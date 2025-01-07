"""
PCE Predictor module for making predictions on new dye molecules using trained models.
"""

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors
from datetime import datetime
import json

from .prediction_model_classes import BasePCEModel, RfPCEModel, XGBoostPCEModel, EnsemblePCEModel
from .preprocessing import FeaturePreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)

# Patterns for extracting values (matching both old and new formats)
PATTERNS = {
    'HOMO': r"(?:Energy of )?Highest Occupied Molecular Orbital:\s*[-\d.]+Ha\s+([-\d.]+)eV",
    'LUMO': r"(?:Energy of )?Lowest Unoccupied Molecular Orbital:\s*[-\d.]+Ha\s+([-\d.]+)eV",
    'Dipole_Moment': r"dipole magnitude:\s*[-\d.]+\s+au\s+([-\d.]+)\s+debye"
}

# Pattern for TDDFT excitation data (matching both old and new formats)
TDDFT_SECTION_PATTERNS = [
    r"TDDFT excitations singlet_alda",
    r"TDDFT excitations alda"
]

# Pattern for excitation lines (handles both +/- and regular orbital numbers)
TDDFT_EXCITATION_PATTERN = r"\s*(\d+)(?:\s*->|→)\s*(\d+)(?:[+-])?\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"

@dataclass
class Constants:
    """Physical constants used in calculations."""
    ECB_TiO2: float = -4.0  # TiO2 conduction band energy (eV)
    Eredox_iodide: float = -4.8  # Redox potential of iodide/triiodide (eV)

class PCEPredictor:
    """Class for predicting PCE values for new dye molecules using trained models."""
    
    def __init__(
        self,
        models_dir: Union[str, Path],
        new_dyes_dft_dir: Union[str, Path],
        new_dyes_mol_dir: Union[str, Path],
        output_dir: Union[str, Path],
        dft_method: str = "PBE"
    ):
        """
        Initialize the PCE predictor.
        
        Args:
            models_dir: Directory containing trained models
            new_dyes_dft_dir: Directory containing DFT outputs for new dyes
            new_dyes_mol_dir: Directory containing mol files for new dyes
            output_dir: Directory to save prediction results
            dft_method: DFT method used (default: "PBE")
        """
        self.models_dir = Path(models_dir)
        self.new_dyes_dft_dir = Path(new_dyes_dft_dir)
        self.new_dyes_mol_dir = Path(new_dyes_mol_dir)
        self.output_dir = Path(output_dir)
        self.dft_method = dft_method
        self.constants = Constants()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model containers
        self.models: Dict[str, Optional[BasePCEModel]] = {
            'RF': None,
            'XGB': None,
            'ENSEMBLE': None
        }
        
        # Define core features used by models
        self.core_features = [
            'HOMO', 'LUMO', 'deltaE_LCB', 'deltaE_RedOxH', 'deltaE_HL',
            'IP', 'EA', 'elnChemPot', 'chemHardness', 'electronegativity',
            'electrophilicityIndex', 'electroacceptingPower'
        ]

    def calculate_molecular_descriptors(self, mol_file: str) -> Optional[Dict]:
        """
        Calculate essential molecular descriptors from mol file using RDKit and Mordred.
        Only calculates most relevant descriptors for PCE prediction.
        
        Args:
            mol_file: Path to the mol file
            
        Returns:
            Dictionary of molecular descriptors or None if calculation fails
        """
        try:
            mol = Chem.MolFromMolFile(mol_file)
            if mol is None:
                logger.error(f"Could not create molecule from file: {mol_file}")
                return None

            # Calculate Mass separately as it's needed for other calculations
            mass = Descriptors.ExactMolWt(mol)
            smiles = Chem.MolToSmiles(mol)

            # Basic RDKit descriptors
            desc_dict = {
                'Mass': mass,  # Keeping Mass as it's needed for other calculations
                'SMILES': smiles,
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol),
                'HBondDonors': Descriptors.NumHDonors(mol),
                'HBondAcceptors': Descriptors.NumHAcceptors(mol),
                'RingCount': Descriptors.RingCount(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol)
            }

            # Create calculator with default descriptors
            calc = Calculator()

            mordred_desc = calc(mol)

            # Filter out any invalid or NaN values
            for key, value in mordred_desc.items():
                if hasattr(value, 'value'):  # Check if descriptor was calculated successfully
                    if value.value is not None and not np.isnan(value.value):
                        desc_dict[key] = value.value

            return desc_dict

        except Exception as e:
            logger.error(f"Error calculating descriptors for mol file {mol_file}: {e}")
            return None

    def extract_dft_data(self) -> pd.DataFrame:
        """Extract data from DFT output files."""
        data = []
        try:
            # Check if directory exists
            if not self.new_dyes_dft_dir.exists():
                raise ValueError(f"DFT directory does not exist: {self.new_dyes_dft_dir}")

            # Get list of txt files
            txt_files = list(self.new_dyes_dft_dir.glob('*.txt'))
            if not txt_files:
                raise ValueError(f"No .txt files found in {self.new_dyes_dft_dir}")

            logger.info(f"Found {len(txt_files)} .txt files in {self.new_dyes_dft_dir}")

            for txt_file in txt_files:
                try:
                    logger.info(f"Processing {txt_file.name}...")
                    record = {'File': txt_file.stem}

                    with open(txt_file, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # Extract HOMO, LUMO, Dipole
                    for key, pattern in PATTERNS.items():
                        matches = re.findall(pattern, content)
                        if matches:
                            try:
                                record[key] = float(matches[-1])
                                logger.debug(f"Extracted {key}: {record[key]} from {txt_file.name}")
                            except ValueError:
                                record[key] = None

                    # Extract TDDFT excitation data
                    excitations = []
                    found_tddft_section = False

                    for line in content.split('\n'):
                        # Look for the start of TDDFT section
                        if any(re.match(pattern, line) for pattern in TDDFT_SECTION_PATTERNS):
                            found_tddft_section = True
                            continue

                        if found_tddft_section:
                            match = re.match(TDDFT_EXCITATION_PATTERN, line)
                            if match:
                                try:
                                    # Groups: from_orbital, to_orbital, td_ev, ks_ev, td_nm, ks_nm, ha, f_osc, overlap
                                    groups = match.groups()
                                    excitations.append({
                                        'wavelength': float(groups[4]),  # td_nm
                                        'oscillator_strength': float(groups[7]),  # f_osc
                                        'energy_ev': float(groups[2])  # td_ev
                                    })
                                except (ValueError, IndexError):
                                    continue

                    if excitations:
                        # Find excitation with maximum oscillator strength
                        max_excitation = max(excitations, key=lambda x: x['oscillator_strength'])
                        record['Max_Absorption_nm'] = max_excitation['wavelength']
                        record['Max_f_osc'] = max_excitation['oscillator_strength']
                        record['Max_Excitation_eV'] = max_excitation['energy_ev']
                    else:
                        record['Max_Absorption_nm'] = None
                        record['Max_f_osc'] = None
                        record['Max_Excitation_eV'] = None

                    # Add validation checks
                    if record.get('Max_Absorption_nm') is not None:
                        if not (200 <= record['Max_Absorption_nm'] <= 800):
                            logger.warning(f"Suspicious absorption value in {txt_file.name}: {record['Max_Absorption_nm']}nm")

                    if record.get('HOMO') is not None and record.get('LUMO') is not None:
                        if not (-10 <= record['HOMO'] <= 0) or not (-10 <= record['LUMO'] <= 0):
                            logger.warning(
                                f"Suspicious HOMO/LUMO values in {txt_file.name}: HOMO={record['HOMO']}eV, LUMO={record['LUMO']}eV")

                    # Log extracted values
                    logger.info(f"Extracted data from {txt_file.name}:")
                    for key, value in record.items():
                        logger.info(f"  {key}: {value}")

                    data.append(record)

                except Exception as e:
                    logger.error(f"Error processing {txt_file}: {str(e)}")
                    logger.error("Full error details:", exc_info=True)
                    continue

            if not data:
                raise ValueError("No valid DFT data was extracted from any files")

            df = pd.DataFrame(data)
            logger.info(f"Successfully created DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Error extracting DFT data: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise

    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare feature matrix for new dyes.
        
        Returns:
            DataFrame containing all required features for prediction
        """
        try:
            # Calculate molecular descriptors
            logger.info("Calculating molecular descriptors...")
            mordred_descriptors_list = []
            for mol_file in self.new_dyes_mol_dir.glob("*.mol"):
                desc = self.calculate_molecular_descriptors(str(mol_file))
                if desc:
                    desc['File'] = mol_file.stem
                    mordred_descriptors_list.append(desc)

            if not mordred_descriptors_list:
                raise ValueError("No valid molecular descriptors were calculated.")

            mordred_df = pd.DataFrame(mordred_descriptors_list)
            logger.info(f"Calculated descriptors for {len(mordred_df)} molecules")

            # Extract DFT data
            logger.info("Extracting DFT data...")
            dft_df = self.extract_dft_data()

            # Merge datasets
            logger.info("Merging DFT and molecular descriptor data...")
            data = dft_df.merge(mordred_df, on='File', how='inner')
            logger.info(f"Successfully merged data for {len(data)} molecules")

            # Calculate additional descriptors
            data['deltaE_LCB'] = data['LUMO'] - self.constants.ECB_TiO2
            data['deltaE_RedOxH'] = self.constants.Eredox_iodide - data['HOMO']
            data['deltaE_HL'] = data['LUMO'] - data['HOMO']
            data['IP'] = -data['HOMO']
            data['EA'] = -data['LUMO']
            data['elnChemPot'] = (-1 / 2) * (data['IP'] + data['EA'])
            data['chemHardness'] = (1 / 2) * (data['IP'] - data['EA'])
            data['electronegativity'] = -data['elnChemPot']
            data['electrophilicityIndex'] = data['elnChemPot'] ** 2 / data['chemHardness']
            data['electroacceptingPower'] = ((data['IP'] + 3 * data['EA']) ** 2) / (16 * (data['IP'] - data['EA']))
            data['electrodonatingPower'] = ((3 * data['IP'] + data['EA']) ** 2) / (16 * (data['IP'] - data['EA']))
            data['LHE'] = (1 - 10 ** -data['Max_f_osc']) * 100

            # Save the complete descriptors for new dyes
            all_descriptors_path = self.output_dir / f"All_descriptors_{self.dft_method}_eth_new_dyes.xlsx"
            data.to_excel(all_descriptors_path, index=False)
            logger.info(f"Saved all descriptors for new dyes to: {all_descriptors_path}")

            # Verify all required features are present
            missing_features = [f for f in self.core_features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            logger.info("Feature preparation completed successfully")
            return data

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise

    def load_models(self) -> None:
        """Load all available trained models from the models directory."""
        try:
            # Find model files
            rf_models = list(self.models_dir.glob(f"pce_prediction_model_RF_{self.dft_method}_eth.joblib"))
            xgb_models = list(self.models_dir.glob(f"pce_prediction_model_XGBoost_{self.dft_method}_eth.joblib"))
            ensemble_models = list(self.models_dir.glob(f"pce_prediction_model_ENSEMBLE_{self.dft_method}_eth.joblib"))
            
            # Load RF model
            if rf_models:
                latest_rf = rf_models[0]
                rf_model = RfPCEModel()
                rf_model.load_model(latest_rf)
                self.models['RF'] = rf_model
                logger.info(f"Loaded RF model from {latest_rf}")
            
            # Load XGBoost model
            if xgb_models:
                latest_xgb = xgb_models[0]
                xgb_model = XGBoostPCEModel()
                xgb_model.load_model(latest_xgb)
                self.models['XGB'] = xgb_model
                logger.info(f"Loaded XGB model from {latest_xgb}")
            
            # Load Ensemble model
            if ensemble_models:
                latest_ensemble = ensemble_models[0]
                ensemble_model = EnsemblePCEModel()
                ensemble_model.load_model(latest_ensemble)
                self.models['ENSEMBLE'] = ensemble_model
                logger.info(f"Loaded ENSEMBLE model from {latest_ensemble}")
            
            # Get core features from any loaded model
            for model in self.models.values():
                if model is not None:
                    self.core_features = model.preprocessor.selected_features
                    break
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Full error details:", exc_info=True)
            raise

    def predict(self) -> None:
        """Make predictions using all available models."""
        try:
            # Load models if not already loaded
            if not any(self.models.values()):
                self.load_models()

            # Prepare features
            data = self.prepare_features()
            
            # Create a results directory if it doesn't exist
            results_dir = self.output_dir / "predictions"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Make predictions with each model
            all_predictions = pd.DataFrame({'File': data['File']})
            
            for model_name, model in self.models.items():
                if model is None:
                    continue

                logger.info(f"\nMaking predictions with {model_name} model...")
                try:
                    # Get features for prediction
                    features = data[self.core_features]
                    
                    # Make predictions
                    predictions = model.predict(features)

                    # Add predictions to results
                    all_predictions[f'PCE_{model_name}'] = predictions

                    # Create detailed results for each model
                    model_results = pd.DataFrame({
                        'File': data['File'],
                        'SMILES': data['SMILES'] if 'SMILES' in data.columns else None,
                        'Predicted_PCE': predictions,
                        'HOMO': data['HOMO'],
                        'LUMO': data['LUMO'],
                        'Max_Absorption_nm': data['Max_Absorption_nm'],
                        'Max_f_osc': data['Max_f_osc'],
                        'Dipole_Moment': data['Dipole_Moment']
                    })

                    # Save model-specific results
                    model_results_path = results_dir / f"PCE_predictions_{model_name}_{self.dft_method}_eth_new_dyes.xlsx"
                    model_results.to_excel(model_results_path, index=False)
                    logger.info(f"Saved {model_name} predictions to: {model_results_path}")

                except Exception as e:
                    logger.error(f"Error making predictions with {model_name} model: {str(e)}")
                    continue

            # Save combined predictions
            if len(all_predictions.columns) > 1:  # Only save if we have predictions
                combined_results_path = results_dir / f"PCE_predictions_ALL_MODELS_{self.dft_method}_eth_new_dyes.xlsx"
                all_predictions.to_excel(combined_results_path, index=False)
                logger.info(f"Saved combined predictions to: {combined_results_path}")

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise
