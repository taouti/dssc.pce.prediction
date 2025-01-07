import os
import re
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from utils.logging_utils import ExecutionLogger
from utils.visualization_utils import PCEVisualizer
from utils.prediction_model_classes import (
    RfPCEModel,
    XGBoostPCEModel,
    EnsemblePCEModel,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pce_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Keyword Configuration
DFT_METHOD = 'PBE'

# File and Directory Configuration
CONFIG = {
    # Input Directories
    'DFT_OUTPUT_DIR': f'./outputs_{DFT_METHOD}_ethanol',
    'MOL_DIR': f'./mol_{DFT_METHOD}_ethanol',

    # Input Files
    'EXPERIMENTAL_DATA': 'dyes_experimental_dataset.xlsx',

    # Output Files
    'DFT_DATASET': f'dyes_DFT_{DFT_METHOD}_eth_dataset.xlsx',
    'COMPREHENSIVE_RESULTS': f'PCE_results_{DFT_METHOD}_eth.xlsx',
    'MODEL_OUTPUT': f'pce_prediction_model_{DFT_METHOD}_eth.joblib',
    
    # Model Parameters
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'CV_FOLDS': 5,
    
    # Random Forest Parameters
    'RF_PARAMS': {
        'n_estimators': 500,
        'max_depth': 6,
        'min_samples_split': 3,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'max_samples': 0.8,
        'n_jobs': -1
    },
    
    # XGBoost Parameters
    'XGB_PARAMS': {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 0.1,
        'scale_pos_weight': 1.0,
    },
    
    # Preprocessing Parameters
    'PREPROCESSING': {
        'force_include_features': ['HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment'],
        'n_features_to_select': 15,
        'correlation_threshold': 0.95,
        'rf_importance_weight': 0.7,
        'mi_importance_weight': 0.3
    },
    
    # Ensemble Parameters
    'ENSEMBLE': {
        'weights': [0.6, 0.4]  # RF weight, XGBoost weight
    }
}


# Physical Constants
CONSTANTS = {
    "Ha_kcalPermol": 627.51,
    "Ha_eV": 27.2114,
    "KB": 1.380649e-23,
    "viscosityHe": 20.0e-6,
    "densityHe": 9.00e-2,
    "NA": 6.022e23,
    "T": 300,
    "ECB_TiO2": -4.00,
    "Eredox_iodide": -4.80,
}


# Updated patterns for better accuracy
PATTERNS = {
    'HOMO': r"Energy of Highest Occupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'LUMO': r"Energy of Lowest Unoccupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'Dipole_Moment': r"dipole magnitude:\s+[-\d.]+\s+au\s+([-\d.]+)\s+debye"
}

COSMO_PATTERNS = {
    'Total_Energy_Hartree': r"Total energy\s*=\s*([-?\d.]+)",
    'Solvation_Energy_eV': r"Dielectric \(solvation\) energy\s+=\s+[-\d.]+\s+([-\d.]+)",
    'Surface_Area_A2': r"Surface area of cavity \[A\*\*2\]\s*=\s+([-\d.]+)",
    'Molecular_Volume_A3': r"Total Volume of cavity \[A\*\*3\]\s*=\s+([-\d.]+)",
}

# New pattern to match TDDFT excitation data format
TDDFT_EXCITATION_PATTERN = r"\s*(\d+)\s*->\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"



def validate_directories():
    """Validate existence of required directories and create if necessary."""
    required_dirs = [CONFIG['DFT_OUTPUT_DIR'], CONFIG['MOL_DIR']]
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} not found. Creating...")
            os.makedirs(directory)


def calculate_molecular_descriptors(mol_file):
    """
    Calculate essential molecular descriptors from mol file using RDKit and Mordred.
    Only calculates most relevant descriptors for PCE prediction.

    Args:
        mol_file (str): Path to the mol file

    Returns:
        dict: Dictionary of molecular descriptors or None if calculation fails
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


def extract_dft_data():
    """Extract data from DFT output files."""
    data = []
    try:
        for filename in os.listdir(CONFIG['DFT_OUTPUT_DIR']):
            if not filename.endswith('.txt'):
                continue

            file_path = os.path.join(CONFIG['DFT_OUTPUT_DIR'], filename)
            record = {'File': filename.replace('.txt', '')}

            with open(file_path, 'r') as file:
                content = file.read()

                # Extract HOMO, LUMO, Dipole
                for key, pattern in PATTERNS.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        try:
                            record[key] = float(matches[-1])
                            logger.debug(f"Extracted {key}: {record[key]} from {filename}")
                        except ValueError:
                            record[key] = None

                # Extract TDDFT excitation data
                excitations = []
                found_tddft_section = False

                for line in content.split('\n'):
                    # Look for the start of TDDFT section
                    if "TDDFT excitations singlet_alda" in line:
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
                        logger.warning(f"Suspicious absorption value in {filename}: {record['Max_Absorption_nm']}nm")

                if record.get('HOMO') is not None and record.get('LUMO') is not None:
                    if not (-10 <= record['HOMO'] <= 0) or not (-10 <= record['LUMO'] <= 0):
                        logger.warning(
                            f"Suspicious HOMO/LUMO values in {filename}: HOMO={record['HOMO']}eV, LUMO={record['LUMO']}eV")

            data.append(record)

        return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"Error during DFT data extraction: {e}")
        raise

def calculate_descriptors(data, constants):
    """
    Calculate quantum chemical descriptors.

    Args:
        data (pd.DataFrame): Input dataframe with HOMO/LUMO values
        constants (dict): Physical constants

    Returns:
        pd.DataFrame: DataFrame with additional calculated descriptors
    """
    try:
        data['deltaE_LCB'] = data['LUMO'] - constants['ECB_TiO2']
        data['deltaE_RedOxH'] = constants['Eredox_iodide'] - data['HOMO']
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
        return data
    except Exception as e:
        logger.error(f"Error calculating descriptors: {e}")
        raise


def prepare_dataset():
    """Prepare and merge all datasets."""
    try:
        # Extract DFT data
        logger.info("Extracting DFT data...")
        dft_df = extract_dft_data()
        dft_df.to_excel(CONFIG['DFT_DATASET'], index=False)

        # Load experimental data
        logger.info("Loading experimental data...")
        experimental_df = pd.read_excel(CONFIG['EXPERIMENTAL_DATA'])
        experimental_df.rename(columns={'Dye': 'File', 'expPCE': 'PCE'}, inplace=True)

        # Calculate molecular descriptors
        logger.info("Calculating molecular descriptors...")
        mordred_descriptors_list = []
        for mol_filename in os.listdir(CONFIG['MOL_DIR']):
            if mol_filename.endswith('.mol'):
                file_name = mol_filename.replace('.mol', '')
                mol_path = os.path.join(CONFIG['MOL_DIR'], mol_filename)

                desc = calculate_molecular_descriptors(mol_path)
                if desc:
                    desc['File'] = file_name
                    mordred_descriptors_list.append(desc)

        if not mordred_descriptors_list:
            raise ValueError("No valid molecular descriptors were calculated")

        mordred_df = pd.DataFrame(mordred_descriptors_list)

        # Remove columns with all NaN values or constant values
        essential_columns = ['Mass', 'SMILES', 'File']
        other_columns = [col for col in mordred_df.columns if col not in essential_columns]
        
        # Clean non-essential columns
        mordred_df_cleaned = mordred_df[essential_columns].copy()
        temp_df = mordred_df[other_columns].copy()
        
        # Remove problematic columns
        temp_df = temp_df.dropna(axis=1, how='all')
        constant_cols = temp_df.columns[temp_df.nunique() == 1]
        temp_df = temp_df.drop(columns=constant_cols)
        
        # Combine back with essential columns
        mordred_df = pd.concat([mordred_df_cleaned, temp_df], axis=1)

        # Normalize File columns
        for df in [dft_df, experimental_df, mordred_df]:
            df['File'] = df['File'].str.strip().str.lower()

        # Merge datasets
        data = dft_df.merge(experimental_df, on='File', how='inner').merge(mordred_df, on='File', how='inner')

        # Calculate additional descriptors
        data = calculate_descriptors(data, CONSTANTS)

        # Handle missing data with more sophisticated imputation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                # Use median for highly skewed features
                if abs(data[col].skew()) > 1:
                    data[col] = data[col].fillna(data[col].median())
                else:
                    data[col] = data[col].fillna(data[col].mean())

        # Save complete dataset before feature selection and scaling
        descriptors_path = os.path.join(
            os.path.dirname(CONFIG['DFT_DATASET']),
            f"All_descriptors_{DFT_METHOD}_eth_training.xlsx"
        )
        data.to_excel(descriptors_path, index=False)
        logger.info(f"Saved complete training dataset to: {descriptors_path}")

        return data, data.copy()

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise


def prepare_pce_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare PCE results DataFrame by selecting relevant columns."""
    pce_results = results_df[['File', 'PCE', 'PredictedPCE', 'Absolute_Error', 'Error_Percentage', 'Dataset']]
    return pce_results


def generate_visualizations(model, model_name: str, output_dir: Path) -> None:
    """Generate and save visualizations for model results."""
    try:
        visualizer = PCEVisualizer(logging.getLogger())
        # Create plots directory at the same level as results directory
        plots_dir = output_dir.parent / "plots"
        enhanced_plots_dir = output_dir.parent / "enhanced_plots"
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(enhanced_plots_dir, exist_ok=True)

        # Generate standard visualizations
        visualizer.output_dir = plots_dir
        visualizer.visualize_results(model, model_name)
        logging.info(f"Generated standard visualizations for {model_name}")

        # Generate enhanced visualizations
        visualizer.output_dir = enhanced_plots_dir
        results = model.get_results()
        
        try:
            # Enhanced prediction plots
            visualizer.plot_predicted_vs_actual_enhanced(results, model_name)
            visualizer.plot_residuals_enhanced(results, model_name)
            visualizer.plot_error_distribution_enhanced(results, model_name)
            
            # Enhanced feature analysis
            if hasattr(model, 'get_feature_importance'):
                importance_df = model.get_feature_importance()
                visualizer.plot_feature_importance_enhanced(importance_df, model_name)
                visualizer.plot_correlation_heatmap_enhanced(results, model_name)
            
            # Descriptor-PCE relationship plots for key features
            key_descriptors = ['HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment']
            for descriptor in key_descriptors:
                if descriptor in results.columns:
                    visualizer.plot_descriptor_pce_relationship(results, descriptor, model_name)
            
            # Dye ranking plot
            visualizer.plot_dye_ranking(results, model_name)
            
            logging.info(f"Generated enhanced visualizations for {model_name}")
        except Exception as e:
            logging.error(f"Error generating enhanced visualizations for {model_name}: {e}")

        # Clean up any remaining figures
        plt.close('all')

    except Exception as e:
        logging.error(f"Error in visualization generation for {model_name}: {e}")
        plt.close('all')  # Ensure cleanup even if there's an error


def main():
    """Main execution function for the PCE prediction pipeline."""
    try:
        start_time = datetime.now()
        logger.info(f"Starting PCE prediction pipeline at {start_time}")

        # Create execution logger with correct parameters
        execution_logger = ExecutionLogger(
            dft=DFT_METHOD,
            base_log_dir="log"
        )

        # Initialize visualizer
        visualizer = PCEVisualizer(execution_logger)

        # Validate directories
        validate_directories()

        # Prepare dataset
        logger.info("Preparing dataset...")
        data, complete_data = prepare_dataset()
        
        # Save all descriptors (using complete data before feature selection and scaling)
        logger.info("Saving all descriptors...")
        descriptors_path = execution_logger.save_descriptors(complete_data, DFT_METHOD)
        logger.info(f"All descriptors saved to: {descriptors_path}")

        # Train and evaluate Random Forest model
        logger.info("\nTraining RF model and generating predictions...")
        rf_model = RfPCEModel(
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            cv_folds=CONFIG['CV_FOLDS'],
            **CONFIG['RF_PARAMS']
        )
        rf_model._create_model()  # Initialize the model
        rf_metrics, rf_results = rf_model.train(data)

        # Prepare and save RF results
        rf_pce_results = prepare_pce_results(rf_results)
        rf_results_path = os.path.join(
            execution_logger.results_dir,
            f"PCE_results_RF_{DFT_METHOD}_eth.xlsx"
        )
        rf_pce_results.to_excel(rf_results_path, index=False)
        logger.info(f"RF results file saved to: {rf_results_path}")

        # Generate visualizations for RF
        logger.info("Generating visualizations for RF...")
        try:
            rf_importance = rf_model.get_feature_importance()
            generate_visualizations(rf_model, "RF", execution_logger.results_dir)
        except Exception as e:
            logger.error(f"Error generating visualizations for RF: {e}")

        # Save RF model
        rf_model_path = os.path.join(
            execution_logger.models_dir,
            f"pce_prediction_model_RF_{DFT_METHOD}_eth.joblib"
        )
        rf_model.save_model(rf_model_path)
        logger.info(f"RF model saved to: {rf_model_path}")

        # Train and evaluate XGBoost model
        logger.info("\nTraining XGBoost model and generating predictions...")
        xgb_model = XGBoostPCEModel(
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            cv_folds=CONFIG['CV_FOLDS'],
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0,
            reg_alpha=0,
            reg_lambda=0.1,
            scale_pos_weight=1.0
        )
        xgb_model._create_model()
        xgb_metrics, xgb_results = xgb_model.train(data)

        # Prepare and save XGBoost results
        xgb_pce_results = prepare_pce_results(xgb_results)
        xgb_results_path = os.path.join(
            execution_logger.results_dir,
            f"PCE_results_XGBoost_{DFT_METHOD}_eth.xlsx"
        )
        xgb_pce_results.to_excel(xgb_results_path, index=False)
        logger.info(f"XGBoost results file saved to: {xgb_results_path}")

        # Generate visualizations for XGBoost
        logger.info("Generating visualizations for XGBoost...")
        try:
            xgb_importance = xgb_model.get_feature_importance()
            generate_visualizations(xgb_model, "XGBoost", execution_logger.results_dir)
        except Exception as e:
            logger.error(f"Error generating visualizations for XGBoost: {e}")

        # Save XGBoost model
        xgb_model_path = os.path.join(
            execution_logger.models_dir,
            f"pce_prediction_model_XGBoost_{DFT_METHOD}_eth.joblib"
        )
        xgb_model.save_model(xgb_model_path)
        logger.info(f"XGBoost model saved to: {xgb_model_path}")

        # Train and evaluate Ensemble model
        logger.info("\nTraining Ensemble model and generating predictions...")
        ensemble_model = EnsemblePCEModel(
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            cv_folds=CONFIG['CV_FOLDS'],
            rf_params=CONFIG['RF_PARAMS'],
            xgb_params=CONFIG['XGB_PARAMS']  # Updated to match config key
        )
        ensemble_metrics, ensemble_results = ensemble_model.train(data)

        # Prepare and save Ensemble results
        ensemble_pce_results = prepare_pce_results(ensemble_results)
        ensemble_results_path = os.path.join(
            execution_logger.results_dir,
            f"PCE_results_ENSEMBLE_{DFT_METHOD}_eth.xlsx"
        )
        ensemble_pce_results.to_excel(ensemble_results_path, index=False)
        logger.info(f"Ensemble results file saved to: {ensemble_results_path}")

        # Generate visualizations for Ensemble
        logger.info("Generating visualizations for Ensemble...")
        try:
            ensemble_importance = ensemble_model.get_feature_importance()
            generate_visualizations(ensemble_model, "ENSEMBLE", execution_logger.results_dir)
        except Exception as e:
            logger.error(f"Error generating visualizations for Ensemble: {e}")

        # Save Ensemble model
        ensemble_model_path = os.path.join(
            execution_logger.models_dir,
            f"pce_prediction_model_ENSEMBLE_{DFT_METHOD}_eth.joblib"
        )
        ensemble_model.save_model(ensemble_model_path)
        logger.info(f"Ensemble model saved to: {ensemble_model_path}")

        # Now predict PCE for new dyes using the trained models
        logger.info("\nPredicting PCE for new dyes...")
        from utils.pce_predictor import PCEPredictor
        
        predictor = PCEPredictor(
            models_dir=execution_logger.models_dir,
            new_dyes_dft_dir="New_dyes_outputs_PBE_ethanol",
            new_dyes_mol_dir="New_dyes_mol_PBE_eth",
            output_dir=execution_logger.results_dir,
            dft_method=DFT_METHOD
        )
        
        predictor.predict()
        logger.info("PCE predictions for new dyes completed.")

        # Log execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"\nPCE prediction pipeline completed at {end_time}")
        logger.info(f"Total execution time: {execution_time}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
