import os
import re
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator
from datetime import datetime
import logging

# Configuration
DFT_METHOD = 'PBE'
CONFIG = {
    'DFT_OUTPUT_DIR': f'./outputs_{DFT_METHOD}_ethanol',
    'MOL_DIR': f'./mol_{DFT_METHOD}_ethanol',
    'MODEL_PATH': f'pce_prediction_model_XGBoost_{DFT_METHOD}_eth.joblib',
    'OUTPUT_FILE': f'PCE_predictions_{DFT_METHOD}_eth.xlsx',
}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regular Expressions for DFT Data
PATTERNS = {
    'HOMO': r"Energy of Highest Occupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'LUMO': r"Energy of Lowest Unoccupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'Dipole_Moment': r"dipole magnitude:\s+[-\d.]+\s+au\s+([-\d.]+)\s+debye",
}
COSMO_PATTERNS = {
    'Total_Energy_Hartree': r"Total energy\s*=\s*([-?\d.]+)",
    'Solvation_Energy_eV': r"Dielectric \(solvation\) energy\s+=\s+[-\d.]+\s+([-\d.]+)",
    'Surface_Area_A2': r"Surface area of cavity \[A\*\*2\]\s*=\s+([-\d.]+)",
    'Molecular_Volume_A3': r"Total Volume of cavity \[A\*\*3\]\s*=\s+([-\d.]+)",
    #'COSMO_Screening_Charge': r"cosmo\s*=\s*([-\d.]+)"
}

EXCITATION_PATTERN = r"\s*\d+ ->\s*\d+\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)"

# Physical Constants
CONSTANTS = {
    "ECB_TiO2": -4.00,
    "Eredox_iodide": -4.80,
}

def calculate_molecular_descriptors(mol_file):
    try:
        mol = Chem.MolFromMolFile(mol_file)
        if mol is None:
            logger.error(f"Could not create molecule from file: {mol_file}")
            return None

        desc_dict = {
            'Mass': Descriptors.ExactMolWt(mol),
            'SMILES': Chem.MolToSmiles(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'HBondDonors': Descriptors.NumHDonors(mol),
            'HBondAcceptors': Descriptors.NumHAcceptors(mol),
            'RingCount': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
        }

        calc = Calculator()
        mordred_desc = calc(mol)
        for key, value in mordred_desc.items():
            if hasattr(value, 'value') and value.value is not None and not np.isnan(value.value):
                desc_dict[key] = value.value

        return desc_dict

    except Exception as e:
        logger.error(f"Error calculating descriptors for mol file {mol_file}: {e}")
        return None


def extract_dft_data():
    data = []
    try:
        for filename in os.listdir(CONFIG['DFT_OUTPUT_DIR']):
            if not filename.endswith('.txt'):
                continue

            file_path = os.path.join(CONFIG['DFT_OUTPUT_DIR'], filename)
            record = {'File': filename.replace('.txt', '')}

            with open(file_path, 'r') as file:
                content = file.read()

                # Extract standard DFT data
                for key, pattern in PATTERNS.items():
                    matches = re.findall(pattern, content)
                    record[key] = float(matches[-1]) if matches else None

                for key, pattern in COSMO_PATTERNS.items():
                    matches = re.findall(pattern, content)
                    record[key] = float(matches[-1]) if matches else None

                # Try different excitation patterns
                # Original pattern
                excitations = re.findall(EXCITATION_PATTERN, content)

                # New pattern for TDDFT format
                tddft_pattern = r"\s*\d+\s*->\s*\d+[+-]\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
                tddft_excitations = re.findall(tddft_pattern, content)

                if excitations:  # If original pattern matches
                    max_excitation = max([(float(ex[0]), float(ex[1]), float(ex[2]))
                                          for ex in excitations if ex[2] != ''],
                                         key=lambda x: x[2],
                                         default=(None, None, None))
                    record['Max_Absorption_nm'] = max_excitation[1]
                    record['Max_f_osc'] = max_excitation[2]

                elif tddft_excitations:  # If TDDFT pattern matches
                    # Convert all valid excitations to floats and filter out any with empty f_osc
                    valid_excitations = []
                    for ex in tddft_excitations:
                        try:
                            # ex[2] is TD-ex[nm], ex[5] is f_osc
                            nm = float(ex[2])
                            f_osc = float(ex[5])
                            valid_excitations.append((nm, f_osc))
                        except (ValueError, IndexError):
                            continue

                    if valid_excitations:
                        # Find excitation with maximum oscillator strength
                        max_excitation = max(valid_excitations, key=lambda x: x[1])
                        record['Max_Absorption_nm'] = max_excitation[0]
                        record['Max_f_osc'] = max_excitation[1]
                    else:
                        record['Max_Absorption_nm'] = None
                        record['Max_f_osc'] = None

                else:  # If no patterns match
                    record['Max_Absorption_nm'] = None
                    record['Max_f_osc'] = None

            data.append(record)

        return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"Error during DFT data extraction: {e}")
        raise

def predict_pce():
    try:
        # Load trained model
        logger.info("Loading trained model...")
        model_data = joblib.load(CONFIG['MODEL_PATH'])

        # The model file should contain both the model and the feature list
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_columns = model_data['feature_columns']
            scaler = model_data.get('scaler')
        else:
            # If it's just the model, we'll need to specify the features explicitly
            model = model_data
            # You'll need to specify the exact features used during training
            feature_columns = [
                'HOMO', 'LUMO', 'Dipole_Moment', 'Total_Energy_Hartree',
                'Solvation_Energy_eV', 'Max_Absorption_nm', 'Max_f_osc',
                'Mass', 'LogP', 'TPSA', 'RotatableBonds', 'HBondDonors',
                'HBondAcceptors', 'RingCount', 'AromaticRings',
                'deltaE_LCB', 'deltaE_RedOxH', 'deltaE_HL', 'IP', 'EA',
                'elnChemPot', 'chemHardness', 'electronegativity',
                'electrophilicityIndex', 'electroacceptingPower',
                'electrodonatingPower', 'LHE'
                # Add any missing features that were used during training
            ]
            scaler = None

        # Extract descriptors from mol files
        logger.info("Calculating molecular descriptors...")
        mordred_descriptors_list = []
        for mol_filename in os.listdir(CONFIG['MOL_DIR']):
            if mol_filename.endswith('.mol'):
                mol_path = os.path.join(CONFIG['MOL_DIR'], mol_filename)
                desc = calculate_molecular_descriptors(mol_path)
                if desc:
                    desc['File'] = mol_filename.replace('.mol', '')
                    mordred_descriptors_list.append(desc)

        if not mordred_descriptors_list:
            raise ValueError("No valid molecular descriptors were calculated.")

        mordred_df = pd.DataFrame(mordred_descriptors_list)

        # Extract DFT data
        logger.info("Extracting DFT data...")
        dft_df = extract_dft_data()

        # Merge datasets
        data = dft_df.merge(mordred_df, on='File', how='inner')

        # Add additional descriptors
        logger.info("Calculating additional quantum chemical descriptors...")
        data['deltaE_LCB'] = data['LUMO'] - CONSTANTS['ECB_TiO2']
        data['deltaE_RedOxH'] = CONSTANTS['Eredox_iodide'] - data['HOMO']
        data['deltaE_HL'] = data['LUMO'] - data['HOMO']
        data['IP'] = -data['HOMO']
        data['EA'] = -data['LUMO']
        data['elnChemPot'] = (-1 / 2) * (data['IP'] + data['EA'])
        data['chemHardness'] = (1 / 2) * (data['IP'] - data['EA'])
        data['electronegativity'] = -data['elnChemPot']
        data['electrophilicityIndex'] = data['elnChemPot'] ** 2 / data['chemHardness']
        data['electroacceptingPower'] = ((data['IP'] + 3 * data['EA']) ** 2) / (16 * (data['IP'] - data['EA']))
        data['electrodonatingPower'] = ((3 * data['IP'] + data['EA']) ** 2) / (16 * (data['IP'] - data['EA']))

        data = prepare_prediction_data()

        # Ensure we have all required features
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select only the features used during training
        features = data[feature_columns]

        # Handle missing values
        features = features.fillna(features.mean())

        # Scale features if a scaler was used during training
        if scaler is not None:
            features = scaler.transform(features)

        # Predict PCE
        logger.info("Predicting PCE values...")
        predictions = model.predict(features)
        data['Predicted_PCE'] = predictions

        # Save results
        logger.info("Saving predictions to file...")
        data.to_excel(CONFIG['OUTPUT_FILE'], index=False)
        logger.info(f"Predictions saved to {CONFIG['OUTPUT_FILE']}.")

    except Exception as e:
        logger.error(f"Error during PCE prediction: {e}")
        raise

def prepare_prediction_data():
    """Prepare the data for prediction, including all feature calculations."""
    # Extract descriptors from mol files
    logger.info("Calculating molecular descriptors...")
    mordred_descriptors_list = []
    for mol_filename in os.listdir(CONFIG['MOL_DIR']):
        if mol_filename.endswith('.mol'):
            mol_path = os.path.join(CONFIG['MOL_DIR'], mol_filename)
            desc = calculate_molecular_descriptors(mol_path)
            if desc:
                desc['File'] = mol_filename.replace('.mol', '')
                mordred_descriptors_list.append(desc)

    if not mordred_descriptors_list:
        raise ValueError("No valid molecular descriptors were calculated.")

    mordred_df = pd.DataFrame(mordred_descriptors_list)

    # Extract DFT data
    logger.info("Extracting DFT data...")
    dft_df = extract_dft_data()

    # Merge datasets
    data = dft_df.merge(mordred_df, on='File', how='inner')

    # Calculate additional descriptors
    logger.info("Calculating additional quantum chemical descriptors...")
    data['deltaE_LCB'] = data['LUMO'] - CONSTANTS['ECB_TiO2']
    data['deltaE_RedOxH'] = CONSTANTS['Eredox_iodide'] - data['HOMO']
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

if __name__ == "__main__":
    predict_pce()
