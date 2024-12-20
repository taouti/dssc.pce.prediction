import dataclasses
import os
import re
import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from utils.logging_utils import ExecutionLogger
from utils.regression_metrics_utils import ModelEvaluator

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

# File and Directory Configuration
CONFIG = {
    # Input Directories
    'DFT_OUTPUT_DIR': './outputs_PBE_ethanol',
    'MOL_DIR': './mol_PBE_ethanol',

    # Input Files
    'EXPERIMENTAL_DATA': 'dyes_experimental_dataset.xlsx',

    # Output Files
    'DFT_DATASET': 'dyes_DFT_PBE_eth_dataset.xlsx',
    'COMPREHENSIVE_RESULTS': 'PCE_results_PBE_eth.xlsx',
    'MODEL_OUTPUT': 'pce_prediction_model_PBE_eth.joblib',

    # Model Parameters
    'TEST_SIZE': 0.15,
    'RANDOM_STATE': 0,
    'N_ESTIMATORS': 100,
    'CV_FOLDS': 5,
    'N_JOBS': -1,
    'MAX_FEATURES': 'sqrt',
    'MAX_SAMPLES': 0.6,
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


# Regular Expression Patterns
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
    'COSMO_Screening_Charge': r"cosmo\s*=\s*([-\d.]+)"
}

EXCITATION_PATTERN = r"\s*\d+ ->\s*\d+\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)"


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

                for key, pattern in PATTERNS.items():
                    matches = re.findall(pattern, content)
                    record[key] = float(matches[-1]) if matches else None

                for key, pattern in COSMO_PATTERNS.items():
                    matches = re.findall(pattern, content)
                    record[key] = float(matches[-1]) if matches else None

                excitations = re.findall(EXCITATION_PATTERN, content)
                if excitations:
                    max_excitation = max([(float(ex[0]), float(ex[1]), float(ex[2]))
                                          for ex in excitations], key=lambda x: x[2])
                    record['Max_Absorption_nm'] = max_excitation[1]
                    record['Max_f_osc'] = max_excitation[2]
                else:
                    record['Max_Absorption_nm'] = None
                    record['Max_f_osc'] = None

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
        # But ensure we keep 'Mass' and 'SMILES' columns
        essential_columns = ['Mass', 'SMILES', 'File']
        other_columns = [col for col in mordred_df.columns if col not in essential_columns]

        # Only clean non-essential columns
        mordred_df_cleaned = mordred_df[essential_columns].copy()
        temp_df = mordred_df[other_columns].copy()

        # Remove problematic columns from non-essential columns only
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

        # Handle missing data
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        return data

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise


def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame, Dict]:
    """
    Train and evaluate the Random Forest model with comprehensive metrics.

    Args:
        data: Input DataFrame containing features and target

    Returns:
        Tuple containing (trained_model, results_dataframe, metrics_dict)
    """
    try:
        # Prepare features and target
        excluded_columns = ['PCE', 'File', 'SMILES', 'expVoc_V', 'expIsc_mAcm-2', 'expFF']
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in excluded_columns]

        X = data[feature_columns]
        y = data['PCE']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE']
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize model and evaluator
        rf_model = RandomForestRegressor(
            random_state=CONFIG['RANDOM_STATE'],
            n_estimators=CONFIG['N_ESTIMATORS'],
            n_jobs=CONFIG['N_JOBS'],
            max_features=CONFIG['MAX_FEATURES'],
            max_samples=CONFIG['MAX_SAMPLES'],
        )
        evaluator = ModelEvaluator(n_splits=5, random_state=CONFIG['RANDOM_STATE'])

        # Train model
        logger.info("Training Random Forest model...")
        rf_model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_metrics, test_metrics = evaluator.evaluate_split_performance(
            rf_model, X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Perform cross-validation
        cv_results = evaluator.cross_validate(rf_model, X_train_scaled, y_train)

        # Compile all metrics
        metrics = {
            'train': dataclasses.asdict(train_metrics),
            'test': dataclasses.asdict(test_metrics),
            'cv': {
                'r2_mean': float(np.mean(cv_results['r2'])),
                'r2_std': float(np.std(cv_results['r2'])),
                'rmse_mean': float(np.mean(cv_results['neg_rmse'])),
                'rmse_std': float(np.std(cv_results['neg_rmse'])),
                'mae_mean': float(np.mean(cv_results['neg_mae'])),
                'mae_std': float(np.std(cv_results['neg_mae']))
            }
        }


        # Log performance metrics
        logger.info("\nModel Performance:")
        logger.info(f"Test R²: {test_metrics.r2:.4f}")
        logger.info(f"Test Adjusted R²: {test_metrics.adjusted_r2:.4f}")
        logger.info(f"Test RMSE: {test_metrics.rmse:.4f}")
        logger.info(f"Test MAE: {test_metrics.mae:.4f}")
        logger.info(f"Cross-validation R² (mean ± std): {metrics['cv']['r2_mean']:.4f} ± {metrics['cv']['r2_std']:.4f}")

        # Prepare results DataFrame
        results = data.copy()
        results_scaled = scaler.transform(X)
        results['Predicted_PCE'] = rf_model.predict(results_scaled)

        return rf_model, results, metrics

    except Exception as e:
        logger.error(f"Error in model training and evaluation: {e}")
        raise


def main():
    """Main execution function for the PCE prediction pipeline."""
    try:
        start_time = datetime.now()
        logger.info(f"Starting PCE prediction pipeline at {start_time}")

        # Initialize execution logger
        execution_logger = ExecutionLogger()

        validate_directories()

        # Prepare dataset
        logger.info("Preparing dataset...")
        data = prepare_dataset()

        # Calculate additional descriptors
        logger.info("Calculating additional descriptors...")
        data = calculate_descriptors(data, CONSTANTS)

        # Train model and get results
        logger.info("Training model and generating predictions...")
        model, results, metrics = train_and_evaluate_model(data)

        # Save model and execution info
        model_path = execution_logger.save_model(model)
        logger.info(f"Model saved to: {model_path}")

        # Add execution metadata to metrics
        metrics['execution_time'] = str(datetime.now() - start_time)
        info_path = execution_logger.save_execution_info(CONFIG, metrics)
        logger.info(f"Execution info saved to: {info_path}")

        # Save results
        logger.info("Saving results...")
        results.to_excel(CONFIG['COMPREHENSIVE_RESULTS'], index=False)

        logger.info(f"Pipeline completed. Total execution time: {metrics['execution_time']}")

        return True

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False


if __name__ == "__main__":
    main()
