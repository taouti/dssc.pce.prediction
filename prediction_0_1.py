import os
import re
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator
from datetime import datetime

#from prediction_1 import rf_model
from utils.logging_utils import ExecutionLogger
#from utils.rf_pce_model import RfPCEModel
from utils.visualization_utils import PCEVisualizer
from utils.prediction_model_classes import (
    RfPCEModel,
    XGBoostPCEModel,
#    LightGBMPCEModel,
#    SvmPCEModel
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
    'TEST_SIZE': 0.05,
    'RANDOM_STATE': 0,
    'N_ESTIMATORS': 400,
    'CV_FOLDS': 10,
    'N_JOBS': -1,
    'MAX_FEATURES': 'sqrt',
    'MAX_SAMPLES': 0.8,
    'LEARNING_RATE': 0.03,
    'max_depth': 8,
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
    #'COSMO_Screening_Charge': r"cosmo\s*=\s*([-\d.]+)"
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


def main():
    """Main execution function for the PCE prediction pipeline."""
    try:
        start_time = datetime.now()
        logger.info(f"Starting PCE prediction pipeline at {start_time}")

        execution_logger = ExecutionLogger(DFT_METHOD)
        visualizer = PCEVisualizer(execution_logger, style='default')

        validate_directories()

        # Prepare dataset
        logger.info("Preparing dataset...")
        data = prepare_dataset()

        # Calculate additional descriptors
        logger.info("Calculating additional descriptors...")
        data = calculate_descriptors(data, CONSTANTS)

        # Initialize models with their parameters
        models = {
            'RF': {
                'model': RfPCEModel(
                    test_size=CONFIG['TEST_SIZE'],
                    random_state=CONFIG['RANDOM_STATE'],
                    n_estimators=CONFIG['N_ESTIMATORS'],
                    n_jobs=CONFIG['N_JOBS'],
                    max_features=CONFIG['MAX_FEATURES'],
                    max_samples=CONFIG['MAX_SAMPLES'],
                    cv_folds=CONFIG['CV_FOLDS']
                ),
                'params': {
                    'n_estimators': CONFIG['N_ESTIMATORS'],
                    'max_features': CONFIG['MAX_FEATURES'],
                    'max_samples': CONFIG['MAX_SAMPLES']
                }
            },
            'XGBoost': {
                'model': XGBoostPCEModel(
                    test_size=CONFIG['TEST_SIZE'],
                    random_state=CONFIG['RANDOM_STATE'],
                    n_estimators=CONFIG['N_ESTIMATORS'],
                    learning_rate=CONFIG['LEARNING_RATE'],
                    max_depth=CONFIG['max_depth'],
                    cv_folds=CONFIG['CV_FOLDS']
                ),
                'params': {
                    'n_estimators': CONFIG['N_ESTIMATORS'],
                    'learning_rate': CONFIG['LEARNING_RATE'],
                    'max_depth': CONFIG['max_depth']
                }
            }
        }

        all_metrics = {}
        all_results = {}
        execution_times = {}

        # Train and evaluate each model
        for model_name, model_info in models.items():
            logger.info(f"\nTraining {model_name} model and generating predictions...")
            model_start_time = datetime.now()

            model = model_info['model']
            model_params = model_info['params']
            metrics, results = model.train(data)

            model_end_time = datetime.now()
            execution_time = str(model_end_time - model_start_time)
            execution_times[model_name] = execution_time

            all_metrics[model_name] = metrics
            all_results[model_name] = results

            # Generate visualizations with model parameters
            logger.info(f"Generating visualizations for {model_name}...")
            try:
                visualizer.plot_actual_vs_predicted(results, model_name, model_params)
                visualizer.plot_error_distribution(results, model_name, model_params)
                visualizer.plot_parity(results, model_name, model_params)
                feature_importance = model.get_feature_importance()
                visualizer.plot_feature_importance(feature_importance, model_name, model_params)
                visualizer.plot_feature_correlation(data, model_name)
                visualizer.plot_residuals(results, model_name, model_params)

            except Exception as e:
                logger.error(f"Error generating visualizations for {model_name}: {e}")

            # Save model and results
            model_data = {
                'model': model.model,
                'feature_columns': model.feature_columns,
                'scaler': model.scaler
            }
            model_filename = f"pce_prediction_model_{model_name}_{DFT_METHOD}_eth"
            model_path = execution_logger.save_model(model_data, model_filename)
            logger.info(f"{model_name} model saved to: {model_path}")

            results_filename = f"PCE_results_{model_name}_{DFT_METHOD}_eth.xlsx"
            results.to_excel(results_filename, index=False)
            results_log_path = execution_logger.save_results_file(results_filename)
            logger.info(f"{model_name} results file saved to: {results_log_path}")

        # Add execution times to metrics
        total_execution_time = str(datetime.now() - start_time)
        for model_name in all_metrics:
            all_metrics[model_name]['execution_time'] = execution_times[model_name]

        # Save comprehensive execution info
        comprehensive_info = {
            'config': CONFIG,
            'metrics': all_metrics,
            'total_execution_time': total_execution_time
        }

        info_path = execution_logger.save_execution_info(
            comprehensive_info,
            all_metrics,
            all_results
        )
        logger.info(f"Comprehensive execution info saved to: {info_path}")

        # Create and save comparison summary
        comparison_df = pd.DataFrame({
            'Model': [],
            'Test_R2': [],
            'Test_RMSE': [],
            'Test_MAE': [],
            'CV_R2_Mean': [],
            'CV_R2_Std': [],
            'Execution_Time': []
        })

        for model_name, metrics in all_metrics.items():
            comparison_df = pd.concat([comparison_df, pd.DataFrame({
                'Model': [model_name],
                'Test_R2': [metrics['test']['r2']],
                'Test_RMSE': [metrics['test']['rmse']],
                'Test_MAE': [metrics['test']['mae']],
                'CV_R2_Mean': [metrics['cv']['r2_mean']],
                'CV_R2_Std': [metrics['cv']['r2_std']],
                'Execution_Time': [metrics['execution_time']]
            })], ignore_index=True)

        comparison_filename = f"model_comparison_{DFT_METHOD}_eth.xlsx"
        comparison_df.to_excel(comparison_filename, index=False)
        comparison_log_path = execution_logger.save_results_file(comparison_filename)
        logger.info(f"\nModel comparison saved to: {comparison_log_path}")

        logger.info(f"\nPipeline completed. Total execution time: {total_execution_time}")

        return True

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False


if __name__ == "__main__":
    main()
