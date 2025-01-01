import os
import re
import logging
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from datetime import datetime

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
    'FEATURE_IMPORTANCE': 'feature_importance_PBE_eth.xlsx',

    # Model Parameters
    'TEST_SIZE': 0.1,
    'RANDOM_STATE': 42,
    'N_ESTIMATORS': 100,
    'CV_FOLDS': 5,
    'N_JOBS': -1,
    'MAX_FEATURES': 'sqrt',
    'MAX_SAMPLES': 0.3,
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

            try:
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

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                continue

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
    """Prepare and merge all datasets with optimized descriptor selection."""
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


def train_and_evaluate_model(data):
    """Train and evaluate the Random Forest model."""
    try:
        # Prepare features and target
        excluded_columns = ['PCE', 'File', 'SMILES', 'expVoc_V', 'expIsc_mAcm-2', 'expFF']
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in excluded_columns]

        X = data[feature_columns]
        y = data['PCE']

        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            X, y, data['File'],
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE']
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        logger.info("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=CONFIG['N_ESTIMATORS'],
            random_state=CONFIG['RANDOM_STATE'],
            n_jobs=CONFIG['N_JOBS'],
            max_features=CONFIG['MAX_FEATURES'],
            max_samples=CONFIG['MAX_SAMPLES'],
        )

        # Perform cross-validation
        cv_scores = cross_val_score(
            rf_model, X_train_scaled, y_train,
            cv=CONFIG['CV_FOLDS']
        )
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

        # Train final model
        rf_model.fit(X_train_scaled, y_train)

        # Feature selection
        selector = SelectFromModel(rf_model, prefit=True)
        selected_features = X_train.columns[selector.get_support()].tolist()
        logger.info(f"Selected features: {selected_features}")

        # Save model
        joblib.dump(rf_model, CONFIG['MODEL_OUTPUT'])

        # Predictions and evaluation
        y_pred_test = rf_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        logger.info(f"Model Performance - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Prepare comprehensive results
        data_scaled = scaler.transform(X)
        data['Predicted_PCE'] = rf_model.predict(data_scaled)
        data['Prediction_Error'] = abs(data['Predicted_PCE'] - data['PCE'])
        data['Prediction_Error_Percentage'] = (data['Prediction_Error'] / data['PCE']) * 100
        data['Prediction_Accuracy_Percentage'] = abs(100 - data['Prediction_Error_Percentage'])

        # Add mean values for prediction metrics
        metrics_mean = data[['Prediction_Error', 'Prediction_Error_Percentage',
                             'Prediction_Accuracy_Percentage']].mean()
        logger.info(f"Mean Prediction Metrics:\n{metrics_mean}")

        # Add dataset split labels
        data['Dataset'] = 'Training'
        data.loc[data['File'].isin(test_files), 'Dataset'] = 'Testing'

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance.to_excel(CONFIG['FEATURE_IMPORTANCE'], index=False)

        # Reorder columns
        column_order = [
                           'File', 'PCE', 'Predicted_PCE', 'Prediction_Error',
                           'Prediction_Error_Percentage', 'Prediction_Accuracy_Percentage', 'Dataset'
                       ] + [col for col in data.columns if col not in [
            'File', 'PCE', 'Predicted_PCE', 'Prediction_Error',
            'Prediction_Error_Percentage', 'Prediction_Accuracy_Percentage', 'Dataset'
        ]]

        data = data[column_order]

        return data, feature_importance

    except Exception as e:
        logger.error(f"Error in model training and evaluation: {e}")
        raise

    def validate_experimental_data(df):
        """
        Validate the structure and content of experimental data.

        Args:
            df (pd.DataFrame): Experimental data DataFrame

        Returns:
            bool: True if validation passes, False otherwise
        """
        required_columns = ['File', 'PCE', 'expVoc_V', 'expIsc_mAcm-2', 'expFF']

        try:
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in experimental data: {missing_columns}")
                return False

            # Check for null values in critical columns
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                logger.warning(f"Null values found in experimental data:\n{null_counts[null_counts > 0]}")

            # Check for negative values in measurements
            for col in ['PCE', 'expVoc_V', 'expIsc_mAcm-2', 'expFF']:
                neg_values = df[df[col] < 0]
                if not neg_values.empty:
                    logger.error(f"Negative values found in {col}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating experimental data: {e}")
            return False

    def validate_dft_output(content):
        """
        Validate DFT output file content.

        Args:
            content (str): Content of DFT output file

        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check for essential DFT calculation components
            required_patterns = [
                r"SCF\s+converged",
                r"Total\s+Energy",
                r"HOMO",
                r"LUMO"
            ]

            for pattern in required_patterns:
                if not re.search(pattern, content):
                    logger.error(f"Missing required DFT output pattern: {pattern}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating DFT output: {e}")
            return False

    def check_file_paths():
        """
        Check existence and permissions of all required files and directories.

        Returns:
            bool: True if all checks pass, False otherwise
        """
        try:
            # Check input files
            required_files = [CONFIG['EXPERIMENTAL_DATA']]
            for file_path in required_files:
                if not os.path.isfile(file_path):
                    logger.error(f"Required file not found: {file_path}")
                    return False
                if not os.access(file_path, os.R_OK):
                    logger.error(f"No read permission for file: {file_path}")
                    return False

            # Check directories
            required_dirs = [CONFIG['DFT_OUTPUT_DIR'], CONFIG['MOL_DIR']]
            for dir_path in required_dirs:
                if not os.path.isdir(dir_path):
                    logger.error(f"Required directory not found: {dir_path}")
                    return False
                if not os.access(dir_path, os.R_OK | os.X_OK):
                    logger.error(f"Insufficient permissions for directory: {dir_path}")
                    return False

            # Check write permissions for output directories
            output_dir = os.path.dirname(CONFIG['COMPREHENSIVE_RESULTS'])
            if not os.access(output_dir, os.W_OK):
                logger.error(f"No write permission for output directory: {output_dir}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking file paths: {e}")
            return False

    def cleanup_temp_files():
        """Clean up temporary files created during processing."""
        try:
            temp_patterns = [
                "*.tmp",
                "*.log",
                "~*.xlsx"
            ]

            cleaned_files = []
            for pattern in temp_patterns:
                for file_path in glob.glob(pattern):
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {file_path}: {e}")

            if cleaned_files:
                logger.info(f"Cleaned up temporary files: {cleaned_files}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # Add any missing imports
    import glob
    from pathlib import Path
    from typing import Dict, List, Tuple, Optional, Union

    # Type hints for key functions
    def extract_dft_data() -> pd.DataFrame:
        """Previous implementation"""
        pass

    def calculate_descriptors(
            data: pd.DataFrame,
            constants: Dict[str, float]
    ) -> pd.DataFrame:
        """Previous implementation"""
        pass

    def prepare_dataset() -> pd.DataFrame:
        """Previous implementation"""
        pass

    def train_and_evaluate_model(
            data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Previous implementation"""
        pass

    if __name__ == "__main__":
        logger.error("This file should be imported, not run directly. Please run main.py instead.")

def main():
    """Main execution function for the PCE prediction pipeline."""
    try:
        # Start time
        start_time = datetime.now()
        logger.info(f"Starting PCE prediction pipeline at {start_time}")

        # Validate directories
        validate_directories()

        # Prepare dataset
        logger.info("Preparing dataset...")
        data = prepare_dataset()

        # Calculate additional descriptors
        logger.info("Calculating additional descriptors...")
        data = calculate_descriptors(data, CONSTANTS)

        # Train model and get results
        logger.info("Training model and generating predictions...")
        results, feature_importance = train_and_evaluate_model(data)

        # Save results
        logger.info("Saving results...")
        results.to_excel(CONFIG['COMPREHENSIVE_RESULTS'], index=False)

        # Calculate and log execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Pipeline completed. Total execution time: {execution_time}")

        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info("-" * 50)
        logger.info(f"Total number of compounds: {len(results)}")
        logger.info(f"Training set size: {len(results[results['Dataset'] == 'Training'])}")
        logger.info(f"Test set size: {len(results[results['Dataset'] == 'Testing'])}")
        logger.info("\nPrediction Metrics:")
        logger.info(f"Mean Prediction Error: {results['Prediction_Error'].mean():.4f}")
        logger.info(f"Mean Prediction Accuracy: {results['Prediction_Accuracy_Percentage'].mean():.2f}%")

        # Print top features
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False


def run_additional_analysis():
    """
    Run additional analysis for different DFT methods and solvation conditions.
    This addresses the TODOs from the original code.
    """
    try:
        # List of DFT methods to analyze
        dft_methods = ['LDA', 'PBE', 'B3LYP']

        # List of solvation conditions
        solvation_conditions = [
            'exp',  # Matches experimental protocols
            'ethanol',  # Unified ethanol solvation
            'gas'  # Gas phase calculations
        ]

        results_summary = []

        for method in dft_methods:
            for solvation in solvation_conditions:
                logger.info(f"\nProcessing {method} with {solvation} solvation...")

                # Update configuration for current analysis
                current_config = CONFIG.copy()
                current_config.update({
                    'DFT_OUTPUT_DIR': f'./outputs_{method}_{solvation}',
                    'MOL_DIR': f'./mol_{method}_{solvation}',
                    'DFT_DATASET': f'dyes_DFT_{method}_{solvation}_dataset.xlsx',
                    'COMPREHENSIVE_RESULTS': f'dyes_comprehensive_PCE_results_{method}_{solvation}.xlsx',
                    'MODEL_OUTPUT': f'pce_prediction_model_{method}_{solvation}.joblib',
                    'FEATURE_IMPORTANCE': f'feature_importance_{method}_{solvation}.xlsx'
                })

                # Skip if directory doesn't exist
                if not os.path.exists(current_config['DFT_OUTPUT_DIR']):
                    logger.warning(f"Directory {current_config['DFT_OUTPUT_DIR']} not found. Skipping...")
                    continue

                try:
                    # Prepare dataset
                    data = prepare_dataset()
                    data = calculate_descriptors(data, CONSTANTS)

                    # Train model and get results
                    results, _ = train_and_evaluate_model(data)

                    # Collect summary statistics
                    summary = {
                        'Method': method,
                        'Solvation': solvation,
                        'Mean_Prediction_Error': results['Prediction_Error'].mean(),
                        'Mean_Prediction_Accuracy': results['Prediction_Accuracy_Percentage'].mean(),
                        'RMSE': np.sqrt(mean_squared_error(results['PCE'], results['Predicted_PCE'])),
                        'MAE': mean_absolute_error(results['PCE'], results['Predicted_PCE'])
                    }

                    results_summary.append(summary)

                except Exception as e:
                    logger.error(f"Error processing {method} with {solvation} solvation: {e}")
                    continue

        # Create summary DataFrame and save
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_df.to_excel('method_comparison_summary.xlsx', index=False)
            logger.info("\nMethod Comparison Summary:")
            logger.info(summary_df.to_string())

        return True

    except Exception as e:
        logger.error(f"Error in additional analysis: {e}")
        return False


if __name__ == "__main__":
    # Run main pipeline
    success = main()

    if success:
        # Run additional analysis if main pipeline succeeds
        logger.info("\nStarting additional analysis for different methods and solvation conditions...")
        run_additional_analysis()
    else:
        logger.error("Main pipeline failed. Skipping additional analysis.")