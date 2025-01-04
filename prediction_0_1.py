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
    'TEST_SIZE': 0.15,  # 0.1 got 4 samples, 0.2 got 8 and 0.15 got 6
    'RANDOM_STATE': 42,
    
    # Random Forest Parameters
    'RF_PARAMS': {
        'n_estimators': 500,  # Increased from 200
        'max_depth': 6,      # Slightly increased
        'min_samples_split': 3,  # Reduced
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'max_samples': 0.8,
        'n_jobs': -1
    },
    
    # XGBoost Parameters
    'XGBOOST_PARAMS': {
        'n_estimators': 1000,           # Increased from 150
        'learning_rate': 0.03,         # Increased from 0.01
        'max_depth': 3,                # Reduced from 6
        'min_child_weight': 1,         # Reduced from 3
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0,                 # Reduced from 0.1
        'reg_alpha': 0,              # Removed L1 regularization
        'reg_lambda': 0.1,             # Reduced L2 regularization
        'scale_pos_weight': 1.0,
    },
    
    # Cross-validation Parameters
    'CV_FOLDS': 5,
    'CV_REPEATS': 3,
    
    # Feature Selection
    'FEATURE_SELECTION': {
        'n_features_to_select': 15,    # Increased from 10
        'importance_threshold': 0.02,   # Decreased from 0.05
        'correlation_threshold': 0.95,  # Keep at 0.95
        'force_include': ['HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment']  # Force include these features
    },
    
    # Ensemble Parameters
    'ENSEMBLE': {
        'weights': [0.6, 0.4]  # RF weight, XGBoost weight
    },
    
    # Feature Scaling
    'FEATURE_SCALING': True
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


def select_features(X, y, feature_names):
    """
    Select the most important features using multiple methods.
    
    Args:
        X (np.array or pd.DataFrame): Feature matrix
        y (np.array or pd.Series): Target values
        feature_names (list): List of feature names
    
    Returns:
        list: Selected feature names
    """
    try:
        # First, ensure forced features are included
        forced_features = [f for f in CONFIG['FEATURE_SELECTION']['force_include'] 
                         if f in feature_names]
        remaining_features = [f for f in feature_names 
                            if f not in forced_features]
        
        # Initialize Random Forest for feature selection on remaining features
        X_remaining = X[remaining_features]
        
        rf_params = {
            'n_estimators': CONFIG['RF_PARAMS']['n_estimators'],
            'max_depth': CONFIG['RF_PARAMS']['max_depth'],
            'min_samples_split': CONFIG['RF_PARAMS']['min_samples_split'],
            'min_samples_leaf': CONFIG['RF_PARAMS']['min_samples_leaf'],
            'max_features': CONFIG['RF_PARAMS']['max_features'],
            'n_jobs': CONFIG['RF_PARAMS']['n_jobs'],
            'random_state': CONFIG['RANDOM_STATE']
        }
        
        rf_model = RfPCEModel(**rf_params)
        rf_model._create_model()
        rf_model.model.fit(X_remaining, y)
        
        # Get feature importance from Random Forest
        rf_importance = pd.DataFrame({
            'feature': remaining_features,
            'importance': rf_model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X_remaining, y)
        mi_importance = pd.DataFrame({
            'feature': remaining_features,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Combine both methods
        combined_importance = pd.DataFrame({
            'feature': remaining_features,
            'rf_importance': rf_importance.set_index('feature').loc[remaining_features, 'importance'],
            'mi_importance': mi_importance.set_index('feature').loc[remaining_features, 'importance']
        })
        
        # Normalize scores
        combined_importance['rf_importance'] = combined_importance['rf_importance'] / combined_importance['rf_importance'].max()
        combined_importance['mi_importance'] = combined_importance['mi_importance'] / combined_importance['mi_importance'].max()
        
        # Calculate combined score
        combined_importance['combined_score'] = (
            combined_importance['rf_importance'] * 0.7 +  # Give more weight to RF
            combined_importance['mi_importance'] * 0.3    # Less weight to MI
        )
        
        # Sort by combined score
        combined_importance = combined_importance.sort_values('combined_score', ascending=False)
        
        # Select top features based on configuration (excluding forced features)
        n_additional_features = min(
            CONFIG['FEATURE_SELECTION']['n_features_to_select'] - len(forced_features),
            len(remaining_features)
        )
        selected_features = forced_features + combined_importance.head(n_additional_features).index.tolist()
        
        # Remove highly correlated features (but never remove forced features)
        if len(selected_features) > 1:
            X_selected = X[selected_features]
            corr_matrix = pd.DataFrame(X_selected).corr().abs()
            
            # Create a mask for highly correlated pairs
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop (never drop forced features)
            to_drop = []
            for i in range(len(upper.columns)):
                for j in range(i + 1, len(upper.columns)):
                    if upper.iloc[i, j] > CONFIG['FEATURE_SELECTION']['correlation_threshold']:
                        feat_i = upper.columns[i]
                        feat_j = upper.columns[j]
                        
                        # Skip if both features are forced
                        if feat_i in forced_features and feat_j in forced_features:
                            continue
                        
                        # Never drop a forced feature
                        if feat_i in forced_features:
                            to_drop.append(feat_j)
                        elif feat_j in forced_features:
                            to_drop.append(feat_i)
                        # For non-forced features, drop the one with lower importance
                        else:
                            if combined_importance.loc[feat_i, 'combined_score'] < combined_importance.loc[feat_j, 'combined_score']:
                                to_drop.append(feat_i)
                            else:
                                to_drop.append(feat_j)
            
            # Remove duplicates from to_drop list
            to_drop = list(set(to_drop))
            
            # Update selected features
            selected_features = [f for f in selected_features if f not in to_drop]
        
        logger.info(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        raise


def prepare_stratified_splits(X, y):
    """
    Prepare stratified splits for cross-validation.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target values
    
    Returns:
        tuple: X, y, and stratification labels
    """
    # Ensure y is a numpy array and reshape it
    y_np = np.array(y).reshape(-1, 1)
    
    # Create bins for stratification
    n_bins = min(5, len(y) // 5)  # Ensure we don't have too many bins for small datasets
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    stratification_labels = kbd.fit_transform(y_np).ravel()
    
    return X, y, stratification_labels


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

        # Save complete dataset with all descriptors before feature selection and scaling
        complete_data = data.copy()

        # Select features
        feature_columns = [col for col in data.columns if col not in ['File', 'PCE', 'SMILES']]
        selected_features = select_features(data[feature_columns], data['PCE'], feature_columns)
        data = data[['File', 'PCE', 'SMILES'] + selected_features].copy()

        # Scale features
        if CONFIG['FEATURE_SCALING']:
            scaler = StandardScaler()
            data[selected_features] = scaler.fit_transform(data[selected_features])

        return data, complete_data

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
        visualizer.output_dir = output_dir.parent / "plots"
        os.makedirs(visualizer.output_dir, exist_ok=True)
        visualizer.visualize_results(model, model_name)
        logging.info(f"Generating visualizations for {model_name}...")
    except Exception as e:
        logging.error(f"Error generating visualizations for {model_name}: {e}")


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
            f"pce_prediction_model_RF_{DFT_METHOD}_eth-{int(datetime.now().timestamp() * 1000)}.joblib"
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
            f"pce_prediction_model_XGBoost_{DFT_METHOD}_eth-{int(datetime.now().timestamp() * 1000)}.joblib"
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
            xgb_params=CONFIG['XGBOOST_PARAMS']
        )
        ensemble_model._create_model()
        ensemble_metrics, ensemble_results = ensemble_model.train(data)

        # Prepare and save Ensemble results
        ensemble_pce_results = prepare_pce_results(ensemble_results)
        ensemble_results_path = os.path.join(
            execution_logger.results_dir,
            f"PCE_results_Ensemble_{DFT_METHOD}_eth.xlsx"
        )
        ensemble_pce_results.to_excel(ensemble_results_path, index=False)
        logger.info(f"Ensemble results file saved to: {ensemble_results_path}")

        # Generate visualizations for Ensemble
        logger.info("Generating visualizations for Ensemble...")
        try:
            ensemble_importance = ensemble_model.get_feature_importance()
            generate_visualizations(ensemble_model, "Ensemble", execution_logger.results_dir)
        except Exception as e:
            logger.error(f"Error generating visualizations for Ensemble: {e}")

        # Save Ensemble model
        ensemble_model_path = os.path.join(
            execution_logger.models_dir,
            f"pce_prediction_model_Ensemble_{DFT_METHOD}_eth-{int(datetime.now().timestamp() * 1000)}.joblib"
        )
        ensemble_model.save_model(ensemble_model_path)
        logger.info(f"Ensemble model saved to: {ensemble_model_path}")

        # Save execution info
        execution_info_path = execution_logger.save_execution_info(
            config={
                'rf': CONFIG['RF_PARAMS'],
                'xgboost': CONFIG['XGBOOST_PARAMS'],
                'ensemble': {
                    'rf_params': CONFIG['RF_PARAMS'],
                    'xgb_params': CONFIG['XGBOOST_PARAMS']
                }
            },
            metrics={
                'rf': rf_metrics,
                'xgboost': xgb_metrics,
                'ensemble': ensemble_metrics
            },
            results={
                'rf': rf_pce_results,
                'xgboost': xgb_pce_results,
                'ensemble': ensemble_pce_results
            }
        )
        logger.info(f"Comprehensive execution info saved to: {execution_info_path}")

        # Create model comparison DataFrame
        model_comparison = pd.DataFrame({
            'Model': ['RF', 'XGBoost', 'Ensemble'],
            'Test_R2': [
                rf_metrics['test']['r2'],
                xgb_metrics['test']['r2'],
                ensemble_metrics['test']['r2']
            ],
            'Test_RMSE': [
                rf_metrics['test']['rmse'],
                xgb_metrics['test']['rmse'],
                ensemble_metrics['test']['rmse']
            ],
            'Test_MAE': [
                rf_metrics['test']['mae'],
                xgb_metrics['test']['mae'],
                ensemble_metrics['test']['mae']
            ],
            'CV_R2_Mean': [
                rf_metrics.get('cv', {}).get('r2_mean', 'N/A'),
                xgb_metrics.get('cv', {}).get('r2_mean', 'N/A'),
                ensemble_metrics.get('cv', {}).get('r2_mean', 'N/A')
            ],
            'CV_R2_Std': [
                rf_metrics.get('cv', {}).get('r2_std', 'N/A'),
                xgb_metrics.get('cv', {}).get('r2_std', 'N/A'),
                ensemble_metrics.get('cv', {}).get('r2_std', 'N/A')
            ],
            'Execution_Time': [
                rf_metrics.get('execution_time', 'N/A'),
                xgb_metrics.get('execution_time', 'N/A'),
                ensemble_metrics.get('execution_time', 'N/A')
            ]
        })

        # Save model comparison
        comparison_path = os.path.join(
            execution_logger.results_dir,
            f"model_comparison_{DFT_METHOD}_eth.xlsx"
        )
        model_comparison.to_excel(comparison_path, index=False)
        logger.info(f"\nModel comparison saved to: {comparison_path}")

        # Log completion
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"\nPipeline completed. Total execution time: {execution_time}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
