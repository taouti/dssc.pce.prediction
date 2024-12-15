import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline


# Configuration
class Config:
    """Configuration parameters for the molecular PCE prediction model."""

    # Computational Chemistry Constants
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

    # Paths
    OUTPUT_DIR = './outputs_LDA'
    MOL_DIR = './mol_LDA'
    EXPERIMENTAL_FILE = 'dyes_experimental_dataset.xlsx'

    # Model Hyperparameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.2

    # GridSearch Parameters
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


# Setup Logging
def setup_logging() -> logging.Logger:
    """Configure and return a logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pce_prediction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def calculate_molecular_descriptors(mol_file: str) -> Optional[Dict]:
    """
    Calculate molecular descriptors from a mol file.

    Args:
        mol_file (str): Path to the mol file

    Returns:
        Optional[Dict]: Dictionary of molecular descriptors
    """
    try:
        # Read molecule from mol file
        mol = Chem.MolFromMolFile(mol_file)

        if mol is None:
            logger.error(f"Could not create molecule from file: {mol_file}")
            return None

        # Convert to SMILES for additional descriptor calculation
        smiles = Chem.MolToSmiles(mol)

        # Mass Calculation
        mass = Descriptors.ExactMolWt(mol)

        # Mordred Descriptor Calculation
        calc = Calculator(descriptors)
        mordred_desc = calc(mol)

        # Convert to dictionary
        desc_dict = mordred_desc.asdict()

        # Add mass and SMILES to descriptors
        desc_dict['Mass'] = mass
        desc_dict['SMILES'] = smiles

        return desc_dict

    except Exception as e:
        logger.error(f"Error calculating descriptors for mol file {mol_file}: {e}")
        return None


def extract_dft_data(output_dir: str) -> pd.DataFrame:
    """
    Extract DFT data from output files.

    Args:
        output_dir (str): Directory containing output files

    Returns:
        pd.DataFrame: DataFrame with extracted DFT data
    """
    patterns = {
        'HOMO': r"Energy of Highest Occupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
        'LUMO': r"Energy of Lowest Unoccupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
        'Dipole_Moment': r"dipole magnitude:\s+[-\d.]+\s+au\s+([-\d.]+)\s+debye"
    }

    cosmo_patterns = {
        'Total_Energy_Hartree': r"Total energy\s*=\s*([-?\d.]+)",
        'Solvation_Energy_eV': r"Dielectric \(solvation\) energy\s+=\s+[-\d.]+\s+([-\d.]+)",
        'Surface_Area_A2': r"Surface area of cavity \[A\*\*2\]\s*=\s+([-\d.]+)",
        'Molecular_Volume_A3': r"Total Volume of cavity \[A\*\*3\]\s*=\s+([-\d.]+)",
        'COSMO_Screening_Charge': r"cosmo\s*=\s*([-\d.]+)"
    }

    excitation_pattern = r"\s*\d+ ->\s*\d+\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)"

    data = []

    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(output_dir, filename)
            record = {'File': filename.replace('.txt', '')}

            try:
                with open(file_path, 'r') as file:
                    content = file.read()

                    for key, pattern in {**patterns, **cosmo_patterns}.items():
                        matches = re.findall(pattern, content)
                        record[key] = float(matches[-1]) if matches else None

                    excitations = re.findall(excitation_pattern, content)
                    if excitations:
                        max_excitation = max(
                            [(float(ex[0]), float(ex[1]), float(ex[2])) for ex in excitations],
                            key=lambda x: x[2]
                        )
                        record['Max_Absorption_nm'] = max_excitation[1]
                        record['Max_f_osc'] = max_excitation[2]
                    else:
                        record['Max_Absorption_nm'] = None
                        record['Max_f_osc'] = None

                data.append(record)

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")

    return pd.DataFrame(data)


def calculate_descriptors(data: pd.DataFrame, constants: Dict) -> pd.DataFrame:
    """
    Calculate additional molecular descriptors.

    Args:
        data (pd.DataFrame): Input DataFrame
        constants (Dict): Dictionary of constants

    Returns:
        pd.DataFrame: DataFrame with additional descriptors
    """
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


def prepare_data(config: Config) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Prepare data for machine learning model.

    Args:
        config (Config): Configuration object

    Returns:
        Tuple of scaled feature matrices and target variables
    """
    # Extract DFT Data
    dft_df = extract_dft_data(config.OUTPUT_DIR)
    dft_df.to_excel('dyes_DFT_LDA_dataset.xlsx', index=False)

    # Load Experimental Dataset
    experimental_df = pd.read_excel(config.EXPERIMENTAL_FILE)
    experimental_df.rename(columns={
        'Dye': 'File',
        'expPCE': 'PCE'
    }, inplace=True)

    # Calculate Mordred Descriptors
    mordred_descriptors_list = []
    for mol_filename in os.listdir(config.MOL_DIR):
        if mol_filename.endswith('.mol'):
            file_name = mol_filename.replace('.mol', '')
            mol_path = os.path.join(config.MOL_DIR, mol_filename)

            desc = calculate_molecular_descriptors(mol_path)
            if desc:
                desc['File'] = file_name
                mordred_descriptors_list.append(desc)

    mordred_df = pd.DataFrame(mordred_descriptors_list)

    # Normalize File Columns
    dft_df['File'] = dft_df['File'].str.strip().str.lower()
    experimental_df['File'] = experimental_df['File'].str.strip().str.lower()
    mordred_df['File'] = mordred_df['File'].str.strip().str.lower()

    # Merge Datasets
    data = dft_df.merge(experimental_df, on='File', how='inner').merge(mordred_df, on='File', how='inner')

    # Handle Missing Data
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Calculate Additional Descriptors
    data = calculate_descriptors(data, config.CONSTANTS)

    # Apply mutual information to filter features
    mi_scores = mutual_info_regression(data[numeric_columns], data["PCE"])
    mi_scores = pd.Series(mi_scores, index=numeric_columns).sort_values(ascending=False)
    selected_features = mi_scores[mi_scores > 0.01].index.tolist()

    # Select features and target
    X = data[selected_features]
    y = data['PCE']

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    param_grid: Dict,
    random_seed: int
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train and evaluate a Random Forest model.

    Args:
        X_train (np.ndarray): Training feature set
        X_test (np.ndarray): Testing feature set
        y_train (pd.Series): Training target values
        y_test (pd.Series): Testing target values
        param_grid (Dict): Hyperparameter grid for GridSearchCV
        random_seed (int): Random seed for reproducibility

    Returns:
        Tuple: Best model and a dictionary of evaluation metrics
    """
    logger.info("Initializing Random Forest Regressor and GridSearchCV.")

    # Initialize model and grid search
    rf = RandomForestRegressor(random_state=random_seed)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    # Train the model
    logger.info("Training the model using GridSearchCV.")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info(f"Best model parameters: {grid_search.best_params_}")

    # Predictions and evaluation
    logger.info("Evaluating the model.")
    y_pred = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    }

    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")

    return best_model, metrics

# Main Execution
if __name__ == "__main__":
    # Prepare Data
    config = Config()
    X_train, X_test, y_train, y_test = prepare_data(config)

    # Train and Evaluate Model
    best_model, evaluation_metrics = train_and_evaluate_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        param_grid=config.PARAM_GRID,
        random_seed=config.RANDOM_SEED
    )

    # Save the best model
    import joblib
    model_path = os.path.join(config.OUTPUT_DIR, 'best_rf_model.pkl')
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        import json
        json.dump(evaluation_metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to {metrics_path}")