import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
constants = {
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

# Function to calculate molecular descriptors from mol file
def calculate_molecular_descriptors(mol_file):
    try:
        # Read molecule from mol file
        mol = Chem.MolFromMolFile(mol_file)

        if mol is None:
            print(f"Could not create molecule from file: {mol_file}")
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
        print(f"Error calculating descriptors for mol file {mol_file}: {e}")
        return None

# Step 1: Extract Data from DFT Outputs
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

output_dir = './outputs_LDA'
data = []

for filename in os.listdir(output_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(output_dir, filename)
        record = {'File': filename.replace('.txt', '')}

        with open(file_path, 'r') as file:
            content = file.read()

            for key, pattern in patterns.items():
                matches = re.findall(pattern, content)
                record[key] = float(matches[-1]) if matches else None

            for key, pattern in cosmo_patterns.items():
                matches = re.findall(pattern, content)
                record[key] = float(matches[-1]) if matches else None

            excitations = re.findall(excitation_pattern, content)
            if excitations:
                max_excitation = max([(float(ex[0]), float(ex[1]), float(ex[2])) for ex in excitations],
                                     key=lambda x: x[2])
                record['Max_Absorption_nm'] = max_excitation[1]
                record['Max_f_osc'] = max_excitation[2]
            else:
                record['Max_Absorption_nm'] = None
                record['Max_f_osc'] = None

        data.append(record)

# Save DFT Extracted Data
dft_df = pd.DataFrame(data)
dft_df.to_excel('dyes_DFT_LDA_dataset.xlsx', index=False)

# Step 2: Load Experimental Dataset
experimental_file = 'dyes_experimental_dataset.xlsx'
experimental_df = pd.read_excel(experimental_file)

# Rename and Preprocess Columns
experimental_df.rename(columns={
    'Dye': 'File',
    'expPCE': 'PCE'
}, inplace=True)

# Calculate Mordred Descriptors from MOL files
mol_dir = './mol_LDA'
mordred_descriptors_list = []

for mol_filename in os.listdir(mol_dir):
    if mol_filename.endswith('.mol'):
        file_name = mol_filename.replace('.mol', '')
        mol_path = os.path.join(mol_dir, mol_filename)

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

# Step 3: Calculate Additional Descriptors
def calculate_descriptors(data, constants):
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

data = calculate_descriptors(data, constants)

# Identify numeric columns for feature selection
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Remove non-predictive, redundant, or experimental columns
excluded_columns = ['PCE', 'File', 'SMILES',
                    'expVoc_V', 'expIsc_mAcm-2', 'expFF']  # Excluded experimental columns
feature_columns = [col for col in numeric_columns if col not in excluded_columns]

# Prepare features and target
X = data[feature_columns]
y = data['PCE']

# Split data into train and test sets with labels
X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
    X, y, data['File'], test_size=0.1, random_state=42
)

# Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred_test = rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"MAE: {mae}, RMSE: {rmse}")

# Predict PCE for Entire Dataset
data_scaled = scaler.transform(X)
data['Predicted_PCE'] = rf_model.predict(data_scaled)

# Calculate Prediction Error and Accuracy
data['Prediction_Error'] = abs(data['Predicted_PCE'] - data['PCE'])
data['Prediction_Error_Percentage'] = (data['Prediction_Error'] / data['PCE']) * 100
data['Prediction_Accuracy_Percentage'] = abs(100 - data['Prediction_Error_Percentage'])

# Add Dataset Split Labels
data['Dataset'] = 'Training'
data.loc[data['File'].isin(test_files), 'Dataset'] = 'Testing'

# Reorder columns to put experimental and predicted values together
column_order = [
    'File', 'PCE', 'Predicted_PCE', 'Prediction_Error',
    'Prediction_Error_Percentage','Prediction_Accuracy_Percentage', 'Dataset'
] + [col for col in data.columns if col not in
    ['File', 'PCE', 'Predicted_PCE', 'Prediction_Error',
     'Prediction_Error_Percentage','Prediction_Accuracy_Percentage', 'Dataset']]

data = data[column_order]

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Save Comprehensive Results
data.to_excel('dyes_comprehensive_PCE_results_LDA.xlsx', index=False)

print("\nPCE prediction completed. Results saved to 'dyes_comprehensive_PCE_results_LDA.xlsx'.")

#TODO Add mean value under Prediction_Error, Prediction_Error_Percentage, and Prediction_Accuracy_Percentage
#TODO Extend the code so it does the prediction using results from LDA, PBE and B3LYP outputs.
#TODO The results should be obtained from:
# 1. Solvated dyes with solvent consistency to the experimental protocols
# 2. Solvated dyes in unified solvent (All results under Ethanol solvation)
# 3. Gas phase calculations on all dyes