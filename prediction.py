import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

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
        record = {'File': filename}

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
                max_excitation = max([(float(ex[0]), float(ex[1]), float(ex[2])) for ex in excitations], key=lambda x: x[2])
                record['Max_Absorption_nm'] = max_excitation[1]
                record['Max_f_osc'] = max_excitation[2]
            else:
                record['Max_Absorption_nm'] = None
                record['Max_f_osc'] = None

        data.append(record)

# Step 2: Read Molecular Files and Calculate Descriptors
mol_dir = './mol_LDA'
calc = Calculator(descriptors, ignore_3D=True)

for filename in os.listdir(mol_dir):
    if filename.endswith('.mol'):
        file_path = os.path.join(mol_dir, filename)
        mol = Chem.MolFromMolFile(file_path, sanitize=True)
        if mol:
            record = {desc[0]: desc[1] for desc in calc(mol).items()}
            record['Mass'] = Descriptors.ExactMolWt(mol)
            record['File'] = filename
            data.append(record)

# Save Extracted Data
df = pd.DataFrame(data)
df.to_excel('dyes_data_PBE.xlsx', index=False)

# Step 3: Load Dataset
input_file = "dyes_data_PBE.xlsx"
data = pd.read_excel(input_file)

# Handle Missing Data
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Step 4: Calculate Descriptors
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

# Step 5: Define Features and Target for Ranking
features = [
    'deltaE_LCB', 'deltaE_RedOxH', 'deltaE_HL', 'IP', 'EA', 
    'elnChemPot', 'chemHardness', 'electronegativity', 
    'electrophilicityIndex', 'electroacceptingPower', 'electrodonatingPower', 'LHE', 
    'Dipole_Moment', 'Solvation_Energy_eV', 'Surface_Area_A2', 'Molecular_Volume_A3', 'Mass'
]

X = data[features]

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Step 7: Train Ranking Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, X_train.mean(axis=1))

# Step 8: Ranking Prediction
data['Ranking_Score'] = rf_model.predict(X_scaled)
data['Ranking'] = data['Ranking_Score'].rank(ascending=False).astype(int)

# Save Results
data.to_excel('dyes_ranking_results_PBE.xlsx', index=False)

print("Ranking completed. Results saved to 'dyes_ranking_results_PBE.xlsx'.")
