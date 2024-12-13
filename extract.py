import os
import re
import csv

# Define the directory containing the files (adjust as needed)
directory_path = './outputs_PBE'

# Define patterns for the required data (HOMO, LUMO, Dipole Moment)
patterns = {
    'HOMO': r"Energy of Highest Occupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'LUMO': r"Energy of Lowest Unoccupied Molecular Orbital:\s+[-\d.]+Ha\s+([-\d.]+)eV",
    'Dipole_Moment': r"dipole magnitude:\s+[-\d.]+\s+au\s+([-\d.]+)\s+debye"
}

# COSMO-related patterns (for last appearance)
cosmo_patterns = {
    'Total_Energy_Hartree': r"Total energy\s*=\s*([-?\d.]+)",
    'Solvation_Energy_eV': r"Dielectric \(solvation\) energy\s+=\s+[-\d.]+\s+([-\d.]+)", 
    'Surface_Area_A2': r"Surface area of cavity \[A\*\*2\]\s*=\s+([-\d.]+)",
    'Molecular_Volume_A3': r"Total Volume of cavity \[A\*\*3\]\s*=\s+([-\d.]+)",
    'COSMO_Screening_Charge': r"cosmo\s*=\s*([-\d.]+)"
}

# Binding Energy pattern (in eV)
binding_energy_pattern = r"binding energy\s*[-\d.]+\s*Ha\s*([-+]?\d*\.\d+|\d+)\s*eV"


# Pattern to extract TDDFT excitation data
excitation_pattern = r"\s*\d+ ->\s*\d+\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)"

# Initialize a list to store results for all files
results = []

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(directory_path, filename)
        data = {'File': filename}

        with open(file_path, 'r') as file:
            content = file.read()

            # Extract HOMO, LUMO, and Dipole Moment (use last occurrence for HOMO and LUMO)
            for key, pattern in patterns.items():
                matches = re.findall(pattern, content)  # Find all matches
                if matches:  # Check if there are matches
                    data[key] = float(matches[-1])  # Use the last match
                else:
                    data[key] = None  # Mark missing data as None

            # Extract COSMO-related descriptors (use last occurrence)
            for key, pattern in cosmo_patterns.items():
                matches = re.findall(pattern, content)  # Find all matches
                if matches:  # Check if there are matches
                    data[key] = float(matches[-1])  # Use the last match
                else:
                    data[key] = None  # Mark missing data as None

            # Extract Binding Energy in eV (last occurrence)
            binding_energy_matches = re.findall(binding_energy_pattern, content)
            if binding_energy_matches:
                data['Binding_Energy_eV'] = float(binding_energy_matches[-1])  # Use last occurrence
            else:
                data['Binding_Energy_eV'] = None  # Mark missing data as None

            # Calculate Surface-to-Volume Ratio using the last Surface Area and Molecular Volume
            if data.get('Surface_Area_A2') and data.get('Molecular_Volume_A3'):
                data['Surface_to_Volume_Ratio'] = data['Surface_Area_A2'] / data['Molecular_Volume_A3']
            else:
                data['Surface_to_Volume_Ratio'] = None

            # Extract TDDFT excitations
            excitations = re.findall(excitation_pattern, content)
            if excitations:
                # Convert to list of tuples with numeric values
                excitations = [(float(ex[0]), float(ex[1]), float(ex[2])) for ex in excitations]
                # Find the row with the maximum oscillator strength
                max_excitation = max(excitations, key=lambda x: x[2])
                data['Max_Absorption_nm'] = max_excitation[1]
                data['Max_f_osc'] = max_excitation[2]
            else:
                data['Max_Absorption_nm'] = None
                data['Max_f_osc'] = None

        results.append(data)

# Save results to a CSV file
output_file = 'extracted_data.csv'
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = [
        'File', 'HOMO', 'LUMO', 'Dipole_Moment',
        'Total_Energy_Hartree', 'Solvation_Energy_eV', 'Surface_Area_A2', 'Molecular_Volume_A3',
        'COSMO_Screening_Charge', 'Surface_to_Volume_Ratio', 'Max_Absorption_nm', 'Max_f_osc', 'Binding_Energy_eV'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Extraction completed. Results saved to '{output_file}'.")
