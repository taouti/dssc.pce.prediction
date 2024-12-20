import os
import re
import logging
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from datetime import datetime

# List of packages to check
packages = [
    "os", "re", "logging", "joblib", "pandas", "numpy", 
    "rdkit", "mordred", "sklearn", "datetime"
]

# Retrieve package versions
def get_package_versions(package_names):
    versions = {}
    for package in package_names:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                versions[package] = module.__version__
            elif hasattr(module, 'version'):
                versions[package] = module.version
            else:
                versions[package] = "Version info not found"
        except ImportError:
            versions[package] = "Not installed"
    return versions

# Save versions to requirements.txt
def save_versions_to_file(versions, filename="requirements.txt"):
    with open(filename, "w") as file:
        for package, version in versions.items():
            # If version info is available, write package and version
            if version != "Version info not found" and version != "Not installed":
                file.write(f"{package}=={version}\n")
            # If not installed, note it in a comment
            elif version == "Not installed":
                file.write(f"# {package} not installed\n")
            else:
                file.write(f"# {package} version info not found\n")

# Get versions and save to file
versions = get_package_versions(packages)
save_versions_to_file(versions)

print(f"Package versions saved to requirements.txt")
