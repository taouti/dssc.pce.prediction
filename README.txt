# DSSC Dye Performance Prediction

This repository contains the implementation of the machine learning pipeline described in:

> DFT and Machine Learning Integration to Predict Efficiency of Modified Metal-Free Dyes in DSSCs  
> Journal of Molecular Graphics and Modelling (2025)  
> DOI: https://doi.org/10.1016/j.jmgm.2025.108975

# Overview

The code implements an integrated computational framework that combines quantum chemical calculations (DFT/TDDFT) and Mordred descriptors with machine learning to predict the Power Conversion Efficiency (PCE) of dye molecules for Dye-Sensitized Solar Cells (DSSCs). The framework achieved a good performance with XGBoost (R² = 0.8904, MAE = 0.0030, RMSE = 0.0038) and successfully identified promising novel dye configurations.

# Key Features

- Integration of quantum-chemical descriptors (DFT) with cheminformatic descriptors (Mordred)
- Random Forest and XGBoost models are used
- Configurable model stability 
- Feature importance analysis for key PCE predictors 
- High-throughput screening capability for novel dye candidates

# Validated Results

The framework was successfully validated on novel dye derivatives of:
- C-PE3: (E)-10-methyl-9-(3-(10-methylacridin-9(10H)-ylidene)prop-1-en-1-yl)acridin-10-ium
- C-PE5: 10-methyl-9-((1E,3E)-5-(10-methylacridin-9(10H)-ylidene)penta-1,3-dien-1-yl)acridin-10-ium
- C-PE7: 10-methyl-9-((1E,3E,5E)-7-(10-methylacridin-9(10H)-ylidene)hepta-1,3,5-trien-1-yl)acridin-10-ium

Top candidates identified by both models:
- C3-PE5: Predicted PCE of 5.49%
- C3-PE7: Predicted PCE of 5.43%

# Features

- Automated extraction of DFT calculation results
- Molecular descriptor calculation using RDKit and Mordred
- Advanced feature selection and preprocessing
- Saved Random Forest and XGBoost trained models
- Cross-validation and robust model evaluation
- Comprehensive visualization of results
- Detailed logging system

# Directory Structure

- 'outputs_*_ethanol/': DFT calculation output files
- 'mol_*_ethanol/': Molecular structure files
- 'utils/': Utility functions for logging, visualization, and model classes
- 'log/': Log files directory
- 'prediction_0_1.py': Main prediction pipeline script

# Installation and Requirements

The code requires Python 3.7+ and the main dependencies found in 'requirements.txt'.


# Usage

## Data Preparation

1. DFT Calculations:
   - Perform DFT and TDDFT calculations for your dye molecules
   - Place output files in the appropriate 'outputs_*_ethanol' directory
   - Store molecular structure files in the corresponding 'mol_*_ethanol' directory

2. Experimental Data:
   - Prepare experimental PCE data in 'dyes_experimental_dataset.xlsx'
   - Ensure consistent formatting with the training dataset and naming for dyes

## Model Training and Prediction

Run the prediction pipeline:

python prediction_0_1.py

The pipeline will:
1. Extract quantum-chemical descriptors from DFT outputs
2. Calculate cheminformatic descriptors using Mordred
3. Perform feature selection and preprocessing
4. Train ensemble models (Random Forest and XGBoost)
5. Generate predictions and performance metrics
6. Create visualization plots for analysis

## Output and Results

The pipeline generates:
- Comprehensive results in Excel format ('PCE_results_*_eth.xlsx')
- Trained model files ('pce_prediction_model_*_eth.joblib')
- Performance visualization plots
- Detailed execution logs ('pce_prediction.log')

## Model Configuration

Key parameters in 'CONFIG' dictionary ('prediction_0_1.py'):
- Random states for model stability analysis
- Feature selection parameters
- Model hyperparameters for Random Forest and XGBoost
- Ensemble weights and preprocessing settings

## Limitations and Future Work

As noted in the research:
- Dataset size and diversity can be expanded
- Environmental effects (solvent interactions, dye-TiO₂ anchoring) could be incorporated
- Model refinements for broader DSSC applications are possible

## Citation

If you use this code in your research, please cite:

Mohammed Madani TAOUTI, Naceur SELMANE, Ali CHEKNANE, Noureddine BENAYA, Hikmat S. HILAL,
DFT and Machine Learning Integration to Predict Efficiency of Modified Metal-Free Dyes in DSSCs,
Journal of Molecular Graphics and Modelling,
2025,
108975,
ISSN 1093-3263,
https://doi.org/10.1016/j.jmgm.2025.108975.
(https://www.sciencedirect.com/science/article/pii/S109332632500035X)


## License

This code is provided for research purposes. Please cite the associated paper if you use this code in your research.
