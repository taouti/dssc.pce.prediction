from abc import ABC, abstractmethod
import dataclasses
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, DMatrix, train
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import os
import pickle
import json
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import seaborn as sns

from .regression_metrics_utils import ModelEvaluator
from .preprocessing import FeaturePreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)

class XGBoostWrapper(XGBRegressor, BaseEstimator):
    """Wrapper for XGBoost that implements scikit-learn's interface."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _more_tags(self):
        """Get more tags for scikit-learn compatibility."""
        return {
            'allow_nan': True,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
            'multioutput_only': False,
            'no_validation': False,
            'non_deterministic': False,
            'pairwise': False,
            'preserves_dtype': [np.float64],
            'requires_fit': True,
            'requires_y': True,
            'stateless': False,
            '_skip_test': False,
            '_xfail_checks': False,
            'poor_score': False,
            'requires_positive_y': False,
            'X_types': ['2darray']
        }

class BasePCEModel(ABC):
    """Base class for PCE prediction models."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            cv_folds: int = 5
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.preprocessor = None
        self.results = None
        self.is_trained = False
        self.evaluator = ModelEvaluator(n_splits=cv_folds, random_state=random_state)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str]]:
        """Prepare data for training using the preprocessor."""
        if self.preprocessor is None:
            # Create preprocessor with default config
            config = PreprocessingConfig(
                force_include_features=['HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment'],
                n_features_to_select=15,
                correlation_threshold=0.95
            )
            self.preprocessor = FeaturePreprocessor(config)

        # Create stratification labels
        y = data['PCE']
        n_bins = min(3, len(y) // 4)
        if n_bins >= 2:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            y_np = np.array(y).reshape(-1, 1)
            stratification_labels = kbd.fit_transform(y_np).ravel()
        else:
            stratification_labels = None

        # Process features
        X = self.preprocessor.fit_transform(data)
        file_names = data['File']
        
        return X, y, file_names, X.columns.tolist(), stratification_labels

    def train(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        """Train the model and return metrics and predictions."""
        X, y, file_names, feature_columns, stratification_labels = self.prepare_data(data)
        self.feature_columns = feature_columns  # Store feature columns for later use

        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            X, y, file_names,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratification_labels
        )

        # Train model
        self._create_model()
        self.model.fit(X_train, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_metrics = {
            'r2': r2_score(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred)
        }
        test_metrics = {
            'r2': r2_score(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred)
        }

        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }

        # Prepare results DataFrame
        results = pd.DataFrame()
        results['File'] = data['File']
        results['PCE'] = data['PCE']
        results['SMILES'] = data['SMILES']
        results['PredictedPCE'] = np.concatenate([train_pred, test_pred])
        results['Absolute_Error'] = abs(results['PredictedPCE'] - results['PCE'])
        results['Error_Percentage'] = (results['Absolute_Error'] / results['PCE']) * 100
        results['Dataset'] = 'Training'
        results.loc[results['File'].isin(test_files), 'Dataset'] = 'Testing'

        # Add selected features to results
        for feature in self.feature_columns:
            results[feature] = data[feature]

        self.results = results
        self.is_trained = True

        return metrics, results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X = self.preprocessor.transform(data)
        return self.model.predict(X)

    def get_results(self) -> pd.DataFrame:
        """Get the results from training."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.results

    def save_model(self, model_path: str):
        """Save the trained model and preprocessor."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model and preprocessor separately
        model_data = {
            'model': self.model,
            'model_type': self.__class__.__name__,
            'feature_columns': self.preprocessor.selected_features
        }

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Save preprocessor
        preprocessor_path = os.path.splitext(model_path)[0] + '_preprocessor.joblib'
        self.preprocessor.save(preprocessor_path)

        # Save metadata
        metadata = {
            'model_type': self.__class__.__name__,
            'n_features': len(self.preprocessor.selected_features),
            'feature_list': self.preprocessor.selected_features,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path
        }
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, model_path: str):
        """Load a trained model and its preprocessor."""
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']

        # Load preprocessor
        preprocessor_path = os.path.splitext(model_path)[0] + '_preprocessor.joblib'
        self.preprocessor = FeaturePreprocessor.load(preprocessor_path)
        self.is_trained = True

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass

class RfPCEModel(BasePCEModel):
    """Random Forest implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            n_estimators: int = 200,
            max_depth: int = 5,
            min_samples_split: int = 4,
            min_samples_leaf: int = 2,
            max_features: str = 'sqrt',
            max_samples: float = 0.8,
            n_jobs: int = -1,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.n_jobs = n_jobs

    def _create_model(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_df = pd.DataFrame({
            'Feature': self.preprocessor.selected_features,
            'Importance': self.model.feature_importances_
        })
        return importance_df.sort_values('Importance', ascending=False)


class XGBoostPCEModel(BasePCEModel):
    """XGBoost implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            n_estimators: int = 1000,
            learning_rate: float = 0.03,
            max_depth: int = 3,
            min_child_weight: int = 1,
            subsample: float = 0.7,
            colsample_bytree: float = 0.7,
            gamma: float = 0,
            reg_alpha: float = 0,
            reg_lambda: float = 0.1,
            scale_pos_weight: float = 1.0,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.feature_columns = None

    def _create_model(self) -> None:
        """Create XGBoost model."""
        pass  # Model will be created during training

    def train(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        """Train the model and return metrics and predictions."""
        X, y, file_names, feature_columns, stratification_labels = self.prepare_data(data)
        self.feature_columns = feature_columns  # Store feature columns for later use

        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test, temp_files, test_files = train_test_split(
            X, y, file_names,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratification_labels
        )

        # Further split temp into train and validation
        val_size = self.test_size / (1 - self.test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state
        )

        # Log dimensions for debugging
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # Create DMatrix objects for XGBoost
        dtrain = DMatrix(X_train, label=y_train)
        dval = DMatrix(X_val, label=y_val)
        dtest = DMatrix(X_test, label=y_test)

        # Set up parameters
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae']
        }

        # Train with early stopping
        self.model = train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Make predictions
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        test_pred = self.model.predict(dtest)

        # Combine train and validation predictions
        all_train_pred = np.concatenate([train_pred, val_pred])

        # Calculate metrics
        train_metrics = {
            'r2': r2_score(np.concatenate([y_train, y_val]), all_train_pred),
            'rmse': np.sqrt(mean_squared_error(np.concatenate([y_train, y_val]), all_train_pred)),
            'mae': mean_absolute_error(np.concatenate([y_train, y_val]), all_train_pred)
        }
        test_metrics = {
            'r2': r2_score(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred)
        }

        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }

        # Log predictions for debugging
        logger.info(f"Training predictions range: {all_train_pred.min():.3f} to {all_train_pred.max():.3f}")
        logger.info(f"Test predictions range: {test_pred.min():.3f} to {test_pred.max():.3f}")

        # Prepare results DataFrame
        results = pd.DataFrame()
        results['File'] = data['File']
        results['PCE'] = data['PCE']
        results['SMILES'] = data['SMILES']
        results['PredictedPCE'] = np.nan  # Initialize with NaN

        # Create a mapping of file to prediction
        train_val_files = pd.Series(temp_files)
        train_val_pred_dict = dict(zip(train_val_files, all_train_pred))
        test_pred_dict = dict(zip(test_files, test_pred))
        all_pred_dict = {**train_val_pred_dict, **test_pred_dict}

        # Set predictions using the mapping
        results['PredictedPCE'] = results['File'].map(all_pred_dict)
        results['Absolute_Error'] = abs(results['PredictedPCE'] - results['PCE'])
        results['Error_Percentage'] = (results['Absolute_Error'] / results['PCE']) * 100
        results['Dataset'] = 'Training'
        results.loc[results['File'].isin(test_files), 'Dataset'] = 'Testing'

        # Add selected features to results
        for feature in self.feature_columns:
            results[feature] = data[feature]

        self.results = results
        self.is_trained = True

        return metrics, results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If required features are missing
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        try:
            # Transform data using preprocessor
            if self.preprocessor:
                X = self.preprocessor.transform(data)
            else:
                X = data

            # Convert to DMatrix
            dtest = DMatrix(X)
            
            # Make predictions
            predictions = self.model.predict(dtest)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with XGB model: {str(e)}")
            raise RuntimeError(f"Error making predictions with XGB model: {str(e)}")

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_scores = self.model.get_score(importance_type='total_gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df

    def generate_visualizations(self, output_dir: str, data: pd.DataFrame) -> None:
        """Generate visualizations for the model."""
        if not self.is_trained:
            logger.error("Model must be trained before generating visualizations.")
            return

        try:
            # Create visualizations directory if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(self.results.loc[self.results['Dataset'] == 'Training', 'PCE'],
                       self.results.loc[self.results['Dataset'] == 'Training', 'PredictedPCE'],
                       alpha=0.6, label='Training')
            plt.scatter(self.results.loc[self.results['Dataset'] == 'Testing', 'PCE'],
                       self.results.loc[self.results['Dataset'] == 'Testing', 'PredictedPCE'],
                       alpha=0.6, label='Testing')
            plt.plot([0, max(self.results['PCE'])], [0, max(self.results['PCE'])],
                     'k--', label='Perfect Prediction')
            plt.xlabel('Actual PCE')
            plt.ylabel('Predicted PCE')
            plt.title('Actual vs Predicted PCE Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'actual_vs_predicted_xgboost.png'))
            plt.close()

            # Plot feature importance
            importance_df = self.get_feature_importance()
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('XGBoost Feature Importance')
            plt.xlabel('Importance Score (Total Gain)')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'feature_importance_xgboost.png'))
            plt.close()

            # Plot prediction errors
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(self.results)), 
                       self.results['Error_Percentage'],
                       c=self.results['Dataset'].map({'Training': 'blue', 'Testing': 'red'}),
                       alpha=0.6)
            plt.axhline(y=0, color='k', linestyle='--')
            plt.xlabel('Sample Index')
            plt.ylabel('Error Percentage')
            plt.title('Prediction Error Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'error_distribution_xgboost.png'))
            plt.close()

        except Exception as e:
            logger.error(f"Error generating visualizations for XGBoost: {str(e)}")


class LightGBMPCEModel(BasePCEModel):
    """LightGBM implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            n_estimators: int = 200,
            learning_rate: float = 0.05,
            num_leaves: int = 31,
            feature_fraction: float = 0.8,
            bagging_fraction: float = 0.8,
            bagging_freq: int = 5,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq

    def _create_model(self) -> None:
        self.model = LGBMRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_df = pd.DataFrame({
            'Feature': self.preprocessor.selected_features,
            'Importance': self.model.feature_importances_
        })
        return importance_df.sort_values('Importance', ascending=False)


class SvmPCEModel(BasePCEModel):
    """Support Vector Machine implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            kernel: str = 'rbf',
            C: float = 1.0,
            epsilon: float = 0.1,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def _create_model(self) -> None:
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if self.kernel != 'linear':
            raise ValueError("Feature importance is only available for linear kernel")

        importance_df = pd.DataFrame({
            'Feature': self.preprocessor.selected_features,
            'Importance': np.abs(self.model.coef_[0])
        })
        return importance_df.sort_values('Importance', ascending=False)


class BasePCEModelEnsemble(ABC):
    """Base class for ensemble PCE prediction models."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.feature_columns = None
        self.results = None
        self.is_trained = False
        self.scaler = StandardScaler()

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str]]:
        """Prepare data for training."""
        # Select features
        selected_features = [
            'HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment',
            'LogP', 'expIsc_mAcm-2', 'chemHardness', 'AromaticRings', 'expVoc_V',
            'elnChemPot', 'RingCount'
        ]
        
        # Store feature columns
        self.feature_columns = selected_features
        
        # Extract features and target
        X = data[selected_features]
        y = data['PCE']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Get file names and stratification labels
        file_names = data['File']
        stratification_labels = data['PCE'].apply(lambda x: int(x * 10) / 10)
        
        return X_scaled, y, file_names, self.feature_columns, stratification_labels

    def get_results(self) -> pd.DataFrame:
        """Get the results from training."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.results

    def save_model(self, path: str) -> None:
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'feature_columns': self.feature_columns,
            'results': self.results,
            'is_trained': self.is_trained,
            'scaler': self.scaler
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str) -> None:
        """Load a trained model from a file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.rf_weight = model_data['rf_weight']
        self.xgb_weight = model_data['xgb_weight']
        self.feature_columns = model_data['feature_columns']
        self.results = model_data['results']
        self.is_trained = model_data['is_trained']
        self.scaler = model_data['scaler']

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        pass


class EnsemblePCEModel(BasePCEModel):
    """Ensemble model combining RF and XGBoost predictions."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            cv_folds: int = 5,
            rf_params: Dict = {},
            xgb_params: Dict = {}
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.rf_params = rf_params
        self.xgb_params = xgb_params
        self.rf_model = None
        self.xgb_model = None
        self.rf_weight = 0.6
        self.xgb_weight = 0.4
        self.is_trained = False
        self.results = None

    def _create_model(self) -> None:
        """Create ensemble model components."""
        self.rf_model = RandomForestRegressor(
            random_state=self.random_state,
            **self.rf_params
        )

        self.xgb_model = XGBRegressor(
            random_state=self.random_state,
            enable_categorical=False,  # Disable categorical feature support
            use_label_encoder=False,   # Disable label encoding
            objective='reg:squarederror',  # Explicitly set regression objective
            **self.xgb_params
        )

    def train(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        """Train the ensemble model and return metrics and predictions."""
        X_train, X_test, y_train, y_test = self._prepare_data(data)
        
        # Train individual models
        self._create_model()
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        
        # Make predictions
        train_pred_rf = self.rf_model.predict(X_train)
        test_pred_rf = self.rf_model.predict(X_test)
        train_pred_xgb = self.xgb_model.predict(X_train)
        test_pred_xgb = self.xgb_model.predict(X_test)
        
        # Combine predictions with weights
        train_pred = (train_pred_rf * self.rf_weight + 
                     train_pred_xgb * self.xgb_weight)
        test_pred = (test_pred_rf * self.rf_weight + 
                    test_pred_xgb * self.xgb_weight)
        
        # Calculate metrics
        train_metrics = {
            'r2': r2_score(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred)
        }
        test_metrics = {
            'r2': r2_score(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred)
        }

        # Add cross-validation metrics
        cv_results = cross_validate(
            self.rf_model, X_train, y_train,
            cv=self.cv_folds,
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        )
        
        cv_metrics = {
            'r2_mean': float(np.mean(cv_results['test_r2'])),
            'r2_std': float(np.std(cv_results['test_r2'])),
            'rmse_mean': float(np.mean(np.sqrt(-cv_results['test_neg_mean_squared_error']))),
            'rmse_std': float(np.std(np.sqrt(-cv_results['test_neg_mean_squared_error']))),
            'mae_mean': float(np.mean(-cv_results['test_neg_mean_absolute_error'])),
            'mae_std': float(np.std(-cv_results['test_neg_mean_absolute_error']))
        }

        metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'cv': cv_metrics
        }
        
        # Prepare results DataFrame
        self.results = self._prepare_results(data, train_pred, test_pred)
        self.is_trained = True
        
        return metrics, self.results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if not self.is_trained:
            raise RuntimeError("Models must be trained before making predictions")
        
        try:
            # Use preprocessor to transform data with consistent features
            X = self.preprocessor.transform(data)
            
            # Make predictions with both models
            rf_pred = self.rf_model.predict(X)
            xgb_pred = self.xgb_model.predict(X)
            
            # Return weighted ensemble prediction
            return rf_pred * self.rf_weight + xgb_pred * self.xgb_weight
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Get feature importance from RF
            rf_importance = pd.DataFrame({
                'Feature': self.preprocessor.selected_features,
                'RF_Importance': self.rf_model.feature_importances_
            })
            
            # Get feature importance from XGBoost
            xgb_importance = pd.DataFrame({
                'Feature': self.preprocessor.selected_features,
                'XGB_Importance': self.xgb_model.feature_importances_
            })
            
            # Combine and calculate weighted importance
            importance_df = pd.merge(rf_importance, xgb_importance, on='Feature')
            importance_df['Importance'] = (
                importance_df['RF_Importance'] * self.rf_weight +
                importance_df['XGB_Importance'] * self.xgb_weight
            )
            
            return importance_df.sort_values('Importance', ascending=False)
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise RuntimeError(f"Failed to calculate feature importance: {str(e)}")

    def generate_visualizations(self, output_dir: str, data: pd.DataFrame) -> None:
        """Generate visualizations for the model."""
        if not self.is_trained:
            logger.error("Model must be trained before generating visualizations")
            return

        try:
            # Create visualizations directory if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(self.results.loc[self.results['Dataset'] == 'Training', 'PCE'],
                       self.results.loc[self.results['Dataset'] == 'Training', 'PredictedPCE'],
                       alpha=0.6, label='Training')
            plt.scatter(self.results.loc[self.results['Dataset'] == 'Testing', 'PCE'],
                       self.results.loc[self.results['Dataset'] == 'Testing', 'PredictedPCE'],
                       alpha=0.6, label='Testing')
            plt.plot([0, max(self.results['PCE'])], [0, max(self.results['PCE'])],
                     'k--', label='Perfect Prediction')
            plt.xlabel('Actual PCE')
            plt.ylabel('Predicted PCE')
            plt.title('Actual vs Predicted PCE Values (Ensemble)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'actual_vs_predicted_ensemble.png'))
            plt.close()

            # Plot feature importance if available
            if hasattr(self, 'get_feature_importance'):
                importance_df = self.get_feature_importance()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=importance_df, x='Importance', y='Feature')
                plt.title('Ensemble Feature Importance')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'feature_importance_ensemble.png'))
                plt.close()

            # Plot prediction errors
            plt.figure(figsize=(10, 6))
            plt.hist(self.results['Error_Percentage'], bins=30, alpha=0.7)
            plt.xlabel('Prediction Error (%)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'prediction_errors_ensemble.png'))
            plt.close()

            logger.info("Successfully generated visualizations for Ensemble model")

        except Exception as e:
            logger.error(f"Error generating visualizations for Ensemble: {str(e)}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        if self.preprocessor is None:
            # Create preprocessor with default config
            config = PreprocessingConfig(
                force_include_features=['HOMO', 'LUMO', 'Max_Absorption_nm', 'Max_f_osc', 'Dipole_Moment'],
                n_features_to_select=15,
                correlation_threshold=0.95
            )
            self.preprocessor = FeaturePreprocessor(config)

        # Check for required columns
        required_columns = ['PCE', 'File', 'SMILES']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Process features
        try:
            X = self.preprocessor.fit_transform(data)
        except Exception as e:
            logger.error(f"Error during feature preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to preprocess features: {str(e)}")

        y = data['PCE']
        file_names = data['File']
        
        # Create stratification labels
        n_bins = min(3, len(y) // 4)
        if n_bins >= 2:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            y_np = np.array(y).reshape(-1, 1)
            stratification_labels = kbd.fit_transform(y_np).ravel()
        else:
            stratification_labels = None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratification_labels
        )
        
        return X_train, X_test, y_train, y_test

    def _prepare_results(self, data: pd.DataFrame, train_pred: np.ndarray, test_pred: np.ndarray) -> pd.DataFrame:
        """Prepare results DataFrame."""
        # Prepare results DataFrame
        results = pd.DataFrame()
        results['File'] = data['File']
        results['PCE'] = data['PCE']
        results['SMILES'] = data['SMILES']
        results['PredictedPCE'] = np.concatenate([train_pred, test_pred])
        results['Absolute_Error'] = abs(results['PredictedPCE'] - results['PCE'])
        results['Error_Percentage'] = (results['Absolute_Error'] / results['PCE']) * 100
        results['Dataset'] = 'Training'
        results.loc[results['File'].isin(data['File'].tail(len(test_pred))), 'Dataset'] = 'Testing'

        # Add selected features to results
        for feature in self.preprocessor.selected_features:
            results[feature] = data[feature]

        return results

    def save_model(self, model_path: str):
        """Save the trained model and preprocessor."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model components
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'model_type': self.__class__.__name__,
            'feature_columns': self.preprocessor.selected_features
        }

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Save preprocessor
        preprocessor_path = os.path.splitext(model_path)[0] + '_preprocessor.joblib'
        self.preprocessor.save(preprocessor_path)

        # Save preprocessor features
        features_path = os.path.splitext(model_path)[0] + '_preprocessor_features.json'
        with open(features_path, 'w') as f:
            json.dump({
                'selected_features': self.preprocessor.selected_features,
                'forced_features': self.preprocessor.config.force_include_features,
                'n_features': len(self.preprocessor.selected_features)
            }, f, indent=2)

        # Save metadata
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
        metadata = {
            'model_type': self.__class__.__name__,
            'n_features': len(self.preprocessor.selected_features),
            'feature_list': self.preprocessor.selected_features,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, model_path: str):
        """Load a trained model and its preprocessor."""
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.rf_model = model_data['rf_model']
            self.xgb_model = model_data['xgb_model']
            self.rf_weight = model_data['rf_weight']
            self.xgb_weight = model_data['xgb_weight']
            self.feature_columns = model_data['feature_columns']

        # Load preprocessor
        preprocessor_path = os.path.splitext(model_path)[0] + '_preprocessor.joblib'
        self.preprocessor = FeaturePreprocessor.load(preprocessor_path)
        self.is_trained = True


class BasePCEModelEnsembleUncertainty:
    """Base class for PCE prediction model ensemble with uncertainty estimation."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            cv_folds: int = 5
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.feature_columns = None
        self.results = None
        self.is_trained = False
        self.scaler = StandardScaler()
        self.evaluator = ModelEvaluator(n_splits=cv_folds, random_state=random_state)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str]]:
        # Define columns to exclude from features
        excluded_columns = ['PCE', 'File', 'SMILES', 'expFF']  
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col not in excluded_columns]

        # Extract features and target
        X = data[self.feature_columns].copy()  
        y = data['PCE'].copy()
        file_names = data['File']

        # Create stratification labels for better split
        n_bins = min(3, len(y) // 4)  
        if n_bins >= 2:  
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            y_np = np.array(y).reshape(-1, 1)
            stratification_labels = kbd.fit_transform(y_np).ravel()
        else:
            stratification_labels = None

        # Handle any missing values
        X = X.fillna(X.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, file_names, self.feature_columns, stratification_labels

    def train(self, data: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        """Train the model and return metrics and predictions."""
        X_scaled, y, file_names, _, stratification_labels = self.prepare_data(data)

        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            X_scaled, y, file_names,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratification_labels
        )

        # Train model
        self._create_model()
        self.model.fit(X_train, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_metrics = {
            'r2': r2_score(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred)
        }
        test_metrics = {
            'r2': r2_score(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred)
        }

        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }

        # Prepare results DataFrame
        results = pd.DataFrame()
        results['File'] = data['File']
        results['PCE'] = data['PCE']
        results['SMILES'] = data['SMILES']
        results['PredictedPCE'] = np.concatenate([train_pred, test_pred])
        results['Absolute_Error'] = abs(results['PredictedPCE'] - results['PCE'])
        results['Error_Percentage'] = (results['Absolute_Error'] / results['PCE']) * 100
        results['Dataset'] = 'Training'
        results.loc[results['File'].isin(test_files), 'Dataset'] = 'Testing'

        # Add selected features to results
        for feature in self.feature_columns:
            results[feature] = data[feature]

        self.results = results
        self.is_trained = True

        return metrics, results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X = data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_results(self) -> pd.DataFrame:
        """Get the results from training."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.results

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass


class EnsemblePCEModelUncertainty(BasePCEModelEnsembleUncertainty):
    """Ensemble model combining RF and XGBoost predictions with uncertainty estimation."""

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            cv_folds: int = 5,
            rf_params: Dict = {},
            xgb_params: Dict = {}
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.rf_model = RfPCEModel(**rf_params)
        self.xgb_model = XGBoostPCEModel(**xgb_params)

    def _create_model(self) -> None:
        """Create ensemble model."""
        self.rf_model._create_model()
        self.xgb_model._create_model()
        self.model = None

    def get_feature_importance(self) -> pd.DataFrame:
        """Get weighted feature importance from both models."""
        rf_importance = self.rf_model.get_feature_importance()
        xgb_importance = self.xgb_model.get_feature_importance()
        
        # Combine importances with weights
        combined_importance = (
            rf_importance['Importance'] * 0.5 +
            xgb_importance['Importance'] * 0.5
        )
        
        return pd.DataFrame({
            'Feature': self.feature_columns,
            'Combined_Importance': combined_importance,
            'RF_Importance': rf_importance['Importance'],
            'XGB_Importance': xgb_importance['Importance']
        }).sort_values('Combined_Importance', ascending=False)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if self.rf_model.model is None or self.xgb_model.model is None:
            raise RuntimeError("Models must be trained before making predictions")
        
        rf_pred = self.rf_model.predict(data)
        xgb_pred = self.xgb_model.predict(data)
        return rf_pred * 0.5 + xgb_pred * 0.5