from abc import ABC, abstractmethod
import dataclasses
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.regression_metrics_utils import ModelEvaluator
from lightgbm import LGBMRegressor

class BasePCEModel(ABC):
    """Base class for PCE prediction models."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            cv_folds: int = 5
    ):
        """
        Initialize base PCE model.

        Args:
            test_size: Proportion of dataset to include in the test split
            random_state: Random state for reproducibility
            cv_folds: Number of folds for cross-validation
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.evaluator = ModelEvaluator(n_splits=cv_folds, random_state=random_state)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str]]:
        """
        Prepare the data for training by selecting features and scaling.

        Args:
            data: Input DataFrame containing features and target

        Returns:
            Tuple containing (X_scaled, y, file_names, feature_columns)
        """
        excluded_columns = ['PCE', 'File', 'SMILES', 'expVoc_V', 'expIsc_mAcm-2', 'expFF']
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col not in excluded_columns]

        X = data[self.feature_columns]
        y = data['PCE']
        file_names = data['File']

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, file_names, self.feature_columns

    @abstractmethod
    def _create_model(self) -> None:
        """Create the specific model instance."""
        pass

    def train_with_engineered_features(self, X_engineered: np.ndarray, y: np.ndarray, file_names: pd.Series) -> Tuple[
        Dict, pd.DataFrame]:
        """
        Train the model using engineered features and evaluate its performance.

        Args:
            X_engineered: Engineered feature matrix
            y: Target values
            file_names: Series containing file identifiers

        Returns:
            Tuple containing (metrics_dict, results_dataframe)
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
                X_engineered, y, file_names,
                test_size=self.test_size,
                random_state=self.random_state
            )

            # Create and train model
            self._create_model()
            self.model.fit(X_train, y_train)

            # Evaluate model
            train_metrics, test_metrics = self.evaluator.evaluate_split_performance(
                self.model, X_train, X_test, y_train, y_test
            )

            # Perform cross-validation
            cv_results = self.evaluator.cross_validate(self.model, X_train, y_train)

            # Compile metrics
            metrics = {
                'train': dataclasses.asdict(train_metrics),
                'test': dataclasses.asdict(test_metrics),
                'cv': {
                    'r2_mean': float(np.mean(cv_results['r2'])),
                    'r2_std': float(np.std(cv_results['r2'])),
                    'rmse_mean': float(np.mean(cv_results['neg_rmse'])),
                    'rmse_std': float(np.std(cv_results['neg_rmse'])),
                    'mae_mean': float(np.mean(cv_results['neg_mae'])),
                    'mae_std': float(np.std(cv_results['neg_mae']))
                }
            }

            # Prepare results DataFrame
            results = pd.DataFrame({
                'File': file_names,
                'PCE': y,
                'Predicted_PCE': self.model.predict(X_engineered),
                'Dataset': 'Training'
            })

            # Mark test set samples
            results.loc[results['File'].isin(test_files), 'Dataset'] = 'Testing'

            # Calculate prediction errors
            results['Prediction_Error'] = abs(results['Predicted_PCE'] - results['PCE'])

            return metrics, results

        except Exception as e:
            raise RuntimeError(f"Error in train_with_engineered_features: {e}")

    def train(self, data: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Train the model and evaluate its performance.

        Args:
            data: Input DataFrame containing features and target

        Returns:
            Tuple containing (metrics_dict, results_dataframe)
        """
        # Prepare data
        X_scaled, y, file_names, _ = self.prepare_data(data)

        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            X_scaled, y, file_names,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Create and train model
        self._create_model()
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_metrics, test_metrics = self.evaluator.evaluate_split_performance(
            self.model, X_train, X_test, y_train, y_test
        )

        # Perform cross-validation
        cv_results = self.evaluator.cross_validate(self.model, X_train, y_train)

        # Compile metrics
        metrics = {
            'train': dataclasses.asdict(train_metrics),
            'test': dataclasses.asdict(test_metrics),
            'cv': {
                'r2_mean': float(np.mean(cv_results['r2'])),
                'r2_std': float(np.std(cv_results['r2'])),
                'rmse_mean': float(np.mean(cv_results['neg_rmse'])),
                'rmse_std': float(np.std(cv_results['neg_rmse'])),
                'mae_mean': float(np.mean(cv_results['neg_mae'])),
                'mae_std': float(np.std(cv_results['neg_mae']))
            }
        }

        # Prepare results DataFrame
        results = data.copy()
        results['Predicted_PCE'] = self.predict(data)
        results['Prediction_Error'] = abs(results['Predicted_PCE'] - results['PCE'])
        results['Dataset'] = 'Training'
        results.loc[results['File'].isin(test_files), 'Dataset'] = 'Testing'

        # Reorder columns
        column_order = [
                           'File', 'PCE', 'Predicted_PCE', 'Prediction_Error', 'Dataset'
                       ] + [col for col in data.columns if col not in [
            'File', 'PCE', 'Predicted_PCE', 'Prediction_Error', 'Dataset'
        ]]
        results = results[column_order]

        return metrics, results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            data: Input DataFrame containing features

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X = data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from the trained model."""
        pass


class RfPCEModel(BasePCEModel):
    """Random Forest implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            n_estimators: int = 100,
            n_jobs: int = -1,
            max_features: str = 'sqrt',
            max_samples: float = 0.6,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.max_samples = max_samples

    def _create_model(self) -> None:
        self.model = RandomForestRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            max_features=self.max_features,
            max_samples=self.max_samples,
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        })
        return importance_df.sort_values('Importance', ascending=False)


class XGBoostPCEModel(BasePCEModel):
    """XGBoost implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 6,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            cv_folds: int = 5
    ):
        super().__init__(test_size, random_state, cv_folds)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def _create_model(self) -> None:
        self.model = XGBRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
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
            'Feature': self.feature_columns,
            'Importance': np.abs(self.model.coef_[0])
        })
        return importance_df.sort_values('Importance', ascending=False)


class LightGBMPCEModel(BasePCEModel):
    """LightGBM implementation of PCE prediction model."""

    def __init__(
            self,
            test_size: float = 0.15,
            random_state: int = 0,
            n_estimators: int = 100,
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
            bagging_freq=self.bagging_freq,
            verbose=-1
        )

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        })
        return importance_df.sort_values('Importance', ascending=False)