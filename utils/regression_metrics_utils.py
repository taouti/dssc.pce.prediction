from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold


@dataclass
class RegressionMetrics:
    """Container for regression performance metrics."""
    r2: float
    adjusted_r2: float
    rmse: float
    mae: float
    std_residuals: float
    mean_residuals: float


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize the model evaluator.

        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> RegressionMetrics:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features used in the model

        Returns:
            RegressionMetrics object containing all calculated metrics
        """
        n_samples = len(y_true)

        # Calculate basic metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Calculate adjusted R²
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

        # Calculate residuals statistics
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        mean_residuals = np.mean(residuals)

        return RegressionMetrics(
            r2=r2,
            adjusted_r2=adjusted_r2,
            rmse=rmse,
            mae=mae,
            std_residuals=std_residuals,
            mean_residuals=mean_residuals
        )

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Perform cross-validation with multiple metrics.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary containing cross-validation scores for each metric
        """
        cv_results = {
            'r2': cross_val_score(model, X, y, cv=self.cv, scoring='r2'),
            'neg_rmse': -np.sqrt(-cross_val_score(model, X, y, cv=self.cv, scoring='neg_mean_squared_error')),
            'neg_mae': -cross_val_score(model, X, y, cv=self.cv, scoring='neg_mean_absolute_error')
        }

        return cv_results

    def evaluate_split_performance(
            self,
            model,
            X_train: np.ndarray,
            X_test: np.ndarray,
            y_train: np.ndarray,
            y_test: np.ndarray
    ) -> Tuple[RegressionMetrics, RegressionMetrics]:
        """
        Evaluate model performance on both training and test sets.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets

        Returns:
            Tuple of (training_metrics, test_metrics)
        """
        n_features = X_train.shape[1]

        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for both sets
        train_metrics = self.calculate_metrics(y_train, y_train_pred, n_features)
        test_metrics = self.calculate_metrics(y_test, y_test_pred, n_features)

        return train_metrics, test_metrics