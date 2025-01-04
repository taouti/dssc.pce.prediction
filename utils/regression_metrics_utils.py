from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedKFold, KFold
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
    """Handles model evaluation and metrics calculation with repeated cross-validation."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 3, random_state: int = 42):
        """
        Initialize the model evaluator with repeated cross-validation.

        Args:
            n_splits: Number of folds for cross-validation
            n_repeats: Number of times to repeat the CV
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.cv = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )

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

        # Calculate adjusted R² with penalty for number of features
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
        Perform repeated cross-validation with multiple metrics.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary containing cross-validation scores for each metric
        """
        from sklearn.ensemble import VotingRegressor

        # Convert inputs to numpy arrays if they're not already
        X_array = X if isinstance(X, np.ndarray) else X.values
        y_array = y if isinstance(y, np.ndarray) else y.values

        # Initialize lists to store scores for each fold and repeat
        r2_scores = []
        rmse_scores = []
        mae_scores = []

        # Perform repeated cross-validation
        for train_idx, val_idx in self.cv.split(X_array):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # Train model on this fold
            if isinstance(model, VotingRegressor):
                model_clone = VotingRegressor(
                    estimators=[(name, est.__class__(**est.get_params())) 
                              for name, est in model.estimators],
                    weights=model.weights
                )
            else:
                model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)

            # Make predictions
            y_pred = model_clone.predict(X_val)

            # Calculate metrics for this fold
            r2_scores.append(r2_score(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae_scores.append(mean_absolute_error(y_val, y_pred))

        # Return all metrics
        return {
            'r2': r2_scores,
            'neg_rmse': [-x for x in rmse_scores],  # Negate for consistency with sklearn
            'neg_mae': [-x for x in mae_scores]  # Negate for consistency with sklearn
        }

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