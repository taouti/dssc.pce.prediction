from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering including PCA and feature selection."""

    def __init__(self, n_components=0.95, n_features_to_select=20):
        """
        Initialize feature engineering pipeline.

        Args:
            n_components: Number of components or variance ratio for PCA
            n_features_to_select: Number of features to select
        """
        self.n_components = n_components
        self.n_features = n_features_to_select
        self.pca = PCA(n_components=n_components)
        self.feature_selector = SelectKBest(
            score_func=mutual_info_regression,
            k=n_features_to_select
        )
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_scores = None
        self.fitted = False
        self.feature_names = None

    def fit_transform(self, X, y=None):
        """Fit and transform the data using PCA and feature selection."""
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)

            # Apply PCA
            X_pca = self.pca.fit_transform(X_scaled)
            n_pca_components = X_pca.shape[1]
            logger.info(f"PCA transformed data to {n_pca_components} components")

            if y is not None:
                # Feature selection on original scaled features
                self.feature_selector.fit(X_scaled, y)
                self.selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                       if self.feature_selector.get_support()[i]]

                # Create feature scores dataframe
                self.feature_scores = pd.DataFrame({
                    'feature': self.feature_names,
                    'score': self.feature_selector.scores_
                }).sort_values('score', ascending=False)

                # Select features using feature names
                X_selected = X_scaled[:, self.feature_selector.get_support()]
                logger.info(f"Selected {len(self.selected_features)} features")

                # Combine PCA components with selected original features
                X_final = np.hstack([X_pca, X_selected])
                logger.info(f"Final transformed data shape: {X_final.shape}")
            else:
                X_final = X_pca

            self.fitted = True
            return X_final

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise

    def transform(self, X):
        """Transform new data using fitted PCA and feature selection."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        try:
            # Scale the data
            X_scaled = self.scaler.transform(X)
            
            # Apply PCA transformation
            X_pca = self.pca.transform(X_scaled)

            if self.selected_features is not None:
                # Get indices of selected features
                selected_indices = [self.feature_names.index(feat) for feat in self.selected_features]
                X_selected = X_scaled[:, selected_indices]
                return np.hstack([X_pca, X_selected])

            return X_pca

        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise

    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before getting feature importance")
        return self.feature_scores

    def get_explained_variance(self):
        """Get explained variance ratio for PCA components."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before getting explained variance")
        return {
            'individual': self.pca.explained_variance_ratio_,
            'cumulative': np.cumsum(self.pca.explained_variance_ratio_)
        }