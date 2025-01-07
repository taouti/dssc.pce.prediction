"""
Preprocessing module for PCE prediction pipeline.
Handles feature selection, scaling, and descriptor processing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    force_include_features: List[str]
    n_features_to_select: int
    correlation_threshold: float
    rf_importance_weight: float = 0.7
    mi_importance_weight: float = 0.3
    excluded_columns: List[str] = None

    def __post_init__(self):
        if self.excluded_columns is None:
            self.excluded_columns = ['PCE', 'File', 'SMILES', 'expVoc_V', 'expIsc_mAcm-2', 'expFF', 'Mass']

class FeaturePreprocessor:
    """Handles feature preprocessing including selection and scaling."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(self, data: pd.DataFrame, target_col: str = 'PCE') -> 'FeaturePreprocessor':
        """Fit the preprocessor on training data."""
        # Get numeric columns excluding specified ones
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_columns if col not in self.config.excluded_columns]

        # Ensure forced features are available
        forced_features = [f for f in self.config.force_include_features if f in available_features]
        remaining_features = [f for f in available_features if f not in forced_features]

        # Calculate feature importance
        importance_scores = self._calculate_feature_importance(
            data[remaining_features], 
            data[target_col],
            remaining_features
        )

        # Select features
        n_additional = min(
            self.config.n_features_to_select - len(forced_features),
            len(remaining_features)
        )
        selected_additional = importance_scores.head(n_additional).index.tolist()
        
        # Combine forced and selected features
        self.selected_features = forced_features + selected_additional

        # Remove highly correlated features
        if len(self.selected_features) > 1:
            self.selected_features = self._remove_correlated_features(
                data[self.selected_features],
                forced_features
            )

        # Fit scaler
        self.scaler.fit(data[self.selected_features])
        self._is_fitted = True
        
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale selected features
        scaled_features = self.scaler.transform(data[self.selected_features])
        return pd.DataFrame(scaled_features, columns=self.selected_features, index=data.index)

    def fit_transform(self, data: pd.DataFrame, target_col: str = 'PCE') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, target_col).transform(data)

    def _calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Calculate combined feature importance using RF and mutual information."""
        # Random Forest importance
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names)

        # Mutual information importance
        mi_scores = mutual_info_regression(X, y)
        mi_importance = pd.Series(mi_scores, index=feature_names)

        # Normalize scores
        rf_importance = rf_importance / rf_importance.max()
        mi_importance = mi_importance / mi_importance.max()

        # Calculate combined score
        combined_score = (
            rf_importance * self.config.rf_importance_weight +
            mi_importance * self.config.mi_importance_weight
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'rf_importance': rf_importance,
            'mi_importance': mi_importance,
            'combined_score': combined_score
        }).sort_values('combined_score', ascending=False)

        return self.feature_importance

    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
        forced_features: List[str]
    ) -> List[str]:
        """Remove highly correlated features while preserving forced features."""
        corr_matrix = X.corr().abs()
        to_drop = []

        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features to drop
        for i in range(len(upper.columns)):
            for j in range(i + 1, len(upper.columns)):
                if upper.iloc[i, j] > self.config.correlation_threshold:
                    feat_i = upper.columns[i]
                    feat_j = upper.columns[j]

                    # Skip if both features are forced
                    if feat_i in forced_features and feat_j in forced_features:
                        continue

                    # Never drop a forced feature
                    if feat_i in forced_features:
                        to_drop.append(feat_j)
                    elif feat_j in forced_features:
                        to_drop.append(feat_i)
                    # For non-forced features, drop the one with lower importance
                    else:
                        if self.feature_importance.loc[feat_i, 'combined_score'] < self.feature_importance.loc[feat_j, 'combined_score']:
                            to_drop.append(feat_i)
                        else:
                            to_drop.append(feat_j)

        # Remove duplicates and return remaining features
        to_drop = list(set(to_drop))
        return [f for f in X.columns if f not in to_drop]

    def save(self, filepath: str):
        """Save preprocessor state to files."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save scaler and selected features
        joblib.dump({
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'config': self.config,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
        }, filepath)

        # Save a human-readable feature list
        features_file = os.path.splitext(filepath)[0] + '_features.json'
        with open(features_file, 'w') as f:
            json.dump({
                'selected_features': self.selected_features,
                'forced_features': self.config.force_include_features,
                'n_features': len(self.selected_features)
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FeaturePreprocessor':
        """Load preprocessor state from file."""
        data = joblib.load(filepath)
        
        # Create instance with saved config
        instance = cls(data['config'])
        
        # Restore state
        instance.scaler = data['scaler']
        instance.selected_features = data['selected_features']
        if data['feature_importance'] is not None:
            instance.feature_importance = pd.DataFrame(data['feature_importance'])
        instance._is_fitted = True
        
        return instance
