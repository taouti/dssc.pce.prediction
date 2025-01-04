import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Dict


class PCEVisualizer:
    def __init__(self, execution_logger, style: str = 'default'):
        """Initialize visualizer with specified style and logger."""
        sns.set_style("whitegrid")  # Use seaborn's whitegrid style
        self.colors = {'Training': '#2ecc71', 'Testing': '#e74c3c'}
        self.log = execution_logger
        self.output_dir = None

    def _format_percentage(self, value: float) -> str:
        """Format float as percentage string."""
        return f"{value:.2f}%"

    def _save_plot(self, fig: plt.Figure, filename: str) -> Path:
        """Save plot to file and return path."""
        if self.output_dir is None:
            raise ValueError("Output directory not set")
        
        output_path = Path(self.output_dir) / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return output_path

    def _add_model_info(self, ax: plt.Axes, model_name: str, model_params: dict) -> None:
        """Add model information to plot."""
        if model_params:
            param_text = "\n".join([f"{k}: {v}" for k, v in model_params.items()])
            ax.text(0.02, 0.98, f"Model: {model_name}\n\nParameters:\n{param_text}",
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=8)

    def plot_actual_vs_predicted(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        """Plot actual vs predicted values with detailed metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for dataset, ax in zip(['Training', 'Testing'], [ax1, ax2]):
            subset = results[results['Dataset'] == dataset].copy()
            subset = subset.sort_values('Absolute_Error', ascending=False)
            x_indices = range(len(subset))

            ax.scatter(x_indices, subset['PCE'] * 100,
                      label='Actual PCE (%)', alpha=0.7, color='blue')
            ax.scatter(x_indices, subset['PredictedPCE'] * 100,
                      label='Predicted PCE (%)', alpha=0.7, color='red')

            # Calculate and display metrics
            r2 = r2_score(subset['PCE'], subset['PredictedPCE'])
            rmse = np.sqrt(mean_squared_error(subset['PCE'], subset['PredictedPCE']))
            mae = mean_absolute_error(subset['PCE'], subset['PredictedPCE'])
            
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title(f'{dataset} Set: Actual vs Predicted PCE\n(Sorted by Absolute Error)')
            ax.set_xlabel('Dye Index')
            ax.set_ylabel('PCE (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add connecting lines between actual and predicted values
            for i, (actual, pred) in enumerate(zip(subset['PCE'], subset['PredictedPCE'])):
                ax.plot([i, i], [actual * 100, pred * 100], 'k-', alpha=0.3)

            # Add dye identifiers as x-tick labels
            ax.set_xticks(x_indices)
            ax.set_xticklabels(subset['File'], rotation=45, ha='right')

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_actual_vs_predicted")

    def plot_error_distribution(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        """Plot error distribution with detailed statistics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            mean_error = subset['Absolute_Error'].mean() * 100
            std_error = subset['Absolute_Error'].std() * 100

            sns.kdeplot(data=subset['Absolute_Error'] * 100,
                       label=f"{dataset} (μ={self._format_percentage(mean_error)}, σ={self._format_percentage(std_error)})",
                       ax=ax, fill=True, alpha=0.3,
                       color=self.colors[dataset])

        ax.set_title('Distribution of Absolute Errors')
        ax.set_xlabel('Error (%)')
        ax.set_ylabel('Density')
        ax.legend()

        self._add_model_info(ax, model_name, model_params)
        return self._save_plot(fig, f"{model_name}_error_distribution")

    def plot_parity(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        """Plot parity with detailed statistics and annotations."""
        fig, ax = plt.subplots(figsize=(8, 8))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            r2 = r2_score(subset['PCE'], subset['PredictedPCE'])
            rmse = np.sqrt(mean_squared_error(subset['PCE'], subset['PredictedPCE']))
            mae = mean_absolute_error(subset['PCE'], subset['PredictedPCE'])
            
            ax.scatter(subset['PCE'] * 100, subset['PredictedPCE'] * 100,
                      label=f"{dataset}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
                      alpha=0.6, color=self.colors[dataset])

            # Add dye identifiers as annotations
            for _, row in subset.iterrows():
                ax.annotate(row['File'], 
                          (row['PCE'] * 100, row['PredictedPCE'] * 100),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)

        # Add parity line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Parity')

        ax.set_title('Parity Plot: Predicted vs Actual PCE')
        ax.set_xlabel('Actual PCE (%)')
        ax.set_ylabel('Predicted PCE (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        self._add_model_info(ax, model_name, model_params)
        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_parity")

    def plot_residuals(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        """Plot residuals with detailed analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            residuals = (subset['PCE'] - subset['PredictedPCE']) * 100
            
            # Residuals vs Predicted
            ax1.scatter(subset['PredictedPCE'] * 100, residuals,
                       label=dataset, alpha=0.6, color=self.colors[dataset])
            
            # Add dye identifiers
            for _, row in subset.iterrows():
                residual = (row['PCE'] - row['PredictedPCE']) * 100
                ax1.annotate(row['File'], 
                           (row['PredictedPCE'] * 100, residual),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax2)

        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Predicted PCE (%)')
        ax1.set_ylabel('Residuals (%)')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Normal Q-Q Plot of Residuals')
        
        self._add_model_info(ax1, model_name, model_params)
        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_residuals")

    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, model_params: dict,
                              top_n: int = 15) -> Path:
        """Plot feature importance with detailed statistics."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort and select top N features
        importance_df = importance_df.sort_values('Importance', ascending=True)
        if len(importance_df) > top_n:
            importance_df = importance_df.tail(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center')
        
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.set_xlabel('Importance Score')
        
        self._add_model_info(ax, model_name, model_params)
        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_feature_importance")

    def plot_feature_correlation(self, data: pd.DataFrame, model_name: str) -> Path:
        """Plot feature correlation matrix."""
        # Calculate correlation matrix
        correlation_matrix = data.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', ax=ax)
        
        plt.title(f'Feature Correlation Matrix - {model_name}')
        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_feature_correlation")

    def visualize_results(self, model, model_name: str) -> List[Path]:
        """Generate all visualizations for a model's results."""
        if not hasattr(model, 'get_results') or not callable(model.get_results):
            raise ValueError("Model must have a get_results() method")
            
        results = model.get_results()
        if results is None or len(results) == 0:
            raise ValueError("No results available for visualization")

        plots = []
        plots.append(self.plot_actual_vs_predicted(results, model_name, {}))
        plots.append(self.plot_error_distribution(results, model_name, {}))
        plots.append(self.plot_parity(results, model_name, {}))
        plots.append(self.plot_residuals(results, model_name, {}))
        
        if hasattr(model, 'get_feature_importance') and callable(model.get_feature_importance):
            importance_df = model.get_feature_importance()
            plots.append(self.plot_feature_importance(importance_df, model_name, {}))
            plots.append(self.plot_feature_correlation(results, model_name))
            
        return plots