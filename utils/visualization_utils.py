import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path


class PCEVisualizer:
    def __init__(self, execution_logger, style: str = 'seaborn'):
        """
        Initialize visualizer with specified style and logger.

        Args:
            execution_logger: ExecutionLogger instance for saving plots
            style: matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {'Training': '#2ecc71', 'Testing': '#e74c3c'}
        self.logger = execution_logger

    def _save_plot(self, fig: plt.Figure, plot_name: str) -> Path:
        """Save plot using the execution logger."""
        plot_path = self.logger.save_plot(fig, plot_name)
        plt.close(fig)
        return plot_path

    def plot_actual_vs_predicted(self, results: pd.DataFrame, model_name: str) -> Path:
        """Plot actual vs predicted PCE values for training and testing sets."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for dataset, ax in zip(['Training', 'Testing'], [ax1, ax2]):
            subset = results[results['Dataset'] == dataset].copy()
            subset = subset.sort_values('Prediction_Error', ascending=False)

            ax.scatter(range(len(subset)), subset['PCE'],
                       label='Actual PCE', alpha=0.7, color='blue')
            ax.scatter(range(len(subset)), subset['Predicted_PCE'],
                       label='Predicted PCE', alpha=0.7, color='red')

            ax.set_title(f'{dataset} Set: Actual vs Predicted PCE\n(Sorted by Prediction Error)')
            ax.set_xlabel('Dye Index')
            ax.set_ylabel('PCE Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            for i, (actual, pred) in enumerate(zip(subset['PCE'], subset['Predicted_PCE'])):
                ax.plot([i, i], [actual, pred], 'k-', alpha=0.3)

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_actual_vs_predicted")

    def plot_error_distribution(self, results: pd.DataFrame, model_name: str) -> Path:
        """Plot error distribution for both datasets."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            sns.kdeplot(data=subset['Prediction_Error'],
                        label=f"{dataset} (Mean: {subset['Prediction_Error'].mean():.3f})",
                        ax=ax, fill=True, alpha=0.3,
                        color=self.colors[dataset])

        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.legend()

        return self._save_plot(fig, f"{model_name}_error_distribution")

    def plot_parity(self, results: pd.DataFrame, model_name: str) -> Path:
        """Create parity plot with error bands."""
        fig, ax = plt.subplots(figsize=(8, 8))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            ax.scatter(subset['PCE'], subset['Predicted_PCE'],
                       label=dataset, alpha=0.6,
                       color=self.colors[dataset])

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Parity')

        max_val = max(lims)
        x = np.linspace(0, max_val, 100)
        ax.fill_between(x, x * 0.9, x * 1.1, alpha=0.1, color='gray', label='±10% Error')

        ax.set_title('Parity Plot: Predicted vs Actual PCE')
        ax.set_xlabel('Experimental PCE')
        ax.set_ylabel('Predicted PCE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_plot(fig, f"{model_name}_parity")

    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, top_n: int = 15) -> Path:
        """Plot top N most important features."""
        fig, ax = plt.subplots(figsize=(12, 6))

        top_features = importance_df.head(top_n)
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)

        ax.set_title(f'Top {top_n} Most Important Features')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')

        return self._save_plot(fig, f"{model_name}_feature_importance")

    def create_summary_report(self, results: pd.DataFrame, metrics: Dict, model_name: str) -> Path:
        """Create a comprehensive summary plot with key metrics and distributions."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # Metrics summary
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_text = (
            f"Model Performance Metrics\n\n"
            f"Training Set:\n"
            f"R² = {metrics['train']['r2']:.4f}\n"
            f"RMSE = {metrics['train']['rmse']:.4f}\n"
            f"MAE = {metrics['train']['mae']:.4f}\n\n"
            f"Testing Set:\n"
            f"R² = {metrics['test']['r2']:.4f}\n"
            f"RMSE = {metrics['test']['rmse']:.4f}\n"
            f"MAE = {metrics['test']['mae']:.4f}\n\n"
            f"Cross-Validation:\n"
            f"R² = {metrics['cv']['r2_mean']:.4f} ± {metrics['cv']['r2_std']:.4f}"
        )
        ax1.text(0.1, 0.5, metrics_text, fontsize=10, va='center')
        ax1.axis('off')

        # Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            sns.kdeplot(data=subset['Prediction_Error'],
                        label=dataset, ax=ax2, fill=True, alpha=0.3,
                        color=self.colors[dataset])
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Absolute Error')

        # Parity plot
        ax3 = fig.add_subplot(gs[1, :])
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            ax3.scatter(subset['PCE'], subset['Predicted_PCE'],
                        label=dataset, alpha=0.6,
                        color=self.colors[dataset])

        lims = [
            np.min([ax3.get_xlim(), ax3.get_ylim()]),
            np.max([ax3.get_xlim(), ax3.get_ylim()])
        ]
        ax3.plot(lims, lims, 'k--', alpha=0.5, label='Parity')
        ax3.set_title('Parity Plot')
        ax3.set_xlabel('Experimental PCE')
        ax3.set_ylabel('Predicted PCE')
        ax3.legend()

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_summary")