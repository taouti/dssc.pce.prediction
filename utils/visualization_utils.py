import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class PCEVisualizer:
    def __init__(self, execution_logger, style: str = 'seaborn-v0_8-whitegrid'):
        # Add format specifier for consistent decimal places
        self.decimal_format = '.2f'
        """
        Initialize visualizer with specified style and logger.
        """
        plt.style.use(style)
        self.colors = {
            'Training': '#2ecc71',
            'Testing': '#e74c3c',
            'Validation': '#3498db'
        }
        self.logger = execution_logger
        self.log = logger

    def _validate_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all data is numeric, converting non-numeric values to NaN and dropping them."""
        try:
            numeric_data = data.apply(pd.to_numeric, errors='coerce')
            if numeric_data.isnull().values.any():
                self.log.warning("Non-numeric values detected and converted to NaN.")
            return numeric_data.dropna()
        except Exception as e:
            self.log.error(f"Error during data validation: {e}")
            raise

    def set_style(self, fig: plt.Figure) -> None:
        """Apply consistent styling to the figure."""
        for ax in fig.get_axes():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.7)

    def _save_plot(self, fig: plt.Figure, filename: str) -> Path:
        """Save the figure using the execution logger."""
        try:
            fig.tight_layout()
            saved_path = self.logger.save_plot(fig, filename)
            plt.close(fig)
            self.log.info(f"Plot saved to {saved_path}")
            return saved_path
        except Exception as e:
            self.log.error(f"Error saving plot: {e}")
            raise

    def plot_feature_correlation(self, data: pd.DataFrame, target_col: str = 'PCE') -> Path:
        """Plot feature correlation heatmap."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols]
            correlation = data_numeric.corr()
            target_corr = abs(correlation[target_col]).sort_values(ascending=False)
            top_features = target_corr.head(15).index

            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation.loc[top_features, top_features]))

            sns.heatmap(correlation.loc[top_features, top_features],
                        mask=mask,
                        annot=True,
                        fmt='.2%',  # Updated to 2 decimal places
                        cmap='RdBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        ax=ax)

            ax.set_title('Feature Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            return self._save_plot(fig, "feature_correlation")
        except Exception as e:
            self.log.error(f"Error in plot_feature_correlation: {e}")
            raise

    def plot_actual_vs_predicted(self, results: pd.DataFrame, model_name: str) -> Path:
        """Plot actual vs predicted values with detailed statistics."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            for dataset in ['Training', 'Testing']:
                mask = results['Dataset'] == dataset
                actual = results.loc[mask, 'PCE'] * 100
                predicted = results.loc[mask, 'Predicted_PCE'] * 100

                ax.scatter(actual, predicted,
                           alpha=0.6,
                           label=f'{dataset} (R² = {r2_score(actual, predicted):.2%})',
                           color=self.colors[dataset])

                # Add trend line
                z = np.polyfit(actual, predicted, 1)
                p = np.poly1d(z)
                ax.plot(actual, p(actual), '--', color=self.colors[dataset], alpha=0.5)

            # Add diagonal line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

            ax.set_xlabel('Actual PCE (%)')
            ax.set_ylabel('Predicted PCE (%)')
            ax.set_title(f'Actual vs Predicted PCE ({model_name})')
            ax.legend()

            self.set_style(fig)
            return self._save_plot(fig, f"actual_vs_predicted_{model_name}")
        except Exception as e:
            self.log.error(f"Error in plot_actual_vs_predicted: {e}")
            raise

    def plot_bland_altman(self, results: pd.DataFrame, model_name: str) -> Path:
        """Create Bland-Altman plot for agreement analysis."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            actual = results['PCE'] * 100
            predicted = results['Predicted_PCE'] * 100

            mean = (actual + predicted) / 2
            diff = predicted - actual

            md = np.mean(diff)
            sd = np.std(diff)

            ax.scatter(mean, diff, alpha=0.6, color='#3498db')
            ax.axhline(md, color='k', linestyle='--')
            ax.axhline(md + 1.96 * sd, color='r', linestyle='--')
            ax.axhline(md - 1.96 * sd, color='r', linestyle='--')

            ax.text(ax.get_xlim()[1], md, f'Mean: {md:.2f}%', ha='right', va='bottom')
            ax.text(ax.get_xlim()[1], md + 1.96 * sd, f'+1.96 SD: {(md + 1.96 * sd):.2f}%', ha='right', va='bottom')
            ax.text(ax.get_xlim()[1], md - 1.96 * sd, f'-1.96 SD: {(md - 1.96 * sd):.2f}%', ha='right', va='top')

            ax.set_xlabel('Mean of Actual and Predicted PCE (%)')
            ax.set_ylabel('Difference (Predicted - Actual) (%)')
            ax.set_title(f'Bland-Altman Plot ({model_name})')

            self.set_style(fig)
            return self._save_plot(fig, f"bland_altman_{model_name}")
        except Exception as e:
            self.log.error(f"Error in plot_bland_altman: {e}")
            raise

    def plot_residuals_analysis(self, results: pd.DataFrame) -> Path:
        """Create comprehensive residuals analysis plots."""
        try:
            results = self._validate_numeric_data(results)

            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # Residuals vs Predicted
            ax1 = fig.add_subplot(gs[0, 0])
            residuals = (results['PCE'] - results['Predicted_PCE']) * 100
            predicted = results['Predicted_PCE'] * 100

            ax1.scatter(predicted, residuals, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')

            # Add trend line
            z = np.polyfit(predicted, residuals, 1)
            p = np.poly1d(z)
            ax1.plot(predicted, p(predicted), '--', color='k', alpha=0.5)

            ax1.set_title('Residuals vs Predicted Values')
            ax1.set_xlabel('Predicted PCE (%)')
            ax1.set_ylabel('Residuals (%)')

            # Q-Q plot
            ax2 = fig.add_subplot(gs[0, 1])
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Normal Q-Q Plot')
            ax2.set_ylabel('Sample Residuals (%)')

            # Residuals histogram
            ax3 = fig.add_subplot(gs[1, 0])
            sns.histplot(residuals, kde=True, ax=ax3)
            ax3.set_title('Residuals Distribution')
            ax3.set_xlabel('Residuals (%)')

            # Add standard deviation lines
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ylims = ax3.get_ylim()
            ax3.vlines([mean_residual, mean_residual + std_residual, mean_residual - std_residual],
                       0, ylims[1], colors=['r', 'k', 'k'], linestyles=['--', ':', ':'])
            ax3.text(mean_residual, ylims[1], f'Mean: {mean_residual:.2f}%', ha='center', va='bottom')

            # Scale-Location plot
            ax4 = fig.add_subplot(gs[1, 1])
            standardized_residuals = residuals / np.std(residuals)
            sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))

            ax4.scatter(predicted, sqrt_std_residuals, alpha=0.5)

            # Add trend line
            z = np.polyfit(predicted, sqrt_std_residuals, 1)
            p = np.poly1d(z)
            ax4.plot(predicted, p(predicted), '--', color='k', alpha=0.5)

            ax4.set_title('Scale-Location Plot')
            ax4.set_xlabel('Predicted PCE (%)')
            ax4.set_ylabel('\u221A|Standardized Residuals|')

            self.set_style(fig)
            return self._save_plot(fig, "residuals_analysis")
        except Exception as e:
            self.log.error(f"Error in plot_residuals_analysis: {e}")
            raise

    def plot_error_distribution(self, results: pd.DataFrame, model_name: str) -> Path:
        """Plot error distribution analysis."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Calculate percentage errors
            actual = results['PCE'] * 100
            predicted = results['Predicted_PCE'] * 100
            absolute_errors = np.abs(predicted - actual)
            percentage_errors = (np.abs(predicted - actual) / actual) * 100

            # Absolute error distribution
            sns.histplot(absolute_errors, kde=True, ax=ax1)
            ax1.set_title('Absolute Error Distribution')
            ax1.set_xlabel('Absolute Error (%)')
            ax1.set_ylabel('Count')

            mean_ae = np.mean(absolute_errors)
            ax1.axvline(mean_ae, color='r', linestyle='--')
            ax1.text(mean_ae, ax1.get_ylim()[1], f'Mean: {mean_ae:.2f}%',
                     rotation=90, va='top', ha='right')

            # Percentage error distribution
            sns.histplot(percentage_errors, kde=True, ax=ax2)
            ax2.set_title('Percentage Error Distribution')
            ax2.set_xlabel('Percentage Error (%)')
            ax2.set_ylabel('Count')

            mean_pe = np.mean(percentage_errors)
            ax2.axvline(mean_pe, color='r', linestyle='--')
            ax2.text(mean_pe, ax2.get_ylim()[1], f'Mean: {mean_pe:.2f}%',
                     rotation=90, va='top', ha='right')

            fig.suptitle(f'Error Distribution Analysis ({model_name})')
            self.set_style(fig)
            return self._save_plot(fig, f"error_distribution_{model_name}")
        except Exception as e:
            self.log.error(f"Error in plot_error_distribution: {e}")
            raise

    def plot_learning_curves(self, train_sizes: np.ndarray,
                             train_scores: np.ndarray,
                             test_scores: np.ndarray) -> Path:
        """Plot learning curves with detailed statistics."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            train_scores_pct = train_scores * 100
            test_scores_pct = test_scores * 100

            # Plot mean scores
            ax.plot(train_sizes, np.mean(train_scores_pct, axis=1), 'o-', label='Training Score',
                    color=self.colors['Training'])
            ax.fill_between(train_sizes,
                            np.mean(train_scores_pct, axis=1) - np.std(train_scores_pct, axis=1),
                            np.mean(train_scores_pct, axis=1) + np.std(train_scores_pct, axis=1),
                            alpha=0.1, color=self.colors['Training'])

            ax.plot(train_sizes, np.mean(test_scores_pct, axis=1), 'o-', label='Cross-Validation Score',
                    color=self.colors['Testing'])
            ax.fill_between(train_sizes,
                            np.mean(test_scores_pct, axis=1) - np.std(test_scores_pct, axis=1),
                            np.mean(test_scores_pct, axis=1) + np.std(test_scores_pct, axis=1),
                            alpha=0.1, color=self.colors['Testing'])

            # Add final scores annotation
            final_train_score = np.mean(train_scores_pct, axis=1)[-1]
            final_test_score = np.mean(test_scores_pct, axis=1)[-1]
            ax.annotate(f'Final training score: {final_train_score:.2f}%',
                        xy=(train_sizes[-1], final_train_score),
                        xytext=(10, 10), textcoords='offset points')
            ax.annotate(f'Final CV score: {final_test_score:.2f}%',
                        xy=(train_sizes[-1], final_test_score),
                        xytext=(10, -10), textcoords='offset points')

            ax.set_title('Learning Curves')
            ax.set_xlabel('Training Examples')
            ax.set_ylabel('Score (%)')
            ax.legend(loc='best')

            self.set_style(fig)
            return self._save_plot(fig, "learning_curves")
        except Exception as e:
            self.log.error(f"Error in plot_learning_curves: {e}")
            raise

    def plot_pca_variance(self, variance_data: Dict) -> Path:
        """Plot PCA explained variance ratio with percentage scales."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Convert variance ratios to percentages
            individual_pct = np.array(variance_data['individual']) * 100
            cumulative_pct = np.array(variance_data['cumulative']) * 100

            # Individual explained variance
            components = range(1, len(individual_pct) + 1)
            ax1.bar(components, individual_pct, alpha=0.8, color='steelblue')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio (%)')
            ax1.set_title('Individual Explained Variance Ratio')

            # Add percentage labels on top of bars
            for i, v in enumerate(individual_pct):
                ax1.text(i + 1, v + 1, f'{v:.1f}%', ha='center', va='bottom')

            # Cumulative explained variance
            ax2.plot(components, cumulative_pct, 'o-', color='steelblue')
            ax2.axhline(y=95, color='r', linestyle='--', label='95% Threshold')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Explained Variance Ratio (%)')
            ax2.set_title('Cumulative Explained Variance Ratio')
            ax2.legend()

            # Add percentage labels for cumulative plot
            for i, v in enumerate(cumulative_pct):
                ax2.text(i + 1, v + 2, f'{v:.1f}%', ha='center', va='bottom')

            self.set_style(fig)
            return self._save_plot(fig, "pca_variance")

        except Exception as e:
            self.log.error(f"Error in plot_pca_variance: {e}")
            raise
