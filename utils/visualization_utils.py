import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score


class PCEVisualizer:
    def __init__(self, execution_logger, style: str = 'seaborn'):
        """
        Initialize visualizer with specified style and logger.
        """
        plt.style.use(style)
        self.colors = {'Training': '#2ecc71', 'Testing': '#e74c3c'}
        self.logger = execution_logger

    def _save_plot(self, fig: plt.Figure, plot_name: str) -> Path:
        """Save plot using the execution logger."""
        plot_path = self.logger.save_plot(fig, plot_name)
        plt.close(fig)
        return plot_path

    def _format_percentage(self, value: float) -> str:
        """Format percentage to two significant figures."""
        return f"{value:.2f}%"

    def _add_model_info(self, ax: plt.Axes, model_name: str, model_params: dict):
        """Add model information to the plot."""
        param_text = "\n".join([f"{k}: {v}" for k, v in model_params.items()])
        ax.text(0.02, 0.98, f"Model: {model_name}\n{param_text}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=8)

    def plot_actual_vs_predicted(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for dataset, ax in zip(['Training', 'Testing'], [ax1, ax2]):
            subset = results[results['Dataset'] == dataset].copy()
            subset = subset.sort_values('Prediction_Error', ascending=False)

            ax.scatter(range(len(subset)), subset['PCE'] * 100,
                       label='Actual PCE (%)', alpha=0.7, color='blue')
            ax.scatter(range(len(subset)), subset['Predicted_PCE'] * 100,
                       label='Predicted PCE (%)', alpha=0.7, color='red')

            ax.set_title(f'{dataset} Set: Actual vs Predicted PCE\n(Sorted by Prediction Error)')
            ax.set_xlabel('Dye Index')
            ax.set_ylabel('PCE (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            self._add_model_info(ax, model_name, model_params)

            for i, (actual, pred) in enumerate(zip(subset['PCE'], subset['Predicted_PCE'])):
                ax.plot([i, i], [actual * 100, pred * 100], 'k-', alpha=0.3)

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_actual_vs_predicted")

    def plot_error_distribution(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            mean_error = subset['Prediction_Error'].mean() * 100
            std_error = subset['Prediction_Error'].std() * 100

            sns.kdeplot(data=subset['Prediction_Error'] * 100,
                        label=f"{dataset} (μ={self._format_percentage(mean_error)}, σ={self._format_percentage(std_error)})",
                        ax=ax, fill=True, alpha=0.3,
                        color=self.colors[dataset])

        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Error (%)')
        ax.set_ylabel('Density')
        ax.legend()

        self._add_model_info(ax, model_name, model_params)

        return self._save_plot(fig, f"{model_name}_error_distribution")

    def plot_parity(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        fig, ax = plt.subplots(figsize=(8, 8))

        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            r2 = np.corrcoef(subset['PCE'], subset['Predicted_PCE'])[0, 1] ** 2
            ax.scatter(subset['PCE'] * 100, subset['Predicted_PCE'] * 100,
                       label=f"{dataset} (R² = {self._format_percentage(r2 * 100)})",
                       alpha=0.6, color=self.colors[dataset])

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Parity')

        max_val = max(lims)
        x = np.linspace(0, max_val, 100)
        ax.fill_between(x, x * 0.9, x * 1.1, alpha=0.1, color='gray', label='±10% Error')

        ax.set_title('Parity Plot: Predicted vs Actual PCE')
        ax.set_xlabel('Experimental PCE (%)')
        ax.set_ylabel('Predicted PCE (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._add_model_info(ax, model_name, model_params)

        return self._save_plot(fig, f"{model_name}_parity")

    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, model_params: dict,
                                top_n: int = 15) -> Path:
        """Plot top N most important features with percentages."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ensure numeric importance values
        top_features = importance_df.head(top_n).copy()
        top_features.loc[:, 'Importance'] = pd.to_numeric(top_features['Importance'], errors='coerce') * 100

        # Remove any rows with NaN values
        top_features = top_features.dropna()

        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)

        ax.set_title(f'Top {top_n} Most Important Features')
        ax.set_xlabel('Relative Importance (%)')
        ax.set_ylabel('Feature')

        self._add_model_info(ax, model_name, model_params)

        # Add percentage labels
        for i, v in enumerate(top_features['Importance']):
            if pd.notnull(v):
                ax.text(v, i, self._format_percentage(v), va='center')

        return self._save_plot(fig, f"{model_name}_feature_importance")

    def plot_feature_correlation(self, data: pd.DataFrame, model_name: str, target_col: str = 'PCE') -> Path:
        """Plot feature correlation heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Calculate correlations with target
        correlations = data.corr()[target_col].sort_values(ascending=False)
        top_features = correlations.index[:15]  # Top 15 correlated features

        # Create correlation matrix for top features
        correlation_matrix = data[top_features].corr()

        # Plot heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2g', ax=ax, cbar_kws={'label': 'Correlation Coefficient'})

        ax.set_title('Feature Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_feature_correlation")

    def plot_residuals(self, results: pd.DataFrame, model_name: str, model_params: dict) -> Path:
        """Plot residuals analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Residuals vs Predicted
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            ax1.scatter(subset['Predicted_PCE'], subset['Prediction_Error'],
                        label=dataset, alpha=0.6, color=self.colors[dataset])

        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Predicted PCE')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.legend()

        # Q-Q plot
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            stats.probplot(subset['Prediction_Error'], dist="norm", plot=ax2)

        ax2.set_title('Normal Q-Q Plot of Residuals')

        self._add_model_info(ax1, model_name, model_params)

        plt.tight_layout()
        return self._save_plot(fig, f"{model_name}_residuals")

    def plot_feature_correlation2(self, data: pd.DataFrame, target_col: str = 'PCE') -> Path:
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

    def plot_actual_vs_predicted2(self, results: pd.DataFrame, model_name: str) -> Path:
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

    def plot_error_distribution2(self, results: pd.DataFrame, model_name: str) -> Path:
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