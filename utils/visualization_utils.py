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
        # Set style to a basic clean style
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        # Publication-ready plot settings
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 600,
            'savefig.dpi': 900,
            'figure.figsize': [8, 6],
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
            'lines.markersize': 8,
            'scatter.marker': 'o',
            'axes.grid': True,
            'grid.alpha': 0.3,
            # Additional settings for better readability
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelpad': 10,
            'figure.autolayout': True
        })
        
        # Enhanced color scheme
        self.colors = {
            'Training': '#2ecc71',
            'Testing': '#e74c3c',
            'Validation': '#3498db',
            'Error_fill': '#fff3f2',
            'Grid': '#cccccc',
            'Highlight': '#f1c40f',
            'Reference': '#34495e',
            'Positive': '#27ae60',
            'Negative': '#c0392b',
            'Neutral': '#7f8c8d'
        }
        
        # Marker styles
        self.markers = {
            'Training': 'o',
            'Testing': 's',
            'Validation': '^',
            'Outlier': 'D'
        }
        
        self.log = execution_logger
        self.output_dir = None

    def _format_percentage(self, value: float) -> str:
        """Format float as percentage string."""
        return f"{value:.2f}%"

    def _format_metric(self, value: float, precision: int = 3) -> str:
        """Format metric values with consistent precision."""
        return f"{value:.{precision}f}"

    def _save_plot(self, fig: plt.Figure, filename: str, enhanced: bool = False) -> List[Path]:
        """Save plot to file in both PNG and SVG formats.
        
        Args:
            fig: matplotlib Figure object
            filename: Base name for the plot file
            enhanced: If True, save to enhanced plots directory
            
        Returns:
            List of paths where the plots were saved
        """
        if self.output_dir is None:
            raise ValueError("Output directory not set")
        
        # Determine output directory based on enhanced flag
        output_dir = Path(self.output_dir)
        if enhanced:
            output_dir = output_dir.parent / "enhanced"
            output_dir.mkdir(exist_ok=True)
        
        saved_paths = []
        
        try:
            # Save as high-resolution PNG
            png_path = output_dir / f"{filename}.png"
            fig.savefig(png_path, dpi=600, bbox_inches='tight', format='png')
            saved_paths.append(png_path)
            
            # Save as SVG
            svg_path = output_dir / f"{filename}.svg"
            fig.savefig(svg_path, bbox_inches='tight', format='svg')
            saved_paths.append(svg_path)
        finally:
            # Close the figure to free up memory
            plt.close(fig)
        
        return saved_paths

    def _add_model_info(self, ax: plt.Axes, model_name: str, model_params: dict) -> None:
        """Add model information to plot."""
        if model_params:
            param_text = "\n".join([f"{k}: {v}" for k, v in model_params.items()])
            ax.text(0.02, 0.98, f"Model: {model_name}\n\nParameters:\n{param_text}",
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=8)

    def _add_metrics_annotation(self, ax: plt.Axes, metrics: Dict[str, float], 
                              loc: str = 'upper left') -> None:
        """Add formatted metrics annotation to plot."""
        metrics_text = "\n".join([f"{k}: {self._format_metric(v)}" 
                                for k, v in metrics.items()])
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
               verticalalignment='top', horizontalalignment='left')

    def _add_publication_styling(self, ax: plt.Axes, title: str = None,
                               xlabel: str = None, ylabel: str = None) -> None:
        """Apply consistent publication-ready styling to axis."""
        if title:
            ax.set_title(title, pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=10)
            
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust tick parameters
        ax.tick_params(direction='out', length=6, width=1.5)

    def plot_actual_vs_predicted(self, results: pd.DataFrame, model_name: str, model_params: dict) -> List[Path]:
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

    def plot_error_distribution(self, results: pd.DataFrame, model_name: str, model_params: dict) -> List[Path]:
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

    def plot_parity(self, results: pd.DataFrame, model_name: str, model_params: dict) -> List[Path]:
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

    def plot_residuals(self, results: pd.DataFrame, model_name: str, model_params: dict) -> List[Path]:
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
                              top_n: int = 15) -> List[Path]:
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

    def plot_feature_correlation(self, data: pd.DataFrame, model_name: str) -> List[Path]:
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

    def visualize_results(self, model, model_name: str) -> List[List[Path]]:
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

    def plot_predicted_vs_actual_enhanced(self, results: pd.DataFrame, model_name: str) -> List[Path]:
        """Create an enhanced scatter plot of predicted vs actual PCE values."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            ax.scatter(subset['PCE'] * 100, subset['PredictedPCE'] * 100,
                      label=dataset, color=self.colors[dataset],
                      marker=self.markers[dataset], alpha=0.7, s=100)
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, '--', color=self.colors['Reference'],
                alpha=0.8, label='Perfect Prediction')
        
        # Calculate and add metrics
        metrics = {
            'R²': r2_score(results['PCE'], results['PredictedPCE']),
            'RMSE': np.sqrt(mean_squared_error(results['PCE'], results['PredictedPCE'])),
            'MAE': mean_absolute_error(results['PCE'], results['PredictedPCE'])
        }
        self._add_metrics_annotation(ax, metrics)
        
        self._add_publication_styling(
            ax,
            title=f'Predicted vs. Actual PCE Values\n{model_name}',
            xlabel='Actual PCE (%)',
            ylabel='Predicted PCE (%)'
        )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return self._save_plot(fig, 'predicted_vs_actual_pce', enhanced=True)

    def plot_residuals_enhanced(self, results: pd.DataFrame, model_name: str) -> List[Path]:
        """Create an enhanced residuals plot with outlier detection."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate residuals
        results = results.copy()
        results['Residuals'] = (results['PCE'] - results['PredictedPCE']) * 100
        
        # Identify outliers (>2 standard deviations)
        std_dev = results['Residuals'].std()
        mean_residual = results['Residuals'].mean()
        results['Is_Outlier'] = abs(results['Residuals'] - mean_residual) > 2 * std_dev
        
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            
            # Plot non-outliers
            normal_points = subset[~subset['Is_Outlier']]
            ax.scatter(normal_points['PredictedPCE'] * 100, normal_points['Residuals'],
                      label=f'{dataset}', color=self.colors[dataset],
                      marker=self.markers[dataset], alpha=0.7, s=100)
            
            # Plot outliers with different marker
            outliers = subset[subset['Is_Outlier']]
            if not outliers.empty:
                ax.scatter(outliers['PredictedPCE'] * 100, outliers['Residuals'],
                          label=f'{dataset} (Outliers)', color=self.colors[dataset],
                          marker=self.markers['Outlier'], alpha=0.9, s=150,
                          edgecolor='black', linewidth=2)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color=self.colors['Reference'], linestyle='--', alpha=0.8)
        
        # Calculate and add metrics
        metrics = {
            'Mean Residual': results['Residuals'].mean(),
            'Std Residual': results['Residuals'].std(),
            'Outliers': results['Is_Outlier'].sum()
        }
        self._add_metrics_annotation(ax, metrics)
        
        self._add_publication_styling(
            ax,
            title=f'Residuals Plot\n{model_name}',
            xlabel='Predicted PCE (%)',
            ylabel='Residuals (%)'
        )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return self._save_plot(fig, 'residuals_plot', enhanced=True)

    def plot_feature_importance_enhanced(self, importance_df: pd.DataFrame, model_name: str,
                                      highlight_top: int = 5) -> List[Path]:
        """Create an enhanced feature importance plot."""
        # Sort features by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Create figure with adjusted height based on number of features
        height = max(8, len(importance_df) * 0.3)
        fig, ax = plt.subplots(figsize=(10, height))
        
        # Create colors array (highlight top N features)
        colors = [self.colors['Highlight'] if i >= len(importance_df) - highlight_top 
                 else self.colors['Neutral'] 
                 for i in range(len(importance_df))]
        
        # Create horizontal bar plot
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                      color=colors, alpha=0.7)
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center',
                   fontsize=10, fontweight='bold')
        
        self._add_publication_styling(
            ax,
            title=f'Feature Importance\n{model_name}',
            xlabel='Importance Score',
            ylabel='Features'
        )
        
        # Add annotation for top features
        top_features = importance_df.tail(highlight_top)['Feature'].tolist()
        annotation = 'Top 5 Features:\n' + '\n'.join(f'{i+1}. {f}' 
                    for i, f in enumerate(reversed(top_features)))
        ax.text(1.05, 0.95, annotation, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
               verticalalignment='top')
        
        plt.tight_layout()
        
        return self._save_plot(fig, 'feature_importance', enhanced=True)

    def plot_correlation_heatmap_enhanced(self, data: pd.DataFrame, model_name: str) -> List[Path]:
        """Create an enhanced correlation heatmap."""
        # Select numeric columns and calculate correlation
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Create figure with size based on number of features
        size = max(8, len(correlation_matrix) * 0.5)
        fig, ax = plt.subplots(figsize=(size, size))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                   annot=True, fmt='.2f', square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   vmin=-1, vmax=1)
        
        self._add_publication_styling(
            ax,
            title=f'Feature Correlation Heatmap\n{model_name}'
        )
        
        plt.tight_layout()
        return self._save_plot(fig, 'correlation_heatmap', enhanced=True)

    def plot_descriptor_pce_relationship(self, data: pd.DataFrame, descriptor: str,
                                       model_name: str) -> List[Path]:
        """Create scatter plot showing relationship between a descriptor and PCE."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for dataset in ['Training', 'Testing']:
            subset = data[data['Dataset'] == dataset]
            ax.scatter(subset[descriptor], subset['PCE'] * 100,
                      label=dataset, color=self.colors[dataset],
                      marker=self.markers[dataset], alpha=0.7, s=100)
        
        # Add regression line
        x = data[descriptor].values.reshape(-1, 1)
        y = data['PCE'].values * 100
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x, y)
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred = reg.predict(x_range)
        
        ax.plot(x_range, y_pred, '--', color=self.colors['Reference'],
               alpha=0.8, label=f'R² = {reg.score(x, y):.3f}')
        
        self._add_publication_styling(
            ax,
            title=f'{descriptor} vs. PCE\n{model_name}',
            xlabel=descriptor,
            ylabel='PCE (%)'
        )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return self._save_plot(fig, f'descriptor_vs_pce_{descriptor}', enhanced=True)

    def plot_error_distribution_enhanced(self, results: pd.DataFrame, model_name: str) -> List[Path]:
        """Create an enhanced error distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate residuals
        residuals = (results['PCE'] - results['PredictedPCE']) * 100
        
        # Plot histogram
        n_bins = int(np.sqrt(len(residuals)))  # Optimal bin size
        hist_data = ax.hist(residuals, bins=n_bins, density=True, alpha=0.6,
                          color=self.colors['Neutral'], label='Observed')
        
        # Fit and plot normal distribution
        from scipy import stats
        mu, std = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, '--', color=self.colors['Reference'],
               label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
        
        # Add KS test results
        ks_statistic, p_value = stats.kstest(residuals, 'norm', args=(mu, std))
        stats_text = (f'Kolmogorov-Smirnov Test:\n'
                     f'Statistic: {ks_statistic:.3f}\n'
                     f'p-value: {p_value:.3f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
               verticalalignment='top', horizontalalignment='right')
        
        self._add_publication_styling(
            ax,
            title=f'Distribution of Prediction Errors\n{model_name}',
            xlabel='Prediction Error (%)',
            ylabel='Density'
        )
        
        ax.legend(loc='upper left')
        plt.tight_layout()
        
        return self._save_plot(fig, 'error_distribution', enhanced=True)

    def plot_model_comparison(self, model_metrics: Dict[str, Dict[str, List[float]]],
                            model_names: List[str]) -> List[Path]:
        """Create an enhanced model comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['R²', 'RMSE', 'MAE']
        x = np.arange(len(model_names))
        width = 0.25
        multiplier = 0
        
        # Plot bars for each metric
        for metric in metrics:
            metric_values = [np.mean(model_metrics[model][metric]) for model in model_names]
            metric_errors = [np.std(model_metrics[model][metric]) for model in model_names]
            
            offset = width * multiplier
            rects = ax.bar(x + offset, metric_values, width, label=metric,
                          yerr=metric_errors, capsize=5)
            multiplier += 1
            
            # Add value labels on the bars
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, height,
                       f'{height:.3f}', ha='center', va='bottom',
                       fontsize=10)
        
        self._add_publication_styling(
            ax,
            title='Model Performance Comparison',
            xlabel='Models',
            ylabel='Metric Value'
        )
        
        ax.set_xticks(x + width, model_names)
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        return self._save_plot(fig, 'model_comparison', enhanced=True)

    def plot_ensemble_comparison(self, results_dict: Dict[str, pd.DataFrame],
                               model_names: List[str]) -> List[Path]:
        """Create an enhanced ensemble vs individual models comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for box plots
        residuals_data = []
        labels = []
        for model_name in model_names:
            results = results_dict[model_name]
            residuals = (results['PCE'] - results['PredictedPCE']) * 100
            residuals_data.append(residuals)
            labels.append(model_name)
        
        # Create box plot
        bp = ax1.boxplot(residuals_data, labels=labels, patch_artist=True)
        
        # Customize box colors
        colors = [self.colors['Training'], self.colors['Testing'],
                 self.colors['Validation'], self.colors['Highlight']]
        for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add scatter plot of predictions
        for i, model_name in enumerate(model_names):
            results = results_dict[model_name]
            ax2.scatter(results['PCE'] * 100, results['PredictedPCE'] * 100,
                       label=model_name, alpha=0.7)
        
        # Add diagonal line to scatter plot
        lims = [
            np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()])
        ]
        ax2.plot(lims, lims, '--', color=self.colors['Reference'],
                alpha=0.8, label='Perfect Prediction')
        
        # Style both plots
        self._add_publication_styling(
            ax1,
            title='Distribution of Prediction Errors',
            xlabel='Models',
            ylabel='Prediction Error (%)'
        )
        
        self._add_publication_styling(
            ax2,
            title='Prediction Comparison',
            xlabel='Actual PCE (%)',
            ylabel='Predicted PCE (%)'
        )
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return self._save_plot(fig, 'ensemble_comparison', enhanced=True)

    def plot_dye_ranking(self, results: pd.DataFrame, model_name: str,
                        highlight_top: int = 5) -> List[Path]:
        """Create an enhanced dye ranking plot."""
        # Sort dyes by predicted PCE
        results = results.sort_values('PredictedPCE', ascending=True)
        
        # Create figure with adjusted height
        height = max(8, len(results) * 0.3)
        fig, ax = plt.subplots(figsize=(12, height))
        
        # Create colors array (highlight top N dyes)
        colors = [self.colors['Highlight'] if i >= len(results) - highlight_top 
                 else self.colors[results.iloc[i]['Dataset']]
                 for i in range(len(results))]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(results)), results['PredictedPCE'] * 100,
                      color=colors, alpha=0.7)
        
        # Add value labels on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}%', ha='left', va='center',
                   fontsize=10, fontweight='bold')
        
        # Set y-tick labels to dye names
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(results['File'])
        
        self._add_publication_styling(
            ax,
            title=f'Ranking of Dyes by Predicted PCE\n{model_name}',
            xlabel='Predicted PCE (%)',
            ylabel='Dye'
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['Training'], alpha=0.7, label='Training Set'),
            Patch(facecolor=self.colors['Testing'], alpha=0.7, label='Testing Set'),
            Patch(facecolor=self.colors['Highlight'], alpha=0.7, label=f'Top {highlight_top}')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        return self._save_plot(fig, 'ranking_of_dyes', enhanced=True)