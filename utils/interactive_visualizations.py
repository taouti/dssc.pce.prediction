"""Interactive visualization utilities for PCE prediction analysis."""

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import shap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

class InteractivePCEVisualizer:
    """Interactive visualization tools for PCE prediction analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the interactive visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
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
    
    def plot_interactive_feature_importance(self, importance_df: pd.DataFrame, 
                                          model_name: str) -> go.Figure:
        """Create interactive feature importance plot with tooltips."""
        fig = go.Figure()
        
        # Sort features by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Add bar trace
        fig.add_trace(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color=self.colors['Highlight'],
            hovertemplate=(
                '<b>Feature:</b> %{y}<br>' +
                '<b>Importance:</b> %{x:.4f}<br>' +
                '<extra></extra>'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            height=max(600, len(importance_df) * 30)
        )
        
        return fig
    
    def plot_prediction_intervals(self, results: pd.DataFrame, 
                                confidence_level: float = 0.95) -> go.Figure:
        """Plot predictions with confidence intervals using bootstrap."""
        # Perform bootstrap to estimate intervals
        n_bootstrap = 1000
        predictions = []
        
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(
                len(results), size=len(results), replace=True
            )
            predictions.append(results.iloc[sample_idx]['PredictedPCE'].values)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, ((1 - confidence_level) / 2) * 100, axis=0)
        upper = np.percentile(predictions, (1 + confidence_level) / 2 * 100, axis=0)
        
        fig = go.Figure()
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=results.index,
            y=upper * 100,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=results.index,
            y=lower * 100,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add actual predictions
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['PredictedPCE'] * 100,
            mode='markers+lines',
            name='Predicted PCE',
            marker=dict(color=self.colors['Highlight']),
            hovertemplate='<b>PCE:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'PCE Predictions with {confidence_level*100}% Confidence Intervals',
            xaxis_title='Sample Index',
            yaxis_title='PCE (%)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_structure_performance_map(self, results: pd.DataFrame, 
                                     smiles_col: str) -> go.Figure:
        """Create interactive scatter plot with molecular structures."""
        def mol_to_img(smiles: str) -> str:
            """Convert SMILES to base64 image."""
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol)
            return img
        
        fig = go.Figure()
        
        for dataset in ['Training', 'Testing']:
            subset = results[results['Dataset'] == dataset]
            
            fig.add_trace(go.Scatter(
                x=subset['PredictedPCE'] * 100,
                y=subset['PCE'] * 100,
                mode='markers',
                name=dataset,
                marker=dict(
                    color=self.colors[dataset],
                    size=10
                ),
                hovertemplate=(
                    '<b>Actual PCE:</b> %{y:.2f}%<br>' +
                    '<b>Predicted PCE:</b> %{x:.2f}%<br>' +
                    '<extra></extra>'
                )
            ))
        
        # Add parity line
        max_val = max(results['PCE'].max(), results['PredictedPCE'].max()) * 100
        min_val = min(results['PCE'].min(), results['PredictedPCE'].min()) * 100
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Parity',
            line=dict(color='black', dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title='Structure-Performance Map',
            xaxis_title='Predicted PCE (%)',
            yaxis_title='Actual PCE (%)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_feature_interactions(self, data: pd.DataFrame, 
                                model, top_features: List[str]) -> go.Figure:
        """Visualize feature interactions using SHAP values."""
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data[top_features])
        
        fig = make_subplots(
            rows=len(top_features),
            cols=len(top_features),
            subplot_titles=[f"{f1} vs {f2}" 
                          for f1 in top_features 
                          for f2 in top_features]
        )
        
        for i, f1 in enumerate(top_features, 1):
            for j, f2 in enumerate(top_features, 1):
                if i != j:
                    fig.add_trace(
                        go.Scatter(
                            x=data[f1],
                            y=data[f2],
                            mode='markers',
                            marker=dict(
                                color=shap_values[:, i-1],
                                colorscale='RdBu',
                                showscale=True
                            ),
                            name=f'{f1} vs {f2}',
                            showlegend=False
                        ),
                        row=i,
                        col=j
                    )
        
        fig.update_layout(
            title='Feature Interaction Analysis',
            height=300*len(top_features),
            width=300*len(top_features),
            template='plotly_white'
        )
        
        return fig
    
    def plot_learning_curves(self, training_history: Dict) -> go.Figure:
        """Plot interactive learning curves."""
        fig = go.Figure()
        
        # Training metrics
        fig.add_trace(go.Scatter(
            x=list(range(len(training_history['train_score']))),
            y=training_history['train_score'],
            mode='lines',
            name='Training Score',
            line=dict(color=self.colors['Training'])
        ))
        
        # Validation metrics
        fig.add_trace(go.Scatter(
            x=list(range(len(training_history['val_score']))),
            y=training_history['val_score'],
            mode='lines',
            name='Validation Score',
            line=dict(color=self.colors['Validation'])
        ))
        
        # Add confidence bands if available
        if 'train_std' in training_history:
            train_upper = np.array(training_history['train_score']) + np.array(training_history['train_std'])
            train_lower = np.array(training_history['train_score']) - np.array(training_history['train_std'])
            
            fig.add_trace(go.Scatter(
                x=list(range(len(training_history['train_score']))) + 
                   list(range(len(training_history['train_score'])))[::-1],
                y=list(train_upper) + list(train_lower)[::-1],
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Training Confidence Band'
            ))
        
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Epoch',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        return fig
    
    def plot_model_disagreement(self, ensemble_predictions: pd.DataFrame) -> go.Figure:
        """Visualize ensemble model disagreement."""
        # Calculate prediction variance across models
        variance = ensemble_predictions.std(axis=1)
        mean_pred = ensemble_predictions.mean(axis=1)
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=mean_pred * 100,
            y=variance * 100,
            mode='markers',
            marker=dict(
                color=variance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Prediction Variance (%)')
            ),
            hovertemplate=(
                '<b>Mean Prediction:</b> %{x:.2f}%<br>' +
                '<b>Variance:</b> %{y:.2f}%<br>' +
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title='Model Disagreement Analysis',
            xaxis_title='Mean Predicted PCE (%)',
            yaxis_title='Prediction Variance (%)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_chemical_property_matrix(self, data: pd.DataFrame, 
                                    property_groups: Dict[str, List[str]]) -> go.Figure:
        """Create interactive correlation matrix for chemical properties."""
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False,
            hovertemplate=(
                '<b>Feature 1:</b> %{x}<br>' +
                '<b>Feature 2:</b> %{y}<br>' +
                '<b>Correlation:</b> %{z:.3f}<br>' +
                '<extra></extra>'
            )
        ))
        
        # Add property group annotations
        y_position = len(corr_matrix)
        for group, properties in property_groups.items():
            fig.add_annotation(
                x=-0.1,
                y=y_position,
                text=group,
                showarrow=False,
                font=dict(size=12)
            )
            y_position -= len(properties)
        
        fig.update_layout(
            title='Chemical Property Correlation Matrix',
            template='plotly_white'
        )
        
        return fig
    
    def plot_error_fingerprint(self, results: pd.DataFrame, 
                             feature_cols: List[str]) -> go.Figure:
        """Generate interactive error pattern visualization."""
        # Calculate normalized feature values
        normalized_features = (results[feature_cols] - 
                             results[feature_cols].mean()) / results[feature_cols].std()
        
        # Calculate absolute errors
        errors = abs(results['PCE'] - results['PredictedPCE']) * 100
        
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=normalized_features.values,
            x=feature_cols,
            y=results.index,
            colorscale='RdBu',
            zmid=0,
            customdata=np.dstack((errors, results['PCE']*100, 
                                results['PredictedPCE']*100))[0],
            hovertemplate=(
                '<b>Feature:</b> %{x}<br>' +
                '<b>Normalized Value:</b> %{z:.2f}<br>' +
                '<b>Prediction Error:</b> %{customdata[0]:.2f}%<br>' +
                '<b>Actual PCE:</b> %{customdata[1]:.2f}%<br>' +
                '<b>Predicted PCE:</b> %{customdata[2]:.2f}%<br>' +
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title='Error Fingerprint Analysis',
            xaxis_title='Features',
            yaxis_title='Samples',
            template='plotly_white'
        )
        
        return fig
    
    def plot_model_evolution(self, version_results: Dict[str, pd.DataFrame]) -> go.Figure:
        """Compare model versions interactively."""
        fig = go.Figure()
        
        metrics = ['R²', 'RMSE', 'MAE']
        versions = list(version_results.keys())
        
        for metric in metrics:
            values = []
            for version, results in version_results.items():
                if metric == 'R²':
                    value = np.corrcoef(results['PCE'], results['PredictedPCE'])[0,1]**2
                elif metric == 'RMSE':
                    value = np.sqrt(np.mean((results['PCE'] - results['PredictedPCE'])**2))
                else:  # MAE
                    value = np.mean(abs(results['PCE'] - results['PredictedPCE']))
                values.append(value)
            
            fig.add_trace(go.Scatter(
                x=versions,
                y=values,
                mode='lines+markers',
                name=metric,
                hovertemplate=(
                    '<b>Version:</b> %{x}<br>' +
                    f'<b>{metric}:</b> %{{y:.4f}}<br>' +
                    '<extra></extra>'
                )
            ))
        
        fig.update_layout(
            title='Model Performance Evolution',
            xaxis_title='Model Version',
            yaxis_title='Metric Value',
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_dashboard(self, results: pd.DataFrame, 
                                   model, feature_cols: List[str]) -> dash.Dash:
        """Create an interactive dashboard with all visualizations."""
        app = dash.Dash(__name__)
        
        # Calculate necessary data
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        })
        
        app.layout = html.Div([
            html.H1('PCE Prediction Analysis Dashboard'),
            
            html.Div([
                html.H2('Feature Importance'),
                dcc.Graph(
                    id='feature-importance',
                    figure=self.plot_interactive_feature_importance(
                        importance_df, 'Current Model'
                    )
                )
            ]),
            
            html.Div([
                html.H2('Prediction Intervals'),
                dcc.Graph(
                    id='prediction-intervals',
                    figure=self.plot_prediction_intervals(results)
                )
            ]),
            
            html.Div([
                html.H2('Feature Interactions'),
                dcc.Graph(
                    id='feature-interactions',
                    figure=self.plot_feature_interactions(
                        results, model, feature_cols[:5]
                    )
                )
            ]),
            
            html.Div([
                html.H2('Error Analysis'),
                dcc.Graph(
                    id='error-fingerprint',
                    figure=self.plot_error_fingerprint(
                        results, feature_cols
                    )
                )
            ])
        ])
        
        return app
