"""
Interactive Plotly dashboard for Pareto frontier visualization.

Provides 2D/3D scatter plots with tolerance visualization and feature selection.
"""

import dash
import logging
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from dash import dcc, html, Input, Output
from typing import Any, Dict, List, Optional


class Dashboard:
    """Interactive Plotly dashboard for Pareto frontier visualization."""
    
    def __init__(self, config: Dict[str, Any], data: np.ndarray, 
                 pareto_indices: List[int], tolerances: np.ndarray):
        """Initialize dashboard with data and configuration."""
        self.config = config
        self.data = data
        self.pareto_indices = pareto_indices
        self.tolerances = tolerances
        self.feature_names = list(config['data'].keys())
        self.logger = logging.getLogger(__name__)
        
        # Create Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Pareto Frontier Selection Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Label("X-axis Feature:"),
                    dcc.Dropdown(
                        id='x-feature',
                        options=[{'label': name, 'value': name} for name in self.feature_names],
                        value=self.feature_names[0] if self.feature_names else None,
                        style={'marginBottom': 10}
                    ),
                    
                    html.Label("Y-axis Feature:"),
                    dcc.Dropdown(
                        id='y-feature',
                        options=[{'label': name, 'value': name} for name in self.feature_names],
                        value=self.feature_names[1] if len(self.feature_names) > 1 else None,
                        style={'marginBottom': 10}
                    ),
                    
                    html.Label("Z-axis Feature (3D):"),
                    dcc.Dropdown(
                        id='z-feature',
                        options=[{'label': 'None', 'value': None}] + 
                               [{'label': name, 'value': name} for name in self.feature_names],
                        value=None,
                        style={'marginBottom': 10}
                    ),
                    
                    html.Label("Visualization Options:"),
                    dcc.Checklist(
                        id='viz-options',
                        options=[
                            {'label': 'Show Frontier', 'value': 'frontier'},
                            {'label': 'Show All Candidates', 'value': 'all_candidates'}
                        ],
                        value=['frontier'],
                        style={'marginBottom': 20}
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Plot area
                html.Div([
                    dcc.Graph(id='pareto-plot', style={'width': '100vw', 'height': '100vh'})
                ], style={'width': '75vw', 'display': 'inline-block', 'height': '100vh', 'verticalAlign': 'top'})
            ]),
            
            # Statistics panel
            html.Div([
                html.H3("Selection Statistics"),
                html.Div(id='stats-panel', style={'marginTop': 20})
            ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#f8f9fa'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            Output('pareto-plot', 'figure'),
            [Input('x-feature', 'value'),
             Input('y-feature', 'value'),
             Input('z-feature', 'value'),
             Input('viz-options', 'value')]
        )
        def update_plot(x_feature, y_feature, z_feature, viz_options):
            return self.create_plot(x_feature, y_feature, z_feature, viz_options)
        
        @self.app.callback(
            Output('stats-panel', 'children'),
            [Input('x-feature', 'value')]  # Dummy input to trigger update
        )
        def update_stats(_):
            return self.create_stats_panel()
    
    def create_plot(self, x_feature: str, y_feature: str, z_feature: Optional[str], 
                   viz_options: List[str]) -> go.Figure:
        """Create scatter plot with selected features."""
        if not x_feature or not y_feature:
            return go.Figure()
        x_idx = self.feature_names.index(x_feature)
        y_idx = self.feature_names.index(y_feature)
        z_idx = self.feature_names.index(z_feature) if z_feature else None
        show_frontier = 'frontier' in viz_options
        show_all = 'all_candidates' in viz_options
        is_3d = bool(z_feature)
        fig = go.Figure()
        pareto_x = self.data[self.pareto_indices, x_idx]
        pareto_y = self.data[self.pareto_indices, y_idx]
        if is_3d:
            pareto_z = self.data[self.pareto_indices, z_idx]
        # --- Dominated region or frontier surface ---
        if show_frontier:
            if not is_3d and len(pareto_x) > 1:
                # Dominated region (2D)
                y_obj = self.config['data'][y_feature]['objective']
                x_sorted_idx = np.argsort(pareto_x)
                x_front = pareto_x[x_sorted_idx]
                y_front = pareto_y[x_sorted_idx]
                if y_obj == 'minimize':
                    y_base = np.max(self.data[:, y_idx])
                    x_poly = np.concatenate(([x_front[0]], x_front, [x_front[-1]], [x_front[0]]))
                    y_poly = np.concatenate(([y_base], y_front, [y_base], [y_base]))
                else:
                    y_base = np.min(self.data[:, y_idx])
                    x_poly = np.concatenate(([x_front[0]], x_front, [x_front[-1]], [x_front[0]]))
                    y_poly = np.concatenate(([y_base], y_front, [y_base], [y_base]))
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly,
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Dominated region',
                    showlegend=True
                ))
            if is_3d and len(pareto_x) >= 4:
                # Frontier surface (3D)
                points3d = np.stack([pareto_x, pareto_y, pareto_z], axis=1)
                try:
                    hull = ConvexHull(points3d)
                    fig.add_trace(go.Mesh3d(
                        x=points3d[:,0], y=points3d[:,1], z=points3d[:,2],
                        i=hull.simplices[:,0],
                        j=hull.simplices[:,1],
                        k=hull.simplices[:,2],
                        opacity=0.3,
                        color='lightblue',
                        name='Frontier surface',
                        showlegend=True
                    ))
                except Exception:
                    pass
        # --- All candidates ---
        if show_all:
            all_x = self.data[:, x_idx]
            all_y = self.data[:, y_idx]
            if is_3d:
                all_z = self.data[:, z_idx]
                fig.add_trace(go.Scatter3d(
                    x=all_x, y=all_y, z=all_z,
                    mode='markers',
                    name='All Candidates',
                    marker=dict(size=4, color='lightblue', opacity=0.6),
                    hovertemplate=f'<b>Candidate</b><br>' +
                                 f'{x_feature}: %{{x:.3f}}<br>' +
                                 f'{y_feature}: %{{y:.3f}}<br>' +
                                 f'{z_feature}: %{{z:.3f}}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=all_x, y=all_y,
                    mode='markers',
                    name='All Candidates',
                    marker=dict(size=4, color='lightblue', opacity=0.6),
                    hovertemplate=f'<b>Candidate</b><br>' +
                                 f'{x_feature}: %{{x:.3f}}<br>' +
                                 f'{y_feature}: %{{y:.3f}}<extra></extra>'
                ))
        # --- Pareto frontier points ---
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=pareto_x, y=pareto_y, z=pareto_z,
                mode='markers',
                name='Pareto Frontier',
                marker=dict(size=8, color='red', symbol='diamond'),
                hovertemplate=f'<b>Pareto Candidate</b><br>' +
                             f'{x_feature}: %{{x:.3f}}<br>' +
                             f'{y_feature}: %{{y:.3f}}<br>' +
                             f'{z_feature}: %{{z:.3f}}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=pareto_x, y=pareto_y,
                mode='markers',
                name='Pareto Frontier',
                marker=dict(size=8, color='red', symbol='diamond'),
                hovertemplate=f'<b>Pareto Candidate</b><br>' +
                             f'{x_feature}: %{{x:.3f}}<br>' +
                             f'{y_feature}: %{{y:.3f}}<extra></extra>'
            ))
        # --- Layout ---
        layout_args = dict(
            title=f"Pareto Frontier: {x_feature} vs {y_feature}" + (f" vs {z_feature}" if is_3d else ''),
            autosize=True,
        )
        if is_3d:
            layout_args['scene'] = dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature,
                aspectmode='cube'
            )
        else:
            layout_args['xaxis_title'] = x_feature
            layout_args['yaxis_title'] = y_feature
        fig.update_layout(**layout_args)
        return fig
    
    def add_tolerance_visualization(self, fig: go.Figure, x_feature: str, y_feature: str, 
                                  z_feature: Optional[str], is_3d: bool):
        """Add tolerance visualization to the plot."""
        x_idx = self.feature_names.index(x_feature)
        y_idx = self.feature_names.index(y_feature)
        z_idx = self.feature_names.index(z_feature) if z_feature else None
        
        # Get tolerance values
        x_tol = self.tolerances[0, x_idx] if self.tolerances.ndim == 2 else self.tolerances[x_idx]
        y_tol = self.tolerances[0, y_idx] if self.tolerances.ndim == 2 else self.tolerances[y_idx]
        z_tol = self.tolerances[0, z_idx] if z_idx is not None and self.tolerances.ndim == 2 else self.tolerances[z_idx] if z_idx is not None else None
        
        # Add tolerance ellipses/ellipsoids for Pareto points
        for idx in self.pareto_indices:
            x_val = self.data[idx, x_idx]
            y_val = self.data[idx, y_idx]
            
            if is_3d and z_idx is not None:
                z_val = self.data[idx, z_idx]
                # Add 3D ellipsoid (simplified as circle)
                fig.add_trace(go.Scatter3d(
                    x=[x_val], y=[y_val], z=[z_val],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='rgba(255,0,0,0.1)',
                        symbol='circle'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                # Add 2D ellipse (simplified as circle)
                fig.add_trace(go.Scatter(
                    x=[x_val], y=[y_val],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='rgba(255,0,0,0.1)',
                        symbol='circle'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def create_stats_panel(self) -> html.Div:
        """Create statistics panel."""
        total_candidates = self.data.shape[0]
        pareto_count = len(self.pareto_indices)
        reduction_ratio = pareto_count / total_candidates * 100
        
        return html.Div([
            html.P(f"Total Candidates: {total_candidates}"),
            html.P(f"Pareto Frontier Candidates: {pareto_count}"),
            html.P(f"Reduction Ratio: {reduction_ratio:.1f}%"),
            html.P(f"Features: {', '.join(self.feature_names)}")
        ])
    
    def run(self, host: str = 'localhost', port: int = 8050, debug: bool = False):
        """Run the dashboard."""
        self.logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug) 