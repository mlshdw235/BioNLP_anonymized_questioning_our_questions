"""
Clinical Performance Data Visualization and Analysis

This module provides functions for analyzing and visualizing clinical task performance
across different models, including heatmaps and distribution plots.
"""
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list

# Global constants
DATA_DIR = "perf_data_pickle"
TASK_ID_MAP_PATH = f"{DATA_DIR}/clinical_task_id_map.pkl"
PERF_DATA_PATH = f"{DATA_DIR}/perf_data_clinical.pkl"
OUTPUT_DF_PATH = f"{DATA_DIR}/perf_data_clinical_df.pkl"
OUTPUT_CSV_PATH = f"{DATA_DIR}/perf_data_clinical_df.csv"

# Visualization constants
TASK_NAME_MAX_LENGTH = 30
PLOT_WIDTH_MULTIPLIER = 1.6
PLOT_HEIGHT_MULTIPLIER = 1.7
BASE_PLOT_WIDTH = 700
BASE_PLOT_HEIGHT = 800
DISTRIBUTION_PLOT_WIDTH = 700
DISTRIBUTION_PLOT_HEIGHT = 420
DISTRIBUTION_HEIGHT_MULTIPLIER = 1.15


def load_data():
    """
    Load pickled data for clinical tasks and performance.
    """
    with open(TASK_ID_MAP_PATH, 'rb') as f:
        clinical_task_id_map = pickle.load(f)
    with open(PERF_DATA_PATH, 'rb') as f:
        clinical_perf_data = pickle.load(f)
    return clinical_task_id_map, clinical_perf_data


def create_dataframe(clinical_task_id_map, clinical_perf_data):
    """
    Convert performance data dict into a pandas DataFrame with task name mapping.
    """
    rows = []
    for task_id, info in clinical_perf_data.items():
        ref_idx, ref_title = info['reference']
        for metric in info['metrics']:
            model_name, score, total_count = metric
            # Map task_id to task_name if available and truncate if needed
            task_name = clinical_task_id_map.get(task_id, {'task_name': f"Task_{task_id}"})['task_name']
            task_name = task_name[:TASK_NAME_MAX_LENGTH] 
            if len(task_name) > TASK_NAME_MAX_LENGTH:
                task_name += '...'
                
            row = {
                'task_id': task_id,
                'task_name': task_name,
                'reference_idx': ref_idx,
                'reference_title': ref_title,
                'model_name': model_name,
                'score': round(score, 2),
                'total_count': total_count
            }
            rows.append(row)
    return pd.DataFrame(rows)


def plot_task_model_heatmap(df, dendrogram_model=True, dendrogram_task=False):
    """
    Plot an interactive heatmap of task vs model performance with optional dendrograms.
    """
    # Pivot: task_name × model_name -> score
    pivot_df = df.pivot_table(
        index='task_name',
        columns='model_name',
        values='score',
        aggfunc='mean'
    )

    # Handle NaN for clustering
    cluster_data = pivot_df.fillna(pivot_df.mean())
    
    # Perform hierarchical clustering
    row_linkage = linkage(cluster_data, method='average', metric='euclidean')
    col_linkage = linkage(cluster_data.T, method='average', metric='euclidean')
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the pivot table according to clustering
    pivot_reordered = pivot_df.iloc[row_order, :].iloc[:, col_order]
    
    # Create z values with NaN filled as -1 (will be displayed as light gray)
    z_values = pivot_reordered.values
    z_display = np.where(np.isnan(z_values), -1, z_values)
    
    # Create custom colorscale with light gray for NaN (-1)
    colorscale = [
        [0, 'rgb(240,240,240)'],  # Light gray for NaN (-1)
        [0.01, 'rgb(68,1,84)'],   # Regular viridis colorscale starts here
        [0.25, 'rgb(59,82,139)'],
        [0.5, 'rgb(33,144,141)'],
        [0.75, 'rgb(94,201,98)'],
        [1, 'rgb(253,231,37)']
    ]
    
    # Create mask for NaN values and prepare hover text
    is_nan = np.isnan(z_values)
    hover_text = np.where(
        is_nan,
        'No Data', 
        np.array([f'Score: {v:.1f}' for v in z_values.flatten()]).reshape(z_values.shape)
    )
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_display,
        x=pivot_reordered.columns,
        y=pivot_reordered.index,
        colorscale=colorscale,
        zmid=50,  # Center the colorscale around 50
        hovertemplate='Model: %{x}<br>Task: %{y}<br>%{customdata}<extra></extra>',
        customdata=hover_text
    ))

    # Add outer border
    shapes = _create_heatmap_border_shapes(pivot_reordered)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Task vs. Model Performance",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        width=BASE_PLOT_WIDTH * PLOT_WIDTH_MULTIPLIER,
        height=BASE_PLOT_HEIGHT * PLOT_HEIGHT_MULTIPLIER,
        xaxis=dict(
            title="Model",
            showgrid=False,
            tickangle=45,
            range=[-0.5, len(pivot_reordered.columns) - 0.5]
        ),
        yaxis=dict(
            showticklabels=False,  # Remove task labels
            showgrid=False,
            title=None,  # Remove y-axis title
            range=[-0.5, len(pivot_reordered.index) - 0.5]
        ),
        margin=dict(l=60, r=50, t=100, b=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
        shapes=shapes
    )
    
    fig.show()


def _create_heatmap_border_shapes(pivot_df):
    """Create border shapes for the heatmap visualization."""
    shapes = []
    # Add outer border
    shapes.extend([
        # Top border
        dict(
            type='line', 
            x0=-0.5, 
            x1=len(pivot_df.columns) - 0.5, 
            y0=-0.5, 
            y1=-0.5, 
            line=dict(color='black', width=1.5, dash='solid')
        ),
        # Bottom border
        dict(
            type='line', 
            x0=-0.5, 
            x1=len(pivot_df.columns) - 0.5, 
            y0=len(pivot_df.index) - 0.5, 
            y1=len(pivot_df.index) - 0.5,
            line=dict(color='black', width=1.5, dash='solid')
        ),
        # Left border
        dict(
            type='line', 
            x0=-0.5, 
            x1=-0.5, 
            y0=-0.5, 
            y1=len(pivot_df.index) - 0.5,
            line=dict(color='black', width=1.5, dash='solid')
        ),
        # Right border
        dict(
            type='line', 
            x0=len(pivot_df.columns) - 0.5, 
            x1=len(pivot_df.columns) - 0.5,
            y0=-0.5, 
            y1=len(pivot_df.index) - 0.5,
            line=dict(color='black', width=1.5, dash='solid')
        )
    ])
    return shapes


def plot_model_wise_distribution(df):
    """
    Plot distribution of scores across models, sorted by descending average performance.
    """
    # Calculate model-wise average score and sort by descending score
    model_mean = df.groupby('model_name')['score'].mean().reset_index()
    model_mean = model_mean.sort_values(by='score', ascending=False)

    # Create mapping for model order and apply to full dataset
    model_order = {model: idx for idx, model in enumerate(model_mean['model_name'])}
    df['model_order'] = df['model_name'].map(model_order)
    model_mean['model_order'] = model_mean['model_name'].map(model_order)

    # Find the index for 'human - doctors' for reference line
    human_doctors_order = model_order.get('human - doctors', None)

    # Create the figure
    fig = go.Figure()
    
    # Add individual points
    fig.add_trace(go.Scatter(
        x=df['model_order'],
        y=df['score'],
        mode='markers',
        name='Individual Scores',
        marker=dict(
            color='red',
            size=5,
            opacity=0.3
        ),
        hovertemplate=(
            'Model: %{customdata[0]}<br>'
            'Score: %{y:.2f}<br>'
            'Task: %{customdata[1]}'
        ),
        customdata=list(zip(df['model_name'], df['task_name']))
    ))

    # Add average line with fill
    fig.add_trace(go.Scatter(
        x=model_mean['model_order'],
        y=model_mean['score'],
        mode='lines',
        name='Average Score',
        line=dict(color='skyblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(135, 206, 235, 0.3)',
        hovertemplate=(
            'Model: %{customdata}<br>'
            'Average Score: %{y:.2f}'
        ),
        customdata=model_mean['model_name']
    ))

    # Add vertical reference line for human-doctors if it exists
    if human_doctors_order is not None:
        fig.add_vline(
            x=human_doctors_order,
            line=dict(
                color='rgba(0, 0, 0, 0.5)',  # Semi-transparent black
                width=1,
                dash='dot'  # Dotted line
            ),
            annotation_text='Human Doctors',
            annotation_position='bottom'
        )

    # Update layout with adjusted size
    fig.update_layout(
        title=dict(
            text="Model Performance Distribution",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        xaxis_title="Model",
        yaxis_title="Score",
        showlegend=True,
        width=DISTRIBUTION_PLOT_WIDTH,
        height=DISTRIBUTION_PLOT_HEIGHT * DISTRIBUTION_HEIGHT_MULTIPLIER,
        margin=dict(l=80, r=50, t=100, b=80),
        xaxis=dict(
            showticklabels=False,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            range=[-0.5, len(model_order) - 0.5]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            range=[0, 100],  # Score range from 0 to 100
            tickfont=dict(size=12)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        font=dict(family="Arial", size=14)
    )
    fig.show()


def main():
    """
    Main execution function that loads data, creates dataframe, and generates visualizations.
    """
    clinical_task_id_map, clinical_perf_data = load_data()
    df = create_dataframe(clinical_task_id_map, clinical_perf_data)
    
    # Calculate mean scores per group to avoid duplicates
    df = df.groupby(
        ['task_id', 'task_name', 'reference_idx', 'reference_title', 'model_name', 'total_count'], 
        as_index=False
    )["score"].mean()
    
    # Save processed data
    with open(OUTPUT_DF_PATH, "wb") as f:
        pickle.dump(df, f)
    df.to_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
    
    # Generate visualizations
    plot_task_model_heatmap(df, dendrogram_model=False, dendrogram_task=False)
    plot_model_wise_distribution(df)


if __name__ == "__main__":
    main()