import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.colors as mcolors
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import matplotlib.lines as mlines

# Directory configurations
DATA_DIR = "."
FIGURE_OUTPUT_DIR = "figures"
BAYESIAN_RESULTS_DIR = "bayesian_analysis_results"
PERFORMANCE_DATA_DIR = "perf_data_pickle"

# File paths
BAYESIAN_PERFORMANCE_FILE = f"{BAYESIAN_RESULTS_DIR}/model_performance.pkl"
CLINICAL_PERFORMANCE_FILE = f"{PERFORMANCE_DATA_DIR}/perf_data_clinical_df.pkl"
OUTPUT_BASE_FILENAME = f"{FIGURE_OUTPUT_DIR}/bayesian_vs_clinical_model_performance_comparison"

# Plot configurations
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8
BAR_WIDTH = 0.35
JITTER_RANGE = 0.1  # Range for scatter point jitter
FONT_SIZE_LABELS = 18
FONT_SIZE_TICKS = 13
LABEL_PADDING = 15
ROTATION_ANGLE = 65
DPI_VALUE = 600
SCATTER_POINT_SIZE = 30
SCATTER_ALPHA = 0.6

# Color configurations
COLOR_BAYESIAN = 'royalblue'
COLOR_CLINICAL = 'lightcoral'
COLOR_INDIVIDUAL = 'darkred'
COLOR_PROPRIETARY = '#4B0082'  # Navy blue
COLOR_OPEN = '#D55E00'  # Dark orange
COLOR_HUMAN = '#008080'  # Teal

# Feature flags
BAYESIAN_WITH_HIGH_CONNECTIVITY = False
MARK_LOW_CONNECTIVITY = True
HIGHLIGHT_MODEL_TYPE = True

# Model property identifiers
PROPRIETARY_MODEL_KEYWORDS = ['gpt', 'claude', 'perplexity', 'gemini', 'palm']
FINETUNED_MODEL_KEYWORDS = ['meditron', 'medalpaca', 'biomistral']
SMALL_MODEL_SIZE_INDICATORS = ['7b', '8b', '13b']


def parsing_model_string(model):
    """
    Parse a model string to identify its properties.
    
    Args:
        model: The model string to parse
        
    Returns:
        tuple: (is_prop_model, is_open_model, is_human_baseline, is_sllm, is_finetuned)
    """
    model_lower = model.lower()
    
    # Check for proprietary models
    is_prop_model = any(prop in model_lower for prop in PROPRIETARY_MODEL_KEYWORDS)
    
    # Check for human baseline
    is_human_baseline = 'human' in model_lower
    
    # Open model check (if not proprietary and not human)
    is_open_model = not (is_prop_model or is_human_baseline)
    
    # Check for small language models based on parameter size indicators
    # Excluding composite models like mixtral-8x7B
    is_sllm = any(size.upper() in model for size in SMALL_MODEL_SIZE_INDICATORS) and '8x' not in model_lower
    
    # Check for domain-specific fine-tuned models
    is_finetuned = any(ft_model in model_lower for ft_model in FINETUNED_MODEL_KEYWORDS)

    return is_prop_model, is_open_model, is_human_baseline, is_sllm, is_finetuned


def format_model_name(model):
    """
    Format model name with visual indicators based on model properties.
    
    Args:
        model: The model name to format
        
    Returns:
        Formatted model name with appropriate styling
    """
    if not HIGHLIGHT_MODEL_TYPE:
        return model
    
    is_prop_model, is_open_model, is_human_baseline, is_sllm, is_finetuned = parsing_model_string(model)
    
    formatted_name = model
    
    # Add prefix symbols for small models
    if is_sllm:
        formatted_name = f"⊙ {formatted_name}"
    
    # Add prefix for fine-tuned models
    if is_finetuned:
        formatted_name = f"★ {formatted_name}"
    
    # Add arrows for human baseline
    if is_human_baseline:
        formatted_name = f"→ {formatted_name} ←"
    
    return formatted_name


def prepare_connectivity_info(adjusted_model_perfs):
    """
    Extract connectivity information from the performance dataframe.
    
    Args:
        adjusted_model_perfs: DataFrame with model performances
        
    Returns:
        dict: Mapping of model names to connectivity status
    """
    connectivity_info = {}
    if 'high_connectivity' in adjusted_model_perfs.columns:
        for model in adjusted_model_perfs.index:
            connectivity_info[model] = adjusted_model_perfs.loc[model, 'high_connectivity']
    return connectivity_info


def extract_scores(adjusted_model_perfs, df_clinical_perfs):
    """
    Extract Bayesian and clinical scores from dataframes.
    
    Args:
        adjusted_model_perfs: DataFrame with Bayesian scores
        df_clinical_perfs: DataFrame with clinical scores
        
    Returns:
        tuple: (bayesian_df, clinical_df, individual_scores)
    """
    # Extract Bayesian scores
    bayesian_scores = {}
    if 'original_scale' in adjusted_model_perfs.columns:
        for model in adjusted_model_perfs.index:
            bayesian_scores[model] = adjusted_model_perfs.loc[model, 'original_scale']
    
    bayesian_df = pd.DataFrame.from_dict(bayesian_scores, orient='index', columns=['Bayesian Score'])
    
    # Extract clinical scores and individual scores
    clinical_scores = {}
    individual_scores = {}
    
    for model in df_clinical_perfs['model_name'].unique():
        model_data = df_clinical_perfs[df_clinical_perfs['model_name'] == model]
        if not model_data.empty:
            clinical_scores[model] = model_data['score'].mean()
            individual_scores[model] = model_data['score'].tolist()
    
    clinical_df = pd.DataFrame.from_dict(clinical_scores, orient='index', columns=['Clinical Score'])
    
    return bayesian_df, clinical_df, individual_scores


def create_hatch_patterns(comparison_df, connectivity_info):
    """
    Create hatch patterns for low connectivity models.
    
    Args:
        comparison_df: DataFrame with model comparisons
        connectivity_info: Dict with connectivity information
        
    Returns:
        list: Hatch patterns for each model
    """
    hatch_patterns = []
    for model in comparison_df.index:
        # Add hatching for low connectivity models when specified
        if (MARK_LOW_CONNECTIVITY and not BAYESIAN_WITH_HIGH_CONNECTIVITY and 
            model in connectivity_info and not connectivity_info[model]):
            hatch_patterns.append('///')
        else:
            hatch_patterns.append('')
    return hatch_patterns


def create_legend_elements(comparison_df, individual_scores):
    """
    Create legend elements for the plot.
    
    Args:
        comparison_df: DataFrame with model comparisons
        individual_scores: Dict with individual clinical scores
        
    Returns:
        list: Legend elements for the plot
    """
    legend_elements = [
        Patch(facecolor=COLOR_BAYESIAN, edgecolor='black', label='Our Scores'),
        Patch(facecolor=COLOR_CLINICAL, edgecolor='black', label='Simple Mean Performances')
    ]
    
    # Add low connectivity legend item if needed
    if MARK_LOW_CONNECTIVITY and not BAYESIAN_WITH_HIGH_CONNECTIVITY:
        legend_elements.append(
            Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='Low Connectivity Models')
        )
    
    # Add scatter point legend if individual scores exist
    if comparison_df.index[0] in individual_scores:
        legend_elements.append(
            mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INDIVIDUAL, 
                         markersize=8, alpha=SCATTER_ALPHA, label='Individual Scores')
        )
    
    # Add model type legend elements if highlighting is enabled
    if HIGHLIGHT_MODEL_TYPE:
        legend_elements.extend([
            Text(0, 0, '→Model←', fontweight='bold', color='black', 
                 label='Human Baseline'),
            Text(0, 0, 'Model', color=COLOR_PROPRIETARY, 
                 label='Proprietary Model'),
            Text(0, 0, 'Model', color=COLOR_OPEN, 
                 label='Open Model'),
            Text(0, 0, '★ Model', color='black', 
                 label='Fine-tuned Model'),
            Text(0, 0, '⊙ Model', color='black', 
                 label='Small LLM')
        ])
    
    return legend_elements


def style_x_tick_labels(comparison_df):
    """
    Apply color and style to x-tick labels based on model type.
    
    Args:
        comparison_df: DataFrame with model comparisons
    """
    if not HIGHLIGHT_MODEL_TYPE:
        return
    
    for i, model in enumerate(comparison_df.index):
        is_prop_model, is_open_model, is_human_baseline, _, _ = parsing_model_string(model)
        
        # Get the text object for this tick label
        text_obj = plt.gca().get_xticklabels()[i]
        
        # Apply appropriate color based on model type and make all bold
        text_obj.set_weight('bold')
        if is_human_baseline:
            text_obj.set_color(COLOR_HUMAN)
        elif is_prop_model:
            text_obj.set_color(COLOR_PROPRIETARY)
        elif is_open_model:
            text_obj.set_color(COLOR_OPEN)


def compare_model_performances(adjusted_model_perfs, df_clinical_perfs):
    """
    Compare model performances between two dataframes and plot the results.
    Only includes models with both Bayesian and Clinical scores.
    
    Args:
        adjusted_model_perfs: DataFrame with model performances from Bayesian analysis
        df_clinical_perfs: DataFrame with clinical performances
        
    Returns:
        tuple: (comparison_df, figure) Comparison DataFrame and matplotlib figure
    """
    # Ensure adjusted_model_perfs has the correct index
    if isinstance(adjusted_model_perfs, pd.DataFrame) and 'model' in adjusted_model_perfs.columns:
        adjusted_model_perfs = adjusted_model_perfs.set_index('model')
    
    # Save connectivity information for hatching
    connectivity_info = prepare_connectivity_info(adjusted_model_perfs)
    
    # Filter by high_connectivity if the global flag is set
    if BAYESIAN_WITH_HIGH_CONNECTIVITY and 'high_connectivity' in adjusted_model_perfs.columns:
        print("Filtering for high_connectivity models only")
        adjusted_model_perfs = adjusted_model_perfs[adjusted_model_perfs['high_connectivity'] == True]
        print(f"Filtered to {len(adjusted_model_perfs)} high_connectivity models")
    
    # Extract scores from dataframes
    bayesian_df, clinical_df, individual_scores = extract_scores(adjusted_model_perfs, df_clinical_perfs)
    
    # Print DataFrames for debugging
    print("\nOur scores:")
    print(bayesian_df.head())
    print("\nSimple mean performances:")
    print(clinical_df.head())
    
    # Merge the two dataframes - only keep models with both scores
    comparison_df = bayesian_df.join(clinical_df, how='inner')
    
    # Print the merged DataFrame
    print("\nMerged comparison DataFrame:")
    print(comparison_df.head())
    
    # Sort by Bayesian Score in descending order
    comparison_df = comparison_df.sort_values('Bayesian Score', ascending=False)
    
    if comparison_df.empty:
        print("Warning: No models found with both Bayesian and Clinical scores!")
        return pd.DataFrame(), plt.figure()
    
    # Create mapping from model names to indices (0, 1, 2, ...)
    model_idx_map = {model: i for i, model in enumerate(comparison_df.index)}
    
    # Plot the comparison
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    x = np.arange(len(comparison_df))
    
    # Determine which models should have hatching (low connectivity models)
    hatch_patterns = create_hatch_patterns(comparison_df, connectivity_info)
    
    # Plot bars with hatching where appropriate
    for i, model in enumerate(comparison_df.index):
        plt.bar(x[i] - BAR_WIDTH/2, comparison_df.loc[model, 'Bayesian Score'], 
                BAR_WIDTH, color=COLOR_BAYESIAN, edgecolor='black', 
                hatch=hatch_patterns[i])
        
        plt.bar(x[i] + BAR_WIDTH/2, comparison_df.loc[model, 'Clinical Score'], 
                BAR_WIDTH, color=COLOR_CLINICAL, edgecolor='black',
                hatch=hatch_patterns[i])
    
    # Create legend elements
    legend_elements = create_legend_elements(comparison_df, individual_scores)
    
    # Add individual clinical scores as scatter points
    for model in comparison_df.index:
        if model in individual_scores:
            # Get position on x-axis for this model
            model_idx = model_idx_map[model]
            # Add small jitter to avoid overlapping points
            x_jitter = [model_idx + BAR_WIDTH/2 + random.uniform(-JITTER_RANGE, JITTER_RANGE) 
                        for _ in individual_scores[model]]
            plt.scatter(x_jitter, individual_scores[model], color=COLOR_INDIVIDUAL, 
                       alpha=SCATTER_ALPHA, s=SCATTER_POINT_SIZE, zorder=3, 
                       label='Individual Scores' if model == comparison_df.index[0] else "")
    
    # Add labels and formatting
    plt.xlabel('Model Name', fontsize=FONT_SIZE_LABELS, labelpad=LABEL_PADDING)
    plt.ylabel('Scores', fontsize=FONT_SIZE_LABELS)
    plt.xlim(min(x) - 1, max(x) + 1)
    
    # Use formatted model names for x-ticks
    formatted_model_names = [format_model_name(model) for model in comparison_df.index]
    plt.xticks(x, formatted_model_names, rotation=ROTATION_ANGLE, ha='right', fontsize=FONT_SIZE_TICKS)
    
    # Style x-tick labels based on model type
    style_x_tick_labels(comparison_df)
    
    # Add legend with transparency
    legend = plt.legend(handles=legend_elements,
                       title=r'$\mathbf{Score\ Type\ and\ Model\ Type}$',
                       loc='upper right', fontsize=10,
                       framealpha=0.8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return comparison_df, plt.gcf()


def save_figures(fig, base_filename):
    """
    Save figure in multiple formats.
    
    Args:
        fig: The matplotlib figure to save
        base_filename: The base filename without extension
    """
    # PNG format with high DPI
    plt.savefig(f'{base_filename}.png', dpi=DPI_VALUE, bbox_inches='tight')

    # TIFF format with high DPI
    plt.savefig(f'{base_filename}.tiff', dpi=DPI_VALUE, bbox_inches='tight')

    # Create PDF with embedded text
    with PdfPages(f'{base_filename}.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        # Add metadata to the PDF
        d = pdf.infodict()
        d['Title'] = 'Model Performance Comparison'
        d['Author'] = 'Automated Analysis'
        d['Subject'] = 'Comparison of Bayesian and Clinical Model Performances'
        d['Keywords'] = 'models, performance, bayesian, clinical, connectivity, model type'
    
    print(f"Figures saved as PNG, TIFF, and PDF formats at {base_filename}.*")


def main():
    """Main function to execute the model performance comparison flow."""
    # Load Bayesian performance data
    with open(BAYESIAN_PERFORMANCE_FILE, 'rb') as f:
        adjusted_model_perfs = pickle.load(f)

    # Load clinical performance data
    with open(CLINICAL_PERFORMANCE_FILE, 'rb') as f:
        df_clinical_perfs = pickle.load(f)

    # Generate comparison and plot
    comparison_df, fig = compare_model_performances(adjusted_model_perfs, df_clinical_perfs)

    # Display comparison table
    print("\nFinal Model Performance Comparison (models with both scores):")
    print(comparison_df)

    # Save figures in multiple formats
    save_figures(fig, OUTPUT_BASE_FILENAME)
    
    plt.show()


if __name__ == "__main__":
    main()
