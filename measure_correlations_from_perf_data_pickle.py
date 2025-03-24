"""
Analyze correlations between benchmark performance data.
This module computes various correlation metrics between general and medical benchmarks
and visualizes the results.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, kendalltau

# Configuration constants
PLOT_AS_PANEL = True  # Set to True for panel visualization, False for individual plots
SAVE_PLOTS = True  # Set to True to save generated plots
NORMALIZE_DATA = True  # Set to True to use normalized data

# Directory and file paths
DATA_DIR = r"G:\내 드라이브\[1] CCADD N CBDL\[1] Personal Research\2025_MedQACorr\perf_data_pickle"
GENERAL_PERF_FILE = 'perf_data_general_fin_normalized.pkl'
MEDICAL_PERF_FILE = 'perf_data_medqa_fin_normalized.pkl'
OUTPUT_DIR = 'figures'
DATASET_DIR = 'final_dataset'

# Adjust filenames based on normalization setting
if not NORMALIZE_DATA:
    GENERAL_PERF_FILE = GENERAL_PERF_FILE.replace('_normalized', 'wo_normalized')
    MEDICAL_PERF_FILE = MEDICAL_PERF_FILE.replace('_normalized', 'wo_normalized')

# Correlation methods to compute
CORR_METHODS = ['pearson', 'spearman', 'kendall', 'lin_ccc']
CORR_METHOD_LABELS = {
    'Pearson Correlation': 'Pearson Correlation', 
    'Spearman Correlation': 'Spearman Correlation', 
    'Kendall Correlation': 'Kendall\'s Tau Correlation', 
    'Lin_ccc Correlation': 'Lin\'s CCC'
}

# Benchmark categories
MED_BENCHES = [
    'MedQA',
    'MedMCQA',
    'MMLU Anatomy',
    'MMLU Clinical Knowledge',
    'MMLU College Biology',
    'MMLU College Medicine',
    'MMLU Medical Genetics',
    'MMLU Professional Medicine',
    'PubMedQA',
]

# Knowledge and Question Answering benchmarks
KNOWLEDGE_BENCHES = [
    'MMLU',            # Massive multitask language understanding
    'MMLU Pro',        # Advanced version of MMLU
]

# Reasoning and Problem Solving benchmarks
REASONING_BENCHES = [
    'BBH',             # Big Bench Hard - diverse reasoning tasks
]

# Code Generation benchmarks
CODE_BENCHES = [
    'HumanEval'        # Code generation and functional correctness
]

# Mathematics benchmarks
MATH_BENCHES = [
    'GSM8K',           # Grade school math word problems
    'MATH',            # Advanced mathematics problems
]

# Common Sense and Language Understanding benchmarks
COMMON_SENSE_BENCHES = []

# Combined general benchmarks
GENERAL_BENCHES = KNOWLEDGE_BENCHES + REASONING_BENCHES + CODE_BENCHES + \
    MATH_BENCHES + COMMON_SENSE_BENCHES

# Visualization constants
MIN_SAMPLE_COUNT = 15  # Minimum number of samples required for cross-correlation
HEATMAP_VMIN = 0       # Minimum value for heatmap
HEATMAP_VMAX = 100     # Maximum value for heatmap


def extract_correlation_values(corr_matrix):
    """Extract numeric correlation values from formatted strings."""
    return corr_matrix.applymap(lambda x: float(x.split(" (")[0])
                               if isinstance(x, str) and " (" in x else np.nan)


def load_and_process_perf_data(filename, selected_benches):
    """
    Load and process performance data from pickle file.
    
    Returns a pivot table with models as rows and benchmarks as columns.
    """
    with open(f"{DATA_DIR}/{filename}", 'rb') as f:
        perf_data = pickle.load(f)

    df = pd.DataFrame(perf_data.items(), columns=['Benchmark_Model', 'Score'])
    df[['Benchmark', 'Model']] = pd.DataFrame(df['Benchmark_Model'].tolist(), index=df.index)
    df.drop(columns=['Benchmark_Model'], inplace=True)
    
    # Filter and sort by selected benchmarks
    df_filtered = df[df['Benchmark'].isin(selected_benches)]
    df_filtered['Benchmark'] = pd.Categorical(df_filtered['Benchmark'],
                                             categories=selected_benches, ordered=True)
    df_filtered = df_filtered.sort_values(by=['Benchmark'])

    return df_filtered.pivot(index='Model', columns='Benchmark', values='Score')


def compute_lins_ccc(x, y):
    """
    Compute Lin's Concordance Correlation Coefficient (CCC).
    
    CCC measures both correlation and agreement between two variables.
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan
        
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pearson_corr, _ = pearsonr(x, y)
    
    return (2 * pearson_corr * np.sqrt(var_x) * np.sqrt(var_y)) / \
        (var_x + var_y + (mean_x - mean_y) ** 2)


def format_correlation_value(corr_val, sample_count):
    """Format correlation value with sample count."""
    if isinstance(corr_val, float):
        if corr_val == 100:
            return f"100 ({sample_count})"
        return f"{corr_val*100:.1f} ({sample_count})"
    else:
        if corr_val[0] == 100:
            return f"100 ({sample_count})"
        return f"{corr_val[0]*100:.1f} ({sample_count})"


def compute_correlation_for_method(data_i, data_j, method):
    """Compute correlation between two data series using specified method."""
    if method == 'lin_ccc':
        return compute_lins_ccc(data_i, data_j)
    elif method == 'pearson':
        return pearsonr(data_i, data_j)[0]
    elif method == 'spearman':
        return spearmanr(data_i, data_j)[0]
    elif method == 'kendall':
        return kendalltau(data_i, data_j)[0]
    return np.nan


def compute_correlation_matrices_with_count(df):
    """
    Compute multiple correlation matrices with sample counts.
    
    Returns a dictionary with correlation matrices for each method.
    """
    benchmarks = df.columns
    correlation_results = {}
    
    # Initialize a separate matrix for each correlation method
    for method in CORR_METHODS:
        correlation_results[f"{method.capitalize()} Correlation"] = pd.DataFrame(
            index=benchmarks, columns=benchmarks)
    
    for bench_i in benchmarks:
        for bench_j in benchmarks:
            valid_data = df[[bench_i, bench_j]].dropna()
            sample_count = len(valid_data)
            
            if not valid_data.empty:
                for method in CORR_METHODS:
                    corr_val = compute_correlation_for_method(
                        valid_data[bench_i], valid_data[bench_j], method)
                    
                    if not pd.isna(corr_val):
                        formatted_val = format_correlation_value(corr_val, sample_count)
                        correlation_results[f"{method.capitalize()} Correlation"].loc[
                            bench_i, bench_j] = formatted_val
                    else:
                        correlation_results[f"{method.capitalize()} Correlation"].loc[
                            bench_i, bench_j] = np.nan
            else:
                for method in CORR_METHODS:
                    correlation_results[f"{method.capitalize()} Correlation"].loc[
                        bench_i, bench_j] = np.nan

    return correlation_results


def compute_cross_correlation_matrices(general_df, medqa_df):
    """
    Compute correlation matrices between general and medical benchmarks.
    
    Returns a dictionary with cross-correlation matrices for each method.
    """
    general_benchmarks = general_df.columns
    medical_benchmarks = medqa_df.columns

    # Initialize correlation matrices for each method
    correlation_results = {}
    for method in CORR_METHODS:
        correlation_results[f"{method.capitalize()} Correlation"] = pd.DataFrame(
            index=general_benchmarks,
            columns=medical_benchmarks
        )

    # Get common models between the two dataframes
    common_models = list(set(general_df.index) & set(medqa_df.index))
    
    # Compute correlations for each pair of benchmarks
    for gen_bench in general_benchmarks:
        for med_bench in medical_benchmarks:
            # Get data for common models
            data = pd.DataFrame({
                'general': general_df.loc[common_models, gen_bench],
                'medical': medqa_df.loc[common_models, med_bench]
            }).dropna()

            sample_count = len(data)
            
            if not data.empty:
                for method in CORR_METHODS:
                    corr_val = compute_correlation_for_method(
                        data['general'], data['medical'], method)
                    
                    if not pd.isna(corr_val):
                        formatted_val = format_correlation_value(corr_val, sample_count)
                        correlation_results[f"{method.capitalize()} Correlation"].loc[
                            gen_bench, med_bench] = formatted_val
                    else:
                        correlation_results[f"{method.capitalize()} Correlation"].loc[
                            gen_bench, med_bench] = np.nan
            else:
                for method in CORR_METHODS:
                    correlation_results[f"{method.capitalize()} Correlation"].loc[
                        gen_bench, med_bench] = np.nan

    return correlation_results


def extract_sample_counts(corr_matrix):
    """Extract sample counts from formatted correlation values."""
    result = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            value = corr_matrix.loc[i, j]
            if isinstance(value, str) and "(" in value and ")" in value:
                count_str = value.split("(")[1].split(")")[0]
                result.loc[i, j] = int(count_str)
            else:
                result.loc[i, j] = 0
    
    return result


def format_correlation_with_samples(corr_matrix, with_linebreak=False):
    """Format correlation values with sample counts, optionally with linebreaks."""
    result = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            value = corr_matrix.loc[i, j]
            if isinstance(value, str) and " (" in value:
                if with_linebreak:
                    result.loc[i, j] = value.replace(" (", "\n(")
                else:
                    result.loc[i, j] = value
            else:
                result.loc[i, j] = value
    
    return result


def filter_cross_correlation(corr_matrix):
    """Filter correlation matrix to include only rows with sufficient samples."""
    sample_counts = extract_sample_counts(corr_matrix)
    valid_rows = [i for i in sample_counts.index 
                 if all(count >= MIN_SAMPLE_COUNT for count in sample_counts.loc[i])]
    
    # Filter the correlation matrix
    return corr_matrix.loc[valid_rows] if valid_rows else pd.DataFrame()


def plot_individual_matrix(corr_matrix, title, with_sample_number=True, is_cross_correlation=False):
    """Plot an individual correlation matrix."""
    # Extract numeric values for the heatmap
    corr_matrix_numeric = extract_correlation_values(corr_matrix)

    # For non-cross-correlation matrices, create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix_numeric), k=1) if not is_cross_correlation else None

    # Prepare annotation data
    if with_sample_number:
        # Include sample counts with line break for better readability
        annot_data = format_correlation_with_samples(corr_matrix, with_linebreak=True)
        fmt_option = ""  # Use string formatting
    else:
        # Show only correlation values
        annot_data = corr_matrix_numeric
        fmt_option = ".2f"  # Standard numerical formatting

    # Adjust figure size based on matrix dimensions
    width = corr_matrix.shape[1] * 0.8 * 1.3
    height = corr_matrix.shape[0] * 0.8
    if is_cross_correlation:
        height = height * 1.35
    plt.figure(figsize=(width, height))

    # Create heatmap
    sns.heatmap(corr_matrix_numeric,
              annot=annot_data,
              fmt=fmt_option,
              cmap="coolwarm",
              center=0,
              vmin=HEATMAP_VMIN,
              vmax=HEATMAP_VMAX,
              mask=mask)

    # Rotate x-axis labels if they're too long
    if max(len(str(label)) for label in corr_matrix.columns) > 10:
        plt.xticks(rotation=45, ha='right')

    # Rotate y-axis labels if they're too long
    if max(len(str(label)) for label in corr_matrix.index) > 10:
        plt.yticks(rotation=0, ha='right')

    # Add title
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot if SAVE_PLOTS is True
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_filename = title.lower().replace(' ', '_')
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.png', dpi=600)
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.tiff', dpi=600)
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.pdf', dpi=600, format='pdf', 
                   metadata={'Creator': 'Matplotlib', 'Producer': 'Matplotlib'})
        print(f"Plot saved as {OUTPUT_DIR}/{base_filename}.png/tiff/pdf")
        
    plt.show()


def plot_panel_matrix(corr_matrices_with_counts, with_sample_number=True, is_cross_correlation=False):
    """Plot correlation matrices as a 2x2 panel."""
    # Use available correlation methods
    corr_methods = list(CORR_METHOD_LABELS.keys())
    available_methods = [method for method in corr_methods if method in corr_matrices_with_counts]
    
    if not available_methods:
        print("No correlation matrices to plot")
        return
        
    # Take the first matrix to get dimensions
    first_matrix = corr_matrices_with_counts[available_methods[0]]
    
    # Filter matrices if needed for cross-correlation
    if is_cross_correlation:
        filtered_matrices = {}
        for method in available_methods:
            filtered_matrix = filter_cross_correlation(corr_matrices_with_counts[method])
            if not filtered_matrix.empty:
                filtered_matrices[method] = filtered_matrix
        
        if not filtered_matrices:
            print("No matrices with sufficient samples to plot")
            return
        
        corr_matrices_with_counts = filtered_matrices
        available_methods = list(filtered_matrices.keys())
        first_matrix = corr_matrices_with_counts[available_methods[0]]
    
    # Set up the panel figure
    plt.rcParams.update({'font.size': 14})
    
    # Configure dimensions for the panel plot
    width_per_subplot = first_matrix.shape[1] * 0.5 * 1.8
    height_per_subplot = first_matrix.shape[0] * 0.5 * 1.8
    if is_cross_correlation:
        width_per_subplot = width_per_subplot * 1.1
        height_per_subplot = height_per_subplot * 1.2
    
    # Create figure with 2x2 grid layout
    figsize = (width_per_subplot * 2 * 1.2, height_per_subplot * 2 * 1.2)
    fig = plt.figure(figsize=figsize)
    
    # Calculate domain-specific spacing in inches
    mm_to_inches = 1/25.4
    
    # Determine the benchmark type and set spacing accordingly
    if is_cross_correlation:
        wspace_inches = 55 * mm_to_inches  # Cross domain spacing
    elif any('Medical' in str(idx) for idx in first_matrix.index):
        wspace_inches = 80 * mm_to_inches  # Medical domain spacing
    else:
        wspace_inches = 45 * mm_to_inches  # General domain spacing
    
    # Convert to fraction of figure width
    wspace_frac = wspace_inches / fig.get_figwidth()
    
    # Create gridspec with domain-specific spacing
    if is_cross_correlation:
        hspace = 0.53
    elif any('Medical' in str(idx) for idx in first_matrix.index):
        hspace = 0.52
    else:
        hspace = 0.5
    gs = fig.add_gridspec(2, 2, wspace=wspace_frac*3, hspace=hspace)
    
    # Common mask for all non-cross-correlation matrices
    common_mask = None
    if not is_cross_correlation and len(first_matrix) > 0:
        first_numeric = extract_correlation_values(first_matrix)
        common_mask = np.triu(np.ones_like(first_numeric), k=1)
    
    # Create custom colormap from white to crimson red
    colors = [(1, 1, 1), (0.996, 0.9, 0.85), (0.99, 0.8, 0.7),
             (0.95, 0.6, 0.5), (0.85, 0.3, 0.3), (0.75, 0.1, 0.1)]
    positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    custom_cmap = LinearSegmentedColormap.from_list('custom_crimson',
                                                    list(zip(positions, colors)))
    
    # Set up normalization for the colormap
    cmap = custom_cmap
    norm = plt.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    
    # Plot each method as a panel
    for i, method in enumerate(available_methods[:4]):
        # Calculate grid position (row, col)
        row = i // 2
        col = i % 2
        
        # Get the correlation matrix
        corr_matrix = corr_matrices_with_counts[method]
        corr_matrix_numeric = extract_correlation_values(corr_matrix)
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Set mask for non-cross-correlation matrices
        mask = common_mask if not is_cross_correlation else None
        
        # Prepare annotation data
        if with_sample_number:
            annot_data = format_correlation_with_samples(corr_matrix, with_linebreak=True)
            fmt_option = ""
        else:
            annot_data = corr_matrix_numeric
            fmt_option = ".2f"
        
        # Create heatmap for this panel
        sns.heatmap(corr_matrix_numeric,
                  annot=annot_data,
                  fmt=fmt_option,
                  cmap=cmap,
                  vmin=HEATMAP_VMIN,
                  vmax=HEATMAP_VMAX,
                  mask=mask,
                  cbar=False,
                  ax=ax)
        
        # Format axis labels
        if max(len(str(label)) for label in corr_matrix.columns) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        
        if max(len(str(label)) for label in corr_matrix.index) > 10:
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
        
        # Add panel label with method name
        panel_label = chr(65 + i)  # ASCII: A=65, B=66, etc.
        method_full_name = CORR_METHOD_LABELS[method]
        ax.text(-0.1, 1.05, f"({panel_label}) {method_full_name}", transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    # Add a single colorbar on the right
    cbar_ax = fig.add_axes([0.95, 0.225, 0.015, 0.55])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.ax.tick_params(labelsize=20)
    
    # Add labels
    plt.figtext(0.5, 0.02, 'Benchmarks', ha='center', fontsize=14)
    plt.figtext(0.02, 0.5, 'Benchmarks', va='center', rotation='vertical', fontsize=14)
    
    # Adjust layout
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    
    # Save the panel plot if SAVE_PLOTS is True
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if is_cross_correlation:
            base_filename = 'cross_correlation_panel'
        else:
            benchmark_type = 'medical' if any('Medical' in str(idx) for idx in first_matrix.index) \
                else 'general'
            base_filename = f'{benchmark_type}_correlation_panel'
        
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.tiff', dpi=600, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/{base_filename}.pdf', dpi=600, format='pdf', bbox_inches='tight',
                   metadata={'Creator': 'Matplotlib', 'Producer': 'Matplotlib'})
        print(f"Panel plot saved as {OUTPUT_DIR}/{base_filename}.png/tiff/pdf")
        
    plt.show()


def plot_correlation_matrices(corr_matrices_with_counts, with_sample_number=True, 
                             is_cross_correlation=False):
    """
    Plot correlation matrices - either as individual plots or as a panel.
    
    Args:
        corr_matrices_with_counts: Dictionary of correlation matrices
        with_sample_number: Boolean to control display of sample counts
        is_cross_correlation: Boolean to indicate if this is a cross-correlation matrix
    """
    if PLOT_AS_PANEL:
        plot_panel_matrix(corr_matrices_with_counts, with_sample_number, is_cross_correlation)
    else:
        for title, corr_matrix in corr_matrices_with_counts.items():
            if is_cross_correlation:
                filtered_matrix = filter_cross_correlation(corr_matrix)
                if filtered_matrix.empty:
                    print(f"Warning: No valid rows with sufficient samples for {title}")
                    continue
                corr_matrix = filtered_matrix
                
            plot_individual_matrix(corr_matrix, title, with_sample_number, is_cross_correlation)


def combine_correlations_to_json(general_corrs, medqa_corrs, cross_corrs):
    """
    Combine all correlation matrices into a single structure for JSON export.
    
    Returns a dictionary with structured correlation data.
    """
    # Initialize the combined structure
    combined_corrs = {
        "benchmark_categories": {
            "general": GENERAL_BENCHES,
            "medical": MED_BENCHES
        },
        "correlations": {}
    }
    
    # Add methods as keys in the correlations section
    for method in CORR_METHODS:
        method_name = f"{method.capitalize()} Correlation"
        combined_corrs["correlations"][method] = {
            "general": {},  # General benchmark correlations
            "medical": {},  # Medical benchmark correlations
            "cross": {}     # Cross-correlations between general and medical
        }
        
        # Process general benchmark correlations
        if method_name in general_corrs:
            gen_matrix = extract_correlation_values(general_corrs[method_name])
            for bench_i in gen_matrix.index:
                combined_corrs["correlations"][method]["general"][bench_i] = {}
                for bench_j in gen_matrix.columns:
                    value = gen_matrix.loc[bench_i, bench_j]
                    if not pd.isna(value):
                        combined_corrs["correlations"][method]["general"][bench_i][bench_j] = value / 100
        
        # Process medical benchmark correlations
        if method_name in medqa_corrs:
            med_matrix = extract_correlation_values(medqa_corrs[method_name])
            for bench_i in med_matrix.index:
                combined_corrs["correlations"][method]["medical"][bench_i] = {}
                for bench_j in med_matrix.columns:
                    value = med_matrix.loc[bench_i, bench_j]
                    if not pd.isna(value):
                        combined_corrs["correlations"][method]["medical"][bench_i][bench_j] = value / 100
        
        # Process cross-correlations
        if method_name in cross_corrs:
            cross_matrix = extract_correlation_values(cross_corrs[method_name])
            
            # Original direction (general -> medical)
            for bench_i in cross_matrix.index:  # bench_i is general benchmark
                if bench_i not in combined_corrs["correlations"][method]["cross"]:
                    combined_corrs["correlations"][method]["cross"][bench_i] = {}
                
                for bench_j in cross_matrix.columns:  # bench_j is medical benchmark
                    value = cross_matrix.loc[bench_i, bench_j]
                    if not pd.isna(value):
                        combined_corrs["correlations"][method]["cross"][bench_i][bench_j] = value / 100
            
            # Reverse direction (medical -> general)
            for bench_j in cross_matrix.columns:  # bench_j is medical benchmark
                if bench_j not in combined_corrs["correlations"][method]["cross"]:
                    combined_corrs["correlations"][method]["cross"][bench_j] = {}
                
                for bench_i in cross_matrix.index:  # bench_i is general benchmark
                    value = cross_matrix.loc[bench_i, bench_j]
                    if not pd.isna(value):
                        combined_corrs["correlations"][method]["cross"][bench_j][bench_i] = value / 100
    
    return combined_corrs


def save_correlations_to_json(combined_corrs, filename):
    """Save the combined correlation data to a JSON file."""
    # Make sure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(combined_corrs, f, indent=2)
    
    print(f"Correlation data successfully saved to {filename}")


def print_dataset_summary(df, name):
    """Print summary statistics for a dataset."""
    non_nan_counts = sum(list(df.notna().sum()))
    print(f'{name} Benchmarks:')
    print(f"  Total models: {df.shape[0]}")
    print(f"  Total benchmarks: {df.shape[1]}")
    print(f"  Total data points: {non_nan_counts}")


def save_processed_datasets(general_df, medqa_df):
    """Save processed datasets to Excel and pickle files."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    date_suffix = '250228'  # Format: YYMMDD
    
    # Save Excel files
    general_df.to_excel(f'{DATASET_DIR}/general_df_fin_{date_suffix}.xlsx')
    medqa_df.to_excel(f'{DATASET_DIR}/medqa_df_fin_{date_suffix}.xlsx')
    
    # Save pickle files
    general_df.to_pickle(f'{DATASET_DIR}/general_df_fin_{date_suffix}.pkl')
    medqa_df.to_pickle(f'{DATASET_DIR}/medqa_df_fin_{date_suffix}.pkl')
    
    print(f"Datasets saved to {DATASET_DIR} directory")


def main():
    """Main function to run the correlation analysis pipeline."""
    # Load and process data
    general_df = load_and_process_perf_data(GENERAL_PERF_FILE, GENERAL_BENCHES)
    print_dataset_summary(general_df, "General")
    
    medqa_df = load_and_process_perf_data(MEDICAL_PERF_FILE, MED_BENCHES)
    print_dataset_summary(medqa_df, "Medical")
    
    # Save processed datasets
    save_processed_datasets(general_df, medqa_df)
    
    # Compute correlation matrices
    general_corrs = compute_correlation_matrices_with_count(general_df)
    medqa_corrs = compute_correlation_matrices_with_count(medqa_df)
    cross_corrs = compute_cross_correlation_matrices(general_df, medqa_df)

    # Generate plots
    plot_correlation_matrices(general_corrs)
    plot_correlation_matrices(medqa_corrs)
    plot_correlation_matrices(cross_corrs, with_sample_number=True, is_cross_correlation=True)
    
    # Combine and save correlation data to JSON
    combined_corrs = combine_correlations_to_json(general_corrs, medqa_corrs, cross_corrs)
    date_suffix = '250228'  # Format: YYMMDD
    save_correlations_to_json(combined_corrs, f'{DATASET_DIR}/correlations_fin_{date_suffix}.json')
    print("All correlation data has been processed and saved.")


if __name__ == '__main__':
    main()