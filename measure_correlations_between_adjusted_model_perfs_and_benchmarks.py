import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# Global constants
DEFAULT_METRIC = 'original_scale'
CORRELATION_TYPES = ['pearson', 'spearman', 'kendall', 'lin_ccc']
DATA_DIR = 'imputed_benchmarks'
ANALYSIS_DIR = 'bayesian_analysis_results'
BENCHMARK_PICKLE_PATH = f'{DATA_DIR}/all_imputation_results.pkl'
MODEL_PERFORMANCE_PATH = f'{ANALYSIS_DIR}/model_performance.pkl'
OUTPUT_PICKLE_PATH = f'{DATA_DIR}/benchmark_correlation_results.pkl'
DEFAULT_FIGSIZE = (14, 7)
BAR_COLORS = {'general': 'blue', 'medical': 'green'}
MIN_DATA_POINTS = 3  # Minimum number of points required for correlation calculation


def lin_ccc(x, y):
    """
    Compute Lin's Concordance Correlation Coefficient.
    
    Args:
        x: First array of measurements
        y: Second array of measurements
        
    Returns:
        Lin's concordance correlation coefficient
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    
    covar = np.cov(x, y)[0, 1]
    
    numerator = 2 * covar
    denominator = var_x + var_y + (mean_x - mean_y)**2
    
    return numerator / denominator


def calculate_correlations(df_model_perfs, benchmark_data, metric=DEFAULT_METRIC):
    """
    Calculate correlations between model effects and benchmark performances.
    
    Args:
        df_model_perfs: DataFrame containing model performance data
        benchmark_data: DataFrame with benchmark results indexed by model names
        metric: The metric column to use from df_model_perfs
        
    Returns:
        Dictionary with benchmark names as keys and correlation measures as values
    """
    model_perfs_models = set(df_model_perfs['model'])
    benchmark_models = set(benchmark_data.index)
    common_models = model_perfs_models.intersection(benchmark_models)
    
    print(f"Common models: {len(common_models)}")
    
    if not common_models:
        return {}
    
    model_perf_dict = {row['model']: row[metric] for _, row in df_model_perfs.iterrows() 
                       if row['model'] in common_models}
    
    results = {}
    
    for benchmark in benchmark_data.columns:
        filtered_benchmark = benchmark_data.loc[list(common_models), benchmark].values
        
        filtered_model_perfs = np.array([model_perf_dict[model] for model in common_models])
        
        valid_indices = ~np.isnan(filtered_benchmark)
        x = filtered_model_perfs[valid_indices]
        y = filtered_benchmark[valid_indices]
        
        if len(x) < MIN_DATA_POINTS:
            continue
            
        results[benchmark] = {
            'pearson': pearsonr(x, y)[0],
            'spearman': spearmanr(x, y)[0],
            'kendall': kendalltau(x, y)[0],
            'lin_ccc': lin_ccc(x, y),
            'n': len(x)
        }
    
    return results


def plot_correlations(general_correlations, medical_correlations, 
                      correlation_type='spearman', figsize=DEFAULT_FIGSIZE):
    """
    Plot correlation results as bar graphs with sorted values.
    
    Args:
        general_correlations: Dictionary of general benchmark correlations
        medical_correlations: Dictionary of medical benchmark correlations
        correlation_type: Type of correlation to plot (pearson, spearman, kendall, lin_ccc)
        figsize: Figure size as a tuple (width, height)
        
    Returns:
        Matplotlib figure object
    """
    plt.figure(figsize=figsize)
    
    benchmarks, correlations, colors = [], [], []
    
    # Prepare data for plotting
    for benchmark_type, corr_dict, color in [
        ('general', general_correlations, BAR_COLORS['general']),
        ('medical', medical_correlations, BAR_COLORS['medical'])
    ]:
        for benchmark, values in corr_dict.items():
            if correlation_type in values and not np.isnan(values[correlation_type]):
                benchmarks.append(benchmark)
                correlations.append(values[correlation_type])
                colors.append(color)
    
    if not benchmarks:
        print(f"No valid data for {correlation_type} correlation plot.")
        return plt
    
    # Sort all data by correlation value (descending)
    sorted_indices = np.argsort(correlations)[::-1]
    benchmarks = [benchmarks[i] for i in sorted_indices]
    correlations = [correlations[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    # Create bar chart
    bars = plt.bar(benchmarks, correlations, color=colors)
    
    # Set plot formatting
    max_corr = max(correlations) if correlations else 1.0
    plt.ylim(0, max_corr * 1.1)  # Add 10% padding to y-axis max
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add correlation values on top of each bar
    label_padding = 0.02 * max_corr  # Padding between bar and text label
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        plt.text(i, corr + label_padding, f"{corr:.3f}", 
                 ha='center', va='bottom', fontsize=9)
    
    # Set title and labels
    plt.title(f"{correlation_type.capitalize()} Correlation by Benchmark")
    plt.xlim(-1, len(bars))
    plt.ylabel('Correlation')
    plt.xlabel('Benchmark')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt


def save_correlations_to_pickle(general_correlations, medical_correlations, filename=OUTPUT_PICKLE_PATH):
    """
    Save correlation results to a pickle file as a nested dictionary.
    
    Args:
        general_correlations: Dictionary of general benchmark correlations
        medical_correlations: Dictionary of medical benchmark correlations
        filename: Path to output pickle file
        
    Returns:
        Dictionary with organized correlation results
    """
    results_dict = {corr_type: {} for corr_type in CORRELATION_TYPES}
    
    # Process and organize correlation results
    for benchmark_type, correlations, prefix in [
        ('general', general_correlations, 'general_'),
        ('medical', medical_correlations, 'medical_')
    ]:
        for benchmark, values in correlations.items():
            for corr_type in CORRELATION_TYPES:
                if corr_type in values and not np.isnan(values[corr_type]):
                    results_dict[corr_type][f"{prefix}{benchmark}"] = values[corr_type]
    
    # Save to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Correlation results saved to {filename}")
    
    return results_dict


def load_data():
    """
    Load model performance and benchmark data from pickle files.
    
    Returns:
        Tuple containing (model_performance_df, general_benchmarks_df, medical_benchmarks_df, metric)
    """
    # Load model performance data
    with open(MODEL_PERFORMANCE_PATH, 'rb') as f:
        df_model_perfs = pickle.load(f)
    
    # Load benchmark data
    with open(BENCHMARK_PICKLE_PATH, 'rb') as f:
        all_imputed_benchmarks = pickle.load(f)
    
    df_general = all_imputed_benchmarks['General_Benchmarks_RF']['multi_imputed']
    df_medical = all_imputed_benchmarks['Medical_Benchmarks_RF']['multi_imputed']

    # Determine appropriate metric column
    metric = DEFAULT_METRIC
    if metric not in df_model_perfs.columns:
        numeric_cols = [col for col in df_model_perfs.columns 
                      if col != 'model' and 
                      np.issubdtype(df_model_perfs[col].dtype, np.number)]
        if numeric_cols:
            metric = numeric_cols[0]
        else:
            raise ValueError("No suitable numeric columns available for correlation analysis")
    
    return df_model_perfs, df_general, df_medical, metric


def save_plots(general_correlations, medical_correlations):
    """
    Generate and save correlation plots for all correlation types.
    
    Args:
        general_correlations: Dictionary of general benchmark correlations
        medical_correlations: Dictionary of medical benchmark correlations
    """
    for corr_type in CORRELATION_TYPES:
        plot = plot_correlations(
            general_correlations, medical_correlations, 
            correlation_type=corr_type, figsize=DEFAULT_FIGSIZE
        )
        
        # Save in multiple formats
        for ext in ['png', 'tiff', 'pdf']:
            filename = f"{corr_type}_correlation_by_benchmark.{ext}"
            dpi = 600
            if ext == 'pdf':
                plot.savefig(filename, dpi=dpi, format=ext)
            else:
                plot.savefig(filename, dpi=dpi)
        
        plt.show()
        plt.close()


def main():
    """
    Main function to load data, calculate correlations, and generate plots.
    """
    try:
        # Load data
        df_model_perfs, df_general, df_medical, metric = load_data()
        
        # Calculate correlations
        general_correlations = calculate_correlations(df_model_perfs, df_general, metric)
        medical_correlations = calculate_correlations(df_model_perfs, df_medical, metric)
        
        # Save correlation results
        save_correlations_to_pickle(general_correlations, medical_correlations)
        
        # Generate and save plots
        save_plots(general_correlations, medical_correlations)
        
        print("Analysis complete. All correlation plots have been saved.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()