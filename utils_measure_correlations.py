import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Global constants
DIR_CLINICAL_PERF = 'clinical_performance'
DIR_FINAL_DATASET = 'final_dataset'
DIR_IMPUTED_BENCHMARKS = 'imputed_benchmarks'
DIR_FIGURES = 'figures'

FNAME_GENERAL_DF = 'general_df_fin_250228.pkl'
FNAME_MEDQA_DF = 'medqa_df_fin_250228.pkl'
FNAME_GENERAL_DF_IMPUTED = 'original_general_benchmarks_imputed_rf.pkl'
FNAME_MEDQA_DF_IMPUTED = 'original_medical_benchmarks_imputed_rf.pkl'
FNAME_CLINICAL_PERF_DF = 'results_fin_250210_추가작업(2.14).xlsx'

# Analysis constants
MIN_MODELS_FOR_CORRELATION = 3
MAX_SAMPLE_SIZE = 1000  # Sample size truncation threshold (95% of data is below this)
BENCHMARK_COVERAGE_THRESHOLD = 0.5  # Default threshold for sufficient benchmark coverage

def load_benchmark_correlations(file_path):
    """Load benchmark correlations from JSON file."""
    try:
        with open(file_path, 'r') as f:
            corr_data = json.load(f)
        return corr_data
    except Exception as e:
        print(f"Error loading benchmark correlations: {e}")
        return None

def load_datasets(config, use_imputated=False):
    """Load all necessary benchmark and clinical performance datasets."""
    # Load benchmark data
    general_df = pd.read_pickle(f'{DIR_FINAL_DATASET}/{FNAME_GENERAL_DF}')
    medqa_df = pd.read_pickle(f'{DIR_FINAL_DATASET}/{FNAME_MEDQA_DF}')
    if use_imputated:
        general_df = pd.read_pickle(f'{DIR_IMPUTED_BENCHMARKS}/{FNAME_GENERAL_DF_IMPUTED}')
        medqa_df = pd.read_pickle(f'{DIR_IMPUTED_BENCHMARKS}/{FNAME_MEDQA_DF_IMPUTED}')

    # Load clinical performance data
    clinical_perf_df = pd.read_excel(f'{config["DIR_CLINICAL_PERF"]}/{FNAME_CLINICAL_PERF_DF}')
    clinical_perf_df = clinical_perf_df[['title', 'task_name',
                                       'task_type', 'therapeutic_area',
                                       'data_source', 'evaluation_type',
                                       'model_full_name2',
                                       'metric_value', 'few-shot', 'sample_size']]
    
    # Truncate sample_size to MAX_SAMPLE_SIZE if larger
    clinical_perf_df['sample_size'] = clinical_perf_df['sample_size'].apply(
        lambda x: MAX_SAMPLE_SIZE if x > MAX_SAMPLE_SIZE else x
    )
    
    # Rename model column for consistency
    clinical_perf_df.rename(columns={'model_full_name2': 'model_name'}, inplace=True)
    
    # Extract model sets for later use
    general_models = set(general_df.index)
    medqa_models = set(medqa_df.index)
    clinical_models = set(clinical_perf_df['model_name'])
    
    return {
        'general_df': general_df,
        'medqa_df': medqa_df,
        'clinical_perf_df': clinical_perf_df,
        'general_models': general_models,
        'medqa_models': medqa_models,
        'clinical_models': clinical_models
    }

def analyze_model_overlap(datasets):
    """Analyze and print model overlap statistics between datasets."""
    print("=== Model Overlap Statistics ===")
    
    general_models = datasets['general_models']
    medqa_models = datasets['medqa_models']
    clinical_models = datasets['clinical_models']
    
    # Count models in each dataset
    print(f"Number of models in general benchmark: {len(general_models)}")
    print(f"Number of models in MedQA benchmark: {len(medqa_models)}")
    print(f"Number of models in clinical performance: {len(clinical_models)}")
    
    # Calculate overlaps
    general_medqa_overlap = general_models.intersection(medqa_models)
    general_clinical_overlap = general_models.intersection(clinical_models)
    medqa_clinical_overlap = medqa_models.intersection(clinical_models)
    all_overlap = general_models.intersection(medqa_models, clinical_models)
    
    # Models unique to each dataset
    general_only = general_models - medqa_models - clinical_models
    medqa_only = medqa_models - general_models - clinical_models
    clinical_only = clinical_models - general_models - medqa_models
    
    # Print overlaps
    print(f"Models in both general and MedQA benchmarks: {len(general_medqa_overlap)}")
    print(f"Models in both general benchmark and clinical performance: {len(general_clinical_overlap)}")
    print(f"Models in both MedQA benchmark and clinical performance: {len(medqa_clinical_overlap)}")
    print(f"Models in all three datasets: {len(all_overlap)}")
    print(f"Models only in general benchmark ({len(general_only)}): {', '.join(list(general_only))}")
    print(f"Models only in MedQA benchmark ({len(medqa_only)}): {', '.join(list(medqa_only))}")
    print(f"Models only in clinical performance ({len(clinical_only)}): {', '.join(list(clinical_only))}")

def calculate_benchmark_coverage(df, clinical_models, benchmark_type):
    """Calculate coverage ratio for benchmarks relative to clinical models."""
    benchmark_coverage = {}
    
    for benchmark in df.columns:
        # Count models in clinical_models that have this benchmark score
        valid_models = [
            model for model in clinical_models 
            if model in df.index and pd.notna(df.loc[model, benchmark])
        ]
        coverage = len(valid_models) / len(clinical_models)
        benchmark_coverage[benchmark] = (coverage, len(valid_models), len(clinical_models))
    
    return benchmark_coverage

def print_benchmark_coverage_stats(general_coverage, medqa_coverage, threshold):
    """Print benchmark coverage statistics and identify included benchmarks."""
    print("\n=== Benchmark Coverage Statistics ===")
    
    print("\nGeneral Benchmarks Coverage:")
    for benchmark, (coverage, valid_count, total) in sorted(
        general_coverage.items(), key=lambda x: x[1][0], reverse=True
    ):
        status = "INCLUDED" if coverage >= threshold else "excluded"
        print(f"  {benchmark}: {valid_count}/{total} models ({coverage*100:.1f}%) - {status}")
    
    print("\nMedQA Benchmarks Coverage:")
    for benchmark, (coverage, valid_count, total) in sorted(
        medqa_coverage.items(), key=lambda x: x[1][0], reverse=True
    ):
        status = "INCLUDED" if coverage >= threshold else "excluded"
        print(f"  {benchmark}: {valid_count}/{total} models ({coverage*100:.1f}%) - {status}")
    
    # Count selected benchmarks
    selected_general = sum(1 for _, (cov, _, _) in general_coverage.items() if cov >= threshold)
    selected_medqa = sum(1 for _, (cov, _, _) in medqa_coverage.items() if cov >= threshold)
    print(f"\nSelected {selected_general} general benchmarks and {selected_medqa} MedQA benchmarks")

def filter_clinical_tasks(clinical_df, general_df, medqa_df, 
                         valid_general_benchmarks, valid_medqa_benchmarks,
                         datasets, config):
    """Filter clinical tasks based on model coverage threshold."""
    clinical_tasks = clinical_df['task_name'].unique()
    valid_clinical_tasks = []
    
    # For strict filtering, track models with data for ALL selected benchmarks
    task_consistent_models = {}
    
    general_models = datasets['general_models']
    medqa_models = datasets['medqa_models']
    
    for task in clinical_tasks:
        # Get models for this clinical task
        task_df = clinical_df[clinical_df['task_name'] == task]
        task_models = set(task_df['model_name'])
        
        if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
            # Find models that have data for ALL valid benchmarks
            models_with_all_general = set()
            models_with_all_medqa = set()
            
            # For general benchmarks
            if valid_general_benchmarks:
                models_with_all_general = task_models.intersection(general_models)
                for benchmark in valid_general_benchmarks:
                    valid_for_this_benchmark = [
                        model for model in models_with_all_general 
                        if pd.notna(general_df.loc[model, benchmark])
                    ]
                    models_with_all_general = set(valid_for_this_benchmark)
                    if len(models_with_all_general) < MIN_MODELS_FOR_CORRELATION:
                        break
            
            # For MedQA benchmarks
            if valid_medqa_benchmarks:
                models_with_all_medqa = task_models.intersection(medqa_models)
                for benchmark in valid_medqa_benchmarks:
                    valid_for_this_benchmark = [
                        model for model in models_with_all_medqa 
                        if pd.notna(medqa_df.loc[model, benchmark])
                    ]
                    models_with_all_medqa = set(valid_for_this_benchmark)
                    if len(models_with_all_medqa) < MIN_MODELS_FOR_CORRELATION:
                        break
            
            has_valid_general = len(models_with_all_general) >= MIN_MODELS_FOR_CORRELATION
            has_valid_medqa = len(models_with_all_medqa) >= MIN_MODELS_FOR_CORRELATION
            
            # Store the models with complete data for this task
            if has_valid_general or has_valid_medqa:
                task_consistent_models[task] = {
                    'general': list(models_with_all_general),
                    'medqa': list(models_with_all_medqa)
                }
        
        else:
            # Standard filtering: check if we have enough overlapping models with ANY valid benchmark
            has_valid_general = False
            has_valid_medqa = False
            
            for benchmark in valid_general_benchmarks:
                valid_models = [
                    model for model in task_models 
                    if model in general_df.index and pd.notna(general_df.loc[model, benchmark])
                ]
                if len(valid_models) >= MIN_MODELS_FOR_CORRELATION:
                    has_valid_general = True
                    break
                    
            for benchmark in valid_medqa_benchmarks:
                valid_models = [
                    model for model in task_models 
                    if model in medqa_df.index and pd.notna(medqa_df.loc[model, benchmark])
                ]
                if len(valid_models) >= MIN_MODELS_FOR_CORRELATION:
                    has_valid_medqa = True
                    break
        
        if has_valid_general or has_valid_medqa:
            valid_clinical_tasks.append(task)
    
    print(f"\nSelected {len(valid_clinical_tasks)} out of {len(clinical_tasks)} clinical tasks with sufficient model coverage")
    
    # For strict filtering, print additional consistency information
    if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False) and task_consistent_models:
        print("\n=== Strict Filtering Model Consistency Report ===")
        for task in valid_clinical_tasks:
            if task in task_consistent_models:
                general_count = len(task_consistent_models[task]['general'])
                medqa_count = len(task_consistent_models[task]['medqa'])
                print(f"Task '{task}': {general_count} consistent models for general benchmarks, "
                      f"{medqa_count} consistent models for MedQA benchmarks")
    
    return valid_clinical_tasks, task_consistent_models

def apply_benchmark_filtering(datasets, config):
    """Apply filtering to benchmarks and clinical tasks based on configuration."""
    general_df = datasets['general_df']
    medqa_df = datasets['medqa_df']
    clinical_perf_df = datasets['clinical_perf_df']
    clinical_models = datasets['clinical_models']
    
    # Default values
    valid_general_benchmarks = general_df.columns.tolist()
    valid_medqa_benchmarks = medqa_df.columns.tolist()
    clinical_tasks = clinical_perf_df['task_name'].unique()
    task_consistent_models = {}
    
    # Define helper functions
    def is_valid_benchmark(benchmark_name, benchmark_type):
        return True
    
    def get_consistent_models(task, benchmark_type):
        return []
    
    # Apply filtering if configured
    if config.get('FOR_VALID_TASKS_AND_PERFS', False) or config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
        filter_type = "STRICT" if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False) else "STANDARD"
        coverage_threshold = config.get('THRESHOLD_BENCHMARK_COVERAGE', BENCHMARK_COVERAGE_THRESHOLD)
        
        print(f"\n=== Applying {filter_type} Filtering for Valid Benchmarks and Clinical Tasks ===")
        print(f"Benchmark coverage threshold: {coverage_threshold*100}%")
        
        # Calculate coverage for general benchmarks
        general_benchmark_coverage = calculate_benchmark_coverage(
            general_df, clinical_models, "general")
        
        # Calculate coverage for medqa benchmarks
        medqa_benchmark_coverage = calculate_benchmark_coverage(
            medqa_df, clinical_models, "medqa")
        
        # Select benchmarks with coverage >= threshold
        valid_general_benchmarks = [
            b for b, (cov, _, _) in general_benchmark_coverage.items() 
            if cov >= coverage_threshold
        ]
        valid_medqa_benchmarks = [
            b for b, (cov, _, _) in medqa_benchmark_coverage.items() 
            if cov >= coverage_threshold
        ]
        
        # Print benchmark coverage statistics
        print_benchmark_coverage_stats(
            general_benchmark_coverage, medqa_benchmark_coverage, coverage_threshold
        )
        
        # Filter clinical tasks
        valid_clinical_tasks, task_consistent_models = filter_clinical_tasks(
            clinical_perf_df, 
            general_df,
            medqa_df,
            valid_general_benchmarks,
            valid_medqa_benchmarks,
            datasets,
            config
        )
        
        # Update clinical tasks
        clinical_tasks = valid_clinical_tasks
        
        # Update benchmark validation function
        def is_valid_benchmark(benchmark_name, benchmark_type):
            if benchmark_type == 'general':
                return benchmark_name in valid_general_benchmarks
            elif benchmark_type == 'medqa':
                return benchmark_name in valid_medqa_benchmarks
            return False
        
        # Add function for strict filtering
        if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
            def get_consistent_models(task, benchmark_type):
                if task in task_consistent_models:
                    return task_consistent_models[task][benchmark_type]
                return []
    
    # Update datasets with filtered values
    datasets.update({
        'valid_general_benchmarks': valid_general_benchmarks,
        'valid_medqa_benchmarks': valid_medqa_benchmarks,
        'clinical_tasks': clinical_tasks,
        'task_consistent_models': task_consistent_models,
        'is_valid_benchmark': is_valid_benchmark,
        'get_consistent_models': get_consistent_models
    })
    
    return datasets

def calculate_correlations(x, y):
    """Calculate various correlation metrics between two variables."""
    if len(x) <= 2:  # Need at least 3 points for meaningful correlation
        return None
        
    try:
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        kendall_corr, kendall_p = kendalltau(x, y)
        lin_corr = lin_ccc(x, y)
        
        return {
            'pearson': (pearson_corr, pearson_p),
            'spearman': (spearman_corr, spearman_p),
            'kendall': (kendall_corr, kendall_p),
            'lin_ccc': lin_corr
        }
    except Exception:
        return None

def lin_ccc(x, y):
    """Calculate Lin's Concordance Correlation Coefficient."""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    
    # Covariance between x and y
    covariance = np.mean((x - mean_x) * (y - mean_y))
    
    # Lin's CCC
    numerator = 2 * covariance
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    
    return numerator / denominator

def calculate_benchmark_correlations(task, benchmark_type, benchmark_df, 
                                    clinical_perf_map, valid_models_base,
                                    is_valid_benchmark, get_consistent_models,
                                    config, sample_size=None):
    """Calculate correlations between benchmark scores and clinical performance."""
    correlations = []
    
    for benchmark in benchmark_df.columns:
        # Skip if not in valid benchmarks list
        if not is_valid_benchmark(benchmark, benchmark_type):
            continue
        
        if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
            # Use only models that have data for ALL selected benchmarks for this task
            valid_models = get_consistent_models(task, benchmark_type)
        else:
            # Get models that have both clinical and this benchmark data
            valid_models = [
                model for model in valid_models_base 
                if pd.notna(benchmark_df.loc[model, benchmark])
            ]
        
        if len(valid_models) <= 2:
            continue
            
        # Get benchmark and clinical performances
        benchmark_scores = [benchmark_df.loc[model, benchmark] for model in valid_models]
        clinical_scores = [clinical_perf_map[model] for model in valid_models]
        
        # Calculate correlations
        corr_results = calculate_correlations(benchmark_scores, clinical_scores)
        
        if corr_results:
            result = {
                'task_name': task,
                'benchmark_type': benchmark_type,
                'benchmark_name': benchmark,
                'num_models': len(valid_models),
                'models': ','.join(valid_models),  # Store model names for verification
                'sample_size': sample_size,
            }
            
            # Add correlation values
            for k, v in corr_results.items():
                if isinstance(v, tuple):
                    result[k] = v[0]  # Correlation value
                    result[f'{k}_pval'] = v[1]  # p-value
                else:
                    result[k] = v
                    
            correlations.append(result)
    
    return correlations

def verify_model_consistency(results_df):
    """Verify model consistency across benchmarks for strict filtering mode."""
    print("\n=== Model Consistency Verification ===")
    
    # Group by task name and benchmark type to check consistency
    for task in results_df['task_name'].unique():
        task_df = results_df[results_df['task_name'] == task]
        
        for benchmark_type in task_df['benchmark_type'].unique():
            type_df = task_df[task_df['benchmark_type'] == benchmark_type]
            
            # Skip if only one benchmark
            if len(type_df) <= 1:
                continue
                
            # Check if all model sets are identical
            model_sets = type_df['models'].unique()
            if len(model_sets) == 1:
                # Convert to string if needed before splitting
                model_set_str = str(model_sets[0]) if not isinstance(model_sets[0], str) else model_sets[0]
                models = model_set_str.split(',') if ',' in model_set_str else [model_set_str]
                print(f"Task '{task}', {benchmark_type} benchmarks: CONSISTENT ({len(type_df)} benchmarks, {len(models)} models)")
            else:
                print(f"Task '{task}', {benchmark_type} benchmarks: INCONSISTENT - found {len(model_sets)} different model sets")
                # Count models in each set
                for i, model_set in enumerate(model_sets):
                    # Convert to string if needed before splitting
                    model_set_str = str(model_set) if not isinstance(model_set, str) else model_set
                    models = model_set_str.split(',') if ',' in model_set_str else [model_set_str]
                    print(f"  Set {i+1}: {len(models)} models")

def print_correlation_summary(results_df, weighting_by_sample_size=False):
    """Print summary statistics of correlation results by benchmark type."""
    print("\n=== Correlation Summary by Benchmark Type ===")
    
    for benchmark_type in ['general', 'medqa']:
        type_results = results_df[results_df['benchmark_type'] == benchmark_type]
        
        if len(type_results) == 0:
            print(f"No correlations calculated for {benchmark_type} benchmarks")
            continue
        
        print(f"\n{benchmark_type.upper()} Benchmark Correlations:")
        print(f"  Number of unique tasks: {type_results['task_name'].nunique()}")
        print(f"  Number of unique benchmarks: {type_results['benchmark_name'].nunique()}")
        print(f"  Total correlation pairs: {len(type_results)}")
        
        # Calculate average correlations
        print("  Average correlations (mean ± std):")
        for corr_type in ['pearson', 'spearman', 'kendall', 'lin_ccc']:
            if corr_type in type_results.columns:
                if weighting_by_sample_size and 'sample_size' in type_results.columns:
                    # Weight calculation: wi = log(1 + sample_sizei)
                    weights = np.log1p(type_results['sample_size'].fillna(1))
                    # Weighted mean
                    mean_corr = np.average(type_results[corr_type], weights=weights)
                    # Weighted std dev
                    weighted_variance = np.average((type_results[corr_type] - mean_corr)**2, weights=weights)
                    std_corr = np.sqrt(weighted_variance)
                    weighting_method = "(weighted by log(1+sample_size))"
                else:
                    mean_corr = type_results[corr_type].mean()
                    std_corr = type_results[corr_type].std()
                    weighting_method = ""
                
                print(f"    {corr_type}: {mean_corr:.4f} ± {std_corr:.4f} {weighting_method}")
        
        # Calculate percentage of significant correlations (p < 0.05)
        for corr_type in ['pearson', 'spearman', 'kendall']:
            p_col = f'{corr_type}_pval'
            if p_col in type_results.columns:
                sig_count = sum(type_results[p_col] < 0.05)
                sig_perc = sig_count / len(type_results) * 100
                print(f"    Significant {corr_type} correlations (p<0.05): {sig_count}/{len(type_results)} ({sig_perc:.1f}%)")

def print_analysis_summary(stats, results_df, config, weighting_by_sample_size=False):
    """Print comprehensive analysis summary and statistics."""
    print("\n=== Analysis Summary ===")
    
    # Print analysis mode
    if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
        print("Analysis mode: STRICTLY FILTERED (using only benchmarks with sufficient coverage, valid clinical tasks, and consistent model sets)")
    elif config.get('FOR_VALID_TASKS_AND_PERFS', False):
        print("Analysis mode: FILTERED (using only benchmarks with sufficient coverage and valid clinical tasks)")
    else:
        print("Analysis mode: UNFILTERED (using all available benchmarks and clinical tasks)")
    
    # Model consistency verification for strict mode
    if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False) and 'models' in results_df.columns:
        verify_model_consistency(results_df)
        
        # Remove the models column as it's no longer needed
        results_df = results_df.drop('models', axis=1)
    
    # Print task statistics
    print(f"Total clinical tasks considered: {len(stats['task_model_counts']) + stats['excluded_tasks']}")
    print(f"Tasks excluded (insufficient data): {stats['excluded_tasks']}")
    print(f"Tasks included in analysis: {stats['included_tasks']}")
    print(f"Total model samples analyzed: {stats['total_model_samples']}")
    
    if stats['included_tasks'] > 0:
        print(f"Average models per task: {stats['total_model_samples'] / stats['included_tasks']:.2f}")
    
    # Print detailed model availability stats
    if stats['task_model_counts']:
        print(f"\nModel availability per task:")
        print(f"  Clinical models: {np.mean(stats['task_model_counts']):.2f} ± {np.std(stats['task_model_counts']):.2f} (min: {np.min(stats['task_model_counts'])}, max: {np.max(stats['task_model_counts'])})")
        print(f"  Models with general benchmark data: {np.mean(stats['general_valid_model_counts']):.2f} ± {np.std(stats['general_valid_model_counts']):.2f} (min: {np.min(stats['general_valid_model_counts'])}, max: {np.max(stats['general_valid_model_counts'])})")
        print(f"  Models with MedQA benchmark data: {np.mean(stats['medqa_valid_model_counts']):.2f} ± {np.std(stats['medqa_valid_model_counts']):.2f} (min: {np.min(stats['medqa_valid_model_counts'])}, max: {np.max(stats['medqa_valid_model_counts'])})")
    
    # Print correlation summary by benchmark type
    print_correlation_summary(results_df, weighting_by_sample_size)

def convert_category_data_category(category_data_category):
    """
    Convert category data to standardized format and capitalize words appropriately.
    """
    category_data_category_converted = []
    prepositions = ['to', 'in', 'at', 'by', 'for', 'with', 'from', 'of', 'on']
    
    for cdc_i in category_data_category:
        if 'therapeutic' in cdc_i:
            cdc_i = 'Treatment'
        elif 'answering (to patients)' in cdc_i:
            cdc_i = 'Answering to Patients'
        elif 'vignettes' in cdc_i:
            cdc_i = 'Clinical Vignettes'
        elif 'else' == cdc_i.strip():
            cdc_i = 'Others'
        else:
            # Split by whitespace and capitalize each word except prepositions
            words = cdc_i.split()
            capitalized_words = []
            
            for i, word in enumerate(words):
                # Capitalize first word regardless of what it is
                if i == 0:
                    capitalized_words.append(word.capitalize())
                # Don't capitalize prepositions unless they're the first word
                elif word.lower() in prepositions:
                    capitalized_words.append(word.lower())
                # Capitalize all other words
                else:
                    capitalized_words.append(word.capitalize())
            cdc_i = ' '.join(capitalized_words)

        if 'mcq' in cdc_i.lower():
            cdc_i = 'MCQ'
            
        # Add line breaks for better readability in charts
        cdc_i = cdc_i.replace('&', '&\n')
        cdc_i = cdc_i.replace('Frequently Asked', 'Frequently Asked\n')
        cdc_i = cdc_i.replace('Board', 'Board\n')
        cdc_i = cdc_i.replace('Clinical', 'Clinical\n')
        cdc_i = cdc_i.replace('Overall', 'Overall\n')
        cdc_i = cdc_i.replace('Answering to', 'Answering to\n')
        cdc_i = cdc_i.replace('Information', 'Information\n')
        cdc_i = cdc_i.replace('General', 'General\n')
        
        category_data_category_converted.append(cdc_i)
    
    return category_data_category_converted

def plot_single_correlation_metric(df, category, metric, ordered_categories,
                                  ax, weighting_by_sample_size=False,
                                  corr_data=None, show_title=True):
    """
    Plot a single correlation metric for a category on the provided axis.
    """
    # Define color for bars
    color_map = {
        'pearson': '#3274A1',
        'spearman': '#E1812C',
        'kendall': '#3A923A',
        'lin_ccc': '#C03D3E'
    }
    # Define colors for benchmark types (for reference lines)
    ref_colors = {'general': 'royalblue', 'medqa': 'forestgreen'}

    # If we're using therapeutic_area_group, recalculate ordered_categories
    if category == 'therapeutic_area_group':
        # Group by the new category for proper ordering
        if weighting_by_sample_size and 'sample_size' in df.columns:
            # Calculate weighted means for each group
            group_means = {}
            for cat, group in df.groupby(category):
                weights = np.log1p(group['sample_size'].fillna(1))
                mean_val = np.average(group[metric], weights=weights) if len(group) > 0 else np.nan
                group_means[cat] = mean_val
            
            # Sort groups by their weighted means
            ordered_categories = sorted(group_means.keys(), key=lambda k: group_means[k], reverse=True)
        else:
            # Regular means
            category_stats = df.groupby(category)[metric].mean().reset_index()
            category_stats = category_stats.sort_values(metric, ascending=False)
            ordered_categories = category_stats[category].tolist()
    
    # Group by category and calculate mean with optional weighting
    if weighting_by_sample_size and 'sample_size' in df.columns:
        # Calculate weighted average and count
        category_data = []
        for cat, group in df.groupby(category):
            # wi = log(1 + sample_sizei)
            weights = np.log1p(group['sample_size'].fillna(1))
            mean_val = np.average(group[metric], weights=weights) if len(group) > 0 else np.nan
            category_data.append({
                category: cat,
                'mean': mean_val,
                'count': len(group)
            })
        category_data = pd.DataFrame(category_data)
    else:
        # Original method - regular average
        category_data = df.groupby(category)[metric].agg(['mean', 'count']).reset_index()
    
    # Reorder according to ordered categories
    category_data[category] = pd.Categorical(
        category_data[category], 
        categories=ordered_categories, 
        ordered=True
    )
    category_data = category_data.sort_values(category)
    
    # Calculate number of tasks per category
    task_counts = df.groupby(category)['task_name'].nunique().reset_index()
    task_counts.columns = [category, 'task_count']
    
    # Merge with main data
    category_data = pd.merge(category_data, task_counts, on=category)
    
    category_data[category] = convert_category_data_category(category_data[category])
    
    # Plot bars
    bars = ax.bar(
        category_data[category], 
        category_data['mean'],
        color=color_map[metric],
        alpha=0.8
    )
    
    # Get average benchmark correlations if data is available
    if corr_data is not None:
        # Map lin_ccc to ccc for benchmark correlation data
        bench_metric = 'ccc' if metric == 'lin_ccc' else metric
        
        # Get average correlations for MedQA benchmark
        benchmark_name = 'MedQA'  # We only care about MedQA here
        benchmark_type = 'medqa'
        
        same_type_avg, other_type_avg = calculate_avg_benchmark_correlations(
            corr_data, benchmark_name, benchmark_type, bench_metric)
        
        # Get x-axis limits
        x_min, x_max = ax.get_xlim()
        
        # Plot same type average correlation line (Medical benchmarks - green)
        if same_type_avg is not None:
            same_color = ref_colors['medqa']
            # Create a more transparent version of the color
            same_color_alpha = list(mcolors.to_rgba(same_color))
            same_color_alpha[3] = 0.5  # Lower alpha for lighter color
            
            # Draw line across the entire graph
            ax.axhline(
                y=same_type_avg,
                color=same_color_alpha,
                linewidth=3.5,
                linestyle='-'
            )
            
            # Add value label at the right edge of the plot
            ax.text(
                x_max * 0.98,  # Slightly inside from the right edge
                same_type_avg,
                f"{same_type_avg:.2f}",
                ha='right',
                va='bottom',
                color=same_color,
                fontsize=14,
                fontweight='bold'
            )
        
        # Plot other type average correlation line (General benchmarks - blue)
        if other_type_avg is not None:
            other_color = ref_colors['general']
            # Create a more transparent version of the color
            other_color_alpha = list(mcolors.to_rgba(other_color))
            other_color_alpha[3] = 0.5  # Lower alpha for lighter color
            
            # Draw line across the entire graph
            ax.axhline(
                y=other_type_avg,
                color=other_color_alpha,
                linewidth=3.5,
                linestyle='-'
            )
            
            # Add value label at the right edge of the plot
            ax.text(
                x_max * 0.98,  # Slightly inside from the right edge
                other_type_avg,
                f"{other_type_avg:.2f}",
                ha='right',
                va='bottom',
                color=other_color,
                fontsize=14,
                fontweight='bold'
            )
    
    # Add values and task counts above bars
    for bar, mean_val, count, task_count in zip(
        bars, 
        category_data['mean'], 
        category_data['count'],
        category_data['task_count']
    ):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{mean_val:.2f}\n({task_count} tasks)",
            ha='center',
            va='bottom',
            fontsize=13,
        )
    
    # Set title and labels only if requested
    if show_title:
        ax.set_title(f"{metric.capitalize()} Correlation", fontsize=18)
    
    ax.set_ylabel("Correlation Coefficient", fontsize=16)
    ax.set_xlabel("Categories", fontsize=16)
    ax.set_ylim(0, 1)
    
    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Format x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=14)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

def calculate_avg_benchmark_correlations(corr_data, benchmark_name, benchmark_type, measure):
    """Calculate average correlations between a benchmark and others of same/different type."""
    if not corr_data or 'correlations' not in corr_data:
        return None, None
    
    corr_method = corr_data['correlations'].get(measure)
    if not corr_method:
        return None, None
    
    # Determine benchmark categories
    benchmark_cat = 'medical' if benchmark_type == 'medqa' else 'general'
    other_cat = 'general' if benchmark_cat == 'medical' else 'medical'
    
    # Get same-category correlations
    same_cat_corrs = []
    if benchmark_cat in corr_method and benchmark_name in corr_method[benchmark_cat]:
        for other_bench, corr_val in corr_method[benchmark_cat][benchmark_name].items():
            if other_bench != benchmark_name:  # Exclude self-correlation
                same_cat_corrs.append(corr_val)
    
    # Get cross-category correlations
    cross_cat_corrs = []
    if 'cross' in corr_method and benchmark_name in corr_method['cross']:
        for other_bench, corr_val in corr_method['cross'][benchmark_name].items():
            cross_cat_corrs.append(corr_val)
    
    # Calculate averages
    same_cat_avg = np.mean(same_cat_corrs) if same_cat_corrs else None
    cross_cat_avg = np.mean(cross_cat_corrs) if cross_cat_corrs else None
    
    return same_cat_avg, cross_cat_avg

def create_category_correlation_plot(df, category, correlation_metrics, config,
                                     use_imputated=False,
                                     weighting_by_sample_size=False,
                                     two_corr_measure=False,
                                     ta_classification=None):
    """Create a figure with subplots for each correlation metric for a given metadata category."""
    # Import matplotlib colors for alpha manipulation
    from matplotlib.lines import Line2D
    
    # Set figure width to 1.2 times the original width
    width_factor = 1.2
    
    # Determine which correlation measures to use based on two_corr_measure flag
    if two_corr_measure:
        used_metrics = [m for m in correlation_metrics if m in ['spearman', 'kendall']]
        fig, axes = plt.subplots(1, 2, figsize=(16 * width_factor, 6))  # Slightly smaller size
    else:
        used_metrics = correlation_metrics
        fig, axes = plt.subplots(2, 2, figsize=(16 * width_factor, 12))  # Slightly smaller size
    
    axes = axes.flatten()
    
    # Filter out categories with 3 or fewer tasks
    task_counts = df.groupby(category)['task_name'].nunique()
    valid_categories = task_counts[task_counts > 3].index.tolist()
    df_filtered = df[df[category].isin(valid_categories)]
    
    # For therapeutic_area with classification enabled, map detailed categories to groups
    if category == 'therapeutic_area' and ta_classification:
        # Create a mapping from detailed TA to group
        ta_to_group = {}
        for group, tas in ta_classification.items():
            for ta in tas:
                ta_to_group[ta] = group
        
        # Apply the mapping to create a new grouped column
        df_filtered = df_filtered.copy()
        df_filtered['therapeutic_area_group'] = df_filtered['therapeutic_area'].map(
            lambda x: ta_to_group.get(x.lower(), 'other') if isinstance(x, str) else 'other'
        )
        
        # Use the grouped column instead
        category = 'therapeutic_area_group'

    # Group by category and calculate mean for each correlation metric
    metrics_agg = {}
    for metric in correlation_metrics:
        metrics_agg[metric] = ['mean', 'count']
    
    category_stats = df_filtered.groupby(category).agg(metrics_agg).reset_index()
    
    # Sort by spearman correlation in descending order
    category_stats = category_stats.sort_values(('spearman', 'mean'), ascending=False)
    
    # Fixed order of categories based on spearman correlation
    ordered_categories = category_stats[category].tolist()
    
    # Load benchmark correlations for comparison lines
    corr_data = load_benchmark_correlations(f'{DIR_FINAL_DATASET}/correlations_fin_250228.json')
    
    # Plot each correlation metric in a separate subplot
    for i, metric in enumerate(used_metrics):
        plot_single_correlation_metric(
            df=df_filtered,
            category=category,
            metric=metric,
            ordered_categories=ordered_categories,
            ax=axes[i],
            weighting_by_sample_size=weighting_by_sample_size,
            corr_data=corr_data
        )
    
    # Add legend for correlation lines with improved title and formatting
    ref_colors = {'general': 'royalblue', 'medqa': 'forestgreen'}
    
    # Calculate the position for legends - 1.5mm to the right of the last plot
    # Convert 1.5mm to figure units (assuming 1 inch = 25.4 mm)
    mm_to_inches = 1 / 25.4
    offset_inches = 0.2 * mm_to_inches
    
    # Get the figure size in inches
    fig_width_inches = fig.get_figwidth()
    
    # Calculate the position as a fraction of figure width
    legend_x_pos = 0.85 + (offset_inches / fig_width_inches)

    # Create legend: Average Correlation with Benchmarks
    avg_legend_elements = [
        Line2D([0], [0], color=ref_colors['medqa'], lw=2, alpha=0.7,
               label='Medical QA'),
        Line2D([0], [0], color=ref_colors['general'], lw=2, alpha=0.7,
               label='General')
    ]
    
    second_legend = fig.legend(
        handles=avg_legend_elements,
        loc='upper left',
        bbox_to_anchor=(legend_x_pos, 0.43),
        ncol=1,
        fontsize=12,
        title="Average Correlation\nwith Benchmarks",
        title_fontsize=13,
        alignment='left'
    )
    
    # Manually set the alignment of the legend title to left
    second_legend._legend_box.align = "left"

    # Adjust layout to accommodate the legends
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])  # Make room for the legends on the right
    
    # Save plot in multiple formats
    if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
        plot_suffix = '_strictly_filtered'
    elif config.get('FOR_VALID_TASKS_AND_PERFS', False):
        plot_suffix = '_filtered'
    else:
        plot_suffix = ''
        
    if use_imputated:
        plot_suffix += '_imputated'
        
    if weighting_by_sample_size:
        plot_suffix += '_weighted'
    
    if two_corr_measure:
        plot_suffix += '_two_measures'
    
    # Base filename
    plot_filename_base = f'medqa_correlation_{category}{plot_suffix}'
    
    # Save in different formats
    formats = {
        'png': {'dpi': 600},
        'tiff': {'dpi': 600},
        'pdf': {'dpi': 600, 'bbox_inches': 'tight', 'format': 'pdf'}
    }
    
    for fmt, params in formats.items():
        filename = f'{DIR_FIGURES}/{plot_filename_base}.{fmt}'
        plt.savefig(filename, **params)
        print(f"Plot saved to {filename}")
    
    # Display the plot
    plt.show()

def create_category_panels_plot(df, categories, correlation_metrics, config,
                               use_imputated=False,
                               two_corr_measure=False,
                               ta_classification=None,
                               weighting_by_sample_size=False):
    """
    Create a figure with panels for different categories, each showing multiple correlation metrics.
    """
    # Set global font size
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    
    # Determine which correlation measures to use
    if two_corr_measure:
        used_metrics = [m for m in correlation_metrics if m in ['spearman', 'kendall']]
    else:
        used_metrics = correlation_metrics
    
    # Calculate total number of subplots (categories × metrics)
    num_metrics = len(used_metrics)
    
    # Create a figure with 2x2 grid for categories, and each cell will have num_metrics columns
    fig = plt.figure(figsize=(num_metrics * 20, 16))
    
    # Create a nested gridspec layout
    # First level is 2x2 for the A, B, C, D panels
    main_gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.1)
    
    # Panel labels (A, B, C, D)
    panel_labels = ['A', 'B', 'C', 'D']
    
    # Capitalize category names for panel titles
    panel_titles = []
    for category in categories:
        # Basic capitalization of category name
        words = category.split('_')
        capitalized = ' '.join(word.capitalize() for word in words)
        panel_titles.append(capitalized)
    
    # Load benchmark correlations for comparison lines
    corr_data = load_benchmark_correlations(f'{DIR_FINAL_DATASET}/correlations_fin_250228.json')
    
    # Plot each category in a 2x2 grid
    for i, (category, panel_label, panel_title) in enumerate(zip(categories, panel_labels, panel_titles)):
        # Calculate row and column for this panel in the 2x2 grid
        row = i // 2
        col = i % 2
        
        # Create a nested gridspec for this panel (for the metrics)
        panel_gs = main_gs[row, col].subgridspec(1, num_metrics, wspace=0.1)
        
        # Filter out categories with 3 or fewer tasks
        task_counts = df.groupby(category)['task_name'].nunique()
        valid_categories = task_counts[task_counts > 3].index.tolist()
        df_filtered = df[df[category].isin(valid_categories)]
        
        # For therapeutic_area with classification enabled, map detailed categories to groups
        if category == 'therapeutic_area' and ta_classification:
            # Create a mapping from detailed TA to group
            ta_to_group = {}
            for group, tas in ta_classification.items():
                for ta in tas:
                    ta_to_group[ta] = group
            
            # Apply the mapping to create a new grouped column
            df_filtered = df_filtered.copy()
            df_filtered['therapeutic_area_group'] = df_filtered['therapeutic_area'].map(
                lambda x: ta_to_group.get(x.lower(), 'other') if isinstance(x, str) else 'other'
            )
            
            # Use the grouped column instead
            category = 'therapeutic_area_group'
    
        # Group by category and calculate mean for each correlation metric
        metrics_agg = {}
        for metric in correlation_metrics:
            metrics_agg[metric] = ['mean', 'count']
        
        category_stats = df_filtered.groupby(category).agg(metrics_agg).reset_index()
        
        # Sort by spearman correlation in descending order
        category_stats = category_stats.sort_values(('spearman', 'mean'), ascending=False)
        
        # Fixed order of categories based on spearman correlation
        ordered_categories = category_stats[category].tolist()
        
        # Plot each correlation metric in a separate column within this panel
        for j, metric in enumerate(used_metrics):
            # Create subplot for this metric within the panel
            ax = fig.add_subplot(panel_gs[0, j])
            
            # If this is the first metric in the panel, add panel label above the subplot
            if j == 0:
                # Add panel label and category name outside the top-left of the subplot
                # Using figure coordinates for consistent positioning
                bbox = ax.get_position()
                fig.text(bbox.x0 - 0.02, bbox.y1 + 0.03, f'({panel_label}) {panel_title}',
                         fontsize=20, fontweight='bold', va='bottom', ha='left')
            
            # Plot the correlation data with the metric title as subplot title
            plot_single_correlation_metric(
                df=df_filtered,
                category=category,
                metric=metric,
                ordered_categories=ordered_categories,
                ax=ax,
                weighting_by_sample_size=weighting_by_sample_size,
                corr_data=corr_data,
                show_title=True  # Show metric title for each subplot
            )
            
            # Only show y-axis label on first column for each panel
            if j > 0:
                ax.set_ylabel("")
    
    # Create a legend for the benchmark correlation lines
    ref_colors = {'general': 'royalblue', 'medqa': 'forestgreen'}
    
    # Average correlation lines legend
    avg_legend_elements = [
        Line2D([0], [0], color=ref_colors['medqa'], lw=3, alpha=0.7,
               label='Medical QA'),
        Line2D([0], [0], color=ref_colors['general'], lw=3, alpha=0.7,
               label='General')
    ]
    
    # Convert 1.5mm to figure units (assuming 1 inch = 25.4 mm)
    mm_to_inches = 1 / 25.4
    offset_inches = 0.1 * mm_to_inches
    
    # Get the figure size in inches
    fig_width_inches = fig.get_figwidth()
    
    # Calculate the position as a fraction of figure width
    legend_x_pos = 0.98 + (offset_inches / fig_width_inches)

    # Add the legend to the right of the figure
    legend = fig.legend(
        handles=avg_legend_elements,
        loc='center right',
        bbox_to_anchor=(legend_x_pos, 0.2),
        fontsize=16,
        title="Average Correlation\nwith Benchmarks",
        title_fontsize=18
    )
    
    # Manually set the alignment of the legend title to left
    legend._legend_box.align = "left"
    # Adjust layout to accommodate the legend
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Save plot in multiple formats
    if config.get('FOR_STRICTLY_VALID_TASKS_AND_PERFS', False):
        plot_suffix = '_strictly_filtered'
    elif config.get('FOR_VALID_TASKS_AND_PERFS', False):
        plot_suffix = '_filtered'
    else:
        plot_suffix = ''

    if use_imputated:
        plot_suffix += '_imputated'
        
    if weighting_by_sample_size:
        plot_suffix += '_weighted'
    
    if two_corr_measure:
        plot_suffix += '_two_measures'
    
    # Base filename
    plot_filename_base = f'medqa_correlation_panels{plot_suffix}'
    
    # Save in different formats
    formats = {
        'png': {'dpi': 600},
        'tiff': {'dpi': 600},
        'pdf': {'dpi': 600, 'bbox_inches': 'tight', 'format': 'pdf'}
    }
    
    for fmt, params in formats.items():
        filename = f'{DIR_FIGURES}/{plot_filename_base}.{fmt}'
        plt.savefig(filename, **params)
        print(f"Plot saved to {filename}")

    # Display the plot
    plt.show()

def create_task_metadata_visualizations(results_df, datasets, config,
                                       use_imputated=False,
                                       weighting_by_sample_size=False,
                                       plot_category_as_panel=False,
                                       two_corr_measure=False,
                                       ta_classification=None):
    """Create visualizations for correlation results by task metadata categories."""
    print("\n=== Creating Task Metadata Visualizations for MedQA Benchmark ===")
    
    # Filter results for only MedQA benchmark
    medqa_results = results_df[
        (results_df['benchmark_type'] == 'medqa') & 
        (results_df['benchmark_name'] == 'MedQA')
    ]
    
    if len(medqa_results) == 0:
        print("No MedQA benchmark results found for visualization")
        return
    
    # Get clinical performance dataframe
    clinical_df = datasets['clinical_perf_df']
    
    # Merge task metadata with correlation results
    task_meta_cols = ['task_name', 'task_type', 'therapeutic_area', 'data_source', 'evaluation_type']
    task_meta = clinical_df[task_meta_cols].drop_duplicates()
    
    medqa_with_meta = pd.merge(
        medqa_results, 
        task_meta,
        on='task_name',
        how='left'
    )
    
    # Check if we have metadata
    if medqa_with_meta.isna().any().any():
        print("Warning: Some tasks are missing metadata information")
    
    # Create visualizations for each metadata category
    metadata_categories = ['task_type', 'therapeutic_area', 'data_source', 'evaluation_type']
    correlation_metrics = ['pearson', 'spearman', 'kendall', 'lin_ccc']
    
    # For each metadata category, create a separate figure with subplots (one for each correlation measure)
    if plot_category_as_panel:
        # Create a single plot with all categories as panels
        create_category_panels_plot(
            medqa_with_meta, 
            metadata_categories,
            correlation_metrics, 
            config,
            use_imputated,
            two_corr_measure,
            ta_classification,
            weighting_by_sample_size
        )
    else:
        # Create individual plots for each category (original behavior)
        for category in metadata_categories:
            create_category_correlation_plot(
                medqa_with_meta, 
                category, 
                correlation_metrics, 
                config,
                use_imputated,
                weighting_by_sample_size,
                two_corr_measure,
                ta_classification
            )
    
    # Create detailed task-type correlation report
    create_task_type_correlation_report(medqa_with_meta, weighting_by_sample_size)

def create_task_type_correlation_report(df, weighting_by_sample_size=False):
    """Create a detailed correlation report by task_type."""
    print("\n=== MedQA Correlation Report by Task Type ===")
    
    # Weighting message based on whether weighting is used
    weighting_msg = " (weighted by log(1+sample_size))" if weighting_by_sample_size else ""
    
    # Group by task_type
    for task_type, group_df in df.groupby('task_type'):
        task_count = len(group_df['task_name'].unique())
        model_count = group_df['num_models'].sum()
        
        # Calculate mean correlations with optional weighting
        if weighting_by_sample_size and 'sample_size' in group_df.columns:
            # Weight calculation: wi = log(1 + sample_sizei)
            weights = np.log1p(group_df['sample_size'].fillna(1))
            
            # Calculate weighted means
            mean_pearson = np.average(group_df['pearson'], weights=weights)
            mean_spearman = np.average(group_df['spearman'], weights=weights)
            mean_kendall = np.average(group_df['kendall'], weights=weights)
            mean_lin_ccc = np.average(group_df['lin_ccc'], weights=weights)
        else:
            # Regular means
            mean_pearson = group_df['pearson'].mean()
            mean_spearman = group_df['spearman'].mean()
            mean_kendall = group_df['kendall'].mean()
            mean_lin_ccc = group_df['lin_ccc'].mean()
        
        # Calculate percentage of significant correlations
        sig_pearson = sum(group_df['pearson_pval'] < 0.05) / len(group_df) * 100
        sig_spearman = sum(group_df['spearman_pval'] < 0.05) / len(group_df) * 100
        sig_kendall = sum(group_df['kendall_pval'] < 0.05) / len(group_df) * 100
        
        print(f"\nTask Type: {task_type}")
        print(f"  Tasks: {task_count}")
        print(f"  Total model samples: {model_count}")
        print(f"  Mean correlations{weighting_msg}:")
        print(f"    Pearson: {mean_pearson:.4f}  (sig: {sig_pearson:.1f}%)")
        print(f"    Spearman: {mean_spearman:.4f}  (sig: {sig_spearman:.1f}%)")
        print(f"    Kendall: {mean_kendall:.4f}  (sig: {sig_kendall:.1f}%)")
        print(f"    Lin's CCC: {mean_lin_ccc:.4f}")
        
        # List the specific tasks
        print(f"  Tasks: {', '.join(group_df['task_name'].unique())}")
