"""Perform imputation on benchmark performances using MICE."""
import os
import pickle
import copy
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the data loading functions
from measure_correlations_from_perf_data_pickle import (
    load_and_process_perf_data,
    FNAME_PERF_GENERAL,
    GENERAL_BENCHES,
    FNAME_PERF_MEDQA,
    MED_BENCHES,
)

from utils_imputation import (
    visualize_missing_pattern_test_results,
    visualize_missing_data,
    visualize_missing_by_model,
    filter_models_by_valid_data_count,
    calculate_missingness_statistics,
    test_missing_patterns,
    littles_mcar_test,
    test_missingness_correlation,
    test_missingness_predictability,
    test_mar_dependency,
    test_mnar_pattern,
    conclude_missing_pattern,
    visualize_imputation_results
)

# Directory configuration
OUTPUT_DIR = 'imputed_benchmarks'
FIGURES_DIR = 'figures'

# Processing parameters
APPLY_FILTERING = False  # Whether to filter models based on valid data count
MIN_VALID_GENERAL = 3    # Minimum valid data points for general benchmarks
MIN_VALID_MEDICAL = 4    # Minimum valid data points for medical benchmarks

# Randomization and simulation parameters
RANDOM_SEED = 42        # Base random seed for reproducibility
MASK_RATIO = 0.1        # Ratio of observed values to mask for evaluation
NUM_IMPUTATIONS = 20    # Number of imputations for multiple imputation

# Benchmark value constraints
VALUE_RANGE = (0, 100)  # Valid range for benchmark values (percentage scores)

# Imputation algorithm parameters
MAX_ITERATIONS = 50     # Maximum iterations for IterativeImputer
CONVERGENCE_TOLERANCE = 1e-6  # Convergence tolerance for imputation algorithm


def create_directories():
    """Create necessary directories for output files."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_data(df):
    """Preprocess dataframe to handle invalid values."""
    # Replace values greater than 100 with NaN
    return df.where(df <= VALUE_RANGE[1])


def perform_mice_imputation(df, estimator, name, random_state=RANDOM_SEED):
    """Perform MICE imputation with the specified estimator.
    
    Args:
        df: DataFrame with missing values
        estimator: Scikit-learn compatible estimator for imputation
        name: Name of the dataset for reporting
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (imputed_dataframe, imputer_object)
    """
    # Suppress convergence warnings during imputation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        imputer = IterativeImputer(
            estimator=estimator,
            initial_strategy='median',
            random_state=random_state,
            sample_posterior=False,
            max_iter=MAX_ITERATIONS,
            tol=CONVERGENCE_TOLERANCE,
            verbose=0
        )
        
        # Fit and transform the data
        imputed_array = imputer.fit_transform(df)
    
    # Clip values to ensure they're in the valid range
    imputed_array = np.clip(imputed_array, VALUE_RANGE[0], VALUE_RANGE[1])
    
    imputed_df = pd.DataFrame(imputed_array, index=df.index, columns=df.columns)
    
    print(f"\n=== MICE Imputation Results for {name} with {estimator.__class__.__name__} ===\n")
    print(f"Shape before imputation: {df.shape}")
    print(f"Shape after imputation: {imputed_df.shape}")
    
    return imputed_df, imputer


def evaluate_with_masking(original_df, imputer, name, estimator_name, mask_ratio=MASK_RATIO):
    """Evaluate imputation quality by masking observed values and comparing results.
    
    Args:
        original_df: Original DataFrame with missing values
        imputer: Fitted IterativeImputer object
        name: Name of the dataset for reporting
        estimator_name: Name of the estimator used
        mask_ratio: Ratio of observed values to mask for evaluation
        
    Returns:
        Tuple of (mae, rmse, r2) evaluation metrics
    """
    # Make a copy of the original data
    eval_df = original_df.copy()
    
    # Get indices of non-missing values
    observed_idx = np.where(~original_df.isna().values)
    
    # Randomly select a subset of observed values to mask
    np.random.seed(RANDOM_SEED)
    mask_size = int(len(observed_idx[0]) * mask_ratio)
    mask_indices = np.random.choice(len(observed_idx[0]), size=mask_size, replace=False)
    
    # Extract the original values before masking
    mask_rows = observed_idx[0][mask_indices]
    mask_cols = observed_idx[1][mask_indices]
    original_values = original_df.values[mask_rows, mask_cols]
    
    # Mask the selected values
    eval_df.values[mask_rows, mask_cols] = np.nan
    
    # Perform imputation with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        imputed_array = imputer.transform(eval_df)
    
    # Clip values to ensure they're in the valid range
    imputed_array = np.clip(imputed_array, VALUE_RANGE[0], VALUE_RANGE[1])
    
    # Extract the imputed values
    imputed_values = imputed_array[mask_rows, mask_cols]
    
    # Calculate metrics
    mae = mean_absolute_error(original_values, imputed_values)
    rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
    r2 = r2_score(original_values, imputed_values)
    
    print(f"\n=== Masking Test Results for {name} with {estimator_name} ===\n")
    print(f"Mask ratio: {mask_ratio}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Visualize results
    create_masked_values_plot(original_values, imputed_values, name, estimator_name, mae, rmse, r2)

    return mae, rmse, r2


def create_masked_values_plot(original_values, imputed_values, name, estimator_name, mae, rmse, r2):
    """Create and save scatter plot of original vs imputed values."""
    plt.figure(figsize=(5.5, 5))
    plt.scatter(original_values, imputed_values, alpha=0.6)
    plt.plot([VALUE_RANGE[0], VALUE_RANGE[1]], [VALUE_RANGE[0], VALUE_RANGE[1]], 'r--')
    plt.xlabel('Original Values')
    plt.ylabel('Imputed Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(VALUE_RANGE)
    plt.ylim(VALUE_RANGE)
    plt.gca().set_aspect('equal')
    plt.text(5, 90, f'MAE: {mae:.3f}', fontsize=9)
    plt.text(5, 85, f'RMSE: {rmse:.3f}', fontsize=9)
    plt.text(5, 80, f'R²: {r2:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/masked_values_scatter_{name}_{estimator_name}.png', dpi=600)
    plt.close()


def configure_estimator_for_multiple_imputation(estimator, imputation_index):
    """Configure estimator for multiple imputation with controlled variability."""
    is_random_forest = isinstance(estimator, RandomForestRegressor)
    
    if is_random_forest:
        # Create a modified copy of the estimator with controlled variability
        modified_estimator = copy.deepcopy(estimator)
        modified_estimator.random_state = RANDOM_SEED + imputation_index
        modified_estimator.bootstrap = True
        modified_estimator.max_samples = 0.85  # Use 85% of samples for training
        modified_estimator.max_features = "sqrt"  # Use sqrt of features for each tree
        modified_estimator.max_depth = 15  # Limit tree depth to prevent overfitting
        return modified_estimator
    else:
        # For BayesianRidge or other models, use the original estimator
        return estimator


def calculate_existing_data_variance(df):
    """Calculate variance statistics for non-missing data in the DataFrame."""
    # Get non-missing values only
    non_missing_mask = ~df.isna()

    # Calculate overall variance of all non-missing values
    all_values = df.values[non_missing_mask.values]
    overall_variance = np.var(all_values)

    # Calculate column-wise (benchmark-wise) variances
    column_variances = df.var(skipna=True)
    mean_column_variance = column_variances.mean()
    max_column_variance = column_variances.max()
    min_column_variance = column_variances.min()

    # Calculate row-wise (model-wise) variances
    row_variances = df.var(axis=1, skipna=True)
    mean_row_variance = row_variances.mean()
    max_row_variance = row_variances.max()
    min_row_variance = row_variances.min()
    
    # Print results
    print("\n=== Existing Data Variance Statistics ===\n")
    print(f"Overall variance of non-missing values: {overall_variance:.6f}")
    print("\nBenchmark-wise variance statistics:")
    print(f"  Mean benchmark variance: {mean_column_variance:.6f}")
    print(f"  Max benchmark variance: {max_column_variance:.6f}")
    print(f"  Min benchmark variance: {min_column_variance:.6f}")
    print("\nModel-wise variance statistics:")
    print(f"  Mean model variance: {mean_row_variance:.6f}")
    print(f"  Max model variance: {max_row_variance:.6f}")
    print(f"  Min model variance: {min_row_variance:.6f}")
    
    # Return statistics as a dictionary
    return {
        'overall_variance': overall_variance,
        'benchmark_variance': {
            'mean': mean_column_variance,
            'max': max_column_variance,
            'min': min_column_variance
        },
        'model_variance': {
            'mean': mean_row_variance,
            'max': max_row_variance,
            'min': min_row_variance
        }
    }


def improved_multiple_imputation(df, estimator, name, n_imputations=NUM_IMPUTATIONS):
    """Perform multiple imputation with robust handling of randomness.
    
    Args:
        df: DataFrame with missing values
        estimator: Scikit-learn compatible estimator for imputation
        name: Name of the dataset for reporting
        n_imputations: Number of imputations to perform
        
    Returns:
        Tuple of (median_imputation_df, (within_var, between_var, total_var), imputations_list)
    """
    imputations = []
    
    # Determine if estimator supports posterior sampling
    sample_posterior = isinstance(estimator, BayesianRidge)
    is_random_forest = isinstance(estimator, RandomForestRegressor)
    
    # Perform multiple imputations with different random seeds
    for i in range(n_imputations):
        current_estimator = configure_estimator_for_multiple_imputation(estimator, i)
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Create a new imputer with the adjusted estimator
            imputer = IterativeImputer(
                estimator=current_estimator,
                initial_strategy='median',
                random_state=RANDOM_SEED + i if not is_random_forest else None,
                sample_posterior=sample_posterior,
                max_iter=MAX_ITERATIONS,
                tol=CONVERGENCE_TOLERANCE,
                verbose=0
            )
            
            imputed_array = imputer.fit_transform(df)
        
        # Clip values to ensure they're in the valid range
        imputed_array = np.clip(imputed_array, VALUE_RANGE[0], VALUE_RANGE[1])
        imputed_df = pd.DataFrame(imputed_array, index=df.index, columns=df.columns)
        imputations.append(imputed_df)
    
    # Calculate variance measures
    stacked_imputations = np.stack([imp.values for imp in imputations])
    
    # Calculate within-imputation variance (average of variances)
    within_var = np.nanmean(np.var(stacked_imputations, axis=0))
    
    # Calculate between-imputation variance (variance of means)
    between_var = np.var(np.mean(stacked_imputations, axis=1))
    
    # Calculate total variance using Rubin's rules
    total_var = within_var + (1 + 1/n_imputations) * between_var
    
    print(f"\n=== Multiple Imputation Results for {name} ===\n")
    print(f"Number of imputations: {n_imputations}")
    print(f"Within-imputation variance: {within_var:.6f}")
    print(f"Between-imputation variance: {between_var:.6f}")
    print(f"Total variance: {total_var:.6f}")
    
    # Return the median of all imputations
    median_imputation = pd.DataFrame(
        np.median(stacked_imputations, axis=0),
        index=df.index,
        columns=df.columns
    )
    
    return median_imputation, (within_var, between_var, total_var), imputations


def save_imputation_results(imputed_df, multi_imputed_df, imputations, data_type, dataset_type, estimator_type):
    """Save imputation results in CSV and pickle formats."""
    # Save CSV file in the current directory
    imputed_df.to_csv(f'{data_type}_{dataset_type}_benchmarks_imputed_{estimator_type}.csv')
    
    # Save PKL files in output directory
    with open(f'{OUTPUT_DIR}/{data_type}_{dataset_type}_benchmarks_imputed_{estimator_type}.pkl', 'wb') as f:
        pickle.dump(imputed_df, f)
    
    with open(f'{OUTPUT_DIR}/{data_type}_{dataset_type}_benchmarks_multi_imputed_{estimator_type}.pkl', 'wb') as f:
        pickle.dump(multi_imputed_df, f)
        
    with open(f'{OUTPUT_DIR}/{data_type}_{dataset_type}_benchmarks_all_imputations_{estimator_type}.pkl', 'wb') as f:
        pickle.dump(imputations, f)


def process_dataset(df, name, data_type, rf_estimator, br_estimator):
    """Process a dataset with both RF and BR estimators and save results.
    
    Args:
        df: DataFrame to process
        name: Name of the dataset
        data_type: 'original' or 'filtered'
        rf_estimator: RandomForest estimator instance
        br_estimator: BayesianRidge estimator instance
        
    Returns:
        Dictionary of imputation results
    """
    dataset_results = {}
    dataset_type = "general" if "General" in name else "medical"
    
    # Process with RandomForest
    print("\n\n" + "="*80)
    print(f"Processing {name} with RandomForest")
    print("="*80)
    
    # MICE imputation
    imputed_rf, imputer_rf = perform_mice_imputation(df, rf_estimator, name)
    
    # Evaluation with masking
    metrics_rf = evaluate_with_masking(df, imputer_rf, name, "RandomForest")
    
    # Multiple imputation
    multi_rf, variance_rf, imputations_rf = improved_multiple_imputation(df, rf_estimator, name)
    
    # Visualization
    visualize_imputation_results(df, imputed_rf, name, "RandomForest")
    
    # Store results
    dataset_results[f"{name}_RF"] = {
        'imputed': imputed_rf,
        'multi_imputed': multi_rf,
        'metrics': metrics_rf,
        'variance': variance_rf,
        'imputations': imputations_rf
    }
    
    # Save results
    save_imputation_results(
        imputed_rf, multi_rf, imputations_rf, 
        data_type, dataset_type, "rf"
    )
    
    # Process with BayesianRidge
    print("\n\n" + "="*80)
    print(f"Processing {name} with BayesianRidge")
    print("="*80)
    
    # MICE imputation
    imputed_br, imputer_br = perform_mice_imputation(df, br_estimator, name)
    
    # Evaluation with masking
    metrics_br = evaluate_with_masking(df, imputer_br, name, "BayesianRidge")
    
    # Multiple imputation
    multi_br, variance_br, imputations_br = improved_multiple_imputation(df, br_estimator, name)
    
    # Visualization
    visualize_imputation_results(df, imputed_br, name, "BayesianRidge")
    
    # Store results
    dataset_results[f"{name}_BR"] = {
        'imputed': imputed_br,
        'multi_imputed': multi_br,
        'metrics': metrics_br,
        'variance': variance_br,
        'imputations': imputations_br
    }
    
    # Save results
    save_imputation_results(
        imputed_br, multi_br, imputations_br, 
        data_type, dataset_type, "br"
    )
    
    return dataset_results


def print_summary(imputation_results, datasets_processed):
    """Print summary of imputation results and saved files."""
    print("\n\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for df_name, estimator in datasets_processed:
        key = f"{df_name.replace(' ', '_')}_{estimator}"
        metrics = imputation_results[key]['metrics']
        print(f"\n{df_name} with {'RandomForest' if estimator == 'RF' else 'BayesianRidge'}:")
        print(f"MAE: {metrics[0]:.4f}, RMSE: {metrics[1]:.4f}, R²: {metrics[2]:.4f}")
    
    print("\nFiles saved:")
    print("- original_general_benchmarks_imputed_rf.csv")
    print("- original_general_benchmarks_imputed_br.csv")
    print("- original_medical_benchmarks_imputed_rf.csv")
    print("- original_medical_benchmarks_imputed_br.csv")
    
    if APPLY_FILTERING:
        print("- filtered_general_benchmarks_imputed_rf.csv")
        print("- filtered_general_benchmarks_imputed_br.csv")
        print("- filtered_medical_benchmarks_imputed_rf.csv")
        print("- filtered_medical_benchmarks_imputed_br.csv")
        
    print("- Multiple PKL files in imputed_benchmarks/ folder including all individual imputations")
    print("- Multiple visualization files in figures/ folder")


def main():
    """Main function to run the imputation analysis workflow."""
    # Suppress warnings during processing
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Create necessary directories
    create_directories()
    
    # Load the data
    general_df = load_and_process_perf_data(FNAME_PERF_GENERAL, GENERAL_BENCHES)
    medqa_df = load_and_process_perf_data(FNAME_PERF_MEDQA, MED_BENCHES)

    # Preprocess both datasets
    general_df = preprocess_data(general_df)
    medqa_df = preprocess_data(medqa_df)
    
    # Calculate variance of existing data
    original_variance = calculate_existing_data_variance(general_df)
    medqa_variance = calculate_existing_data_variance(medqa_df)
    print(f"overall variance - general: {original_variance['overall_variance']:.6f}")
    print(f"overall variance - medqa: {medqa_variance['overall_variance']:.6f}")

    # Check for invalid values after preprocessing
    print("Invalid values in general_df:", general_df[general_df > 100].count().sum())
    print("Invalid values in medqa_df:", medqa_df[medqa_df > 100].count().sum())

    # Visualize missing data patterns
    visualize_missing_data(general_df, "General Benchmarks")
    visualize_missing_data(medqa_df, "Medical Benchmarks")
    visualize_missing_by_model(general_df, "General Benchmarks")
    visualize_missing_by_model(medqa_df, "Medical Benchmarks")

    # Test missing patterns
    print("Testing missing patterns of General Benchmarks")
    general_results = test_missing_patterns(general_df)
    visualize_missing_pattern_test_results(general_df, general_results)
    
    print("Testing missing patterns of Medical Benchmarks")
    medqa_results = test_missing_patterns(medqa_df)
    visualize_missing_pattern_test_results(medqa_df, medqa_results)
    
    # Initialize filtered dataframes
    general_filtered_df = None
    medqa_filtered_df = None

    if APPLY_FILTERING:
        # Apply the filtering criteria to both dataframes
        general_filtered_df = filter_models_by_valid_data_count(
            general_df, min_valid_count=MIN_VALID_GENERAL, dataset_name="General Benchmarks"
        )
        medqa_filtered_df = filter_models_by_valid_data_count(
            medqa_df, min_valid_count=MIN_VALID_MEDICAL, dataset_name="Medical Benchmarks"
        )

        # Visualize the filtered dataframes
        visualize_missing_data(general_filtered_df, "Filtered General Benchmarks")
        visualize_missing_data(medqa_filtered_df, "Filtered Medical Benchmarks")

        # Print basic info
        print("\nGeneral benchmarks dataset shape:", general_df.shape)
        print("Medical benchmarks dataset shape:", medqa_df.shape)
        print("Filtered general benchmarks dataset shape:", general_filtered_df.shape)
        print("Filtered medical benchmarks dataset shape:", medqa_filtered_df.shape)

        # Calculate missingness statistics
        general_model_miss, general_bench_miss = calculate_missingness_statistics(
            general_df, "General Benchmarks"
        )
        medqa_model_miss, medqa_bench_miss = calculate_missingness_statistics(
            medqa_df, "Medical Benchmarks"
        )
        general_filtered_model_miss, general_filtered_bench_miss = calculate_missingness_statistics(
            general_filtered_df, "Filtered General Benchmarks"
        )
        medqa_filtered_model_miss, medqa_filtered_bench_miss = calculate_missingness_statistics(
            medqa_filtered_df, "Filtered Medical Benchmarks"
        )

    # Define estimators
    rf_estimator = RandomForestRegressor(
        n_estimators=100, 
        random_state=RANDOM_SEED,
        n_jobs=-1  # Use all available cores
    )
    br_estimator = BayesianRidge()
    
    # Prepare datasets for processing
    datasets_to_process = [
        (general_df, "General_Benchmarks", "original"),
        (medqa_df, "Medical_Benchmarks", "original"),
    ]
    
    if APPLY_FILTERING and general_filtered_df is not None and medqa_filtered_df is not None:
        datasets_to_process += [
            (general_filtered_df, "Filtered_General_Benchmarks", "filtered"),
            (medqa_filtered_df, "Filtered_Medical_Benchmarks", "filtered")
        ]
    
    # Dictionary to store all imputation results
    imputation_results = {}
    
    # Track datasets processed for summary
    datasets_processed = []
    
    # Process each dataset
    for df, name, data_type in datasets_to_process:
        results = process_dataset(df, name, data_type, rf_estimator, br_estimator)
        imputation_results.update(results)
        
        # Add to processed list for summary
        if "General" in name:
            display_name = "General Benchmarks" if data_type == "original" else "Filtered General Benchmarks"
        else:
            display_name = "Medical Benchmarks" if data_type == "original" else "Filtered Medical Benchmarks"
            
        datasets_processed.append((display_name, "RF"))
        datasets_processed.append((display_name, "BR"))
    
    # Save all imputation results in one file
    with open(f'{OUTPUT_DIR}/all_imputation_results.pkl', 'wb') as f:
        pickle.dump(imputation_results, f)
    
    # Print summary of results
    print_summary(imputation_results, datasets_processed)


if __name__ == "__main__":
    main()