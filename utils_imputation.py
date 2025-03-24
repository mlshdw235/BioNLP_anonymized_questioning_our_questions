"""Perform imputation on benchmark performances using MICE."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats

# Global constants
OUTPUT_DIR = 'figures'
DEFAULT_ALPHA = 0.05
CORRELATION_THRESHOLD = 0.3  
# Threshold for significant correlation between missingness indicators; values > 0.3 suggest non-MCAR pattern
PREDICTABILITY_THRESHOLD = 0.7  
# AUC threshold for determining if missingness can be predicted from other variables; AUC > 0.7 suggests MAR pattern
SKEW_THRESHOLD = 1.5  
# Threshold for assessing if distribution skewness indicates systematic missingness; |skew| > 1.5 suggests possible MNAR pattern
MIN_SAMPLES_FOR_TEST = 10  
# Minimum samples needed for statistical tests to ensure reliability of t-tests and other statistical measures
MIN_SAMPLES_FOR_PREDICTION = 20  
# Minimum samples needed for machine learning predictability tests; higher than MIN_SAMPLES_FOR_TEST due to ML requirements
HEATMAP_COLORS = ['#ffffff', '#ff5555']  # White for present, red for missing

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_missing_pattern_test_results(df, results):
    """
    Create visualizations to help understand missing data patterns.
    
    Args:
        df: Original DataFrame with missing values
        results: Results from test_missing_patterns
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Heatmap of missing values
    plt.subplot(2, 2, 1)
    missing_mask = df.isna()
    sns.heatmap(missing_mask, cbar=False, cmap=HEATMAP_COLORS)
    plt.title('Missing Data Pattern')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    
    # 2. Correlation heatmap of missingness indicators
    plt.subplot(2, 2, 2)
    if 'correlation_test' in results and 'correlation_matrix' in results['correlation_test']:
        corr_matrix = results['correlation_test']['correlation_matrix']
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={'shrink': .5})
        plt.title('Correlation Between Missingness Indicators')
    else:
        plt.text(0.5, 0.5, 'Correlation data not available', ha='center', va='center')
    
    # 3. Predictability of missingness
    plt.subplot(2, 2, 3)
    if 'predictability_test' in results and 'column_scores' in results['predictability_test']:
        scores = results['predictability_test']['column_scores']
        cols = list(scores.keys())
        vals = list(scores.values())
        
        if cols and vals:
            plt.barh(cols, vals, color='skyblue')
            plt.axvline(x=PREDICTABILITY_THRESHOLD, color='red', linestyle='--', label='Predictability threshold')
            plt.xlim(0.5, 1.0)
            plt.title('Predictability of Missingness (AUC)')
            plt.xlabel('AUC Score')
        else:
            plt.text(0.5, 0.5, 'No predictability scores available', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'Predictability data not available', ha='center', va='center')
    
    # 4. Conclusion
    plt.subplot(2, 2, 4)
    if 'conclusion' in results and results['conclusion']:
        conclusion = results['conclusion']
        plt.axis('off')
        plt.text(0.5, 0.9, f"Missing Data Pattern: {conclusion['pattern']}", 
                 ha='center', va='center', fontsize=14, fontweight='bold')
        plt.text(0.5, 0.8, f"Confidence: {conclusion['confidence']}", 
                 ha='center', va='center', fontsize=12)
        
        y_pos = 0.7
        for line in conclusion['explanation']:
            plt.text(0.1, y_pos, line, ha='left', va='center', fontsize=10)
            y_pos -= 0.05
    else:
        plt.text(0.5, 0.5, 'Conclusion not available', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/missing_pattern_analysis.png', dpi=300)
    plt.show()

def visualize_missing_data(df, title):
    """
    Create a simplified heatmap visualization of missing data patterns.
    
    Args:
        df: pandas DataFrame with missing values
        title: string for the plot title
    """
    ensure_output_directory()
    
    # Create a boolean mask for missing values
    missing_mask = df.isna()

    # Set up the matplotlib figure with reduced width
    plt.figure(figsize=(6, 8))
    
    # Generate a heatmap with the mask and no ytick labels
    ax = sns.heatmap(missing_mask, 
                cmap=HEATMAP_COLORS,
                cbar=False,
                yticklabels=False)
    
    # Add black border around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        
    # Customize the plot
    plt.title(f'Missing Data Pattern: {title}', fontsize=16)
    plt.xlabel('Benchmarks', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f'{title.lower().replace(" ", "_")}_missing_data.png'
    plt.savefig(f'{OUTPUT_DIR}/{plot_filename}', dpi=600)
    plt.show()

    # Print missing data statistics
    print(f"Total model count: {df.shape[0]}")
    missing_percentage = missing_mask.mean() * 100
    print(f"\nMissing Data Percentage for each benchmark in {title}:")
    for col, pct in missing_percentage.items():
        print(f"{col}: {pct:.1f}%")

    # Overall statistics
    total_cells = np.prod(df.shape)
    total_missing = missing_mask.sum().sum()
    print(f"\nTotal missing values: {total_missing} out of {total_cells} ({total_missing/total_cells*100:.1f}%)")

def visualize_missing_by_model(df, title):
    """
    Create a horizontal bar chart of missing benchmarks per model.
    
    Args:
        df: pandas DataFrame with missing values
        title: string for the plot title
    """
    ensure_output_directory()
    
    # Count missing values for each model
    missing_counts = df.isna().sum(axis=1).sort_values(ascending=False)
    
    # Create figure with appropriate dimensions based on number of models
    plt.figure(figsize=(10, max(8, len(missing_counts) * 0.25)))
    
    # Create horizontal bar chart
    ax = sns.barplot(
        x=missing_counts.values,
        y=missing_counts.index,
        palette=['#ff5555'],  # Red bars for consistency with heatmap
        orient='h'
    )
    
    # Add benchmark count annotations to each bar
    for i, count in enumerate(missing_counts.values):
        if count > 0:  # Only annotate non-zero values
            ax.text(
                count + 0.1,
                i,
                f"{count}/{df.shape[1]} ({count/df.shape[1]*100:.1f}%)",
                va='center'
            )
    
    # Customize the plot
    plt.title(f'Missing Benchmarks by Model: {title}', fontsize=16)
    plt.xlabel('Number of Missing Benchmarks', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xlim(0, df.shape[1] + 1)  # Set x-axis limit to number of benchmarks + buffer
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f'{title.lower().replace(" ", "_")}_missing_by_model.png'
    plt.savefig(f'{OUTPUT_DIR}/{plot_filename}', dpi=600)
    plt.show()
    
    # Print summary statistics
    complete_models = (missing_counts == 0).sum()
    total_models = len(missing_counts)
    print(f"\n{complete_models} out of {total_models} models ({complete_models/total_models*100:.1f}%) "
          f"have complete benchmark data in {title}")
    print(f"Average missing benchmarks per model: {missing_counts.mean():.2f} out of {df.shape[1]}")

def filter_models_by_valid_data_count(df, min_valid_count, dataset_name):
    """
    Filter models based on minimum number of valid benchmark results.
    
    Args:
        df: pandas DataFrame with benchmark performance data
        min_valid_count: minimum number of non-NaN benchmark values required
        dataset_name: name of the dataset for reporting
        
    Returns:
        filtered_df: DataFrame containing only models meeting the criteria
    """
    # Count valid (non-NaN) values for each model
    valid_counts = df.notna().sum(axis=1)
    
    # Filter models with at least min_valid_count valid entries
    qualified_models = valid_counts[valid_counts >= min_valid_count].index
    filtered_df = df.loc[qualified_models]
    
    # Print summary statistics
    print(f"\n--- {dataset_name} Dataset Filtering Results ---")
    print(f"Original model count: {len(df)}")
    print(f"Models with at least {min_valid_count} valid benchmarks: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} models")
    
    # Calculate percentage of missing values before and after filtering
    missing_before = df.isna().sum().sum() / np.prod(df.shape) * 100
    missing_after = filtered_df.isna().sum().sum() / np.prod(filtered_df.shape) * 100
    print(f"Missing data percentage before filtering: {missing_before:.2f}%")
    print(f"Missing data percentage after filtering: {missing_after:.2f}%")
    
    return filtered_df

def calculate_missingness_statistics(df, name):
    """
    Calculate and print missingness statistics.
    
    Args:
        df: pandas DataFrame with missing values
        name: name of the dataset for reporting
        
    Returns:
        model_missingness: Series with missingness percentage per model
        benchmark_missingness: Series with missingness percentage per benchmark
    """
    print(f"\n=== Missingness Statistics for {name} ===\n")
    
    # Overall missingness
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    overall_missingness = missing_cells / total_cells * 100
    print(f"Overall missingness: {overall_missingness:.2f}%")
    
    # Model-wise missingness
    model_missingness = df.isna().mean(axis=1) * 100
    print(f"Model-wise missingness: Mean = {model_missingness.mean():.2f}%, Std = {model_missingness.std():.2f}%")
    
    # Benchmark-wise missingness
    benchmark_missingness = df.isna().mean(axis=0) * 100
    print(f"Benchmark-wise missingness: Mean = {benchmark_missingness.mean():.2f}%, Std = {benchmark_missingness.std():.2f}%")
    
    # Top and bottom 5 models by missingness
    top5_models = model_missingness.sort_values(ascending=False).head(5)
    bottom5_models = model_missingness.sort_values(ascending=True).head(5)
    
    print("\nModels with highest missingness:")
    for model, miss in top5_models.items():
        print(f"  {model}: {miss:.2f}%")
    
    print("\nModels with lowest missingness:")
    for model, miss in bottom5_models.items():
        print(f"  {model}: {miss:.2f}%")
    
    # Benchmarks sorted by missingness
    print("\nBenchmarks sorted by missingness (descending):")
    for bench, miss in benchmark_missingness.sort_values(ascending=False).items():
        print(f"  {bench}: {miss:.2f}%")
        
    return model_missingness, benchmark_missingness

def test_missing_patterns(df, alpha=DEFAULT_ALPHA):
    """
    Test if missing values are MCAR, MAR, or MNAR.
    
    Args:
        df: pandas DataFrame with missing values
        alpha: significance level for hypothesis tests
        
    Returns:
        dict: Results of the various missing data tests
    """
    results = {
        'little_mcar_test': None,
        'correlation_test': None,
        'predictability_test': None,
        'mar_var_dependency': None,
        'conclusion': None,
        'details': {}
    }
    
    # Create missing indicators
    missing_indicators = pd.DataFrame(index=df.index)
    for col in df.columns:
        missing_indicators[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Test 1: Little's MCAR test (approximation)
    littles_test_result = littles_mcar_test(df)
    results['little_mcar_test'] = littles_test_result
    results['details']['littles_test'] = littles_test_result
    
    # Test 2: Correlation between missingness indicators
    corr_test_result = test_missingness_correlation(missing_indicators)
    results['correlation_test'] = corr_test_result
    results['details']['correlation_test'] = corr_test_result
    
    # Test 3: Predictability of missingness
    predict_test_result = test_missingness_predictability(df, missing_indicators)
    results['predictability_test'] = predict_test_result
    results['details']['predictability_test'] = predict_test_result
    
    # Test 4: Test if missingness depends on observed values in other variables (MAR)
    mar_test_result = test_mar_dependency(df)
    results['mar_var_dependency'] = mar_test_result
    results['details']['mar_dependency'] = mar_test_result
    
    # Summarize the results to determine the most likely missing pattern
    results['conclusion'] = conclude_missing_pattern(results, alpha)
    
    return results

def littles_mcar_test(df):
    """
    Implementation of Little's MCAR test (simplified version).
    
    The full implementation of Little's test is complex. This is a simplified version
    that approximates the test by comparing the means of cases with and without missing values.
    
    Args:
        df: pandas DataFrame with missing values
        
    Returns:
        dict: Test results including p-value and conclusion
    """
    result = {
        'test_statistic': 0,
        'p_value': 1.0,
        'is_mcar': True,
        'details': {}
    }
    
    # Create a copy of the dataframe to work with
    data = df.copy()
    
    # For each variable with missing data
    p_values = []
    
    for col in data.columns:
        if data[col].isna().sum() > 0:
            # Get the mask for missing values in this column
            missing_mask = data[col].isna()
            
            # Skip if all values are missing or no values are missing
            if missing_mask.all() or not missing_mask.any():
                continue
                
            # Compare other variables for cases with and without missing values in this column
            col_results = {}
            
            for other_col in data.columns:
                if other_col != col:
                    # Get values where col is missing and not missing
                    vals_when_missing = data.loc[missing_mask, other_col].dropna()
                    vals_when_not_missing = data.loc[~missing_mask, other_col].dropna()
                    
                    # Skip if not enough data points
                    if len(vals_when_missing) < MIN_SAMPLES_FOR_TEST or len(vals_when_not_missing) < MIN_SAMPLES_FOR_TEST:
                        continue
                    
                    # Perform t-test to compare means
                    try:
                        t_stat, p_val = stats.ttest_ind(
                            vals_when_missing, 
                            vals_when_not_missing,
                            equal_var=False  # Use Welch's t-test
                        )
                        p_values.append(p_val)
                        col_results[other_col] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'mean_when_missing': vals_when_missing.mean(),
                            'mean_when_not_missing': vals_when_not_missing.mean()
                        }
                    except:
                        # Skip if t-test fails
                        continue
            
            result['details'][col] = col_results
    
    # Combine p-values using Fisher's method
    if p_values:
        k = len(p_values)
        fisher_stat = -2 * sum(np.log(p_values))
        combined_p_value = 1 - stats.chi2.cdf(fisher_stat, 2*k)
        
        result['test_statistic'] = fisher_stat
        result['p_value'] = combined_p_value
        result['is_mcar'] = combined_p_value > DEFAULT_ALPHA
    
    return result

def test_missingness_correlation(missing_indicators):
    """
    Test if missingness indicators are correlated with each other.
    
    Significant correlations between missingness indicators suggest
    the data is not MCAR.
    
    Args:
        missing_indicators: DataFrame of binary indicators for missingness
        
    Returns:
        dict: Test results including correlation matrix and conclusion
    """
    result = {
        'significant_correlations': 0,
        'total_correlations': 0,
        'max_correlation': 0,
        'is_indicative_of_mcar': True,
        'correlation_matrix': None
    }
    
    # Calculate correlation matrix
    corr_matrix = missing_indicators.corr()
    result['correlation_matrix'] = corr_matrix
    
    # Count significant correlations (using absolute correlation > CORRELATION_THRESHOLD as a heuristic)
    # Exclude diagonal elements
    n_cols = len(corr_matrix.columns)
    significant_corr = 0
    total_corr = 0
    max_corr = 0
    
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > max_corr:
                max_corr = corr_val
            
            if corr_val > CORRELATION_THRESHOLD:
                significant_corr += 1
            
            total_corr += 1
    
    result['significant_correlations'] = significant_corr
    result['total_correlations'] = total_corr
    result['max_correlation'] = max_corr
    
    # If more than 10% of correlations are significant, likely not MCAR
    threshold = 0.1 * total_corr
    result['is_indicative_of_mcar'] = significant_corr <= threshold
    
    return result

def test_missingness_predictability(df, missing_indicators, threshold=PREDICTABILITY_THRESHOLD):
    """
    Test if missingness can be predicted from observed values, which
    would suggest MAR rather than MCAR.
    
    Args:
        df: Original DataFrame with missing values
        missing_indicators: DataFrame of binary indicators for missingness
        threshold: AUC threshold for considering missingness predictable
        
    Returns:
        dict: Test results including predictability scores
    """
    result = {
        'predictable_columns': 0,
        'total_columns_tested': 0,
        'max_auc_score': 0,
        'column_scores': {},
        'is_indicative_of_mar': False
    }
    
    predictable_cols = 0
    total_cols_tested = 0
    max_auc = 0
    
    # For each column with missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            missing_indicator = missing_indicators[f'{col}_missing']
            
            # If too few missings or non-missings, skip
            if missing_indicator.sum() < MIN_SAMPLES_FOR_TEST or (len(missing_indicator) - missing_indicator.sum()) < MIN_SAMPLES_FOR_TEST:
                continue
                
            # Prepare data: use all other columns to predict missingness
            X = df.drop(columns=[col]).copy()
            
            # Remove columns with missing values for this specific test
            X = X.loc[~missing_indicator.index.isin(df[df[col].isna()].index)]
            y = missing_indicator.loc[~missing_indicator.index.isin(df[df[col].isna()].index)]
            
            # If too little data remains, skip
            if len(X) < MIN_SAMPLES_FOR_PREDICTION:
                continue
                
            total_cols_tested += 1
            
            try:
                # Use RandomForest to check if missingness is predictable
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(clf, X, y, cv=min(5, len(X)//10), scoring='roc_auc')
                
                # Calculate mean AUC score
                mean_auc = np.mean(scores)
                
                result['column_scores'][col] = mean_auc
                
                if mean_auc > max_auc:
                    max_auc = mean_auc
                
                # If AUC is above threshold, consider missingness predictable
                if mean_auc > threshold:
                    predictable_cols += 1
            except:
                # Skip if prediction fails
                continue
    
    result['predictable_columns'] = predictable_cols
    result['total_columns_tested'] = total_cols_tested
    result['max_auc_score'] = max_auc
    
    # If any column's missingness is highly predictable, suggest MAR
    result['is_indicative_of_mar'] = predictable_cols > 0
    
    return result

def test_mar_dependency(df):
    """
    Test if missingness in each variable depends on observed values
    of other variables (MAR).
    
    Args:
        df: DataFrame with missing values
        
    Returns:
        dict: Test results for MAR dependency
    """
    result = {
        'variable_dependencies': {},
        'columns_with_mar_pattern': 0,
        'total_columns_tested': 0,
        'is_indicative_of_mar': False
    }
    
    # For each column with missing values
    mar_columns = 0
    tested_columns = 0
    
    for col in df.columns:
        # Skip if no missing values
        if df[col].isna().sum() == 0:
            continue
            
        tested_columns += 1
        col_result = {'dependent_variables': []}
        
        # Test if missingness in col depends on values in other columns
        missing_mask = df[col].isna()
        
        for other_col in df.columns:
            if other_col != col:
                # Skip if other_col has too many missing values
                if df[other_col].isna().mean() > 0.5:  # Skip if more than 50% missing
                    continue
                
                # Get values of other_col where col is missing vs. not missing
                vals_when_missing = df.loc[missing_mask, other_col].dropna()
                vals_when_not_missing = df.loc[~missing_mask, other_col].dropna()
                
                # Skip if not enough data
                if len(vals_when_missing) < MIN_SAMPLES_FOR_TEST or len(vals_when_not_missing) < MIN_SAMPLES_FOR_TEST:
                    continue
                
                # Perform statistical test to check dependency
                try:
                    # For continuous variables
                    t_stat, p_val = stats.ttest_ind(
                        vals_when_missing, 
                        vals_when_not_missing,
                        equal_var=False
                    )
                    
                    if p_val < DEFAULT_ALPHA:
                        col_result['dependent_variables'].append({
                            'variable': other_col,
                            'p_value': p_val,
                            'mean_when_missing': vals_when_missing.mean(),
                            'mean_when_not_missing': vals_when_not_missing.mean()
                        })
                except:
                    # Skip if test fails
                    continue
        
        # If missingness depends on at least one other variable, suggest MAR
        if len(col_result['dependent_variables']) > 0:
            mar_columns += 1
            
        result['variable_dependencies'][col] = col_result
    
    result['columns_with_mar_pattern'] = mar_columns
    result['total_columns_tested'] = tested_columns
    result['is_indicative_of_mar'] = mar_columns > 0
    
    return result

def test_mnar_pattern(df):
    """
    Test for MNAR pattern by checking correlation between values and missingness 
    within the same variable.
    
    This is challenging to test directly, but we can check for patterns
    suggesting MNAR, such as systematic missingness at high or low values.
    
    Args:
        df: DataFrame with missing values
        
    Returns:
        dict: Test results for MNAR pattern
    """
    result = {
        'variable_tests': {},
        'columns_with_mnar_pattern': 0,
        'total_columns_tested': 0,
        'is_indicative_of_mnar': False
    }
    
    mnar_columns = 0
    tested_columns = 0
    
    # For each column with missing values
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
            
        # We can't directly test MNAR (since we don't have the missing values)
        # But we can look for patterns suggestive of MNAR
        
        # Look at distribution of non-missing values
        non_missing_vals = df[col].dropna()
        
        # Skip if too few values
        if len(non_missing_vals) < MIN_SAMPLES_FOR_PREDICTION:
            continue
            
        tested_columns += 1
        
        # Check for truncation pattern (e.g., all values below a threshold are missing)
        percentile_5 = non_missing_vals.quantile(0.05)
        percentile_95 = non_missing_vals.quantile(0.95)
        
        col_result = {
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'mean': non_missing_vals.mean(),
            'median': non_missing_vals.median(),
            'suggest_mnar': False,
            'truncation_pattern': False
        }
        
        # Check if distribution looks truncated (suggesting MNAR)
        # This is a heuristic - compare the 5% and 95% percentiles to the min and max
        skew = non_missing_vals.skew()
        
        if abs(skew) > SKEW_THRESHOLD:
            col_result['suggest_mnar'] = True
            col_result['truncation_pattern'] = True
            mnar_columns += 1
        
        result['variable_tests'][col] = col_result
    
    result['columns_with_mnar_pattern'] = mnar_columns
    result['total_columns_tested'] = tested_columns
    result['is_indicative_of_mnar'] = mnar_columns > 0
    
    return result

def conclude_missing_pattern(results, alpha=DEFAULT_ALPHA):
    """
    Determine the most likely missing data pattern based on test results.
    
    Args:
        results: Dict containing results of all missing data tests
        alpha: Significance level
        
    Returns:
        dict: Conclusion and explanation
    """
    conclusion = {
        'pattern': None,
        'confidence': None,
        'explanation': []
    }
    
    # Evidence for each pattern
    mcar_evidence = []
    mar_evidence = []
    mnar_evidence = []
    
    # Check Little's MCAR test
    if results['little_mcar_test']['p_value'] > alpha:
        mcar_evidence.append(f"Little's MCAR test p-value ({results['little_mcar_test']['p_value']:.4f}) > {alpha}")
    else:
        mar_evidence.append(f"Little's MCAR test p-value ({results['little_mcar_test']['p_value']:.4f}) <= {alpha}")
    
    # Check correlation test
    if results['correlation_test']['is_indicative_of_mcar']:
        mcar_evidence.append("Low correlation between missingness indicators")
    else:
        mar_evidence.append("Significant correlation between missingness indicators")
    
    # Check predictability test
    if results['predictability_test']['is_indicative_of_mar']:
        mar_evidence.append("Missingness can be predicted from observed values")
    else:
        mcar_evidence.append("Missingness is not predictable from observed values")
    
    # Check MAR dependency test
    if results['mar_var_dependency']['is_indicative_of_mar']:
        mar_evidence.append("Missingness depends on observed values in other variables")
    else:
        mcar_evidence.append("Missingness doesn't depend on other observed values")
    
    # Determine most likely pattern
    mcar_score = len(mcar_evidence)
    mar_score = len(mar_evidence)
    mnar_score = len(mnar_evidence)
    
    # MNAR is hardest to test directly, so we infer it if not MCAR or MAR
    # For simplicity, we're assuming if not clearly MCAR or MAR, it might be MNAR
    if mcar_score > mar_score and mcar_score > mnar_score:
        conclusion['pattern'] = 'MCAR'
        conclusion['confidence'] = 'high' if mcar_score >= 3 else 'medium'
    elif mar_score > mcar_score and mar_score > mnar_score:
        conclusion['pattern'] = 'MAR'
        conclusion['confidence'] = 'high' if mar_score >= 3 else 'medium'
    else:
        # Default to MNAR if evidence is inconclusive
        conclusion['pattern'] = 'MNAR (tentative)'
        conclusion['confidence'] = 'low'
        conclusion['explanation'].append("MNAR is inferred due to lack of strong evidence for MCAR or MAR")
    
    # Add evidence to explanation
    if mcar_evidence:
        conclusion['explanation'].append("Evidence for MCAR:")
        conclusion['explanation'].extend([f"- {e}" for e in mcar_evidence])
    
    if mar_evidence:
        conclusion['explanation'].append("Evidence for MAR:")
        conclusion['explanation'].extend([f"- {e}" for e in mar_evidence])
    
    if mnar_evidence:
        conclusion['explanation'].append("Evidence for MNAR:")
        conclusion['explanation'].extend([f"- {e}" for e in mnar_evidence])
    
    return conclusion
