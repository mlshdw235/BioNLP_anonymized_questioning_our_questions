"""
Medical QA Performance Analysis

This module contains functions for analyzing and visualizing model performance
on medical QA tasks compared to general benchmarks.
"""
import pickle
import re
from collections import Counter, defaultdict
from pprint import pprint
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
from itertools import combinations

# Configuration constants
# Base directory and file paths
DIR_BASE = r'G:\내 드라이브\[1] CCADD N CBDL\[1] Personal Research\2025_MedQACorr'
DIR_BENCHMARK_PERFORMANCE = f'{DIR_BASE}\\medqaperformance\\benchmark_performances(2.8).xlsx'
DIR_CLINICAL_PERFORMANCE = f'{DIR_BASE}\\serach_documents\\fulltext_review_results\\results_fin_250210_추가작업(2.14).xlsx'

# Constants for data analysis
SIMILARITY_THRESHOLD = 0.8  # Threshold for task name similarity
COT_KEYWORD = 'cot'  # Keyword to identify Chain of Thought in memos

# Visualization constants
FIGURE_WIDTH = 1000
FIGURE_HEIGHT = 700
SCALE_FACTOR = 3/5  # For resizing figures
MARKER_SIZE = 8

def preprocess_dataframe(df):
    """Preprocess DataFrame by stripping spaces from string columns."""
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' and 'metric' not in x.name else x)
    return df


def normalize_model_name(df, col_names=('model_full_name2', 'model')):
    """Normalize model name column for consistent formatting."""
    target_col = next((col for col in col_names if col in df.columns), None)
    if not target_col:
        raise ValueError(f"Valid column name not found in DataFrame: {df.columns}")

    df[target_col] = df[target_col].astype(str).apply(
        lambda name: re.sub(r'(\d+)b', r'\1B', name)
                       .replace('Instruct', 'instruct') 
                       if isinstance(name, str) else name
    )
    return df


def load_and_preprocess_data():
    """Load Excel data and preprocess for analysis."""
    df_general = preprocess_dataframe(
        pd.read_excel(DIR_BENCHMARK_PERFORMANCE, sheet_name='general')
    )
    df_medqa = preprocess_dataframe(
        pd.read_excel(DIR_BENCHMARK_PERFORMANCE, sheet_name='medqa')
    )
    df_clinical = preprocess_dataframe(
        pd.read_excel(DIR_CLINICAL_PERFORMANCE)
    )

    df_general = normalize_model_name(df_general, ('model',))
    df_medqa = normalize_model_name(df_medqa, ('model',))
    df_clinical = normalize_model_name(df_clinical, ('model_full_name2',))

    return df_general, df_medqa, df_clinical


def check_duplicate_performance(df, dataset_name):
    """Check for duplicate model performance entries in a dataset."""
    perf_data = defaultdict(list)
    
    for _, row in df.iterrows():
        key = (row['benchmark'], row['model'])
        perf_data[key].append((row['metrics'], row.get('memo', '')))

    duplicates_found = False
    for key, values in perf_data.items():
        perf_data[key] = list(set(values))
        if len(perf_data[key]) > 1:
            if not duplicates_found:
                print(f"\nDuplicate entries found in {dataset_name}:")
                duplicates_found = True
            print(f"Key: {key}")
            pprint(perf_data[key])
            print("=" * 20)


def plot_performance_vs_shots(perf_data, benchmark_name='all'):
    """Plot performance vs. shots with interactive hover information."""
    # Filter data by benchmark if needed
    if not benchmark_name.startswith('all'):
        perf_data = {k: v for k, v in perf_data.items() if k[0] == benchmark_name}
        figsize = (FIGURE_WIDTH * SCALE_FACTOR, FIGURE_HEIGHT * SCALE_FACTOR)
    else:
        figsize = (FIGURE_WIDTH, FIGURE_HEIGHT)

    # Extract performance changes between different shot counts
    perf_changes = []
    for (benchmark, model), perfs in perf_data.items():
        perfs = sorted(set(perfs), key=lambda x: x[1])  # Sort by shot count
        for i in range(len(perfs) - 1):
            shot1, perf1 = perfs[i][1], perfs[i][0]
            shot2, perf2 = perfs[i + 1][1], perfs[i + 1][0]

            try:
                shot1_val = int(shot1[0]) if isinstance(shot1, str) and shot1[0].isdigit() else -1
                shot2_val = int(shot2[0]) if isinstance(shot2, str) and shot2[0].isdigit() else -1
            except ValueError:
                continue

            if shot1_val < shot2_val:
                perf_changes.append((benchmark, model,
                                     shot1_val, shot2_val, perf1, perf2, shot1, shot2))

    if perf_changes:
        fig = go.Figure()
        for benchmark, model, shot1, shot2, perf1, perf2, shot1_ori, shot2_ori in perf_changes:
            fig.add_trace(go.Scatter(
                x=[shot1, shot2],
                y=[perf1, perf2],
                mode='markers+lines',
                marker=dict(size=MARKER_SIZE, color=['green', 'blue']),
                line=dict(color='red', width=1),
                hoverinfo='text',
                text=[f'Benchmark: {benchmark}<br>Model: {model}<br>Perf: {perf1} ({shot1_ori})',
                      f'Benchmark: {benchmark}<br>Model: {model}<br>Perf: {perf2} ({shot2_ori})']
            ))

        fig.update_layout(
            title=f'Performance vs. Shots ({benchmark_name})',
            xaxis_title='Shots',
            yaxis_title='Performance',
            hovermode='closest',
            template='plotly_white',
            showlegend=False,
            width=figsize[0],
            height=figsize[1],
        )
        fig.show()


def plot_outliers_scatter(perf_data, benchmark_name='all', after_normalize=False):
    """Plot interactive scatter plot for detecting outliers in performance data."""
    fig = go.Figure()
    filtered_data = {k: v for k, v in perf_data.items()
                     if benchmark_name == 'all' or k[0] == benchmark_name}
    all_shot_vals = []
    
    for (benchmark, model), perfs in filtered_data.items():
        x_vals = []
        y_vals = []
        text_vals = []
        colors = []
        
        if isinstance(perfs, list):
            for metric, memo in perfs:
                try:
                    metric_val = float(metric)
                    shot_val = int(memo[0]) if memo and memo[0].isdigit() else None
                except (ValueError, IndexError):
                    continue

                if shot_val is None:
                    continue
                    
                all_shot_vals.append(shot_val)
                jitter = np.random.normal(0, 0.05)  # Add slight jitter to x-axis
                x_vals.append(shot_val + jitter)
                y_vals.append(metric_val)
                colors.append('orange' if COT_KEYWORD in memo.lower() else 'blue')
                text_vals.append(
                    f"Benchmark: {benchmark}<br>Model: {model}<br>Metric: {metric_val}<br>Shots: {shot_val}<br>Memo: {memo}"
                )
        else:  # For normalized data (numeric)
            memo = '0-shot (normalized)'
            metric_val = float(perfs)
            shot_val = 0
            all_shot_vals.append(shot_val)
            jitter = np.random.normal(0, 0.05)
            x_vals.append(shot_val + jitter)
            y_vals.append(metric_val)
            colors.append('blue')  # Default color for normalized data
            text_vals.append(
                f"Benchmark: {benchmark}<br>Model: {model}<br>Metric: {metric_val}<br>Shots: {shot_val}<br>Memo: {memo}"
            )
            
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(size=MARKER_SIZE, color=colors, opacity=0.7),
            hoverinfo='text',
            text=text_vals
        ))

    # Determine x-axis range (special case for normalized data)
    x_range = [-2, 2] if all(val == 0 for val in all_shot_vals) else None

    # Set title with normalization info if applicable
    title = f'Outlier Detection Scatter Plot - {benchmark_name}'
    if after_normalize:
        title += "<br>- After normalizing to 0-shot"
        
    fig.update_layout(
        title=title,
        xaxis_title='Shots',
        yaxis_title='Performance Metric',
        hovermode='closest',
        template='plotly_white',
        showlegend=False,
        width=FIGURE_WIDTH * SCALE_FACTOR,
        height=FIGURE_HEIGHT * SCALE_FACTOR,
        xaxis=dict(range=x_range)
    )
    fig.show()


def parse_memo(memo):
    """Parse shot count and CoT status from memo string."""
    if not memo or not isinstance(memo, str):
        return None, False
        
    memo_stripped = memo.strip()
    shot_match = re.match(r'(\d+)', memo_stripped)
    shot = int(shot_match.group(1)) if shot_match else None
    is_cot = COT_KEYWORD in memo_stripped.lower()
    
    return shot, is_cot


def analyze_few_shot_cot_impact(perf_data, benchmark_name='all'):
    """
    Perform regression analysis to estimate the impact of few-shot increase 
    and CoT usage on model performance.
    """
    # Filter data by benchmark
    filtered_data = {
        k: v for k, v in perf_data.items()
        if benchmark_name.startswith('all') or k[0] == benchmark_name
    }

    data_points = []
    for (_, model), perfs in filtered_data.items():
        sorted_perfs = sorted(set(perfs), key=lambda x: x[1])  # Sort by shot count
        for i in range(len(sorted_perfs) - 1):
            metric1, memo1 = sorted_perfs[i]
            metric2, memo2 = sorted_perfs[i + 1]
            
            try:
                shot1 = int(memo1[0]) if memo1 and memo1[0].isdigit() else None
                shot2 = int(memo2[0]) if memo2 and memo2[0].isdigit() else None
                metric1 = float(metric1)
                metric2 = float(metric2)
            except (ValueError, IndexError):
                continue
                
            if shot1 is None or shot2 is None or shot1 >= shot2:
                continue
                
            shot_diff = shot2 - shot1
            perf_diff = metric2 - metric1
            cot1 = COT_KEYWORD in memo1.lower()
            cot2 = COT_KEYWORD in memo2.lower()
            
            if cot1 == cot2:
                cot = 0
            elif not cot1 and cot2:
                cot = 1
            else:
                cot = -1
                
            data_points.append((shot_diff, cot, perf_diff))

    if not data_points:
        print(f"No valid data for benchmark: {benchmark_name}")
        return None

    # Prepare data for regression
    data_points = np.array(data_points)
    x = data_points[:, :2]  # Features: shot_diff, cot
    y = data_points[:, 2]   # Target: perf_diff
    
    # Add intercept for regression
    x = sm.add_constant(x)
    
    # Perform linear regression
    model = sm.OLS(y, x).fit()

    # Print results
    num_samples = len(data_points)
    print(f"\nRegression Analysis for Benchmark: {benchmark_name}")
    print("--------------------------------------------------")
    print(f"Number of data samples used: {num_samples}")
    print(f"Performance gain per additional shot: {model.params[1]:.4f} (p-value: {model.pvalues[1]:.4f})")
    
    if len(model.params) > 2:
        print(f"Average performance impact from CoT (-1: removed, 1: added): {model.params[2]:.4f} (p-value: {model.pvalues[2]:.4f})")
    else:
        print("CoT effect could not be estimated due to lack of variation in CoT usage.")
        
    print(f"R-squared: {model.rsquared:.4f}")
    print("--------------------------------------------------\n")

    return model, num_samples


def compute_regression_slopes_for_benchmark(perf_data, benchmark):
    """Compute regression slopes for a specific benchmark."""
    X_data = []
    Y_data = []
    
    for (bm, model), measurements in perf_data.items():
        if bm != benchmark:
            continue
            
        parsed_measurements = []
        for metric, memo in measurements:
            shot, is_cot = parse_memo(memo)
            if shot is None:
                continue
                
            try:
                perf = float(metric)
            except ValueError:
                continue
                
            parsed_measurements.append((shot, perf, is_cot))
            
        if len(parsed_measurements) < 2:
            continue
            
        parsed_measurements.sort(key=lambda x: x[0])
        for i in range(len(parsed_measurements) - 1):
            shot1, perf1, cot1 = parsed_measurements[i]
            shot2, perf2, cot2 = parsed_measurements[i + 1]
            
            if shot2 > shot1:
                shot_diff = shot2 - shot1
                perf_diff = perf2 - perf1
                delta_cot = (1 if cot2 else 0) - (1 if cot1 else 0)
                X_data.append([shot_diff, delta_cot])
                Y_data.append(perf_diff)
                
    if len(X_data) < 1:
        return 0, 0
        
    X = sm.add_constant(np.array(X_data))
    Y = np.array(Y_data)
    model_reg = sm.OLS(Y, X).fit()
    
    b1 = model_reg.params[1] if len(model_reg.params) > 1 else 0
    b2 = model_reg.params[2] if len(model_reg.params) > 2 else 0
    
    return b1, b2


def normalize_perf_data_to_zero_shot(perf_data, without_normalize=False):
    """
    Normalize performance data to zero-shot equivalent values 
    using regression coefficients.
    """
    final_data = {}
    benchmarks = set(key[0] for key in perf_data.keys())
    
    # Compute regression slopes for each benchmark
    reg_slopes = {}
    for benchmark in benchmarks:
        b1, b2 = compute_regression_slopes_for_benchmark(perf_data, benchmark)
        reg_slopes[benchmark] = (b1, b2)
    
    # Normalize each model's performance
    for (bm, model), measurements in perf_data.items():
        zero_shot_values = []
        nonzero_estimates = []
        
        for metric, memo in measurements:
            shot, is_cot = parse_memo(memo)
            if shot is None:
                continue
                
            try:
                perf_val = float(metric)
            except ValueError:
                continue
                
            if shot == 0:
                zero_shot_values.append(perf_val)
            else:
                b1, b2 = reg_slopes.get(bm, (0, 0))
                if without_normalize:
                    est_zero_shot = perf_val
                else:
                    est_zero_shot = perf_val - shot * b1 - (b2 if is_cot else 0)
                nonzero_estimates.append(est_zero_shot)
                
        if zero_shot_values:
            final_data[(bm, model)] = np.mean(zero_shot_values)
        elif nonzero_estimates:
            final_data[(bm, model)] = np.mean(nonzero_estimates)
    
    return final_data


def create_regression_summary_tables(perf_data_general, perf_data_medqa):
    """
    Create summary tables for regression analysis of few-shot and CoT impact
    on model performance.
    """
    # Define columns for the summary tables
    columns = [
        'Benchmark', 'Shot Performance Gain', 'Shot P-value', 
        'CoT Performance Gain', 'CoT P-value', 'R-squared', 'Sample Size'
    ]
    
    # Initialize lists to store results
    general_results = []
    medqa_results = []
    
    # Analyze general benchmarks
    print("\n=== General Benchmarks Regression Summary ===\n")
    benchmarks_general = set(key[0] for key in perf_data_general.keys())
    for benchmark in benchmarks_general:
        result = analyze_few_shot_cot_impact(perf_data_general, benchmark)
        if result:  # If regression was successful
            model, num_samples = result
            # Extract regression parameters
            shot_gain = model.params[1] if len(model.params) > 1 else 0
            shot_pvalue = model.pvalues[1] if len(model.pvalues) > 1 else 1.0
            cot_gain = model.params[2] if len(model.params) > 2 else 0
            cot_pvalue = model.pvalues[2] if len(model.pvalues) > 2 else 1.0
            r_squared = model.rsquared
            
            # Add to results list
            general_results.append([
                benchmark, 
                shot_gain, 
                shot_pvalue, 
                cot_gain, 
                cot_pvalue, 
                r_squared, 
                num_samples
            ])
    
    # Analyze MedQA benchmarks
    print("\n=== MedQA Benchmarks Regression Summary ===\n")
    benchmarks_medqa = set(key[0] for key in perf_data_medqa.keys())
    for benchmark in benchmarks_medqa:
        result = analyze_few_shot_cot_impact(perf_data_medqa, benchmark)
        if result:  # If regression was successful
            model, num_samples = result
            # Extract regression parameters
            shot_gain = model.params[1] if len(model.params) > 1 else 0
            shot_pvalue = model.pvalues[1] if len(model.pvalues) > 1 else 1.0
            cot_gain = model.params[2] if len(model.params) > 2 else 0
            cot_pvalue = model.pvalues[2] if len(model.pvalues) > 2 else 1.0
            r_squared = model.rsquared
            
            # Add to results list
            medqa_results.append([
                benchmark, 
                shot_gain, 
                shot_pvalue, 
                cot_gain, 
                cot_pvalue, 
                r_squared, 
                num_samples
            ])
    
    # Create DataFrames
    general_df = pd.DataFrame(general_results, columns=columns)
    medqa_df = pd.DataFrame(medqa_results, columns=columns)
    
    # Sort tables by shot performance gain (descending)
    general_df = general_df.sort_values(by='Shot Performance Gain', ascending=False)
    medqa_df = medqa_df.sort_values(by='Shot Performance Gain', ascending=False)
    
    # Format numeric columns
    for df in [general_df, medqa_df]:
        df['Shot Performance Gain'] = df['Shot Performance Gain'].map(lambda x: f"{x:.4f}")
        df['Shot P-value'] = df['Shot P-value'].map(lambda x: f"{x:.4f}")
        df['CoT Performance Gain'] = df['CoT Performance Gain'].map(lambda x: f"{x:.4f}")
        df['CoT P-value'] = df['CoT P-value'].map(lambda x: f"{x:.4f}")
        df['R-squared'] = df['R-squared'].map(lambda x: f"{x:.4f}")
    
    # Print tables
    print("\nGeneral Benchmarks Summary Table:")
    print(general_df.to_string(index=False))
    
    print("\nMedQA Benchmarks Summary Table:")
    print(medqa_df.to_string(index=False))
    
    # Add All summary
    print("\n=== All Benchmarks Combined Regression Summary ===\n")
    analyze_few_shot_cot_impact(perf_data_general, 'all - General')
    analyze_few_shot_cot_impact(perf_data_medqa, 'all - MedQA')
    
    return general_df, medqa_df


def organize_clinical_performance(df):
    """
    Organizes clinical performance data with enhanced metadata and quality checks.
    
    Returns:
        tuple: (task_id_map, organized_data)
            - task_id_map: Dictionary mapping task IDs to metadata
            - organized_data: Dictionary of performance metrics by task
    """
    # Define metadata columns for consistency checks
    metadata_cols = ['task_type', 'therapeutic_area', 'data_source', 'evaluation_type']

    # Check for inconsistent metadata within same task_name
    inconsistencies = []
    for task_name in df['task_name'].unique():
        task_data = df[df['task_name'] == task_name]
        for col in metadata_cols:
            unique_values = task_data[col].unique()
            if len(unique_values) > 1:
                inconsistencies.append({
                    'task_name': task_name,
                    'column': col,
                    'values': unique_values,
                    'examples': task_data[['task_name'] + metadata_cols].head(3).to_dict('records')
                })
                
    # Print inconsistencies if found
    if inconsistencies:
        print("\nFound inconsistent metadata for same task_names:")
        print("=" * 80)
        for i, inc in enumerate(inconsistencies[:10], 1):
            print(f"\n{i}. Task: {inc['task_name']}")
            print(f"Column: {inc['column']}")
            print(f"Different values: {inc['values']}")
            print("\nExample rows:")
            for j, example in enumerate(inc['examples'], 1):
                print(f"Row {j}: {example}")

    # Check for similar task names
    task_names = list(df['task_name'].unique())
    similar_tasks = []
    
    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            score = SequenceMatcher(None, task_names[i].lower(), task_names[j].lower()).ratio()
            if score > SIMILARITY_THRESHOLD:
                similar_tasks.append((task_names[i], task_names[j], score))
                
    # Print similar task names if found
    if similar_tasks:
        print("\nFound similar task names:")
        print("=" * 80)
        for task1, task2, score in sorted(similar_tasks, key=lambda x: x[2], reverse=True):
            print(f"\nSimilarity score {score:.3f}:")
            print(f"Task 1: {task1}")
            print(f"Task 2: {task2}")

    # Create enhanced task ID mapping
    task_id_map = {}
    for i, task_name in enumerate(task_names, 1):
        task_data = df[df['task_name'] == task_name].iloc[0]
        task_id_map[str(i)] = {
            'task_name': task_name,
            'task_type': task_data['task_type'],
            'therapeutic_area': task_data['therapeutic_area'],
            'data_source': task_data['data_source'],
            'evaluation_type': task_data['evaluation_type']
        }
        
    # Create reverse mapping for organizing performance data
    task_name_to_id = {v['task_name']: k for k, v in task_id_map.items()}

    # Initialize the output dictionary
    clinical_perf_data = {}
    
    # Process each unique task
    for task_name in task_names:
        task_id = task_name_to_id[task_name]
        task_data = df[df['task_name'] == task_name]
        
        # Get the reference information
        first_occurrence = task_data.iloc[0]
        reference = (first_occurrence.name, first_occurrence['title'])
        
        # Get metrics information
        metrics = []
        for _, row in task_data.iterrows():
            metric_tuple = (row['model_full_name2'], row['metric_value'], row['sample_size'])
            metrics.append(metric_tuple)
            
        # Organize the data for this task
        clinical_perf_data[task_id] = {
            'reference': reference,
            'metrics': metrics
        }

    return task_id_map, clinical_perf_data


def visualize_clinical_metadata(task_id_map):
    """
    Create comprehensive visualizations for clinical task metadata distributions.
    
    Args:
        task_id_map: Dictionary mapping task IDs to metadata
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Convert task_id_map to DataFrame
    df = pd.DataFrame.from_dict(task_id_map, orient='index')
    metadata_cols = ['task_type', 'therapeutic_area', 'data_source', 'evaluation_type']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15), dpi=600)
    
    # Create two separate GridSpec objects
    gs_top = plt.GridSpec(1, 4, figure=fig, height_ratios=[1], top=0.95, bottom=0.55)
    gs_bottom = plt.GridSpec(2, 3, figure=fig, top=0.45, bottom=0.05)
    
    # Plot individual distributions (pie charts)
    for idx, col in enumerate(metadata_cols):
        ax = fig.add_subplot(gs_top[0, idx])
        counts = df[col].value_counts()
        
        # Calculate percentages for labels
        total = counts.sum()
        labels = [f'{label}\n({count}, {count/total*100:.1f}%)' 
                 for label, count in counts.items()]
        
        wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='',
                                         textprops={'fontsize': 8})
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}',
                    pad=20, fontsize=10)
    
    # Create heatmaps for pairwise relationships
    for idx, (col1, col2) in enumerate(combinations(metadata_cols, 2)):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs_bottom[row, col])
        
        # Create contingency table
        contingency = pd.crosstab(df[col1], df[col2])
        
        # Plot heatmap
        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu',
                   ax=ax, cbar=False)
        ax.set_title(f'{col1.replace("_", " ").title()} vs\n'
                    f'{col2.replace("_", " ").title()}',
                    pad=20, fontsize=10)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Adjust font sizes
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for col in metadata_cols:
        print(f"\n{col.replace('_', ' ').title()}:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            print(f"  {value}: {count} tasks ({count/len(df)*100:.1f}%)")
    
    # Add main title
    plt.suptitle('Clinical Task Metadata Distribution Analysis', 
                fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def process_and_save_data(perf_data_general, perf_data_medqa, clinical_task_id_map, clinical_perf_data):
    """Process and save performance data to pickle files for later use."""
    # Normalize data
    perf_data_general_normalized = normalize_perf_data_to_zero_shot(perf_data_general)
    perf_data_medqa_normalized = normalize_perf_data_to_zero_shot(perf_data_medqa)
    perf_data_general_wonormalized = normalize_perf_data_to_zero_shot(perf_data_general, without_normalize=True)
    perf_data_medqa_wonormalized = normalize_perf_data_to_zero_shot(perf_data_medqa, without_normalize=True)

    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('perf_data_pickle'):
        os.makedirs('perf_data_pickle')

    # Save data to pickle files
    output_files = {
        'perf_data_pickle/clinical_task_id_map.pkl': clinical_task_id_map,
        'perf_data_pickle/perf_data_clinical.pkl': clinical_perf_data,
        'perf_data_pickle/perf_data_general_fin_normalized.pkl': perf_data_general_normalized,
        'perf_data_pickle/perf_data_medqa_fin_normalized.pkl': perf_data_medqa_normalized,
        'perf_data_pickle/perf_data_general_fin_wonormalized.pkl': perf_data_general_wonormalized,
        'perf_data_pickle/perf_data_medqa_fin_wonormalized.pkl': perf_data_medqa_wonormalized
    }
    
    for file_path, data in output_files.items():
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"Data saved to {len(output_files)} pickle files in perf_data_pickle directory.")

def run_comprehensive_analysis():
    """Run comprehensive analysis on all datasets."""
    print("Starting comprehensive medical QA performance analysis...")
    
    # Load and preprocess data
    df_general_perf, df_medqa_perf, df_clinical_perf = load_and_preprocess_data()
    clinical_task_id_map, clinical_perf_data = organize_clinical_performance(df_clinical_perf)
    
    # Visualize clinical metadata
    visualize_clinical_metadata(clinical_task_id_map)
    
    # Extract model names
    model_names_general = df_general_perf['model'].dropna().tolist()
    model_names_medqa = df_medqa_perf['model'].dropna().tolist()
    model_names_clinical = df_clinical_perf['model_full_name2'].dropna().tolist()
    model_names_total = sorted(list(set(model_names_general + model_names_medqa + model_names_clinical)))

    # Print model name frequency
    print("\nModel Names Distribution in Clinical Dataset:")
    pprint(Counter(model_names_clinical))

    # Extract dataset names
    datasets_general = df_general_perf['benchmark'].dropna().tolist()
    datasets_medqa = df_medqa_perf['benchmark'].dropna().tolist()
    
    print("\nGeneral Datasets Distribution:")
    pprint(Counter(datasets_general))
    
    print("\nMedical QA Datasets Distribution:")
    pprint(Counter(datasets_medqa))

    # Check for duplicate performance entries
    check_duplicate_performance(df_general_perf, 'General')
    check_duplicate_performance(df_medqa_perf, 'MedQA')

    # Prepare performance data for visualization
    perf_data_general = defaultdict(list)
    for _, row in df_general_perf.iterrows():
        perf_data_general[(row['benchmark'], row['model'])].append((row['metrics'], row.get('memo', '')))

    perf_data_medqa = defaultdict(list)
    for _, row in df_medqa_perf.iterrows():
        perf_data_medqa[(row['benchmark'], row['model'])].append((row['metrics'], row.get('memo', '')))

    # Plot performance vs shots for both domains
    plot_performance_vs_shots(perf_data_general, 'all - General')
    analyze_few_shot_cot_impact(perf_data_general, 'all - General')
    
    plot_performance_vs_shots(perf_data_medqa, 'all - MedQA')
    analyze_few_shot_cot_impact(perf_data_medqa, 'all - MedQA')

    # Create regression summary tables
    print("\nCreating regression summary tables...")
    general_summary, medqa_summary = create_regression_summary_tables(perf_data_general, perf_data_medqa)
    
    # Save summary tables to CSV
    general_summary.to_csv("regression_summary_general.csv", index=False)
    medqa_summary.to_csv("regression_summary_medqa.csv", index=False)
    
    # Process and save data
    process_and_save_data(perf_data_general, perf_data_medqa, clinical_task_id_map, clinical_perf_data)
    
    # Perform outlier analysis
    print("\nPerforming outlier analysis...")
    # General benchmarks
    for benchmark in set(df_general_perf['benchmark'].dropna()):
        plot_performance_vs_shots(perf_data_general, benchmark)
        analyze_few_shot_cot_impact(perf_data_general, benchmark)
        plot_outliers_scatter(perf_data_general, benchmark)
    
    # MedQA benchmarks    
    for benchmark in set(df_medqa_perf['benchmark'].dropna()):
        plot_performance_vs_shots(perf_data_medqa, benchmark)
        analyze_few_shot_cot_impact(perf_data_medqa, benchmark)
        plot_outliers_scatter(perf_data_medqa, benchmark)
    
    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_comprehensive_analysis()