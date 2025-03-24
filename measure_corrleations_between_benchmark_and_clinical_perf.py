"""measure_corrleations_between_benchmark_and_clinical_perf_with_bias_consideration.py"""
import pickle

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from utils_measure_correlations import (
    load_benchmark_correlations,
    load_datasets,
    analyze_model_overlap,
    apply_benchmark_filtering,
    calculate_benchmark_correlations,
    print_analysis_summary,
    create_task_metadata_visualizations,
    calculate_avg_benchmark_correlations
)


USE_IMPUTATED = False
WEIGHTING_BY_SAMPLE_SIZE = True
TWO_CORR_MEASURE = False
PLOT_MAIN_AS_PANELS = False
PLOT_CATEGORY_AS_PANELS = True
WITHOUT_NUM_MODELS = True
TA_CLASSIFICATION = {
    "general internal medicine": [
        "general medicine",
        "infectious diseases",
        "endocrinology & metabolic disorders"
    ],
    "cardiopulmonary & renal medicine": [
        "cardiology",
        "pulmonology",
        "nephrology",
        "hematology",
        "gastroenterology & hepatology"
    ],
    "neurology & psychiatry": [
        "neurology",
        "psychiatric"
    ],
    "orthopedic & procedural medicine": [
        "orthopedics & musculoskeletal",
        "dentistry",
        "otolaryngology"
    ],
    "dermatology & imaging": [
        "dermatology",
        "radiology"
    ],
    "oncology": [
        "oncology"
    ],
    "emergency & critical care": [
        "emergency medicine"
    ],
    "pediatrics & genetic disorders": [
        "pediatrics",
        "genetic disorders"
    ]
}

def __init__():
    """Initialize configuration and load datasets"""
    # Configuration constants
    config = {
        'FOR_VALID_TASKS_AND_PERFS': False,
        'FOR_STRICTLY_VALID_TASKS_AND_PERFS': False,
        'THRESHOLD_BENCHMARK_COVERAGE': 0.5,
        'DIR_CLINICAL_PERF': r'G:\내 드라이브\[1] CCADD N CBDL\[1] Personal Research\2025_MedQACorr\serach_documents\fulltext_review_results'
    }
    
    # Load datasets
    datasets = load_datasets(config, use_imputated=USE_IMPUTATED)
    datasets['general_df'] = \
        datasets['general_df'][['MMLU', 'MMLU Pro', 'BBH', 
                                # 'ARC-Challenge',
                                'HumanEval',
                                'GSM8K', 'MATH', 
                                # 'HellaSwag'
                                ]]
    # Analyze model overlap
    analyze_model_overlap(datasets)
    
    # Apply filtering if needed
    datasets = apply_benchmark_filtering(datasets, config)
    
    # Calculate correlations
    correlation_results = calculate_all_correlations(datasets, config)
    
    # Generate visualizations
    if PLOT_MAIN_AS_PANELS:
        create_combined_panel_visualizations(correlation_results, config)
    else:
        horizontal_line_data = create_visualizations(correlation_results, config)
        create_visualizations_with_bayesian_modeling(correlation_results, config, horizontal_line_data)
    
    # # Generate task metadata visualizations
    # create_task_metadata_visualizations(correlation_results, datasets, config,
    #                                     use_imputated=USE_IMPUTATED,
    #                                     weighting_by_sample_size=WEIGHTING_BY_SAMPLE_SIZE,
    #                                     plot_category_as_panel=PLOT_CATEGORY_AS_PANELS,
    #                                     two_corr_measure=TWO_CORR_MEASURE,
    #                                     ta_classification=TA_CLASSIFICATION)

def calculate_all_correlations(datasets, config):
    """Calculate correlations between clinical performance and benchmarks"""
    print("\n=== Correlation Analysis ===")
    
    clinical_tasks = datasets['clinical_tasks']
    clinical_perf_df = datasets['clinical_perf_df']
    general_df = datasets['general_df']
    medqa_df = datasets['medqa_df']
    general_models = datasets['general_models']
    medqa_models = datasets['medqa_models']
    is_valid_benchmark = datasets['is_valid_benchmark']
    get_consistent_models = datasets['get_consistent_models']
    
    print(f"Starting with {len(clinical_tasks)} unique clinical tasks")
    
    # Prepare to store results
    correlation_results = []
    
    # For analysis summary
    analysis_stats = {
        'excluded_tasks': 0,
        'included_tasks': 0,
        'total_model_samples': 0,
        'task_model_counts': [],
        'general_valid_model_counts': [],
        'medqa_valid_model_counts': []
    }
    
    for task in clinical_tasks:
        # Get clinical performance for this task
        task_df = clinical_perf_df[clinical_perf_df['task_name'] == task]
        
        # Check if we have enough models for this task
        if len(task_df) <= 2:
            analysis_stats['excluded_tasks'] += 1
            continue
        
        # Check which models we have benchmark data for
        task_models = set(task_df['model_name'])
        
        # Models with data in both clinical task and general benchmark
        general_valid_models = task_models.intersection(general_models)
        
        # Models with data in both clinical task and MedQA benchmark
        medqa_valid_models = task_models.intersection(medqa_models)
        
        if len(general_valid_models) <= 2 and len(medqa_valid_models) <= 2:
            analysis_stats['excluded_tasks'] += 1
            continue
        
        # Store counts for summary
        analysis_stats['task_model_counts'].append(len(task_df))
        analysis_stats['general_valid_model_counts'].append(len(general_valid_models))
        analysis_stats['medqa_valid_model_counts'].append(len(medqa_valid_models))
            
        analysis_stats['included_tasks'] += 1
        analysis_stats['total_model_samples'] += len(task_df)
        
        # Create model-to-performance mapping for this clinical task
        clinical_perf_map = dict(zip(task_df['model_name'], task_df['metric_value']))
        
        # Get sample size for this task
        sample_size = None
        if WEIGHTING_BY_SAMPLE_SIZE:
            # Get the sample size for this task
            sample_sizes = task_df['sample_size'].unique()
            if len(sample_sizes) == 1:
                sample_size = sample_sizes[0]
            else:
                # If there are multiple sample sizes, use the median
                sample_size = task_df['sample_size'].median()
        
        # Calculate correlations with general benchmarks
        correlation_results.extend(
            calculate_benchmark_correlations(
                task, 'general', general_df, clinical_perf_map, 
                general_valid_models, is_valid_benchmark, get_consistent_models,
                config, sample_size
            )
        )
        
        # Calculate correlations with MedQA benchmarks
        correlation_results.extend(
            calculate_benchmark_correlations(
                task, 'medqa', medqa_df, clinical_perf_map, 
                medqa_valid_models, is_valid_benchmark, get_consistent_models,
                config, sample_size
            )
        )
    
    # Convert results to dataframe
    results_df = pd.DataFrame(correlation_results)
    
    # Print analysis summaries
    print_analysis_summary(analysis_stats, results_df, config,
                           weighting_by_sample_size=WEIGHTING_BY_SAMPLE_SIZE)
    
    # Save results
    save_path = save_results(results_df, config)
    print(f"\nResults saved to {save_path}")
    
    return results_df

def save_results(results_df, config):
    """Save correlation results to file"""
    if config['FOR_STRICTLY_VALID_TASKS_AND_PERFS']:
        output_file = 'correlation_analysis_results_strictly_filtered'
    elif config['FOR_VALID_TASKS_AND_PERFS']:
        output_file = 'correlation_analysis_results_filtered'
    else:
        output_file = 'correlation_analysis_results'
        
    if USE_IMPUTATED:
        output_file += '_imputated'
        
    if WEIGHTING_BY_SAMPLE_SIZE:
        output_file += '_weighted'
        
    output_file += '.csv'
    
    results_df.to_csv(output_file, index=False)
    return output_file

def create_combined_panel_visualizations(results_df, config):
    """
    Create a combined panel visualization with regular correlation results 
    and Bayesian modeling in a vertical layout (A and B panels).
    
    Args:
        results_df: DataFrame with correlation results
        config: Configuration dictionary
    """
    if len(results_df) == 0:
        print("No results to visualize")
        return
    
    # Load benchmark correlations
    corr_data = load_benchmark_correlations('final_dataset/correlations_fin_250228.json')
    if corr_data is None:
        print("Warning: Could not load benchmark correlations data. Proceeding without comparison lines.")
    
    # Load Bayesian benchmark correlations from pickle file
    try:
        with open('bayesian_analysis_results/benchmark_correlation_results.pkl', 'rb') as f:
            bayesian_corr_data = pickle.load(f)
        print("Successfully loaded Bayesian correlation data")
    except Exception as e:
        print(f"Failed to load Bayesian correlation data: {e}")
        bayesian_corr_data = None
    
    # Set figure parameters with increased size
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    
    # Determine which correlation measures to use based on TWO_CORR_MEASURE flag
    if TWO_CORR_MEASURE:
        corr_measures = ['spearman', 'kendall']
        n_cols = 2
        figsize = (22, 14)  # Increased size for panel layout
    else:
        corr_measures = ['pearson', 'spearman', 'kendall', 'lin_ccc']
        n_cols = 2
        figsize = (24, 32)  # 2x2 layout for 4 measures
    
    # Create figure with gridspec for panel layout
    fig = plt.figure(figsize=figsize)
    
    # Create main gridspec with 2 rows (for A and B panels)
    main_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.72)
    
    # Create nested gridspecs for plots in each panel
    wspace = 0.1
    panel_a_gs = main_gs[0].subgridspec(1, n_cols, wspace=wspace)
    panel_b_gs = main_gs[1].subgridspec(1, n_cols, wspace=wspace)
    
    # Define colors for benchmark types - reversed order to make Medical QA appear first
    colors = {'medqa': 'forestgreen', 'general': 'royalblue'}

    # Clean benchmark names by removing (medqa) and (general) suffixes
    def clean_benchmark_name(name):
        name = name.replace("(medqa)", "").replace("(general)", "").strip()
        return name
    
    # Helper function to map benchmark names from results_df to bayesian_corr_data format
    def map_benchmark_to_bayesian_key(b_type, b_name):
        clean_name = clean_benchmark_name(b_name)
        prefix = "general_" if b_type == "general" else "medical_"
        return f"{prefix}{clean_name}"
    
    # Prepare horizontal line data storage
    horizontal_line_data = {measure: {} for measure in corr_measures}
    
    # Create axes for all panels first
    panel_a_axes = [fig.add_subplot(panel_a_gs[0, i]) for i in range(n_cols)]
    panel_b_axes = [fig.add_subplot(panel_b_gs[0, i]) for i in range(n_cols)]
    
    # Get the position of the leftmost subplot in each panel for title positioning
    left_a_pos = panel_a_axes[0].get_position()
    left_b_pos = panel_b_axes[0].get_position()
    
    # Add panel titles using the position of the leftmost subplot
    fontsize_panel = 18
    fig.text(left_a_pos.x0 - 0.04, left_a_pos.y1 + 0.04,
             "(A) Using the Originally Reported Clinical Performances",
             fontsize=fontsize_panel, fontweight='bold')
    
    fig.text(left_b_pos.x0 - 0.04, left_b_pos.y1 + 0.04,
             "(B) Using Representative Clinical Performances of Each Model Estimated via Bayesian Modeling",
             fontsize=fontsize_panel, fontweight='bold')
    
    # Store avg_corr_legend_elements for later use
    avg_corr_legend_elements = []
    
    # Plot Panel A - Standard Correlation Results
    for i, measure in enumerate(corr_measures):
        ax_a = panel_a_axes[i]
        
        # Get the plot elements and clean x-axis labels
        legend_elements_a, measure_h_lines = plot_correlation_measure(
            results_df, 
            measure, 
            colors, 
            ax_a, 
            corr_data, 
            clean_benchmark_func=clean_benchmark_name
        )
        
        # Store horizontal line data for this measure
        horizontal_line_data[measure] = measure_h_lines
        
        # Store avg_corr_legend_elements for the legend
        if i == 0:
            avg_corr_legend_elements = legend_elements_a
        
        # Set x-axis label
        ax_a.set_xlabel("Benchmarks", fontsize=14)
        ax_a.set_ylabel("Correlation Coefficient", fontsize=14)
        
        # Rotate x-tick labels and increase font size
        ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=30, ha='right', fontsize=12)
        ax_a.tick_params(axis='y', labelsize=12)
        
        # Add title with increased font size
        if 'kendall' in measure.lower():
            title_str = "Kendall's Tau Correlation"
        elif 'lin' in measure.lower():
            title_str = "Lin's CCC"
        else:
            title_str = f"{measure.capitalize()} Correlation"
        ax_a.set_title(title_str, fontsize=16)
    
    # Plot Panel B - Bayesian Correlation Results
    if bayesian_corr_data:
        # Plot individual correlation measures in Panel B
        for i, measure in enumerate(corr_measures):
            ax_b = panel_b_axes[i]
            
            # Get horizontal line data for this measure
            measure_h_lines = horizontal_line_data.get(measure, {})
            print(measure)
            print(measure_h_lines)
            
            # Plot with Bayesian data but original horizontal lines
            _ = plot_with_bayesian_data(
                results_df, 
                measure, 
                colors, 
                ax_b, 
                measure_h_lines,
                bayesian_corr_data,
                map_benchmark_to_bayesian_key,
                clean_benchmark_func=clean_benchmark_name
            )
            
            # Set x-axis label (without bold)
            ax_b.set_xlabel("Benchmarks", fontsize=14)
            ax_b.set_ylabel("Correlation Coefficient", fontsize=14)
            
            # Rotate x-tick labels and increase font size
            ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=30, ha='right', fontsize=12)
            ax_b.tick_params(axis='y', labelsize=12)
            
            # Add title with increased font size (without bold)
            if 'kendall' in measure.lower():
                title_str = "Kendall's Tau Correlation"
            elif 'lin' in measure.lower():
                title_str = "Lin's CCC"
            else:
                title_str = f"{measure.capitalize()} Correlation"
            ax_b.set_title(title_str, fontsize=16)
    else:
        # If no Bayesian data, display message
        for i in range(n_cols):
            ax_b = panel_b_axes[i]
            ax_b.text(0.5, 0.5, "Bayesian data not available", ha='center', va='center', fontsize=16)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
    
    # Create benchmark type legend items - reversed order to make Medical QA appear first
    benchmark_legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors['medqa'], label='Medical QA'),
        plt.Rectangle((0, 0), 1, 1, color=colors['general'], label='General'),
        Line2D([0], [0], color=colors['medqa'], lw=3.5, alpha=0.5, label='Medical QA Reference'),
        Line2D([0], [0], color=colors['general'], lw=3.5, alpha=0.5, label='General Reference')
    ]
    
    # Calculate position for legends
    mm_to_inches = 1 / 25.4
    offset_inches = 1.5 * mm_to_inches
    fig_width_inches = fig.get_figwidth()
    legend_x_pos = 0.905 + (offset_inches / fig_width_inches)
    
    # Determine which correlation measures to use based on TWO_CORR_MEASURE flag
    if TWO_CORR_MEASURE:
        legend_y_position_1 = 0.27
        legend_y_position_2 = 0.19
    else:
        legend_y_position_1 = 0.13
        legend_y_position_2 = 0.05
    print(f"legend_y_position_1: {legend_y_position_1}")
    print(f"legend_y_position_2: {legend_y_position_2}")
    
    # Add the first legend for benchmark types
    first_legend = fig.legend(
        handles=benchmark_legend_elements[:2],  # Only the benchmark type rectangles
        loc='upper left',
        bbox_to_anchor=(legend_x_pos, legend_y_position_1),
        ncol=1,
        fontsize=12,
        title="Benchmark Types",
        title_fontsize=13,
        alignment='left'
    )
    
    # Manually set the alignment of the legend title to left
    first_legend._legend_box.align = "left"
    
    # Add the second legend for average correlation
    if avg_corr_legend_elements:
        avg_corr_legend_elements = sorted(avg_corr_legend_elements,
                                          key=lambda line: line.get_label(), reverse=True)
        second_legend = fig.legend(
            handles=avg_corr_legend_elements,
            loc='upper left',
            bbox_to_anchor=(legend_x_pos, legend_y_position_2),
            ncol=1,
            fontsize=12,
            title="Average Correlation\nwith Benchmarks",
            title_fontsize=13,
            alignment='left'
        )
        
        # Manually set the alignment of the legend title to left
        second_legend._legend_box.align = "left"
    
    # Add the first legend to the figure
    fig.add_artist(first_legend)
    
    # Adjust layout to accommodate the legends
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    # Save plot in multiple formats
    if config['FOR_STRICTLY_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_panel_plot_strictly_filtered'
    elif config['FOR_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_panel_plot_filtered'
    else:
        plot_filename_base = 'correlation_panel_plot'
        
    if USE_IMPUTATED:
        plot_filename_base += '_imputated'
        
    if WEIGHTING_BY_SAMPLE_SIZE:
        plot_filename_base += '_weighted'
        
    if TWO_CORR_MEASURE:
        plot_filename_base += '_two_measures'
    
    # Save in different formats
    formats = {
        'png': {'dpi': 600},
        'tiff': {'dpi': 600},
        'pdf': {'dpi': 600, 'bbox_inches': 'tight', 'format': 'pdf'}
    }
    
    for fmt, params in formats.items():
        filename = f'figures/{plot_filename_base}.{fmt}'
        plt.savefig(filename, **params)
        print(f"Plot saved to {filename}")
    plt.show()
    plt.close()

def create_visualizations(results_df, config):
    """
    Create visualizations for correlation results with benchmark comparisons
    
    Returns:
        Dictionary containing horizontal line data for each measure and benchmark
    """
    if len(results_df) == 0:
        print("No results to visualize")
        return {}
        
    # Load benchmark correlations
    corr_data = load_benchmark_correlations('final_dataset/correlations_fin_250228.json')
    if corr_data is None:
        print("Warning: Could not load benchmark correlations data. Proceeding without comparison lines.")
        
    # Set figure width to 1.2 times the original width
    width_factor = 1.2
    
    # Determine which correlation measures to use based on TWO_CORR_MEASURE flag
    if TWO_CORR_MEASURE:
        corr_measures = ['spearman', 'kendall']
        fig, axes = plt.subplots(1, 2, figsize=(20 * width_factor, 8))
        axes = axes.flatten()
    else:
        corr_measures = ['pearson', 'spearman', 'kendall', 'lin_ccc']
        fig, axes = plt.subplots(2, 2, figsize=(20 * width_factor, 16))
        axes = axes.flatten()
        
    # Define colors for benchmark types
    colors = {'general': 'royalblue', 'medqa': 'forestgreen'}

    # Clean benchmark names by removing (medqa) and (general) suffixes
    def clean_benchmark_name(name):
        return name.replace("(medqa)", "").replace("(general)", "").strip()
    
    # Dictionary to store horizontal line data
    horizontal_line_data = {measure: {} for measure in corr_measures}
    
    # Plot correlations for each measure
    avg_corr_legend_elements = []
    for i, measure in enumerate(corr_measures):
        # Map lin_ccc to ccc for benchmark correlation data
        bench_measure = 'ccc' if measure == 'lin_ccc' else measure
        
        # Update the x-axis labels
        ax = axes[i]
        
        # Get the plot elements and clean x-axis labels
        avg_legend_elements, measure_h_lines = plot_correlation_measure(
            results_df, 
            measure, 
            colors, 
            ax, 
            corr_data, 
            clean_benchmark_func=clean_benchmark_name
        )
        
        # Store horizontal line data for this measure
        horizontal_line_data[measure] = measure_h_lines
        
        avg_corr_legend_elements = avg_legend_elements
        
        # Set x-axis label with bold font
        ax.set_xlabel("Benchmarks", fontsize=14)
        
        # Rotate x-tick labels for better spacing
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=12)
    
    # Create legend for benchmark types with a title
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors['general'], label='General'),
        plt.Rectangle((0, 0), 1, 1, color=colors['medqa'], label='Medical QA')
    ]
    
    # Calculate position for legends
    mm_to_inches = 1 / 25.4
    offset_inches = 1.5 * mm_to_inches
    fig_width_inches = fig.get_figwidth()
    legend_x_pos = 0.85 + (offset_inches / fig_width_inches)
    
    if TWO_CORR_MEASURE:
        legend_y_position_1 = 0.56 
        legend_y_position_2 = 0.43
    else:
        legend_y_position_1 = 0.30 
        legend_y_position_2 = 0.23

    # First legend (benchmark types)
    first_legend = fig.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(legend_x_pos, legend_y_position_1),
        ncol=1,
        fontsize=12,
        title="Benchmark Types",
        title_fontsize=13,
        alignment='left'
    )
    
    # Manually set the alignment of the legend title to left
    first_legend._legend_box.align = "left"
    
    # Add the second legend if we have average correlation elements
    if avg_corr_legend_elements:
        second_legend = fig.legend(
            handles=avg_corr_legend_elements,
            loc='upper left',
            bbox_to_anchor=(legend_x_pos, legend_y_position_2),
            ncol=1,
            fontsize=12,
            title="Average Correlation\nwith Benchmarks",
            title_fontsize=13,
            alignment='left'
        )
        
        # Manually set the alignment of the legend title to left
        second_legend._legend_box.align = "left"
        
    # Add the first legend to the figure
    fig.add_artist(first_legend)
    
    # Adjust layout to accommodate the legends
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    # Save plot in multiple formats
    if config['FOR_STRICTLY_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_plot_strictly_filtered'
    elif config['FOR_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_plot_filtered'
    else:
        plot_filename_base = 'correlation_plot'
        
    if USE_IMPUTATED:
        plot_filename_base += '_imputated'
        
    if WEIGHTING_BY_SAMPLE_SIZE:
        plot_filename_base += '_weighted'
        
    if TWO_CORR_MEASURE:
        plot_filename_base += '_two_measures'
        
    # Save in different formats
    formats = {
        'png': {'dpi': 600},
        'tiff': {'dpi': 600},
        'pdf': {'dpi': 600, 'bbox_inches': 'tight', 'format': 'pdf'}
    }
    
    for fmt, params in formats.items():
        filename = f'figures/{plot_filename_base}.{fmt}'
        plt.savefig(filename, **params)
        print(f"Plot saved to {filename}")

    return horizontal_line_data

def create_visualizations_with_bayesian_modeling(results_df, config, horizontal_line_data):
    """
    Create visualizations for correlation results using benchmark data from pickle file.
    This function uses horizontal line data from the original visualization.
    
    Args:
        results_df: DataFrame with correlation results
        config: Configuration dictionary
        horizontal_line_data: Dictionary with horizontal line data from create_visualizations
    """
    import pickle
    
    if len(results_df) == 0:
        print("No results to visualize")
        return
    
    # Load benchmark correlations from pickle file
    try:
        with open('bayesian_analysis_results/benchmark_correlation_results.pkl', 'rb') as f:
            bayesian_corr_data = pickle.load(f)
        print("Successfully loaded Bayesian correlation data")
    except Exception as e:
        print(f"Failed to load Bayesian correlation data: {e}")
        return
    
    # Set figure width to 1.2 times the original width
    width_factor = 1.2
    
    # Determine which correlation measures to use based on TWO_CORR_MEASURE flag
    if TWO_CORR_MEASURE:
        corr_measures = ['spearman', 'kendall']
        fig, axes = plt.subplots(1, 2, figsize=(20 * width_factor, 8))
        axes = axes.flatten()
    else:
        corr_measures = ['pearson', 'spearman', 'kendall', 'lin_ccc']
        fig, axes = plt.subplots(2, 2, figsize=(20 * width_factor, 16))
        axes = axes.flatten()
    
    # Define colors for benchmark types
    colors = {'general': 'royalblue', 'medqa': 'forestgreen'}

    # Clean benchmark names by removing (medqa) and (general) suffixes
    def clean_benchmark_name(name):
        return name.replace("(medqa)", "").replace("(general)", "").strip()
    
    # Helper function to map benchmark names from results_df to bayesian_corr_data format
    def map_benchmark_to_bayesian_key(b_type, b_name):
        clean_name = clean_benchmark_name(b_name)
        prefix = "general_" if b_type == "general" else "medical_"
        return f"{prefix}{clean_name}"
    
    # Plot correlations for each measure
    avg_corr_legend_elements = []
    for i, measure in enumerate(corr_measures):
        # Get horizontal line data for this measure
        measure_h_lines = horizontal_line_data.get(measure, {})
        
        # Update the x-axis labels
        ax = axes[i]
        
        # Plot with Bayesian data but original horizontal lines
        avg_legend_elements = plot_with_bayesian_data(
            results_df, 
            measure, 
            colors, 
            ax, 
            measure_h_lines,
            bayesian_corr_data,
            map_benchmark_to_bayesian_key,
            clean_benchmark_func=clean_benchmark_name
        )
        
        avg_corr_legend_elements = avg_legend_elements
        
        # Set x-axis label with bold font
        ax.set_xlabel("Benchmarks", fontsize=14)
        
        # Rotate x-tick labels for better spacing
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=12)
    
    # Create legend for benchmark types with a title
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors['general'], label='General (Bayesian)'),
        plt.Rectangle((0, 0), 1, 1, color=colors['medqa'], label='Medical QA (Bayesian)')
    ]
    
    # Calculate position for legends
    mm_to_inches = 1 / 25.4
    offset_inches = 1.5 * mm_to_inches
    fig_width_inches = fig.get_figwidth()
    legend_x_pos = 0.85 + (offset_inches / fig_width_inches)
    
    # First legend (benchmark types)
    first_legend = fig.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(legend_x_pos, 0.56),
        ncol=1,
        fontsize=12,
        title="Benchmark Types",
        title_fontsize=13,
        alignment='left'
    )
    
    # Manually set the alignment of the legend title to left
    first_legend._legend_box.align = "left"
    
    # Add the second legend if we have average correlation elements
    if avg_corr_legend_elements:
        second_legend = fig.legend(
            handles=avg_corr_legend_elements,
            loc='upper left',
            bbox_to_anchor=(legend_x_pos, 0.43),
            ncol=1,
            fontsize=12,
            title="Original Correlation\nwith Benchmarks",
            title_fontsize=13,
            alignment='left'
        )
        
        # Manually set the alignment of the legend title to left
        second_legend._legend_box.align = "left"
        
    # Add the first legend to the figure
    fig.add_artist(first_legend)
    
    # Adjust layout to accommodate the legends
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    # Save plot in multiple formats
    if config['FOR_STRICTLY_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_plot_bayesian_strictly_filtered'
    elif config['FOR_VALID_TASKS_AND_PERFS']:
        plot_filename_base = 'correlation_plot_bayesian_filtered'
    else:
        plot_filename_base = 'correlation_plot_bayesian'
        
    if USE_IMPUTATED:
        plot_filename_base += '_imputated'
        
    if WEIGHTING_BY_SAMPLE_SIZE:
        plot_filename_base += '_weighted'
        
    if TWO_CORR_MEASURE:
        plot_filename_base += '_two_measures'
        
    # Save in different formats
    formats = {
        'png': {'dpi': 600},
        'tiff': {'dpi': 600},
        'pdf': {'dpi': 600, 'bbox_inches': 'tight', 'format': 'pdf'}
    }
    
    for fmt, params in formats.items():
        filename = f'figures/{plot_filename_base}.{fmt}'
        plt.savefig(filename, **params)
        print(f"Plot saved to {filename}")

def plot_with_bayesian_data(results_df, measure, colors, ax, horizontal_line_data, 
                           bayesian_corr_data, map_benchmark_func, clean_benchmark_func=None):
    """
    Plot correlation measure using Bayesian values for bars but original horizontal lines
    
    Args:
        results_df: DataFrame with correlation results
        measure: Correlation measure name
        colors: Dictionary mapping benchmark types to colors
        ax: Matplotlib axis to plot on
        horizontal_line_data: Dictionary of horizontal line data from original plot
        bayesian_corr_data: Dictionary of Bayesian correlation values
        map_benchmark_func: Function to map benchmark names to Bayesian keys
        clean_benchmark_func: Function to clean benchmark names for display
    
    Returns:
        List of legend elements for average correlation lines
    """
    # Calculate original means for model counts
    if WEIGHTING_BY_SAMPLE_SIZE and 'sample_size' in results_df.columns:
        original_means_df = pd.DataFrame([
            {
                'benchmark_type': b_type,
                'benchmark_name': b_name,
                'mean_val': np.average(group[measure], weights=np.log1p(group['sample_size'].fillna(1))) if len(group) > 0 else np.nan
            }
            for (b_type, b_name), group in results_df.groupby(['benchmark_type', 'benchmark_name'])
        ])
    else:
        original_means_df = results_df.groupby(['benchmark_type', 'benchmark_name'])[measure].mean().reset_index()
        original_means_df = original_means_df.rename(columns={measure: 'mean_val'})
    
    # Create DataFrame with Bayesian values
    benchmark_corrs = []
    for _, row in original_means_df.iterrows():
        b_type = row['benchmark_type']
        b_name = row['benchmark_name']
        
        # Map to Bayesian data key format
        bayesian_key = map_benchmark_func(b_type, b_name)
        
        # Get Bayesian correlation value if available
        bayesian_val = bayesian_corr_data.get(measure, {}).get(bayesian_key, np.nan)
        
        # Create combined key for horizontal line lookup
        combined_key = f"{b_type}_{b_name}"
        
        benchmark_corrs.append({
            'benchmark_type': b_type,
            'benchmark_name': b_name,
            measure: bayesian_val,  # Use Bayesian value for bars
            'horizontal_data_key': combined_key  # Key for horizontal line data
        })
    
    benchmark_corrs = pd.DataFrame(benchmark_corrs)
    
    # Filter out NaN values
    benchmark_corrs = benchmark_corrs.dropna(subset=[measure])
    
    if len(benchmark_corrs) == 0:
        ax.text(0.5, 0.5, f"No data for {measure}", ha='center', va='center')
        return None
    
    # Sort by Bayesian correlation value descending
    benchmark_corrs = benchmark_corrs.sort_values(measure, ascending=False)
    
    # Get sample sizes for each benchmark
    benchmark_counts = results_df.groupby(['benchmark_type', 'benchmark_name'])['num_models'].sum().reset_index()
    benchmark_corrs = pd.merge(benchmark_corrs, benchmark_counts, on=['benchmark_type', 'benchmark_name'])
    
    # Create bar colors based on benchmark type
    bar_colors = [colors[btype] for btype in benchmark_corrs['benchmark_type']]
    
    # Plot bars using Bayesian values
    bars = ax.bar(
        range(len(benchmark_corrs)), 
        benchmark_corrs[measure],
        color=bar_colors
    )
    
    # Add correlation values above bars
    value_text_font = 12
    for j, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f"{benchmark_corrs[measure].iloc[j]:.3f}" if WITHOUT_NUM_MODELS else f"{benchmark_corrs[measure].iloc[j]:.3f}\n({int(benchmark_corrs['num_models'].iloc[j])})",
            ha='center',
            va='bottom',
            fontsize=value_text_font
        )
    
    # Create legend elements for average correlations
    avg_legend_elements = []
    
    # Legend tracking
    legend_added = {'general': False, 'medqa': False}
    
    # Add horizontal lines from original plot data
    for j, (_, row) in enumerate(benchmark_corrs.iterrows()):
        bar = bars[j]
        bar_width = bar.get_width()
        bar_x = bar.get_x()
        benchmark_type = row['benchmark_type']
        horizontal_key = row['horizontal_data_key']
        
        # Get horizontal line values from data
        h_line_data = horizontal_line_data.get(horizontal_key, {})
        
        # Plot horizontal lines if available
        if h_line_data:
            # Same type horizontal line
            if 'same_type' in h_line_data:
                same_color = colors[benchmark_type]
                same_color_alpha = list(mcolors.to_rgba(same_color))
                same_color_alpha[3] = 0.5
                
                extended_width = bar_width * 1.05
                ax.plot(
                    [bar_x - 0.05 * bar_width, bar_x + extended_width], 
                    [h_line_data['same_type'], h_line_data['same_type']], 
                    color=same_color_alpha, 
                    linewidth=3.5,
                    linestyle='-'
                )
            
            # Other type horizontal line
            if 'other_type' in h_line_data:
                other_type = 'medqa' if benchmark_type == 'general' else 'general'
                other_color = colors[other_type]
                other_color_alpha = list(mcolors.to_rgba(other_color))
                other_color_alpha[3] = 0.5
                
                extended_width = bar_width * 1.05
                ax.plot(
                    [bar_x - 0.05 * bar_width, bar_x + extended_width], 
                    [h_line_data['other_type'], h_line_data['other_type']], 
                    color=other_color_alpha, 
                    linewidth=3.5,
                    linestyle='-'
                )
                
                # Add to legend if not already added
                if not legend_added[other_type]:
                    legend_label = 'General Original' if other_type == 'general' else 'Medical QA Original'
                    avg_legend_elements.append(
                        Line2D([0], [0], color=other_color, lw=3.5, linestyle='-', 
                              label=f'{legend_label}')
                    )
                    legend_added[other_type] = True
    
    # Clean benchmark names if function provided
    x_labels = []
    for _, row in benchmark_corrs.iterrows():
        benchmark_name = row['benchmark_name']
        if clean_benchmark_func:
            benchmark_name = clean_benchmark_func(benchmark_name)
        # if 'MMLU ' in benchmark_name and benchmark_name!='MMLU Pro':
        #     benchmark_name = benchmark_name.replace('MMLU', '').strip()
        #     benchmark_name += '\n(MMLU)'
        x_labels.append(benchmark_name)
    
    # Add benchmark names as x-tick labels with rotation
    ax.set_xticks(range(len(benchmark_corrs)))
    ax.set_xticklabels(
        x_labels,
        rotation=30,
        ha='right',
        fontsize=12
    )
    
    # Add title and labels with bold font
    if 'kendall' in measure.lower():
        title_str = "Kendall's Tau Correlation"
    elif 'lin' in measure.lower():
        title_str = "Lin's CCC"
    else:
        title_str = f"{measure.capitalize()} Correlation"
    ax.set_title(title_str, fontsize=14)
    ax.set_ylabel("Correlation Coefficient", fontsize=14)
    ax.set_xlabel("Benchmarks", fontsize=14)
    
    # Set y range from 0 to 1
    ax.set_xlim(-1, len(benchmark_corrs))
    ax.set_ylim(0, 1)  
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    return avg_legend_elements

def plot_correlation_measure(results_df, measure, colors, ax, corr_data=None,
                             clean_benchmark_func=None):
    """
    Plot correlation measure for all benchmarks with comparison lines
    
    Args:
        results_df: DataFrame with correlation results
        measure: Correlation measure name
        colors: Dictionary mapping benchmark types to colors
        ax: Matplotlib axis to plot on
        corr_data: Benchmark correlation data for comparison
        clean_benchmark_func: Function to clean benchmark names
    
    Returns:
        Tuple of (legend_elements, horizontal_line_data) where horizontal_line_data is
        a dictionary with horizontal line values
    """
    # Calculate weighted means if sample size available
    if WEIGHTING_BY_SAMPLE_SIZE and 'sample_size' in results_df.columns:
        benchmark_corrs = []
        for (b_type, b_name), group in results_df.groupby(['benchmark_type', 'benchmark_name']):
            weights = np.log1p(group['sample_size'].fillna(1))
            mean_val = np.average(group[measure], weights=weights) if len(group) > 0 else np.nan
            benchmark_corrs.append({
                'benchmark_type': b_type,
                'benchmark_name': b_name,
                measure: mean_val
            })
        benchmark_corrs = pd.DataFrame(benchmark_corrs)
    else:
        benchmark_corrs = results_df.groupby(['benchmark_type', 'benchmark_name'])[measure].mean().reset_index()
    
    # Filter out NaN values
    benchmark_corrs = benchmark_corrs.dropna()
    
    if len(benchmark_corrs) == 0:
        ax.text(0.5, 0.5, f"No data for {measure}", ha='center', va='center')
        return None, {}
        
    # Sort by correlation value descending
    benchmark_corrs = benchmark_corrs.sort_values(measure, ascending=False)
    
    # Get sample sizes for each benchmark
    benchmark_counts = results_df.groupby(['benchmark_type', 'benchmark_name'])['num_models'].sum().reset_index()
    benchmark_corrs = pd.merge(benchmark_corrs, benchmark_counts, on=['benchmark_type', 'benchmark_name'])
    
    # Create bar colors based on benchmark type
    bar_colors = [colors[btype] for btype in benchmark_corrs['benchmark_type']]
    
    # Plot bars
    bars = ax.bar(
        range(len(benchmark_corrs)), 
        benchmark_corrs[measure],
        color=bar_colors
    )
    
    # Add correlation values above bars
    value_text_font = 12
    for j, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f"{benchmark_corrs[measure].iloc[j]:.3f}" if WITHOUT_NUM_MODELS else f"{benchmark_corrs[measure].iloc[j]:.3f}\n({int(benchmark_corrs['num_models'].iloc[j])})",
            ha='center',
            va='bottom',
            fontsize=value_text_font
        )
    
    # Create legend elements for average correlations
    avg_legend_elements = []
    
    # Dictionary to store horizontal line data
    horizontal_line_data = {}
    
    # Add benchmark correlation lines if data available
    if corr_data:
        # Keep track of whether we've added legend elements for each type
        legend_added = {'general': False, 'medqa': False}
        
        for j, (_, row) in enumerate(benchmark_corrs.iterrows()):
            benchmark_name = row['benchmark_name']
            benchmark_type = row['benchmark_type']
            
            # Calculate average correlations
            same_type_avg, other_type_avg = calculate_avg_benchmark_correlations(
                corr_data, benchmark_name, benchmark_type, measure)
            
            # Store horizontal line data
            combined_key = f"{benchmark_type}_{benchmark_name}"
            horizontal_line_data[combined_key] = {
                'same_type': same_type_avg,
                'other_type': other_type_avg
            }
            
            bar = bars[j]
            bar_width = bar.get_width()
            bar_x = bar.get_x()
            
            # Plot same type average correlation line
            if same_type_avg is not None:
                same_color = colors[benchmark_type]
                # Create a more transparent version of the color
                same_color_alpha = list(mcolors.to_rgba(same_color))
                same_color_alpha[3] = 0.5  # Lower alpha for lighter color
                
                # Extend line beyond bar width
                extended_width = bar_width * 1.05  # 5% longer on each side
                ax.plot(
                    [bar_x - 0.05 * bar_width, bar_x + extended_width], 
                    [same_type_avg, same_type_avg], 
                    color=same_color_alpha, 
                    linewidth=3.5,
                    linestyle='-'
                )
                
            # Plot other type average correlation line
            if other_type_avg is not None:
                other_type = 'medqa' if benchmark_type == 'general' else 'general'
                other_color = colors[other_type]
                # Create a more transparent version of the color
                other_color_alpha = list(mcolors.to_rgba(other_color))
                other_color_alpha[3] = 0.5  # Lower alpha for lighter color
                
                # Extend line beyond bar width
                ax.plot(
                    [bar_x - 0.05 * bar_width, bar_x + extended_width], 
                    [other_type_avg, other_type_avg], 
                    color=other_color_alpha, 
                    linewidth=3.5,
                    linestyle='-'
                )

                if not legend_added[other_type]:
                    legend_label = 'General' \
                        if other_type == 'general' else 'Medical QA'
                    avg_legend_elements.append(
                        Line2D([0], [0], color=other_color, lw=3.5, linestyle='-', 
                            label=f'{legend_label}')
                    )
                    legend_added[other_type] = True
    
    # Clean benchmark names if function provided
    x_labels = []
    for _, row in benchmark_corrs.iterrows():
        benchmark_name = row['benchmark_name']
        if clean_benchmark_func:
            benchmark_name = clean_benchmark_func(benchmark_name)
        # if 'MMLU ' in benchmark_name and benchmark_name!='MMLU Pro':
        #     benchmark_name = benchmark_name.replace('MMLU', '').strip()
        #     benchmark_name += '\n(MMLU)'
        x_labels.append(benchmark_name)
    
    # Add benchmark names as x-tick labels with rotation
    ax.set_xticks(range(len(benchmark_corrs)))
    ax.set_xticklabels(
        x_labels,
        rotation=30,
        ha='right',
        fontsize=12
    )
    
    # Add title and labels with bold font
    if 'kendall' in measure.lower():
        title_str = "Kendall's Tau Correlation"
    elif 'lin' in measure.lower():
        title_str = "Lin's CCC"
    else:
        title_str = f"{measure.capitalize()} Correlation"
    ax.set_title(title_str, fontsize=14)
    ax.set_ylabel("Correlation Coefficient", fontsize=14)
    ax.set_xlabel("Benchmarks", fontsize=14)
    
    # Set y range from 0 to 1
    ax.set_xlim(-1, len(benchmark_corrs))
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    return avg_legend_elements, horizontal_line_data

# Run the analysis
if __name__ == "__main__":
    __init__()
