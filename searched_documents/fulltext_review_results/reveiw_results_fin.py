"""
Clinical data analysis tools for processing and visualizing metadata distributions,
task consistency, and performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
import glob

# Suppress warnings to avoid cluttering the output
warnings.filterwarnings('ignore')

# Global constants
DEFAULT_DATA_FILENAME = 'clinical_llm_data.xlsx'
KOREAN_FILENAME = 'results_fin_250210_추가작업(2.14).xlsx'
THERAPEUTIC_AREA_THRESHOLD = 3.0  # Percentage threshold for consolidating areas
META_COLUMNS = ['task_type', 'therapeutic_area', 'data_source', 'evaluation_type']

# Set a consistent color palette and improve plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors


def consolidate_therapeutic_areas(df):
    """
    Consolidate therapeutic areas with less than 3% representation into larger categories.
    
    Args:
        df: DataFrame containing clinical data with 'therapeutic_area' column
        
    Returns:
        DataFrame with updated therapeutic_area values
    """
    # Copy dataframe to avoid modifying the original
    df_consolidated = df.copy()
    
    # Calculate the current distribution
    area_counts = df['therapeutic_area'].value_counts()
    total = len(df)
    area_percentages = (area_counts / total * 100).round(1)
    
    # Define the mapping for consolidation
    # Areas under threshold will be mapped to appropriate larger categories
    area_mapping = {
        # Keeping original areas that are > 3%
        'general medicine': 'general medicine',
        'oncology': 'oncology',
        'ophthalmology': 'ophthalmology',
        'orthopedics & musculoskletal': 'orthopedics & musculoskletal',
        'emergency medicine': 'emergency medicine',
        'neurology': 'neuropsychiatric',
        'genetic disorders': 'genetic disorders',
        'radiology': 'radiology',
        'otolaryngology': 'dental & otolaryngology',
        'pediatrics': 'pediatrics',
        'cardiology': 'cardiology',
        'dermatology': 'dermatology',
        
        # Consolidating smaller areas (< 3%)
        'dentistry': 'dental & otolaryngology',
        'nephrology': 'internal medicine subspecialties',
        'gastroenterology & hepatology': 'internal medicine subspecialties',
        'psychiatric': 'neuropsychiatric',
        'infectious': 'general medicine',
        'pulmonology': 'internal medicine subspecialties',
        'urology': 'internal medicine subspecialties',
        'rhematology': 'internal medicine subspecialties',
        'cardiovascular': 'cardiology',
        'hematology': 'oncology',
        'endorionology': 'internal medicine subspecialties',
        'orolaryngology': 'dental & otolaryngology',
        'forensic': 'specialty medicine',
    }
    
    # Apply the mapping
    df_consolidated['original_area'] = df_consolidated['therapeutic_area']
    df_consolidated['therapeutic_area'] = df_consolidated['therapeutic_area'].map(area_mapping)
    
    # Print the consolidation details
    print("\n=== Therapeutic Area Consolidation ===")
    print("Original areas mapped to new consolidated categories:")
    
    # Group by new categories and list original areas
    area_grouping = defaultdict(list)
    for orig, new in area_mapping.items():
        if orig != new:  # Only show those that changed
            area_grouping[new].append(orig)
    
    for new_area, orig_areas in area_grouping.items():
        print(f"\n{new_area}:")
        for orig in orig_areas:
            count = area_counts.get(orig, 0)
            pct = area_percentages.get(orig, 0)
            print(f"  - {orig}: {count} entries ({pct}%)")
    
    # Calculate and print the new distribution
    new_area_counts = df_consolidated['therapeutic_area'].value_counts()
    new_area_percentages = (new_area_counts / total * 100).round(1)
    
    print("\nNew therapeutic area distribution:")
    for area, count in new_area_counts.items():
        pct = new_area_percentages[area]
        print(f"{area}: {count} entries ({pct}%)")
    
    return df_consolidated


def check_task_consistency(df):
    """
    Check if tasks with the same name have consistent metadata.
    
    Args:
        df: DataFrame containing clinical task data
        
    Returns:
        Dictionary of tasks with inconsistencies in their metadata
    """
    # Group by task_name
    task_groups = df.groupby('task_name')
    
    # Dictionary to store inconsistencies
    inconsistencies = {}
    
    # Check each task group
    for task_name, group in task_groups:
        task_inconsistencies = {}
        
        # Check each metadata column
        for col in META_COLUMNS:
            unique_values = group[col].unique()
            
            # If more than one unique value, there's an inconsistency
            if len(unique_values) > 1:
                task_inconsistencies[col] = unique_values.tolist()
        
        # If any inconsistencies found, add to the main dictionary
        if task_inconsistencies:
            inconsistencies[task_name] = task_inconsistencies
    
    return inconsistencies


def analyze_metadata_distributions(df):
    """
    Analyze and visualize distributions and relationships of metadata columns.
    
    Args:
        df: DataFrame containing clinical task data
    """
    # Create figure for individual distributions (pie charts)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Clinical Task Metadata Distribution Analysis', fontsize=16, y=0.98)
    
    # Plot individual distributions as pie charts
    for i, col in enumerate(META_COLUMNS):
        plt.subplot(2, 2, i+1)
        
        # Calculate percentages for print and labels
        value_counts = df[col].value_counts()
        total = value_counts.sum()
        percentages = (value_counts / total * 100).round(1)
        
        # Print summary statistics for each category
        print(f"\n=== Distribution of {col} ===")
        for value, count in value_counts.items():
            pct = percentages[value]
            print(f"{value}: {count} entries ({pct}%)")
        
        # Create labels with percentages
        labels = [f"{val}\n({pct}%)" for val, pct in zip(value_counts.index, percentages)]
        
        # Plot pie chart
        plt.pie(value_counts, labels=labels, autopct='%1.1f%%', 
                startangle=90, colors=COLORS[:len(value_counts)],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Create separate figures for each pair of metadata
    analyze_metadata_relationships(df)


def analyze_metadata_relationships(df):
    """
    Analyze and visualize relationships between pairs of metadata columns.
    
    Args:
        df: DataFrame containing clinical task data
    """
    for i, col1 in enumerate(META_COLUMNS):
        for j, col2 in enumerate(META_COLUMNS):
            if i < j:  # Only plot each pair once
                # Create a new figure for each pair
                plt.figure(figsize=(10, 6))
                
                # Create cross-tabulation
                cross_tab = pd.crosstab(df[col1], df[col2])
                cross_tab = cross_tab.astype(int)
                print(cross_tab)
                
                # Print cross-tabulation summary
                print(f"\n=== Relationship between {col1} and {col2} ===")
                print(f"Top combinations:")
                flat_data = []
                for idx1, row in cross_tab.iterrows():
                    for idx2, val in row.items():
                        if val > 0:
                            flat_data.append((idx1, idx2, val))
                
                # Sort by count and print top 5
                top_combinations_count = 5
                for item1, item2, count in sorted(flat_data, key=lambda x: x[2], reverse=True)[:top_combinations_count]:
                    print(f"{item1} + {item2}: {count} entries")
                
                # Plot heatmap with annotations
                sns.heatmap(cross_tab,
                            annot=True,
                            cmap='Blues',
                            fmt='.0f',
                            cbar=False,
                            annot_kws={
                                'fontsize': 12,
                                'color': 'black'
                            })

                plt.title(f'{col1} vs\n{col2}')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()


def analyze_task_performance(df):
    """
    Analyze task performance counts and visualize top tasks.
    
    Args:
        df: DataFrame containing clinical task data
    """
    # Count rows per task
    task_counts = df.groupby('task_name').size().sort_values(ascending=False)
    
    # Get top 10 tasks by count
    top_task_count = 10
    top_tasks = task_counts.head(top_task_count)
    
    # Print task counts summary
    print("\n=== Task Performance Report Summary ===")
    print(f"Total number of tasks: {len(task_counts)}")
    print(f"Total number of performance reports: {len(df)}")
    print(f"Average reports per task: {len(df) / len(task_counts):.2f}")
    print(f"Maximum reports for a single task: {task_counts.max()} ({task_counts.idxmax()})")
    print(f"Minimum reports for a single task: {task_counts.min()} ({task_counts.idxmin()})")
    
    print("\nTop 10 tasks by number of reports:")
    for task, count in top_tasks.items():
        print(f"{task}: {count} reports")
    
    # Visualize top tasks
    plt.figure(figsize=(14, 8))
    top_tasks.plot(kind='bar', color=COLORS[:len(top_tasks)])
    plt.title('Number of Performance Reports for Top 10 Tasks')
    plt.xlabel('Task Name')
    plt.ylabel('Number of Reports')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Analyze by metadata groups
    analyze_metadata_groups(df)


def analyze_metadata_groups(df):
    """
    Analyze and print statistics for tasks grouped by metadata columns.
    
    Args:
        df: DataFrame containing clinical task data
    """
    # Calculate and print metadata group statistics (text only, no graphs)
    print("\n=== Metadata Group Statistics ===")
    
    for col in META_COLUMNS:
        # Group by metadata column
        meta_counts = df.groupby(col).size().sort_values(ascending=False)
        
        # Calculate unique tasks per group
        tasks_per_group = df.groupby(col)['task_name'].nunique().sort_values(ascending=False)
        
        # Calculate average reports per task in each group
        avg_reports = pd.DataFrame({
            'total_reports': meta_counts,
            'unique_tasks': tasks_per_group
        })
        avg_reports['avg_reports_per_task'] = avg_reports['total_reports'] / avg_reports['unique_tasks']
        
        # Print summary
        print(f"\nStatistics for {col}:")
        for meta_value, row in avg_reports.iterrows():
            print(f"  {meta_value}:")
            print(f"    Total reports: {row['total_reports']}")
            print(f"    Unique tasks: {row['unique_tasks']}")
            print(f"    Average reports per task: {row['avg_reports_per_task']:.2f}")


def load_data():
    """
    Load clinical data from available Excel files or use sample data if unavailable.
    
    Returns:
        DataFrame containing the loaded data
    """
    try:
        # First try with the pattern filename
        try:
            files = glob.glob(KOREAN_FILENAME)
            if files:
                df = pd.read_excel(files[0], sheet_name='Sheet1')
                print(f"Successfully loaded file: {files[0]}")
            else:
                # If pattern doesn't match, try a direct filename
                df = pd.read_excel(KOREAN_FILENAME, sheet_name='Sheet1')
                print(f"Successfully loaded file: {KOREAN_FILENAME}")
        except:
            # Fallback to a simple filename
            df = pd.read_excel(DEFAULT_DATA_FILENAME, sheet_name='Sheet1')
            print(f"Successfully loaded file: {DEFAULT_DATA_FILENAME}")
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Using sample data for demonstration...")
        # Create sample data for demonstration
        df = pd.read_csv('paste.txt', sep='\t')
    
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    return df


def main():
    """
    Main function to execute the clinical data analysis workflow.
    """
    # Load the data
    df = load_data()
    
    # Apply therapeutic area consolidation before running the analysis
    df = consolidate_therapeutic_areas(df)
    
    # Check task consistency
    print("\n=== Checking Task Metadata Consistency ===")
    inconsistencies = check_task_consistency(df)
    
    if inconsistencies:
        print("\nWARNING: Inconsistencies detected in task metadata:")
        for task_name, issues in inconsistencies.items():
            print(f"\nTask: {task_name}")
            for meta_col, values in issues.items():
                print(f"  {meta_col} has multiple values: {values}")
    else:
        print("✓ All tasks have consistent metadata values")
    
    # Analyze metadata distributions
    print("\n=== Analyzing Metadata Distributions ===")
    analyze_metadata_distributions(df)
    
    # Analyze task performance counts
    print("\n=== Analyzing Task Performance Reports ===")
    analyze_task_performance(df)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()