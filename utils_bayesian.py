"""utils_bayesian.py: Utility functions for Bayesian optimization."""
import os
import pickle
import traceback

import jax.numpy as jnp
import numpyro

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy import stats

# Constants
HIGH_CONNECTIVITY_THRESHOLD = 50  # Percentile threshold for high connectivity models
DEFAULT_HIST_BINS = 30  # Default number of bins for histograms
MAJOR_MODEL_THRESHOLD_PERCENTILE = 75  # Percentile threshold for major models
R_HAT_THRESHOLD = 1.1  # Threshold for R-hat convergence diagnostic
ESS_THRESHOLD = 300  # Threshold for effective sample size
TOP_BOTTOM_TASK_LIMIT = 20  # Number of top/bottom tasks to display when there are too many


def setup_environment():
    """Configure PyTensor environment to avoid compilation issues."""
    # Completely disable C compilation
    os.environ["PYTENSOR_FLAGS"] = "mode=FAST_COMPILE"
    os.environ["PYTENSOR_COMPILER"] = "python"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Disable PyTensor cache to avoid compilation attempts
    os.environ["PYTENSOR_BASE_COMPILEDIR"] = ""
    print("Environment configured for Python-only mode.")


def load_and_prepare_data(file_path,
                          drop_models_with_one_records=False,
                          drop_models_with_two_records=False,
                          modeling_task_id=False):
    """
    Load and prepare clinical performance data for modeling.
    
    Args:
        file_path: Path to Excel file containing clinical performance data
        drop_models_with_one_records: If True, drop models with only one record
        drop_models_with_two_records: If True, drop models with only two records
        modeling_task_id: If True, add task_id column based on task_name
    
    Returns:
        DataFrame with prepared clinical performance data
    """
    # Load data
    clinical_perf_df = pd.read_excel(file_path)

    # Select necessary columns
    clinical_perf_df = clinical_perf_df[['task_name', 'task_type', 'therapeutic_area',
                                         'data_source', 'evaluation_type',
                                         'model_full_name2',
                                         'metric_value', 'sample_size']]

    # Filter models with few records if requested
    if drop_models_with_one_records:
        print("Dropping models with only one record...")
        clinical_perf_df = _filter_models_by_record_count(clinical_perf_df, 1)
        
    if drop_models_with_two_records:
        print("Dropping models with two records...")
        clinical_perf_df = _filter_models_by_record_count(clinical_perf_df, 2)

    # Plot metric value distribution
    plot_metric_distribution(clinical_perf_df, 'metric_value',
                            'Distribution of Model Performance Metrics', 'Metric Value')

    # Plot model metric count distribution
    model_counts_df = clinical_perf_df['model_full_name2'].value_counts().reset_index()
    model_counts_df.columns = ['model_full_name2', 'count']
    plot_metric_distribution(model_counts_df, 'count',
                            'Distribution of Number of Records per Model', 'Number of Records',
                            bins=50)

    # Get major models
    major_models = get_major_models(clinical_perf_df)

    # Compute model connectivity ranking
    model_connectivity_ranking = compute_model_connectivity(clinical_perf_df)
    plot_metric_distribution(model_connectivity_ranking, 'combined_score',
                             'Distribution of Model Connectivity Scores',
                             'Combined Connectivity Score', bins=100)

    # Add column for top N% models by connectivity
    threshold_score = np.percentile(model_connectivity_ranking['combined_score'],
                                    HIGH_CONNECTIVITY_THRESHOLD)
    model_connectivity_ranking['high_connectivity'] = \
        model_connectivity_ranking['combined_score'] >= threshold_score

    # Add the high_connectivity flag to the main dataframe
    clinical_perf_df = _add_connectivity_to_dataframe(clinical_perf_df, model_connectivity_ranking)

    # Add task_id column based on task_name if modeling_task_id is enabled
    if modeling_task_id:
        print("Adding task_id based on unique task_name...")
        clinical_perf_df['task_id'] = clinical_perf_df['task_name'].astype('category').cat.codes
        print(f"Created {clinical_perf_df['task_id'].nunique()} unique task IDs")

    # Convert categorical variables to numeric codes
    for col in ['task_type', 'data_source', 'evaluation_type',
                'model_full_name2', 'task_name']:
        clinical_perf_df[f'{col}_code'] = clinical_perf_df[col].astype('category').cat.codes

    # Z-score normalization for performance metric
    clinical_perf_df['metric_value_scaled'] = (
        clinical_perf_df['metric_value'] - clinical_perf_df['metric_value'].mean()
    ) / clinical_perf_df['metric_value'].std()

    return clinical_perf_df


def _filter_models_by_record_count(df, count):
    """Filter out models with specified record count."""
    model_counts = df['model_full_name2'].value_counts()
    filtered_models = model_counts[model_counts == count].index
    return df[~df['model_full_name2'].isin(filtered_models)]


def _add_connectivity_to_dataframe(df, connectivity_ranking):
    """Add high_connectivity flag to the main dataframe."""
    # Ensure model names column in connectivity ranking matches model_full_name2
    model_name_col = connectivity_ranking.columns[0]  # Assuming first column contains model names
    if model_name_col != 'model_full_name2':
        connectivity_ranking = connectivity_ranking.rename(
            columns={model_name_col: 'model_full_name2'})

    df = df.merge(
        connectivity_ranking[['model_full_name2', 'high_connectivity']],
        on='model_full_name2',
        how='left'
    )
    df['high_connectivity'] = df['high_connectivity'].fillna(False)
    return df


def plot_metric_distribution(df, column, title, xlabel, bins=DEFAULT_HIST_BINS):
    """
    Plot histogram for a given column in the dataset.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        title: Title of the histogram
        xlabel: X-axis label
        bins: Number of bins for the histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def get_major_models(df, threshold_percentile=MAJOR_MODEL_THRESHOLD_PERCENTILE):
    """
    Identify major models based on the number of records (metric count).
    
    Args:
        df: DataFrame containing the data
        threshold_percentile: Percentile threshold for major models
    
    Returns:
        List of major model names
    """
    model_counts = df['model_full_name2'].value_counts()
    threshold = np.percentile(model_counts, threshold_percentile)
    major_models = model_counts[model_counts >= threshold].index.tolist()
    print(f"Threshold for major models: {threshold}, Number of major models: {len(major_models)}")
    return major_models


def compute_model_connectivity(df):
    """
    Compute model connectivity ranking using network analysis.
    
    Args:
        df: DataFrame containing the model-task relationships
    
    Returns:
        DataFrame with model connectivity rankings and centrality measures
    """
    G = nx.Graph()

    # Add edges between models and tasks
    for _, row in df.iterrows():
        model = row['model_full_name2']
        task = row['task_name']
        if pd.notna(model) and pd.notna(task):
            G.add_edge(model, task)

    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Extract model-specific rankings
    model_ranking = pd.DataFrame({
        'model': list(degree_centrality.keys()),
        'degree_centrality': list(degree_centrality.values()),
        'betweenness_centrality': list(betweenness_centrality.values()),
        'closeness_centrality': list(closeness_centrality.values())
    })

    # Filter only models (exclude tasks from ranking)
    model_ranking = model_ranking[model_ranking['model'].isin(df['model_full_name2'].unique())]

    # Rank models by combined centrality score
    model_ranking['combined_score'] = (
        model_ranking['degree_centrality'] +
        model_ranking['betweenness_centrality'] +
        model_ranking['closeness_centrality']
    )
    model_ranking = model_ranking.sort_values('combined_score', ascending=False)
    
    return model_ranking


def check_convergence(mcmc, threshold_rhat=R_HAT_THRESHOLD, threshold_ess=ESS_THRESHOLD):
    """
    Check MCMC convergence using R-hat and ESS metrics.
    
    Args:
        mcmc: MCMC object after sampling
        threshold_rhat: Threshold for R-hat convergence diagnostic
        threshold_ess: Threshold for effective sample size
    
    Returns:
        Dictionary with convergence diagnostics
    """
    try:
        # Get samples per chain to keep shape = (chains, samples, ...)
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        param_names = list(samples_by_chain.keys())

        rhat_violations = {}
        ess_dict = {}
        ess_violations = {}
        good_convergence_params = []

        for param in param_names:
            # param_samples has shape (chains, samples, [extra dims...])
            param_samples = samples_by_chain[param]

            # Skip parameters that can't be properly analyzed
            if not _can_analyze_parameter(param_samples):
                continue

            param_converged = True

            # Calculate R-hat
            param_converged = _check_rhat(param, param_samples, threshold_rhat, 
                                          rhat_violations, param_converged)

            # Calculate ESS
            param_converged = _check_ess(param, param_samples, threshold_ess, 
                                         ess_dict, ess_violations, param_converged)

            if param_converged:
                good_convergence_params.append(param)

        # Print summary
        _print_convergence_summary(threshold_rhat, threshold_ess, 
                                  rhat_violations, ess_violations, good_convergence_params)

        return {
            'rhat_violations': rhat_violations,
            'ess_violations': ess_violations,
            'ess': ess_dict,
            'converged': (len(rhat_violations) == 0 and len(ess_violations) == 0),
            'good_convergence_params': good_convergence_params
        }

    except Exception as e:
        print(f"Error in convergence diagnostics: {str(e)}")
        traceback.print_exc()
        return {
            'rhat_violations': {},
            'ess_violations': {},
            'ess': {},
            'converged': False,
            'error': str(e),
            'good_convergence_params': []
        }


def _can_analyze_parameter(param_samples):
    """Check if parameter can be analyzed for convergence."""
    # Check sample dimension
    if param_samples.shape[1] < 2:
        print(f"Skipping R-hat/ESS (not enough samples per chain).")
        return False

    # Check if constant
    if jnp.all(param_samples == param_samples[0, 0]):
        print(f"Skipping R-hat/ESS (constant value).")
        return False
        
    return True


def _check_rhat(param, param_samples, threshold, violations, is_converged):
    """Check R-hat convergence diagnostic for a parameter."""
    try:
        r_hat = numpyro.diagnostics.split_gelman_rubin(param_samples)
        if r_hat.ndim == 0:
            # Scalar
            val_rhat = float(r_hat)
            print(f"R-hat for {param}: {val_rhat:.4f}")
            if val_rhat > threshold:
                violations[param] = val_rhat
                is_converged = False
        else:
            # Array (e.g., shape = (n_models, ) etc.)
            max_rhat = float(jnp.max(r_hat))
            print(f"R-hat for {param}: {max_rhat:.4f} (max over {r_hat.shape} array)")
            if jnp.any(r_hat > threshold):
                violations[param] = max_rhat
                is_converged = False
    except Exception as e:
        print(f"Could not calculate R-hat for {param}: {str(e)}")
        is_converged = False
        
    return is_converged


def _check_ess(param, param_samples, threshold, ess_dict, violations, is_converged):
    """Check effective sample size for a parameter."""
    try:
        ess = numpyro.diagnostics.effective_sample_size(param_samples)
        if jnp.isnan(ess).any():
            print(f"Skipping ESS for {param} (NaN encountered).")
            is_converged = False
        else:
            if ess.ndim == 0:
                # Scalar
                val_ess = float(ess)
                print(f"ESS for {param}: {val_ess:.1f}")
                ess_dict[param] = val_ess
                if val_ess < threshold:
                    violations[param] = val_ess
                    is_converged = False
            else:
                # Array
                mean_ess = float(jnp.mean(ess))
                min_ess = float(jnp.min(ess))
                print(f"ESS for {param}: mean={mean_ess:.1f}, min={min_ess:.1f} (over {ess.shape})")
                ess_dict[param] = mean_ess
                if min_ess < threshold:
                    violations[param] = min_ess
                    is_converged = False
    except Exception as e:
        print(f"Could not calculate ESS for {param}: {str(e)}")
        is_converged = False
        
    return is_converged


def _print_convergence_summary(threshold_rhat, threshold_ess, 
                              rhat_violations, ess_violations, good_params):
    """Print summary of convergence diagnostics."""
    print("\n=== MCMC Convergence Diagnostics ===")
    print(f"R-hat threshold: {threshold_rhat}")
    print(f"ESS threshold: {threshold_ess}")

    if not rhat_violations:
        print("✓ All parameters have R-hat < threshold (good convergence).")
    else:
        print(f"✗ {len(rhat_violations)} parameters have R-hat > threshold:")
        for p, r in rhat_violations.items():
            print(f"  - {p}: {r:.4f}")

    if not ess_violations:
        print("✓ All parameters have ESS > threshold (good sampling).")
    else:
        print(f"✗ {len(ess_violations)} parameters have ESS < threshold:")
        for p, v in ess_violations.items():
            print(f"  - {p}: {v:.1f}")

    print("\n=== Parameters with Good Convergence ===")
    if good_params:
        for param in good_params:
            print(f"  - {param}")
    else:
        print("No parameters have both good R-hat and ESS values.")


def plot_qq(residuals, title="Q-Q Plot of Residuals", output_dir=None):
    """
    Create a Q-Q plot to check normality of residuals.
    
    Args:
        residuals: Residuals from model fit
        title: Plot title
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate quantiles and create Q-Q plot
    fig = stats.probplot(residuals, dist="norm", plot=plt)
    
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'residuals_qq_plot.png')
    
    plt.show()


def _save_plot(output_dir, filename, dpi=300):
    """Save plot to specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')


def posterior_predictive_check(mcmc, model, data, output_dir=None):
    """
    Perform posterior predictive checks to assess model fit.
    
    Args:
        mcmc: MCMC object after sampling
        model: NumPyro model function
        data: Dictionary containing model inputs
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (residuals, predicted_values)
    """
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import matplotlib.pyplot as plt
        from numpyro.infer import Predictive
        
        # Get posterior samples
        posterior_samples = mcmc.get_samples()
        
        # Create predictive distribution - return only observation site
        predictive = Predictive(model, posterior_samples, return_sites=["obs"])
        
        # Generate predictions (excluding obs key from data)
        prediction_data = {k: v for k, v in data.items() if k != 'obs'}
        rng_key = jax.random.PRNGKey(0)
        
        # Predict without observed data
        predictions = predictive(rng_key, **prediction_data)
        
        # Extract observed data and predicted values
        y_obs = data.get('obs', None)
        y_pred = predictions.get('obs', None)
        
        # Print debug info
        _print_predictive_debug_info(y_obs, y_pred)
        
        if y_obs is None or y_pred is None:
            print("Warning: Could not extract observed or predicted values for PPC.")
            return None, None
        
        # Convert JAX arrays to numpy for easier handling
        y_obs, y_pred = _convert_to_numpy(y_obs, y_pred)
        
        # Calculate posterior predictive p-values
        ppp_values = np.mean(y_pred > y_obs, axis=0)
        
        # Compute mean predicted values across all posterior samples
        y_pred_mean = np.mean(y_pred, axis=0) if len(y_pred.shape) > 1 else y_pred
        
        # Calculate residuals
        residuals = y_obs - y_pred_mean
        
        # Create visualization plots
        _create_posterior_predictive_plots(y_obs, y_pred, y_pred_mean, residuals, ppp_values, output_dir)
        
        # Calculate and print summary statistics
        _print_predictive_summary(residuals, y_obs, y_pred_mean, ppp_values)
        
        # Analyze model effects if available
        _analyze_model_effects(posterior_samples)
        
        return residuals, y_pred_mean
    except Exception as e:
        print(f"Error in posterior predictive check: {str(e)}")
        traceback.print_exc()
        return None, None


def _print_predictive_debug_info(y_obs, y_pred):
    """Print debug information for predictive check."""
    print("\n=== Posterior Predictive Check Debug Info ===")
    print(f"Observed shape: {y_obs.shape if y_obs is not None else None}")
    print(f"Predicted shape: {y_pred.shape if y_pred is not None else None}")
    
    # Compare first few values
    if y_obs is not None and y_pred is not None:
        print("\nComparison of first 5 values:")
        for i in range(min(5, len(y_obs))):
            if isinstance(y_pred, jnp.ndarray) and len(y_pred.shape) > 1:
                pred_mean = float(jnp.mean(y_pred[:, i]))
                print(f"  Index {i}: Observed={float(y_obs[i]):.4f}, "
                      f"Predicted (mean)={pred_mean:.4f}, "
                      f"Diff={float(y_obs[i] - pred_mean):.4f}")
            else:
                print(f"  Index {i}: Observed={float(y_obs[i]):.4f}, "
                      f"Predicted={float(y_pred[i]):.4f}, "
                      f"Diff={float(y_obs[i] - y_pred[i]):.4f}")


def _convert_to_numpy(y_obs, y_pred):
    """Convert JAX arrays to numpy arrays."""
    if isinstance(y_obs, jnp.ndarray):
        y_obs = np.array(y_obs)
    if isinstance(y_pred, jnp.ndarray):
        y_pred = np.array(y_pred)
    return y_obs, y_pred


def _create_posterior_predictive_plots(y_obs, y_pred, y_pred_mean, residuals, ppp_values, output_dir):
    """Create plots for posterior predictive check."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of observed vs predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_obs, y_pred_mean, alpha=0.5)
    
    # Add diagonal line for reference
    min_val = min(np.min(y_obs), np.min(y_pred_mean))
    max_val = max(np.max(y_obs), np.max(y_pred_mean))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values (Mean)')
    plt.title('Observed vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(y_obs, y_pred_mean)[0, 1]
    plt.annotate(f'Correlation: {corr:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Plot histograms
    plt.subplot(2, 2, 2)
    plt.hist(y_obs, bins=30, alpha=0.5, label='Observed')
    
    # Flatten predicted values for histogram
    if len(y_pred.shape) > 1:
        y_pred_flat = y_pred.reshape(-1)
        plt.hist(y_pred_flat, bins=30, alpha=0.5, label='Predicted')
    else:
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Observed vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 2, 3)
    plt.scatter(y_pred_mean, residuals, alpha=0.5)
    plt.axhline(0, color='k', linestyle='--')
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # Plot PPP values
    plt.subplot(2, 2, 4)
    plt.hist(ppp_values, bins=20)
    plt.axvline(0.5, color='r', linestyle='--')
    
    plt.xlabel('Posterior Predictive p-value')
    plt.ylabel('Frequency')
    plt.title('Posterior Predictive p-values (ideal: centered at 0.5)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'posterior_predictive_check.png')
    
    plt.show()


def _print_predictive_summary(residuals, y_obs, y_pred_mean, ppp_values):
    """Print summary statistics for predictive check."""
    print("\n=== Posterior Predictive Check Summary ===")
    print(f"Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
    print(f"Root mean squared error: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"Mean of PPP values: {np.mean(ppp_values):.4f} (ideal: 0.5)")
    print(f"Std of PPP values: {np.std(ppp_values):.4f}")
    
    # Correlation between observed and predicted
    corr = np.corrcoef(y_obs, y_pred_mean)[0, 1]
    print(f"Correlation between observed and predicted: {corr:.4f}")
    
    # Check for potential overfitting
    mean_abs_diff = np.mean(np.abs(y_obs - y_pred_mean))
    obs_std = np.std(y_obs)
    relative_diff = mean_abs_diff / obs_std
    
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")
    print(f"Relative to observed std: {relative_diff:.4f}")
    
    if relative_diff < 0.1:
        print("WARNING: Predictions very similar to observations - possible data leakage or overfitting")


def _analyze_model_effects(posterior_samples):
    """Analyze model effects from posterior samples."""
    print("\n=== Model Effect Analysis ===")
    try:
        # Extract model effects if available
        model_effect = posterior_samples.get("model_effect")
        if model_effect is not None:
            model_effect_mean = np.array(model_effect.mean(axis=0))
            model_effect_std = np.array(model_effect.std(axis=0))
            print(f"Model effect range: {model_effect_mean.min():.4f} to {model_effect_mean.max():.4f}")
            print(f"Model effect mean std: {model_effect_std.mean():.4f}")
            
            # Test for normality
            _, p_value = stats.normaltest(model_effect_mean)
            print(f"Model effect normality test p-value: {p_value:.4f}")
            
            # Compare top and bottom models
            if len(model_effect_mean) >= 10:
                top_effects = np.sort(model_effect_mean)[-5:]
                bottom_effects = np.sort(model_effect_mean)[:5]
                effect_diff = top_effects.mean() - bottom_effects.mean()
                print(f"Difference between top 5 and bottom 5 models: {effect_diff:.4f}")
    except Exception as e:
        print(f"Could not analyze model effects: {e}")


def visualize_effects(model_performance, task_type_df, data_source_df,
                      evaluation_df, output_dir=None, task_id_df=None):
    """
    Visualize component effects on performance metrics.
    
    Args:
        model_performance: DataFrame with model performance data
        task_type_df: DataFrame with task type effects
        data_source_df: DataFrame with data source effects
        evaluation_df: DataFrame with evaluation type effects
        output_dir: Directory to save plots
        task_id_df: DataFrame with task ID effects
    """
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Model Effects
    if model_performance is not None and not model_performance.empty:
        _visualize_model_effects(model_performance, output_dir)

    # 2. Task Type Effects
    if task_type_df is not None and not task_type_df.empty:
        _visualize_category_effects(task_type_df, 'Task Type', 'lightgreen', output_dir, 
                                   filename='task_type_effects.png')
    
    # 3. Data Source Effects
    if data_source_df is not None and not data_source_df.empty:
        _visualize_category_effects(data_source_df, 'Data Source', 'salmon', output_dir,
                                   filename='data_source_effects.png')
    
    # 4. Evaluation Type Effects
    if evaluation_df is not None and not evaluation_df.empty:
        _visualize_category_effects(evaluation_df, 'Evaluation Type', 'mediumpurple', output_dir,
                                   filename='evaluation_effects.png')
    
    # 5. Task ID Effects
    if task_id_df is not None and not task_id_df.empty:
        _visualize_task_id_effects(task_id_df, output_dir)


def _visualize_model_effects(model_performance, output_dir):
    """Visualize model effects on performance."""
    # Filter for high connectivity models
    high_connectivity_models = model_performance[model_performance['high_connectivity'] == True]
    if len(high_connectivity_models) == 0:
        print("Warning: No models with high_connectivity=True found. Showing all models.")
        high_connectivity_models = model_performance
    
    # Determine which value to plot
    plot_value, effect_label = _get_plot_value_and_label(high_connectivity_models)
    
    # Plot high connectivity models
    plt.figure(figsize=(12, len(high_connectivity_models) * 0.3 + 2))

    # Sort by plot_value in descending order (highest performance at top)
    model_performance_sorted = high_connectivity_models.sort_values(plot_value, ascending=True)

    # Plot horizontal bar chart
    bars = plt.barh(model_performance_sorted['model'],
                    model_performance_sorted[plot_value],
                    color='skyblue')
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
              f'{width:.2f}', va='center')

    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'Model Effects on Performance (High Connectivity Models) - {effect_label}',
              fontsize=14)
    plt.xlabel(effect_label)
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'model_effects_high_connectivity.png')
    plt.show()
    
    # Also create a visualization with all models for comparison
    _visualize_all_models(model_performance, plot_value, effect_label, output_dir)


def _get_plot_value_and_label(models_df):
    """Determine which value to plot and the appropriate label."""
    if 'mu_original_scale' in models_df.columns and not models_df['mu_original_scale'].isna().all():
        return 'mu_original_scale', 'Model Performance (Original Scale)'
    elif 'clipped_prediction' in models_df.columns and not models_df['clipped_prediction'].isna().all():
        return 'clipped_prediction', 'Clipped Prediction'
    else:
        return 'effect', 'Effect (in metric units)'


def _visualize_all_models(model_performance, plot_value, effect_label, output_dir):
    """Visualize effects for all models."""
    plt.figure(figsize=(12, len(model_performance) * 0.3 + 2))
    
    # Sort by effect for better visualization
    all_models_sorted = model_performance.sort_values(plot_value, ascending=True)
    
    # Use different colors for high vs low connectivity models
    colors = ['skyblue' if hc else 'lightgray' for hc in all_models_sorted['high_connectivity']]
    
    # Plot horizontal bar chart
    bars = plt.barh(all_models_sorted['model'], 
                  all_models_sorted[plot_value],
                 color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
              f'{width:.2f}', va='center')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'Model Effects on Performance (All Models) - {effect_label}', fontsize=14)
    plt.xlabel(effect_label)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='High Connectivity'),
        Patch(facecolor='lightgray', label='Low Connectivity')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'model_effects_all.png')
    plt.show()


def _visualize_category_effects(category_df, category_type, color, output_dir, filename):
    """Visualize effects for categorical variables."""
    plt.figure(figsize=(10, len(category_df) * 0.4 + 2))
    
    # Sort by effect for better visualization
    df_sorted = category_df.sort_values('effect', ascending=True)
    
    # Plot horizontal bar chart
    bars = plt.barh(df_sorted['category'], 
                    df_sorted['effect'],
                    color=color)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'{category_type} Effects on Performance', fontsize=14)
    plt.xlabel('Effect (in metric units)')
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, filename)
    plt.show()


def _visualize_task_id_effects(task_id_df, output_dir):
    """Visualize task ID effects."""
    # If too many task IDs, limit to top and bottom 20
    if len(task_id_df) > 40:
        print(f"Too many task IDs to display ({len(task_id_df)}). Showing top and bottom 20.")
        top_tasks = task_id_df.nlargest(TOP_BOTTOM_TASK_LIMIT, 'effect')
        bottom_tasks = task_id_df.nsmallest(TOP_BOTTOM_TASK_LIMIT, 'effect')
        task_id_df_display = pd.concat([top_tasks, bottom_tasks])
    else:
        task_id_df_display = task_id_df

    plt.figure(figsize=(10, len(task_id_df_display) * 0.3 + 2))
    
    # Sort by effect for better visualization
    task_id_df_sorted = task_id_df_display.sort_values('effect', ascending=True)
    
    # Plot horizontal bar chart
    bars = plt.barh(task_id_df_sorted['category'], 
                    task_id_df_sorted['effect'],
                    color='gold')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Task ID Effects on Performance', fontsize=14)
    plt.xlabel('Effect (in metric units)')
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'task_id_effects.png')
    plt.show()


def save_results(model_performance, task_type_df, data_source_df,
                 evaluation_df, output_dir, task_id_df=None):
    """
    Save analysis results to CSV files.
    
    Args:
        model_performance: DataFrame with model performance data
        task_type_df: DataFrame with task type effects
        data_source_df: DataFrame with data source effects
        evaluation_df: DataFrame with evaluation type effects
        output_dir: Directory to save results
        task_id_df: DataFrame with task ID effects
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save each dataframe to CSV
    model_performance.to_csv(os.path.join(output_dir, 'model_effects.csv'), index=False)
    
    if task_type_df is not None:
        task_type_df.to_csv(os.path.join(output_dir, 'task_type_effects.csv'), index=False)
    
    if data_source_df is not None:
        data_source_df.to_csv(os.path.join(output_dir, 'data_source_effects.csv'), index=False)
    
    if evaluation_df is not None:
        evaluation_df.to_csv(os.path.join(output_dir, 'evaluation_effects.csv'), index=False)
    
    # Save task_id effects if available
    if task_id_df is not None:
        task_id_df.to_csv(os.path.join(output_dir, 'task_id_effects.csv'), index=False)
    
    # Save selected model performance data to pickle
    _save_model_performance_pickle(model_performance, output_dir)

    print(f"Results saved to {output_dir}")


def _save_model_performance_pickle(model_performance, output_dir):
    """Save selected model performance data to pickle file."""
    # Select only the required columns
    selected_data = model_performance[['model', 'effect', 'high_connectivity']] 
    
    # Add mu_original_scale if available
    if 'mu_original_scale' in model_performance.columns:
        selected_data['original_scale'] = model_performance['mu_original_scale']
    
    # Save to pickle file
    pickle_path = os.path.join(output_dir, 'model_performance.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(selected_data, f)
    
    print(f"Model performance data saved to {pickle_path}")


def visualize_residuals(observed, predicted, residuals, output_dir=None):
    """
    Visualize model residuals.
    
    Args:
        observed: Observed values
        predicted: Predicted values
        residuals: Residuals (observed - predicted)
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 10))
    
    # 1. Residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add LOWESS smoother if available
    _add_lowess_smoother(predicted, residuals)
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 2. Histogram of Residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', density=True)
    
    # Add normal curve for comparison
    x = np.linspace(min(residuals), max(residuals), 100)
    mean, std = np.mean(residuals), np.std(residuals)
    plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
    
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Observed vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(observed, predicted, alpha=0.5)
    
    # Add diagonal line
    min_val = min(min(observed), min(predicted))
    max_val = max(max(observed), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values')
    plt.title('Observed vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        _save_plot(output_dir, 'residual_diagnostics.png')
    
    plt.show()
    
    # Print summary statistics for residuals
    _print_residual_diagnostics(residuals)


def _add_lowess_smoother(x, y):
    """Add LOWESS smoother to the plot if statsmodels is available."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y, x, frac=0.2)
        plt.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
    except ImportError:
        print("Skipping LOWESS smoother - statsmodels not available")


def _print_residual_diagnostics(residuals):
    """Print diagnostic statistics for residuals."""
    print("\n=== Residual Diagnostics ===")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Std of residuals: {np.std(residuals):.4f}")
    print(f"Min of residuals: {np.min(residuals):.4f}")
    print(f"Max of residuals: {np.max(residuals):.4f}")
    
    # Shapiro-Wilk test for normality (with sample size limit)
    if len(residuals) <= 5000:  # Shapiro-Wilk has a sample size limit
        stat, p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk test: stat={stat:.4f}, p-value={p:.4f}")
        if p < 0.05:
            print("  Residuals are not normally distributed (p < 0.05)")
        else:
            print("  Residuals appear normally distributed (p >= 0.05)")