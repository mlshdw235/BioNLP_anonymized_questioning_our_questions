    """
Adjusts model performance metrics through Bayesian modeling.
This module implements hierarchical Bayesian models to account for various factors
affecting model performance in machine learning evaluation tasks.
"""
import traceback
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils_bayesian import (
    setup_environment,
    load_and_prepare_data,
    plot_metric_distribution,
    get_major_models,
    compute_model_connectivity,
    HIGH_CONNECTIVITY_THRESHOLD,
    check_convergence,
    plot_qq,
    visualize_effects,
    visualize_residuals,
    save_results
)

# Global configuration settings
OUTPUT_DIR = './bayesian_analysis_results'
MODELING_PARAMS = [
    'task_type',
    'data_source',
    'evaluation'
]
DROP_MODELS_WITH_ONE_RECORDS = True
DROP_MODELS_WITH_TWO_RECORDS = False
MODELING_TASK_ID = False
NUM_CHAINS = 8  # Number of MCMC chains to run in parallel
MCMC_WARMUP_STEPS = 1000
MCMC_SAMPLE_STEPS = 2000
VALIDATION_SUBSET_SIZE = 0.5  # Fraction of data to use for validation

# Initialize NumPyro with the specified number of chains
numpyro.set_host_device_count(NUM_CHAINS)

print(f"HIGH_CONNECTIVITY_THRESHOLD: upper {100 - HIGH_CONNECTIVITY_THRESHOLD}% ")


def run_non_mcmc_estimation(df):
    """
    Use a simple linear regression approach as a fallback when MCMC fails.
    
    Args:
        df: DataFrame with prepared data
        
    Returns:
        Tuple of DataFrames with model performance and category effects
    """
    # Extract features
    features = ['task_type_code', 'data_source_code', 'evaluation_type_code']
    
    # Add task_id to features if it's being modeled
    if MODELING_TASK_ID and 'task_id' in df.columns:
        features.append('task_id')
        
    X = pd.get_dummies(df[features])

    # Create model dummy variables
    model_dummies = pd.get_dummies(df['model_full_name2_code'])
    X = pd.concat([X, model_dummies], axis=1)

    # Fix for sklearn feature names error - convert all column names to strings
    X.columns = X.columns.astype(str)

    # Print dimensions for debugging
    print(f"Non-MCMC approach - Sample count: {len(df)}")
    print(f"Non-MCMC approach - Feature count: {X.shape[1]}")
    print(f"Non-MCMC approach - Sample:Feature ratio: {len(df)/X.shape[1]:.2f}")

    # Target variable
    y = df['metric_value_scaled']

    # Fit regression
    model = LinearRegression()
    model.fit(X, y)

    # Extract model coefficients
    model_cols = [col for col in model_dummies.columns.astype(str)]
    model_effects = model.coef_[-len(model_cols):]

    # Map indices back to model names
    model_effects_df = _create_model_effects_dataframe(df, model_cols, model_effects)

    # Extract other feature effects and map them to categories
    task_type_df, data_source_df, evaluation_df, task_id_df = _extract_category_effects(
        df, X.columns[:-len(model_cols)], model.coef_[:len(X.columns)-len(model_cols)]
    )
        
    # Calculate model predictions for residual analysis
    predictions = model.predict(X)
    residuals = y - predictions

    return model_effects_df, task_type_df, data_source_df, evaluation_df, task_id_df, residuals, predictions


def _create_model_effects_dataframe(df, model_cols, model_effects):
    """
    Create a DataFrame mapping model indices to their names and effects.
    
    Args:
        df: Source DataFrame with model information
        model_cols: Column names for model dummy variables
        model_effects: Effect values for each model
        
    Returns:
        DataFrame with model names and their effects
    """
    # Map back to model names
    model_mapping = dict(zip(df['model_full_name2_code'], df['model_full_name2']))
    model_names = []
    
    for col in model_cols:
        try:
            idx = int(col.split('_')[-1])
            model_names.append(model_mapping.get(idx, f"Unknown-{idx}"))
        except ValueError:
            model_names.append(f"Unknown-{col}")

    # Create results dataframe
    model_performance = pd.DataFrame({
        'model': model_names,
        'effect': model_effects
    })

    # Sort by effect in descending order
    return model_performance.sort_values('effect', ascending=False)


def _extract_category_effects(df, feature_names, feature_effects):
    """
    Extract and map effect values for various categorical features.
    
    Args:
        df: Source DataFrame with category information
        feature_names: Names of feature columns
        feature_effects: Effect values for each feature
        
    Returns:
        Tuple of DataFrames for each category type
    """
    # Initialize effect dictionaries
    task_type_effects = {}
    data_source_effects = {}
    evaluation_effects = {}
    task_id_effects = {}
    
    # Map effects to their original categories
    for feature, effect in zip(feature_names, feature_effects):
        if feature.startswith('task_type_code_'):
            _map_effect_to_category(df, feature, effect, 'task_type', task_type_effects)
        elif feature.startswith('data_source_code_'):
            _map_effect_to_category(df, feature, effect, 'data_source', data_source_effects)
        elif feature.startswith('evaluation_type_code_'):
            _map_effect_to_category(df, feature, effect, 'evaluation_type', evaluation_effects)
        elif MODELING_TASK_ID and feature.startswith('task_id_'):
            _map_effect_to_category(df, feature, effect, 'task_name', task_id_effects, 'task_id')

    # Create DataFrames for each effect type
    task_type_df = _create_category_effect_dataframe(task_type_effects)
    data_source_df = _create_category_effect_dataframe(data_source_effects)
    evaluation_df = _create_category_effect_dataframe(evaluation_effects)
    
    # Create task_id DataFrame if modeled
    if MODELING_TASK_ID and task_id_effects:
        task_id_df = _create_category_effect_dataframe(task_id_effects)
    else:
        task_id_df = None
        
    return task_type_df, data_source_df, evaluation_df, task_id_df


def _map_effect_to_category(df, feature, effect, category_col, effect_dict, code_col=None):
    """
    Map an effect value to its corresponding category name.
    
    Args:
        df: Source DataFrame
        feature: Feature column name
        effect: Effect value
        category_col: Column containing category names
        effect_dict: Dictionary to store the mapping
        code_col: Optional column with category codes (if different from category_col+'_code')
    """
    try:
        idx = int(feature.split('_')[-1])
        code_column = code_col if code_col else f"{category_col}_code"
        category = df[df[code_column] == idx][category_col].iloc[0]
        effect_dict[category] = effect
    except (ValueError, IndexError):
        pass


def _create_category_effect_dataframe(effect_dict):
    """
    Create a DataFrame from a dictionary of category effects.
    
    Args:
        effect_dict: Dictionary mapping categories to effects
        
    Returns:
        DataFrame with categories and their effects
    """
    return pd.DataFrame({
        'category': list(effect_dict.keys()),
        'effect': list(effect_dict.values())
    }).sort_values('effect', ascending=False)


def try_jax_numpyro_approach(df):
    """
    Implement a Bayesian hierarchical model using NumPyro with JAX backend.
    Uses a Normal distribution for modeling performance metrics directly.
    
    Args:
        df: DataFrame with prepared data
        
    Returns:
        Tuple containing model performance results and diagnostics
    """
    try:
        # Print version info for debugging
        print(f"JAX version: {jax.__version__}")
        print(f"NumPyro version: {numpyro.__version__}")
        
        # Extract indices for categorical variables
        category_indices = _extract_category_indices(df)
        
        # Use scaled metric values directly
        perf_values = jnp.array(df['metric_value_scaled'].values)

        # Get dimension sizes and print model statistics
        dimensions = _get_model_dimensions(df, category_indices)
        _print_model_statistics(df, dimensions)
        
        # Define the Bayesian model
        model = _create_bayesian_model(df, category_indices, perf_values, dimensions)

        # Prepare model data dictionary
        model_data = _prepare_model_data(category_indices, perf_values)

        # Run inference
        mcmc = _run_mcmc_inference(model, model_data)

        # Check MCMC convergence
        convergence_results = check_convergence(mcmc)
        
        # Extract posterior samples
        samples = mcmc.get_samples()

        # Extract and process effects from samples
        effects = _extract_effects_from_samples(samples, dimensions)
        
        # Extract predictions
        predicted_values = _extract_predictions(samples, category_indices['model_idx'])
        
        # Rescale effects and create result DataFrames
        model_results = _create_result_dataframes(df, effects, dimensions, predicted_values)
        
        # Extract residuals and predicted observations (if available)
        residuals = None
        predicted_obs = None

        # Plot Q-Q plot for model residuals
        if residuals is not None:
            plot_qq(residuals, "Q-Q Plot of Model Residuals", OUTPUT_DIR)
        
        return (*model_results, residuals, predicted_obs, convergence_results)

    except Exception as e:
        print(f"JAX/NumPyro approach failed: {str(e)}")
        traceback.print_exc()
        return None


def _extract_category_indices(df):
    """
    Extract category indices from the DataFrame.
    
    Args:
        df: Source DataFrame
        
    Returns:
        Dictionary of category indices
    """
    indices = {
        'task_type_idx': jnp.array(df['task_type_code'].values),
        'data_source_idx': jnp.array(df['data_source_code'].values),
        'evaluation_idx': jnp.array(df['evaluation_type_code'].values),
        'model_idx': jnp.array(df['model_full_name2_code'].values),
    }
    
    # Add task_id if modeling it
    if MODELING_TASK_ID and 'task_id' in df.columns:
        indices['task_id_idx'] = jnp.array(df['task_id'].values)
    
    return indices


def _get_model_dimensions(df, category_indices):
    """
    Calculate dimensions for different categorical variables.
    
    Args:
        df: Source DataFrame
        category_indices: Dictionary of category indices
        
    Returns:
        Dictionary of dimension sizes
    """
    dimensions = {
        'n_models': len(df['model_full_name2_code'].unique()),
        'n_task_types': len(df['task_type_code'].unique()) if 'task_type' in MODELING_PARAMS else 0,
        'n_data_sources': len(df['data_source_code'].unique()) if 'data_source' in MODELING_PARAMS else 0,
        'n_evaluation_methods': len(df['evaluation_type_code'].unique()) if 'evaluation' in MODELING_PARAMS else 0,
        'n_task_ids': 0
    }
    
    # Calculate task ID dimension if modeling it
    if MODELING_TASK_ID and 'task_id_idx' in category_indices:
        dimensions['n_task_ids'] = len(df['task_id'].unique())
        print(f"Unique task IDs: {dimensions['n_task_ids']}")
    
    return dimensions


def _print_model_statistics(df, dimensions):
    """
    Print statistical information about the model dimensions.
    
    Args:
        df: Source DataFrame
        dimensions: Dictionary of dimension sizes
    """
    print(f"Sample count: {len(df)}")
    
    if 'task_type' in MODELING_PARAMS:
        print(f"Unique task types: {dimensions['n_task_types']}")
    if 'data_source' in MODELING_PARAMS:
        print(f"Unique data sources: {dimensions['n_data_sources']}")
    if 'evaluation' in MODELING_PARAMS:
        print(f"Unique evaluation methods: {dimensions['n_evaluation_methods']}")
    
    print(f"Unique models: {dimensions['n_models']}")
    
    # Calculate total parameter count
    total_params = dimensions['n_models']
    if 'task_type' in MODELING_PARAMS:
        total_params += dimensions['n_task_types'] + 1  # +1 for sigma_type
    if 'data_source' in MODELING_PARAMS:
        total_params += dimensions['n_data_sources'] + 1  # +1 for sigma_source
    if 'evaluation' in MODELING_PARAMS:
        total_params += dimensions['n_evaluation_methods'] + 1  # +1 for sigma_eval
    if MODELING_TASK_ID and dimensions['n_task_ids'] > 0:
        total_params += dimensions['n_task_ids'] + 1  # +1 for sigma_task_id
    
    total_params += 2  # Add 2 for other sigma parameters (sigma_model, sigma_obs)
    
    print(f"Total parameters: {total_params}")
    print(f"Sample:Parameter ratio: {len(df)/total_params:.2f}")
    print(f"Using {NUM_CHAINS} MCMC chains based on available devices")


def _create_bayesian_model(df, category_indices, perf_values, dimensions):
    """
    Create a Bayesian hierarchical model for performance metrics.
    
    Args:
        df: Source DataFrame
        category_indices: Dictionary of category indices
        perf_values: Array of performance values
        dimensions: Dictionary of dimension sizes
        
    Returns:
        Bayesian model function
    """
    # Get metric mean and std for rescaling
    metric_mean = df['metric_value'].mean()
    metric_std = df['metric_value'].std()
    
    # Extract indices for easier access
    task_type_idx = category_indices['task_type_idx']
    data_source_idx = category_indices['data_source_idx']
    evaluation_idx = category_indices['evaluation_idx']
    model_idx = category_indices['model_idx']
    task_id_idx = category_indices.get('task_id_idx', None)
    
    # Extract dimensions
    n_task_types = dimensions['n_task_types']
    n_data_sources = dimensions['n_data_sources']
    n_evaluation_methods = dimensions['n_evaluation_methods']
    n_models = dimensions['n_models']
    n_task_ids = dimensions['n_task_ids']

    def model(task_type_idx=task_type_idx, data_source_idx=data_source_idx,
              evaluation_idx=evaluation_idx, model_idx=model_idx,
              task_id_idx=task_id_idx, obs=perf_values):
        """
        Bayesian hierarchical model for normalized performance metrics.
        
        This model works with pre-normalized data (mean=0, std=1) and:
        - Uses Normal distribution with mean 0 for task effects
        - Allows model effects to have a flexible mean parameter
        - Optionally includes task ID effects
        """
        # Hyperpriors for hierarchical variance
        sigma_model = numpyro.sample("sigma_model", dist.HalfNormal(1))
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1))
        
        # Add hyperpriors based on MODELING_PARAMS
        if 'task_type' in MODELING_PARAMS:
            sigma_type = numpyro.sample("sigma_type", dist.HalfNormal(1))
        
        if 'data_source' in MODELING_PARAMS:
            sigma_source = numpyro.sample("sigma_source", dist.HalfNormal(1))
        
        if 'evaluation' in MODELING_PARAMS:
            sigma_eval = numpyro.sample("sigma_eval", dist.HalfNormal(1))
        
        # Add hyperprior for task ID if modeling it
        if MODELING_TASK_ID and task_id_idx is not None:
            sigma_task_id = numpyro.sample("sigma_task_id", dist.HalfNormal(0.2))

        # Model effects - with a global offset parameter
        model_effect_mean = numpyro.sample("model_effect_mean", dist.Normal(0.0, 1))
        model_effect = numpyro.sample("model_effect",
                                      dist.Normal(model_effect_mean, sigma_model),
                                      sample_shape=(n_models,))
        
        # Task effects (Hierarchical decomposition)
        task_effect_components = []
        
        if 'task_type' in MODELING_PARAMS:
            task_type_effect = numpyro.sample("task_type_effect",
                                            dist.Normal(0, sigma_type),
                                            sample_shape=(n_task_types,))
            task_effect_components.append(task_type_effect[task_type_idx])
        
        if 'data_source' in MODELING_PARAMS:
            data_source_effect = numpyro.sample("data_source_effect",
                                                dist.Normal(0, sigma_source),
                                                sample_shape=(n_data_sources,))
            task_effect_components.append(data_source_effect[data_source_idx])
        
        if 'evaluation' in MODELING_PARAMS:
            evaluation_effect = numpyro.sample("evaluation_effect",
                                            dist.Normal(0, sigma_eval),
                                            sample_shape=(n_evaluation_methods,))
            task_effect_components.append(evaluation_effect[evaluation_idx])

        # Add task_id_effect if needed
        if MODELING_TASK_ID and task_id_idx is not None:
            task_id_effect = numpyro.sample("task_id_effect",
                                            dist.Normal(0, sigma_task_id),
                                            sample_shape=(n_task_ids,))
            task_effect_components.append(task_id_effect[task_id_idx])
        
        # Calculate task effect as the sum of components
        task_effect = sum(task_effect_components) if task_effect_components else jnp.zeros_like(model_idx, dtype=jnp.float32)

        # Compute mean performance estimate in normalized scale
        mu = model_effect[model_idx] + task_effect + numpyro.sample("mu_noise", dist.Normal(0, 0.1))

        # Transform model predictions back to original scale for interpretation
        mu_original_scale = mu * metric_std + metric_mean
        numpyro.deterministic("mu_original_scale", mu_original_scale)

        # Normal likelihood on normalized scale
        with numpyro.handlers.scale(scale=0.9):  # Downweight the likelihood slightly for robustness
            numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=obs)

        # Prepare effects to return
        effects = [model_effect]
        if 'task_type' in MODELING_PARAMS:
            effects.append(task_type_effect)
        if 'data_source' in MODELING_PARAMS:
            effects.append(data_source_effect)
        if 'evaluation' in MODELING_PARAMS:
            effects.append(evaluation_effect)
        if MODELING_TASK_ID and task_id_idx is not None:
            effects.append(task_id_effect)

        return tuple(effects)
    
    return model


def _prepare_model_data(category_indices, perf_values):
    """
    Prepare data dictionary for the Bayesian model.
    
    Args:
        category_indices: Dictionary of category indices
        perf_values: Array of performance values
        
    Returns:
        Dictionary of model data
    """
    model_data = {
        'model_idx': category_indices['model_idx'],
        'obs': perf_values
    }
    
    if 'task_type' in MODELING_PARAMS:
        model_data['task_type_idx'] = category_indices['task_type_idx']
    if 'data_source' in MODELING_PARAMS:
        model_data['data_source_idx'] = category_indices['data_source_idx']
    if 'evaluation' in MODELING_PARAMS:
        model_data['evaluation_idx'] = category_indices['evaluation_idx']
    if MODELING_TASK_ID and 'task_id_idx' in category_indices:
        model_data['task_id_idx'] = category_indices['task_id_idx']
    
    return model_data


def _run_mcmc_inference(model, model_data):
    """
    Run MCMC inference using the NUTS sampler.
    
    Args:
        model: Bayesian model function
        model_data: Dictionary of model data
        
    Returns:
        MCMC object with samples
    """
    rng_key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
    
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        target_accept_prob=0.8,  # Standard target acceptance rate
        init_strategy=init_to_median()  # Use median initialization for stability
    )
    
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=MCMC_WARMUP_STEPS, 
        num_samples=MCMC_SAMPLE_STEPS,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        progress_bar=True
    )
    
    mcmc.run(rng_key, **model_data)
    return mcmc


def _extract_effects_from_samples(samples, dimensions):
    """
    Extract effect values from MCMC samples.
    
    Args:
        samples: Dictionary of MCMC samples
        dimensions: Dictionary of dimension sizes
        
    Returns:
        Dictionary of effect values
    """
    effects = {
        'model_effects': samples["model_effect"].mean(axis=0),
    }
    
    if 'task_type' in MODELING_PARAMS:
        effects['task_type_effects'] = samples["task_type_effect"].mean(axis=0)
    
    if 'data_source' in MODELING_PARAMS:
        effects['data_source_effects'] = samples["data_source_effect"].mean(axis=0)
    
    if 'evaluation' in MODELING_PARAMS:
        effects['evaluation_effects'] = samples["evaluation_effect"].mean(axis=0)
    
    if MODELING_TASK_ID and 'task_id_effect' in samples:
        effects['task_id_effects'] = samples["task_id_effect"].mean(axis=0)
    
    return effects


def _extract_predictions(samples, model_idx):
    """
    Extract and aggregate predictions from MCMC samples.
    
    Args:
        samples: Dictionary of MCMC samples
        model_idx: Array of model indices
        
    Returns:
        Dictionary of prediction values
    """
    predictions = {}
    
    # Extract mu_original_scale predictions
    mu_original_scale_values = samples.get("mu_original_scale", None)
    if mu_original_scale_values is not None:
        predictions['model_original_scale'] = _aggregate_predictions_by_model(
            mu_original_scale_values, model_idx
        )
    
    # Extract clipped predictions if available
    mu_clipped_values = samples.get("mu_clipped", None)
    if mu_clipped_values is not None:
        predictions['model_clipped'] = _aggregate_predictions_by_model(
            mu_clipped_values, model_idx
        )
    
    return predictions


def _aggregate_predictions_by_model(prediction_values, model_idx):
    """
    Aggregate prediction values by model index.
    
    Args:
        prediction_values: Array of prediction values
        model_idx: Array of model indices
        
    Returns:
        Dictionary mapping model indices to average predictions
    """
    # Create a dict to store sum and count for each model index
    model_to_prediction = {}
    
    # Go through all observations
    for i, model_id in enumerate(model_idx):
        model_id_int = int(model_id)
        if model_id_int not in model_to_prediction:
            model_to_prediction[model_id_int] = {"sum": 0.0, "count": 0}
        
        # Add all values for this model across all samples
        model_to_prediction[model_id_int]["sum"] += float(prediction_values[:, i].mean())
        model_to_prediction[model_id_int]["count"] += 1
    
    # Compute average prediction for each model
    return {
        model_id: data["sum"] / data["count"]
        for model_id, data in model_to_prediction.items()
    }


def _create_result_dataframes(df, effects, dimensions, predicted_values):
    """
    Create DataFrames for model and category effects with appropriate rescaling.
    
    Args:
        df: Source DataFrame
        effects: Dictionary of effect values
        dimensions: Dictionary of dimension sizes
        predicted_values: Dictionary of prediction values
        
    Returns:
        Tuple of DataFrames for model and category effects
    """
    # Get metric mean and std for rescaling
    metric_std = df['metric_value'].std()
    
    # Rescale effects to original metric scale
    model_effects_rescaled = effects['model_effects'] * metric_std
    
    # Create connectivity mapping
    model_names, connectivity_info = _get_model_names_and_connectivity(df, dimensions['n_models'])
    
    # Create model performance DataFrame
    model_performance = pd.DataFrame({
        'model': model_names,
        'effect_scaled': effects['model_effects'],
        'effect': model_effects_rescaled,
        'high_connectivity': [connectivity_info.get(i, False) for i in range(dimensions['n_models'])]
    })
    
    # Add predictions if available
    if 'model_original_scale' in predicted_values:
        model_performance['mu_original_scale'] = [
            predicted_values['model_original_scale'].get(i, None) for i in range(dimensions['n_models'])
        ]
    
    if 'model_clipped' in predicted_values:
        model_performance['clipped_prediction'] = [
            predicted_values['model_clipped'].get(i, None) for i in range(dimensions['n_models'])
        ]
    
    # Sort by effect
    model_performance = model_performance.sort_values('effect', ascending=False)
    
    # Create DataFrames for category effects
    task_type_df, data_source_df, evaluation_df, task_id_df = _create_category_effect_dataframes(
        df, effects, dimensions, metric_std
    )
    
    return model_performance, task_type_df, data_source_df, evaluation_df, task_id_df


def _get_model_names_and_connectivity(df, n_models):
    """
    Map model indices to names and connectivity information.
    
    Args:
        df: Source DataFrame
        n_models: Number of unique models
        
    Returns:
        Tuple of model names array and connectivity dictionary
    """
    # Map to model names
    model_mapping = dict(zip(df['model_full_name2_code'], df['model_full_name2']))
    model_names = [model_mapping.get(i) for i in range(n_models)]
    
    # Get connectivity mapping
    high_connectivity_mapping = dict(zip(
        df['model_full_name2'], df['high_connectivity']
    ))
    
    # Create connectivity info
    connectivity_info = {
        i: high_connectivity_mapping.get(model_name, False) 
        for i, model_name in enumerate(model_names)
    }
    
    return model_names, connectivity_info


def _create_category_effect_dataframes(df, effects, dimensions, metric_std):
    """
    Create DataFrames for all category effects.
    
    Args:
        df: Source DataFrame
        effects: Dictionary of effect values
        dimensions: Dictionary of dimension sizes
        metric_std: Standard deviation of the original metric
        
    Returns:
        Tuple of DataFrames for category effects
    """
    task_type_df = None
    if 'task_type' in MODELING_PARAMS:
        task_type_names = _map_indices_to_categories(df, 'task_type', dimensions['n_task_types'])
        task_type_df = pd.DataFrame({
            'category': task_type_names,
            'effect_scaled': effects['task_type_effects'],
            'effect': effects['task_type_effects'] * metric_std
        }).sort_values('effect', ascending=False)
    
    data_source_df = None
    if 'data_source' in MODELING_PARAMS:
        data_source_names = _map_indices_to_categories(df, 'data_source', dimensions['n_data_sources'])
        data_source_df = pd.DataFrame({
            'category': data_source_names,
            'effect_scaled': effects['data_source_effects'],
            'effect': effects['data_source_effects'] * metric_std
        }).sort_values('effect', ascending=False)
    
    evaluation_df = None
    if 'evaluation' in MODELING_PARAMS:
        evaluation_names = _map_indices_to_categories(df, 'evaluation_type', dimensions['n_evaluation_methods'])
        evaluation_df = pd.DataFrame({
            'category': evaluation_names,
            'effect_scaled': effects['evaluation_effects'],
            'effect': effects['evaluation_effects'] * metric_std
        }).sort_values('effect', ascending=False)
    
    task_id_df = None
    if MODELING_TASK_ID and 'task_id_effects' in effects:
        task_id_names = _map_indices_to_categories(df, 'task_name', dimensions['n_task_ids'], 'task_id')
        task_id_df = pd.DataFrame({
            'category': task_id_names,
            'effect_scaled': effects['task_id_effects'],
            'effect': effects['task_id_effects'] * metric_std
        }).sort_values('effect', ascending=False)
    
    return task_type_df, data_source_df, evaluation_df, task_id_df


def _map_indices_to_categories(df, category_col, count, code_col=None):
    """
    Map indices to their corresponding category names.
    
    Args:
        df: Source DataFrame
        category_col: Column containing category names
        count: Number of unique categories
        code_col: Optional column with category codes (defaults to category_col+'_code')
        
    Returns:
        List of category names
    """
    code_column = code_col if code_col else f"{category_col}_code"
    category_mapping = dict(zip(df[code_column], df[category_col]))
    return [category_mapping.get(i, f'Unknown-{i}') for i in range(count)]


def run_validation_mcmc(df, num_chains=2):
    """
    Run a validation MCMC on a subset of data to quickly check if the model works.
    
    Args:
        df: Prepared dataframe with coded indices
        num_chains: Number of MCMC chains for validation
        
    Returns:
        Results from the non-MCMC estimation on a subset of data
    """
    # Take a random subset of the data
    subset_idx = np.random.choice(
        np.arange(len(df)), 
        size=int(len(df) * VALIDATION_SUBSET_SIZE), 
        replace=False
    )
    df_subset = df.iloc[subset_idx].reset_index(drop=True)
    
    # Run with a simple approach for quick validation
    return run_non_mcmc_estimation(df_subset)


def main():
    """
    Main function that coordinates the Bayesian modeling process.
    
    Returns:
        Analysis results or None if an error occurs
    """
    try:
        # Setup environment first
        setup_environment()

        # Load and prepare data
        df = load_and_prepare_data(
            filepath=None,  # Should be provided by the calling code
            drop_models_with_one_records=DROP_MODELS_WITH_ONE_RECORDS,
            drop_models_with_two_records=DROP_MODELS_WITH_TWO_RECORDS,
            modeling_task_id=MODELING_TASK_ID
        )
        print(f"Loaded data with {len(df)} records")

        # Ensure all required code columns exist
        _prepare_code_columns(df)

        # Try NumPyro approach first
        print("Attempting JAX/NumPyro approach...")
        result = try_jax_numpyro_approach(df)

        # If NumPyro fails, fall back to non-MCMC approach
        if result is None:
            print("Falling back to non-MCMC approach...")
            result = run_non_mcmc_estimation(df)
            
            # Unpack results (non-MCMC approach)
            model_performance, task_type_df, data_source_df, evaluation_df, task_id_df, residuals, predictions = result
            
            # Visualize residuals
            visualize_residuals(df['metric_value_scaled'].values, predictions, residuals, OUTPUT_DIR)
        else:
            # Unpack results (MCMC approach)
            (model_performance, task_type_df, data_source_df,
            evaluation_df, task_id_df, residuals, predictions,
            convergence_results) = result
            
            # Visualize residuals (if available)
            if residuals is not None:
                visualize_residuals(df['metric_value_scaled'].values, predictions, residuals, OUTPUT_DIR)

        # Print results
        _print_analysis_results(model_performance, task_type_df, data_source_df, evaluation_df, task_id_df)

        # Visualize effects
        visualize_effects(
            model_performance, task_type_df, data_source_df, evaluation_df, 
            OUTPUT_DIR, task_id_df=task_id_df if MODELING_TASK_ID else None
        )

        # Save results
        save_results(
            model_performance, task_type_df, data_source_df, evaluation_df, 
            OUTPUT_DIR, task_id_df=task_id_df if MODELING_TASK_ID else None
        )

        return result

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return None


def _prepare_code_columns(df):
    """
    Ensure all required code columns exist in the DataFrame.
    
    Args:
        df: Source DataFrame to check and modify
    """
    for col_base in ['task_type', 'data_source', 'evaluation_type', 'model_full_name2', 'task_name']:
        code_col = f'{col_base}_code'
        if code_col not in df.columns:
            print(f"Creating missing column {code_col}")
            df[code_col] = df[col_base].astype('category').cat.codes


def _print_analysis_results(model_performance, task_type_df, data_source_df, evaluation_df, task_id_df):
    """
    Print formatted analysis results.
    
    Args:
        model_performance: DataFrame with model effects
        task_type_df: DataFrame with task type effects
        data_source_df: DataFrame with data source effects
        evaluation_df: DataFrame with evaluation type effects
        task_id_df: DataFrame with task ID effects (optional)
    """
    print("\nModel Performance Ranking:")
    print(model_performance)
    
    if task_type_df is not None:
        print("\nTask Type Effects:")
        print(task_type_df)
    
    if data_source_df is not None:
        print("\nData Source Effects:")
        print(data_source_df)
    
    if evaluation_df is not None:
        print("\nEvaluation Type Effects:")
        print(evaluation_df)
    
    if MODELING_TASK_ID and task_id_df is not None:
        print("\nTask ID Effects:")
        print(task_id_df)


if __name__ == "__main__":
    results = main()