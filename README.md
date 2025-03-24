# MedQABench: Evaluating Medical QA Benchmarks Against Clinical Capabilities
This repository contains the code for the paper "Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?" submitted to the BioNLP Workshop at ACL 2025.

## Research Overview
Language models have shown impressive performance on medical question-answering benchmarks, but how well do these results translate to real-world clinical settings? This research presents a systematic analysis of the correlation between benchmark performance and actual clinical capabilities.

![Research Overview](https://github.com/mlshdw235/BioNLP_anonymized_questioning_our_questions/raw/main/figures/research_overview.png)

Our study demonstrates that medical QA benchmarks (like MedQA) show a moderate correlation with clinical performance (Spearman's ρ = 0.59), but this correlation is notably lower than those observed within benchmarks themselves. Among various benchmarks, MedQA is the most predictive of clinical performance, yet it still fails to adequately capture essential competencies such as patient communication, longitudinal care, and clinical information extraction.

![Benchmark-Clinical Correlation](https://github.com/mlshdw235/BioNLP_anonymized_questioning_our_questions/raw/main/figures/correlation_plot_imputated_weighted_two_measures-1.png)

Through Bayesian hierarchical modeling, we've estimated representative clinical performance across different models, identifying GPT-4 and GPT-4o as consistently top-performing models that often match or exceed human physicians' performance.

![Model Performance](https://github.com/mlshdw235/BioNLP_anonymized_questioning_our_questions/raw/main/figures/bayesian_vs_clinical_model_performance_comparison-1.png)

## Key Findings
- Medical QA benchmarks only moderately predict clinical capabilities (ρ = 0.59)
- MedQA benchmark shows the strongest correlation with clinical performance
- Significant gaps exist in benchmark coverage of critical clinical skills
- GPT-4 and GPT-4o demonstrate performance comparable to human physicians

## Repository Structure

- `bayesian_modeling.py`: Implementation of hierarchical Bayesian models for adjusting model performance metrics
- `explore_perf_data_clinical.py`: Visualization and analysis tools for clinical performance data
- `measure_correlations_between_adjusted_model_perfs_and_benchmarks.py`: Correlates Bayesian-adjusted model performances with benchmark scores
- `measure_correlations_between_benchmark_and_clinical_perf.py`: Measures correlations between raw benchmark and clinical performances
- `measure_correlations_from_perf_data_pickle.py`: Analyzes correlations between different benchmark performance data
- `perform_imputation_on_benchmark_perfs.py`: Uses MICE (Multiple Imputation by Chained Equations) to handle missing benchmark performance data
- `plot_bayesian_vs_raw_performances.py`: Creates visualizations comparing Bayesian-adjusted vs raw performance metrics
- `preprocess_performances.py`: Preprocessing module for model performance data across various benchmarks
- `utils_bayesian.py`: Utility functions for Bayesian optimization and analysis
- `utils_imputation.py`: Utility functions for data imputation techniques
- `utils_measure_correlations.py`: Utility functions for correlation analysis

## Requirements

```
Python 3.10+
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.10.0
seaborn>=0.13.0
scipy>=1.15.0
statsmodels>=0.14.0
numpyro>=0.17.0
jax>=0.5.0
plotly>=5.13.0
networkx>=3.0
```

## Data Preparation

Before running the analysis, performance data should be organized in the following directory structure:

```
- perf_data_pickle/
  - clinical_task_id_map.pkl
  - perf_data_clinical.pkl
  - perf_data_clinical_df.pkl
  - perf_data_general_fin_normalized.pkl
  - perf_data_medqa_fin_normalized.pkl
- final_dataset/
  - correlations_fin.json
  - general_df_fin.pkl
  - medqa_df_fin.pkl
```

Note: The repository does not include the raw data files due to their large size and licensing restrictions. Please contact the authors for access to the processed data.

## Usage

### 1. Data Preprocessing

```bash
python preprocess_performances.py
```

This script processes raw benchmark and clinical performance data, normalizes scores, and produces the pickle files needed for subsequent analyses.

### 2. Performance Data Imputation

```bash
python perform_imputation_on_benchmark_perfs.py
```

This script handles missing values in the benchmark performance data using MICE imputation.

### 3. Bayesian Modeling

```bash
python bayesian_modeling.py
```

Implements hierarchical Bayesian models to adjust for model, task type, data source, and evaluation method effects on performance metrics.

### 4. Correlation Analysis

```bash
python measure_correlations_between_benchmark_and_clinical_perf.py
```

Calculates correlations between benchmark performances and clinical task performances.

### 5. Visualization

```bash
python plot_bayesian_vs_raw_performances.py
```

Creates visualizations comparing Bayesian-adjusted performance metrics with raw performance metrics.

## Key Findings

Our research reveals several important insights:

1. **Moderate clinical relevance**: Medical QA benchmarks show a moderate correlation (ρ = 0.59) with clinical performance, suggesting they capture some but not all relevant abilities.

2. **Missing competencies**: Current benchmarks inadequately assess critical clinical skills like patient communication, longitudinal care management, and information extraction.

3. **Model performance**: Through Bayesian modeling, we found that proprietary models (particularly GPT-4 and GPT-4o) consistently outperform open-source alternatives in clinical settings.

4. **Benchmark selection**: Among benchmarks, MedQA demonstrates the strongest alignment with clinical performance, especially for clinical knowledge, treatment, and diagnosis tasks.

## Citation

If you use this code or find our research helpful, please consider citing our paper:

```
@inproceedings{anonymous2025questioning,
  title={Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?},
  author={Anonymous},
  booktitle={Proceedings of the BioNLP Workshop at ACL 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank all the researchers who have developed medical benchmarks and shared their clinical evaluation results, making this meta-analysis possible.
