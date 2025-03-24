"""
Module for analyzing research papers that evaluate Large Language Models.
This module provides functions to process model names, analyze paper characteristics,
and generate statistics about model usage across papers.
"""
import pickle
from collections import Counter
from itertools import combinations
from datetime import datetime


# Dictionary to normalize model names
MODEL_NORMALIZE = {
    # GPT-4 and variants
    'GPT-4': ['GPT-4', 'GPT4', 'GPT 4', 'GPT-4.0', 'GPT 4.0', 'GPT4.0',
              'Generative Pre-trained Transformer 4', 
              'Generative Pretrained Transformer-4',
              'GPT-4 (OpenAI)', 'GPT-4 AI', 'GPT-4 based ChatGPT',
              'Generative Pre-trained Transformer 4 (GPT-4)',
              'GPT-4 turbo', 'GPT-4 Turbo', 'GPT-4 HPO',
              'GPT-4 OpenAI API', 'gpt-4-1106-preview', 'GPT4V', 
              'GPT-4(V)', 'GPT-4V', 'GPT-4 Vision',
              'Generative Pretrained Transformer 4 with Vision capabilities (GPT-4V)',
              'GPT-4 Vision (GPT-4V)',
              'GPT-4V(ision)', 'GPT-4 1106 Vision Preview',
              'GPT-4 Turbo with Vision', 'text-only GPT-4 Turbo',
              'GPT-4 omni', 'GPT-4 vision', 'GPT-4 + GPT-4V', 'GPT-4V-based ChatGPT',
              'ChatGPT-4', 'ChatGPT4', 'ChatGPT 4', 'Chat-GPT4', 'ChatGPT-4.0', 'ChatGPT 4.0',
              'ChatGPT-4.o', 'ChatGPT-4o', 'ChatGPT-4.0o',
              'ChatGPT plus 4.0', 'ChatGPT Plus v4.0',
              'ChatGPT (GPT-4)', 'ChatGPT-4 (OpenAI)', 'ChatGPT with GPT-4', 'ChatGPT-4V',
              'ChatGPT-4 With Vision', 'ChatGPT-4 Vision'],
    # ChatGPT and variants
    'ChatGPT': ['ChatGPT', 'Chat GPT', 'Chat-GPT', 'chatGPT',
                'ChatGPT©', 'OpenAI ChatGPT', "OpenAI's ChatGPT",
                'Chat Generative Pre-trained Transformer',
                'Chat Generative Pretrained Transformer',
                'Chat-Generative Pre-Trained Transformer',
                'Conversational Generative Pre-trained Transformer',
                'ChatGPT (OpenAI)', 'chatGPT™', 'GPT-3.5', 'GPT3.5', 'GPT-3.5-turbo',
                'GPT-3.5 Turbo', 'GPT-3.5 turbo', 'gpt-3.5-turbo-1106',
                'OpenAI GPT3.5', 'GPT-3.5-Turbo',
                'Generative Pre-trained Transformer model (gpt-3.5-turbo)',
                'OpenAI gpt-3.5-turbo', 'GTP-3.5', 'GPT-3·5',
                'generative pre-trained transformer-3.5',
                'Generative Pretraining Transformer 3.5', 'ChatGPT 3.5 Turbo',
                'ChatGPT-3.5', 'ChatGPT3.5', 'ChatGPT 3.5', 'ChatGPT-3',
                'ChatGPT3', 'ChatGPT v3.5',
                'ChatGPT-3.5 (OpenAI)', 'ChatGPT-3.5-turbo', 'ChatGPT (GPT-3.5)',
                'Chat-GPT 3.5'],
    # Claude and variants
    'Claude': ['Claude', 'Claude.AI', 'Claude AI', 'CLAUDE', 'Claude-2',
               'Claude2', 'Claude 2', 'Claude-2.1', 'Claude 2.1', 'Claude-3',
               'Claude 3', 'Claude-3.5-sonnet', 'Claude 3.5 Sonnet',
               'Claude-3.5 Sonnet', 'Claude Sonnet 3.5', 'Claude 3 Opus','Claude 3 Sonnet',
               'Claude 3 Haiku', 'Claude-3-haiku', 'Claude Pro', 'Claude+', 'Claude Instant'],
    # Bard/Gemini and variants
    'Gemini': ['Bard', 'BARD', 'Google Bard', 'Google BARD',
               "Google's BARD", 'Bard AI', 'Bard (Google AI)',
               'Bard (Google LLC)', 'Bard (Versions 1 and 2)',
               'Bard Gemini Pro', 'Gemini', 'Google Gemini',
               'GEMINI', 'Google GEMINI', 'Gemini Pro', 'Gemini Pro 1.0', 'Gemini Advanced',
               'Google Gemini Advanced', 'Gemini Ultra', 'Gemini 1.0',
               'Gemini 1.5', 'Gemini 1.5 Pro',
               'Gemini 1.5 Flash', 'Gemini (Pro)', 'GeminiPro', 'Gemini Adv',
               'Google Bard (currently Google Gemini)', 'Gemini (2024.2)'],
    # Llama and variants
    'Llama': ['Llama', 'LLaMA', 'LLAMA', 'Llama2', 'LLAMA2', 'LLaMA2',
              'Llama-2', 'LLaMA-2', 'LLaMA 1', 'LLaMA 2', 'Llama 2', 'Llama-2-70B',
              'Llama-2-70B-chat', 'Llama 2-70b', 'Llama 2 70B',
              'Llama-2 70B', 'LLaMA-2-7b', 'Llama-2 7B', 'Llama2-7b', 'Llama2-13b', 'LLaMA-3B',
              'Llama3', 'LLaMA3', 'Llama 3', 'Llama-3 8B', 'Llama3-8B', 'Llama 3 8B',
              'Llama-3-70B', 'Llama3-70b', 'Llama 3 70B', 'Llama-3.1-70B',
              'LLaMA 3.1', 'Meta Llama 3 70b'],
    # BERT and variants
    'BERT': ['BERT', 'Bidirectional Encoder Representations from Transformers',
             'BERT-base', 'BERT-medium',
             'BERT-small', 'BERT-mini', 'BERT-tiny', 'BioBERT', 'ClinicalBERT', 'Clinical BERT',
             'Bio-Clinical BERT', 'BioClinical-BERT', 'BioClinicalBERT',
             'PubMedBERT', 'BlueBERT', 'Clinical Longformer', 'ClinicalBLIP', 'DistilBERT',
             'distilBERT', 'DistilBert', 'distilbert-base-uncased', 'RoBERTa',
             'DeBERTa', 'ELECTRA', 'XLNet'],
    # Microsoft products
    'Microsoft': ['Bing', 'Bing Chat', 'Bing AI', 'BingAI', 'Bing AI Chat', 'Microsoft Bing',
                  'Microsoft Bing AI', 'New Bing', 'New Bing Chat', 'Copilot', 'CoPilot',
                  'Co-Pilot', 'Microsoft Copilot', 'Copilot Pro', 'COPILOT'],

    # Mistral and variants
    'Mistral': ['Mistral', 'Mistral-7b', 'Mistral-7B', 'Mistral-7B-Instruct-v0.2',
                'Mistral-Large', 'Mistral-Medium', 'Mixtral', 'Mixtral-8x7B',
                'Mixtral 8x7b', 'Mixtral 8x22b'],

    # Generic/Other LLMs
    'Other LLMs': ['large language model', 'Large Language Model',
                   'LLM', 'Large Language Models', 'large language models',
                   'Large Language Models (LLMs)', 'LLMs', 'Language models (LM)',
                   'Neural language models', 'AI LLM', 'AI-LLM',
                   'Large Language Model (LLM)', 'large language model (LLM)',
                   'large language model (LLM) based AI']
}

# Global constants
OUTPUT_DIR = '.'  # Directory for saving output files
CLINICAL_LLM_ORIGINAL = 'Clinical LLM Performance Evaluation - Original'
PAPER_TYPES_TO_DISPLAY = [
    CLINICAL_LLM_ORIGINAL,
    "Clinical LLM Performance Review",
    "Non-Clinical LLM Evaluation",
    "None",
    None
]
MIN_MODELS_FOR_PAIR = 2  # Minimum number of models required to form a pair
TOP_MODEL_PAIRS_TO_SHOW = 10  # Number of top model pairs to display in results


def normalize_model_name(model_name, MODEL_NORMALIZE):
    """
    Normalize a model name to its standardized form using the MODEL_NORMALIZE dictionary.
    
    Returns the standardized name or 'Other' if not found in the dictionary.
    """
    if model_name is None:
        return None
        
    for standard_name, variants in MODEL_NORMALIZE.items():
        if model_name in variants or model_name.strip() in variants:
            return standard_name

    return 'Other'


def normalize_model_list(model_list, MODEL_NORMALIZE):
    """
    Normalize a list of model names to their standardized forms.
    
    Returns None if the input list is None, otherwise returns a list of normalized names.
    """
    if model_list is None:
        return None
        
    return [normalize_model_name(model, MODEL_NORMALIZE) for model in model_list]


def get_model_pairs(model_list, MODEL_NORMALIZE):
    """
    Get all possible pairs of normalized model names from a list.
    
    Returns pairs in alphabetical order to ensure consistent counting.
    Returns an empty list if input list has fewer than 2 models.
    """
    if model_list is None or len(model_list) < MIN_MODELS_FOR_PAIR:
        return []
        
    normalized_models = normalize_model_list(model_list, MODEL_NORMALIZE)
    pairs = []
    
    for pair in combinations(normalized_models, 2):
        # Sort pair to ensure consistent ordering
        pairs.append(tuple(sorted(pair)))
        
    return pairs


def get_analysis_stats(df):
    """
    Calculate various statistics from the analyzed papers DataFrame.
    
    Returns a dictionary containing counts of papers, model usage, and paper types.
    """
    # Convert string boolean values to actual boolean values
    df['analyzed_multiple_models_usage'] = df['analyzed_multiple_models_usage'].map({
        'true': True,
        'false': False,
        'NA': None
    })

    # Normalize model names
    df['normalized_models'] = df['analyzed_models'].apply(
        lambda x: normalize_model_list(x, MODEL_NORMALIZE)
    )

    # Count papers with multiple models
    multiple_models_count = sum(
        1 for models in df['normalized_models']
        if models is not None and len(models) >= MIN_MODELS_FOR_PAIR
    )

    # Create normalized model counts
    all_normalized_models = [
        model 
        for models in df['normalized_models']
        if models is not None
        for model in models
    ]
    normalized_model_counts = Counter(all_normalized_models)

    # Count model pairs
    all_model_pairs = [
        pair 
        for models in df['analyzed_models']
        if models is not None and len(models) >= MIN_MODELS_FOR_PAIR
        for pair in get_model_pairs(models, MODEL_NORMALIZE)
    ]
    model_pair_counts = Counter(all_model_pairs)

    return {
        'total_papers': len(df),
        'multiple_models_count': multiple_models_count,
        'paper_type_counts': Counter(df['analyzed_paper_type']),
        'normalized_model_counts': normalized_model_counts,
        'model_pair_counts': model_pair_counts
    }


def print_analysis_results(stats):
    """
    Print the analysis results in a formatted way.
    """
    print(f"Total papers: {stats['total_papers']}")
    print(f"Number of papers using multiple models: {stats['multiple_models_count']}")

    print("\nPaper Type Distribution:")
    print("-" * 50)
    for paper_type, count in stats['paper_type_counts'].most_common():
        display_type = 'None' if paper_type is None else paper_type
        if paper_type in PAPER_TYPES_TO_DISPLAY:
            print(f"{display_type:<40} : {count:>5}")

    print("\nNormalized Model Distribution:")
    print("-" * 50)
    for model, count in stats['normalized_model_counts'].most_common():
        model_name = 'None' if model is None else model
        print(f"{model_name:<40} : {count:>5}")

    print("\nMost Common Model Pairs:")
    print("-" * 50)
    for pair, count in stats['model_pair_counts'].most_common(TOP_MODEL_PAIRS_TO_SHOW):
        print(f"{pair[0]} + {pair[1]:<35} : {count:>5}")


def get_papers_for_review(df):
    """
    Get papers that meet review criteria.
    
    Criteria include:
    - Original clinical LLM evaluations
    - Papers using multiple models (explicitly or implicitly)
    - Papers with undefined paper type
    
    Returns a filtered DataFrame.
    """
    # Papers with original clinical LLM evaluation
    paper_original_clinical_llm = df['analyzed_paper_type'].apply(
        lambda x: x is not None and x.strip() == CLINICAL_LLM_ORIGINAL
    )
    
    # Papers with multiple models (explicit)
    multiple_models_mask_explicitly = df['normalized_models'].apply(
        lambda x: x is not None and len(x) >= MIN_MODELS_FOR_PAIR
    )
    
    # Papers with multiple models (implicit)
    multiple_models_mask_implicitly = df['analyzed_multiple_models_usage'].apply(
        lambda x: x is True
    )
    
    # Papers with undefined paper type
    none_paper_type_mask = df['analyzed_paper_type'].isna()
    
    # Combined mask for all criteria
    combined_mask = (
        paper_original_clinical_llm | 
        multiple_models_mask_explicitly | 
        multiple_models_mask_implicitly | 
        none_paper_type_mask
    )
    
    # Print counts for each criterion
    print(f"Original clinical LLM papers: {sum(paper_original_clinical_llm)}")
    print(f"Papers with multiple models (explicit): {sum(multiple_models_mask_explicitly)}")
    print(f"Papers with multiple models (implicit): {sum(multiple_models_mask_implicitly)}")
    print(f"Papers with undefined type: {sum(none_paper_type_mask)}")
    print(f"Total papers meeting criteria: {sum(combined_mask)}")

    return df[combined_mask]


def save_papers_for_review(df_for_review):
    """
    Save the filtered papers to pickle and CSV files with timestamp.
    
    Returns tuple of (pickle_filename, csv_filename).
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'{OUTPUT_DIR}/final_papers_for_fulltext_review_{timestamp}'
    
    # Save as pickle
    pickle_filename = f'{base_filename}.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(df_for_review, f)
    
    # Save as CSV
    csv_filename = f'{base_filename}.csv'
    df_for_review.to_csv(csv_filename, encoding='utf-8-sig', index=False)

    return pickle_filename, csv_filename


def main():
    """
    Main function to run the analysis pipeline.
    """
    input_file = f'{OUTPUT_DIR}/analyzed_papers_results_full_20250116_165806.pkl'
    
    # Load the pickle file
    with open(input_file, 'rb') as f:
        df_analyzed_papers = pickle.load(f)
    
    # Get analysis statistics
    stats = get_analysis_stats(df_analyzed_papers)
    
    # Print results
    print_analysis_results(stats)
    
    # Get and save papers for review
    df_for_review = get_papers_for_review(df_analyzed_papers)
    pickle_file, csv_file = save_papers_for_review(df_for_review)

    print(f"\nNumber of papers for full-text review: {len(df_for_review)}")
    print(f"Results saved to:\n{pickle_file}\n{csv_file}")


if __name__ == "__main__":
    main()