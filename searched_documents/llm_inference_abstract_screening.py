"""
Module for analyzing academic paper abstracts using LLM inference.
This module provides functionality for abstract screening and categorization
using local LLM inference via Ollama.
"""
import pickle
import json
import time
from typing import Dict, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm
import pandas as pd

# Global configurations
VERBOSE_LOGGING = 1  # Level of logging detail (0=off, 1=on)
REPROCESS_UNPARSED = True  # Flag for reprocessing None paper_types
INPUT_PICKLE_FILE = 'analyzed_papers_results_full.pkl'
PROMPT_DIRECTORY = 'prompts'
PROMPT_FILE = 'abstract_screening.txt'
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi4:latest"
MAX_RETRY_ATTEMPTS = 10
CHECKPOINT_SIZE = 10  # Number of papers processed before saving checkpoint
TEMPERATURE = 0.1  # LLM generation temperature (lower = more deterministic)
TOP_P = 0.9  # Top-p sampling parameter

class PaperAnalyzer:
    """PaperAnalyzer for abstract screening using LLM inference."""
    
    def __init__(self, model_name=DEFAULT_MODEL, max_retries=MAX_RETRY_ATTEMPTS):
        """
        Initialize the paper analyzer.
        
        Args:
            model_name: Name of the Ollama model to use
            max_retries: Maximum number of retry attempts for API calls
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_url = OLLAMA_BASE_URL

    def read_prompt_template(self, file_path):
        """
        Read prompt template from file.
        
        Args:
            file_path: Path to the prompt template file
            
        Returns:
            String containing the prompt template
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def generate_prompt(self, title, abstract, template):
        """
        Generate a prompt by filling template with paper title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            template: Prompt template string
            
        Returns:
            Completed prompt string
        """
        return template.replace("{Title}", title).replace("{Abstract}", abstract)

    def call_ollama(self, prompt):
        """
        Call Ollama API to generate LLM response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If API call fails
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": TEMPERATURE,
            "top_p": TOP_P
        }
        response = requests.post(self.base_url, json=payload)
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"API call failed with status code {response.status_code}")

    def parse_analysis_output(self, output):
        """
        Parse the LLM analysis output to extract structured data.
        
        Args:
            output: Raw LLM response string
            
        Returns:
            Dictionary containing parsed analysis fields or None if parsing fails
        """
        try:
            # Extract content between markers
            start_idx = output.find("ANALYSIS_START")
            end_idx = output.find("ANALYSIS_END")
            if start_idx == -1 or end_idx == -1:
                if VERBOSE_LOGGING:
                    print("Failed to find ANALYSIS_START or ANALYSIS_END markers")
                    print("Raw output:", output)
                return None

            content = output[start_idx + len("ANALYSIS_START"):end_idx].strip()
            
            # Initialize result dictionary with raw response
            result = {"raw_response": output}
            
            # Parse XML content
            try:
                root = ET.fromstring(f"<root>{content}</root>")
            except ET.ParseError as e:
                if VERBOSE_LOGGING:
                    print("\nXML parsing failed!")
                    print(f"XML Parse Error: {str(e)}")
                return None
            
            # Define fields to parse with their parser functions and default values
            fields_to_parse = {
                "paper_type": (lambda x: x.strip(), None),
                "models": (json.loads, "[]"),
                "multiple_models_usage": (lambda x: x.strip(), None),
                "human_groups": (json.loads, "[]"),
                "evaluation_tasks": (json.loads, "[]"),
                "performance_results": (json.loads, "[]")
            }
            
            # Parse each field
            for field, (parser, default) in fields_to_parse.items():
                try:
                    element = root.find(field.upper())
                    if element is None:
                        if VERBOSE_LOGGING:
                            print(f"\nField '{field}' not found in XML")
                        result[field] = default
                        continue
                        
                    text_content = element.text.strip() if element.text else ""
                    if not text_content and VERBOSE_LOGGING:
                        print(f"\nEmpty content for field '{field}'")
                        result[field] = default
                        continue
                        
                    result[field] = parser(text_content)
                except (json.JSONDecodeError, AttributeError, ValueError) as e:
                    if VERBOSE_LOGGING:
                        print(f"\nParsing failed for field '{field}':")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {str(e)}")
                        print(f"Problematic content: {text_content if 'text_content' in locals() else 'No content'}")
                    result[field] = default
            
            return result

        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"\nUnexpected error during parsing:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Full content:", content if 'content' in locals() else "Content extraction failed")
            return None

    def analyze_paper(self, title, abstract, prompt_template):
        """
        Analyze a paper using LLM with retry mechanism.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            prompt_template: Template for generation prompt
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                prompt = self.generate_prompt(title, abstract, prompt_template)
                response = self.call_ollama(prompt)
                result = self.parse_analysis_output(response)
                if result is not None:
                    return result
                if VERBOSE_LOGGING:
                    print(f"Attempt {attempt + 1} failed to parse output, retrying...")
                time.sleep(1)
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                time.sleep(1)

        raise Exception(f"Failed to analyze paper after {self.max_retries} attempts")


def save_results(df, retry=False):
    """
    Save results to CSV and pickle files with appropriate filenames.
    
    Args:
        df: DataFrame containing results to save
        retry: Whether this is a retry save (affects filename)
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = 'analyzed_papers_results_retry' if retry else 'analyzed_papers_results'
    suffix = '_full'
    
    output_csv_file = f'{prefix}{suffix}_{now}.csv'
    output_pkl_file = f'{prefix}{suffix}_{now}.pkl'
    
    df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    df.to_pickle(output_pkl_file)
    
    if VERBOSE_LOGGING:
        print(f"Results saved to {output_csv_file} and {output_pkl_file}")


def main():
    """Main execution function for paper analysis."""
    analyzer = PaperAnalyzer(model_name=DEFAULT_MODEL, max_retries=MAX_RETRY_ATTEMPTS)

    # Read prompt template
    prompt_path = f'{PROMPT_DIRECTORY}/{PROMPT_FILE}'
    prompt_template = analyzer.read_prompt_template(prompt_path)

    # Load papers for analysis
    if REPROCESS_UNPARSED:
        # Load the previously analyzed papers
        with open(INPUT_PICKLE_FILE, 'rb') as f:
            df_papers = pickle.load(f)
        
        # Filter papers with None paper_type
        mask_none_type = df_papers['analyzed_paper_type'].isna()
        df_papers_to_analyze = df_papers[mask_none_type].copy()
        
        if VERBOSE_LOGGING:
            print(f"Found {len(df_papers_to_analyze)} papers with None paper_type to reanalyze")
    else:
        # Original flow for initial analysis
        df_papers = pd.read_pickle('filtered_papers_with_valid_doi_abstract_pdfs_250115.pkl')
        df_papers = df_papers[['title', 'authors', 'year', 'venue', 'abstract',
                              'fields_of_study', 'doi', 'source', 'valid_pdf_fname']]
        df_papers_to_analyze = df_papers

    # Define result columns
    results_columns = {
        'paper_type': [],
        'models': [],
        'multiple_models_usage': [],
        'human_groups': [],
        'evaluation_tasks': [],
        'performance_results': [],
        'raw_response': []
    }

    total_papers = len(df_papers_to_analyze)
    analyzed_columns = [f'analyzed_{key}' for key in results_columns]
    
    # Pre-initialize DataFrame for storing results
    df_temp = pd.DataFrame(
        index=range(total_papers),
        columns=analyzed_columns,
        dtype=object
    )

    # Process each paper
    for idx, row in tqdm(df_papers_to_analyze.iterrows(), total=total_papers):
        try:
            result = analyzer.analyze_paper(row['title'], row['abstract'], prompt_template)
            for key in results_columns:
                if key not in result:
                    if VERBOSE_LOGGING:
                        print(f"Warning: Key '{key}' missing from analysis result for paper {idx}")
                    result[key] = None
                df_temp.at[idx, f'analyzed_{key}'] = result[key]
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error processing paper {idx}: {str(e)}")
            for key in results_columns:
                df_temp.at[idx, f'analyzed_{key}'] = None

    # Save final results
    if REPROCESS_UNPARSED:
        # Update only the reanalyzed papers in the original DataFrame
        for col in analyzed_columns:
            df_papers.loc[mask_none_type, col] = df_temp[col]
        save_results(df_papers, retry=True)
    else:
        # Save results for initial analysis
        df_papers_final = df_papers_to_analyze.copy()
        for col in analyzed_columns:
            df_papers_final[col] = df_temp[col]
        save_results(df_papers_final, retry=False)

    print("Analysis complete.")


if __name__ == "__main__":
    main()