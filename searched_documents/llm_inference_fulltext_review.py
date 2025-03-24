"""
LLM inference for medical paper full-text review.
Processes PDF papers, extracts text, and analyzes them using LLM API.
"""
import os
import pickle
import time
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List

import PyPDF2
import pandas as pd
import requests
from tqdm import tqdm

from utils_fulltext_review import parse_llm_response, merge_parsed_data

# Configuration constants
# These paths should be modified by the user before running the script
DIR_BASE = "."  # Base directory path (change this to your project directory)
PROMPT_PATH = os.path.join(DIR_BASE, "prompts", "fulltext_review.txt")
PAPERS_PATH = "final_papers_for_fulltext_review.pkl"
PDF_DIR = "full_text_generalmed_medqa_both/pdfs"

# LLM configuration
MODEL_NAME = "phi4:latest"
OLLAMA_API = "http://localhost:11434/api/generate"
MAX_RETRIES = 10  # Maximum number of API call retries

# Processing parameters
MAX_CHUNK_LENGTH = 40000  # Maximum characters per chunk for text processing
INTERMEDIATE_SAVE_INTERVAL = 30  # Save interval for intermediate results
SAMPLE_N = None  # Set to int value for testing with a subset of papers


def load_prompt(prompt_path):
    """Load prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def load_papers(papers_path):
    """Load papers dataframe from pickle file."""
    with open(papers_path, 'rb') as f:
        return pickle.load(f)


def should_retry_inference(response, parsed):
    """
    Check if the response needs to be retried based on performance results format.
    Returns True if response contains problematic template data.
    """
    if not response:
        return False
        
    problematic_start = \
        "[{'value': '70%', 'metric': 'accuracy in examinations', 'subject': 'ChatGPT'"
        
    if ("PERFORMANCE_RESULTS" in parsed and 
        problematic_start in str(parsed['PERFORMANCE_RESULTS'])):
        print("Warning: Example performance result detected in output - will retry inference")
        return True
        
    return False


def extract_text_from_pdf(pdf_path):
    """
    Extract text content from PDF file.
    Returns extracted text and boolean indicating if EOF error occurred.
    """
    text = ""
    eof_error = False
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        if "EOF marker not found" in str(e):
            eof_error = True
        print(f"Error reading PDF {pdf_path}: {e}")
        return "", eof_error
        
    return text, eof_error


def split_text_into_chunks(text, max_length):
    """
    Split text into chunks of maximum length while preserving words.
    Returns list of text chunks.
    """
    chunks = []
    current_chunk = ""
    words = text.split()
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


def call_ollama(prompt, temperature=0.0):
    """
    Make API call to Ollama with provided prompt and temperature.
    Returns LLM response or None if error occurs.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "system": "You are a helpful research assistant analyzing medical papers.",
        "stream": False,
        "options": {
            "num_ctx": 16384  # Context window size
        }
    }

    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def process_paper(title, pdf_path, prompt_template):
    """
    Process a single paper with retry logic and chunk handling.
    Returns parsed data, raw response, and EOF error status.
    """
    text, eof_error = extract_text_from_pdf(pdf_path)
    
    if eof_error:
        return parse_llm_response("", eof_error=True), None, eof_error
        
    if not text:
        return None, None, eof_error

    chunks = split_text_into_chunks(text, MAX_CHUNK_LENGTH)
    parsed_results = []
    responses = []

    if len(chunks) > 1:
        print(f"More than one chunks ({len(chunks)}) for {title}!")
        
    for chunk in chunks:
        full_prompt = prompt_template.replace("{Title}", title)
        full_prompt = full_prompt.replace("{Full_Text}", f"\n{chunk}\n\n")

        # Try with zero temperature first, then increase slightly for retries
        first_attempt = True
        for temp in [0.0] + [0.1] * (MAX_RETRIES - 1):
            response = call_ollama(full_prompt, temperature=temp)
            
            if response:
                if "ANALYSIS_START" in response and "ANALYSIS_END" in response:
                    parsed = parse_llm_response(response)
                    if should_retry_inference(response, parsed):
                        continue
                    if parsed:
                        parsed_results.append(parsed)
                        responses.append(response)
                        break
                    elif first_attempt:
                        print(f"First parsing attempt failed for paper chunk: {title}")
                        
            first_attempt = False
            time.sleep(1)  # Delay between retries

    merged_parsed = merge_parsed_data(parsed_results)
    merged_response = "\n".join(responses) if responses else None

    return merged_parsed, merged_response, eof_error


def create_expanded_dataframe(papers_df, results):
    """
    Create expanded dataframe with parsed data as separate columns.
    Returns dataframe with extracted fields as columns.
    """
    expanded_df = papers_df.copy()
    expanded_df['fulltext_review_data'] = None
    expanded_df.iloc[:len(results), expanded_df.columns.get_loc('fulltext_review_data')] = results

    # Find all unique keys in results
    all_keys = set()
    for result in results:
        if result is not None:
            all_keys.update(result.keys())
            
    # Create columns for each key
    for key in all_keys:
        column_name = f"fulltext_{key.lower()}"
        expanded_df[column_name] = None
        
    # Fill in values
    for idx, paper_data in enumerate(results):
        if paper_data is not None:
            for key, value in paper_data.items():
                column_name = f"fulltext_{key.lower()}"
                # Keep PAPER_TYPE as is, convert others to string
                if key == "PAPER_TYPE":
                    expanded_df.iloc[idx, expanded_df.columns.get_loc(column_name)] = value
                else:
                    expanded_df.iloc[idx, expanded_df.columns.get_loc(column_name)] = str(value)

    return expanded_df


def cleanup_intermediate_files(start_time_str):
    """
    Clean up intermediate files created between start time and now.
    Removes temporary files to avoid disk clutter.
    """
    start_time = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
    current_time = datetime.now()

    intermediate_files = glob.glob('fulltext_review_results/intermediate_*.{csv,pkl}')
    for file_path in intermediate_files:
        try:
            timestamp_str = file_path.split('intermediate_')[1].split('.')[0]
            file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            if start_time <= file_time <= current_time:
                os.remove(file_path)
                print(f"Cleaned up intermediate file: {file_path}")
        except (IndexError, ValueError):
            print(f"Could not parse timestamp from filename: {file_path}")


def save_intermediate_results(papers_df, results, responses, current_time):
    """
    Save intermediate results to CSV and PKL files.
    Creates checkpoint files during long-running processes.
    """
    expanded_df = create_expanded_dataframe(papers_df.iloc[:len(results)], results)
    expanded_df['fulltext_review_response'] = None
    expanded_df.iloc[:len(responses), 
                    expanded_df.columns.get_loc('fulltext_review_response')] = responses

    base_path = f"fulltext_review_results/intermediate_{current_time}"
    expanded_df.to_csv(f"{base_path}.csv", index=False)
    expanded_df.to_pickle(f"{base_path}.pkl")
    print(f"\nSaved intermediate results at: {current_time}")


def main():
    """Main execution function for paper processing pipeline."""
    start_time = time.strftime('%Y%m%d_%H%M%S')
    prompt_template = load_prompt(PROMPT_PATH)
    papers_df = load_papers(PAPERS_PATH)
    start_idx = None  # Set this to an integer to resume from a specific index

    # Use subset for testing if specified
    if SAMPLE_N:
        papers_df = papers_df.head(SAMPLE_N)

    results = []
    responses = []
    eof_error_count = 0
    total_papers = len(papers_df)
    os.makedirs("fulltext_review_results", exist_ok=True)

    if start_idx is None:
        start_idx = 0

    # Process papers with progress tracking
    for idx, (_, row) in enumerate(tqdm(papers_df.iloc[start_idx:].iterrows(),
                                        total=total_papers - start_idx,
                                        desc="Processing papers"),
                                  start=start_idx):
        title = row['title']
        pdf_path = os.path.join(PDF_DIR, row['valid_pdf_fname'])
        parsed_data, response, eof_error = process_paper(title, pdf_path, prompt_template)
        
        if eof_error:
            eof_error_count += 1
            
        results.append(parsed_data)
        responses.append(response)
        
        # Save intermediate results at regular intervals
        if (idx + 1) % INTERMEDIATE_SAVE_INTERVAL == 0:
            current_time = time.strftime('%Y%m%d_%H%M%S')
            save_intermediate_results(papers_df.iloc[:idx+1], results, responses, current_time)
            cleanup_intermediate_files(start_time)

    # Create final results
    expanded_df = create_expanded_dataframe(papers_df, results)
    expanded_df['fulltext_review_response'] = responses

    # Save final output
    output_base = f"fulltext_review_results/fulltext_review_results_{time.strftime('%Y%m%d_%H%M%S')}"
    expanded_df.to_csv(f"{output_base}.csv", index=False)
    expanded_df.to_pickle(f"{output_base}.pkl")

    # Clean up temporary files
    cleanup_intermediate_files(start_time)

    print(f"\nTotal number of PDFs with EOF errors: {eof_error_count}")


if __name__ == "__main__":
    main()