"""
Module for retrieving DOIs and abstracts for academic papers from various sources.
"""
from difflib import SequenceMatcher
import time
import re

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from habanero import Crossref
import backoff

import pandas as pd

# Constants
MIN_ABSTRACT_LENGTH = 30
DEFAULT_MIN_SIMILARITY = 0.95
DEFAULT_SAVE_INTERVAL = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 1
ERROR_STATUS_CODES = [429, 500, 502, 503, 504]
TITLE_PREVIEW_LENGTH = 50

# API URLs
EUROPEPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# LLM-related keywords for filtering
LLM_KEYWORDS = [
    'language model', 'llm',
    'artificial intelligence', 'natural language processing',
    'claude', 'bard', 'bing', 'gemini',
    'chatgpt', 'gpt', 'gpt-4', 'gpt4', 'chatgpt-3.5',
    'perplexity', 'llama', 'mistral', 'med-palm', 'deepseek'
]


def setup_http_session():
    """
    Configure and return HTTP session with retry strategy.
    
    Returns:
        requests.Session: Configured session with retry strategy
    """
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=ERROR_STATUS_CODES
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def calculate_similarity(str1, str2):
    """
    Calculate string similarity using SequenceMatcher.
    
    Args:
        str1: First string to compare
        str2: Second string to compare
        
    Returns:
        float: Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def is_valid_abstract(abstract):
    """
    Check if abstract is valid (non-empty and sufficient length).
    
    Args:
        abstract: Abstract text to validate
        
    Returns:
        bool: True if abstract is valid, False otherwise
    """
    if not abstract:
        return False
    if isinstance(abstract, str) and len(abstract.strip()) >= MIN_ABSTRACT_LENGTH:
        return True
    return False


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.Timeout),
    max_tries=MAX_RETRIES
)
def get_doi_from_crossref(cr, title, min_similarity=DEFAULT_MIN_SIMILARITY):
    """
    Get DOI from Crossref with retry logic.
    
    Args:
        cr: Crossref client
        title: Paper title to search for
        min_similarity: Minimum similarity threshold
        
    Returns:
        tuple: (DOI string or None, source string or None, similarity score)
    """
    try:
        results = cr.works(
            query=title,
            select=['DOI', 'title'],
            cursor='*',
            cursor_max=1,
            limit=1
        )
        if results['message']['items']:
            first_result = results['message']['items'][0]
            if 'title' in first_result and first_result['title']:
                similarity = calculate_similarity(title, first_result['title'][0])
                if similarity >= min_similarity:
                    return first_result.get('DOI'), 'crossref', similarity
        return None, None, 0
    except Exception as e:
        print(f"Crossref error for title '{title[:TITLE_PREVIEW_LENGTH]}...': {str(e)}")
        return None, None, 0


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.Timeout),
    max_tries=MAX_RETRIES
)
def get_doi_from_europepmc(session, title, min_similarity=DEFAULT_MIN_SIMILARITY):
    """
    Get DOI from Europe PMC with retry logic.
    
    Args:
        session: HTTP session
        title: Paper title to search for
        min_similarity: Minimum similarity threshold
        
    Returns:
        tuple: (DOI string or None, source string or None, similarity score)
    """
    try:
        params = {
            'query': f'title:"{title}"',
            'format': 'json'
        }
        response = session.get(EUROPEPMC_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.ok:
            data = response.json()
            if data['resultList']['result']:
                first_result = data['resultList']['result'][0]
                if 'title' in first_result:
                    similarity = calculate_similarity(title, first_result['title'])
                    if similarity >= min_similarity:
                        return first_result.get('doi'), 'europepmc', similarity
        return None, None, 0
    except Exception as e:
        print(f"Europe PMC error for title '{title[:TITLE_PREVIEW_LENGTH]}...': {str(e)}")
        return None, None, 0


def get_abstract_from_source(source, identifier, session=None, cr=None):
    """
    Get abstract using source-specific methods.
    
    Args:
        source: Source name ('crossref' or 'europepmc')
        identifier: DOI or other identifier
        session: HTTP session (for europepmc)
        cr: Crossref client (for crossref)
        
    Returns:
        str or None: Abstract text if found, None otherwise
    """
    try:
        if source == 'crossref' and cr is not None:
            result = cr.works(ids=[identifier])
            if result.get('message', {}).get('abstract'):
                return result['message']['abstract']
        elif source == 'europepmc' and session is not None:
            params = {
                'query': f'DOI:"{identifier}"',
                'format': 'json',
                'resultType': 'core'
            }
            response = session.get(EUROPEPMC_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.ok:
                data = response.json()
                if data['resultList']['result']:
                    return data['resultList']['result'][0].get('abstractText', '')
        return None
    except Exception as e:
        print(f"Error getting abstract from {source} for {identifier}: {str(e)}")
        return None


def get_best_doi(cr, session, title, min_similarity=DEFAULT_MIN_SIMILARITY):
    """
    Get DOI from multiple sources and return the best match.
    
    Args:
        cr: Crossref client
        session: HTTP session
        title: Paper title to search for
        min_similarity: Minimum similarity threshold
        
    Returns:
        tuple: (DOI string or None, source string or None, similarity score)
    """
    doi_crossref, source_crossref, sim_crossref = get_doi_from_crossref(cr, title, min_similarity)
    time.sleep(0.1)  # Rate limiting
    doi_europepmc, source_europepmc, sim_europepmc = \
        get_doi_from_europepmc(session, title, min_similarity)

    if sim_crossref > sim_europepmc:
        return doi_crossref, source_crossref, sim_crossref
    elif sim_europepmc > 0:
        return doi_europepmc, source_europepmc, sim_europepmc
    else:
        return None, None, 0


def clean_jats_abstract(text):
    """
    Remove JATS XML tags from abstract text while preserving content structure.
    
    Args:
        text: Abstract text containing JATS XML tags
        
    Returns:
        str: Cleaned abstract text with tags removed
    """
    if not text:
        return text
    
    # Replace specific tags with appropriate formatting
    text = re.sub(r'<jats:title>Abstract</jats:title>', '', text)
    text = re.sub(r'<jats:title>(.*?)</jats:title>', r'\n\1: ', text)
    text = re.sub(r'<jats:italic>(.*?)</jats:italic>', r'\1', text)
    text = re.sub(r'<jats:sec>|</jats:sec>', '', text)
    text = re.sub(r'<jats:p>(.*?)</jats:p>', r'\1', text)
    
    # Remove any remaining XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s+', '\n', text)  # Remove extra spaces after newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    text = text.strip()
    
    # Fix any unicode spaces
    text = re.sub(r'\u2009=\u2009', '=', text)

    return text


def filter_by_llm_keywords(df):
    """
    Filter DataFrame rows based on the presence of LLM-related keywords in titles.
    
    Args:
        df: DataFrame containing paper information
    
    Returns:
        DataFrame: Filtered DataFrame with LLM-related papers
    """
    def contains_llm_keyword(text):
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in LLM_KEYWORDS)
    
    df = df.dropna(subset=['Year'])
    initial_count = len(df)
    df_filtered = df[df['Title'].apply(contains_llm_keyword)]
    filtered_count = len(df_filtered)

    print(f"Initial number of papers: {initial_count}")
    print(f"Number of papers after filtering by LLM keywords: {filtered_count}")
    print(f"Number of papers excluded: {initial_count - filtered_count}")

    return df_filtered


def print_summary(df, output_filename):
    """
    Print summary statistics of the processing results.
    
    Args:
        df: Processed DataFrame
        output_filename: Path to the output file
    """
    print("\nSummary:")
    print(f"Total papers processed: {len(df)}")
    print(f"DOIs found: {df['doi'].notna().sum()}")
    print(f"Original valid abstracts: {df['Abstract'].apply(is_valid_abstract).sum()}")
    print(f"API-retrieved valid abstracts: {df['API_Abstract'].apply(is_valid_abstract).sum()}")
    print(f"Average similarity score: {df['title_similarity'].mean():.3f}")
    print("DOI sources distribution:")
    print(df['doi_source'].value_counts())
    print(f"\nResults saved to: {output_filename}")


def process_papers_for_doi_and_abstract(input_filename, 
                                        min_similarity=DEFAULT_MIN_SIMILARITY,
                                        save_interval=DEFAULT_SAVE_INTERVAL):
    """
    Main function to process papers and find DOIs and abstracts.
    
    Args:
        input_filename: Path to the input CSV file
        min_similarity: Minimum title similarity threshold for matching
        save_interval: Number of rows to process before saving intermediate results
        
    Returns:
        DataFrame: Processed DataFrame with DOIs and abstracts
    """
    # Initialize clients
    cr = Crossref(timeout=REQUEST_TIMEOUT)
    session = setup_http_session()

    # Read data
    df_papers = pd.read_csv(input_filename)
    df_papers = filter_by_llm_keywords(df_papers)

    # Initialize new columns if they don't exist
    if 'doi' not in df_papers.columns:
        df_papers['doi'] = None
    if 'doi_source' not in df_papers.columns:
        df_papers['doi_source'] = None
    if 'title_similarity' not in df_papers.columns:
        df_papers['title_similarity'] = 0.0
    if 'API_Abstract' not in df_papers.columns:
        df_papers['API_Abstract'] = None

    # Process papers
    for idx, row in tqdm(df_papers.iterrows(), total=len(df_papers)):
        try:
            # Always try to get DOI
            doi, source, similarity = get_best_doi(cr, session, row['Title'], min_similarity)
            
            # Update DOI related information
            df_papers.at[idx, 'doi'] = doi
            df_papers.at[idx, 'doi_source'] = source
            df_papers.at[idx, 'title_similarity'] = similarity

            # Only try to get abstract if we don't have a valid one already
            if not is_valid_abstract(row.get('Abstract')) and doi and source:
                abstract = get_abstract_from_source(source, doi, session, cr)
                abstract = clean_jats_abstract(abstract)
                if is_valid_abstract(abstract):
                    df_papers.at[idx, 'API_Abstract'] = abstract

            # Periodic save
            if idx % save_interval == 0 and idx > 0:
                intermediate_filename = input_filename.replace('.csv', '_intermediate.csv')
                df_papers.to_csv(intermediate_filename, index=False)
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    # Save final results
    output_filename = input_filename.replace('.csv', '_with_doi_abstract.csv')
    df_papers.to_csv(output_filename, index=False)

    # Print summary
    print_summary(df_papers, output_filename)

    return df_papers


if __name__ == "__main__":
    # Default input file
    DEFAULT_INPUT_FILE = 'medical_llm_papers_20250110_162549_medqa.csv'
    
    # Process papers
    df_processed = process_papers_for_doi_and_abstract(DEFAULT_INPUT_FILE)