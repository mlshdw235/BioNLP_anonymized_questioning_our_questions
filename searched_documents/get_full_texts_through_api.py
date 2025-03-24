"""
Module for processing academic paper data, filtering by LLM-related keywords,
and downloading PDFs based on DOIs.

Before using, update the configuration constants with your own API keys and credentials.
"""
from typing import Optional
from dataclasses import dataclass
from pprint import pprint
import pandas as pd

from pdf_donwloader import EnhancedPaperDownloader

# Configuration constants
OUTPUT_DIR = "full_text_generalmed_medqa_both"
UNPAYWALL_EMAIL = "your.email@example.com"
ELSEVIER_API_KEY = "your_elsevier_api_key"
WILEY_API_KEY = "your_wiley_api_key"
SPRINGER_API_KEY = "your_springer_api_key"
SCIENCEDIRECT_API_KEY = "your_sciencedirect_api_key"
RESEARCHGATE_CREDENTIALS = {
    "username": "your_username",
    "password": "your_password"
}

# Input/output file paths
INPUT_CSV_PATH = 'data/papers.csv'
OUTPUT_CSV_PATH = 'output/filtered_papers.csv'

# Filtering constants
MIN_ABSTRACT_LENGTH = 10  # Minimum abstract length to be considered meaningful


@dataclass
class DownloadResult:
    """Represents the result of a paper download attempt"""
    success: bool
    source: str
    pdf_content: Optional[bytes] = None
    error_message: str = ""


def process_csv_file(input_path: str) -> pd.DataFrame:
    """
    Process paper data from a CSV file.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        Processed DataFrame
    """
    # Load CSV file
    df = pd.read_csv(input_path)
    
    # Clean the data
    df = df.dropna(subset=['doi'])
    df = df.drop_duplicates(subset=['doi'], keep='first')

    # Process abstracts
    df = _process_abstracts(df)
    
    # Standardize column names
    df = _standardize_columns(df)

    return df


def _process_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process abstracts in the dataframe, using API_Abstract when needed.
    
    Args:
        df: DataFrame with Abstract and API_Abstract columns
        
    Returns:
        DataFrame with processed abstracts
    """
    def has_meaningful_info(text):
        return isinstance(text, str) and len(text) >= MIN_ABSTRACT_LENGTH

    # Create a mask for rows to keep
    rows_to_keep = []
    for index, row in df.iterrows():
        if has_meaningful_info(row['Abstract']):
            rows_to_keep.append(True)
        elif has_meaningful_info(row['API_Abstract']):
            df.at[index, 'Abstract'] = row['API_Abstract']
            rows_to_keep.append(True)
        else:
            rows_to_keep.append(False)
            
    # Filter the DataFrame using the boolean mask
    return df[rows_to_keep]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and select required columns.
    
    Args:
        df: DataFrame with original column names
        
    Returns:
        DataFrame with standardized column names
    """
    columns_mapping = {
        'Title': 'title',
        'Authors': 'authors',
        'Year': 'year',
        'Venue': 'venue',
        'Abstract': 'abstract',
        'Fields_of_Study': 'fields_of_study',
        'doi': 'doi',
        'source': 'source'
    }
    df = df.rename(columns=columns_mapping)
    return df[columns_mapping.values()]


def filter_by_llm_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame rows based on LLM-related keywords in the abstract.
    
    Args:
        df: DataFrame containing paper information
    
    Returns:
        Filtered DataFrame with LLM-related papers
    """
    llm_keywords = [
        'language model', 'llm',
        'artificial intelligence', 'natural language processing',
        'claude', 'bard', 'bing', 'gemini',
        'chatgpt', 'gpt', 'gpt-4', 'gpt4', 'chatgpt-3.5',
        'perplexity', 'llama', 'mistral', 'med-palm', 'deepseek', 
        # 'phi' excluded as too short, might cause inaccurate filtering
    ]

    df = df.dropna(subset=['year'])
    initial_count = len(df)
    
    df_filtered = df[df['abstract'].apply(
        lambda abstract: any(keyword in abstract.lower() for keyword in llm_keywords)
    )]
    
    filtered_count = len(df_filtered)

    print(f"Initial number of papers: {initial_count}")
    print(f"Number of papers after filtering by LLM keywords: {filtered_count}")
    print(f"Number of papers excluded: {initial_count - filtered_count}")

    return df_filtered


def initialize_downloader():
    """Initialize paper downloader with API credentials"""
    return EnhancedPaperDownloader(
        output_dir=OUTPUT_DIR,
        unpaywall_email=UNPAYWALL_EMAIL,
        elsevier_api_key=ELSEVIER_API_KEY,
        wiley_api_key=WILEY_API_KEY,
        tandf_api_key=None,
        springer_api_key=SPRINGER_API_KEY,
        sciencedirect_api_key=SCIENCEDIRECT_API_KEY,
        oxford_academic_api_key=None,
        researchgate_username=RESEARCHGATE_CREDENTIALS["username"],
        researchgate_password=RESEARCHGATE_CREDENTIALS["password"]
    )


def main():
    """Main function to process academic papers and download PDFs"""
    # Initialize downloader
    downloader = initialize_downloader()

    # Process input CSV file
    df = process_csv_file(input_path=INPUT_CSV_PATH)
    
    # Filter by LLM keywords
    df_filtered = filter_by_llm_keywords(df)
    
    # Save filtered results
    df_filtered.to_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
    
    # Download PDFs for all DOIs
    dois = list(df_filtered['doi'])
    results = downloader.download_papers(dois)
    
    # Print download results
    success_count = sum(1 for status in results.values() if "Successfully downloaded" in status)
    already_count = sum(1 for status in results.values() if "Already downloaded" in status)
    print(f"Results: {success_count} new downloads, {already_count} already present")
    pprint(f"Raw results: \n{results}")


if __name__ == "__main__":
    main()