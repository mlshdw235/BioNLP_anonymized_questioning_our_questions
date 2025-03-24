import os
import json
import pandas as pd

# Global configuration variables
CSV_FILENAMES = [
    'combined_medical_papers_with_doi.csv',
    'medical_llm_papers_medqa_with_doi_abstract.csv',
    'medical_llm_papers_generalmed_with_doi_abstract.csv',
    'reference_information/medical_llm_papers_generalmed_with_doi_abstract.csv',
    'reference_information/medical_llm_papers_medqa_with_doi_abstract.csv'
]
FULL_TEXT_DIR = 'full_text_generalmed_medqa_both'
PDF_SUBDIR = 'pdfs'
MIN_PDF_SIZE_KB = 180  # Minimum PDF file size in kilobytes
OUTPUT_CSV = 'filtered_papers_with_valid_doi_abstract_pdfs.csv'
OUTPUT_PICKLE = 'filtered_papers_with_valid_doi_abstract_pdfs.pkl'


def load_and_combine_csv_files(filenames):
    """
    Load multiple CSV files and combine them into a single DataFrame.
    
    Args:
        filenames: List of CSV filenames to process
        
    Returns:
        Combined pandas DataFrame
    """
    df_list = [pd.read_csv(fname) for fname in filenames]
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"Number of rows after combining CSV files: {df_combined.shape[0]}")
    return df_combined


def remove_duplicates_and_nulls(dataframe):
    """
    Remove duplicate entries based on DOI and filter out rows with null DOI or abstract.
    
    Args:
        dataframe: DataFrame to process
        
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicate DOIs
    df_clean = dataframe.drop_duplicates(subset='doi', inplace=False)
    print(f"Number of rows after dropping duplicates: {df_clean.shape[0]}")
    
    # Remove rows with null DOI or abstract
    df_clean = df_clean.dropna(subset=['doi', 'abstract'])
    print(f"Number of rows after filtering rows with non-null 'doi' and 'abstract': {df_clean.shape[0]}")
    
    return df_clean


def get_valid_pdf_files(directory, min_size_kb):
    """
    Get list of PDF files that meet the minimum size requirement.
    
    Args:
        directory: Directory containing PDF files
        min_size_kb: Minimum file size in kilobytes
        
    Returns:
        List of valid PDF filenames
    """
    pdf_dir = os.path.join(directory, PDF_SUBDIR)
    pdf_infos = [
        (fname, os.path.getsize(os.path.join(pdf_dir, fname)) / 1024)
        for fname in os.listdir(pdf_dir) if fname.endswith('.pdf')
    ]
    print(f"Number of PDF files: {len(pdf_infos)}")
    
    valid_pdf_infos = [info for info in pdf_infos if info[1] >= min_size_kb]
    valid_pdf_fnames = [info[0] for info in valid_pdf_infos]
    print(f"Number of valid PDF files (>= {min_size_kb} KB): {len(valid_pdf_fnames)}")
    
    return valid_pdf_fnames


def load_metadata(directory):
    """
    Load metadata from JSON file.
    
    Args:
        directory: Directory containing metadata.json
        
    Returns:
        Loaded metadata as dictionary
    """
    metadata_path = os.path.join(directory, 'metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def create_pdf_to_doi_mapping(metadata, valid_pdf_files):
    """
    Create a mapping from PDF filenames to DOI from metadata.
    
    Args:
        metadata: Metadata dictionary
        valid_pdf_files: List of valid PDF filenames
        
    Returns:
        Dictionary mapping PDF filenames to DOIs
    """
    pdf_to_doi = {
        f"{key}.pdf": value['doi'] 
        for key, value in metadata.items() if 'doi' in value
    }
    return {k: v for k, v in pdf_to_doi.items() if k in valid_pdf_files}


def filter_dataframe_by_valid_pdfs(dataframe, pdf_to_doi_mapping):
    """
    Filter DataFrame to include only rows with valid PDF files.
    
    Args:
        dataframe: DataFrame to filter
        pdf_to_doi_mapping: Mapping from PDF filenames to DOIs
        
    Returns:
        Filtered DataFrame with added PDF filename column
    """
    # Create reverse mapping from DOI to PDF filename
    doi_to_pdf = {v: k for k, v in pdf_to_doi_mapping.items()}
    
    # Add PDF filename column based on DOI
    dataframe['valid_pdf_fname'] = dataframe['doi'].map(doi_to_pdf)
    
    # Keep only rows with valid PDF filenames
    filtered_df = dataframe.dropna(subset=['valid_pdf_fname'])
    print(f"Number of rows in the final DataFrame: {filtered_df.shape[0]}")
    
    return filtered_df


def save_output(dataframe, csv_path, pickle_path):
    """
    Save the final DataFrame to CSV and pickle files.
    
    Args:
        dataframe: DataFrame to save
        csv_path: Path for CSV output
        pickle_path: Path for pickle output
    """
    dataframe.to_csv(csv_path, index=False, encoding='utf-8-sig')
    dataframe.to_pickle(pickle_path)


def process_medical_papers():
    """
    Main function to process medical papers data.
    
    Returns:
        Final filtered DataFrame
    """
    # Load and combine CSV files
    df_combined = load_and_combine_csv_files(CSV_FILENAMES)
    
    # Clean the combined DataFrame
    df_clean = remove_duplicates_and_nulls(df_combined)
    
    # Get valid PDF files
    valid_pdf_files = get_valid_pdf_files(FULL_TEXT_DIR, MIN_PDF_SIZE_KB)
    
    # Load metadata
    metadata = load_metadata(FULL_TEXT_DIR)
    
    # Create PDF to DOI mapping
    pdf_to_doi = create_pdf_to_doi_mapping(metadata, valid_pdf_files)
    
    # Filter DataFrame by valid PDFs
    df_final = filter_dataframe_by_valid_pdfs(df_clean, pdf_to_doi)
    
    # Save results
    save_output(df_final, OUTPUT_CSV, OUTPUT_PICKLE)
    
    return df_final


if __name__ == "__main__":
    df_fin = process_medical_papers()