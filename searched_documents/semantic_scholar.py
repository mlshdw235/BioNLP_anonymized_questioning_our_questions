"""Module for searching and processing papers from Semantic Scholar API."""
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import json

import requests
import pandas as pd
from ratelimit import limits, sleep_and_retry

# Constants
LOG_FILE = 'semantic_scholar_search.log'
LOG_MAX_SIZE = 10_000_000  # 10MB
LOG_BACKUP_COUNT = 5
DATA_EXPORT_PREFIX = "medical_llm_papers"

# API Configuration
API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json"
}
if API_KEY and API_KEY != "YOUR_API_KEY":
    HEADERS["x-api-key"] = API_KEY

# API Rate Limiting
CALLS_PER_MINUTE = 30
ONE_MINUTE = 60
MAX_RETRIES = 3
RETRY_DELAY = 2

# Search Parameters
SEARCH_YEAR_RANGE = "2022-2025"
RESULTS_PER_PAGE = 100
MAX_RESULTS_LIMIT = 10000  # API won't return more than 10k results

# Feature Flags
VERBOSE = 0  # Logging verbosity level
ONLY_FOR_FIRST_QUERY = False  # Limit search to just the first query for testing
ABOUT_MEDQA = False  # Switch between MedQA and clinical applications

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_SIZE,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_log(message):
    """Log message safely, handling encoding issues."""
    try:
        logger.info(message)
    except UnicodeEncodeError:
        logger.info(message.encode('ascii', 'replace').decode())

@dataclass
class Paper:
    """Paper class with Semantic Scholar metadata."""
    title: str
    authors: List[str]
    year: int
    venue: Optional[str] = None
    citations: int = 0
    url: Optional[str] = None
    abstract: Optional[str] = None
    paper_id: Optional[str] = None
    doi: Optional[str] = None
    fields_of_study: List[str] = None
    references_count: int = 0
    citations_count: int = 0

    def __hash__(self):
        return hash(self.paper_id if self.paper_id else self.title)

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return NotImplemented
        return self.title and self.title.lower().strip() == other.title.lower().strip()

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
def make_api_request(endpoint: str, params: Dict = None) -> Dict:
    """Make rate-limited API request to Semantic Scholar."""
    url = f"{API_BASE_URL}/{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            if "x-api-key" not in HEADERS:
                time.sleep(3)  # Add delay for unauthorized requests
            
            response = requests.get(url, headers=HEADERS, params=params)
            
            if response.status_code == 403 and "x-api-key" not in HEADERS:
                logger.warning("Rate limit exceeded for unauthorized request. Waiting longer...")
                time.sleep(10)  # Longer wait for rate limit issues
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                raise
            backoff_time = 2 ** attempt  # Exponential backoff
            time.sleep(backoff_time)

    return {}

def search_papers(query: str, offset: int = 0, limit: int = RESULTS_PER_PAGE, max_results: int = None) -> Dict:
    """Search for papers with pagination.
    
    Args:
        query: Search query string
        offset: Starting point for results
        limit: Number of results per page
        max_results: Maximum total results to retrieve (None for all available)
    """
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "fields": "title,authors,year,venue,abstract,url,citations,references,fieldsOfStudy",
        "year": SEARCH_YEAR_RANGE
    }

    try:
        response = make_api_request("paper/search", params)
        total_available = response.get('total', 0)
        
        if VERBOSE >= 1:
            safe_log(f"Query: {query}")
            safe_log(f"Total available results: {total_available}")
            safe_log(f"Current page results ({offset}-{offset+limit}):")
            for paper in response.get('data', []):
                title = paper.get('title', 'No title')
                year = paper.get('year', 'No year')
                safe_log(f"- [{year}] {title}")
                
        return response

    except Exception as e:
        logger.error(f"Error in search_papers: {e}")
        return {"total": 0, "data": []}

def get_paper_details(paper_id: str) -> Dict:
    """Get detailed paper information by ID."""
    return make_api_request(f"paper/{paper_id}")

def get_query_groups() -> Dict[str, List[str]]:
    """Get the query term groups for search construction."""
    llm_terms = [
        "large language model", "language model",
        "GPT-4", "ChatGPT", 
        # "Llama"
    ]
    
    # MedQA related terms
    medqa_terms = [
        "medical question answering", "USMLE", "MedQA",
        "medical benchmark", "clinical reasoning"
    ]

    # Extended clinical application terms
    clinical_terms = [
        # Medical Specialties
        "internal medicine", "surgery", "pediatrics", "obstetrics", "gynecology",
        "ophthalmology", "dentistry", "psychiatry", "neurology", "cardiology",
        "dermatology", "orthopedics", "urology", "radiology",
        "endocrinology", "gastroenterology", "hematology", "oncology",
        "rheumatology", "pulmonology", "nephrology", "infectious disease",
        "allergy immunology", "palliative care", "geriatrics",
        "anatomical pathology", "clinical pathology", "anesthesiology",
        "emergency medicine", "family medicine", "preventive medicine",

        # Surgery settings
        "surgery", "pediatric surgery", "breast surgery",
        "colorectal surgery", "vascular surgery",
        "neurosurgery", "plastic surgery", "hand surgery",
        "spine surgery", "bariatric surgery",

        # Clinical Settings
        "emergency department", "icu", "operating room", "outpatient",
        "primary care", "trauma center", 

        # Clinical Documents and Tasks
        "electronic health record", "clinical notes", "discharge summary",
        "medical history", "radiology report", "medication orders",
        "care plans", "treatment protocols", "clinical decision support",
        "clinical guidelines", "medical coding",

        # Common Diseases and Conditions
        "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
        "leukemia", "lymphoma", "multiple myeloma",
        "diabetes mellitus", "hypertension",
        "heart failure", "coronary artery disease", "atrial fibrillation", "stroke",
        "alzheimer disease", "parkinson disease",
        "multiple sclerosis", "epilepsy", "depression", "anxiety disorder",
        "asthma", "copd", "pneumonia", "tuberculosis",
        "hiv aids", "hepatitis", "cirrhosis", "inflammatory bowel disease",
        "rheumatoid arthritis", "osteoarthritis",
        "chronic kidney disease", "osteoporosis", "thyroid disorders", "cystic fibrosis",

        # Clinical Procedures and Interventions
        "chemotherapy", "radiation therapy",
        "transplantation", "dialysis", "ventilation", "ecmo",

        # Age-Specific Care (Pediatrics)
        "newborn care", "child development", "growth disorders", "birth defects",

        # Age-Specific Care (Geriatrics)
        "falls prevention", "memory disorders", "polypharmacy management", 

        # Special Populations and Considerations
        "maternal health", "prenatal care", "postpartum care",
        "women's health", "social determinants", "medical ethics",
    ]

    eval_terms = [
        "evaluation", "accuracy", "benchmark",
        "validation", "application",
    ]

    return {
        "llm": llm_terms,
        "medical_qa": medqa_terms if ABOUT_MEDQA else list(set(clinical_terms)),
        "evaluation": eval_terms
    }

def build_queries() -> List[str]:
    """Build search queries combining terms from all groups."""
    groups = get_query_groups()
    queries = []
    
    for llm in groups["llm"]:
        for qa in groups["medical_qa"]:
            for eval_term in groups["evaluation"]:
                base_query = f'{llm} AND {qa} AND {eval_term}'
                queries.append(base_query)

    return queries

def process_paper_data(data: Dict) -> Optional[Paper]:
    """Convert API response to Paper object."""
    try:
        if not data.get('title') or not data.get('year'):
            return None

        # Extract author names with robust error handling
        authors = []
        raw_authors = data.get('authors', [])
        if raw_authors:
            for author in raw_authors:
                if isinstance(author, dict):
                    author_name = author.get('name', '')
                elif isinstance(author, str):
                    author_name = author
                else:
                    continue
                    
                if author_name:
                    authors.append(author_name)

        # Safe type conversion for year
        year = int(data['year'])
        if not 2022 <= year <= 2025:  # Filter by year range
            return None

        return Paper(
            title=data['title'],
            authors=authors,
            year=year,
            venue=data.get('venue', {}).get('name', '') \
                if isinstance(data.get('venue'), dict) else data.get('venue', ''),
            abstract=data.get('abstract', ''),
            paper_id=data.get('paperId'),
            doi=data.get('doi'),
            fields_of_study=data.get('fieldsOfStudy', []),
            references_count=len(data.get('references', [])),
            citations_count=len(data.get('citations', []))
        )

    except Exception as e:
        logger.error(f"Error processing paper data: {e}")
        logger.debug(f"Problematic data: {data}")
        return None

def add_papers_from_response(papers_set: Set[Paper], response_data: Dict) -> int:
    """Process papers from API response and add to collection."""
    added_count = 0
    skipped_count = 0
    
    for paper_data in response_data.get('data', []):
        paper = process_paper_data(paper_data)
        
        if paper and paper.year >= 2022:
            if paper not in papers_set:
                papers_set.add(paper)
                added_count += 1
                logger.debug(f"Added paper: {paper.title}")
            else:
                skipped_count += 1
                logger.debug(f"Skipped duplicate: {paper.title}")
        else:
            logger.debug(f"Skipped paper (year/processing): {paper_data.get('title')}")

    logger.info(f"Added: {added_count}, Skipped: {skipped_count}")
    return added_count

def save_results(papers: Set[Paper], prefix: str = DATA_EXPORT_PREFIX):
    """Save results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert papers to DataFrame-compatible format
    df_data = []
    for paper in papers:
        df_data.append({
            "Title": paper.title,
            "Authors": "; ".join(paper.authors),
            "Year": paper.year,
            "Venue": paper.venue,
            "Citations": paper.citations_count,
            "References": paper.references_count,
            "Abstract": paper.abstract,
            "Paper_ID": paper.paper_id,
            "Fields_of_Study": "; ".join(paper.fields_of_study or [])
        })

    df = pd.DataFrame(df_data)
    
    # Create CSV filename based on search type
    csv_path = f"{prefix}_{timestamp}"
    if ABOUT_MEDQA:
        csv_path += '_medqa.csv'
    else:
        csv_path += '_generalmed.csv'
        
    # Save CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Build search metadata
    metadata = {
        "timestamp": timestamp,
        "total_papers": len(papers),
        "year_distribution": df["Year"].value_counts().to_dict(),
        "venue_distribution": df["Venue"].value_counts().to_dict(),
        "fields_distribution": df["Fields_of_Study"]
            .str.split("; ")
            .explode()
            .value_counts()
            .to_dict()
    }

    # Create JSON filename based on search type
    metadata_path = f"{prefix}_metadata_{timestamp}"
    if ABOUT_MEDQA:
        metadata_path += '_medqa.json'
    else:
        metadata_path += '_generalmed.json'
        
    # Save JSON metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main execution function for the Semantic Scholar search process."""
    try:
        logger.info("Starting Semantic Scholar medical LLM paper search")
        papers_collection = set()
        queries = build_queries()

        # Limit to first query if testing
        if ONLY_FOR_FIRST_QUERY:
            queries = queries[:1]
            
        total_queries = len(queries)
        total_papers = 0
        
        # Process each query
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{total_queries}: {query}")
            try:
                # Initial search
                response = search_papers(query)
                papers_added = add_papers_from_response(papers_collection, response)
                total_papers += papers_added
                
                # Get total count and paginate if necessary
                total_results = response.get('total', 0)
                if total_results > RESULTS_PER_PAGE:
                    # API won't return more than 10k results 
                    max_offset = min(total_results, MAX_RESULTS_LIMIT)
                    
                    for offset in range(RESULTS_PER_PAGE, max_offset, RESULTS_PER_PAGE):
                        logger.info(
                            f"Fetching results {offset}-{offset+RESULTS_PER_PAGE} of {total_results}"
                        )
                        # Add delay between requests to respect rate limits
                        time.sleep(1)
                        response = search_papers(query, offset=offset)
                        
                        if not response.get('data'):
                            logger.warning(f"No more results after offset {offset}")
                            break
                            
                        papers_added = add_papers_from_response(papers_collection, response)
                        total_papers += papers_added
                        
                        # Log progress
                        logger.info(f"Retrieved {papers_added} papers (Total: {total_papers})")

            except Exception as e:
                logger.error(f"Error processing query: {e}, continuing with next query")
                continue

        # Save results if papers were found
        if papers_collection:
            save_results(papers_collection)
            logger.info(f"\nSearch completed. Total unique papers found: {total_papers}")
        else:
            logger.warning("No results found in the search")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()