"""
Paper Downloader - A tool to download academic papers using various APIs

This module provides functionality to download academic papers using multiple sources
including Unpaywall, PMC, CrossRef, and publisher-specific APIs.
"""

import os
import json
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
import requests
from tqdm import tqdm
import cloudscraper

# Global Constants
OUTPUT_DIR = "papers"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
DEFAULT_EMAIL = "your-email@domain.com"
REQUEST_DELAY = 10  # Seconds between API requests
MINI_DELAY = 1  # Seconds between trying different sources

# Logging level: 0=none, 1=basic, 2=verbose
VERBOSE = 1

# Default HTTP headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Headers for bypassing Cloudflare protection
CLOUDFLARE_BYPASS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'DNT': '1',
}


@dataclass
class DownloadResult:
    """Store the result of a paper download attempt"""
    success: bool
    source: str
    pdf_content: bytes = None
    error_message: str = ""


def initialize_directories(output_dir=OUTPUT_DIR, pdf_dir=PDF_DIR):
    """Create necessary directories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)


def load_metadata(metadata_file=METADATA_FILE):
    """Load metadata with review_id based structure"""
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {
        "reviews": {},  # Will contain review_id based structure
        "files": {}     # Will maintain file path mapping
    }


def save_metadata(metadata, metadata_file=METADATA_FILE):
    """Save metadata to file"""
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_identifier(doi):
    """Generate a unique identifier for a DOI"""
    hash_object = hashlib.md5(doi.encode())
    return hash_object.hexdigest()[:10]


def create_cloudscraper():
    """Create a cloudscraper instance to bypass Cloudflare protection"""
    return cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
    )


def bypass_cloudflare(url, scraper):
    """
    Bypass Cloudflare protection and download PDF
    
    Args:
        url: PDF download URL
        scraper: Cloudscraper instance
            
    Returns:
        Tuple[bool, Optional[bytes], Optional[str]]: 
        - Success status
        - PDF content (if successful)
        - Error message (if failed)
    """
    try:
        response = scraper.get(url, headers=CLOUDFLARE_BYPASS_HEADERS)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type or response.content.startswith(b'%PDF'):
                return True, response.content, None
            else:
                return False, None, "Response is not a PDF"
        return False, None, f"Status code: {response.status_code}"
    except Exception as e:
        return False, None, str(e)


def try_unpaywall(doi, unpaywall_email=DEFAULT_EMAIL):
    """
    Try to download from Unpaywall API
    
    Args:
        doi: DOI of the paper
        unpaywall_email: Email for Unpaywall API
        
    Returns:
        DownloadResult object with download status and content
    """
    try:
        # Get PDF URL from Unpaywall API
        url = f"https://api.unpaywall.org/v2/{doi}?email={unpaywall_email}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pdf_url = None
            # Check best_oa_location first
            if data.get("best_oa_location"):
                pdf_url = (data["best_oa_location"].get("url_for_pdf") or 
                        data["best_oa_location"].get("pdf_url"))
            # Check all oa_locations if no PDF found
            if not pdf_url and data.get("oa_locations"):
                for location in data["oa_locations"]:
                    pdf_url = location.get("url_for_pdf") or location.get("pdf_url")
                    if pdf_url:
                        break
            if not pdf_url:
                return DownloadResult(False, "Unpaywall", None, "No PDF URL found")
            # Try to download PDF
            pdf_response = requests.get(pdf_url, headers=DEFAULT_HEADERS)
            if pdf_response.status_code == 200:
                return DownloadResult(True, "Unpaywall", pdf_response.content)
            if VERBOSE >= 1:
                print(f"Failed in Unpaywall: Status code {pdf_response.status_code}")
            return DownloadResult(False, "Unpaywall", None, "PDF download failed")
    except Exception as e:
        return DownloadResult(False, "Unpaywall", None, str(e))


def get_pmcid_from_doi(doi):
    """Get PMC ID from DOI using NCBI E-utilities"""
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={doi}[doi]"
        response = requests.get(search_url)
        tree = ET.fromstring(response.content)

        pmid = tree.find('.//Id').text
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(fetch_url)
        tree = ET.fromstring(response.content)

        pmc_id = tree.find(".//ArticleId[@IdType='pmc']")
        return pmc_id.text if pmc_id is not None else None
    except Exception:
        return None


def try_pmc(doi):
    """
    Try to download PDF from PMC using DOI
    
    Args:
        doi: DOI of the paper
        
    Returns:
        DownloadResult object with download status and content
    """
    try:
        # Try to get PMC ID from DOI
        pmc_id = get_pmcid_from_doi(doi)
        if not pmc_id:
            return DownloadResult(False, "PMC", None, "No PMC ID found")
        # Try to download PDF from PMC
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf"
        pdf_response = requests.get(pdf_url, headers=DEFAULT_HEADERS)
        if pdf_response.status_code == 200:
            return DownloadResult(True, "PMC", pdf_response.content)

        if VERBOSE >= 1:
            print(f"Failed in PMC: Status code {pdf_response.status_code}")
        return DownloadResult(False, "PMC", None, "PDF download failed")
    except Exception as e:
        return DownloadResult(False, "PMC", None, str(e))


def get_publisher_url(doi):
    """Get publisher URL from DOI using Crossref API"""
    try:
        api_url = f'https://api.crossref.org/works/{doi}'
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data['message'].get('resource', {}).get('primary', {}).get('URL')
        return None
    except Exception:
        return None


def try_url_based(doi, scraper):
    """
    Try to download PDF by appending /pdf to publisher URL obtained from Crossref
    
    Args:
        doi: DOI of the paper
        scraper: Cloudscraper instance
        
    Returns:
        DownloadResult object with download status and content
    """
    try:
        # Try to get publisher URL from DOI
        pub_url = get_publisher_url(doi)
        if not pub_url:
            return DownloadResult(False, "URL_BASED", None, "No publisher URL found")

        if 'tandfonline' in pub_url:
            pdf_url = f"{pub_url.replace('/full/', '/pdf/')}?download=true"
            success, content, error = bypass_cloudflare(pdf_url, scraper)
        elif 'academic.oup.com' in pub_url:
            pdf_url = \
                f"{pub_url.replace('/article/', '/article-pdf/')}/{doi.split('/')[-1]}.pdf"
            success, content, error = bypass_cloudflare(pdf_url, scraper)
        else:
            pdf_url = f"{pub_url}/pdf"
            pdf_response = requests.get(pdf_url, headers=DEFAULT_HEADERS)
            success = pdf_response.status_code == 200
            content = pdf_response.content if success else None
            error = None if success else "PDF download failed"

        if success:
            return DownloadResult(True, "URL_BASED", content)
        if VERBOSE >= 1:
            print(f"Failed in URL_BASED: {error}")
        return DownloadResult(False, "URL_BASED", None, error)

    except Exception as e:
        return DownloadResult(False, "URL_BASED", None, str(e))


def try_crossref(doi):
    """
    Try to get PDF URL from CrossRef API
    
    Args:
        doi: DOI of the paper
        
    Returns:
        DownloadResult object with download status and content
    """
    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            links = data.get("message", {}).get("link", [])
            for link in links:
                if link.get("content-type", "").lower() == "application/pdf":
                    pdf_url = link.get("URL")
                    if pdf_url:
                        pdf_response = requests.get(pdf_url, headers=DEFAULT_HEADERS)
                        if pdf_response.status_code == 200:
                            return DownloadResult(True, "CrossRef", pdf_response.content)
        if VERBOSE >= 1:
            print(f"Failed in CrossRef: No PDF URL found or download failed")
        return DownloadResult(False, "CrossRef", None, "No PDF available")
    except Exception as e:
        return DownloadResult(False, "CrossRef", None, str(e))


def try_wiley(doi, wiley_api_key=None):
    """
    Try to download from Wiley TDM API
    
    Args:
        doi: DOI of the paper
        wiley_api_key: API key for Wiley TDM API
        
    Returns:
        DownloadResult object with download status and content
    """
    if not wiley_api_key:
        return DownloadResult(False, "Wiley", None, "No API key provided")
    try:
        url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
        headers = {
            'Authorization': f'Bearer {wiley_api_key}',
            'Accept': 'application/pdf',
            'Wiley-TDM-Client-Token': wiley_api_key
        }

        if VERBOSE >= 1:
            print(f"\nTrying Wiley TDM API with URL: {url}")
            print(f"Headers: {headers}")
        pdf_response = requests.get(url, headers=headers)

        if VERBOSE >= 1:
            print(f"PDF response status: {pdf_response.status_code}")

        if pdf_response.status_code == 200 and pdf_response.content.startswith(b'%PDF'):
            return DownloadResult(True, "Wiley", pdf_response.content)

        if VERBOSE >= 1:
            print(f"Failed in Wiley: Unable to download PDF")
        return DownloadResult(False, "Wiley", None, "No PDF available")
    except Exception as e:
        return DownloadResult(False, "Wiley", None, str(e))


def try_taylor_francis(doi, tandf_api_key=None):
    """
    Try to download from Taylor & Francis Online API
    
    Args:
        doi: DOI of the paper
        tandf_api_key: API key for Taylor & Francis API
        
    Returns:
        DownloadResult object with download status and content
    """
    if not tandf_api_key:
        return DownloadResult(False, "Taylor & Francis", None, "No API key provided")
    
    try:
        url = f"https://api.tandfonline.com/articles/{doi}"
        headers = {
            'X-ApiKey': tandf_api_key,
            'Accept': 'application/pdf',
            'User-Agent': 'Your-Application-Name/1.0',
            'Origin': 'https://api.tandfonline.com'
        }
        
        # Add proper session handling
        session = requests.Session()
        response = session.get(url, headers=headers)
        
        if response.status_code == 200 and response.content:
            return DownloadResult(True, "Taylor & Francis", response.content)
            
        if response.status_code == 403:
            return DownloadResult(False, "Taylor & Francis", None, "Authentication failed - check API key")
            
        return DownloadResult(False, "Taylor & Francis", None, f"HTTP {response.status_code}")
        
    except Exception as e:
        return DownloadResult(False, "Taylor & Francis", None, str(e))


def try_springer(doi, springer_api_key=None):
    """
    Try to download from Springer API
    
    Args:
        doi: DOI of the paper
        springer_api_key: API key for Springer API
        
    Returns:
        DownloadResult object with download status and content
    """
    if not springer_api_key:
        return DownloadResult(False, "Springer", None, "No API key provided")
    try:
        url = f"https://api.springernature.com/metadata/json"
        params = {
            'q': f'doi:{doi}',
            's': 1,
            'p': 1,
            'api_key': springer_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('records'):
                pdf_url = data['records'][0].get('url', [{}])[0].get('value')
                if pdf_url:
                    pdf_response = requests.get(pdf_url, headers=DEFAULT_HEADERS)
                    if pdf_response.status_code == 200:
                        return DownloadResult(True, "Springer", pdf_response.content)
        if VERBOSE >= 1:
            print(f"Failed in Springer: No PDF URL found or download failed")
        return DownloadResult(False, "Springer", None, "No PDF URL found")
    except Exception as e:
        return DownloadResult(False, "Springer", None, str(e))


def try_sciencedirect(doi, sciencedirect_api_key=None):
    """
    Try to download from ScienceDirect API
    
    Args:
        doi: DOI of the paper
        sciencedirect_api_key: API key for ScienceDirect API
        
    Returns:
        DownloadResult object with download status and content
    """
    if not sciencedirect_api_key:
        return DownloadResult(False, "ScienceDirect", None, "No API key provided")
    try:
        url = f"https://api.elsevier.com/content/article/doi/{doi}"
        headers = {
            'X-ELS-APIKey': sciencedirect_api_key,
            'Accept': 'application/pdf'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.content:
            return DownloadResult(True, "ScienceDirect", response.content)
        if VERBOSE >= 1:
            print(f"Failed in ScienceDirect: Status code {response.status_code}")
        return DownloadResult(False, "ScienceDirect", None, f"HTTP {response.status_code}")
    except Exception as e:
        return DownloadResult(False, "ScienceDirect", None, str(e))


def try_oxford_academic(doi, oxford_academic_api_key=None):
    """
    Try to download from Oxford Academic API
    
    Args:
        doi: DOI of the paper
        oxford_academic_api_key: API key for Oxford Academic API
        
    Returns:
        DownloadResult object with download status and content
    """
    if not oxford_academic_api_key:
        return DownloadResult(False, "Oxford Academic", None, "No API key provided")
    try:
        url = f"https://academic.oup.com/api/pdf/{doi}"
        headers = {
            'Authorization': f'Bearer {oxford_academic_api_key}',
            'Accept': 'application/pdf'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.content:
            return DownloadResult(True, "Oxford Academic", response.content)
        if VERBOSE >= 1:
            print(f"Failed in Oxford Academic: Status code {response.status_code}")
        return DownloadResult(False, "Oxford Academic", None, f"HTTP {response.status_code}")
    except Exception as e:
        return DownloadResult(False, "Oxford Academic", None, str(e))


def download_papers(dois, output_dir=OUTPUT_DIR, pdf_dir=PDF_DIR, unpaywall_email=DEFAULT_EMAIL,
                   elsevier_api_key=None, wiley_api_key=None, tandf_api_key=None,
                   springer_api_key=None, sciencedirect_api_key=None, oxford_academic_api_key=None):
    """
    Download papers using multiple sources in order of preference
    
    Args:
        dois: List of DOIs to download
        output_dir: Directory to save output
        pdf_dir: Directory to save PDFs
        unpaywall_email: Email for Unpaywall API
        *_api_key: API keys for various publishers
        
    Returns:
        Dictionary with DOIs as keys and download status as values
    """
    # Initialize directories and metadata
    initialize_directories(output_dir, pdf_dir)
    metadata = load_metadata(os.path.join(output_dir, "metadata.json"))
    metadata_file = os.path.join(output_dir, "metadata.json")
    
    # Create cloudscraper instance
    scraper = create_cloudscraper()
    
    results = {}
    for doi in tqdm(dois):
        if not doi or '/' not in doi:
            results[doi] = "Invalid DOI format"
            continue

        identifier = generate_identifier(doi)
        pdf_path = os.path.join(pdf_dir, f"{identifier}.pdf")
        if identifier in metadata:
            results[doi] = "Already downloaded"
            if VERBOSE >= 1:
                print(f"DOI ({doi}) is already downloaded!")
            continue

        # Define source functions with their required parameters
        sources = [
            lambda d=doi: try_crossref(d),
            lambda d=doi: try_wiley(d, wiley_api_key),
            lambda d=doi: try_pmc(d),
            lambda d=doi: try_url_based(d, scraper),
            # lambda d=doi: try_taylor_francis(d, tandf_api_key),
            # lambda d=doi: try_springer(d, springer_api_key),
            # lambda d=doi: try_sciencedirect(d, sciencedirect_api_key),
            # lambda d=doi: try_oxford_academic(d, oxford_academic_api_key),
            lambda d=doi: try_unpaywall(d, unpaywall_email),
        ]

        time.sleep(REQUEST_DELAY)  # Rate limiting delay
        success = False
        for source_func in sources:
            source_name = source_func.__name__ if hasattr(source_func, '__name__') else source_func.__class__.__name__
            if VERBOSE > 0:
                print(f"Trying {doi} with {source_name}...")
            result = source_func()
            if result is None:
                result = DownloadResult(False, "Unknown", None, "Result is None")
            if result.success and result.pdf_content:
                try:
                    with open(pdf_path, 'wb') as f:
                        f.write(result.pdf_content)
                    success = True
                    # Update metadata
                    metadata[identifier] = {
                        "doi": doi,
                        "downloaded_from": result.source,
                        "download_date": datetime.now().isoformat(),
                        "file_path": pdf_path
                    }
                    save_metadata(metadata, metadata_file)
                    results[doi] = f"Successfully downloaded from {result.source}"
                    break
                except Exception as e:
                    print(f"Error saving PDF: {e}")
            time.sleep(MINI_DELAY)  # Rate limiting delay between sources

        if not success:
            results[doi] = "Failed to download from all sources"
            print(f"DOI ({doi}) failed to download from all sources!")

    return results


# Usage example
# results = download_papers(
#     dois=["10.1038/nature12373", "10.1126/science.1259855"],
#     output_dir="research_papers",
#     unpaywall_email="your-email@example.com",
#     elsevier_api_key="your-elsevier-api-key",
#     wiley_api_key="your-wiley-api-key",
#     springer_api_key="your-springer-api-key",
#     sciencedirect_api_key="your-sciencedirect-api-key"
# )