"""
Utility functions for parsing and processing LLM responses for fulltext review.
This module provides functions to handle structured data extraction from LLM responses.
"""
import re
from copy import deepcopy
from typing import Optional, Dict, Any, List

# Constants
SECTION_TAGS = [
    "PAPER_TYPE", "BIBLIOGRAPHIC_DATES", "CLINICAL_DOMAIN",
    "MODELS", "EXPERIMENTAL_SETTINGS", "HUMAN_GROUPS",
    "EVALUATION_TASKS", "PERFORMANCE_RESULTS"
]

FIRST_OCCURRENCE_FIELDS = ["PAPER_TYPE", "BIBLIOGRAPHIC_DATES", "CLINICAL_DOMAIN"]
LIST_FIELDS = ["HUMAN_GROUPS", "EVALUATION_TASKS", "PERFORMANCE_RESULTS"]

# Default empty response structure
DEFAULT_EOF_ERROR_RESPONSE = {
    "PAPER_TYPE": "EOF PDF Error",
    "BIBLIOGRAPHIC_DATES": None,
    "CLINICAL_DOMAIN": None,
    "MODELS": None,
    "EXPERIMENTAL_SETTINGS": None,
    "HUMAN_GROUPS": None,
    "EVALUATION_TASKS": None,
    "PERFORMANCE_RESULTS": None
}


def clean_section_content(content: str) -> str:
    """Clean section content by removing newlines and extra whitespace."""
    return ' '.join(content.split())


def _remove_comments(json_string: str) -> str:
    """
    Remove comments and normalize boolean values in JSON strings.
    
    Args:
        json_string: JSON string that may contain comments
    
    Returns:
        Cleaned JSON string with comments removed and normalized boolean values
    """
    # Remove comments that start with # or // and end before any quote
    json_string = re.sub(r'(?:#|//)[^"\'\n]*(?=["\'\n]|$)', '', json_string)
    
    # Remove extra whitespace and empty lines
    json_string = '\n'.join(line.strip() for line in json_string.split('\n') if line.strip())
    
    # Normalize boolean and null values to Python format
    replacements = {
        'true': 'True', 'TRUE': 'True',
        'false': 'False', 'FALSE': 'False',
        'null': 'None'
    }
    
    for old, new in replacements.items():
        json_string = json_string.replace(old, new)
        
    return json_string


def parse_llm_response(response: str, eof_error: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response into structured data.
    
    Args:
        response: Raw LLM response text
        eof_error: Flag indicating if EOF error occurred during processing
        
    Returns:
        Dictionary with parsed data or None if parsing failed
    """
    if eof_error:
        return DEFAULT_EOF_ERROR_RESPONSE

    try:
        # Extract content between ANALYSIS_START and ANALYSIS_END markers
        content = response.split("ANALYSIS_START")[1].split("ANALYSIS_END")[0].strip()
        parsed_data = {}
        
        # Parse each section
        for section in SECTION_TAGS:
            start_tag = f"<{section}>"
            end_tag = f"</{section}>"
            
            if start_tag in content and end_tag in content:
                section_content = content.split(start_tag)[1].split(end_tag)[0].strip()
                section_content = clean_section_content(section_content)
                section_content = _remove_comments(section_content)
                
                try:
                    parsed_data[section] = eval(section_content)
                except Exception as e:
                    print(f"Error parsing LLM response section {section}: {e}")
                    print(f"Section content: {section_content}")
                    parsed_data[section] = section_content

        return parsed_data
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None


def merge_parsed_data(data_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Merge multiple parsed data dictionaries with specific rules for each field.
    
    Args:
        data_list: List of dictionaries containing parsed data
        
    Returns:
        Merged dictionary or None if no valid data
    """
    # Handle empty or invalid input
    valid_data = [d for d in data_list if d is not None]
    if not valid_data:
        return None

    merged = {}
    
    # 1. Use first occurrence for certain fields
    for field in FIRST_OCCURRENCE_FIELDS:
        for data in valid_data:
            if data.get(field):
                merged[field] = deepcopy(data[field])
                break
        if field not in merged:
            merged[field] = None

    # 2. Process and deduplicate MODELS
    merged["MODELS"] = _merge_models_data(valid_data)

    # 3. Process EXPERIMENTAL_SETTINGS with voting
    merged["EXPERIMENTAL_SETTINGS"] = _merge_experimental_settings(valid_data)

    # 4. Process and deduplicate list fields
    for field in LIST_FIELDS:
        merged[field] = _merge_list_field(valid_data, field)
    
    return merged


def _merge_models_data(valid_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Merge and deduplicate model data based on common_name.
    
    Args:
        valid_data: List of valid data dictionaries
        
    Returns:
        List of unique model dictionaries or None if no models
    """
    all_models = []
    
    for data in valid_data:
        models = data.get("MODELS")
        if isinstance(models, list):
            all_models.extend(deepcopy(models))
    
    # No models found
    if not all_models:
        return None
        
    # Remove duplicates based on common_name
    seen_models = {}
    unique_models = []
    
    for model in all_models:
        if isinstance(model, dict) and "common_name" in model:
            if model["common_name"] not in seen_models:
                seen_models[model["common_name"]] = model
                unique_models.append(model)
                
    return unique_models


def _merge_experimental_settings(valid_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Merge experimental settings by voting on the most frequent value for each setting.
    
    Args:
        valid_data: List of valid data dictionaries
        
    Returns:
        Dictionary with merged experimental settings or None if no settings
    """
    # Track vote counts for each setting value
    settings_count = {}
    
    for data in valid_data:
        exp_setting = data.get("EXPERIMENTAL_SETTINGS")
        if not isinstance(exp_setting, dict):
            continue
            
        for key, value in exp_setting.items():
            if key not in settings_count:
                settings_count[key] = {}
                
            str_value = str(value)  # Convert to string for counting
            settings_count[key][str_value] = settings_count[key].get(str_value, 0) + 1
    
    # No settings found
    if not settings_count:
        return None
        
    # Select most frequent value for each setting
    merged_settings = {}
    
    for key, value_counts in settings_count.items():
        # Find most frequent value (or first in case of ties)
        most_frequent = max(
            value_counts.items(),
            key=lambda x: (x[1], -list(value_counts.keys()).index(x[0]))
        )
        
        # Convert back to appropriate type
        value_str = most_frequent[0]
        if value_str.lower() == 'true':
            merged_settings[key] = True
        elif value_str.lower() == 'false':
            merged_settings[key] = False
        else:
            merged_settings[key] = value_str
            
    return merged_settings


def _merge_list_field(valid_data: List[Dict[str, Any]], field: str) -> Optional[List[Any]]:
    """
    Merge and deduplicate list fields.
    
    Args:
        valid_data: List of valid data dictionaries
        field: Field name to merge
        
    Returns:
        List of unique items or None if no items
    """
    all_items = []
    
    for data in valid_data:
        items = data.get(field)
        if isinstance(items, list):
            all_items.extend(deepcopy(items))
    
    # No items found
    if not all_items:
        return None
        
    # Remove duplicates based on string representation
    seen = set()
    unique_items = []
    
    for item in all_items:
        item_str = str(item)
        if item_str not in seen:
            seen.add(item_str)
            unique_items.append(item)
            
    return unique_items