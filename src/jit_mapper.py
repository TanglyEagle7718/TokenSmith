import json
import os
import pathlib
from typing import List, Tuple, Set

def get_keywords(query: str) -> List[str]:
    stopwords = {
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what",
        "how", "can", "i", "do", "you", "tell", "me", "about", "explain", "describe",
        "why", "when", "where", "who", "does", "be"
    }
    words = query.lower().split()
    return [word.strip('.,!?()[]:"\'') for word in words if word not in stopwords]

def identify_sections_to_embed(
    query: str, 
    extracted_index_path: str, 
    chunk_map_path: str
) -> Tuple[List[int], List[int]]:
    if not os.path.exists(extracted_index_path):
        return [], []
    
    if not os.path.exists(chunk_map_path):
        return [], []

    keywords = get_keywords(query)
    
    with open(extracted_index_path, 'r') as f:
        extracted_index = json.load(f)
    with open(chunk_map_path, 'r') as f:
        chunk_map = json.load(f)

    normalized_index = {k.lower(): v for k, v in extracted_index.items()}
        
    relevant_pages = set()
    for kw in keywords:
        if kw in normalized_index:
            pages = normalized_index[kw]
            if isinstance(pages, list):
                relevant_pages.update(pages)
            else:
                relevant_pages.add(pages)
        
        for entry, pages in normalized_index.items():
            if kw in entry.split():
                if isinstance(pages, list):
                    relevant_pages.update(pages)
                else:
                    relevant_pages.add(pages)
    
    chunks_to_embed = set()
    for page in relevant_pages:
        page_str = str(page)
        if page_str in chunk_map:
            chunks_to_embed.update(chunk_map[page_str])
            
    return sorted(list(chunks_to_embed)), sorted(list(relevant_pages))
