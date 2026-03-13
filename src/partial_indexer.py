from abc import ABC, abstractmethod
import os
import pathlib
import json
import re
from typing import List, Optional, Dict
from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown

class PartialIndexer(ABC):
    @abstractmethod
    def build_index(
        self,
        markdown_file: str,
        *,
        chunker: DocumentChunker,
        chunk_config: ChunkConfig,
        embedding_model_path: str,
        artifacts_dir: os.PathLike,
        index_prefix: str,
        use_multiprocessing: bool = False,
        use_headings: bool = False,
        chapters_to_index: Optional[List[int]] = None
    ) -> None:
        pass

    @abstractmethod
    def add_to_index(
        self,
        markdown_file: str,
        *,
        chunker: DocumentChunker,
        chunk_config: ChunkConfig,
        embedding_model_path: str,
        artifacts_dir: os.PathLike,
        index_prefix: str,
        chapters_to_add: List[int],
        use_multiprocessing: bool = False,
        use_headings: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def jit_index(
        self,
        question: str,
        *,
        markdown_file: str,
        chunker: DocumentChunker,
        chunk_config: ChunkConfig,
        embedding_model_path: str,
        artifacts_dir: os.PathLike,
        index_prefix: str,
        extracted_index_path: str = "data/extracted_index.json",
        use_headings: bool = False,
    ) -> List[int]:
        pass

    def get_page_to_chapter_map(self, markdown_file: str) -> Dict[str, int]:
        """Shared helper to build map of Page -> Chapter."""
        cache_path = pathlib.Path("index/cache/page_to_chapter_map.json")
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        print("Building page-to-chapter map...")
        sections = extract_sections_from_markdown(markdown_file)
        page_to_chapter = {}
        current_page = 1
        page_pattern = re.compile(r'--- Page (\d+) ---')
        
        for s in sections:
            chap = s.get('chapter', 0)
            content = s.get('content', '')
            page_to_chapter[str(current_page)] = chap
            matches = page_pattern.findall(content)
            for m in matches:
                current_page = int(m) + 1
                page_to_chapter[str(current_page)] = chap
                
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(page_to_chapter, f, indent=2)
        return page_to_chapter

    def _get_keywords(self, question: str) -> List[str]:
        """Shared keyword extraction."""
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what",
            "how", "can", "i", "do", "you", "tell", "me", "about"
        }
        words = question.lower().split()
        return [word.strip('.,!?()[]') for word in words if word not in stopwords]
