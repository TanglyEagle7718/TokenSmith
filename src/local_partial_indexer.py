import os
import pathlib
import json
from typing import List, Optional
from src.partial_indexer import PartialIndexer
from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.index_builder import build_index
from src.index_updater import add_to_index

class LocalPartialIndexer(PartialIndexer):
    def build_index(self, markdown_file, **kwargs):
        build_index(markdown_file=markdown_file, **kwargs)

    def add_to_index(self, markdown_file, **kwargs):
        add_to_index(markdown_file=markdown_file, **kwargs)

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
        keywords = self._get_keywords(question)
        if not os.path.exists(extracted_index_path): return []
            
        with open(extracted_index_path, 'r') as f:
            extracted_index = json.load(f)
            
        relevant_pages = set()
        for word in keywords:
            if word in extracted_index:
                relevant_pages.update(extracted_index[word])
        
        if not relevant_pages: return []
            
        page_to_chapter = self.get_page_to_chapter_map(markdown_file)
        target_chapters = {page_to_chapter[str(p)] for p in relevant_pages if str(p) in page_to_chapter}
        
        artifacts_dir = pathlib.Path(artifacts_dir)
        info_path = artifacts_dir / f"{index_prefix}_info.json"
        existing_chapters = []
        if info_path.exists():
            with open(info_path, 'r') as f:
                existing_chapters = json.load(f).get("chapters", [])
        
        chapters_to_add = sorted(list(target_chapters - set(existing_chapters)))
        if not chapters_to_add: return []
            
        print(f"Local JIT Indexing chapters: {chapters_to_add}")
        self.add_to_index(
            markdown_file=markdown_file,
            chunker=chunker,
            chunk_config=chunk_config,
            embedding_model_path=embedding_model_path,
            artifacts_dir=artifacts_dir,
            index_prefix=index_prefix,
            chapters_to_add=chapters_to_add,
            use_headings=use_headings
        )
        return chapters_to_add
