import os
import pathlib
import pickle
import json
import re
from typing import List, Dict, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.partial_indexer import PartialIndexer
from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.embedder import GeminiEmbedder
from src.index_builder import preprocess_for_bm25

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

class CloudPartialIndexer(PartialIndexer):
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
        artifacts_dir = pathlib.Path(artifacts_dir)
        sections = extract_sections_from_markdown(markdown_file, exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS)
        if chapters_to_index:
            sections = [s for s in sections if s.get('chapter') in chapters_to_index]

        all_chunks, sources, metadata = [], [], []
        page_to_chunk_ids, current_page, total_chunks = {}, 1, 0
        heading_stack, page_pattern = [], re.compile(r'--- Page (\d+) ---')

        for c in sections:
            while heading_stack and heading_stack[-1][0] >= c.get('level', 1): heading_stack.pop()
            if c['heading'] != "Introduction": heading_stack.append((c.get('level', 1), c['heading']))
            full_path = f"Chapter {c.get('chapter', 0)} " + " ".join([h[1] for h in heading_stack])
            sub_chunks = chunker.chunk(c['content'])

            for sub_id, sub_chunk in enumerate(sub_chunks):
                frags = page_pattern.split(sub_chunk)
                if frags[0].strip():
                    page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks + sub_id)
                for j in range(1, len(frags), 2):
                    current_page = int(frags[j]) + 1
                    if frags[j+1].strip(): page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks + sub_id)

                clean = re.sub(page_pattern, '', sub_chunk).strip()
                if c["heading"] == "Introduction": continue
                all_chunks.append((f"Description: {full_path} Content: " if use_headings else "") + clean)
                sources.append(markdown_file)
                metadata.append({"filename": markdown_file, "section": c['heading'], "section_path": full_path, "chunk_id": total_chunks + sub_id})
            total_chunks += len(sub_chunks)

        with open(artifacts_dir / f"{index_prefix}_page_to_chunk_map.json", "w") as f:
            json.dump({p: sorted(list(ids)) for p, ids in page_to_chunk_ids.items()}, f, indent=2)

        print(f"Embedding {len(all_chunks):,} chunks via Gemini Cloud...")
        embeddings = GeminiEmbedder().encode(all_chunks, show_progress_bar=True)
        faiss_idx = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_idx.add(embeddings)
        faiss.write_index(faiss_idx, str(artifacts_dir / f"{index_prefix}.faiss"))

        bm25 = BM25Okapi([preprocess_for_bm25(c) for c in all_chunks])
        with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f: pickle.dump(bm25, f)
        with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f: pickle.dump(all_chunks, f)
        with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f: pickle.dump(sources, f)
        with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f: pickle.dump(metadata, f)
        with open(artifacts_dir / f"{index_prefix}_info.json", "w") as f:
            json.dump({"chapters": chapters_to_index or ["all"], "type": "cloud"}, f, indent=2)

    def add_to_index(self, markdown_file, **kwargs):
        artifacts_dir = pathlib.Path(kwargs['artifacts_dir'])
        if not (artifacts_dir / f"{kwargs['index_prefix']}.faiss").exists():
            return self.build_index(markdown_file=markdown_file, chapters_to_index=kwargs.get('chapters_to_add'), **kwargs)

        print(f"Adding chapters {kwargs.get('chapters_to_add')} to existing cloud index...")
        with open(artifacts_dir / f"{kwargs['index_prefix']}_chunks.pkl", "rb") as f: existing_chunks = pickle.load(f)
        with open(artifacts_dir / f"{kwargs['index_prefix']}_info.json", "r") as f: index_info = json.load(f)
        
        new_chaps = list(set(kwargs.get('chapters_to_add', [])) - set(index_info.get("chapters", [])))
        if not new_chaps: return

        sections = [s for s in extract_sections_from_markdown(markdown_file, exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS) if s.get('chapter') in new_chaps]
        new_chunks, new_sources, new_meta = [], [], []
        for c in sections:
            for sc in kwargs['chunker'].chunk(c['content']):
                if c["heading"] == "Introduction": continue
                new_chunks.append(sc.strip())
                new_sources.append(markdown_file)
                new_meta.append({"filename": markdown_file, "section": c['heading'], "chunk_id": len(existing_chunks) + len(new_chunks)})

        if not new_chunks: return
        new_embs = GeminiEmbedder().encode(new_chunks)
        faiss_idx = faiss.read_index(str(artifacts_dir / f"{kwargs['index_prefix']}.faiss"))
        faiss_idx.add(new_embs)
        faiss.write_index(faiss_idx, str(artifacts_dir / f"{kwargs['index_prefix']}.faiss"))

        all_chunks = existing_chunks + new_chunks
        with open(artifacts_dir / f"{kwargs['index_prefix']}_chunks.pkl", "wb") as f: pickle.dump(all_chunks, f)
        index_info["chapters"] = sorted(list(set(index_info.get("chapters", []) + new_chaps)))
        with open(artifacts_dir / f"{kwargs['index_prefix']}_info.json", "w") as f: json.dump(index_info, f, indent=2)

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
        with open(extracted_index_path, 'r') as f: extracted_index = json.load(f)
        relevant_pages = set()
        for word in keywords:
            if word in extracted_index: relevant_pages.update(extracted_index[word])
        
        if not relevant_pages: return []
        page_to_chapter = self.get_page_to_chapter_map(markdown_file)
        target_chapters = {page_to_chapter[str(p)] for p in relevant_pages if str(p) in page_to_chapter}
        
        artifacts_dir = pathlib.Path(artifacts_dir)
        info_path = artifacts_dir / f"{index_prefix}_info.json"
        existing_chapters = []
        if info_path.exists():
            with open(info_path, 'r') as f: existing_chapters = json.load(f).get("chapters", [])
        
        chapters_to_add = sorted(list(target_chapters - set(existing_chapters)))
        if not chapters_to_add: return []
            
        print(f"Cloud JIT Indexing chapters: {chapters_to_add}")
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
