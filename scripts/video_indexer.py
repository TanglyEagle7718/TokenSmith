#!/usr/bin/env python3
"""
video_indexer.py
Processes videos, transcribes them, extracts frames, and adds embeddings to the FAISS index.
"""

import os
import cv2
import pickle
import pathlib
import json
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Optional
from llama_cpp import Llama

# Assuming Whisper or similar for transcription if not using a multimodal Qwen model for it.
# For this script, we'll assume a generic interface for the multimodal models.

from src.index_builder import preprocess_for_bm25
from rank_bm25 import BM25Okapi

from src.preprocessing.chunking import DocumentChunker, SectionRecursiveStrategy, SectionRecursiveConfig

class VideoIndexer:
    def __init__(
        self,
        text_embedding_model_path: str,
        multimodal_model_path: str,
        artifacts_dir: str = "index/partial_sections",
        index_prefix: str = "textbook_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.artifacts_dir = pathlib.Path(artifacts_dir)
        self.index_prefix = index_prefix
        
        # Initialize Chunker
        config = SectionRecursiveConfig(recursive_chunk_size=chunk_size, recursive_overlap=chunk_overlap)
        strategy = SectionRecursiveStrategy(config)
        self.chunker = DocumentChunker(strategy=strategy)

        # Initialize models
        print(f"Loading text embedding model from {text_embedding_model_path}...")
        self.text_embedder = Llama(
            model_path=text_embedding_model_path,
            embedding=True,
            verbose=False,
            n_ctx=4096,
            n_gpu_layers=-1
        )
        
        print(f"Loading multimodal model from {multimodal_model_path}...")
        # Placeholder for multimodal model loading (e.g. Qwen3-VL)
        self.multimodal_model = None 

    def extract_transcript(self, video_path: str) -> str:
        """Transcribes the video audio."""
        print(f"Transcribing {video_path}...")
        # Placeholder for transcription logic (e.g. whisper)
        return f"This is a placeholder transcript for {os.path.basename(video_path)}."

    def extract_frames(self, video_path: str, interval_sec: int = 5) -> List[np.ndarray]:
        """Extracts frames from the video at regular intervals."""
        print(f"Extracting frames from {video_path} every {interval_sec} seconds...")
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30 # Fallback
        interval_frames = int(fps * interval_sec)
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval_frames == 0:
                frames.append(frame)
            count += 1
        cap.release()
        return frames

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of text strings."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 4096)
        embeddings = []
        for text in tqdm(texts, desc="Embedding text"):
            emb = self.text_embedder.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Embeds a list of video frames."""
        if not frames:
            return np.array([], dtype=np.float32).reshape(0, 4096)
        print(f"Embedding {len(frames)} frames...")
        dim = 4096 
        return np.random.rand(len(frames), dim).astype(np.float32)

    def update_index(
        self, 
        new_embeddings: np.ndarray, 
        new_chunks: List[str], 
        new_metadata: List[Dict],
        new_sources: List[str]
    ):
        """Updates the existing FAISS, BM25, and other artifacts."""
        faiss_index_path = self.artifacts_dir / f"{self.index_prefix}.faiss"
        bm25_path = self.artifacts_dir / f"{self.index_prefix}_bm25.pkl"
        chunks_path = self.artifacts_dir / f"{self.index_prefix}_chunks.pkl"
        sources_path = self.artifacts_dir / f"{self.index_prefix}_sources.pkl"
        meta_path = self.artifacts_dir / f"{self.index_prefix}_meta.pkl"

        # 1. Update FAISS Index
        if faiss_index_path.exists():
            index = faiss.read_index(str(faiss_index_path))
            index.add(new_embeddings)
            faiss.write_index(index, str(faiss_index_path))
            print(f"Updated FAISS index at {faiss_index_path}")
        else:
            dim = new_embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(new_embeddings)
            faiss.write_index(index, str(faiss_index_path))
            print(f"Created new FAISS index at {faiss_index_path}")

        # 2. Update Chunks, Sources, and Metadata
        if chunks_path.exists():
            with open(chunks_path, "rb") as f: existing_chunks = pickle.load(f)
            all_chunks = existing_chunks + new_chunks
        else:
            all_chunks = new_chunks
        
        if sources_path.exists():
            with open(sources_path, "rb") as f: existing_sources = pickle.load(f)
            all_sources = existing_sources + new_sources
        else:
            all_sources = new_sources

        if meta_path.exists():
            with open(meta_path, "rb") as f: existing_meta = pickle.load(f)
            all_meta = existing_meta + new_metadata
        else:
            all_meta = new_metadata

        with open(chunks_path, "wb") as f: pickle.dump(all_chunks, f)
        with open(sources_path, "wb") as f: pickle.dump(all_sources, f)
        with open(meta_path, "wb") as f: pickle.dump(all_meta, f)

        # 3. Update BM25 Index
        print("Updating BM25 index...")
        tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        
        print("Successfully updated all index artifacts.")

    def process_videos(self, video_dir: str):
        video_dir_path = pathlib.Path(video_dir)
        video_files = list(video_dir_path.glob("*.mp4"))
        
        all_new_embeddings = []
        all_new_chunks = []
        all_new_metadata = []
        all_new_sources = []

        for video_file in video_files:
            video_path = str(video_file)
            filename = video_file.name
            
            # 1. Transcription
            transcript = self.extract_transcript(video_path)
            
            # 2. Chunk Transcript using DocumentChunker
            transcript_chunks = self.chunker.chunk(transcript)
            
            # 3. Embed Transcript Chunks
            if transcript_chunks:
                text_embeddings = self.embed_text(transcript_chunks)
                all_new_embeddings.append(text_embeddings)
                
                for i, chunk in enumerate(transcript_chunks):
                    all_new_chunks.append(f"Video Transcript ({filename}): {chunk}")
                    all_new_sources.append(video_path)
                    all_new_metadata.append({
                        "filename": filename,
                        "type": "video_transcript",
                        "chunk_id": i,
                        "text_preview": chunk[:100]
                    })

            # 4. Extract and Embed Frames
            frames = self.extract_frames(video_path)
            if frames:
                frame_embeddings = self.embed_frames(frames)
                all_new_embeddings.append(frame_embeddings)
                
                for i, _ in enumerate(frames):
                    all_new_chunks.append(f"Video Frame ({filename}) at {i*5}s")
                    all_new_sources.append(video_path)
                    all_new_metadata.append({
                        "filename": filename,
                        "type": "video_frame",
                        "timestamp_sec": i * 5,
                        "chunk_id": len(transcript_chunks) + i
                    })

        if all_new_embeddings:
            final_embeddings = np.vstack(all_new_embeddings)
            self.update_index(
                final_embeddings,
                all_new_chunks,
                all_new_metadata,
                all_new_sources
            )

def main():
    # Paths for the server (as requested, assuming they exist there)
    TEXT_MODEL = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    MM_MODEL = "models/qwen3_embedding/qwen3_vl_model" # Placeholder
    VIDEO_DIR = "data/videos"
    
    indexer = VideoIndexer(TEXT_MODEL, MM_MODEL)
    indexer.process_videos(VIDEO_DIR)

if __name__ == "__main__":
    main()
