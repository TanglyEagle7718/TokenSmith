import os
import cv2
import pickle
import pathlib
import json
import numpy as np
import faiss
import time
import whisper
import warnings
import base64
from tqdm import tqdm
from typing import List, Dict, Optional
from llama_cpp import Llama
import torch
import re

from src.config import RAGConfig
from rank_bm25 import BM25Okapi
from src.preprocessing.chunking import DocumentChunker, SectionRecursiveStrategy, SectionRecursiveConfig

# pulls heavily from index_builder, index_updater, and embedder

def preprocess_for_bm25(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    tokens = text.split()
    return tokens

class VideoIndexer:
    def __init__(
        self,
        config: RAGConfig,
        artifacts_dir: pathlib.Path,
        index_prefix: str,
    ):
        self.artifacts_dir = artifacts_dir
        self.index_prefix = index_prefix
        
        chunk_config = SectionRecursiveConfig(recursive_chunk_size=config.chunk_size, recursive_overlap=config.chunk_overlap)
        strategy = SectionRecursiveStrategy(chunk_config)
        self.chunker = DocumentChunker(strategy=strategy)

        # https://github.com/openai/whisper & 
        whisper_model_name = getattr(config, "whisper_model", "base")
        print(f"Loading Whisper model '{whisper_model_name}'...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.whisper_model = whisper.load_model(whisper_model_name)

        video_model_path = "models/Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf"
        
        start_time = time.time()
        print(f"Loading GGUF model from {video_model_path}...")
        if not os.path.exists(video_model_path):
            raise FileNotFoundError(f"Model file not found: {video_model_path}")
            
        self.model = Llama(
            model_path=video_model_path,
            embedding=True,
            logits_all=True,
            clip_model_path=video_model_path,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False
        )
        print(f"Model loaded in {time.time() - start_time:.2f}s")

    def extract_transcript(self, video_path: str) -> List[Dict]:
        print(f"Transcribing {video_path}...")
        start_time = time.time()
        result = self.whisper_model.transcribe(video_path, fp16=torch.cuda.is_available())
        print(f"Transcribed in {time.time() - start_time:.2f}s")
        return result["segments"]

    def extract_frames(self, video_path: str, interval_sec: int = 5) -> List[np.ndarray]:
        start_time = time.time()
        images = []
        # https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
        #https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return []
            
        duration_sec = total_frames / fps
        for sec in range(0, int(duration_sec), interval_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret: break
            images.append(frame)
            
        cap.release()
        print(f"Extracted {len(images)} frames in {time.time() - start_time:.2f}s")
        return images

    def embed_text(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([], dtype=np.float32)
        start_time = time.time()
        
        embeddings = []
        for text in tqdm(texts, desc="Embedding Chunks", leave=False):
            try:
                response = self.model.create_embedding(text)
                emb_list = response['data'][0]['embedding']
                
                emb_array = np.array(emb_list, dtype=np.float32)
                if emb_array.ndim == 2:
                    emb = emb_array.mean(axis=0)
                else:
                    emb = emb_array
                
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Failed to embed chunk: {e}")
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
        
        if not embeddings:
            return np.array([], dtype=np.float32)
            
        res = np.array(embeddings, dtype=np.float32)
        print(f"Embedded {len(res)}/{len(texts)} text chunks in {time.time() - start_time:.2f}s")
        return res

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Generating real embeddings for frames using multimodal Qwen3-VL."""
        if not frames: return np.array([], dtype=np.float32)
        start_time = time.time()
        
        embeddings = []
        for frame in tqdm(frames, desc="Embedding Frames", leave=False):
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            },
                            {"type": "text", "text": "Represent this image for retrieval;"}
                        ]
                    }
                ]
                self.model.create_chat_completion(messages=messages, max_tokens=1)
                
                emb_list = self.model.embed("")
                emb = np.array(emb_list, dtype=np.float32).flatten()
                
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Failed to embed frame: {e}")
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
        
        if not embeddings:
            return np.array([], dtype=np.float32)
            
        res = np.array(embeddings, dtype=np.float32)
        print(f"Embedded {len(res)}/{len(frames)} frames in {time.time() - start_time:.2f}s")
        return res

    def update_index(self, new_embeddings, new_chunks, new_metadata, new_sources):
        start_time = time.time()
        faiss_path = self.artifacts_dir / f"{self.index_prefix}.faiss"
        bm25_path = self.artifacts_dir / f"{self.index_prefix}_bm25.pkl"
        chunks_path = self.artifacts_dir / f"{self.index_prefix}_chunks.pkl"
        sources_path = self.artifacts_dir / f"{self.index_prefix}_sources.pkl"
        meta_path = self.artifacts_dir / f"{self.index_prefix}_meta.pkl"

        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            if new_embeddings.shape[1] != index.d:
                print(f"Warning: Embedding dimension mismatch ({new_embeddings.shape[1]} vs {index.d}). Re-building index.")
                index = faiss.IndexFlatL2(new_embeddings.shape[1])
            index.add(new_embeddings)
        else:
            index = faiss.IndexFlatL2(new_embeddings.shape[1])
            index.add(new_embeddings)
        faiss.write_index(index, str(faiss_path))

        def load_pkl(p): return pickle.load(open(p, "rb")) if p.exists() else []
        all_chunks = load_pkl(chunks_path) + new_chunks
        all_sources = load_pkl(sources_path) + new_sources
        all_meta = load_pkl(meta_path) + new_metadata

        pickle.dump(all_chunks, open(chunks_path, "wb"))
        pickle.dump(all_sources, open(sources_path, "wb"))
        pickle.dump(all_meta, open(meta_path, "wb"))

        print(f"Re-building BM25 for {len(all_chunks)} total entries...")
        tokenized = [preprocess_for_bm25(c) for c in all_chunks]
        pickle.dump(BM25Okapi(tokenized), open(bm25_path, "wb"))
        print(f"Index updated in {time.time() - start_time:.2f}s")

    def process_videos(self, video_dir: str):
        if not os.path.exists(video_dir):
            print(f"Video directory {video_dir} not found. Skipping.")
            return

        meta_path = self.artifacts_dir / f"{self.index_prefix}_meta.pkl"
        existing_filenames = set()
        
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                existing_meta = pickle.load(f)
                existing_filenames = {m.get("filename") for m in existing_meta if "filename" in m}

        v_files = list(pathlib.Path(video_dir).glob("*.mp4"))
        new_files = [f for f in v_files if f.name not in existing_filenames]
        
        if not new_files:
            print(f"No new videos found in {video_dir}. Index is up to date.")
            return

        print(f"Found {len(new_files)} new videos to process (skipping {len(v_files) - len(new_files)} already indexed).")
        
        total_embeddings, total_chunks, total_meta, total_sources = [], [], [], []
        
        for v_file in new_files:
            print(f"\n--- Processing: {v_file.name} ---")
            
            title_text = f"Video Title: {v_file.name}"
            title_emb = self.embed_text([title_text])
            if title_emb.size > 0:
                total_embeddings.append(title_emb)
                total_chunks.append(title_text)
                total_sources.append(str(v_file))
                total_meta.append({
                    "filename": v_file.name,
                    "type": "video_title",
                    "timestamp_sec": 0
                })

            segments = self.extract_transcript(str(v_file))
            if segments:
                group_size = 3
                for i in range(0, len(segments), group_size):
                    group = segments[i:i + group_size]
                    combined_text = " ".join([s["text"].strip() for s in group])
                    start_time = int(group[0]["start"])
                    
                    chunk_text = f"Transcript ({v_file.name}) at {start_time}s: {combined_text}"
                    embs = self.embed_text([chunk_text])
                    
                    if embs.size > 0:
                        total_embeddings.append(embs)
                        total_chunks.append(chunk_text)
                        total_sources.append(str(v_file))
                        total_meta.append({
                            "filename": v_file.name, 
                            "type": "video_transcript", 
                            "timestamp_sec": start_time
                        })

            frames = self.extract_frames(str(v_file))
            if frames:
                f_embs = self.embed_frames(frames)
                if f_embs.size > 0:
                    total_embeddings.append(f_embs)
                    for i in range(len(frames)):
                        timestamp = i * 5
                        total_chunks.append(f"Video Frame ({v_file.name}) at {timestamp}s")
                        total_sources.append(str(v_file))
                        total_meta.append({
                            "filename": v_file.name, 
                            "type": "video_frame", 
                            "timestamp_sec": timestamp
                        })

        if total_embeddings:
            self.update_index(np.vstack(total_embeddings), total_chunks, total_meta, total_sources)
