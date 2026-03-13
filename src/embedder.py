import sqlite3
import hashlib
import multiprocessing
import multiprocessing.pool
import numpy as np
from pathlib import Path
import os
from typing import List, Union, Optional
from llama_cpp import Llama
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Global variables for worker processes
_worker_model: Optional[Llama] = None
_worker_embedding_dim: int = 0

def _init_worker(model_path: str, n_ctx: int, n_threads: int):
# ... (rest of methods until SentenceTransformer)
    """
    Initializes the model inside a worker process.
    """
    global _worker_model, _worker_embedding_dim

    _worker_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False,
        use_mmap=True # Allows OS to share model weights across processes
    )
    
    # Cache dimension
    test_emb = _worker_model.create_embedding("test")['data'][0]['embedding']
    _worker_embedding_dim = len(test_emb)

def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of text using the worker's local model instance.
    """
    global _worker_model, _worker_embedding_dim
    if _worker_model is None:
        return []
        
    embeddings = []
    for text in texts:
        try:
            # Create embedding
            emb = _worker_model.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        except Exception:
            # Return zero vector on failure
            embeddings.append([0.0] * _worker_embedding_dim)
            
    return embeddings

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        """
        Initialize with a local GGUF model file path.
        
        Args:
            model_path: Path to your local .gguf file
            n_ctx: Context window size (increased to match Qwen3 training context)
            n_threads: Number of threads to use (None = auto-detect)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=True,
            use_mmap=True,
            n_gpu_layers=-1 # use GPU if available
        )
        self._embedding_dimension = None
        
        _ = self.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def encode(self, 
           texts: Union[str, List[str]], 
           batch_size: int = 16,  # Adjusted for 4B model
           normalize: bool = False,
           show_progress_bar: bool = False,
           **kwargs) -> np.ndarray:

        """
        Encode texts to embeddings with batch processing.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
            show_progress_bar: Whether to show progress bar
            Returns:
            numpy.ndarray: Float32 embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        # Process in batches
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            try:
                # IMPORTANT CHANGE: Pass the entire LIST to the model at once.
                # This triggers the native C++/Metal batch processing logic.
                response = self.model.create_embedding(batch_texts)
                
                # Extract the list of embedding vectors from the response
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch: {e}")
                # Fallback: encode one by one if batch fails, or append zeros
                for _ in batch_texts:
                    embeddings.append([0.0] * self.embedding_dimension)
                
        vecs = np.array(embeddings, dtype=np.float32)
        
        if normalize: # do L2 normalization
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
            
        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension

    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        """
        Starts a pool of worker processes.
        """
        if num_workers:
            workers = num_workers
        else:
            # Default to CPU count - 2 (leave room for OS/Main process)
            workers = max(1, multiprocessing.cpu_count() - 2)

        print(f"Creating {workers} worker processes...")
        
        # Use 1 thread per worker to avoid CPU thrashing
        worker_threads = 1
        
        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, worker_threads)
        )
        return pool

    def encode_multi_process(self, texts: List[str], pool: multiprocessing.pool.Pool, batch_size: int = 32) -> np.ndarray:
        """
        Distributes work across the pool.
        """
        # Sort by length to minimize padding/processing waste
        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        # Create batches
        chunks = [sorted_texts[i : i + batch_size] for i in range(0, len(sorted_texts), batch_size)]

        # Process with progress bar
        results = []
        print(f"Dispatching {len(chunks)} batches to pool...")
        for batch_result in tqdm(
            pool.imap(_encode_batch_worker, chunks), 
            total=len(chunks), 
            desc="Parallel Encoding"
        ):
            results.append(batch_result)

        flat_embeddings = [emb for batch in results for emb in batch]

        # Restore original order
        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered_embeddings = [flat_embeddings[i] for i in inverse_indices]
        
        return np.array(ordered_embeddings, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        pool.close()
        pool.join()


class EmbeddingCache:
    """Persistent SQLite cache for embeddings."""
    
    def __init__(self, cache_dir: str = "index/cache"):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT,
                    model_hash TEXT,
                    query_text TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_hash, query_text)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
    
    def get(self, model_path: str, query: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if it exists."""
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                (model_hash, query)
            ).fetchone()
            
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None
    
    def set(self, model_path: str, query: str, embedding: np.ndarray):
        """Store embedding in cache."""
        model_name = Path(model_path).stem
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        blob = embedding.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (model_name, model_hash, query_text, embedding) VALUES (?,?,?,?)",
                (model_name, model_hash, query, blob)
            )


class CachedEmbedder:
    """
    Wrapper around SentenceTransformer that caches query embeddings.
    Drop-in replacement for SentenceTransformer.
    """
    
    def __init__(self, model_path: str, **kwargs):
        self.embedder = SentenceTransformer(model_path, **kwargs)
        self.cache = EmbeddingCache()
        self.model_path = model_path
    
    def encode(self, texts, **kwargs):
        """Encode texts with caching support."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_path, text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)
        
        # Compute missing embeddings
        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                self.cache.set(self.model_path, text, emb)
                results.append((idx, emb))
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])
    
    def __getattr__(self, name):
        """Delegate other methods to wrapped embedder."""
        return getattr(self.embedder, name)

import os
import time
from typing import Union, List
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types

class GeminiEmbedder:
    """
    Embedding model that uses the Google Gemini API.
    """
    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize the Gemini embedding model.
        
        Args:
            model_name: The name of the Gemini embedding model to use.
        """
        self.model_name = model_name
        
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please export GEMINI_API_KEY in your environment.")

        self._client = genai.Client(api_key=self.api_key)
        self._embedding_dimension = None

        print("--- Available Embedding Models ---")
        for m in self._client.models.list():
            if hasattr(m, 'supported_actions') and 'embedContent' in m.supported_actions:
                print(f"Model ID: {m.name}")
                print(f"  > Display Name: {m.display_name}")
                print("-" * 30)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            res = self._client.models.embed_content(
                model=self.model_name,
                contents="test"
            )
            self._embedding_dimension = len(res.embeddings[0].values)
        return self._embedding_dimension

    def encode(self, 
           texts: Union[str, List[str]], 
           batch_size: int = 20,  
           normalize: bool = False,
           show_progress_bar: bool = False,
           max_retries: int = 5,      # Added parameter for retries
           base_delay: float = 2.0,   # Added parameter for backoff delay
           **kwargs) -> np.ndarray:
        """
        Encode texts using the Gemini API with exponential backoff for rate limits.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Cloud Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # --- Retry Logic Loop ---
            for attempt in range(max_retries):
                try:
                    response = self._client.models.embed_content(
                        model=self.model_name,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT" if len(batch_texts) > 1 else "RETRIEVAL_QUERY"
                        )
                    )
                    
                    batch_embeddings = [e.values for e in response.embeddings]
                    embeddings.extend(batch_embeddings)
                    
                    # Optional: A tiny sleep between successful batches to pace out RPM
                    time.sleep(0.5) 
                    break  # Break out of the retry loop on success
                    
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if the error is a rate limit (429) issue
                    if "429" in error_str or "exhausted" in error_str:
                        if attempt < max_retries - 1:
                            sleep_time = base_delay * (2 ** attempt)
                            print(f"\nRate limit hit. Retrying batch {i+1} in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
                            print(f"\nFailed batch {i+1} after {max_retries} retries: {e}")
                            # Fallback: append zeros on ultimate failure
                            for _ in batch_texts:
                                embeddings.append([0.0] * self.embedding_dimension)
                    else:
                        # If it's a different error (e.g., 400 Bad Request), don't retry, just fail gracefully
                        print(f"\nError encoding cloud batch {i+1}: {e}")
                        for _ in batch_texts:
                            embeddings.append([0.0] * self.embedding_dimension)
                        break 
        
        vecs = np.array(embeddings, dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
            
        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension