import os
import json
import hashlib
import logging
from typing import Optional, List, Dict
import asyncio

from config import LOCAL_CACHE_PATH

logger = logging.getLogger("embedding_cache")

class EmbeddingCache:
    """
    A local file-based cache for storing generated text embeddings.
    Uses MD5 hashes of chunk texts as lookup keys to prevent recomputing embeddings.
    """
    def __init__(self, cache_file_path: str = LOCAL_CACHE_PATH):
        self.cache_file_path = cache_file_path
        self.cache: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()
        self._load_cache()

    def _load_cache(self):
        """Loads cached embeddings from disk."""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from local storage.")
            except Exception as e:
                logger.warning(f"Failed to read local embedding cache file: {e}. Starting fresh.")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        """Saves current cache dictionary to disk."""
        try:
            # Ensure containing directory exists
            dir_name = os.path.dirname(self.cache_file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            with open(self.cache_file_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to persist embedding cache to disk: {e}")

    def _get_hash(self, text: str) -> str:
        """Generates MD5 hash for a given text."""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    async def get(self, text: str) -> Optional[List[float]]:
        """
        Looks up the embedding for a given text from the cache.
        """
        text_hash = self._get_hash(text)
        async with self.lock:
            val = self.cache.get(text_hash)
            if val is not None:
                return val
        return None

    async def set(self, text: str, embedding: List[float]):
        """
        Saves the embedding for a given text to the local cache and persists to disk.
        """
        text_hash = self._get_hash(text)
        async with self.lock:
            self.cache[text_hash] = embedding
            # Run blocking write in executor or directly (since cache writes are relatively small and fast)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_cache)

# Global embedding cache instance
global_embedding_cache = EmbeddingCache()
