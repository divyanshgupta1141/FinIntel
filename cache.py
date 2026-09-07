import os
import json
import hashlib
import asyncio
import logging
from typing import Optional, Dict, Any, List
import redis.asyncio as aioredis
import numpy as np

# Redis Search imports
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from database import get_embedding_dimension

logger = logging.getLogger("cache")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INDEX_NAME = "finintel:cache:idx"

class SemanticCache:
    """
    Semantic Caching layer using Redis Stack Vector Search (RediSearch).
    Uses FT.CREATE to set up a vector index and FT.SEARCH with KNN to find similar queries.
    Gracefully degrades and bypasses caching if Redis is unavailable or fails to connect.
    """
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.is_available: bool = False

    async def connect(self):
        """
        Establishes connection to the Redis server and initializes the Vector Search index.
        Fails gracefully without raising exceptions if Redis is down or unreachable.
        """
        if self.is_available and self.redis_client:
            return

        if not REDIS_URL:
            logger.warning("REDIS_URL environment variable is not configured. Bypassing semantic cache.")
            self.redis_client = None
            self.is_available = False
            return

        client = None
        try:
            logger.info(f"Connecting to Redis Stack at {REDIS_URL}...")
            client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3.0,
                socket_timeout=3.0,
            )
            # Verify connectivity with a strict timeout
            await asyncio.wait_for(client.ping(), timeout=3.0)
            logger.info("Connected to Redis successfully.")
            
            self.redis_client = client
            # Verify or create the RediSearch index
            await self._init_index()
            self.is_available = True
            logger.info("Redis semantic cache initialized and ready.")
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis Stack or setup index: {e}. "
                "FastAPI will start up successfully with the semantic cache bypassed."
            )
            if client:
                try:
                    await client.aclose()
                except Exception:
                    pass
            self.redis_client = None
            self.is_available = False

    async def _init_index(self):
        """
        Initializes the RediSearch HNSW vector index if it doesn't exist.
        """
        if not self.redis_client:
            return

        try:
            # Check if index exists
            await self.redis_client.ft(INDEX_NAME).info()
            logger.info(f"RediSearch index '{INDEX_NAME}' already exists.")
        except Exception as info_err:
            err_msg = str(info_err).lower()
            if "unknown command" in err_msg:
                logger.warning(
                    f"Connected Redis instance does not support RediSearch commands ({info_err}). "
                    "Semantic caching requires Redis Stack / RediSearch. Bypassing cache."
                )
                raise info_err

            # Dynamically get embedding dimension
            dim = await get_embedding_dimension()
            logger.info(f"RediSearch index '{INDEX_NAME}' not found. Creating a new HNSW vector index with dimension {dim}...")
            
            # Fields for index
            fields = [
                TextField("query_text"),
                VectorField(
                    "vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dim,          # Dynamic dimension
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                    }
                ),
                TextField("response_json")
            ]
            
            # Create index definition (all hashes starting with finintel:cache:)
            definition = IndexDefinition(
                prefix=["finintel:cache:"],
                index_type=IndexType.HASH
            )
            
            # Execute index creation
            await self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=definition
            )
            logger.info(f"RediSearch HNSW vector index '{INDEX_NAME}' created successfully.")

    async def close(self):
        """
        Closes the Redis connection pool.
        """
        if self.redis_client:
            try:
                await self.redis_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self.redis_client = None
                self.is_available = False
                logger.info("Redis connection closed.")

    async def get(self, query_text: str, query_embedding: List[float], similarity_threshold: float = 0.96) -> Optional[Dict[str, Any]]:
        """
        Performs semantic lookup in the cache using Redis Stack KNN search.
        1. Formulates KNN query.
        2. Executes FT.SEARCH on the index.
        3. Cosine similarity score = 1.0 - Cosine Distance.
           If Cosine Distance is <= (1.0 - threshold), it is a match.
        """
        if not self.is_available or not self.redis_client:
            return None

        try:
            # Convert embedding to float32 binary format
            query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Calculate distance threshold (e.g. similarity >= 0.96 -> distance <= 0.04)
            distance_threshold = round(1.0 - similarity_threshold, 4)
            
            # Formulate query: KNN search returns nearest neighbor
            # We explicitly retrieve only query_text, response_json, and vector_score to avoid
            # retrieving/decoding the raw binary vector field.
            q = (
                Query("*=>[KNN 1 @vector $query_vector AS vector_score]")
                .sort_by("vector_score")
                .return_fields("query_text", "response_json", "vector_score")
                .dialect(2)
            )
            
            logger.info(f"Executing KNN Search on Redis for distance threshold: {distance_threshold}...")
            
            # Run search query inside Redis
            result = await self.redis_client.ft(INDEX_NAME).search(
                q,
                query_params={"query_vector": query_vector_bytes}
            )
            
            if result.total > 0:
                doc = result.docs[0]
                distance = float(doc.vector_score)
                similarity = 1.0 - distance
                
                logger.info(f"Redis Cache Lookup Match: Query: '{doc.query_text}' | Distance: {distance:.4f} (Similarity: {similarity:.4f})")
                
                if distance <= distance_threshold:
                    logger.info(f"Semantic Cache HIT! Cosine similarity {similarity:.4f} >= threshold {similarity_threshold}")
                    return json.loads(doc.response_json)
                else:
                    logger.info(f"Semantic Cache MISS. Similarity {similarity:.4f} < threshold {similarity_threshold}")
            else:
                logger.info("Semantic Cache MISS. No documents returned from index.")
                
            return None

        except Exception as e:
            logger.warning(f"Error checking semantic cache in Redis: {e}. Bypassing cache.")
            return None

    async def set(self, query_text: str, query_embedding: List[float], response_data: Dict[str, Any], ttl: int = 86400):
        """
        Stores the query text, its embedding vector, and the generated response JSON in Redis.
        The vector is written as float32 binary format to support RediSearch vector indexing.
        TTL defaults to 24 hours.
        """
        if not self.is_available or not self.redis_client:
            return

        try:
            # Generate deterministic hash of query text for the key name
            query_hash = hashlib.md5(query_text.encode("utf-8")).hexdigest()
            key = f"finintel:cache:{query_hash}"
            
            # Convert embedding to float32 binary format for HNSW indexing
            query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

            payload = {
                "query_text": query_text,
                "vector": query_vector_bytes,
                "response_json": json.dumps(response_data)
            }

            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=payload)
                pipe.expire(key, ttl)
                await pipe.execute()

            logger.info(f"Stored query in Redis semantic cache with TTL {ttl}s. Key: {key}")
        except Exception as e:
            logger.warning(f"Error saving to Redis semantic cache: {e}")

# Global cache instance
cache = SemanticCache()
