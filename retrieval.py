"""
Hybrid Retrieval Module for FinIntel.

Merges dense semantic vector search (PostgreSQL pgvector) with lexical full-text
search (PostgreSQL tsvector / websearch_to_tsquery) using Reciprocal Rank Fusion (RRF).

Reciprocal Rank Fusion Formula:
    Score = sum(1 / (k + rank))
    Where k = 60 (standard smoothing constant per Cormack et al., SIGIR 2009).

Architecture:
    1. Dense Vector Search: Computes cosine distance (<=>) over 768-dim embeddings
       indexed via HNSW to capture semantic meaning and conceptual similarity.
    2. Lexical Search: Evaluates ts_vector @@ websearch_to_tsquery('english', query)
       ranked via ts_rank over GIN indexes to capture exact financial terms, acronyms,
       and numerical metrics.
    3. Fusion: CTE executes both searches, assigns dense_rank and text_rank over top 20
       candidates, and combines scores via FULL OUTER JOIN into top-k chunks.
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any

logger = logging.getLogger("retrieval")


def reciprocal_rank_fusion(
    dense_ranked_ids: List[str],
    lexical_ranked_ids: List[str],
    k: int = 60
) -> Dict[str, float]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores across dense and lexical result sets.

    Formula:
        Score = sum(1 / (k + rank))

    Where:
        - k is the ranking constant (default 60, per Cormack et al.), which dampens
          the impact of high rankings from either list and prevents single-modality dominance.
        - rank is the 1-based ordinal rank of document d in each retrieval modality.

    Args:
        dense_ranked_ids: List of document IDs ordered by dense pgvector cosine similarity.
        lexical_ranked_ids: List of document IDs ordered by BM25/FTS keyword relevance.
        k: Hyperparameter smoothing constant (default 60).

    Returns:
        Dictionary mapping document IDs to their merged RRF score.
    """
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    for rank, doc_id in enumerate(lexical_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    return scores


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_embedding: List[float]
) -> List[Dict[str, Any]]:
    """
    Executes a hybrid search query in PostgreSQL.
    Combines:
    1. Vector semantic search (using pgvector <=> cosine distance operator)
    2. Full-text search (using ts_vector @@ websearch_to_tsquery)

    Rank fusion is computed in-database using Reciprocal Rank Fusion (RRF):
        Score = sum(1 / (60 + rank))
        RRF_Score = 1 / (60 + vector_rank) + 1 / (60 + text_rank)

    Dense semantic search maps conceptual nuances, while lexical search preserves
    exact numerical disclosures, ticker symbols, and financial terminology.
    Merging them via RRF eliminates score scale discrepancies between cosine distance
    and ts_rank.

    Args:
        session: Async SQLAlchemy session
        query_text: Raw user query string for keyword search
        query_embedding: 768-dimension embedding of the query

    Returns:
        List of dictionaries with document chunk data and RRF scores.
        Hardcoded to return exactly top 3 chunks.
    """
    top_k = 3
    # Convert query embedding to a string format that pgvector understands: '[v1, v2, ...]'
    embedding_str = f"[{','.join(map(str, query_embedding))}]"
    
    # We use websearch_to_tsquery as it is robust against syntax errors
    # and supports quotes for phrase matching and minus signs for exclusion.
    query_sql = text("""
        WITH vector_search AS (
            SELECT 
                id, 
                document_name, 
                page_number, 
                chunk_text,
                ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:embedding_val AS vector)) as rank
            FROM document_chunks
            ORDER BY embedding <=> CAST(:embedding_val AS vector)
            LIMIT 20
        ),
        text_search AS (
            SELECT 
                id, 
                document_name, 
                page_number, 
                chunk_text,
                ROW_NUMBER() OVER (ORDER BY ts_rank(ts_vector, websearch_to_tsquery('english', :query_str)) DESC) as rank
            FROM document_chunks
            WHERE ts_vector @@ websearch_to_tsquery('english', :query_str)
            ORDER BY ts_rank(ts_vector, websearch_to_tsquery('english', :query_str)) DESC
            LIMIT 20
        )
        SELECT 
            COALESCE(v.id, t.id) as id,
            COALESCE(v.document_name, t.document_name) as document_name,
            COALESCE(v.page_number, t.page_number) as page_number,
            COALESCE(v.chunk_text, t.chunk_text) as chunk_text,
            (COALESCE(1.0 / (60.0 + v.rank), 0.0) + COALESCE(1.0 / (60.0 + t.rank), 0.0)) as rrf_score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
        ORDER BY rrf_score DESC
        LIMIT :top_k;
    """)
    
    try:
        result = await session.execute(
            query_sql,
            {
                "embedding_val": embedding_str,
                "query_str": query_text,
                "top_k": top_k
            }
        )
        
        chunks = []
        for row in result.fetchall():
            chunks.append({
                "id": str(row.id),
                "document_name": row.document_name,
                "page_number": row.page_number,
                "chunk_text": row.chunk_text,
                "rrf_score": float(row.rrf_score)
            })
            
        logger.info(f"Hybrid search returned {len(chunks)} results for query: '{query_text}'")
        return chunks
        
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        # Fallback: if FTS or HNSW fails (e.g. extension or indexes not configured properly),
        # perform a simple semantic vector-only query.
        logger.info("Attempting vector-only fallback search...")
        try:
            fallback_sql = text("""
                SELECT id, document_name, page_number, chunk_text,
                       (1.0 / (1.0 + (embedding <=> CAST(:embedding_val AS vector)))) as rrf_score
                FROM document_chunks
                ORDER BY embedding <=> CAST(:embedding_val AS vector)
                LIMIT :top_k;
            """)
            result = await session.execute(
                fallback_sql,
                {
                    "embedding_val": embedding_str,
                    "top_k": top_k
                }
            )
            chunks = []
            import math
            for row in result.fetchall():
                score = row.rrf_score
                if score is None or math.isnan(score):
                    score = 0.0
                chunks.append({
                    "id": str(row.id),
                    "document_name": row.document_name,
                    "page_number": row.page_number,
                    "chunk_text": row.chunk_text,
                    "rrf_score": float(score)
                })
            return chunks
        except Exception as fallback_e:
            logger.error(f"Fallback search also failed: {fallback_e}")
            return []
