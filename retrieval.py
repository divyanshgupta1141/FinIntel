import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any

logger = logging.getLogger("retrieval")

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
    RRF_Score = 1 / (60 + vector_rank) + 1 / (60 + text_rank)
    
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
        for row in result.fetchall():
            chunks.append({
                "id": str(row.id),
                "document_name": row.document_name,
                "page_number": row.page_number,
                "chunk_text": row.chunk_text,
                "rrf_score": float(row.rrf_score)
            })
        return chunks
