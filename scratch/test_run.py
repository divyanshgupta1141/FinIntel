import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to sys.path so we can import workspace modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, get_session
from ingest import ingest_document, get_gemini_client, get_embedding_async
from retrieval import hybrid_search
from cache import cache
from workflow import workflow_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_run")

async def main():
    # Verify GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is missing! Please set it to run this test.")
        return

    logger.info("--- Step 1: Database Setup and Mock Ingestion ---")
    # This automatically creates 'finintel' database and document_chunks table with indexes
    await ingest_document()

    logger.info("--- Step 2: Query Generation & Vector Embedding ---")
    query = "What is Apple's total revenue for FY2025 and what are the main risk factors?"
    client = get_gemini_client()
    query_embedding = await get_embedding_async(client, query)
    logger.info(f"Generated query embedding: {len(query_embedding)} dimensions.")

    logger.info("--- Step 3: Database Hybrid Search Verification ---")
    async for session in get_session():
        chunks = await hybrid_search(session, query, query_embedding)
        logger.info(f"Retrieved {len(chunks)} chunks via Hybrid RRF Search:")
        for i, chunk in enumerate(chunks):
            logger.info(f"  [{i+1}] Doc: {chunk['document_name']}, Page: {chunk['page_number']}, RRF: {chunk['rrf_score']:.4f}")
            logger.info(f"      Text excerpt: {chunk['chunk_text'][:120]}...")

    logger.info("--- Step 4: Redis Semantic Cache Verification ---")
    await cache.connect()
    
    # 4a. Initial Cache Check (Should Miss)
    logger.info("Checking semantic cache (expecting MISS)...")
    cached_val = await cache.get(query, query_embedding)
    logger.info(f"Initial Cache Result: {cached_val}")

    logger.info("--- Step 5: LangGraph Pipeline Orchestration ---")
    state = {
        "query": query,
        "query_embedding": query_embedding,
        "chunks": [],
        "analysis": None
    }
    
    # Invoke LangGraph workflow
    final_state = await workflow_app.ainvoke(state)
    analysis = final_state.get("analysis")
    
    logger.info("LangGraph execution completed. Structured Output:")
    if analysis:
        logger.info(f"  Key Metric Summary: {analysis.key_metric_summary}")
        logger.info(f"  Impact Score: {analysis.financial_impact_score}")
        logger.info(f"  Risk Factors: {analysis.risk_factors}")
        logger.info(f"  Impact Assessment: {analysis.impact_assessment}")
        logger.info("  Citations:")
        for cit in analysis.source_citations:
            logger.info(f"    - Doc: {cit.document_name}, Page: {cit.page_number}")
            logger.info(f"      Verbatim Excerpt: '{cit.excerpt}'")
            
        # 4b. Write to Cache
        logger.info("Writing result to semantic cache...")
        await cache.set(query, query_embedding, analysis.model_dump())
        
        # 4c. Second Cache Check (Should Hit)
        logger.info("Checking semantic cache again (expecting HIT)...")
        cached_val = await cache.get(query, query_embedding)
        logger.info(f"Second Cache Result (HIT): {cached_val is not None}")
        if cached_val:
            logger.info(f"  Cached Key Metric Summary: {cached_val.get('key_metric_summary')}")
    else:
        logger.error("LangGraph pipeline returned empty analysis.")

    await cache.close()
    logger.info("Integration test run finished successfully!")

if __name__ == "__main__":
    asyncio.run(main())
