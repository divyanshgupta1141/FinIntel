import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Centralized configuration & structured logging setup
import config
import logging

logger = logging.getLogger("main")

from cache import cache
from ingest import get_gemini_client, get_embedding_async
from workflow import workflow_app, FinancialReportAnalysis

# ---------------------------------------------------------
# FastAPI Lifespan (for Redis Connection Management)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Establish connection to Redis on startup
    await cache.connect()
    if cache.redis_client:
        logger.info("Clearing Redis Cache on startup to purge poisoned cache records...")
        try:
            await cache.redis_client.flushall()
            logger.info("Redis cache flushed successfully.")
        except Exception as e:
            logger.error(f"Failed to flush Redis cache: {e}")
    yield
    # Clean up Redis connection on shutdown
    await cache.close()

app = FastAPI(
    title="FinIntel: Financial Document Intelligence Platform",
    description="Enterprise-grade token-optimized document intelligence platform leveraging FastAPI, pgvector, Redis, LangGraph, and PydanticAI.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Redis Sliding-Window Rate Limiter (Issue 8)
# ---------------------------------------------------------
LIMIT_REQUESTS = 10     # Max requests per client
LIMIT_WINDOW = 60       # Window size in seconds (1 minute)

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """
    HTTP middleware that enforces rate limiting on all API routes using Redis sorted sets.
    Implements a sliding-window counter and returns HTTP 429 if the limit is exceeded.
    Fails open to ensure resilience in case Redis suffers connection issues.
    """
    if request.url.path.startswith("/api/v1/"):
        client_ip = request.client.host if request.client else "unknown"
        rate_limit_key = f"rate_limit:{client_ip}"
        
        if cache.redis_client:
            try:
                now = time.time()
                clear_before = now - LIMIT_WINDOW
                
                async with cache.redis_client.pipeline(transaction=True) as pipe:
                    # 1. Clear timestamps older than the sliding window boundary
                    pipe.zremrangebyscore(rate_limit_key, 0, clear_before)
                    # 2. Add current request timestamp
                    pipe.zadd(rate_limit_key, {str(now): now})
                    # 3. Retrieve request count within current window
                    pipe.zcard(rate_limit_key)
                    # 4. Set key expiration to clear memory automatically
                    pipe.expire(rate_limit_key, LIMIT_WINDOW + 5)
                    
                    # Execute atomic pipeline
                    _, _, request_count, _ = await pipe.execute()
                    
                if request_count > LIMIT_REQUESTS:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip} ({request_count} requests in {LIMIT_WINDOW}s)")
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"Too many requests. Rate limit exceeded ({LIMIT_REQUESTS} requests per minute)."}
                    )
            except Exception as e:
                # Log error and fail open to protect API availability
                logger.error(f"Rate limiting evaluation failed: {e}. Failing open...")
                
    return await call_next(request)

# ---------------------------------------------------------
# Global Exception Handlers (Issue 8)
# ---------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Exception handler for standard HTTPExceptions.
    """
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions, logs the error stack trace, 
    and returns a clean JSON error response.
    """
    logger.error(f"Unhandled Exception caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again later."}
    )

# ---------------------------------------------------------
# Request/Response Schemas
# ---------------------------------------------------------
class AnalysisRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The financial question or analysis instruction regarding the corporate report.",
        examples=["What are Apple's main supply chain risks for fiscal year 2025?"]
    )

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API operation status.
    """
    return {"status": "healthy", "service": "FinIntel"}

@app.post("/api/v1/analyze", response_model=FinancialReportAnalysis)
async def analyze_document(request: AnalysisRequest):
    """
    Endpoint to analyze ingested financial documents based on a user query.
    
    Workflow:
    1. Generate embedding for query text.
    2. Check Redis Semantic Cache (> 0.96 cosine similarity match).
       - Cache Hit: Bypasses LLM and DB, returning the cached JSON directly (0-token cost).
    3. Cache Miss: Run LangGraph pipeline:
       - Retrieve: Query postgres using RRF Hybrid Search (pgvector + FTS).
       - Generate: Call Gemini API using native google-genai response_schema (0-agent loop).
    4. Save the generated analysis JSON to the Redis Semantic Cache mapped against the query embedding.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # Get the Gemini client to generate query embeddings
        client = get_gemini_client()
        logger.info(f"Generating query embedding for: '{query}'")
        query_embedding = await get_embedding_async(client, query)
    except ValueError as val_err:
        logger.error(f"Configuration error: {val_err}")
        raise HTTPException(status_code=500, detail=str(val_err))
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to call embedding service.")

    # 1. Intercept with Redis Semantic Caching
    try:
        cached_result = await cache.get(query, query_embedding)
        if cached_result:
            logger.info("Serving response directly from Redis Semantic Cache (0-token cost).")
            return FinancialReportAnalysis(**cached_result)
    except Exception as e:
        logger.warning(f"Error checking cache: {e}. Proceeding to pipeline.")

    # 2. Cache Miss: Execute LangGraph Workflow
    logger.info("Cache Miss: Executing LangGraph workflow pipeline...")
    state = {
        "query": query,
        "query_embedding": query_embedding,
        "chunks": [],
        "analysis": None,
        "error": None
    }
    
    try:
        final_state = await workflow_app.ainvoke(state)
        analysis_result = final_state.get("analysis")
        error_occurred = final_state.get("error")
        
        if not analysis_result:
            raise HTTPException(status_code=500, detail="LangGraph pipeline failed to generate an analysis.")
            
        # 3. Store result in semantic cache for future requests ONLY if no error occurred
        if not error_occurred:
            try:
                await cache.set(query, query_embedding, analysis_result.model_dump())
            except Exception as cache_err:
                logger.warning(f"Failed to save result to semantic cache: {cache_err}")
        else:
            logger.info("Generation failed; skipping semantic caching of error response.")
            
        return analysis_result
        
    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use uvicorn to run the server locally
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
