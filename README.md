# FinIntel: Financial Document Intelligence Platform

FinIntel is an enterprise-grade, token-optimized financial document intelligence platform designed to ingest, retrieve, and analyze complex corporate disclosures. It leverages **FastAPI** for HTTP endpoints, **PostgreSQL with pgvector** for hybrid RRF retrieval, **Redis Stack** for high-efficiency semantic caching, and **LangGraph with Groq (Llama 3.1)** for native structured JSON analysis.

---

## System Architecture Flow

The following diagram maps out the data path of a user inquiry:

```text
                  +------------------+
                  |  Client Query    |
                  +--------+---------+
                           |
                           v
                  +--------+---------+
                  |    FastAPI       |
                  |  (main:analyze)  |
                  +--------+---------+
                           |
                           | 1. Query Embedding
                           v
             +-------------+-------------+
             |  Gemini embedding-001 API |
             +-------------+-------------+
                           |
                           | 2. Search Vectors
                           v
           +---------------+---------------+
           |    Redis Semantic Cache       |
           |      (RediSearch HNSW)        |
           +---------------+---------------+
                           |
            +--------------+--------------+
            |                             |
      (Cache Hit)                   (Cache Miss)
            |                             |
            v                             v
  +---------+---------+         +---------+---------+
  | Return Cached JSON|         | Hybrid Search (DB)|
  |  (0-token cost)   |         | pgvector + FTS RRF|
  +-------------------+         +---------+---------+
                                          |
                                          | 3. Retrieve Top 3
                                          v
                                +---------+---------+
                                | Context Assembly  |
                                |  (max 3000 chars) |
                                +---------+---------+
                                          |
                                          | 4. Groq Inference
                                          v
                                +---------+---------+
                                |  Groq Llama 3.1   |
                                |  (llama-3.1-8b)   |
                                +---------+---------+
                                          |
                                          | 5. Raw JSON Output
                                          v
                                +---------+---------+
                                | Pydantic Schema   |
                                |   Validation      |
                                +---------+---------+
                                          |
                                          | 6. Save cache &
                                          |    Return result
                                          v
```

---

## Key Technical Features

### 1. Zero-Token Redis Semantic Caching
- Integrates **RediSearch** vector indexing using an HNSW configuration.
- Measures similarity between the query embedding and previously cached queries using **Cosine Distance**.
- Any query with a similarity match $> 0.96$ skips database calls and LLM generation entirely, returning the structured JSON directly from Redis.
- Bypasses caching dynamically if the execution pipeline experiences any runtime or API errors.

### 2. Hybrid Retrieval with pgvector & Full-Text Search (FTS)
- Combines semantic vector similarity search (using `pgvector` HNSW indexes) and keyword-based Full-Text Search (using PostgreSQL GIN indexes on `tsvector` columns).
- Blends scores using Reciprocal Rank Fusion (RRF) to prioritize context chunks that have both high semantic similarity and exact keyword matches.
- Strictly bounds chunk sizes and retrievable limits (exactly top 3 chunks, truncated to 3,000 characters) to optimize the LLM's context window.

### 3. Deterministic Structured JSON Output
- Swaps out complex model wrappers for native **Groq API JSON Mode** (`response_format={"type": "json_object"}`).
- Automatically validates the JSON schema generated from the Pydantic `FinancialReportAnalysis` model at the inference layer.

### 4. Resilient Free-Tier Rate Limiting
- Implements an async-safe **Token-Bucket Rate Limiter** configured to enforce an RPM ceiling (e.g. 5 requests/minute).
- Handles transient network errors, Google API rate limits (`429`), and Groq spikes in demand (`503 Service Unavailable`) using exponential backoff with random jitter.
- Distinguishes between minute-level rate limits (which are retried) and true daily quota limits (which exit gracefully).

---

## Local Quickstart Guide

### 1. Set Up Environment Configuration
Clone the repository and copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and fill in your API credentials:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Start the Docker Compose Stack
Launch the database, semantic cache, and web application services:
```bash
docker compose up --build -d
```
The services will initialize in order:
- `db`: PostgreSQL container initialized with the `pgvector` extension.
- `cache`: Redis Stack container providing vector search features.
- `web`: FastAPI application container. On startup, it automatically provisions tables, tsvector mappings, HNSW search indices, and seeds mock financial data.

### 3. Access Swagger API Documentation
Open your browser and navigate to:
```text
http://localhost:8000/docs
```
You can execute test requests directly through the interactive Swagger UI.

---

## Design Rationales & Engineering Trade-offs

### 1. Swapping Gemini for Groq LLM Generation
- **Latency & Throughput**: While the Google Gemini API has massive context windows, its free-tier rate limits (both RPM and daily caps) are highly restrictive for active developer iteration and production workloads. Swapping the LLM generation step to Groq (`llama-3.1-8b-instant`) provides sub-second reasoning and generation latency with generous throughput.
- **Hybrid Embedding Model Setup**: Because Groq does not currently provide text embedding models natively, we retain Gemini (`gemini-embedding-001`) for the vector creation step of our chunks and queries. This dual-provider approach leverages the best strengths of both platforms (Gemini for high-quality embedding metrics, Groq for lightning-fast text completions).

### 2. Choosing Redis Stack over Standard Alpine Redis
- **Search Capabilities**: Standard `redis:7-alpine` does not contain the RediSearch engine required to execute vector search indexes. Using `redis/redis-stack-server` ensures we can leverage native HNSW indexing, Cosine Distance calculations, and K-Nearest Neighbors (KNN) searches directly in Redis memory, maintaining our 0-token overhead constraint on repeated queries.

### 3. Offloading Evaluation Concurrency
- **Ragas Concurrency Limits**: Running comprehensive RAG evaluations can easily trigger `429 Too Many Requests` when executing multiple concurrent LLM calls. We configure our evaluation suite (`eval.py`) with `RunConfig(max_workers=2)` and async pauses to cleanly respect developer-tier quotas.
