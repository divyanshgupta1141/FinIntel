import os
import uuid
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

import asyncio

from config import DATABASE_URL

# Async database engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class DocumentChunk(Base):
    """
    SQLAlchemy model representing a semantic text chunk of a financial document.
    """
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_name = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_text = Column(String, nullable=False)
    
    # Dimension is set dynamically during database init
    embedding = Column(Vector(768), nullable=False)


async def get_embedding_dimension() -> int:
    """
    Calls the Gemini Embedding API for a test text to dynamically inspect 
    the vector size of the configured embedding model. Defaults to 768 if offline/no key.
    """
    from config import EMBEDDING_MODEL
    from utils.rate_limiter import execute_with_retry
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY not found. Defaulting embedding dimension to 768.")
        return 768
        
    try:
        from google import genai
        from google.genai import types
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        def call_embed():
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents="dimension_test",
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            
        response = await execute_with_retry(call_embed)
        dim = len(response.embeddings[0].values)
        logger.info(f"Dynamically detected embedding dimension: {dim}")
        return dim
    except Exception as e:
        logger.warning(f"Failed to dynamically check embedding dimension: {e}. Defaulting to 768.")
        return 768


async def init_db():
    """
    Initializes the database:
    1. Connects to the default 'postgres' database to check if 'finintel' exists, and creates it if missing.
    2. Dynamically detects embedding dimension and updates column definitions.
    3. Attempts to create the vector extension if not present.
    4. Creates the document_chunks table.
    5. Adds a generated ts_vector column for Hybrid full-text search.
    6. Creates an HNSW index on the vector embedding column for fast approximate nearest neighbor search.
    7. Creates a GIN index on the full-text search column.
    """
    # 0. Check and create the finintel database if it doesn't exist
    # Parse the host from DATABASE_URL
    default_url = DATABASE_URL
    # Construct default postgres connection URL by changing database name to postgres
    if "/finintel" in default_url:
        default_url = default_url.replace("/finintel", "/postgres")
    
    logger.info(f"Connecting to default database to check/create 'finintel' db...")
    try:
        # We need AUTOCOMMIT isolation level to run CREATE DATABASE
        temp_engine = create_async_engine(default_url, isolation_level="AUTOCOMMIT")
        async with temp_engine.connect() as conn:
            db_exists = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'finintel'")
            )
            if not db_exists.scalar():
                logger.info("Database 'finintel' does not exist. Creating database...")
                await conn.execute(text("CREATE DATABASE finintel;"))
            else:
                logger.info("Database 'finintel' already exists.")
        await temp_engine.dispose()
    except Exception as e:
        logger.warning(f"Could not verify/create 'finintel' database via 'postgres' default db: {e}. Proceeding directly...")

    # 0b. Dynamically detect and update embedding column dimensions before table creation
    dim = await get_embedding_dimension()
    DocumentChunk.embedding.type.dim = dim
    logger.info(f"SQLAlchemy: Configured DocumentChunk.embedding column with dimension {dim}")

    # 1. Connect to finintel and create extension and tables
    async with engine.begin() as conn:
        logger.info("Initializing database: creating vector extension if not exists...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        logger.info("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
        
        # 2. Add generated ts_vector column for text search if it doesn't exist
        logger.info("Checking and adding generated ts_vector column for full-text search...")
        await conn.execute(text("""
            ALTER TABLE document_chunks 
            ADD COLUMN IF NOT EXISTS ts_vector tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
        """))

        # 3. Create HNSW index for sub-50ms vector query execution
        logger.info("Creating HNSW index for vector cosine similarity...")
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS doc_chunks_hnsw_idx 
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """))

        # 4. Create GIN index for full-text search
        logger.info("Creating GIN index for full-text search...")
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS doc_chunks_fts_idx 
            ON document_chunks USING gin (ts_vector);
        """))
        
    logger.info("Database initialization completed successfully.")


async def get_session() -> AsyncSession:
    """
    Dependency helper to provide an async session for database interactions.
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
