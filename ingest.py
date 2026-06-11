import os
import sys
import argparse
import asyncio
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

# FORCE RAILWAY PUBLIC PRODUCTION ROUTING
# We set this before importing internal database modules to override local configurations
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:bDkTQgIIqVnGlDZygspuZnoTUrXOxxRw@acela.proxy.rlwy.net:20295/railway"
else:
    raw_url = os.environ["DATABASE_URL"]
    if raw_url.startswith("postgresql://"):
        os.environ["DATABASE_URL"] = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif raw_url.startswith("postgres://"):
        os.environ["DATABASE_URL"] = raw_url.replace("postgres://", "postgresql+asyncpg://", 1)

from database import init_db, get_session, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion")

# Try importing Google GenAI clients
try:
    from google import genai
    GENAI_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GENAI_NEW_SDK = False
    except ImportError:
        logger.error("Neither google-genai nor google-generativeai is installed. Please check requirements.txt.")
        sys.exit(1)

def get_gemini_client():
    """
    Initializes and returns a Gemini client based on available packages.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to run embeddings.")
    
    if GENAI_NEW_SDK:
        return genai.Client(api_key=api_key)
    else:
        genai_legacy.configure(api_key=api_key)
        return genai_legacy

async def get_embedding_async(client, text_content: str) -> List[float]:
    """
    Retrieves a vector embedding for a text chunk.
    First checks the local persistent cache; if not cached, calls the Gemini Embedding API
    with rate-limiting and exponential backoff retry logic, then stores the result.
    """
    from config import EMBEDDING_MODEL
    from utils.embedding_cache import global_embedding_cache
    from utils.rate_limiter import execute_with_retry
    
    # Check local cache first
    cached_emb = await global_embedding_cache.get(text_content)
    if cached_emb is not None:
        return cached_emb

    # Cache miss: generate embedding
    if GENAI_NEW_SDK:
        from google.genai import types
        
        def call_embed():
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text_content,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            
        response = await execute_with_retry(call_embed)
        embedding = response.embeddings[0].values
    else:
        def call_embed_legacy():
            return client.embed_content(
                model=f"models/{EMBEDDING_MODEL}",
                content=text_content
            )
            
        response = await execute_with_retry(call_embed_legacy)
        embedding = response['embedding']

    # Save to local cache
    await global_embedding_cache.set(text_content, embedding)
    return embedding

def chunk_text(text: str, page_number: int, chunk_size_words: int = 500, overlap_words: int = 50) -> List[Dict[str, Any]]:
    """
    Segments page text into semantic chunks of approx 500-800 tokens (estimated at 500 words)
    with a 10% (50 words) overlap.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return []
        
    if len(words) <= chunk_size_words:
        chunks.append({
            "text": text,
            "page_number": page_number
        })
    else:
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size_words]
            if len(chunk_words) < 100 and chunks:
                break
            chunks.append({
                "text": " ".join(chunk_words),
                "page_number": page_number
            })
            i += (chunk_size_words - overlap_words)
            
    return chunks

def extract_pdf_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Opens a local PDF and extracts text chunks along with page number metadata.
    """
    logger.info(f"Opening PDF file: {pdf_path}")
    reader = PdfReader(pdf_path)
    all_chunks = []
    
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue
            
        page_chunks = chunk_text(page_text, page_num)
        all_chunks.extend(page_chunks)
        logger.info(f"Extracted {len(page_chunks)} chunks from page {page_num}")
        
    return all_chunks

def generate_mock_financial_data() -> List[Dict[str, Any]]:
    """
    Generates high-quality mock financial report data for Apple Inc. FY2025
    to enable out-of-the-box testing without needing a PDF.
    """
    logger.info("Generating mock corporate annual report data for seeding...")
    mock_data = [
        {
            "page_number": 1,
            "text": """
            Apple Inc. FY2025 Annual Report. Item 1. Business Overview.
            Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services.
            During the fiscal year 2025, Apple recorded total revenue of $395.0 billion, representing a 4% growth year-over-year. Services revenue reached an all-time high of $98.5 billion, driven by active subscriptions in Apple Music, iCloud, Apple Pay, and App Store transactions. iPhone net sales accounted for $201.2 billion, maintaining its position as the primary hardware revenue driver. Operating income for the fiscal year was $115.0 billion with a net margin of 26.5%.
            """
        },
        {
            "page_number": 2,
            "text": """
            Apple Inc. FY2025 Annual Report. Item 7. Management's Discussion and Analysis (MD&A).
            Key Metric Summary: Gross margin for the year was 46.2%, expanding by 150 basis points due to favorable product mix shifts towards high-margin Services and Pro hardware models. Research and Development (R&D) expenses rose to $32.4 billion, reflecting intensive capital deployment in generative AI technologies, spatial computing (Vision Pro updates), and custom silicon development. 
            International sales accounted for 58% of total revenue. Cash, cash equivalents, and marketable securities totaled $165.0 billion, offset by a term debt of $95.0 billion, yielding a net cash position of $70.0 billion. The company returned $90.0 billion to shareholders through share repurchases and dividends.
            """
        },
        {
            "page_number": 3,
            "text": """
            Apple Inc. FY2025 Annual Report. Item 1A. Risk Factors.
            Risk Factor 1: Supply Chain Vulnerabilities. The company's manufacturing operations are concentrated in specific geographic regions, making it susceptible to disruptions from natural disasters, geopolitical tensions, trade disputes, or regulatory changes. Any disruption in semiconductor supply or assembly logistics could severely impact shipment timelines.
            Risk Factor 2: Currency Fluctuations. Apple operates globally, and a strengthening US Dollar (USD) represents a major foreign exchange headwind, impacting gross margins in international markets.
            Risk Factor 3: Regulatory Compliance and Antitrust. Increasing regulatory scrutiny regarding App Store policies, fee structures, and default search engine contracts in the US and European Union represents a significant litigation risk. Fines or mandated changes to business models could reduce Services margins.
            """
        }
    ]
    return mock_data

async def ingest_document(pdf_path: str = None, doc_name: str = "Annual_Report_2025"):
    """
    Performs the entire ingestion workflow targeting live Railway instance.
    """
    if pdf_path:
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return
        chunks_data = extract_pdf_chunks(pdf_path)
    else:
        chunks_data = generate_mock_financial_data()
        doc_name = "Apple_Inc_FY2025_Mock_Report"
        
    if not chunks_data:
        logger.warning("No chunks found to ingest.")
        return
        
    logger.info(f"Total chunks targeted for remote production upload: {len(chunks_data)}")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment. Generating mock embeddings (768-dim zero-vectors).")
        client = None
    else:
        try:
            client = get_gemini_client()
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to mock embeddings.")
            client = None
            
    logger.info("Initializing connection parameters and verifying production schema layout...")
    await init_db()
    
    async for session in get_session():
        logger.info("Pinging Railway database cluster. Pushing vector records asynchronously...")
        processed_count = 0
        
        for item in chunks_data:
            text_content = item["text"].strip()
            page_num = item["page_number"]
            
            logger.info(f"Streaming chunk {processed_count+1}/{len(chunks_data)} into live DB...")
            
            if client is None:
                embedding = [0.0] * 768
            else:
                embedding = await get_embedding_async(client, text_content)
            
            chunk_obj = DocumentChunk(
                document_name=doc_name,
                page_number=page_num,
                chunk_text=text_content,
                embedding=embedding
            )
            session.add(chunk_obj)
            processed_count += 1
            
            if client is not None:
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.05)
            
        await session.commit()
        logger.info(f"🎉 Production sync complete! Successfully injected {processed_count} vector objects into Railway.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial PDF documents into the FinIntel PostgreSQL vector store.")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file. If omitted, mock corporate financial data is ingested.")
    parser.add_argument("--name", type=str, default="Annual_Report_2025", help="Name metadata for the document.")
    args = parser.parse_args()
    
    asyncio.run(ingest_document(pdf_path=args.pdf, doc_name=args.name))