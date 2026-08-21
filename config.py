import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup root structured logger level and configuration
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger("config")

# Model settings
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen/qwen3.6-27b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "qwen/qwen3.6-27b")
EMBEDDING_DIMENSION = 768

# Throttling & Evaluation mode settings
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
EVAL_MODE = os.getenv("EVAL_MODE", "fast")  # "fast" = faithfulness only, "full" = faithfulness & context precision

# DB & Redis connection strings
raw_db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/finintel")

# Sanitize query parameters for asyncpg compatibility:
# 1. Convert sslmode to ssl (e.g. sslmode=require -> ssl=require)
# 2. Remove channel_binding which is not supported by asyncpg
if "?" in raw_db_url:
    base_url, query_str = raw_db_url.split("?", 1)
    params = query_str.split("&")
    new_params = []
    for p in params:
        if p.startswith("sslmode="):
            val = p.split("=", 1)[1]
            new_params.append(f"ssl={val}")
        elif p.startswith("channel_binding="):
            continue
        else:
            new_params.append(p)
    raw_db_url = f"{base_url}?{'&'.join(new_params)}"

if raw_db_url.startswith("postgresql://"):
    DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif raw_db_url.startswith("postgres://"):
    DATABASE_URL = raw_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_db_url

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Embedding cache settings
LOCAL_CACHE_PATH = os.getenv("LOCAL_CACHE_PATH", ".cache/embeddings_cache.json")

def validate_environment():
    """
    Verifies that all required environment variables are set.
    Gracefully handles missing variables with clean user-friendly errors.
    """
    # 1. Verify Gemini API Key (needed for embeddings)
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("=" * 60, file=sys.stderr)
        print("❌ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is missing!", file=sys.stderr)
        print("   This is required for generating document embeddings.", file=sys.stderr)
        print("   Please check your .env file or environment variables.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
        
    # Sync GOOGLE_API_KEY and GEMINI_API_KEY to ensure both SDKs and integrations use the same verified key
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["GOOGLE_API_KEY"] = gemini_key

    # 2. Verify Groq API Key (needed for LLM generation)
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("=" * 60, file=sys.stderr)
        print("❌ ERROR: GROQ_API_KEY is missing!", file=sys.stderr)
        print("   This is required for running LLM inference on Groq.", file=sys.stderr)
        print("   Please check your .env file or environment variables.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    # 2. Verify LangSmith Tracing configurations
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith: LANGCHAIN_API_KEY not found. Tracing disabled automatically to prevent runtime warning spam.")
    else:
        # If API key is present, enable or keep standard configuration
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith: Tracing configuration loaded successfully.")

# Run validation on import
validate_environment()
