import os
import sys
import asyncio
import logging
from types import ModuleType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# MONKEYPATCH: Resolve LangChain/Ragas Python 3.14 compat
# ---------------------------------------------------------
# Dynamic injection to satisfy Ragas' import of ChatVertexAI from langchain_community
try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    class MockChatVertexAI:
        pass
    ChatVertexAI = MockChatVertexAI

vertexai_mock = ModuleType('langchain_community.chat_models.vertexai')
vertexai_mock.ChatVertexAI = ChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = vertexai_mock

# Now we can safely import Ragas and HuggingFace datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.run_config import RunConfig
from google import genai
from config import EVALUATION_MODEL, EMBEDDING_MODEL, EVAL_MODE

from database import async_session
from retrieval import hybrid_search
from ingest import get_gemini_client, get_embedding_async
from workflow import workflow_app, FinancialReportAnalysis

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("evaluation")

# ---------------------------------------------------------
# Golden Test Dataset
# ---------------------------------------------------------

GOLDEN_DATASET = [
    {
        "question": "What was Apple's total revenue for the fiscal year 2025 and its growth rate?",
        "ground_truth": "Total revenue was $395.0 billion, which represents a 4% growth year-over-year."
    },
    {
        "question": "What was the expansion in gross margin for Apple in FY2025, and what drove it?",
        "ground_truth": "Gross margin expanded by 150 basis points to 46.2%, driven by favorable product mix shifts towards high-margin Services and Pro hardware models."
    },
    {
        "question": "Identify the main risk factors discussed in Apple's FY2025 report.",
        "ground_truth": "The three main risks are: Supply Chain Vulnerabilities (geographic concentration, assembly logistics), Currency Fluctuations (USD strengthening headwind), and Regulatory Compliance/Antitrust (App Store policies, litigation risk in US/EU)."
    },
    {
        "question": "How much did Apple spend on Research and Development (R&D) in FY2025, and what was the focus?",
        "ground_truth": "Apple spent $32.4 billion on R&D, focusing on generative AI, spatial computing (Vision Pro updates), and custom silicon."
    },
    {
        "question": "What is Apple's net cash position in FY2025 and how much was returned to shareholders?",
        "ground_truth": "Apple had a net cash position of $70.0 billion ($165.0 billion cash minus $95.0 billion term debt) and returned $90.0 billion to shareholders."
    }
]

# ---------------------------------------------------------
# Evaluation Setup & Run
# ---------------------------------------------------------

async def run_evaluation():
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not gemini_key or not groq_key:
        print("Error: GEMINI_API_KEY or GROQ_API_KEY environment variable is not set. Cannot run evaluation.")
        return
        
    print("=" * 75)
    print("STARTING OFFICIAL RAGAS EVALUATION RUN (USING GROQ + GEMINI)")
    print("=" * 75)
    
    # 1. Initialize RAGAS configurations natively
    from groq import Groq
    groq_client = Groq(api_key=groq_key)
    gemini_client = genai.Client(api_key=gemini_key)
    
    eval_llm = llm_factory(model=EVALUATION_MODEL, provider="groq", client=groq_client)
    eval_embeddings = GoogleEmbeddings(client=gemini_client, model=EMBEDDING_MODEL)
    
    # Configure metrics with native providers (injecting them as properties to support legacy Ragas wrapper validation)
    faithfulness_metric = Faithfulness()
    faithfulness_metric.llm = eval_llm
    
    context_precision_metric = ContextPrecision()
    context_precision_metric.llm = eval_llm
    context_precision_metric.embeddings = eval_embeddings
    
    # Select metrics based on configuration mode
    if EVAL_MODE == "fast":
        metrics = [faithfulness_metric]
        print(f"Evaluation Mode: 'fast' (running Faithfulness only with model '{EVALUATION_MODEL}')")
    else:
        metrics = [faithfulness_metric, context_precision_metric]
        print(f"Evaluation Mode: 'full' (running Faithfulness and Context Precision with model '{EVALUATION_MODEL}')")
    
    # 2. Collect pipeline runs
    questions = []
    contexts = []
    answers = []
    ground_truths = []
    
    client = get_gemini_client()
    
    for i, pair in enumerate(GOLDEN_DATASET):
        q = pair["question"]
        gt = pair["ground_truth"]
        print(f"Gathering RAG execution data for Case {i+1}/5: '{q[:50]}...'")
        
        # A. Query Embedding
        query_embedding = await get_embedding_async(client, q)
        
        # B. Retrieve chunks
        async with async_session() as session:
            chunks = await hybrid_search(session, q, query_embedding)
            
        # C. Run LangGraph workflow
        state = {
            "query": q,
            "query_embedding": query_embedding,
            "chunks": chunks,
            "analysis": None
        }
        final_state = await workflow_app.ainvoke(state)
        analysis: FinancialReportAnalysis = final_state.get("analysis")
        
        # Collect outputs
        questions.append(q)
        # Contexts inside Ragas HuggingFace dataset must be a list of lists of strings: List[List[str]]
        contexts.append([c["chunk_text"] for c in chunks])
        answers.append(analysis.key_metric_summary)
        ground_truths.append(gt)
        
        # Avoid free-tier Gemini rate limits by pausing briefly between runs
        await asyncio.sleep(1.0)
        
    print("\nCompiling HuggingFace Dataset...")
    # 3. Create HuggingFace Dataset
    dataset_dict = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print("Running official Ragas evaluation pipeline. This may take a minute...")
    # 4. Execute Ragas Evaluate with RunConfig to control concurrency and rate limits
    run_config = RunConfig(max_workers=2, max_retries=5)
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config
    )
    
    print("\n" + "=" * 75)
    print("OFFICIAL RAGAS EVALUATION METRICS SUMMARY")
    print("=" * 75)
    print(eval_result)
    print("=" * 75)
    
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_evaluation())
