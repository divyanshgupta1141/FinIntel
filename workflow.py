import os
import logging
from typing import List, Dict, Any, Optional, Literal, TypedDict
from pydantic import BaseModel, Field

# Groq SDK imports
from groq import Groq

# Centralized config and utility imports
from config import INFERENCE_MODEL
from utils.rate_limiter import execute_with_retry

# LangGraph imports
from langgraph.graph import StateGraph, START, END

from retrieval import hybrid_search

logger = logging.getLogger("workflow")

# ---------------------------------------------------------
# 1. Pydantic Output Schemas (Unchanged)
# ---------------------------------------------------------

class SourceCitation(BaseModel):
    """
    Explicit citation of the source document chunk to ground claims.
    """
    document_name: str = Field(description="The exact name of the source document.")
    page_number: int = Field(description="The page number in the document where the excerpt is found.")
    excerpt: str = Field(description="Verbatim excerpt from the source text showing the evidence.")


class FinancialReportAnalysis(BaseModel):
    """
    Structured extraction of key financial insights, metrics, and risks.
    """
    key_metric_summary: str = Field(
        description="A clear textual synthesis summarizing key financial metrics found in the retrieved context."
    )
    financial_impact_score: int = Field(
        description="An integer rating from 1 to 10 indicating the financial severity or opportunity scale.",
        ge=1,
        le=10
    )
    risk_factors: List[str] = Field(
        description="A list of operational, strategic, or external risk factors explicitly mentioned in the context."
    )
    impact_assessment: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Strictly constrained categorical classification of the risk factor severity."
    )
    source_citations: List[SourceCitation] = Field(
        description="Verbatim source citations verifying every claim in this analysis. Must not be empty if claims are made."
    )

# ---------------------------------------------------------
# 2. LangGraph State Definition
# ---------------------------------------------------------

class GraphState(TypedDict):
    """
    Type-safe LangGraph state dictionary.
    """
    query: str
    query_embedding: List[float]
    chunks: List[Dict[str, Any]]
    analysis: Optional[FinancialReportAnalysis]
    error: Optional[str]

# ---------------------------------------------------------
# 3. LangGraph Nodes
# ---------------------------------------------------------

async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve node: Connects to the database and pulls the top 3 hybrid search chunks.
    Note: Fix 4 requires us to hardcode the retrieval limit to exactly top 3.
    """
    logger.info("LangGraph: Running Retrieve Node...")
    from database import async_session
    
    query_text = state["query"]
    query_embedding = state["query_embedding"]
    
    async with async_session() as session:
        # Retrieve exactly top 3 chunks (hardcoded inside retrieval.py)
        chunks = await hybrid_search(
            session=session,
            query_text=query_text,
            query_embedding=query_embedding
        )
        
    return {"chunks": chunks}


async def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate node: Refactored to use native Groq SDK.
    Concatenates chunks, strictly slices context_str to 3000 chars (Fix 4),
    and instructs the model to return JSON conforming to the schema natively.
    """
    logger.info("LangGraph: Running Generate Node...")
    
    chunks = state["chunks"]
    query = state["query"]
    
    if not chunks:
        logger.warning("No context chunks available for generation.")
        empty_analysis = FinancialReportAnalysis(
            key_metric_summary="No relevant document chunks found in database.",
            financial_impact_score=1,
            risk_factors=["No document context found."],
            impact_assessment="Low",
            source_citations=[]
        )
        return {"analysis": empty_analysis}
        
    # Format chunks for prompt injection
    context_sections = []
    for idx, c in enumerate(chunks):
        section = (
            f"Chunk {idx+1} (Doc: {c['document_name']}, Page: {c['page_number']}):\n"
            f"{c['chunk_text']}\n"
        )
        context_sections.append(section)
        
    raw_context_str = "\n".join(context_sections)
    
    # Explicitly slice context_str to enforce a maximum of 3,000 characters (Fix 4)
    context_str = raw_context_str[:3000]
    if len(raw_context_str) > 3000:
        logger.info(f"Context truncated from {len(raw_context_str)} characters to 3000 characters.")
    
    system_prompt = (
        "You are an elite corporate finance analyst. Your task is to analyze the retrieved financial document chunks "
        "and answer the user's query with high precision.\n\n"
        "CRITICAL GROUNDING RULES:\n"
        "1. Every metric, risk factor, and claim you provide in the JSON must be strictly cited from the context.\n"
        "2. Fill the `source_citations` array with objects mapping to the correct document name, page number, "
        "and a direct, word-for-word verbatim excerpt (`excerpt`) from the source chunks.\n"
        "3. DO NOT extrapolate, assume, or hallucinate. If the context does not contain the answer, "
        "leave the fields empty/neutral and write a note in key_metric_summary stating the information is missing.\n"
        "4. Do not generate any claim, metric, or assessment that cannot be directly cited from the retrieved context chunks.\n"
        "5. BUDGET CONSTRAINT: You MUST keep your response extremely concise to fit within a strict token budget. "
        "Keep `key_metric_summary` under 30 words. Keep each item in `risk_factors` under 5 words. "
        "Limit the number of citations in `source_citations` to at most 2, and keep their `excerpt` under 15 words.\n\n"
        f"You MUST output raw JSON that conforms EXACTLY to this JSON Schema:\n"
        f"{FinancialReportAnalysis.model_json_schema()}"
    )
    
    prompt = (
        f"User Financial Inquiry: {query}\n\n"
        f"Retrieved Context Chunks (sliced to fit budget):\n"
        f"{context_str}\n\n"
        f"Generate a grounded financial analysis based on the inquiry and the context."
    )
    
    try:
        # Initialize native Groq client (picks up GROQ_API_KEY from environment)
        client = Groq()
        
        # Call chat completions using native Groq SDK with retry protection
        def call_groq():
            return client.chat.completions.create(
                model=INFERENCE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )

        response = await execute_with_retry(call_groq)
        
        raw_text = response.choices[0].message.content
        logger.info(f"Raw response text: {raw_text}")
        analysis_result = FinancialReportAnalysis.model_validate_json(raw_text)
        return {"analysis": analysis_result, "error": None}
        
    except Exception as e:
        if 'response' in locals() and hasattr(response, 'choices') and len(response.choices) > 0:
            logger.error(f"Error during native Groq generation. Raw response: {response.choices[0].message.content}")
        logger.error(f"Error during native Groq generation: {e}")
        error_analysis = FinancialReportAnalysis(
            key_metric_summary=f"Analysis failed due to model or parsing error: {str(e)}",
            financial_impact_score=1,
            risk_factors=["Execution error"],
            impact_assessment="Low",
            source_citations=[]
        )
        return {"analysis": error_analysis, "error": str(e)}

# ---------------------------------------------------------
# 4. LangGraph Workflow Construction
# ---------------------------------------------------------

workflow = StateGraph(GraphState)

# Add Retrieve and Generate Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# START --> retrieve --> generate --> END
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

workflow_app = workflow.compile()
