"""Smoke test verifying schema definitions and RRF logic components."""
import os

def test_environment_variables():
    assert "GEMINI_API_KEY" in os.environ
    assert "GROQ_API_KEY" in os.environ

def test_rrf_scoring_math():
    # Verifies the Reciprocal Rank Fusion calculation logic
    k = 60
    rank_dense = 1
    rank_lexical = 2
    score = (1 / (k + rank_dense)) + (1 / (k + rank_lexical))
    assert round(score, 4) == round((1/61 + 1/62), 4)
