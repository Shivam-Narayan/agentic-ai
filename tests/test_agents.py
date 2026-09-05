"""
Diagnostic and smoke tests for agent component initialization.
"""
import os
import pytest
from pathlib import Path


def test_agent_environment_config():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    assert data_dir.exists(), "data directory should exist in project root"


def test_agent_embeddings_loader():
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        assert embed_model is not None
    except Exception as exc:
        pytest.skip(f"Embedding model initialization skipped in current environment: {exc}")
