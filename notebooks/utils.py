"""
Shared Utilities for Medical Billing ML System

This module contains common functions used across multiple notebooks
to eliminate code duplication and ensure consistency.

Functions:
- get_embedding(): Generate embeddings using HF Inference API
- init_database(): Initialize database connection
- init_pinecone(): Initialize Pinecone vector database
- init_hf_client(): Initialize HuggingFace Inference client
- init_anthropic_client(): Initialize Anthropic client
- ensure_package_installed(): Check and install Python packages
"""

import os
import sys
import time
import subprocess
import importlib.util
from typing import Optional, Tuple
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config import get_config


def ensure_package_installed(package_name: str, import_name: Optional[str] = None,
                            upgrade: bool = False, uninstall_first: Optional[str] = None):
    """
    Ensure a Python package is installed, install if missing

    Args:
        package_name: Package name for pip install
        import_name: Name used for import (if different from package_name)
        upgrade: Whether to upgrade if already installed
        uninstall_first: Package to uninstall before installing (for migrations)
    """
    check_name = import_name or package_name

    if importlib.util.find_spec(check_name) is None or upgrade:
        if uninstall_first:
            print(f"   📦 Uninstalling {uninstall_first}...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", uninstall_first],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"   📦 Installing {package_name}...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("-U")
        cmd.append(package_name)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   ✅ {package_name} installed")


def get_embedding(text: str, hf_client, model_id: str, embedding_dim: int,
                 retry_count: int = 0, max_retries: int = 3,
                 retry_delay: int = 2) -> Optional[np.ndarray]:
    """
    Generate embedding using HF Inference API with retry logic.

    Args:
        text: Text to embed
        hf_client: HuggingFace InferenceClient instance
        model_id: Model ID to use
        embedding_dim: Expected embedding dimension
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries (seconds)

    Returns:
        numpy array of embedding, or None if all retries fail
    """
    try:
        result = hf_client.feature_extraction(text, model=model_id)
        embedding = np.array(result)

        # Mean pooling if token-level embeddings returned
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)
        embedding = embedding.astype(np.float32)

        # Validate dimension
        if embedding.shape[0] != embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != expected {embedding_dim}"
            )

        return embedding

    except Exception as e:
        if retry_count < max_retries:
            print(f"\n   ⚠️ API error (attempt {retry_count + 1}/{max_retries}): {e}")
            time.sleep(retry_delay * (retry_count + 1))  # Exponential backoff
            return get_embedding(text, hf_client, model_id, embedding_dim,
                               retry_count + 1, max_retries, retry_delay)
        else:
            print(f"\n   ❌ Failed after {max_retries} attempts: {e}")
            return None


def init_database(database_url: str, verbose: bool = True) -> Tuple[Engine, int]:
    """
    Initialize database connection and return engine and total claims count

    Args:
        database_url: PostgreSQL connection URL
        verbose: Whether to print connection info

    Returns:
        (engine, total_claims) tuple
    """
    if verbose:
        print("\n🔌 Connecting to Vercel Postgres...")

    engine = create_engine(database_url)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM claims"))
        total_claims = result.fetchone()[0]

        if verbose:
            print(f"   ✅ Connected! Found {total_claims:,} claims in database")

    return engine, total_claims


def init_pinecone(api_key: str, index_name: str, dimension: int,
                 verbose: bool = True):
    """
    Initialize Pinecone client and return index

    Args:
        api_key: Pinecone API key
        index_name: Name of the index
        dimension: Embedding dimension
        verbose: Whether to print initialization info

    Returns:
        (pc_client, index) tuple
    """
    # Ensure package is installed
    ensure_package_installed("pinecone", uninstall_first="pinecone-client")

    from pinecone import Pinecone, ServerlessSpec

    if verbose:
        print("\n🌲 Initializing Pinecone...")

    pc = Pinecone(api_key=api_key)

    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        if verbose:
            print(f"   Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier region
            )
        )
        time.sleep(10)
        if verbose:
            print(f"   ✅ Index created")
    else:
        if verbose:
            print(f"   ✅ Index '{index_name}' exists")

    index = pc.Index(index_name)

    if verbose:
        stats = index.describe_index_stats()
        print(f"   Vectors in index: {stats.total_vector_count:,}")

    return pc, index


def init_hf_client(token: str, model_id: str, verbose: bool = True):
    """
    Initialize HuggingFace Inference client

    Args:
        token: HuggingFace API token
        model_id: Model ID to verify
        verbose: Whether to print initialization info

    Returns:
        InferenceClient instance
    """
    # Ensure package is installed
    ensure_package_installed("huggingface_hub")

    from huggingface_hub import HfApi, InferenceClient

    if verbose:
        print(f"\n🤗 Setting up HuggingFace embeddings...")
        print(f"   Model: {model_id}")

    # Verify model availability
    if verbose:
        try:
            info = HfApi(token=token).model_info(model_id)
            pipeline = getattr(info, "pipeline_tag", None)
            print(f"   ✅ Model available (pipeline: {pipeline or 'unknown'})")
        except Exception as exc:
            print(f"   ⚠️ Could not verify model ({exc})")

    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
    )

    return client


def init_anthropic_client(api_key: Optional[str], model_id: str,
                          verbose: bool = True):
    """
    Initialize Anthropic client for Claude API

    Args:
        api_key: Anthropic API key (optional)
        model_id: Claude model ID
        verbose: Whether to print initialization info

    Returns:
        Anthropic client instance or None if no API key
    """
    if not api_key:
        if verbose:
            print("⚠️ ANTHROPIC_API_KEY not set - will use generic category names")
        return None

    # Ensure package is installed
    ensure_package_installed("anthropic")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    if verbose:
        print(f"   ✅ Anthropic: {model_id}")
        print(f"   💡 Model configured via config.py or CLAUDE_MODEL env var")

    return client


def validate_environment_variables(required_vars: list, verbose: bool = True) -> bool:
    """
    Validate that required environment variables are set

    Args:
        required_vars: List of required environment variable names
        verbose: Whether to print validation info

    Returns:
        True if all required vars are set, False otherwise
    """
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        if verbose:
            print("❌ Missing environment variables:")
            for var in missing:
                print(f"   - {var}")
            print("\n📋 Setup instructions:")
            if "HF_TOKEN" in missing:
                print("   HF_TOKEN: https://huggingface.co/settings/tokens")
            if "PINECONE_API_KEY" in missing:
                print("   PINECONE_API_KEY: https://www.pinecone.io/")
            if "VERCEL_POSTGRES_URL" in missing:
                print("   VERCEL_POSTGRES_URL: PostgreSQL connection string")
        return False

    return True


def test_embedding_generation(hf_client, model_id: str, embedding_dim: int,
                              verbose: bool = True) -> bool:
    """
    Test embedding generation with a sample text

    Args:
        hf_client: HuggingFace InferenceClient instance
        model_id: Model ID to test
        embedding_dim: Expected embedding dimension
        verbose: Whether to print test results

    Returns:
        True if test passed, False otherwise
    """
    if verbose:
        print("\n🧪 Testing embedding generation...")

    try:
        test_emb = get_embedding(
            "Patient with Type 2 diabetes mellitus",
            hf_client, model_id, embedding_dim
        )

        if test_emb is not None:
            assert test_emb.shape == (embedding_dim,)
            if verbose:
                print(f"   ✅ Shape: {test_emb.shape}")
                print(f"   ✅ Sample: [{test_emb[0]:.4f}, {test_emb[1]:.4f}, ...]")
            return True
        else:
            raise Exception("Embedding generation failed")
    except Exception as e:
        if verbose:
            print(f"   ❌ FAILED: {e}")
        return False
