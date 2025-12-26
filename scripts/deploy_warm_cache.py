#!/usr/bin/env python3
"""
Deploy Warm Cache Script
=========================
Pre-populate Vercel Blob storage with frequently-used models and embeddings.
Run this after each deployment to minimize cold start times.

Usage:
    python scripts/deploy_warm_cache.py

Prerequisites:
    - BLOB_READ_WRITE_TOKEN set in environment
    - VERCEL_POSTGRES_URL set in environment
"""

import os
import sys
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def warm_code_sets():
    """Pre-download and cache ICD-10/HCPCS code sets."""
    print("\n📦 Warming code set cache...")

    try:
        from src.data.cloud_cache import BlobStorage
        from src.data.build_embeddings import download_icd10_codes, download_hcpcs_codes

        blob = BlobStorage()

        # Download code sets
        icd10_df = download_icd10_codes(year=2025)
        hcpcs_df = download_hcpcs_codes(year=2025)

        # Cache in Blob storage
        blob.put("codesets/icd10_2025.pkl", pickle.dumps(icd10_df))
        blob.put("codesets/hcpcs_2025.pkl", pickle.dumps(hcpcs_df))

        print(f"✅ Cached {len(icd10_df)} ICD-10 codes")
        print(f"✅ Cached {len(hcpcs_df)} HCPCS codes")
        return True

    except Exception as e:
        print(f"⚠️  Code set caching failed: {e}")
        return False

def warm_embedding_model():
    """Pre-load embedding model to cache."""
    print("\n📦 Warming embedding model cache...")

    try:
        from src.data.cloud_cache import get_model_with_cache
        from sentence_transformers import SentenceTransformer
        from config.settings import embedding_config

        model = get_model_with_cache(
            embedding_config.model_name,
            lambda: SentenceTransformer(embedding_config.model_name),
            version="v1.0"
        )

        print(f"✅ Cached model: {embedding_config.model_name}")

        # Test encoding
        test_embedding = model.encode("Test clinical note")
        print(f"✅ Model verification passed (dim: {len(test_embedding)})")
        return True

    except Exception as e:
        print(f"⚠️  Model caching failed: {e}")
        return False

def warm_spacy_models():
    """Pre-load spaCy models."""
    print("\n📦 Warming spaCy models...")

    try:
        import spacy

        # Load both models
        nlp_general = spacy.load("en_core_web_sm")
        print("✅ Loaded en_core_web_sm")

        nlp_medical = spacy.load("en_core_sci_md")
        print("✅ Loaded en_core_sci_md")

        return True

    except Exception as e:
        print(f"⚠️  spaCy model warming failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 Warm Cache Deployment Script")
    print("=" * 60)

    # Check required env vars
    required_vars = ["BLOB_READ_WRITE_TOKEN", "VERCEL_POSTGRES_URL"]
    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("\nRun: vercel env pull")
        sys.exit(1)

    # Warm caches
    results = []
    results.append(("Code Sets", warm_code_sets()))
    results.append(("Embedding Model", warm_embedding_model()))
    results.append(("spaCy Models", warm_spacy_models()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 WARM CACHE SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    success_count = sum(1 for _, s in results if s)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n✅ All caches warmed successfully")
        print("\nYour deployment is now optimized for fast cold starts!")
    else:
        print(f"\n⚠️  {success_count}/{total_count} caches warmed")
        print("\nSome caches failed. Check errors above.")

    print("=" * 60)

if __name__ == "__main__":
    main()
