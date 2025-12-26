"""
Cloud Model Cache - Vercel Blob/KV Integration
================================================
Caches NLP models and embeddings using Vercel managed services to avoid
re-downloading large models on each serverless function invocation.

Storage Strategy:
- Vercel Blob: Large binary files (models, code embeddings)
- Vercel KV: Small metadata, ephemeral caching
- Vercel Postgres: Structured data (already in use)

Environment Variables Required:
- BLOB_READ_WRITE_TOKEN: Vercel Blob access token
- KV_REST_API_URL: Vercel KV REST API URL
- KV_REST_API_TOKEN: Vercel KV access token
"""

import os
import hashlib
import json
from typing import Optional, Dict, Any
from pathlib import Path

class CloudModelCache:
    """
    Manages model and embedding caching using Vercel services.

    Usage:
        cache = CloudModelCache()

        # Cache a model
        cache.store_model("bioclinical-bert", model_data)

        # Retrieve cached model
        model = cache.get_model("bioclinical-bert")
    """

    def __init__(self):
        self.use_blob = os.getenv("BLOB_READ_WRITE_TOKEN") is not None
        self.use_kv = os.getenv("KV_REST_API_URL") is not None

        if not (self.use_blob or self.use_kv):
            print("⚠️  No Vercel Blob/KV tokens found. Using local cache only.")
            print("   Set BLOB_READ_WRITE_TOKEN and KV_REST_API_URL for cloud caching.")

        self.local_cache_dir = Path.home() / ".cache" / "billing_nlp"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, identifier: str) -> str:
        """Generate consistent cache key."""
        return hashlib.md5(identifier.encode()).hexdigest()

    def store_model_metadata(self, model_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Store model metadata in Vercel KV (or local cache).

        Args:
            model_name: Unique model identifier
            metadata: Model metadata (version, size, last_updated, etc.)

        Returns:
            True if successful
        """
        cache_key = self._get_cache_key(model_name)

        if self.use_kv:
            try:
                import requests
                url = f"{os.getenv('KV_REST_API_URL')}/set/{cache_key}"
                headers = {"Authorization": f"Bearer {os.getenv('KV_REST_API_TOKEN')}"}
                response = requests.post(url, json=metadata, headers=headers)
                return response.status_code == 200
            except Exception as e:
                print(f"⚠️  KV storage failed: {e}")

        # Fallback to local cache
        cache_file = self.local_cache_dir / f"{cache_key}_metadata.json"
        with open(cache_file, 'w') as f:
            json.dump(metadata, f)
        return True

    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata from cache."""
        cache_key = self._get_cache_key(model_name)

        if self.use_kv:
            try:
                import requests
                url = f"{os.getenv('KV_REST_API_URL')}/get/{cache_key}"
                headers = {"Authorization": f"Bearer {os.getenv('KV_REST_API_TOKEN')}"}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json().get("result")
            except Exception as e:
                print(f"⚠️  KV retrieval failed: {e}")

        # Fallback to local cache
        cache_file = self.local_cache_dir / f"{cache_key}_metadata.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        return None

    def get_local_model_path(self, model_name: str) -> Path:
        """
        Get local path for model storage.

        For Vercel/serverless, models should be loaded from:
        1. /tmp directory (ephemeral)
        2. HuggingFace cache (auto-managed)
        3. Custom cache directory
        """
        # Use /tmp for serverless environments
        if os.path.exists("/tmp"):
            cache_dir = Path("/tmp") / "nlp_models"
        else:
            cache_dir = self.local_cache_dir / "models"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / model_name.replace("/", "_")

    def should_download_model(self, model_name: str, version: str) -> bool:
        """
        Check if model needs to be downloaded.

        Args:
            model_name: Model identifier
            version: Model version

        Returns:
            True if download needed
        """
        metadata = self.get_model_metadata(model_name)
        if metadata is None:
            return True

        if metadata.get("version") != version:
            return True

        # Check if local file exists
        local_path = self.get_local_model_path(model_name)
        return not local_path.exists()


def get_model_with_cache(model_name: str, model_loader_fn, version: str = "latest"):
    """
    Load model with caching strategy.

    Args:
        model_name: HuggingFace model name or identifier
        model_loader_fn: Function to load model (e.g., SentenceTransformer)
        version: Model version

    Returns:
        Loaded model

    Example:
        from sentence_transformers import SentenceTransformer

        model = get_model_with_cache(
            "NeuML/bioclinical-modernbert-base-embeddings",
            lambda: SentenceTransformer("NeuML/bioclinical-modernbert-base-embeddings"),
            version="v1.0"
        )
    """
    cache = CloudModelCache()

    # Check if we need to download
    if cache.should_download_model(model_name, version):
        print(f"📥 Downloading model: {model_name}...")
        model = model_loader_fn()

        # Store metadata
        cache.store_model_metadata(model_name, {
            "version": version,
            "model_name": model_name,
            "last_updated": str(os.path.getmtime(cache.get_local_model_path(model_name)))
        })

        print(f"✅ Model cached: {model_name}")
        return model
    else:
        print(f"♻️  Using cached model: {model_name}")
        return model_loader_fn()


# Vercel Blob helper for large binary storage
class BlobStorage:
    """
    Store large binary files (embeddings, models) in Vercel Blob.

    Usage:
        blob = BlobStorage()

        # Store embeddings
        blob.put("embeddings/icd10.bin", embeddings_data)

        # Retrieve
        data = blob.get("embeddings/icd10.bin")
    """

    def __init__(self):
        self.token = os.getenv("BLOB_READ_WRITE_TOKEN")
        if not self.token:
            raise ValueError("BLOB_READ_WRITE_TOKEN not set")

    def put(self, key: str, data: bytes) -> bool:
        """Upload binary data to Vercel Blob."""
        try:
            import requests
            url = f"https://blob.vercel-storage.com/{key}"
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.put(url, data=data, headers=headers)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Blob upload failed: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """Download binary data from Vercel Blob."""
        try:
            import requests
            url = f"https://blob.vercel-storage.com/{key}"
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            print(f"❌ Blob download failed: {e}")
            return None
