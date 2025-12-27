"""
HuggingFace Inference API Client
=================================
Uses HuggingFace Inference API instead of local models.
No need to download large model files - everything runs in the cloud.

Benefits:
- No local model downloads (saves ~400MB per model)
- No GPU needed
- Faster cold starts
- Automatic model scaling
- Free tier: 30K requests/month
"""

import os
import requests
import time
from typing import List, Union, Optional
import numpy as np


class HuggingFaceEmbeddings:
    """
    HuggingFace Inference API client for embeddings.

    Usage:
        embedder = HuggingFaceEmbeddings(
            model="NeuML/bioclinical-modernbert-base-embeddings"
        )

        # Single text
        embedding = embedder.encode("Patient with diabetes")

        # Batch
        embeddings = embedder.encode([
            "Type 2 diabetes mellitus",
            "Essential hypertension"
        ])
    """

    def __init__(
        self,
        model: str = "NeuML/bioclinical-modernbert-base-embeddings",
        api_token: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize HF Inference API client.

        Args:
            model: HuggingFace model identifier
            api_token: HF API token (reads from HF_TOKEN env var if not provided)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.timeout = timeout

        # Get API token
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

        # Cache for dimension (fetched on first use)
        self._dimension = None

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            show_progress_bar: Show progress (for compatibility)
            batch_size: Batch size for API calls

        Returns:
            numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            return self._encode_single(texts)

        # Handle batch
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._encode_batch(batch)
            embeddings.extend(batch_embeddings)

            if show_progress_bar and i > 0:
                print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)} texts")

        return np.array(embeddings)

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        response = self._make_request({"inputs": text})

        if response.status_code == 200:
            embedding = response.json()
            return np.array(embedding)
        elif response.status_code == 503:
            # Model loading - wait and retry
            print("  ⏳ Model loading (first use)... waiting 20s")
            time.sleep(20)
            response = self._make_request({"inputs": text})
            if response.status_code == 200:
                return np.array(response.json())

        raise Exception(f"API error {response.status_code}: {response.text}")

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode batch of texts."""
        # HF API accepts array of inputs
        response = self._make_request({"inputs": texts})

        if response.status_code == 200:
            embeddings = response.json()
            return [np.array(emb) for emb in embeddings]
        elif response.status_code == 503:
            # Model loading - wait and retry
            time.sleep(20)
            response = self._make_request({"inputs": texts})
            if response.status_code == 200:
                embeddings = response.json()
                return [np.array(emb) for emb in embeddings]

        raise Exception(f"API error {response.status_code}: {response.text}")

    def _make_request(self, payload: dict) -> requests.Response:
        """Make API request."""
        return requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension from a test encoding
            test_emb = self.encode("test")
            self._dimension = len(test_emb)
        return self._dimension


def get_embeddings_model(
    model_name: str = "NeuML/bioclinical-modernbert-base-embeddings"
) -> HuggingFaceEmbeddings:
    """
    Get HuggingFace embeddings model.

    Drop-in replacement for sentence-transformers SentenceTransformer.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFaceEmbeddings instance

    Example:
        # Old (local):
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer("NeuML/bioclinical-modernbert-base-embeddings")

        # New (API):
        from src.data.hf_embeddings import get_embeddings_model
        model = get_embeddings_model()

        # Same interface!
        embedding = model.encode("Patient with diabetes")
    """
    return HuggingFaceEmbeddings(model=model_name)


# For backward compatibility
SentenceTransformer = HuggingFaceEmbeddings
