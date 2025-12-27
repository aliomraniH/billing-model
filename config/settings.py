"""
Cloud-aware configuration for Medical Billing NLP System.
Designed for Deepnote + Vercel Postgres deployment.
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration - auto-detects cloud vs local."""
    url: str = os.getenv(
        "VERCEL_POSTGRES_URL",
        os.getenv("DATABASE_URL", "postgresql://localhost:5432/billing_db")
    )

    @property
    def is_cloud(self) -> bool:
        return "neon.tech" in self.url or "vercel" in self.url.lower()

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    # Using PubMedBERT - clinically validated, Inference API compatible
    # Alternative: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" for semantic search
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    dimension: int = 768
    max_seq_length: int = 512  # PubMedBERT standard
    batch_size: int = 32

@dataclass
class NLPConfig:
    """Clinical NLP configuration."""
    spacy_model: str = "en_core_sci_md"
    confidence_threshold: float = 0.85
    consensus_threshold: float = 0.70

@dataclass
class ProcessingConfig:
    """Processing behavior configuration."""
    batch_size: int = 64
    max_claims_per_run: int = 1000
    enable_deidentification: bool = True

# Global config instances
db_config = DatabaseConfig()
embedding_config = EmbeddingConfig()
nlp_config = NLPConfig()
processing_config = ProcessingConfig()

def get_hf_token() -> Optional[str]:
    """Get HuggingFace token (optional for public models)."""
    return os.getenv("HF_TOKEN")

def validate_environment():
    """Validate required environment variables."""
    if not db_config.url or db_config.url == "postgresql://localhost:5432/billing_db":
        print("⚠️  WARNING: Using default database URL. Set VERCEL_POSTGRES_URL in Deepnote.")
        return False
    print(f"✅ Database: {'Cloud (Vercel/Neon)' if db_config.is_cloud else 'Local'}")
    print(f"✅ Embedding Model: {embedding_config.model_name}")
    return True
