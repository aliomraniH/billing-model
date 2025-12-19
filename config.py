"""
Centralized Model Configuration for Medical Billing ML System

This file contains ALL model configurations, API endpoints, and version information.
NO models are hardcoded in the notebooks - everything is configured here.

Usage:
    from config import ModelConfig
    config = ModelConfig.load()
    embedding_model = config.embedding.model_id
    claude_model = config.llm.model_id

Configuration Priority:
    1. Environment variables (highest priority)
    2. config.json file (if exists)
    3. Default values in this file (fallback)

Model Updates:
    To update models, either:
    - Update this file and commit
    - Set environment variables
    - Create a config.json file (not committed to git)
"""

import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import timedelta


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    provider: str = "huggingface"
    model_id: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    api_endpoint: str = "https://api-inference.huggingface.co"
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: int = 2


@dataclass
class LLMConfig:
    """Configuration for LLM (Claude) model"""
    provider: str = "anthropic"
    model_id: str = "claude-sonnet-4-5-20250929"
    api_version: str = "2023-06-01"
    max_tokens: int = 500
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay_seconds: int = 2


@dataclass
class RefreshConfig:
    """Configuration for data refresh intervals"""
    # Embedding refresh intervals (in hours)
    default_embedding_refresh_hours: int = 12
    category_refresh_hours: int = 24
    cluster_refresh_hours: int = 48

    # Minimum time before allowing re-embedding (prevents too frequent refreshes)
    min_embedding_age_hours: int = 1

    # Auto-refresh settings
    auto_refresh_enabled: bool = True
    refresh_on_startup: bool = False

    # Batch refresh settings
    refresh_batch_size: int = 100
    max_refresh_per_run: int = 1000  # Limit refreshes per execution


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database"""
    index_name: str = "medical-billing-notes"
    dimension: int = 384  # Must match embedding dimension
    metric: str = "cosine"
    namespace: str = ""  # Default namespace


@dataclass
class ClusteringConfig:
    """Configuration for HDBSCAN clustering"""
    min_cluster_size: int = 5
    min_samples: int = 2
    cluster_selection_method: str = "eom"

    # Adaptive sizing (overrides min_cluster_size if dataset is large)
    adaptive_sizing: bool = True
    adaptive_size_ratio: float = 0.02  # 2% of dataset size


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    max_claims_to_process: int = 1000  # -1 for all
    batch_size: int = 100
    parallel_workers: int = 4


class ModelConfig:
    """Main configuration class - loads from env, file, or defaults"""

    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.llm = LLMConfig()
        self.refresh = RefreshConfig()
        self.pinecone = PineconeConfig()
        self.clustering = ClusteringConfig()
        self.processing = ProcessingConfig()

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> 'ModelConfig':
        """
        Load configuration with priority: env vars > config file > defaults

        Args:
            config_file: Optional path to JSON config file

        Returns:
            ModelConfig instance
        """
        config = cls()

        # Step 1: Try to load from JSON file
        if config_file and os.path.exists(config_file):
            config._load_from_file(config_file)
        elif os.path.exists('config.json'):
            config._load_from_file('config.json')

        # Step 2: Override with environment variables (highest priority)
        config._load_from_env()

        return config

    def _load_from_file(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Update embedding config
        if 'embedding' in data:
            for key, value in data['embedding'].items():
                if hasattr(self.embedding, key):
                    setattr(self.embedding, key, value)

        # Update LLM config
        if 'llm' in data:
            for key, value in data['llm'].items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)

        # Update refresh config
        if 'refresh' in data:
            for key, value in data['refresh'].items():
                if hasattr(self.refresh, key):
                    setattr(self.refresh, key, value)

        # Update Pinecone config
        if 'pinecone' in data:
            for key, value in data['pinecone'].items():
                if hasattr(self.pinecone, key):
                    setattr(self.pinecone, key, value)

        # Update clustering config
        if 'clustering' in data:
            for key, value in data['clustering'].items():
                if hasattr(self.clustering, key):
                    setattr(self.clustering, key, value)

        # Update processing config
        if 'processing' in data:
            for key, value in data['processing'].items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Embedding config
        self.embedding.model_id = os.getenv('HF_EMBEDDING_MODEL', self.embedding.model_id)
        self.embedding.dimension = int(os.getenv('HF_EMBEDDING_DIM', self.embedding.dimension))
        self.embedding.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', self.embedding.batch_size))
        self.embedding.api_endpoint = os.getenv('HF_API_ENDPOINT', self.embedding.api_endpoint)

        # LLM config
        self.llm.model_id = os.getenv('CLAUDE_MODEL', self.llm.model_id)
        self.llm.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', self.llm.max_tokens))
        self.llm.temperature = float(os.getenv('CLAUDE_TEMPERATURE', self.llm.temperature))

        # Refresh config
        self.refresh.default_embedding_refresh_hours = int(
            os.getenv('EMBEDDING_REFRESH_HOURS', self.refresh.default_embedding_refresh_hours)
        )
        self.refresh.category_refresh_hours = int(
            os.getenv('CATEGORY_REFRESH_HOURS', self.refresh.category_refresh_hours)
        )
        self.refresh.auto_refresh_enabled = os.getenv('AUTO_REFRESH_ENABLED', 'true').lower() == 'true'

        # Pinecone config
        self.pinecone.index_name = os.getenv('PINECONE_INDEX', self.pinecone.index_name)
        self.pinecone.namespace = os.getenv('PINECONE_NAMESPACE', self.pinecone.namespace)

        # Clustering config
        self.clustering.min_cluster_size = int(os.getenv('MIN_CLUSTER_SIZE', self.clustering.min_cluster_size))
        self.clustering.min_samples = int(os.getenv('MIN_SAMPLES', self.clustering.min_samples))

        # Processing config
        self.processing.max_claims_to_process = int(
            os.getenv('MAX_CLAIMS_TO_PROCESS', self.processing.max_claims_to_process)
        )
        self.processing.batch_size = int(os.getenv('BATCH_SIZE', self.processing.batch_size))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'embedding': asdict(self.embedding),
            'llm': asdict(self.llm),
            'refresh': asdict(self.refresh),
            'pinecone': asdict(self.pinecone),
            'clustering': asdict(self.clustering),
            'processing': asdict(self.processing),
        }

    def save(self, filepath: str = 'config.json'):
        """Save current configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_config(self):
        """Print current configuration"""
        print("=" * 70)
        print("📋 MODEL CONFIGURATION")
        print("=" * 70)

        print("\n🤖 Embedding Model:")
        print(f"   Provider: {self.embedding.provider}")
        print(f"   Model: {self.embedding.model_id}")
        print(f"   Dimension: {self.embedding.dimension}")
        print(f"   Batch size: {self.embedding.batch_size}")

        print("\n🧠 LLM Model:")
        print(f"   Provider: {self.llm.provider}")
        print(f"   Model: {self.llm.model_id}")
        print(f"   Max tokens: {self.llm.max_tokens}")
        print(f"   Temperature: {self.llm.temperature}")

        print("\n🔄 Refresh Configuration:")
        print(f"   Embedding refresh: every {self.refresh.default_embedding_refresh_hours} hours")
        print(f"   Category refresh: every {self.refresh.category_refresh_hours} hours")
        print(f"   Cluster refresh: every {self.refresh.cluster_refresh_hours} hours")
        print(f"   Auto-refresh: {'enabled' if self.refresh.auto_refresh_enabled else 'disabled'}")

        print("\n📊 Pinecone Configuration:")
        print(f"   Index: {self.pinecone.index_name}")
        print(f"   Dimension: {self.pinecone.dimension}")
        print(f"   Metric: {self.pinecone.metric}")

        print("\n🔬 Clustering Configuration:")
        print(f"   Min cluster size: {self.clustering.min_cluster_size}")
        print(f"   Min samples: {self.clustering.min_samples}")
        print(f"   Adaptive sizing: {self.clustering.adaptive_sizing}")

        print("\n⚙️  Processing Configuration:")
        print(f"   Max claims: {'ALL' if self.processing.max_claims_to_process == -1 else self.processing.max_claims_to_process:,}")
        print(f"   Batch size: {self.processing.batch_size}")

        print("\n💡 To override:")
        print("   - Set environment variables (e.g., export CLAUDE_MODEL='claude-opus-4-5-20251101')")
        print("   - Create config.json file in project root")
        print("   - Edit notebooks/config.py defaults")
        print("=" * 70)


# Convenience function for quick access
def get_config(config_file: Optional[str] = None) -> ModelConfig:
    """Get configuration instance"""
    return ModelConfig.load(config_file)


# Example usage
if __name__ == "__main__":
    config = get_config()
    config.print_config()

    # Save example config file
    print("\n📝 Saving example config.json...")
    config.save('config.example.json')
    print("✅ Saved to config.example.json")
