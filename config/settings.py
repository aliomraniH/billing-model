"""Centralized configuration for the medical billing system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Database - Use Vercel Postgres (Neon) for cloud deployment
# Supports both VERCEL_POSTGRES_URL (Deepnote/cloud) and DATABASE_URL (local)
DATABASE_URL = os.getenv(
    "VERCEL_POSTGRES_URL",
    os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/dbname")
)

# Models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "NeuML/bioclinical-modernbert-base-embeddings"
)
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 8192

# Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.70"))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "10"))

# Processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# CMS Data URLs (Public Domain)
CMS_ICD10_URL = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
CMS_HCPCS_URL = "https://www.cms.gov/files/zip/2025-alpha-numeric-hcpcs-file.zip"
NLM_ICD10_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
NLM_HCPCS_API = "https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search"
