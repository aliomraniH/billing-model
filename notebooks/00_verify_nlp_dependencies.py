"""
Notebook 00: Verify NLP Dependencies
=====================================
Run this FIRST before any other NLP notebooks.
Checks that all required packages are installed in Deepnote.

Time: ~30 seconds
"""

print("🔍 Medical Billing NLP - Dependency Verification")
print("=" * 60)

# Track results
results = []

# 1. Core NLP
print("\n📦 Checking Core NLP packages...")
try:
    import medspacy
    results.append(("medspacy", "✅", medspacy.__version__))
except ImportError as e:
    results.append(("medspacy", "❌", "pip install medspacy"))

try:
    import scispacy
    results.append(("scispacy", "✅", "installed"))
except ImportError:
    results.append(("scispacy", "❌", "pip install scispacy"))

try:
    import spacy
    results.append(("spacy", "✅", spacy.__version__))
except ImportError:
    results.append(("spacy", "❌", "pip install spacy"))

# 2. Embedding Model
print("📦 Checking Embedding packages...")
try:
    from sentence_transformers import SentenceTransformer
    results.append(("sentence-transformers", "✅", "installed"))
except ImportError:
    results.append(("sentence-transformers", "❌", "pip install sentence-transformers"))

try:
    from huggingface_hub import hf_hub_download
    results.append(("huggingface_hub", "✅", "installed"))
except ImportError:
    results.append(("huggingface_hub", "❌", "pip install huggingface_hub"))

# 3. Database
print("📦 Checking Database packages...")
try:
    import psycopg2
    results.append(("psycopg2", "✅", psycopg2.__version__))
except ImportError:
    results.append(("psycopg2", "❌", "pip install psycopg2-binary"))

try:
    from pgvector.psycopg2 import register_vector
    results.append(("pgvector", "✅", "installed"))
except ImportError:
    results.append(("pgvector", "❌", "pip install pgvector"))

try:
    import sqlalchemy
    results.append(("sqlalchemy", "✅", sqlalchemy.__version__))
except ImportError:
    results.append(("sqlalchemy", "❌", "pip install sqlalchemy"))

# 4. De-identification
print("📦 Checking De-identification packages...")
try:
    from presidio_analyzer import AnalyzerEngine
    results.append(("presidio-analyzer", "✅", "installed"))
except ImportError:
    results.append(("presidio-analyzer", "❌", "pip install presidio-analyzer"))

try:
    from presidio_anonymizer import AnonymizerEngine
    results.append(("presidio-anonymizer", "✅", "installed"))
except ImportError:
    results.append(("presidio-anonymizer", "❌", "pip install presidio-anonymizer"))

# 5. spaCy Models
print("📦 Checking spaCy models...")
try:
    nlp = spacy.load("en_core_web_sm")
    results.append(("en_core_web_sm", "✅", "loaded"))
except:
    results.append(("en_core_web_sm", "❌", "python -m spacy download en_core_web_sm"))

try:
    nlp = spacy.load("en_core_sci_md")
    results.append(("en_core_sci_md", "✅", "loaded"))
except:
    results.append(("en_core_sci_md", "❌", "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"))

# 6. Environment Variables
print("📦 Checking Environment Variables...")
import os
db_url = os.getenv("VERCEL_POSTGRES_URL")
if db_url and "neon.tech" in db_url:
    results.append(("VERCEL_POSTGRES_URL", "✅", "configured (Neon)"))
elif db_url:
    results.append(("VERCEL_POSTGRES_URL", "✅", "configured"))
else:
    results.append(("VERCEL_POSTGRES_URL", "❌", "Set in Deepnote Project Settings"))

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    results.append(("HF_TOKEN", "✅", "configured"))
else:
    results.append(("HF_TOKEN", "⚠️", "Optional (for private models)"))

# Print Results
print("\n" + "=" * 60)
print("📋 DEPENDENCY CHECK RESULTS")
print("=" * 60)

missing = []
for name, status, info in results:
    print(f"{status} {name:25} {info}")
    if status == "❌":
        missing.append(info)

print("\n" + "=" * 60)
if missing:
    print("❌ MISSING DEPENDENCIES - Run these commands:")
    print("-" * 60)
    for cmd in missing:
        print(f"  {cmd}")
    print("\nThen re-run this notebook.")
else:
    print("✅ ALL DEPENDENCIES SATISFIED")
    print("   You can proceed to notebook 07_nlp_knowledge_base_setup.py")
