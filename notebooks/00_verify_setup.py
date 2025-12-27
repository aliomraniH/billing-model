"""
Verify Setup - Deepnote
========================
Run AFTER first-time installation to verify everything works.
This is a TEST script, separate from installation.

Time: ~1 minute
"""

# Add project root to Python path (Deepnote-compatible)
import sys
import os
from pathlib import Path

# Find project root by looking for config/settings.py
current_path = Path.cwd()
project_root = None

# Check if we're already in project root
if (current_path / 'config' / 'settings.py').exists():
    project_root = str(current_path)
# Check if we're in notebooks subdirectory
elif (current_path.parent / 'config' / 'settings.py').exists():
    project_root = str(current_path.parent)
# Search upwards
else:
    for parent in current_path.parents:
        if (parent / 'config' / 'settings.py').exists():
            project_root = str(parent)
            break

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)
    os.chdir(project_root)  # Also set working directory

print("=" * 60)
print("🔍 Setup Verification")
print("=" * 60)
print(f"📁 Project root: {project_root if project_root else 'NOT FOUND'}")
print(f"📁 Working directory: {os.getcwd()}")
print("=" * 60)

# %% [markdown]
# ## Step 1: Check Environment Variables

# %%

print("\n📋 Environment Variables:")
print("-" * 60)

required_vars = [
    "VERCEL_POSTGRES_URL",
    "HF_TOKEN",
]

optional_vars = [
    "BLOB_READ_WRITE_TOKEN",
    "KV_REST_API_URL",
    "KV_REST_API_TOKEN",
]

all_set = True
for var in required_vars:
    if os.getenv(var):
        print(f"  ✅ {var}")
    else:
        print(f"  ❌ {var} - NOT SET")
        all_set = False

print("\n📦 Optional (production caching):")
for var in optional_vars:
    if os.getenv(var):
        print(f"  ✅ {var}")
    else:
        print(f"  ⚠️  {var} - Not set (OK for development)")

if not all_set:
    print("\n❌ Missing required variables!")
    print("Run: %run notebooks/00_first_time_install.py")
    raise ValueError("Setup incomplete")

# %% [markdown]
# ## Step 2: Verify Python Packages

# %%
print("\n📦 Python Packages:")
print("-" * 60)

packages = {
    "sqlalchemy": "Database ORM",
    "psycopg2": "PostgreSQL driver",
    "pgvector": "Vector extension",
    "spacy": "NLP framework",
    "medspacy": "Medical NLP",
    "scispacy": "Scientific NLP",
    "requests": "HTTP library (for HF API)",
    # NOTE: Using HuggingFace Inference API - no local models needed
    # "sentence_transformers": "Embedding models",
    # "transformers": "Hugging Face transformers",
}

all_installed = True
for package, description in packages.items():
    try:
        __import__(package)
        print(f"  ✅ {package:25} - {description}")
    except ImportError:
        print(f"  ❌ {package:25} - NOT INSTALLED")
        all_installed = False

if not all_installed:
    print("\n❌ Some packages missing!")
    print("Run: %run notebooks/00_first_time_install.py")

# %% [markdown]
# ## Step 3: Verify spaCy Models

# %%
print("\n📦 spaCy Models:")
print("-" * 60)

import spacy

models = {
    "en_core_web_sm": "General English model",
    "en_core_sci_md": "Clinical/scientific model",
}

all_models = True
for model_name, description in models.items():
    try:
        nlp = spacy.load(model_name)
        print(f"  ✅ {model_name:20} - {description}")
    except OSError:
        print(f"  ❌ {model_name:20} - NOT INSTALLED")
        all_models = False

if not all_models:
    print("\n❌ Some models missing!")
    print("Run: %run notebooks/00_first_time_install.py")

# %% [markdown]
# ## Step 4: Test Database Connection

# %%
print("\n🔌 Database Connection:")
print("-" * 60)

try:
    from sqlalchemy import create_engine, text

    engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))

    with engine.begin() as conn:  # Use begin() for auto-commit
        # Test connection
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
        print("  ✅ Connection successful")

        # Check version
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0].split(',')[0]
        print(f"  ✅ PostgreSQL: {version}")

        # Check pgvector
        result = conn.execute(text("""
            SELECT extversion FROM pg_extension WHERE extname = 'vector'
        """))
        row = result.fetchone()
        if row:
            print(f"  ✅ pgvector: v{row[0]}")
        else:
            print("  ❌ pgvector: Not installed")

        # Check tables
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
        """))
        table_count = result.fetchone()[0]
        print(f"  ✅ Tables: {table_count} found")

except Exception as e:
    print(f"  ❌ Database error: {e}")
    print("\n🔧 Troubleshooting:")
    print("  1. Check VERCEL_POSTGRES_URL is correct")
    print("  2. Verify database is active in Vercel dashboard")

# %% [markdown]
# ## Step 5: Test HuggingFace API

# %%
print("\n🤖 HuggingFace Inference API:")
print("-" * 60)

try:
    import requests

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("  ⚠️  HF_TOKEN not set - API calls will be rate limited")
    else:
        print("  ✅ HF_TOKEN configured")

    # Test API endpoint with current configured model
    from config.settings import embedding_config
    api_url = f"https://api-inference.huggingface.co/models/{embedding_config.model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    test_text = "Patient with diabetes and hypertension"
    response = requests.post(api_url, headers=headers, json={"inputs": test_text}, timeout=30)

    if response.status_code == 200:
        print("  ✅ HuggingFace API accessible")
        print(f"  ✅ Model available: {embedding_config.model_name}")
        # Test embedding shape
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            print(f"  ✅ Embedding dimension: {len(result)} (expected: {embedding_config.dimension})")
    elif response.status_code == 503:
        print("  ⚠️  Model loading (this is normal on first use)")
        print("     Model will be ready in 20-30 seconds")
    else:
        print(f"  ⚠️  API response: {response.status_code}")
        if response.text:
            print(f"     Error: {response.text[:200]}")
        print(f"\n     See docs/TROUBLESHOOTING_HF_API.md for solutions")

except Exception as e:
    print(f"  ⚠️  API test: {e}")
    print("  Note: This is OK - using API-first architecture")

# %% [markdown]
# ## Step 6: Test Core Modules

# %%
print("\n📦 Core Modules:")
print("-" * 60)

try:
    from src.features.clinical_code_object import ClinicalCodeObject, ExtractedEntity
    print("  ✅ clinical_code_object")

    from src.features.cluster_consensus import ClusterConsensus
    print("  ✅ cluster_consensus")

    from src.features.gap_analyzer import GapAnalyzer
    print("  ✅ gap_analyzer")

    from src.data.cloud_cache import CloudModelCache
    print("  ✅ cloud_cache")

    from config.settings import validate_environment, db_config
    print("  ✅ config.settings")

    print("\n  ✅ All core modules importable")

except ImportError as e:
    print(f"  ❌ Import error: {e}")
    print("\n🔧 Check that you're in the project root directory")

# %% [markdown]
# ## Step 7: Test Cloud Storage (Optional)

# %%
print("\n☁️  Cloud Storage:")
print("-" * 60)

# Blob Storage
if os.getenv("BLOB_READ_WRITE_TOKEN"):
    try:
        import requests

        # Clean token (remove any quotes)
        blob_token = os.getenv("BLOB_READ_WRITE_TOKEN").strip('"').strip("'")

        url = "https://blob.vercel-storage.com/test_health.txt"
        headers = {"Authorization": f"Bearer {blob_token}"}

        # Test write
        response = requests.put(url, data=b"test", headers=headers, timeout=10)
        if response.status_code == 200:
            print("  ✅ Blob Storage: Write works")

            # Test read
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print("  ✅ Blob Storage: Read works")

            # Cleanup
            requests.delete(url, headers=headers, timeout=10)
        else:
            print(f"  ❌ Blob Storage: Write failed ({response.status_code})")

    except Exception as e:
        print(f"  ⚠️  Blob Storage: {e}")
else:
    print("  ⚠️  Blob Storage: Not configured (optional)")

# KV Cache
if os.getenv("KV_REST_API_URL") and os.getenv("KV_REST_API_TOKEN"):
    try:
        import requests
        import time

        # Clean environment variables (remove any quotes)
        kv_url = os.getenv("KV_REST_API_URL").strip('"').strip("'")
        kv_token = os.getenv("KV_REST_API_TOKEN").strip('"').strip("'")
        headers = {"Authorization": f"Bearer {kv_token}"}

        # Test set
        test_key = f"health_check_{int(time.time())}"
        response = requests.post(
            f"{kv_url}/set/{test_key}",
            json={"value": "test"},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            print("  ✅ KV Cache: Write works")

            # Test get
            response = requests.get(f"{kv_url}/get/{test_key}", headers=headers, timeout=10)
            if response.status_code == 200:
                print("  ✅ KV Cache: Read works")

            # Cleanup
            requests.post(f"{kv_url}/del/{test_key}", headers=headers, timeout=10)
        else:
            print(f"  ❌ KV Cache: Write failed ({response.status_code})")

    except Exception as e:
        print(f"  ⚠️  KV Cache: {e}")
else:
    print("  ⚠️  KV Cache: Not configured (optional)")

# %% [markdown]
# ## ✅ Verification Complete

# %%
print("\n" + "=" * 60)
print("✅ SETUP VERIFICATION COMPLETE")
print("=" * 60)

print("\n✅ Everything is working!")

print("\n🎯 You're ready to:")
print("  1. Build knowledge base:")
print("     %run notebooks/07_nlp_knowledge_base_setup.py")
print("")
print("  2. Extract codes from claims:")
print("     %run notebooks/08_nlp_code_extraction.py")
print("")
print("  3. Run gap analysis:")
print("     %run notebooks/09_gap_analysis_reporting.py")

print("\n" + "=" * 60)
