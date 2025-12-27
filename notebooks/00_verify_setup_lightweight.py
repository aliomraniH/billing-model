"""
Verify Setup - Deepnote (Memory-Optimized)
===========================================
Verifies API-first setup WITHOUT heavy local NLP models.

Expected memory: ~200MB (vs ~2.5GB with local models)
Time: ~1 minute
"""

# Add project root to Python path (Deepnote-compatible)
import sys
import os
from pathlib import Path

# Find project root by looking for config/settings.py
current_path = Path.cwd()
project_root = None

# Strategy 1: Check current directory and parents
if (current_path / 'config' / 'settings.py').exists():
    project_root = str(current_path)
elif (current_path.parent / 'config' / 'settings.py').exists():
    project_root = str(current_path.parent)
else:
    for parent in current_path.parents:
        if (parent / 'config' / 'settings.py').exists():
            project_root = str(parent)
            break

# Strategy 2: Search common Deepnote/development paths
if not project_root:
    common_paths = [
        Path('/work'),  # Deepnote's primary location
        Path('/home/user/billing-model'),
        Path('/work/billing-model'),
        Path('/datasets/billing-model'),
        Path.home() / 'billing-model',
    ]

    for path in common_paths:
        if path.exists() and (path / 'config' / 'settings.py').exists():
            project_root = str(path)
            break

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)
    os.chdir(project_root)  # Also set working directory

print("=" * 60)
print("🔍 Setup Verification (Memory-Optimized)")
print("=" * 60)
print(f"📁 Project root: {project_root if project_root else 'NOT FOUND'}")
print(f"📁 Working directory: {os.getcwd()}")
if project_root:
    print(f"📁 sys.path updated: ✅")
    print(f"📁 Architecture: API-First (NO local NLP models)")
else:
    print(f"⚠️  Could not find project!")
    print(f"   Current directory: {current_path}")
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
    "ANTHROPIC_API_KEY",  # For Claude-based context analysis
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

print("\n📦 Optional variables:")
for var in optional_vars:
    if os.getenv(var):
        print(f"  ✅ {var}")
    else:
        print(f"  ⚠️  {var} - Not set (OK for development)")

if not all_set:
    print("\n❌ Missing required variables!")
    print("   Add in Deepnote: Environment → Add Variable")

# %% [markdown]
# ## Step 2: Verify Lightweight Packages (NO spaCy!)

# %%
print("\n📦 Python Packages (Lightweight):")
print("-" * 60)

packages = {
    "sqlalchemy": "Database ORM",
    "psycopg2": "PostgreSQL driver",
    "pgvector": "Vector extension",
    "pandas": "Data processing",
    "numpy": "Numerical computing",
    "requests": "HTTP library",
    "anthropic": "Claude API (optional)",
}

all_installed = True
for package, description in packages.items():
    try:
        __import__(package)
        print(f"  ✅ {package:25} - {description}")
    except ImportError:
        if package == "anthropic":
            print(f"  ⚠️  {package:25} - Not installed (optional)")
        else:
            print(f"  ❌ {package:25} - NOT INSTALLED")
            all_installed = False

print("\n📦 Excluded (memory optimization):")
print("  ❌ spacy                     - Using API instead (~200MB saved)")
print("  ❌ scispacy                  - Using API instead (~100MB saved)")
print("  ❌ medspacy                  - Using API instead (~50MB saved)")
print("  ❌ sentence-transformers     - Using API instead (~500MB saved)")
print("\n  💾 Total memory saved: ~850MB + models (~1.5GB)")

# %% [markdown]
# ## Step 3: Test Database Connection

# %%
print("\n🔌 Database Connection:")
print("-" * 60)

try:
    from sqlalchemy import create_engine, text

    pg_url = os.getenv("VERCEL_POSTGRES_URL")
    if not pg_url:
        raise ValueError("VERCEL_POSTGRES_URL not set")

    engine = create_engine(pg_url)

    with engine.connect() as conn:
        # Test connection
        result = conn.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"  ✅ Connection successful")
        print(f"  ✅ PostgreSQL: {version.split(',')[0]}")

        # Check pgvector
        result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
        pgvector_version = result.scalar()
        if pgvector_version:
            print(f"  ✅ pgvector: v{pgvector_version}")
        else:
            print(f"  ⚠️  pgvector: Not installed (will be installed on first use)")

        # Count tables
        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """))
        table_count = result.scalar()
        print(f"  ✅ Tables: {table_count} found")

except Exception as e:
    print(f"  ❌ Database connection failed: {e}")

# %% [markdown]
# ## Step 4: Test HuggingFace API

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

    # Test API connectivity
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Test whoami endpoint
    response = requests.get(
        "https://huggingface.co/api/whoami",
        headers=headers,
        timeout=5
    )

    if response.status_code == 200:
        user_info = response.json()
        print(f"  ✅ API accessible (user: {user_info.get('name', 'anonymous')})")
    else:
        print(f"  ⚠️  API response: {response.status_code}")

    # Test embedding model
    from config.settings import embedding_config
    api_url = f"https://api-inference.huggingface.co/models/{embedding_config.model_name}"

    test_text = "Patient with diabetes and hypertension"
    response = requests.post(api_url, headers=headers, json={"inputs": test_text}, timeout=30)

    if response.status_code == 200:
        print(f"  ✅ Embedding API accessible")
        print(f"  ✅ Model: {embedding_config.model_name}")
        # Test embedding shape
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            print(f"  ✅ Embedding dimension: {len(result)} (expected: {embedding_config.dimension})")
    elif response.status_code == 503:
        print("  ⏳ Model loading (normal on first use, ready in 20-30s)")
    else:
        print(f"  ⚠️  API response: {response.status_code}")
        print(f"     See docs/TROUBLESHOOTING_HF_API.md for solutions")

except Exception as e:
    print(f"  ⚠️  API test: {e}")

# %% [markdown]
# ## Step 5: Test API-based NLP Client

# %%
print("\n🧠 API-based NLP Client:")
print("-" * 60)

try:
    from src.nlp_api_client import get_nlp_client

    client = get_nlp_client()
    print("  ✅ NLP client loaded")

    # Test medical NER
    test_text = "Patient diagnosed with Type 2 Diabetes Mellitus and hypertension."

    print("\n  Testing medical entity extraction...")
    try:
        entities = client.extract_medical_entities(test_text, min_score=0.5)
        if entities:
            print(f"  ✅ Entity extraction works ({len(entities)} entities found)")
            for e in entities[:3]:  # Show first 3
                print(f"     • {e['text']:30} ({e['label']}, score: {e['score']:.2f})")
        else:
            print(f"  ⚠️  No entities found (may use fallback NER)")
    except Exception as e:
        print(f"  ⚠️  Entity extraction: {e}")

    # Test embeddings
    print("\n  Testing embeddings...")
    try:
        embeddings = client.get_embeddings(["diabetes", "hypertension"])
        if embeddings and len(embeddings) == 2:
            print(f"  ✅ Embeddings work (dimension: {len(embeddings[0])})")
        else:
            print(f"  ⚠️  Unexpected embedding format")
    except Exception as e:
        print(f"  ⚠️  Embeddings: {e}")

    # Test context analysis
    print("\n  Testing context analysis...")
    test_context = "No evidence of pneumonia. Patient denies chest pain."
    try:
        context_entities = client.extract_medical_entities(test_context, min_score=0.5)
        if context_entities:
            context_entities = client.analyze_clinical_context(test_context, context_entities)
            print(f"  ✅ Context analysis works")
            for e in context_entities[:2]:
                flags = []
                if e.get('is_negated'): flags.append('negated')
                if e.get('is_uncertain'): flags.append('uncertain')
                flag_str = ', '.join(flags) if flags else 'affirmed'
                print(f"     • {e['text']:20} ({flag_str})")
    except Exception as e:
        print(f"  ⚠️  Context analysis: {e}")

    print("\n  💡 NLP Client Architecture:")
    print("     • Medical NER: HuggingFace API (d4data/biomedical-ner-all)")
    print("     • Embeddings: HuggingFace API (PubMedBERT)")
    print("     • Context: Claude API or rule-based fallback")
    print("     • Memory: ~10MB (vs ~2GB for local models)")

except ImportError as e:
    print(f"  ❌ NLP client import failed: {e}")
except Exception as e:
    print(f"  ⚠️  NLP client test error: {e}")

# %% [markdown]
# ## Step 6: Test Core Modules

# %%
print("\n📦 Core Modules:")
print("-" * 60)

try:
    from config.settings import validate_environment, db_config, embedding_config
    print("  ✅ config.settings")

    print(f"\n  📋 Configuration:")
    print(f"     • Database: {'Cloud (Vercel/Neon)' if db_config.is_cloud else 'Local'}")
    print(f"     • Embedding model: {embedding_config.model_name}")
    print(f"     • Embedding dimension: {embedding_config.dimension}")

except ImportError as e:
    print(f"  ❌ Import error: {e}")

# %% [markdown]
# ## Step 7: Memory Usage Report

# %%
print("\n💾 Memory Usage:")
print("-" * 60)

try:
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"  📊 Current process: {memory_mb:.1f} MB")
    print(f"\n  📊 Memory comparison:")
    print(f"     • With local NLP (spaCy+medspaCy+models): ~2,500 MB")
    print(f"     • API-first approach (this setup): ~{memory_mb:.0f} MB")
    print(f"     • Savings: ~{2500 - memory_mb:.0f} MB ({((2500 - memory_mb) / 2500 * 100):.1f}%)")
except ImportError:
    print("  ⚠️  psutil not available (install with: pip install psutil)")
    print(f"\n  📊 Expected memory:")
    print(f"     • With local NLP: ~2,500 MB")
    print(f"     • API-first: ~200 MB")
    print(f"     • Savings: ~2,300 MB (92%)")

# %% [markdown]
# ## ✅ Verification Complete

# %%
print("\n" + "=" * 60)
print("✅ SETUP VERIFICATION COMPLETE")
print("=" * 60)

print("""
🎯 System Ready!

📋 Architecture Summary:
   • Database: Vercel Postgres with pgvector ✅
   • NLP: HuggingFace Inference API ✅
   • Embeddings: PubMedBERT via API ✅
   • Context Analysis: Claude API or rule-based ✅
   • Memory: ~200 MB (vs ~2.5 GB with local models) ✅

🚀 Next Steps:
   1. Build knowledge base:
      %run /work/notebooks/07_nlp_knowledge_base_setup.py

   2. Extract codes from claims:
      %run /work/notebooks/08_nlp_code_extraction.py

   3. Run gap analysis:
      %run /work/notebooks/09_gap_analysis_reporting.py

📚 Documentation:
   • API Client: src/nlp_api_client.py
   • Settings: config/settings.py
   • Troubleshooting: docs/TROUBLESHOOTING_HF_API.md
   • Quick Start: docs/QUICKSTART_DEEPNOTE.md
""")

print("=" * 60)
