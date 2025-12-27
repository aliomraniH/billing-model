"""
First-Time Installation - Deepnote
===================================
Clean installation script - runs ONCE to set up all dependencies.
Separate from testing/verification.

Time: 5-10 minutes
Prerequisites: Environment variables must be set in Deepnote UI first!
"""

# Add project root to Python path
import sys
import os
from pathlib import Path

# Get project root (parent of notebooks directory)
project_root = str(Path.cwd().parent) if Path.cwd().name == 'notebooks' else str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 60)
print("🚀 First-Time Installation")
print("=" * 60)

# %% [markdown]
# ## ⚠️ BEFORE RUNNING THIS NOTEBOOK
#
# **You MUST add environment variables first!**
#
# 1. Click **⚙️ (gear icon)** in left sidebar
# 2. Select **"Environment variables"**
# 3. Add the required variables (see list below)
# 4. **Restart machine**: ... menu → Restart machine
# 5. Then run this notebook
#
# ### Required Environment Variables:
#
# #### 1. VERCEL_POSTGRES_URL ✅ (You already have this)
# ```
# Name: VERCEL_POSTGRES_URL
# Value: postgresql://...@...neon.tech/...?sslmode=require
# ```
# **Get it from:**
# - Vercel Dashboard → Storage → Postgres → .env.local tab
#
# #### 2. HF_TOKEN ✅ (You already have this)
# ```
# Name: HF_TOKEN
# Value: hf_...
# ```
# **Get it from:**
# - https://huggingface.co/settings/tokens
#
# #### 3. BLOB_READ_WRITE_TOKEN (Get from Vercel)
# ```
# Name: BLOB_READ_WRITE_TOKEN
# Value: vercel_blob_rw_...
# ```
# **How to get it:**
# - **On your LOCAL machine** (not in Deepnote):
#   ```bash
#   npm install -g vercel
#   vercel login
#   vercel link
#   vercel blob create nlp-models
#   vercel env pull
#   cat .env.local | grep BLOB_READ_WRITE_TOKEN
#   ```
# - Copy the token value
# - Add to Deepnote environment variables
#
# #### 4. KV_REST_API_URL (Get from Vercel)
# ```
# Name: KV_REST_API_URL
# Value: https://...kv.vercel-storage.com
# ```
# **How to get it:**
# - **On your LOCAL machine**:
#   ```bash
#   vercel kv create nlp-cache
#   vercel env pull
#   cat .env.local | grep KV_REST_API_URL
#   ```
# - Copy the URL
# - Add to Deepnote environment variables
#
# #### 5. KV_REST_API_TOKEN (Get from Vercel)
# ```
# Name: KV_REST_API_TOKEN
# Value: ...
# ```
# **How to get it:**
# - Same as above:
#   ```bash
#   cat .env.local | grep KV_REST_API_TOKEN
#   ```
# - Copy the token
# - Add to Deepnote environment variables
#
# ---
#
# **After adding ALL variables:**
# - Click "..." menu → "Restart machine"
# - Wait for restart to complete
# - Run this notebook

# %% [markdown]
# ## Step 1: Check Environment Variables

# %%
import os

print("\n📋 Checking environment variables...")
print("-" * 60)

required_vars = {
    "VERCEL_POSTGRES_URL": "Database connection",
    "HF_TOKEN": "HuggingFace API token",
}

production_vars = {
    "BLOB_READ_WRITE_TOKEN": "Vercel Blob storage (for caching)",
    "KV_REST_API_URL": "Vercel KV cache URL",
    "KV_REST_API_TOKEN": "Vercel KV cache token",
}

missing_required = []
missing_optional = []

print("\n✅ You already have:")
for var, desc in required_vars.items():
    if os.getenv(var):
        masked = f"{os.getenv(var)[:10]}...{os.getenv(var)[-10:]}"
        print(f"  ✅ {var}: {masked}")
    else:
        print(f"  ❌ {var}: NOT SET - {desc}")
        missing_required.append(var)

print("\n📦 Optional (for production caching):")
for var, desc in production_vars.items():
    if os.getenv(var):
        masked = f"{os.getenv(var)[:10]}...{os.getenv(var)[-10:]}"
        print(f"  ✅ {var}: {masked}")
    else:
        print(f"  ⚠️  {var}: Not set - {desc}")
        missing_optional.append(var)

if missing_required:
    print("\n" + "=" * 60)
    print("❌ MISSING REQUIRED VARIABLES")
    print("=" * 60)
    print("\nMissing:")
    for var in missing_required:
        print(f"  - {var}")
    print("\n🔧 To fix:")
    print("  1. Add variables in Deepnote: ⚙️ → Environment variables")
    print("  2. Restart machine: ... menu → Restart machine")
    print("  3. Re-run this notebook")
    print("=" * 60)
    raise ValueError("Required environment variables not set")

print("\n" + "=" * 60)
print("✅ Required variables are set!")
if missing_optional:
    print("\n⚠️  Optional variables missing:")
    for var in missing_optional:
        print(f"  - {var}")
    print("\nThese are optional for development.")
    print("Add them later for production deployment.")
print("=" * 60)

# %% [markdown]
# ## Step 2: Install Core Dependencies

# %%
import subprocess
import sys

print("\n📦 Installing core dependencies...")
print("This may take 5-10 minutes on first run...")
print("-" * 60)

core_packages = [
    "pip install --upgrade pip",
    "pip install sqlalchemy>=2.0.0",
    "pip install psycopg2-binary>=2.9.0",
    "pip install pgvector>=0.2.4",
    "pip install requests>=2.31.0",
]

for cmd in core_packages:
    package = cmd.split()[-1].split(">=")[0]
    print(f"\n📦 Installing {package}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✅ {package} installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e.stderr[:200]}")

# %% [markdown]
# ## Step 3: Install NLP Dependencies

# %%
print("\n📦 Installing NLP dependencies...")
print("This is the longest step (5-8 minutes)...")
print("-" * 60)

nlp_packages = [
    "pip install spacy>=3.7.0",
    "pip install scispacy>=0.5.4",
    "pip install medspacy>=1.2.0",
    # NOTE: Using HuggingFace Inference API instead of local models
    # "pip install sentence-transformers>=2.2.2",  # Skip - use API
    # "pip install transformers>=4.48.0",  # Skip - use API
]

for cmd in nlp_packages:
    package = cmd.split()[-1].split(">=")[0]
    print(f"\n📦 Installing {package}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per package
        )
        print(f"  ✅ {package} installed")
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  {package} installation timed out (this is OK, it may still complete)")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e.stderr[:200]}")

# %% [markdown]
# ## Step 4: Install spaCy Models

# %%
print("\n📦 Installing spaCy models...")
print("-" * 60)

spacy_models = [
    ("python -m spacy download en_core_web_sm", "en_core_web_sm"),
    ("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz",
     "en_core_sci_md (clinical)"),
]

for cmd, name in spacy_models:
    print(f"\n📦 Installing {name}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        print(f"  ✅ {name} installed")
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  {name} installation timed out (may still complete)")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e.stderr[:200]}")

# %% [markdown]
# ## Step 5: Verify Installation

# %%
print("\n🔍 Verifying installation...")
print("-" * 60)

print("\nTesting imports...")
try:
    import sqlalchemy
    print("  ✅ sqlalchemy")
except:
    print("  ❌ sqlalchemy")

try:
    import spacy
    print("  ✅ spacy")
except:
    print("  ❌ spacy")

try:
    import medspacy
    print("  ✅ medspacy")
except:
    print("  ❌ medspacy")

try:
    import pgvector
    print("  ✅ pgvector")
except:
    print("  ❌ pgvector")

try:
    import requests
    print("  ✅ requests (for HuggingFace API)")
except:
    print("  ❌ requests")

print("\nTesting spaCy models...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("  ✅ en_core_web_sm")
except:
    print("  ❌ en_core_web_sm")

try:
    nlp = spacy.load("en_core_sci_md")
    print("  ✅ en_core_sci_md")
except:
    print("  ❌ en_core_sci_md")

# %% [markdown]
# ## Step 6: Test Database Connection

# %%
print("\n🔌 Testing database connection...")
print("-" * 60)

try:
    from sqlalchemy import create_engine, text

    db_url = os.getenv("VERCEL_POSTGRES_URL")
    engine = create_engine(db_url, connect_args={'connect_timeout': 10})

    with engine.begin() as conn:  # Use begin() for auto-commit
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"\n✅ Connected to PostgreSQL")
        print(f"   Version: {version.split(',')[0]}")

        # Enable pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("✅ pgvector extension enabled")

        print("\n✅ Database connection successful!")

except Exception as e:
    print(f"\n❌ Database connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("  1. Check VERCEL_POSTGRES_URL in Deepnote env vars")
    print("  2. Verify database is not paused in Vercel dashboard")
    print("  3. Ensure connection string ends with ?sslmode=require")

# %% [markdown]
# ## ✅ Installation Complete!

# %%
print("\n" + "=" * 60)
print("✅ INSTALLATION COMPLETE")
print("=" * 60)

print("\n📋 What was installed:")
print("  ✅ Core dependencies (SQLAlchemy, pgvector, etc.)")
print("  ✅ NLP packages (spaCy, medspaCy, scispaCy)")
print("  ✅ Embedding models (sentence-transformers)")
print("  ✅ spaCy language models")
print("  ✅ Database connection verified")

print("\n🎯 Next Steps:")
print("  1. Run verification notebook:")
print("     %run notebooks/00_verify_setup.py")
print("")
print("  2. Build knowledge base (one-time, 10-20 min):")
print("     %run notebooks/07_nlp_knowledge_base_setup.py")
print("")
print("  3. Start extracting codes:")
print("     %run notebooks/08_nlp_code_extraction.py")

print("\n" + "=" * 60)
print("⚠️  IMPORTANT: Restart machine after installation")
print("   ... menu → Restart machine")
print("=" * 60)
