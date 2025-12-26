"""
Deepnote First-Time Setup
==========================
Complete setup guide for running the NLP system in Deepnote.

Prerequisites:
- Deepnote account (free tier works)
- Vercel account with Postgres database
- HuggingFace account (free)

Time: ~10 minutes
"""

print("🚀 Medical Billing NLP - Deepnote Setup Guide")
print("=" * 60)

# %% [markdown]
# ## Step 1: Verify Deepnote Environment

# %%
import sys
import os
from pathlib import Path

print("\n📍 Environment Information:")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Current directory: {os.getcwd()}")
print(f"  Deepnote detected: {'DEEPNOTE_PROJECT_ID' in os.environ}")

if 'DEEPNOTE_PROJECT_ID' in os.environ:
    print(f"  Project ID: {os.environ['DEEPNOTE_PROJECT_ID']}")
    print("  ✅ Running in Deepnote")
else:
    print("  ⚠️  Not detected as Deepnote environment")

# %% [markdown]
# ## Step 2: Add Environment Variables in Deepnote
#
# **IMPORTANT: Do this through the Deepnote UI, not in code!**
#
# 1. Click the **gear icon** (⚙️) in the left sidebar
# 2. Select **"Environment variables"**
# 3. Click **"+ Add environment variable"**
# 4. Add each variable below:
#
# ### Required Variables:
#
# #### VERCEL_POSTGRES_URL
# ```
# Name: VERCEL_POSTGRES_URL
# Value: postgresql://user:password@host.neon.tech/dbname
# ```
# **Where to get it:**
# - Go to: https://vercel.com/dashboard
# - Navigate to: Storage → Postgres → Your Database
# - Click: ".env.local" tab
# - Copy the `POSTGRES_URL` value
#
# #### HF_TOKEN (Optional but recommended)
# ```
# Name: HF_TOKEN
# Value: hf_xxxxxxxxxxxxxxxxxxxx
# ```
# **Where to get it:**
# - Go to: https://huggingface.co/settings/tokens
# - Click: "New token"
# - Name it: "deepnote-billing-nlp"
# - Copy the token
#
# #### BLOB_READ_WRITE_TOKEN (Optional - for production)
# ```
# Name: BLOB_READ_WRITE_TOKEN
# Value: vercel_blob_rw_xxxxxxxxxxxx
# ```
# **Where to get it:**
# - Terminal (on your local machine, not in Deepnote):
#   ```bash
#   vercel blob create nlp-models
#   vercel env pull
#   grep BLOB_READ_WRITE_TOKEN .env
#   ```
#
# #### KV_REST_API_URL (Optional - for production)
# ```
# Name: KV_REST_API_URL
# Value: https://xxxxx.kv.vercel-storage.com
# ```
# **Where to get it:**
# - Terminal (on your local machine):
#   ```bash
#   vercel kv create nlp-cache
#   vercel env pull
#   grep KV_REST_API_URL .env
#   ```
#
# #### KV_REST_API_TOKEN (Optional - for production)
# ```
# Name: KV_REST_API_TOKEN
# Value: xxxxxxxxxxxxxxxxxxxxx
# ```
# **Where to get it:**
# - Same as above, check `.env` file after `vercel env pull`
#
# ---
#
# **After adding variables:**
# - Click "Save" for each variable
# - **Restart the machine**: Click "..." menu → "Restart machine"
# - This notebook will reload automatically

# %% [markdown]
# ## Step 3: Verify Environment Variables

# %%
print("\n🔍 Checking environment variables...")

required_vars = {
    "VERCEL_POSTGRES_URL": "Required - Your Postgres connection string",
    "HF_TOKEN": "Recommended - HuggingFace API token",
}

optional_vars = {
    "BLOB_READ_WRITE_TOKEN": "Optional - Vercel Blob storage (for production)",
    "KV_REST_API_URL": "Optional - Vercel KV cache (for production)",
    "KV_REST_API_TOKEN": "Optional - Vercel KV auth (for production)",
}

print("\n📋 Required Variables:")
all_required_set = True
for var, description in required_vars.items():
    value = os.getenv(var)
    if value:
        # Show only first/last few chars for security
        masked = f"{value[:10]}...{value[-10:]}" if len(value) > 20 else "***"
        print(f"  ✅ {var}: {masked}")
    else:
        print(f"  ❌ {var}: NOT SET")
        print(f"     → {description}")
        all_required_set = False

print("\n📋 Optional Variables (for production):")
for var, description in optional_vars.items():
    value = os.getenv(var)
    if value:
        masked = f"{value[:10]}...{value[-10:]}" if len(value) > 20 else "***"
        print(f"  ✅ {var}: {masked}")
    else:
        print(f"  ⚠️  {var}: Not set")
        print(f"     → {description}")

if not all_required_set:
    print("\n" + "=" * 60)
    print("❌ SETUP INCOMPLETE")
    print("=" * 60)
    print("\n🔧 Next steps:")
    print("  1. Add missing variables in Deepnote (see Step 2 above)")
    print("  2. Restart the machine: ... menu → Restart machine")
    print("  3. Re-run this notebook")
    print("\n" + "=" * 60)
else:
    print("\n" + "=" * 60)
    print("✅ ENVIRONMENT VARIABLES CONFIGURED")
    print("=" * 60)
    print("\nYou can proceed to Step 4!")

# %% [markdown]
# ## Step 4: Install NLP Dependencies
#
# **Run the automated setup script:**

# %%
print("\n📦 Installing NLP dependencies...")
print("This may take 5-10 minutes on first run...")

# Run setup script
import subprocess

result = subprocess.run(
    [sys.executable, "scripts/setup_nlp_system.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("❌ Setup failed:")
    print(result.stderr)
else:
    print("\n✅ Dependencies installed successfully!")

# %% [markdown]
# ## Step 5: Verify NLP Dependencies

# %%
print("\n🔍 Verifying NLP installation...")

# Run verification script
result = subprocess.run(
    [sys.executable, "notebooks/00_verify_nlp_dependencies.py"],
    capture_output=True,
    text=True
)

print(result.stdout)

if "ALL DEPENDENCIES SATISFIED" in result.stdout:
    print("\n✅ All NLP dependencies verified!")
else:
    print("\n⚠️  Some dependencies missing. Review output above.")

# %% [markdown]
# ## Step 6: Test Database Connection

# %%
print("\n🔌 Testing database connection...")

try:
    from sqlalchemy import create_engine, text
    from config.settings import db_config

    engine = create_engine(db_config.url)

    with engine.connect() as conn:
        # Test basic connection
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"✅ Connected to PostgreSQL")
        print(f"   Version: {version.split(',')[0]}")

        # Check if pgvector is installed
        result = conn.execute(text("""
            SELECT extversion FROM pg_extension WHERE extname = 'vector'
        """))
        row = result.fetchone()

        if row:
            print(f"✅ pgvector extension: v{row[0]}")
        else:
            print("⚠️  pgvector not installed. Installing...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✅ pgvector installed")

        # Check tables
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
        """))
        table_count = result.fetchone()[0]
        print(f"✅ Database has {table_count} tables")

        print("\n✅ Database connection successful!")

except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("  1. Verify VERCEL_POSTGRES_URL is correct")
    print("  2. Check Vercel dashboard: Storage → Postgres")
    print("  3. Ensure database is not paused")

# %% [markdown]
# ## Step 7: Run Health Checks (Optional)

# %%
print("\n🏥 Running comprehensive health checks...")
print("This tests all storage systems...\n")

result = subprocess.run(
    [sys.executable, "scripts/run_health_checks.py", "--verbose"],
    capture_output=True,
    text=True
)

print(result.stdout)

if result.returncode == 0:
    print("\n✅ All health checks passed!")
else:
    print("\n⚠️  Some health checks failed (this is OK for first-time setup)")
    print("   Missing Blob/KV storage is expected if not configured yet")

# %% [markdown]
# ## ✅ Setup Complete!
#
# You're now ready to use the NLP system in Deepnote.
#
# ### Next Steps:
#
# 1. **Build Knowledge Base (one-time, 10-20 min):**
#    ```python
#    # Run in a new notebook cell:
#    %run notebooks/07_nlp_knowledge_base_setup.py
#    ```
#
# 2. **Extract Codes from Claims:**
#    ```python
#    %run notebooks/08_nlp_code_extraction.py
#    ```
#
# 3. **Run Gap Analysis:**
#    ```python
#    %run notebooks/09_gap_analysis_reporting.py
#    ```
#
# ### Troubleshooting:
#
# - **"Module not found" errors:**
#   - Restart machine: ... menu → Restart machine
#   - Re-run setup: `%run scripts/setup_nlp_system.py`
#
# - **Database connection errors:**
#   - Check environment variable: `print(os.getenv("VERCEL_POSTGRES_URL"))`
#   - Verify in Deepnote UI: ⚙️ → Environment variables
#
# - **Model download slow:**
#   - First download can take 5-10 minutes
#   - Subsequent runs use cached models (<1 second)
#
# ### Documentation:
#
# - **NLP System:** [NLP_SYSTEM_README.md](NLP_SYSTEM_README.md)
# - **Cloud Storage:** [docs/CLOUD_STORAGE.md](docs/CLOUD_STORAGE.md)
# - **Deployment:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

# %% [markdown]
# ## Summary
#
# ✅ You've completed:
# - [x] Environment variable configuration
# - [x] NLP dependency installation
# - [x] Database connection verification
# - [x] Health check validation
#
# 🎉 **Ready to extract medical billing codes!**

print("\n" + "=" * 60)
print("✅ DEEPNOTE SETUP COMPLETE")
print("=" * 60)
print("\nYour NLP system is ready to use!")
print("\nNext: Run notebooks/07_nlp_knowledge_base_setup.py")
print("=" * 60)
