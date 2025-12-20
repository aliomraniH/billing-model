# Quick Start Guide: Configuration & Refresh System

This guide walks you through setting up and using the centralized configuration system and timestamp-based refresh functionality.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Environment Setup](#step-1-environment-setup)
3. [Step 2: Database Migration](#step-2-database-migration)
4. [Step 3: Verify Configuration](#step-3-verify-configuration)
5. [Step 4: Run Notebooks](#step-4-run-notebooks)
6. [Step 5: Test Refresh System](#step-5-test-refresh-system)
7. [Configuration Options](#configuration-options)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Ensure you have:
- ✅ Python 3.8+
- ✅ PostgreSQL database (Vercel Postgres)
- ✅ Required API keys (Hugging Face, Pinecone, Anthropic)

---

## Step 1: Environment Setup

### 1.1 Set Required Environment Variables

Create a `.env` file in the project root (or set in your environment):

```bash
# Database
export VERCEL_POSTGRES_URL="postgresql://user:pass@host:5432/db"

# API Keys
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
export PINECONE_API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxx"
```

### 1.2 Load Environment Variables

```bash
# If using .env file
source .env

# Or manually export each variable
export VERCEL_POSTGRES_URL="your_db_url"
export HF_TOKEN="your_hf_token"
# ... etc
```

### 1.3 Verify Environment Variables

```bash
# Quick check
echo $VERCEL_POSTGRES_URL
echo $HF_TOKEN
echo $PINECONE_API_KEY
echo $ANTHROPIC_API_KEY
```

All should print values (not empty).

---

## Step 2: Database Migration

The migration adds timestamp columns and creates the refresh configuration table.

### 2.1 Navigate to Migrations Directory

```bash
cd notebooks/migrations
```

### 2.2 Run the Migration

```bash
python 01_add_refresh_timestamps.py
```

**Expected Output:**
```
======================================================================
🔄 DATABASE MIGRATION: Add Refresh Timestamps
======================================================================

✅ Database URL loaded

📋 Migration Steps:
   1. Add timestamp columns to clinical_notes
   2. Add refresh configuration to claim_categories
   3. Create embedding_refresh_config table
   4. Create indexes for efficient refresh queries
   5. Populate initial timestamp values

======================================================================

STEP 1: Updating clinical_notes table...
   ✅ Added last_embedded_at column
   ✅ Added embedding_version column
   ✅ Added embedding_model column

STEP 2: Updating claim_categories table...
   ✅ Added last_refreshed_at column
   ✅ Added refresh_interval_hours column
   ✅ Added cluster_version column

STEP 3: Creating embedding_refresh_config table...
   ✅ Created embedding_refresh_config table
   ✅ Inserted default refresh configurations

STEP 4: Creating indexes for efficient refresh queries...
   ✅ Created index on clinical_notes.last_embedded_at
   ✅ Created index on claim_categories.last_refreshed_at
   ✅ Created index on embedding_refresh_config.next_run_at

STEP 5: Populating initial values...
   ✅ Updated 2,312 clinical_notes records

======================================================================
✅ MIGRATION COMPLETED SUCCESSFULLY
======================================================================
```

### 2.3 Verify Migration

```bash
cd ../..  # Return to project root
python -c "
from notebooks.config import get_config
from notebooks.refresh_manager import get_refresh_manager

manager = get_refresh_manager(dry_run=True)
manager.print_status()
"
```

**Expected Output:**
```
======================================================================
🔄 REFRESH MANAGER
======================================================================
Mode: DRY RUN (no changes)
======================================================================

📊 Embedding Status:
   Total notes: 2,312
   Embedded notes: 2,312
   Stale embeddings: 0 (or some number)
   Average age: X.X hours

🏷️  Category Status:
   Total categories: 19
   Refreshed categories: 19
   Stale categories: 0

⚙️  Configuration:
   Embedding refresh interval: 12 hours
   Category refresh interval: 24 hours
   Auto-refresh: enabled

📋 Refresh Rules:
   archive_notes: 168h (✅ enabled)
   category_clusters: 24h (✅ enabled)
   critical_notes: 6h (✅ enabled)
   default_embeddings: 12h (✅ enabled)

======================================================================
```

---

## Step 3: Verify Configuration

### 3.1 Check Default Configuration

```bash
python notebooks/config.py
```

**Expected Output:**
```
======================================================================
📋 MODEL CONFIGURATION
======================================================================

🤖 Embedding Model:
   Provider: huggingface
   Model: BAAI/bge-small-en-v1.5
   Dimension: 384
   Batch size: 100

🧠 LLM Model:
   Provider: anthropic
   Model: claude-sonnet-4-5-20250929
   Max tokens: 500
   Temperature: 0.0

🔄 Refresh Configuration:
   Embedding refresh: every 12 hours
   Category refresh: every 24 hours
   Cluster refresh: every 48 hours
   Auto-refresh: enabled

📊 Pinecone Configuration:
   Index: medical-billing-notes
   Dimension: 384
   Metric: cosine

🔬 Clustering Configuration:
   Min cluster size: 5
   Min samples: 2
   Adaptive sizing: True

⚙️  Processing Configuration:
   Max claims: 1,000
   Batch size: 100

💡 To override:
   - Set environment variables (e.g., export CLAUDE_MODEL='claude-opus-4-5-20251101')
   - Create config.json file in project root
   - Edit notebooks/config.py defaults
======================================================================

📝 Saving example config.json...
✅ Saved to config.example.json
```

### 3.2 Test Configuration Overrides (Optional)

#### Via Environment Variables:
```bash
export CLAUDE_MODEL='claude-opus-4-5-20251101'
export EMBEDDING_REFRESH_HOURS=6

python -c "
from notebooks.config import get_config
cfg = get_config()
print(f'Claude Model: {cfg.llm.model_id}')
print(f'Refresh Hours: {cfg.refresh.default_embedding_refresh_hours}')
"
```

**Expected Output:**
```
Claude Model: claude-opus-4-5-20251101
Refresh Hours: 6
```

#### Via JSON File:
```bash
cat > config.json <<EOF
{
  "llm": {
    "model_id": "claude-opus-4-5-20251101",
    "max_tokens": 1000
  },
  "refresh": {
    "default_embedding_refresh_hours": 6
  }
}
EOF

python -c "
from notebooks.config import get_config
cfg = get_config()
cfg.print_config()
"
```

---

## Step 4: Run Notebooks

### 4.1 Run Notebook 5: Embeddings & Similarity Search

```bash
cd notebooks
python 05_embeddings_similarity_search.py
```

**What it does:**
- Loads configuration from `config.py`
- Connects to Vercel Postgres and Pinecone
- Generates embeddings for clinical notes
- Only re-embeds notes that are stale (older than refresh interval)
- Updates `last_embedded_at` timestamps

**Expected Output (abbreviated):**
```
======================================================================
📦 MEDICAL BILLING ML - EMBEDDINGS & SIMILARITY SEARCH (v2.0)
======================================================================
✅ Architecture: Vercel Postgres + Pinecone (hybrid)
✅ HF API: router.huggingface.co (December 2025)
✅ Production-ready: Full dataset processing with validation
======================================================================

✅ All environment variables loaded
   Batch size: 100
   Max claims: 1,000

🔌 Connecting to Vercel Postgres...
   ✅ Connected! Found 10,000 claims in database

🌲 Initializing Pinecone...
   ✅ Connected to Pinecone
   ✅ Index 'medical-billing-notes' ready (384 dimensions)

📝 Generating clinical notes for claims...
   ✅ Generated 2,312 clinical notes

🚀 Starting embedding generation...
   Processing batch 1/24 (100 notes)...
   ✅ Batch 1: 100 embeddings generated
   ...

✅ Embedding generation complete!
   Total notes: 2,312
   Embedded: 2,312
   Skipped (fresh): 1,500
   Time: 45.2s

📊 Similarity search test...
   Query: "diabetes with complications"
   Top 5 similar notes:
   1. [Score: 0.92] Patient with Type 2 diabetes mellitus...
   ...

✅ ALL TESTS PASSED
```

### 4.2 Run Notebook 6: LLM Clustering & Caching

```bash
python 06_llm_clustering_cache.py
```

**What it does:**
- Loads configuration from `config.py`
- Uses dynamic Claude model from config
- Performs HDBSCAN clustering on embeddings
- Uses Claude to generate category names
- Caches results in PostgreSQL
- Updates `last_refreshed_at` timestamps

**Expected Output (abbreviated):**
```
======================================================================
🧠 MEDICAL BILLING ML - LLM CLUSTERING & CACHING (v2.0)
======================================================================
✅ Dynamic Configuration System
✅ Production-ready: Full dataset processing
✅ Cost-optimized: Caching & duplicate prevention
======================================================================

✅ Configuration loaded
   Claude Model: claude-sonnet-4-5-20250929
   Min cluster size: 5 (adaptive)

🔌 Connecting to databases...
   ✅ Vercel Postgres connected
   ✅ Pinecone connected

🔍 Fetching embeddings from Pinecone...
   ✅ Retrieved 2,312 embeddings

🤖 Running HDBSCAN clustering...
   ✅ Found 19 clusters (95% of notes clustered)

💡 Generating category names with Claude...
   Processing cluster 1/19...
   ✅ Category: "Diabetes Management & Complications"
   ...

💾 Saving to database...
   ✅ Saved 19 categories to claim_categories

✅ ALL TESTS PASSED
```

---

## Step 5: Test Refresh System

### 5.1 Check Refresh Status

```bash
python notebooks/refresh_manager.py
```

**Expected Output:**
```
======================================================================
🔄 REFRESH MANAGER
======================================================================
Mode: DRY RUN (no changes)
======================================================================

📊 Embedding Status:
   Total notes: 2,312
   Embedded notes: 2,312
   Stale embeddings: 0
   Average age: 2.5 hours

🏷️  Category Status:
   Total categories: 19
   Refreshed categories: 19
   Stale categories: 0

⚙️  Configuration:
   Embedding refresh interval: 12 hours
   Category refresh interval: 24 hours
   Auto-refresh: enabled

📋 Refresh Rules:
   archive_notes: 168h (✅ enabled)
   category_clusters: 24h (✅ enabled)
   critical_notes: 6h (✅ enabled)
   default_embeddings: 12h (✅ enabled)

======================================================================

🔍 Finding stale embeddings (top 5):
(No stale embeddings found - all are fresh)

🔍 Finding stale categories (top 5):
(No stale categories found - all are fresh)
```

### 5.2 Simulate Stale Data (Optional Testing)

```bash
# Mark some notes as stale by backdating their timestamps
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.begin() as conn:
    # Make 100 notes appear stale (13 hours old)
    conn.execute(text('''
        UPDATE clinical_notes
        SET last_embedded_at = NOW() - INTERVAL '13 hours'
        WHERE note_id IN (
            SELECT note_id FROM clinical_notes
            ORDER BY note_id LIMIT 100
        )
    '''))
    print('✅ Marked 100 notes as stale')
"
```

Now check refresh status again:
```bash
python notebooks/refresh_manager.py
```

**Expected Output:**
```
📊 Embedding Status:
   Total notes: 2,312
   Embedded notes: 2,312
   Stale embeddings: 100  ← Now shows stale notes
   Average age: 6.2 hours

🔍 Finding stale embeddings (top 5):
   1. Note 1 (progress_note): 13.0h old
   2. Note 2 (discharge_summary): 13.0h old
   3. Note 3 (radiology): 13.0h old
   4. Note 4 (progress_note): 13.0h old
   5. Note 5 (discharge_summary): 13.0h old
```

### 5.3 Run Refresh (Dry Run)

The refresh system is automatically integrated into the notebooks. When you run Notebook 5 again, it will:
- Detect stale embeddings
- Re-generate only the stale ones
- Skip fresh embeddings (saves API calls and time)

```bash
python 05_embeddings_similarity_search.py
```

Look for output like:
```
🔄 Checking for stale embeddings...
   Found 100 stale embeddings (older than 12 hours)
   Skipping 2,212 fresh embeddings

🚀 Refreshing stale embeddings...
   Processing batch 1/1 (100 notes)...
   ✅ Refreshed 100 embeddings
   ✅ Updated timestamps
```

---

## Configuration Options

### Option 1: Environment Variables (Recommended for Production)

```bash
# Embedding Model
export HF_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export HF_EMBEDDING_DIM=384
export EMBEDDING_BATCH_SIZE=100

# LLM Model
export CLAUDE_MODEL="claude-sonnet-4-5-20250929"
export CLAUDE_MAX_TOKENS=500
export CLAUDE_TEMPERATURE=0.0

# Refresh Intervals
export EMBEDDING_REFRESH_HOURS=12
export CATEGORY_REFRESH_HOURS=24
export AUTO_REFRESH_ENABLED=true

# Pinecone
export PINECONE_INDEX="medical-billing-notes"

# Clustering
export MIN_CLUSTER_SIZE=5
export MIN_SAMPLES=2

# Processing
export MAX_CLAIMS_TO_PROCESS=1000
export BATCH_SIZE=100
```

### Option 2: JSON Configuration File (Recommended for Development)

Create `config.json` in project root:

```json
{
  "embedding": {
    "model_id": "BAAI/bge-small-en-v1.5",
    "dimension": 384,
    "batch_size": 100
  },
  "llm": {
    "model_id": "claude-opus-4-5-20251101",
    "max_tokens": 1000,
    "temperature": 0.0
  },
  "refresh": {
    "default_embedding_refresh_hours": 6,
    "category_refresh_hours": 12,
    "auto_refresh_enabled": true
  },
  "clustering": {
    "min_cluster_size": 10,
    "adaptive_sizing": true
  }
}
```

Then run notebooks normally - they'll use this config.

### Option 3: Edit Python Defaults (Least Recommended)

Edit `notebooks/config.py` directly and change the default values in the dataclass definitions.

---

## Troubleshooting

### Error: "Missing environment variables"

**Problem:**
```
❌ Missing environment variables:
   - VERCEL_POSTGRES_URL
   - HF_TOKEN
```

**Solution:**
1. Check that variables are set: `echo $VERCEL_POSTGRES_URL`
2. If using `.env` file, run: `source .env`
3. Export variables manually if needed

---

### Error: "column 'last_embedded_at' does not exist"

**Problem:**
```
psycopg2.errors.UndefinedColumn: column "last_embedded_at" does not exist
```

**Solution:**
Run the database migration:
```bash
python notebooks/migrations/01_add_refresh_timestamps.py
```

---

### Error: "column 'embedding' does not exist"

**Problem:**
```
psycopg2.errors.UndefinedColumn: column "embedding" does not exist
```

**Solution:**
This is expected! The system migrated from pgvector (with `embedding` column) to Pinecone. The migration script removes the old `embedding` column. This error means you're using old code - pull the latest changes.

---

### Configuration Not Loading

**Problem:**
Config changes don't take effect.

**Solution:**
1. Check configuration priority: ENV > JSON > Python defaults
2. If using ENV vars, ensure they're exported: `export VAR=value`
3. If using JSON, ensure file is named `config.json` and in project root
4. Verify config is loaded:
```bash
python -c "from notebooks.config import get_config; get_config().print_config()"
```

---

### No Stale Embeddings Found (But There Should Be)

**Problem:**
Refresh manager shows 0 stale embeddings but you expect some.

**Solution:**
1. Check refresh interval: `echo $EMBEDDING_REFRESH_HOURS`
2. Check actual timestamps:
```bash
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    result = conn.execute(text('''
        SELECT
            COUNT(*) as total,
            COUNT(last_embedded_at) as with_timestamp,
            MAX(last_embedded_at) as most_recent,
            MIN(last_embedded_at) as oldest
        FROM clinical_notes
    '''))
    row = result.fetchone()
    print(f'Total: {row[0]}')
    print(f'With timestamp: {row[1]}')
    print(f'Most recent: {row[2]}')
    print(f'Oldest: {row[3]}')
"
```

---

### Notebooks Running Too Slowly

**Problem:**
Embedding generation takes too long.

**Solution:**
1. Reduce batch size: `export MAX_CLAIMS_TO_PROCESS=100`
2. Process in smaller batches: `export BATCH_SIZE=50`
3. The refresh system helps - it skips fresh embeddings on subsequent runs

---

## Next Steps

1. ✅ **Migration Complete** - Database has timestamp columns
2. ✅ **Configuration Working** - Can override models via ENV/JSON
3. ✅ **Notebooks Running** - Using centralized config
4. ✅ **Refresh System Active** - Only processing stale data

### Regular Workflow:

```bash
# Daily/scheduled run
python notebooks/05_embeddings_similarity_search.py  # Only refreshes stale embeddings
python notebooks/06_llm_clustering_cache.py          # Only re-clusters if needed

# Check what needs refresh
python notebooks/refresh_manager.py

# Update models (no code changes!)
export CLAUDE_MODEL="claude-opus-4-5-20251101"
python notebooks/06_llm_clustering_cache.py
```

---

## Summary

✅ **Configuration System** - All models configurable in one place
✅ **Refresh System** - Intelligent staleness tracking
✅ **Cost Optimization** - Skip processing fresh data
✅ **Zero Code Changes** - Update models via ENV/JSON
✅ **Production Ready** - Timestamps, monitoring, validation

For detailed documentation, see:
- `notebooks/CONFIGURATION_AND_REFRESH_SYSTEM.md` - Architecture details
- `notebooks/TESTING_GUIDE.md` - Comprehensive test scenarios
- `notebooks/REGENERATION_GUIDE.md` - Reset and regeneration workflows
