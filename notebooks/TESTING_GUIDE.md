# Step-by-Step Testing Guide for Updated Notebooks

## 🎯 Overview

This guide walks you through testing the new dynamic configuration system and timestamp-based refresh functionality.

**What's New**:
- ✅ No hardcoded models (everything configurable)
- ✅ Automatic refresh based on timestamps
- ✅ Skip fresh embeddings (save API calls)
- ✅ Intelligent staleness detection

**Estimated Time**: 30-45 minutes

---

## 📋 Pre-Testing Checklist

Before you begin, ensure you have:
- [ ] PostgreSQL database with existing claims data
- [ ] `VERCEL_POSTGRES_URL` environment variable set
- [ ] `HF_TOKEN` environment variable set (HuggingFace)
- [ ] `PINECONE_API_KEY` environment variable set
- [ ] `ANTHROPIC_API_KEY` environment variable set (optional)
- [ ] Python 3.8+ installed
- [ ] Latest code pulled from git

---

## 🚀 Step-by-Step Testing

### **STEP 1: Pull Latest Code**

```bash
# Navigate to project directory
cd /home/user/billing-model

# Fetch latest changes
git fetch origin

# Switch to the feature branch
git checkout claude/review-notebook-6-clustering-D8RjT

# Pull latest commits
git pull origin claude/review-notebook-6-clustering-D8RjT
```

**Verify**:
```bash
# Check you have the latest commit
git log --oneline -1
```

**Expected output**:
```
a53f1e1 Add dynamic model config & timestamp-based refresh system
```

**Verify new files exist**:
```bash
ls -la notebooks/config.py
ls -la notebooks/refresh_manager.py
ls -la notebooks/migrations/01_add_refresh_timestamps.py
```

✅ **Success criteria**: All files exist, commit hash matches

---

### **STEP 2: Install/Update Dependencies**

```bash
# Install required packages
pip install -U sqlalchemy psycopg2-binary pandas numpy
pip install -U pinecone huggingface_hub anthropic
pip install -U scikit-learn>=1.3.0
```

**Verify installation**:
```bash
python -c "import sqlalchemy; print(f'SQLAlchemy: {sqlalchemy.__version__}')"
python -c "import pinecone; print('Pinecone: OK')"
python -c "import anthropic; print('Anthropic: OK')"
python -c "from sklearn.cluster import HDBSCAN; print('HDBSCAN: OK')"
```

✅ **Success criteria**: All imports succeed, no errors

---

### **STEP 3: Run Database Migration**

This adds timestamp columns for the refresh system.

```bash
cd notebooks
python migrations/01_add_refresh_timestamps.py
```

**Expected output**:
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

STEP 5: Populating initial timestamp values...
   ✅ Set last_embedded_at for X existing notes
   ✅ Set last_refreshed_at for Y categories

======================================================================
✅ MIGRATION COMPLETE
======================================================================

clinical_notes:
   Total notes: 15,000
   With embeddings: 0
   With timestamps: 0

claim_categories:
   Total categories: 0
   With timestamps: 0

embedding_refresh_config:
   default_embeddings: 12h (✅ enabled)
   category_clusters: 24h (✅ enabled)
   critical_notes: 6h (✅ enabled)
   archive_notes: 168h (✅ enabled)

======================================================================
✅ MIGRATION COMPLETE
======================================================================
```

**Verify migration**:
```bash
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    # Check new columns exist
    result = conn.execute(text('''
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'clinical_notes'
        AND column_name IN ('last_embedded_at', 'embedding_model', 'embedding_version')
    '''))
    columns = [r[0] for r in result.fetchall()]
    print(f'New columns in clinical_notes: {columns}')

    # Check new table exists
    result = conn.execute(text('''
        SELECT COUNT(*) FROM embedding_refresh_config
    '''))
    count = result.fetchone()[0]
    print(f'Refresh config entries: {count}')
"
```

**Expected output**:
```
New columns in clinical_notes: ['last_embedded_at', 'embedding_model', 'embedding_version']
Refresh config entries: 4
```

✅ **Success criteria**:
- Migration completes without errors
- New columns exist in database
- Refresh config table has 4 entries

**⚠️ If migration fails**:
- Check database connection
- Ensure no other processes are using the database
- Check PostgreSQL logs for errors

---

### **STEP 4: Test Configuration System**

```bash
# Test default configuration
python -c "from config import get_config; get_config().print_config()"
```

**Expected output**:
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
   - Set environment variables
   - Create config.json file
   - Edit notebooks/config.py defaults
======================================================================
```

**Test environment variable override**:
```bash
# Override Claude model
export CLAUDE_MODEL='claude-opus-4-5-20251101'
python -c "from config import get_config; print(f\"LLM Model: {get_config().llm.model_id}\")"
```

**Expected output**:
```
LLM Model: claude-opus-4-5-20251101
```

**Test JSON config override**:
```bash
# Create test config
cat > test-config.json <<EOF
{
  "llm": {
    "model_id": "test-model-from-json"
  },
  "refresh": {
    "default_embedding_refresh_hours": 6
  }
}
EOF

# Load and verify
python -c "from config import get_config; cfg = get_config('test-config.json'); print(f\"LLM: {cfg.llm.model_id}, Refresh: {cfg.refresh.default_embedding_refresh_hours}h\")"

# Clean up
rm test-config.json
```

**Expected output**:
```
LLM: test-model-from-json, Refresh: 6h
```

✅ **Success criteria**:
- Default config prints correctly
- Environment variables override defaults
- JSON config overrides defaults
- Priority order works: ENV > JSON > defaults

---

### **STEP 5: Test Notebook 5 (Embeddings) - Small Dataset**

**Set test configuration**:
```bash
# Use small dataset for testing
export MAX_CLAIMS_TO_PROCESS=100
export EMBEDDING_BATCH_SIZE=25
export AUTO_REFRESH_ENABLED=true
export EMBEDDING_REFRESH_HOURS=12

# Verify settings
echo "Max claims: $MAX_CLAIMS_TO_PROCESS"
echo "Batch size: $EMBEDDING_BATCH_SIZE"
echo "Auto refresh: $AUTO_REFRESH_ENABLED"
```

**Run Notebook 5**:
```bash
cd notebooks
python 05_embeddings_similarity_search.py
```

**Expected output (key sections)**:

1. **Configuration Loading**:
```
======================================================================
📋 MODEL CONFIGURATION
======================================================================

🤖 Embedding Model:
   Provider: huggingface
   Model: BAAI/bge-small-en-v1.5
   Dimension: 384
   Batch size: 25

🔄 Refresh Configuration:
   Embedding refresh: every 12 hours
   Auto-refresh: enabled
======================================================================
```

2. **Database Connection**:
```
🔌 Connecting to Vercel Postgres...
   ✅ Connected to database
   ✅ Total claims in database: 15,000
   Will process 100 claims
```

3. **Batch Processing**:
```
🔄 Processing claims in batches...

   Batch 1/4 (claims 1-25)...
   ✅ Uploaded 25 vectors to Pinecone
   📊 Progress: 25/100 (X.X claims/sec, ETA: X.Xmin)

   Batch 2/4 (claims 26-50)...
   ✅ Uploaded 25 vectors to Pinecone
   📊 Progress: 50/100 (X.X claims/sec, ETA: X.Xmin)

   ...
```

4. **Statistics**:
```
📊 PROCESSING STATISTICS
======================================================================
   Total processed: 100
   Notes created: 100
   Embeddings created: 100
   Errors: 0
   Duration: XX.Xs (X.X claims/sec)
======================================================================
```

5. **Data Quality Validation**:
```
🔍 DATA QUALITY VALIDATION
----------------------------------------------------------------------
   Vector coverage: 100/15,000 (0.7%)
   ✅ Embedding dimensions validated: 384
   ✅ No zero vectors found
   ⚠️ WARNING: Found ~8-12 duplicate vectors  ← Should be <15%, not 64%!

   📊 Vector Statistics:
      • Mean L2 norm: 1.000
      • Std L2 norm: 0.000
```

**Verify in database**:
```bash
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    # Check embeddings were timestamped
    result = conn.execute(text('''
        SELECT
            COUNT(*) as total,
            COUNT(last_embedded_at) as timestamped,
            COUNT(DISTINCT embedding_model) as models
        FROM clinical_notes
        WHERE last_embedded_at IS NOT NULL
    '''))
    row = result.fetchone()
    print(f'Total notes: {row[0]}')
    print(f'With timestamps: {row[1]}')
    print(f'Unique models: {row[2]}')

    # Show sample
    result = conn.execute(text('''
        SELECT claim_id, embedding_model, last_embedded_at
        FROM clinical_notes
        WHERE last_embedded_at IS NOT NULL
        LIMIT 3
    '''))
    print('\nSample entries:')
    for row in result:
        print(f'  Claim {row[0]}: {row[1]}, embedded at {row[2]}')
"
```

**Expected output**:
```
Total notes: 100
With timestamps: 100
Unique models: 1

Sample entries:
  Claim 1: BAAI/bge-small-en-v1.5, embedded at 2025-12-18 10:30:45+00:00
  Claim 2: BAAI/bge-small-en-v1.5, embedded at 2025-12-18 10:30:46+00:00
  Claim 3: BAAI/bge-small-en-v1.5, embedded at 2025-12-18 10:30:47+00:00
```

✅ **Success criteria**:
- Configuration loads correctly
- 100 embeddings created
- Duplicates < 15% (should be ~8-12 duplicates out of 100)
- All notes have timestamps
- All notes have embedding_model set
- No errors during processing

**⚠️ If duplicates are still ~64%**:
This means old data is still present. Run the reset script:
```bash
python 00_reset_and_regenerate.py
```

---

### **STEP 6: Test Auto-Refresh (Skip Fresh Embeddings)**

Now test that fresh embeddings are skipped.

```bash
# Run Notebook 5 AGAIN immediately
python 05_embeddings_similarity_search.py
```

**Expected output (key difference)**:
```
   Batch 1/4 (claims 1-25)...
   ⏭️  Skipped 25 fresh embeddings (< 12 hours old)
   ✅ Uploaded 0 vectors to Pinecone

   Batch 2/4 (claims 26-50)...
   ⏭️  Skipped 25 fresh embeddings (< 12 hours old)
   ✅ Uploaded 0 vectors to Pinecone

📊 PROCESSING STATISTICS
======================================================================
   Total processed: 100
   Notes created: 0        ← Skipped all!
   Embeddings created: 0   ← Skipped all!
   Fresh skipped: 100      ← All were fresh!
   Errors: 0
======================================================================
```

**Verify skipping logic**:
```bash
python -c "
from sqlalchemy import create_engine, text
from datetime import datetime
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    result = conn.execute(text('''
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (
                WHERE last_embedded_at > NOW() - INTERVAL '12 hours'
            ) as fresh,
            COUNT(*) FILTER (
                WHERE last_embedded_at <= NOW() - INTERVAL '12 hours'
            ) as stale
        FROM clinical_notes
        WHERE last_embedded_at IS NOT NULL
    '''))
    row = result.fetchone()
    print(f'Total: {row[0]}')
    print(f'Fresh (<12h): {row[1]}')
    print(f'Stale (>12h): {row[2]}')
"
```

**Expected output**:
```
Total: 100
Fresh (<12h): 100
Stale (>12h): 0
```

✅ **Success criteria**:
- Second run skips all 100 embeddings
- No API calls made (saves costs!)
- Timestamps indicate all are fresh

---

### **STEP 7: Test Forced Refresh (Disable Auto-Refresh)**

```bash
# Disable auto-refresh to force regeneration
export AUTO_REFRESH_ENABLED=false

# Run again
python 05_embeddings_similarity_search.py
```

**Expected output**:
```
🔄 Refresh Configuration:
   Auto-refresh: disabled  ← Changed!

   Batch 1/4 (claims 1-25)...
   ✅ Uploaded 25 vectors to Pinecone  ← Regenerated despite being fresh!

📊 PROCESSING STATISTICS
======================================================================
   Total processed: 100
   Notes created: 100      ← All regenerated!
   Embeddings created: 100 ← All regenerated!
======================================================================
```

**Re-enable auto-refresh**:
```bash
export AUTO_REFRESH_ENABLED=true
```

✅ **Success criteria**:
- With AUTO_REFRESH=false, all embeddings regenerate
- With AUTO_REFRESH=true, fresh embeddings are skipped

---

### **STEP 8: Test Notebook 6 (Clustering) - Small Dataset**

```bash
# Ensure we have embeddings first
export MAX_CLAIMS_TO_PROCESS=100
python 05_embeddings_similarity_search.py

# Now run clustering
python 06_llm_clustering_cache.py
```

**Expected output (key sections)**:

1. **Configuration Loading**:
```
======================================================================
📋 MODEL CONFIGURATION
======================================================================

🧠 LLM Model:
   Provider: anthropic
   Model: claude-sonnet-4-5-20250929
   Max tokens: 500
   Temperature: 0.0

🔬 Clustering Configuration:
   Min cluster size: 5
   Min samples: 2
   Adaptive sizing: True
======================================================================
```

2. **Clustering Results**:
```
🔬 Clustering with HDBSCAN...
   Parameters: min_cluster_size=5, min_samples=2
   ✅ Found 5-8 clusters  ← Should be 5-8, not 19 or 38!
   ⚠️ Noise points: ~10-15 (~10-15%)
   ✅ Centroids shape: (5-8, 384)

📊 CLUSTERING QUALITY METRICS
----------------------------------------------------------------------
   • Silhouette Score: 0.35-0.50  ← Should be >0.35, not 0.249!
   • Davies-Bouldin Index: <1.5
   • Calinski-Harabasz Score: >100
```

3. **LLM Labeling** (if ANTHROPIC_API_KEY is set):
```
🤖 Labeling clusters with LLM...
   [1/6] Labeling cluster 0... 'Diabetes Management' (18 items)
   [2/6] Labeling cluster 1... 'Cardiac Procedures' (22 items)
   ...
   ✅ Labeled 6 categories

🔧 Checking for duplicate category names...
   ✅ No duplicate category names found
```

4. **Metadata Verification**:
```
🔄 Updating Pinecone metadata with new categories...
   ✅ Updated 90 vectors with new categories
   ✅ Metadata update verified: 'diabetes_management'  ← Should pass now!
```

**Verify in database**:
```bash
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    # Check categories
    result = conn.execute(text('''
        SELECT
            category_name,
            claim_count,
            last_refreshed_at,
            refresh_interval_hours
        FROM claim_categories
        ORDER BY claim_count DESC
        LIMIT 5
    '''))
    print('Top categories:')
    for row in result:
        print(f'  {row[0]}: {row[1]} claims, interval: {row[3]}h')
"
```

**Expected output**:
```
Top categories:
  diabetes_management: 25 claims, interval: 24h
  cardiac_procedures: 22 claims, interval: 24h
  respiratory_conditions: 18 claims, interval: 24h
  ...
```

✅ **Success criteria**:
- Clustering creates 5-8 clusters (not over-clustering)
- Silhouette score > 0.35 (better quality)
- No duplicate category names
- Metadata verification passes
- Categories have timestamps and refresh intervals

---

### **STEP 9: Test Refresh Manager**

```bash
# Check refresh status
python refresh_manager.py --check
```

**Expected output**:
```
======================================================================
🔄 REFRESH MANAGER
======================================================================
Mode: DRY RUN (no changes)
======================================================================

🔍 CHECKING REFRESH STATUS

📊 Embeddings:
   Total embedded notes: 100
   Stale (need refresh): 0 (0.0%)
   Threshold: Older than 12 hours

📊 Categories:
   Total categories: 6
   Stale (need re-clustering): 0 (0.0%)
   Threshold: Older than 24 hours

======================================================================
```

**Simulate stale data (lower threshold)**:
```bash
# Change threshold to 0 hours (everything is stale)
export EMBEDDING_REFRESH_HOURS=0

# Check again
python refresh_manager.py --check
```

**Expected output**:
```
📊 Embeddings:
   Total embedded notes: 100
   Stale (need refresh): 100 (100.0%)  ← All stale now!
   Threshold: Older than 0 hours

   Oldest 5 stale embeddings:
      [1] Note 1234: 0.5h old
      [2] Note 1235: 0.5h old
      ...
```

**Test dry-run refresh**:
```bash
python refresh_manager.py --refresh-embeddings --max-refresh 10 --dry-run
```

**Expected output**:
```
🔄 AUTO-REFRESHING STALE EMBEDDINGS

   Found 100 stale embeddings

   Batch 1/1 (10 notes)...
   [DRY RUN] Would mark 10 notes for re-embedding

✅ Refresh complete:
   Total marked for refresh: 10
```

**Test actual refresh**:
```bash
# Restore normal threshold
export EMBEDDING_REFRESH_HOURS=12

# Force some embeddings to be stale (update timestamp)
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    # Make 5 embeddings appear 13 hours old
    conn.execute(text('''
        UPDATE clinical_notes
        SET last_embedded_at = NOW() - INTERVAL '13 hours'
        WHERE claim_id IN (
            SELECT claim_id FROM clinical_notes LIMIT 5
        )
    '''))
    conn.commit()
    print('✅ Made 5 embeddings stale (13 hours old)')
"

# Check status
python refresh_manager.py --check

# Refresh them
python refresh_manager.py --refresh-embeddings --max-refresh 5
```

**Expected output**:
```
📊 Embeddings:
   Total embedded notes: 100
   Stale (need refresh): 5 (5.0%)

🔄 AUTO-REFRESHING STALE EMBEDDINGS

   Found 5 stale embeddings

   Batch 1/1 (5 notes)...
   ✅ Marked 5 notes for re-embedding

✅ Refresh complete:
   Total marked for refresh: 5
```

✅ **Success criteria**:
- Refresh manager detects stale data
- Dry-run mode works correctly
- Actual refresh marks notes for re-embedding
- Thresholds are configurable

---

### **STEP 10: Test Model Change Detection**

```bash
# Change embedding model
export HF_EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
export HF_EMBEDDING_DIM=384
export MAX_CLAIMS_TO_PROCESS=10

# Run Notebook 5
python 05_embeddings_similarity_search.py
```

**Expected behavior**:
```
🤖 Embedding Model:
   Model: sentence-transformers/all-MiniLM-L6-v2  ← New model!

   Batch 1/1 (claims 1-10)...
   🔄 Model changed (was: BAAI/bge-small-en-v1.5)
   ✅ Regenerating embeddings with new model
   ✅ Uploaded 10 vectors to Pinecone

📊 PROCESSING STATISTICS
======================================================================
   Total processed: 10
   Notes created: 10
   Embeddings created: 10  ← All regenerated despite being fresh!
   Model changes: 10       ← Detected model change!
======================================================================
```

**Restore original model**:
```bash
export HF_EMBEDDING_MODEL='BAAI/bge-small-en-v1.5'
```

✅ **Success criteria**:
- Model change is detected
- Fresh embeddings are regenerated when model changes
- New model name is stored in database

---

### **STEP 11: Test Large Dataset (Optional)**

**⚠️ WARNING**: This will make many API calls and may take 15-30 minutes.

```bash
# Process 1,000 claims
export MAX_CLAIMS_TO_PROCESS=1000
export EMBEDDING_BATCH_SIZE=100

# Run Notebook 5
python 05_embeddings_similarity_search.py
```

**Monitor for**:
- Batch processing working correctly
- Progress updates showing ETA
- Duplicate rate < 10% (not 64%)
- No API rate limit errors
- All embeddings get timestamps

**Run Notebook 6**:
```bash
python 06_llm_clustering_cache.py
```

**Monitor for**:
- 10-15 clusters (not 38)
- Silhouette score > 0.35
- No duplicate category names
- Metadata verification passes

✅ **Success criteria**:
- 1,000 embeddings created successfully
- Duplicate vectors < 10%
- Clustering quality improved
- All features working at scale

---

## 📊 Verification Checklist

After testing, verify all features work:

### Configuration System
- [ ] Default config loads correctly
- [ ] Environment variables override defaults
- [ ] JSON config overrides defaults
- [ ] Priority order: ENV > JSON > defaults
- [ ] config.print_config() shows current settings

### Timestamp System
- [ ] Database migration completed
- [ ] New columns exist (last_embedded_at, embedding_model, etc.)
- [ ] Timestamps are set when embeddings created
- [ ] Model name is stored in database

### Auto-Refresh System
- [ ] Fresh embeddings are skipped (age < threshold)
- [ ] Stale embeddings are regenerated (age > threshold)
- [ ] Model change detection works
- [ ] AUTO_REFRESH_ENABLED toggle works

### Refresh Manager
- [ ] `--check` shows correct stale count
- [ ] `--dry-run` doesn't make changes
- [ ] `--refresh-embeddings` marks notes for refresh
- [ ] `--refresh-categories` updates timestamps
- [ ] `--refresh-all` works

### Notebook Improvements
- [ ] Duplicate vectors < 10% (not 64%)
- [ ] Clustering creates 10-15 clusters (not 38)
- [ ] Silhouette score > 0.35 (not 0.249)
- [ ] No duplicate category names
- [ ] Metadata verification passes

---

## 🐛 Troubleshooting

### Issue: Migration fails with "column already exists"

**Solution**: Migration is idempotent, just continue
```bash
# Migration will skip existing columns
python migrations/01_add_refresh_timestamps.py
```

### Issue: "Column 'last_embedded_at' does not exist"

**Solution**: Run migration again
```bash
python migrations/01_add_refresh_timestamps.py
```

### Issue: Duplicates still ~64%

**Solution**: Old data still present, reset and regenerate
```bash
python 00_reset_and_regenerate.py
export MAX_CLAIMS_TO_PROCESS=100
python 05_embeddings_similarity_search.py
```

### Issue: "Config module not found"

**Solution**: Ensure you're in notebooks directory
```bash
cd notebooks
python 05_embeddings_similarity_search.py
```

### Issue: All embeddings regenerating every time

**Solution**: Check AUTO_REFRESH setting
```bash
export AUTO_REFRESH_ENABLED=true
export EMBEDDING_REFRESH_HOURS=12
```

### Issue: Refresh manager shows 0 stale

**Solution**: Embeddings are actually fresh, increase threshold to test
```bash
export EMBEDDING_REFRESH_HOURS=0  # Everything is stale
python refresh_manager.py --check
export EMBEDDING_REFRESH_HOURS=12  # Restore normal
```

### Issue: Metadata verification still failing

**Solution**: Check Pinecone indexing delay
```bash
# Increase wait time in Notebook 6, line 682
time.sleep(10)  # Instead of 5
```

---

## 📈 Expected Results Summary

| Metric | Before | After | Test Result |
|--------|--------|-------|-------------|
| Duplicate vectors | 64.3% | <10% | ___% |
| Silhouette score | 0.249 | 0.35-0.45 | ___ |
| Clusters (100 claims) | 19 | 5-8 | ___ |
| Metadata verification | ❌ Failed | ✅ Passed | ___ |
| Model update process | Edit code | Change env var | ___ |
| Fresh embedding handling | Regenerate all | Skip fresh | ___ |
| Staleness visibility | None | Full | ___ |

---

## ✅ Final Validation

Run this comprehensive check:

```bash
cd notebooks

echo "1. Testing configuration..."
python -c "from config import get_config; get_config().print_config()" || echo "❌ FAILED"

echo -e "\n2. Testing refresh manager..."
python refresh_manager.py --check || echo "❌ FAILED"

echo -e "\n3. Checking database migration..."
python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM embedding_refresh_config'))
    count = result.fetchone()[0]
    print(f'✅ Refresh configs: {count}')
" || echo "❌ FAILED"

echo -e "\n4. Testing notebooks with 10 claims..."
export MAX_CLAIMS_TO_PROCESS=10
python 05_embeddings_similarity_search.py > /tmp/nb5.log 2>&1 && echo "✅ Notebook 5 OK" || echo "❌ Notebook 5 FAILED"
python 06_llm_clustering_cache.py > /tmp/nb6.log 2>&1 && echo "✅ Notebook 6 OK" || echo "❌ Notebook 6 FAILED"

echo -e "\n5. Testing auto-refresh (should skip all)..."
python 05_embeddings_similarity_search.py 2>&1 | grep -q "Skipped" && echo "✅ Auto-refresh OK" || echo "⚠️  No skips (maybe disabled)"

echo -e "\n✅ ALL TESTS COMPLETE"
```

---

## 🎯 Success Criteria

Your testing is successful if:

1. ✅ Database migration completes
2. ✅ Configuration system loads and overrides work
3. ✅ Notebook 5 creates embeddings with timestamps
4. ✅ Fresh embeddings are skipped on second run
5. ✅ Notebook 6 clusters without duplicate names
6. ✅ Metadata verification passes
7. ✅ Refresh manager detects stale data
8. ✅ Duplicate vectors < 10%
9. ✅ Silhouette score > 0.35
10. ✅ Model changes via env vars work

---

## 📚 Reference

- **Configuration docs**: `notebooks/CONFIGURATION_AND_REFRESH_SYSTEM.md`
- **Regeneration guide**: `notebooks/REGENERATION_GUIDE.md`
- **Analysis**: `notebooks/ANALYSIS_NOTEBOOKS_5_6_OUTPUTS.md`

---

**Testing Duration**: 30-45 minutes
**Last Updated**: 2025-12-18
**Version**: 1.0.0
