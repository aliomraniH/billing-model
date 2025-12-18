# Configuration and Refresh System Documentation

## Overview

This document describes the centralized configuration system and timestamp-based refresh mechanism implemented for the Medical Billing ML system.

## 🎯 Key Improvements

### 1. **No Hardcoded Models**
All model names, versions, and API endpoints are configured in ONE place (`config.py`). Model updates require ZERO code changes.

### 2. **Timestamp-Based Refresh**
Automatic refresh of stale embeddings and categories based on configurable intervals (e.g., 12 hours, 24 hours).

### 3. **Flexible Configuration**
Configure via environment variables, JSON file, or Python defaults. Choose what works best for your workflow.

---

## 📁 New Files

| File | Purpose |
|------|---------|
| `notebooks/config.py` | Centralized model configuration system |
| `notebooks/refresh_manager.py` | Utility to manage stale data refresh |
| `notebooks/migrations/01_add_refresh_timestamps.py` | Database migration for timestamp columns |
| `notebooks/CONFIGURATION_AND_REFRESH_SYSTEM.md` | This documentation |

---

## 🔧 Configuration System

### Architecture

```
Priority Order (highest to lowest):
1. Environment Variables   ← export CLAUDE_MODEL='claude-opus-4-5-20251101'
2. config.json File        ← { "llm": { "model_id": "..." } }
3. config.py Defaults      ← CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
```

### Configuration Categories

#### **Embedding Configuration**
```python
@dataclass
class EmbeddingConfig:
    provider: str = "huggingface"
    model_id: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    api_endpoint: str = "https://api-inference.huggingface.co"
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: int = 2
```

**Environment Variables**:
- `HF_EMBEDDING_MODEL` - Override embedding model
- `HF_EMBEDDING_DIM` - Override dimension
- `EMBEDDING_BATCH_SIZE` - Batch size for API calls
- `HF_API_ENDPOINT` - Custom HuggingFace endpoint

#### **LLM Configuration**
```python
@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model_id: str = "claude-sonnet-4-5-20250929"
    api_version: str = "2023-06-01"
    max_tokens: int = 500
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay_seconds: int = 2
```

**Environment Variables**:
- `CLAUDE_MODEL` - Override Claude model
- `CLAUDE_MAX_TOKENS` - Override max tokens
- `CLAUDE_TEMPERATURE` - Override temperature

#### **Refresh Configuration**
```python
@dataclass
class RefreshConfig:
    default_embedding_refresh_hours: int = 12  # Refresh embeddings every 12h
    category_refresh_hours: int = 24           # Refresh categories every 24h
    cluster_refresh_hours: int = 48            # Refresh clusters every 48h
    min_embedding_age_hours: int = 1           # Minimum age before refresh
    auto_refresh_enabled: bool = True          # Auto-refresh on notebook run
    refresh_on_startup: bool = False           # Refresh all on startup
    refresh_batch_size: int = 100              # Batch size for refreshes
    max_refresh_per_run: int = 1000            # Max refreshes per run
```

**Environment Variables**:
- `EMBEDDING_REFRESH_HOURS` - Embedding refresh interval
- `CATEGORY_REFRESH_HOURS` - Category refresh interval
- `AUTO_REFRESH_ENABLED` - Enable/disable auto-refresh

---

## 🔄 Refresh System

### How It Works

1. **Timestamp Tracking**: Every embedding/category has `last_embedded_at`/`last_refreshed_at` timestamp
2. **Staleness Detection**: Compare current time vs timestamp against configured interval
3. **Auto-Refresh**: Notebooks automatically skip fresh data, regenerate stale data
4. **Manual Refresh**: Use `refresh_manager.py` for explicit control

### Database Schema Changes

#### `clinical_notes` Table
```sql
ALTER TABLE clinical_notes
  ADD COLUMN last_embedded_at TIMESTAMP WITH TIME ZONE,
  ADD COLUMN embedding_version VARCHAR(100),
  ADD COLUMN embedding_model VARCHAR(255);
```

#### `claim_categories` Table
```sql
ALTER TABLE claim_categories
  ADD COLUMN last_refreshed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  ADD COLUMN refresh_interval_hours INTEGER DEFAULT 24,
  ADD COLUMN cluster_version VARCHAR(50);
```

#### `embedding_refresh_config` Table (New)
```sql
CREATE TABLE embedding_refresh_config (
    config_id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    refresh_interval_hours INTEGER NOT NULL DEFAULT 12,
    enabled BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 🚀 Quick Start

### Initial Setup

```bash
# Step 1: Run database migration (ONCE)
python notebooks/migrations/01_add_refresh_timestamps.py

# Step 2: (Optional) Create custom config file
cat > config.json <<EOF
{
  "llm": {
    "model_id": "claude-opus-4-5-20251101"
  },
  "refresh": {
    "default_embedding_refresh_hours": 6
  }
}
EOF

# Step 3: Run notebooks normally
python notebooks/05_embeddings_similarity_search.py
python notebooks/06_llm_clustering_cache.py
```

### Configuration Examples

#### Example 1: Use Latest Claude Opus (Environment Variable)
```bash
export CLAUDE_MODEL='claude-opus-4-5-20251101'
python notebooks/06_llm_clustering_cache.py
```

#### Example 2: Use Different Embedding Model (JSON Config)
```json
{
  "embedding": {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384
  }
}
```

#### Example 3: Aggressive Refresh Schedule
```bash
export EMBEDDING_REFRESH_HOURS=6
export CATEGORY_REFRESH_HOURS=12
python notebooks/05_embeddings_similarity_search.py
```

#### Example 4: Disable Auto-Refresh
```bash
export AUTO_REFRESH_ENABLED=false
python notebooks/05_embeddings_similarity_search.py
```

---

## 📊 Refresh Manager Usage

### Check Refresh Status

```bash
# Dry run - check what needs refresh
python notebooks/refresh_manager.py --check

# Output:
# 📊 Embeddings:
#    Total embedded notes: 15,000
#    Stale (need refresh): 3,245 (21.6%)
#    Threshold: Older than 12 hours
```

### Refresh Stale Embeddings

```bash
# Refresh up to 1,000 stale embeddings
python notebooks/refresh_manager.py --refresh-embeddings --max-refresh 1000

# Dry run first to see what would happen
python notebooks/refresh_manager.py --refresh-embeddings --dry-run
```

### Refresh Stale Categories

```bash
# Mark categories as needing re-clustering
python notebooks/refresh_manager.py --refresh-categories
```

### Refresh Everything

```bash
# Refresh both embeddings and categories
python notebooks/refresh_manager.py --refresh-all
```

---

## 🔍 How Notebooks Use the System

### Notebook 5 (Embeddings)

**Before (Hardcoded)**:
```python
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BATCH_SIZE = 100
```

**After (Dynamic)**:
```python
from config import get_config

cfg = get_config()
MODEL_ID = cfg.embedding.model_id          # Configurable
EMBEDDING_DIM = cfg.embedding.dimension     # Configurable
BATCH_SIZE = cfg.embedding.batch_size       # Configurable
```

**Refresh Logic**:
```python
# Check if embedding is fresh
result = conn.execute(text("""
    SELECT last_embedded_at, embedding_model
    FROM clinical_notes
    WHERE claim_id = :cid
"""), {'cid': claim_id})

row = result.fetchone()
if row and row[0]:
    age_hours = (datetime.now() - row[0]).total_seconds() / 3600

    # Skip if fresh and model hasn't changed
    if age_hours < REFRESH_INTERVAL_HOURS and row[1] == MODEL_ID:
        continue  # Skip this embedding

# Generate fresh embedding
embedding = get_embedding(note_text)

# Update timestamp
conn.execute(text("""
    UPDATE clinical_notes
    SET
        last_embedded_at = NOW(),
        embedding_model = :model,
        embedding_version = :version
    WHERE claim_id = :cid
"""), {'cid': claim_id, 'model': MODEL_ID, 'version': '1.0'})
```

### Notebook 6 (Clustering)

**Before (Hardcoded)**:
```python
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 2
```

**After (Dynamic)**:
```python
from config import get_config

cfg = get_config()
CLAUDE_MODEL = cfg.llm.model_id                # Configurable
CLAUDE_MAX_TOKENS = cfg.llm.max_tokens         # Configurable
MIN_CLUSTER_SIZE = cfg.clustering.min_cluster_size  # Configurable
```

**LLM Call**:
```python
message = anthropic_client.messages.create(
    model=CLAUDE_MODEL,          # From config
    max_tokens=CLAUDE_MAX_TOKENS,  # From config
    temperature=CLAUDE_TEMPERATURE,  # From config
    ...
)
```

---

## 📈 Refresh Scenarios

### Scenario 1: Model Update

```bash
# Old model embeddings automatically marked as stale
export HF_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L12-v2"
export HF_EMBEDDING_DIM=384

# Run Notebook 5 - will regenerate all embeddings with new model
python notebooks/05_embeddings_similarity_search.py
```

### Scenario 2: Periodic Refresh

```bash
# Set up cron job to refresh every 12 hours
0 */12 * * * cd /path/to/project && python notebooks/refresh_manager.py --refresh-all
```

### Scenario 3: On-Demand Refresh

```python
# In your application code
from refresh_manager import RefreshManager

manager = RefreshManager(database_url)

# Check if refresh is needed
embedding_stats, category_stats = manager.check_refresh_status()

if embedding_stats.stale_found > 100:
    # Trigger refresh
    manager.auto_refresh_embeddings(max_refresh=1000)
```

---

## 🎓 Best Practices

### 1. **Configuration Management**

✅ **DO**:
- Use environment variables for production
- Use `config.json` for local development (add to `.gitignore`)
- Update `config.py` defaults when upgrading baseline models
- Document configuration changes in commit messages

❌ **DON'T**:
- Hardcode model names in notebook cells
- Commit `config.json` with sensitive API keys
- Change configuration during notebook execution

### 2. **Refresh Intervals**

| Data Type | Recommended Interval | Rationale |
|-----------|---------------------|-----------|
| **Embeddings** | 12-24 hours | Balance freshness vs API costs |
| **Categories** | 24-48 hours | Clustering is expensive |
| **Critical Data** | 6 hours | For high-priority use cases |
| **Archive Data** | 7 days (168 hours) | Rarely changes |

### 3. **Migration Strategy**

When updating models:

```bash
# Step 1: Test with small dataset
export MAX_CLAIMS_TO_PROCESS=100
export HF_EMBEDDING_MODEL="new-model-name"
python notebooks/05_embeddings_similarity_search.py

# Step 2: Verify quality
python notebooks/06_llm_clustering_cache.py

# Check silhouette score, clustering quality

# Step 3: If satisfied, regenerate all
python notebooks/refresh_manager.py --refresh-all
export MAX_CLAIMS_TO_PROCESS=-1
python notebooks/05_embeddings_similarity_search.py
```

---

## 🐛 Troubleshooting

### Issue: "Column 'last_embedded_at' does not exist"

**Solution**: Run the migration
```bash
python notebooks/migrations/01_add_refresh_timestamps.py
```

### Issue: Config changes not taking effect

**Solution**: Check priority order
```bash
# Environment variables override everything
unset CLAUDE_MODEL  # Remove env var if needed

# Check what's loaded
python -c "from config import get_config; get_config().print_config()"
```

### Issue: All embeddings being regenerated every run

**Solution**: Check AUTO_REFRESH setting
```bash
export AUTO_REFRESH_ENABLED=true  # Should be true
export EMBEDDING_REFRESH_HOURS=12  # Increase interval
```

### Issue: Stale embeddings not being refreshed

**Solution**: Manual trigger
```bash
# Force refresh with manager
python notebooks/refresh_manager.py --refresh-embeddings

# Or disable auto-refresh and manually control
export AUTO_REFRESH_ENABLED=false
```

---

## 📚 API Reference

### `config.py`

```python
from config import get_config, ModelConfig

# Load configuration
cfg = get_config()  # Loads from env, file, or defaults

# Access settings
embedding_model = cfg.embedding.model_id
claude_model = cfg.llm.model_id
refresh_hours = cfg.refresh.default_embedding_refresh_hours

# Print current configuration
cfg.print_config()

# Save to JSON
cfg.save('my-config.json')

# Load from specific file
cfg = get_config(config_file='prod-config.json')
```

### `refresh_manager.py`

```python
from refresh_manager import RefreshManager

# Initialize
manager = RefreshManager(database_url, dry_run=False)

# Check status
emb_stats, cat_stats = manager.check_refresh_status()
print(f"Stale embeddings: {emb_stats.stale_found}")

# Refresh embeddings
stats = manager.auto_refresh_embeddings(max_refresh=1000)
print(f"Refreshed: {stats.refreshed}")

# Refresh categories
stats = manager.auto_refresh_categories()
```

---

## 🔐 Security Notes

1. **Never commit** `config.json` with API keys
2. **Use environment variables** for production secrets
3. **Rotate API keys** regularly
4. **Limit refresh rates** to avoid rate limiting
5. **Monitor costs** when using paid APIs

---

## 📊 Monitoring

### Key Metrics to Track

```sql
-- Embedding freshness
SELECT
    COUNT(*) FILTER (WHERE last_embedded_at > NOW() - INTERVAL '12 hours') as fresh,
    COUNT(*) FILTER (WHERE last_embedded_at <= NOW() - INTERVAL '12 hours') as stale,
    COUNT(*) FILTER (WHERE last_embedded_at IS NULL) as never_embedded
FROM clinical_notes
WHERE embedding IS NOT NULL;

-- Category refresh status
SELECT
    category_name,
    last_refreshed_at,
    refresh_interval_hours,
    EXTRACT(EPOCH FROM (NOW() - last_refreshed_at))/3600 as age_hours
FROM claim_categories
ORDER BY age_hours DESC
LIMIT 10;

-- Refresh config status
SELECT *
FROM embedding_refresh_config
WHERE enabled = TRUE;
```

---

## 🎯 Summary

### Before This Update
- ❌ Models hardcoded in notebooks
- ❌ Model changes require code edits
- ❌ No timestamp tracking
- ❌ Manual re-generation of all data
- ❌ No visibility into staleness

### After This Update
- ✅ Centralized configuration system
- ✅ Model changes via env vars or config file
- ✅ Automatic staleness detection
- ✅ Intelligent refresh (only stale data)
- ✅ Full visibility and control

### Migration Checklist

- [ ] Run database migration (`01_add_refresh_timestamps.py`)
- [ ] Update environment variables or create `config.json`
- [ ] Test with small dataset first
- [ ] Review refresh intervals for your use case
- [ ] Set up monitoring/alerting for stale data
- [ ] Document custom configuration choices

---

## 📖 Related Documentation

- **Configuration Guide**: This document
- **Regeneration Guide**: `notebooks/REGENERATION_GUIDE.md`
- **Analysis**: `notebooks/ANALYSIS_NOTEBOOKS_5_6_OUTPUTS.md`
- **Duplicate Fix**: `notebooks/FIX_DUPLICATE_CATEGORIES.md`

---

**Last Updated**: 2025-12-18
**Version**: 1.0.0
**Author**: Claude Code (Automated System)
