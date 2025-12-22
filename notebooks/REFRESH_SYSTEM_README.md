# Embedding Refresh System - Documentation

## 🚀 Quick Start: Changing Configuration in Deepnote

### Fix config.py Formatting Error (Run Once)

If you encounter `ValueError: Cannot specify ',' with 's'.`, run this in a Deepnote cell:

```python
# Quick fix for config.py formatting bug
import os

config_path = '/work/config.py'

with open(config_path, 'r') as f:
    content = f.read()

# Check if already patched
if 'max_claims_display = ' in content:
    print("✅ config.py is already patched!")
else:
    # Apply the fix
    old_code = """        print("\\n⚙️  Processing Configuration:")
        print(f"   Max claims: {'ALL' if self.processing.max_claims_to_process == -1 else self.processing.max_claims_to_process:,}")
        print(f"   Batch size: {self.processing.batch_size}")"""

    new_code = """        print("\\n⚙️  Processing Configuration:")
        max_claims_display = 'ALL' if self.processing.max_claims_to_process == -1 else f'{self.processing.max_claims_to_process:,}'
        print(f"   Max claims: {max_claims_display}")
        print(f"   Batch size: {self.processing.batch_size}")"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(config_path, 'w') as f:
            f.write(content)
        print("✅ config.py patched successfully!")

print("✅ Ready to run Notebook 5!")
```

### Change Settings with Environment Variables (Recommended)

```python
# Set configuration BEFORE running Notebook 5
import os

# Processing settings
os.environ['MAX_CLAIMS_TO_PROCESS'] = '15000'  # or '-1' for all claims
os.environ['BATCH_SIZE'] = '100'

# Refresh settings
os.environ['AUTO_REFRESH_ENABLED'] = 'false'  # 'true' or 'false'
os.environ['EMBEDDING_REFRESH_HOURS'] = '12'

# Model settings (optional)
os.environ['HF_EMBEDDING_MODEL'] = 'BAAI/bge-small-en-v1.5'
os.environ['CLAUDE_MODEL'] = 'claude-sonnet-4-5-20250929'

# Now run Notebook 5
%run /work/notebooks/05_embeddings_similarity_search.py
```

**Available Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CLAIMS_TO_PROCESS` | `1000` | Number of claims to process (`-1` for all) |
| `AUTO_REFRESH_ENABLED` | `true` | Enable smart refresh system |
| `EMBEDDING_REFRESH_HOURS` | `12` | Hours before re-embedding |
| `BATCH_SIZE` | `100` | Batch size for processing |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CLAUDE_MODEL` | `claude-sonnet-4-5-20250929` | LLM model |
| `MIN_CLUSTER_SIZE` | `5` | Minimum cluster size |
| `CATEGORY_REFRESH_HOURS` | `24` | Category refresh interval |

---

## Problem You're Experiencing

When you run Notebook 5, you see a low success rate like **8.4%** with output like:

```
• Total claims processed: 1,000
• Notes created: 84
• Embeddings created: 84
• Success rate: 8.4%
```

**Why is this happening?**

The auto-refresh system is **skipping** most claims because they already have "fresh" embeddings (less than 12 hours old). This is working as designed to avoid unnecessary re-processing and API costs.

## Understanding the Refresh System

### How It Works

1. **Timestamp Tracking**: Each clinical note has a `last_embedded_at` timestamp
2. **Refresh Interval**: By default, embeddings are only refreshed every 12 hours
3. **Smart Skipping**: Claims with fresh embeddings are automatically skipped
4. **Cost Optimization**: Prevents redundant API calls to HuggingFace and Pinecone

### Configuration

The refresh system is controlled by `notebooks/config.py`:

```python
@dataclass
class RefreshConfig:
    default_embedding_refresh_hours: int = 12  # Refresh interval
    auto_refresh_enabled: bool = True          # Enable/disable auto-refresh
```

## Solutions

### Option 1: Disable Auto-Refresh (Recommended for Initial Setup)

**Temporarily disable auto-refresh to process all claims:**

```bash
# Set environment variable
export AUTO_REFRESH_ENABLED=false

# Then run notebook 5
python notebooks/05_embeddings_similarity_search.py
```

Or modify `notebooks/config.py`:
```python
auto_refresh_enabled: bool = False  # Changed from True
```

### Option 2: Clear All Timestamps (Force Re-processing)

**Method A: Command line (with confirmation):**
```bash
# Check current status
python notebooks/fix_embeddings_refresh.py --status

# Clear all timestamps (will prompt for confirmation)
python notebooks/fix_embeddings_refresh.py --clear-all

# Or skip confirmation (for automation/notebooks)
python notebooks/fix_embeddings_refresh.py --clear-all --yes

# Then run notebook 5 normally
python notebooks/05_embeddings_similarity_search.py
```

**Method B: Jupyter/Deepnote notebooks (no confirmation needed):**
```python
# In a notebook cell, simply run:
%run notebooks/clear_embeddings.py

# Or use the command-line version with --yes flag:
!python notebooks/fix_embeddings_refresh.py --clear-all --yes
```

### Option 3: Clear Only Missing Embeddings

**Selectively clear timestamps for claims missing from Pinecone:**

```bash
# This checks Pinecone and only clears timestamps for missing vectors
python notebooks/fix_embeddings_refresh.py --clear-missing
```

### Option 4: Wait for Embeddings to Become Stale

If you can wait 12 hours, the embeddings will automatically become "stale" and be re-processed on the next run.

## Production Workflow

### Initial Setup (First Time)

```bash
# 1. Disable auto-refresh for initial load
export AUTO_REFRESH_ENABLED=false

# 2. Process all claims
export MAX_CLAIMS_TO_PROCESS=-1  # Process all claims
python notebooks/05_embeddings_similarity_search.py

# 3. Re-enable auto-refresh for future runs
unset AUTO_REFRESH_ENABLED  # Will use default (True)
```

### Ongoing Maintenance

Once initial embeddings are created, auto-refresh works automatically:

```bash
# Run notebook normally - only stale embeddings will refresh
python notebooks/05_embeddings_similarity_search.py
```

### Force Full Refresh

When you need to regenerate all embeddings (model update, etc.):

```bash
# Clear timestamps
python notebooks/fix_embeddings_refresh.py --clear-all

# Re-run notebook
python notebooks/05_embeddings_similarity_search.py
```

## Expected Output (With Auto-Refresh Enabled)

```
📝 Generating clinical notes for claims...
   Will process 1,000 claims
   Existing notes in database: 2,758

   🔄 Auto-refresh enabled (interval: 12h)
   Fresh embeddings (<12h): 916 (will be SKIPPED)
   Stale/missing embeddings: 84 (will be PROCESSED)

   💡 To process all claims, either:
      1. Set AUTO_REFRESH_ENABLED=false in environment
      2. Run: python notebooks/fix_embeddings_refresh.py --clear-all
      3. Wait 12 hours for embeddings to become stale

🔄 Processing claims in batches...
   Batch size: 100
   Target: 1,000 claims

   Batch 1/10 (claims 1-100)...
   ✅ Uploaded 8 vectors to Pinecone

📊 Processing Summary:
   • Total claims processed: 1,000
   • Skipped (fresh embeddings): 916
   • Notes created: 84
   • Embeddings created: 84
   • Success rate: 100.0% (of non-skipped claims)
```

Notice the **100% success rate of non-skipped claims** - the 8.4% you saw was misleading!

## Database Schema

The refresh system adds these columns to `clinical_notes`:

```sql
CREATE TABLE clinical_notes (
    note_id SERIAL PRIMARY KEY,
    claim_id INTEGER REFERENCES claims(claim_id),
    note_type VARCHAR(50),
    note_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Refresh system columns (added automatically)
    last_embedded_at TIMESTAMP WITH TIME ZONE,
    embedding_model VARCHAR(255),
    embedding_version VARCHAR(100),

    UNIQUE(claim_id, note_type)
);
```

## Troubleshooting

### "Column does not exist" errors

The notebook now automatically creates missing columns. If you still see errors:

```bash
# Manually run the migration
python notebooks/migrations/01_add_refresh_timestamps.py
```

### Running in Jupyter/Deepnote notebooks

If you get confirmation prompt issues or `EOFError`:

```python
# Use the notebook-friendly script (no confirmation needed):
%run notebooks/clear_embeddings.py

# OR use the --yes flag:
!python notebooks/fix_embeddings_refresh.py --clear-all --yes
```

### "No module named 'pinecone'" warning

This is just a warning - the script will still work. The Pinecone check is optional:

```bash
# Install pinecone if you want full status checks:
pip install pinecone

# Or ignore the warning - timestamps are still cleared correctly
```

### All claims being skipped

```bash
# Check status
python notebooks/fix_embeddings_refresh.py --status

# Clear timestamps if needed
python notebooks/fix_embeddings_refresh.py --clear-all
```

### Database and Pinecone out of sync

```bash
# This will identify and fix discrepancies
python notebooks/fix_embeddings_refresh.py --clear-missing
```

## Configuration Reference

### Environment Variables

```bash
# Disable auto-refresh
export AUTO_REFRESH_ENABLED=false

# Change refresh interval (hours)
export EMBEDDING_REFRESH_HOURS=24

# Process all claims
export MAX_CLAIMS_TO_PROCESS=-1

# Process specific number
export MAX_CLAIMS_TO_PROCESS=5000
```

### Config File (config.json)

Create `config.json` in project root:

```json
{
  "refresh": {
    "auto_refresh_enabled": false,
    "default_embedding_refresh_hours": 24
  },
  "processing": {
    "max_claims_to_process": -1
  }
}
```

### Code Modification (config.py)

Edit `notebooks/config.py` directly:

```python
@dataclass
class RefreshConfig:
    default_embedding_refresh_hours: int = 24  # Changed from 12
    auto_refresh_enabled: bool = False          # Changed from True
```

## Best Practices

1. **Initial Setup**: Disable auto-refresh and process all claims once
2. **Development**: Keep auto-refresh disabled for testing
3. **Production**: Enable auto-refresh to optimize costs
4. **Model Updates**: Clear timestamps when changing embedding models
5. **Monitoring**: Regularly check `fix_embeddings_refresh.py --status`

## Cost Optimization

The refresh system saves costs by:

- **Avoiding redundant API calls** to HuggingFace
- **Preventing duplicate vectors** in Pinecone
- **Smart batching** of refresh operations
- **Configurable intervals** based on your needs

Example savings with 15,000 claims:
- Without refresh: 15,000 API calls per run
- With 12h refresh: ~125 API calls per run (99% reduction!)

## Support

If you're still experiencing issues:

1. Check the output of `python notebooks/fix_embeddings_refresh.py --status`
2. Review the notebook output for the refresh warning message
3. Try disabling auto-refresh temporarily
4. Check that environment variables are set correctly

---

**Last Updated**: December 2025
**Version**: 2.0
