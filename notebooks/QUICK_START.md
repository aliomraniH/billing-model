# Quick Start Guide - Medical Billing ML System

**Last Updated:** December 21, 2025
**Branch:** `claude/cleanup-billing-duplicates-M2Iqt`
**Status:** ✅ Production Ready

---

## 🚀 One-Minute Overview

This system uses machine learning to automatically categorize medical billing claims:

1. **Notebook 05** → Generates embeddings from clinical notes
2. **Notebook 06** → Clusters embeddings and labels with Claude AI
3. **Result** → Organized categories like "Diabetes Management", "Cardiac Procedures"

---

## 📁 File Structure

All files are in `/home/user/billing-model/notebooks/`:

```
notebooks/
├── config.py           # Configuration (models, API keys)
├── utils.py            # Shared utilities (NEW! 8 functions)
├── refresh_manager.py  # Auto-refresh stale data
├── 05_*.py            # Generate embeddings
└── 06_*.py            # Cluster & categorize
```

---

## 🔧 Setup (First Time)

```bash
# 1. Install dependencies
pip install numpy pandas sqlalchemy psycopg2-binary
pip install pinecone huggingface_hub anthropic scikit-learn

# 2. Set environment variables
export VERCEL_POSTGRES_URL="postgres://user:pass@host:5432/db"
export HF_TOKEN="hf_..."
export PINECONE_API_KEY="..."
export ANTHROPIC_API_KEY="sk-..."  # Optional

# 3. Setup database (one-time)
cd /home/user/billing-model/notebooks
python 01_setup_database.py
python 02_load_synthea_data.py
```

---

## ▶️ Run the Pipeline

```bash
cd /home/user/billing-model/notebooks

# Process 100 claims for testing
export MAX_CLAIMS_TO_PROCESS=100

# Step 1: Generate embeddings
python 05_embeddings_similarity_search.py

# Step 2: Cluster and categorize
python 06_llm_clustering_cache.py
```

**Expected results:**
- ✅ 100 embeddings created
- ✅ 5-8 categories created
- ✅ All integration tests pass

---

## 🛠️ New Utilities Module

The new `utils.py` provides 8 shared functions:

```python
from utils import (
    get_embedding,              # Generate embeddings with retry
    init_database,              # Connect to PostgreSQL
    init_pinecone,              # Setup Pinecone
    init_hf_client,             # HuggingFace client
    init_anthropic_client,      # Claude client
    validate_environment_variables,
    test_embedding_generation,
    ensure_package_installed
)

# Example usage
engine, total_claims = init_database(DATABASE_URL)
hf_client = init_hf_client(HF_TOKEN, MODEL_ID)
embedding = get_embedding(text, hf_client, MODEL_ID, 384)
```

**Benefits:**
- ✅ Eliminates ~270 lines of duplicate code
- ✅ Single source of truth
- ✅ Consistent error handling
- ✅ Easy to maintain

---

## 📊 What Each Notebook Does

### Notebook 05: Embeddings
```python
# What it does:
1. Generates clinical notes for claims
2. Creates 384-dim embeddings (BAAI/bge-small-en-v1.5)
3. Stores in Pinecone + PostgreSQL
4. Auto-refreshes stale embeddings (>12h old)

# Output:
✅ Vectors in Pinecone
✅ Notes in clinical_notes table
```

### Notebook 06: Clustering
```python
# What it does:
1. Loads all embeddings from Pinecone
2. Clusters with HDBSCAN
3. Labels clusters using Claude AI
4. Updates Pinecone metadata
5. Creates categories in database

# Output:
✅ Categories (e.g., "Diabetes Management")
✅ claim_categories table populated
✅ Search functions ready
```

---

## 🔄 Workflow Sequence

```
Setup Database → Load Data → Generate Embeddings → Cluster & Label
     ↓              ↓              ↓                    ↓
  01_setup   02_load_data   05_embeddings      06_clustering
```

**Maintenance:** Run `refresh_manager.py --check` periodically

---

## 🧪 Testing

```bash
# Verify imports work
cd /home/user/billing-model/notebooks
python test_imports.py

# Test with small dataset
export MAX_CLAIMS_TO_PROCESS=10
python 05_embeddings_similarity_search.py
python 06_llm_clustering_cache.py
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `CLEANUP_SUMMARY.md` | Complete refactoring details |
| `IMPORT_REFERENCE.md` | Import paths & structure |
| `QUICK_START.md` | This file! |
| `test_imports.py` | Import verification script |

---

## ⚡ Common Commands

```bash
# Check refresh status
python refresh_manager.py --check

# Process all claims
export MAX_CLAIMS_TO_PROCESS=-1
python 05_embeddings_similarity_search.py

# Use different Claude model
export CLAUDE_MODEL="claude-opus-4-5-20251101"
python 06_llm_clustering_cache.py

# View configuration
python -c "from config import get_config; get_config().print_config()"
```

---

## 🔍 Troubleshooting

**Import errors?**
```bash
# Ensure you're in the right directory
cd /home/user/billing-model/notebooks
python test_imports.py
```

**Missing dependencies?**
```bash
pip install -r requirements.txt
# Or install individually (see Setup section)
```

**Environment variables not set?**
```bash
# Check what's missing
python -c "from utils import validate_environment_variables; validate_environment_variables(['VERCEL_POSTGRES_URL', 'HF_TOKEN', 'PINECONE_API_KEY'])"
```

---

## 📈 Performance Tips

- **Start small:** Use `MAX_CLAIMS_TO_PROCESS=100` for testing
- **Auto-refresh saves costs:** Skips fresh embeddings (<12h old)
- **Adjust clustering:** Set `MIN_CLUSTER_SIZE=20` for better clusters
- **Monitor quality:** Check Silhouette score in notebook 06 output

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Generate embeddings | `python 05_embeddings_similarity_search.py` |
| Cluster & categorize | `python 06_llm_clustering_cache.py` |
| Check stale data | `python refresh_manager.py --check` |
| Verify imports | `python test_imports.py` |
| View config | `python -c "from config import get_config; get_config().print_config()"` |

---

**Ready to go!** Start with `python test_imports.py` to verify everything is set up correctly.

For detailed information, see:
- `CLEANUP_SUMMARY.md` - Complete refactoring details
- `IMPORT_REFERENCE.md` - Import structure
- `TESTING_GUIDE.md` - Comprehensive testing guide
