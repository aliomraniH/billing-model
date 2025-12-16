# Medical Billing ML Pipeline - Notebooks 05 & 06

## Overview

Notebooks 05 and 06 have been completely restructured to use a **Hybrid Architecture** with December 2025 API updates:

- **Vercel Postgres**: Structured data (claims, diagnoses, clinical notes text)
- **Pinecone**: Vector embeddings (free tier, purpose-built for similarity search)
- **HuggingFace Router API**: December 2025 updated endpoint for embeddings
- **Claude 4.5**: LLM-powered intelligent cluster labeling

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEDICAL BILLING ML PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐         ┌──────────────────────────────────┐  │
│  │   Vercel Postgres    │         │         Pinecone (Free)           │  │
│  ├──────────────────────┤         ├──────────────────────────────────┤  │
│  │ • claims             │         │ Index: medical-billing-notes      │  │
│  │ • diagnoses          │◄───────►│ • 384-dim embeddings              │  │
│  │ • procedures         │  ref    │ • metadata.claim_id               │  │
│  │ • clinical_notes     │         │ • metadata.note_type              │  │
│  │ • claim_categories   │         │ • metadata.category               │  │
│  │ • predictions        │         │                                   │  │
│  └──────────────────────┘         └──────────────────────────────────┘  │
│           ▲                                    ▲                         │
│           │                                    │                         │
│           └────────────┬───────────────────────┘                         │
│                        │                                                 │
│              ┌─────────▼─────────┐                                       │
│              │  HuggingFace API  │                                       │
│              │  (Embeddings)     │                                       │
│              │  router.hf.co     │                                       │
│              └───────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Pinecone**: Purpose-built for vectors, free tier, no pgvector setup issues
- **Vercel Postgres**: Relational data, no extension headaches
- **Separation of concerns**: Vectors vs structured data
- **Faster similarity search** at scale

---

## CRITICAL December 2025 API Updates

### Fix 1: HuggingFace Inference API Endpoint

**OLD (Broken - Returns 410):**
```python
API_URL = "https://api-inference.huggingface.co/models/{model}"
client = InferenceClient(token=HF_TOKEN)
```

**NEW (Working):**
```python
API_URL = "https://router.huggingface.co/hf-inference/models/{model}"
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)
```

### Fix 2: Model Compatibility for Feature Extraction

**OLD (Broken - SentenceSimilarity pipeline error):**
```python
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # ❌ NOT supported
```

**NEW (Working models for feature_extraction):**
```python
# Option 1: Same 384 dimensions (recommended - no schema changes)
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
```

### Fix 3: sklearn HDBSCAN (v1.3+)

**OLD:**
```python
import hdbscan  # Standalone package (deprecated)
```

**NEW:**
```python
from sklearn.cluster import HDBSCAN  # Built into sklearn 1.3+
```

### Fix 4: Claude Model Update

**OLD:**
```python
model="claude-3-sonnet-20240229"
```

**NEW:**
```python
model="claude-sonnet-4-5-20250929"
```

---

## Environment Variables Required

### Required for Both Notebooks

```bash
# Database (existing)
export VERCEL_POSTGRES_URL="postgresql://..."

# HuggingFace (update token permissions)
export HF_TOKEN="hf_xxxxx"
# Get at: https://huggingface.co/settings/tokens
# MUST enable: "Make calls to Inference Providers"

# Pinecone (new - free tier)
export PINECONE_API_KEY="pcsk_xxxxx"
# Get at: https://www.pinecone.io/ (sign up free)
```

### Optional for Notebook 06

```bash
# Anthropic (for LLM clustering labels - optional)
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
# Get at: https://console.anthropic.com/
# If not set, will use generic category names
```

---

## Setup Instructions

### 1. Get Pinecone API Key (Free)

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up for free tier
3. Create a project
4. Copy API key from dashboard

### 2. Update HuggingFace Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create new token or edit existing
3. **IMPORTANT**: Enable "Make calls to Inference Providers" permission
4. Copy token

### 3. Get Anthropic API Key (Optional)

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Create API key
3. Copy key

### 4. Set Environment Variables

#### Option A: Vercel Project Settings
```bash
# Add to Vercel → Project Settings → Environment Variables
VERCEL_POSTGRES_URL=postgresql://...
HF_TOKEN=hf_xxxxx
PINECONE_API_KEY=pcsk_xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx  # Optional
```

#### Option B: Local Development (.env)
```bash
echo "HF_TOKEN=hf_xxxxx" >> .env
echo "PINECONE_API_KEY=pcsk_xxxxx" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-xxxxx" >> .env
```

#### Option C: Current Session
```bash
export HF_TOKEN="hf_xxxxx"
export PINECONE_API_KEY="pcsk_xxxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
```

---

## Notebook 05: Embeddings & Similarity Search

### What It Does

1. ✅ Connects to Vercel Postgres and Pinecone
2. ✅ Creates/migrates `clinical_notes` table (text only, no vectors)
3. ✅ Generates embeddings using HF Router API (BAAI/bge-small-en-v1.5)
4. ✅ Stores text in Postgres, vectors in Pinecone
5. ✅ Implements semantic similarity search
6. ✅ Validates billing codes against clinical documentation

### Features

- **Hybrid Storage**: Text in Postgres, vectors in Pinecone
- **December 2025 API**: Uses router.huggingface.co
- **No pgvector**: Avoids extension setup issues
- **384-dim embeddings**: BAAI/bge-small-en-v1.5 model
- **Billing validation**: Semantic similarity matching

### Usage Example

```python
# Search similar notes
results = search_similar_notes("diabetes insulin", top_k=5)

# Filter by category
results = search_similar_notes("heart attack", category_filter="cardiac", top_k=3)

# Validate billing code
validation = validate_billing_code(claim_id=123, icd_code='E11.9')
```

---

## Notebook 06: LLM Clustering & Category Cache

### What It Does

1. ✅ Loads vectors from Pinecone
2. ✅ Clusters using sklearn HDBSCAN (v1.3+)
3. ✅ Labels clusters with Claude 4.5 Sonnet
4. ✅ Creates category tables (claim_categories, claim_category_membership)
5. ✅ Enables fast category-filtered search

### Features

- **sklearn HDBSCAN**: Built-in clustering (no standalone package)
- **Claude 4.5**: Intelligent medical category labeling
- **Auto-categorization**: Assign new claims to categories
- **Category search**: Filter similarity search by category
- **Fallback support**: Generic names if no ANTHROPIC_API_KEY

### Database Tables Created

```sql
-- Master category definitions
CREATE TABLE claim_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE,
    display_name VARCHAR(200),
    description TEXT,
    centroid_json TEXT,
    claim_count INTEGER
);

-- Claim-to-category mappings
CREATE TABLE claim_category_membership (
    membership_id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT,
    category_id INTEGER REFERENCES claim_categories(category_id),
    similarity_score FLOAT,
    UNIQUE(claim_id, category_id)
);
```

### Usage Example

```python
# Search within a category
results = search_by_category("heart condition", category_name="cardiac", top_k=5)

# Auto-categorize a new claim
category = auto_categorize_claim(
    claim_id=456,
    note_text="Patient with chest pain and elevated troponin"
)
# → {'category_name': 'cardiac_conditions', 'display_name': 'Cardiac Conditions', 'similarity': 0.87}
```

---

## Quick Reference: December 2025 Changes

| Component | OLD (Broken) | NEW (Working) |
|-----------|--------------|---------------|
| **HF Endpoint** | `api-inference.huggingface.co` | `router.huggingface.co/hf-inference` |
| **HF Client** | `InferenceClient(token=...)` | `InferenceClient(provider="hf-inference", api_key=...)` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | `BAAI/bge-small-en-v1.5` |
| **Vector Storage** | pgvector (Postgres extension) | Pinecone (dedicated vector DB) |
| **HDBSCAN** | `import hdbscan` | `from sklearn.cluster import HDBSCAN` |
| **Claude Model** | `claude-3-sonnet-20240229` | `claude-sonnet-4-5-20250929` |

---

## Dependencies

Install required packages:

```bash
pip install \
    huggingface_hub \
    pinecone \
    anthropic \
    numpy pandas \
    sqlalchemy psycopg2-binary \
    scikit-learn>=1.3
```

> 📌 **Pinecone package update:** The official client is now published as `pinecone`. Uninstall any `pinecone-client` dependency to avoid import errors. Notebooks 05/06 will auto-install `pinecone` if it's missing, but it's best to install it once up front.

> 🤗 **Hugging Face client:** If you hit `ModuleNotFoundError: No module named 'huggingface_hub'`, install it once with `pip install -U huggingface_hub`. The notebooks also auto-install it when missing.

**Using a different embedding model?**

Set these environment variables so Pinecone dimensions stay aligned with your chosen model, and optionally pass `model_id` into the search helpers:

```bash
export HF_EMBEDDING_MODEL="your/model"
export HF_EMBEDDING_DIM=768  # must match the Pinecone index dimension
```

Both `search_similar_notes(...)` (Notebook 05) and `search_by_category(...)` (Notebook 06) accept an optional `model_id` argument for experimentation.

**Note**: No PyTorch required! All embeddings generated via serverless API.

---

## Migration from Old Version

If you were using the old notebooks with pgvector:

1. ✅ Run notebook 05 - it will automatically:
   - Drop old `embedding` column from `clinical_notes`
   - Migrate to Pinecone architecture
   - Regenerate embeddings using new model

2. ✅ Run notebook 06 - it will:
   - Use Pinecone vectors instead of reading from Postgres
   - Create new category tables with JSON centroids (no pgvector dependency)

---

## Troubleshooting

### Pinecone Index Not Found
```
Error: Index 'medical-billing-notes' not found
```
**Solution**: Notebook 05 auto-creates the index. Wait 10 seconds after creation.

### Pinecone package rename error
```
Exception: The official Pinecone python package has been renamed from `pinecone-client` to `pinecone`.
```
**Solution (recommended before running notebooks):** Remove the legacy dependency and install the new one:
```bash
pip uninstall -y pinecone-client
pip install -U pinecone
```

**Auto-fix fallback:** Notebooks 05/06 will automatically uninstall `pinecone-client` (if present) and install the `pinecone` package at runtime when the import is missing.

### HuggingFace 410 Error
```
HTTP Error 410: Gone
```
**Solution**: You're using old API endpoint. Update to `router.huggingface.co/hf-inference`.

### Model Pipeline Error
```
Error: Model does not support SentenceSimilarity pipeline
```
**Solution**: Switch to `BAAI/bge-small-en-v1.5` (feature_extraction compatible).

### HDBSCAN Import Error
```
ModuleNotFoundError: No module named 'hdbscan'
```
**Solution**: Update scikit-learn: `pip install scikit-learn>=1.3`

### Claude API Error
```
NotFoundError: model not found
```
**Solution**: Update to `claude-sonnet-4-5-20250929`.

---

## Performance

### Embedding Generation
- **Throughput**: ~5 embeddings/second (HF Inference API)
- **Latency**: ~200ms per embedding
- **Cost**: Free tier (with HF_TOKEN)

### Similarity Search
- **Pinecone Query**: <100ms (serverless)
- **Top-K Results**: Instant (HNSW index)
- **Filtering**: Category filters add <10ms

### LLM Clustering
- **HDBSCAN**: <1 second (10-100 vectors)
- **Claude Labeling**: ~2 seconds per cluster
- **Total**: ~30 seconds for 10 clusters

---

## Next Steps

After running notebooks 05 & 06:

1. **Build Search UI**: Use `search_similar_notes()` for semantic search
2. **Auto-categorize**: Use `auto_categorize_claim()` for new claims
3. **Billing Validation**: Use `validate_billing_code()` for fraud detection
4. **Analytics**: Query `claim_categories` for insights

---

## Support

- **Pinecone Docs**: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- **HuggingFace Inference**: [https://huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers)
- **Anthropic API**: [https://docs.anthropic.com/](https://docs.anthropic.com/)
- **sklearn HDBSCAN**: [https://scikit-learn.org/stable/modules/clustering.html#hdbscan](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)

---

## License

MIT License - Medical Billing ML Pipeline
