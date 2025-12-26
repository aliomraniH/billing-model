# Cloud Storage & Caching Architecture

## Overview

The NLP system uses a multi-tier caching strategy optimized for serverless/cloud environments:

```
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE HIERARCHY                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Tier 1: Vercel KV (Redis)                                  │
│  ├─ Model metadata (version, size, last_updated)            │
│  ├─ Cache hit/miss statistics                               │
│  └─ Ephemeral session data                                  │
│                                                               │
│  Tier 2: Vercel Blob Storage                                │
│  ├─ Pre-computed code embeddings (ICD-10, HCPCS)            │
│  ├─ Cached model weights (BioClinical BERT)                 │
│  └─ Large binary files (>1MB)                               │
│                                                               │
│  Tier 3: Vercel Postgres + pgvector                         │
│  ├─ Code embeddings table (searchable vectors)              │
│  ├─ NLP extractions (ClinicalCodeObject JSON)               │
│  └─ Gap analysis results                                    │
│                                                               │
│  Tier 4: HuggingFace Hub (Read-Only)                        │
│  ├─ Original model weights (BioClinical BERT)               │
│  ├─ Tokenizers and configs                                  │
│  └─ Fallback for model loading                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

### Problem: Serverless Cold Starts

In serverless environments (Vercel Functions, Replit deployments), each invocation may start with:
- Empty `/tmp` directory
- No persistent filesystem
- 1-5 second timeout for cold starts
- Large model downloads (400MB+) exceed limits

### Solution: Multi-Tier Caching

| Tier | Storage | Use Case | Latency | Cost |
|------|---------|----------|---------|------|
| KV | Vercel KV | Metadata, flags | <1ms | Free tier: 100K reads/day |
| Blob | Vercel Blob | Models, embeddings | 10-50ms | $0.15/GB-month |
| Postgres | Vercel Postgres | Structured vectors | 5-20ms | $20/month (Neon free tier) |
| HuggingFace | External CDN | Original models | 500-2000ms | Free (public models) |

## Setup Instructions

### 1. Enable Vercel Blob Storage

```bash
# Install Vercel CLI
npm install -g vercel

# Login and link project
vercel login
vercel link

# Enable Blob storage
vercel blob create nlp-models

# Get token (copy to .env)
vercel env pull
```

Add to `.env`:
```bash
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...
```

### 2. Enable Vercel KV (Redis)

```bash
# Create KV store
vercel kv create nlp-cache

# Get credentials (auto-added to .env)
vercel env pull
```

Add to `.env`:
```bash
KV_REST_API_URL=https://...kv.vercel-storage.com
KV_REST_API_TOKEN=...
```

### 3. Update Environment Variables

In Vercel Dashboard → Settings → Environment Variables:
- `BLOB_READ_WRITE_TOKEN`
- `KV_REST_API_URL`
- `KV_REST_API_TOKEN`
- `VERCEL_POSTGRES_URL` (already set)
- `HF_TOKEN` (optional)

## Usage Examples

### Cache Model Weights

```python
from src.data.cloud_cache import get_model_with_cache
from sentence_transformers import SentenceTransformer

# Load model with automatic caching
model = get_model_with_cache(
    "NeuML/bioclinical-modernbert-base-embeddings",
    lambda: SentenceTransformer("NeuML/bioclinical-modernbert-base-embeddings"),
    version="v1.0"
)
```

**First run:** Downloads from HuggingFace (30-60 seconds)
**Subsequent runs:** Loads from cache (<1 second)

### Store Pre-Computed Embeddings

```python
from src.data.cloud_cache import BlobStorage
import numpy as np

blob = BlobStorage()

# Compute embeddings once
embeddings = model.encode(["diabetes", "hypertension", ...])

# Store in Blob (faster retrieval than re-computing)
blob.put("embeddings/common_conditions_v1.npy", embeddings.tobytes())

# Later: retrieve instantly
cached = blob.get("embeddings/common_conditions_v1.npy")
if cached:
    embeddings = np.frombuffer(cached, dtype=np.float32)
```

### Session Caching with KV

```python
from src.data.cloud_cache import CloudModelCache

cache = CloudModelCache()

# Store processing results
cache.store_model_metadata("claim_batch_123", {
    "processed_count": 1000,
    "extraction_time": 45.2,
    "status": "completed"
})

# Check status in another function
metadata = cache.get_model_metadata("claim_batch_123")
print(metadata["processed_count"])  # 1000
```

## Best Practices

### 1. Pre-Warm Critical Assets

Create a deployment script that pre-populates Blob storage:

```python
# scripts/deploy_warm_cache.py
from src.data.cloud_cache import BlobStorage
from src.data.build_embeddings import download_icd10_codes
import pickle

blob = BlobStorage()

# Download and cache code sets
icd10_df = download_icd10_codes()
blob.put("codesets/icd10_2025.pkl", pickle.dumps(icd10_df))

print("✅ Cache warmed for deployment")
```

Run after each deployment:
```bash
vercel deploy --prod
python scripts/deploy_warm_cache.py
```

### 2. Version Your Cached Data

Include version in cache keys:
```python
blob.put("embeddings/icd10_v2025.1.npy", data)  # Good
blob.put("embeddings/icd10.npy", data)           # Bad (no versioning)
```

### 3. Set Expiration for Ephemeral Data

```python
import requests
import os

url = f"{os.getenv('KV_REST_API_URL')}/set/temp_claim_123"
headers = {"Authorization": f"Bearer {os.getenv('KV_REST_API_TOKEN')}"}
data = {"value": "...", "ex": 3600}  # Expire in 1 hour
requests.post(url, json=data, headers=headers)
```

### 4. Monitor Cache Hit Rates

```python
def track_cache_metrics(cache_hit: bool, operation: str):
    cache = CloudModelCache()
    metrics = cache.get_model_metadata("cache_stats") or {"hits": 0, "misses": 0}

    if cache_hit:
        metrics["hits"] += 1
    else:
        metrics["misses"] += 1

    cache.store_model_metadata("cache_stats", metrics)

    hit_rate = metrics["hits"] / (metrics["hits"] + metrics["misses"])
    print(f"Cache hit rate: {hit_rate:.1%}")
```

## Cost Optimization

### Free Tier Limits (Vercel Hobby Plan)

| Service | Free Tier | Overage Cost |
|---------|-----------|--------------|
| Blob Storage | 1 GB | $0.15/GB-month |
| KV Reads | 100K/day | $0.20/100K |
| KV Writes | 100K/day | $0.25/100K |
| Postgres | 500MB (Neon) | $20/month (Pro) |

### Estimated Usage (1000 claims/day)

| Operation | Storage | Daily Ops | Monthly Cost |
|-----------|---------|-----------|--------------|
| Model cache (one-time) | 400MB Blob | - | $0.06 |
| Code embeddings | 50MB Postgres | - | Free |
| Metadata lookups | - | 5K reads | Free |
| Session data | - | 2K writes | Free |
| **Total** | | | **~$0.10/month** |

## Fallback Strategy

If Blob/KV unavailable, system gracefully degrades:

```python
from src.data.cloud_cache import CloudModelCache

cache = CloudModelCache()
# Prints: ⚠️ No Vercel Blob/KV tokens found. Using local cache only.

# Still works, using local filesystem
cache.store_model_metadata("model", {...})  # Saves to ~/.cache/
```

## References

- [Vercel Blob Documentation](https://vercel.com/docs/storage/vercel-blob)
- [Vercel KV Documentation](https://vercel.com/docs/storage/vercel-kv)
- [Vercel Postgres Documentation](https://vercel.com/docs/storage/vercel-postgres)
- [HuggingFace Hub Caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

## Troubleshooting

### "BLOB_READ_WRITE_TOKEN not set"

```bash
# Re-pull environment variables
vercel env pull

# Manually set in .env
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...
```

### "Model download still slow"

Check that Blob storage is enabled:
```bash
vercel blob ls
# Should show cached models
```

If empty, run warm cache script:
```bash
python scripts/deploy_warm_cache.py
```

### "KV connection timeout"

Verify KV credentials:
```bash
curl -H "Authorization: Bearer $KV_REST_API_TOKEN" \
     "$KV_REST_API_URL/get/test"
```

Should return `{"result": null}` (not 401/403)
