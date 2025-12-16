# Notebook 05: Embeddings & Similarity Search - Setup Guide

## Overview
Notebook 05 has been updated to use **Hugging Face Serverless Inference API** instead of local PyTorch/sentence-transformers to avoid dependency conflicts.

## The Problem (FIXED)
Previous version used `sentence-transformers` which requires PyTorch 2.2+ and has compatibility issues with transformers library versions. This caused the error:
```
AttributeError: module 'torch' has no attribute 'register_pytree_node'
```

## The Solution
Use Hugging Face's serverless infrastructure to generate embeddings - **no PyTorch installation required!**

## Setup Instructions

### Required Environment Variables

1. **VERCEL_POSTGRES_URL** (Required)
   - Your PostgreSQL database connection string
   - Already configured in your project

2. **HF_TOKEN** (Recommended for best performance)
   - Hugging Face API token for authenticated requests
   - Get your token at: https://huggingface.co/settings/tokens

### How to Get HF_TOKEN

1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. Give it a name (e.g., "Medical Billing ML")
4. Select permissions:
   - ✅ **Make calls to Inference Providers** (required)
   - Can leave others unchecked
5. Click "Create token"
6. Copy the token (starts with `hf_...`)

### Setting the Environment Variable

#### Option A: In Vercel Project Settings
```bash
# Add to Vercel → Project Settings → Environment Variables
Name:  HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### Option B: Local Development (.env file)
```bash
# Create or edit .env file in project root
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" >> .env
```

#### Option C: Temporary (Current Session Only)
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Dependencies
Install minimal dependencies (no PyTorch needed!):
```bash
pip install huggingface_hub numpy pandas sqlalchemy psycopg2-binary requests
```

## Features

### 1. Embedding Generation
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- 384-dimensional embeddings
- Serverless compute (no local GPU needed)

### 2. Semantic Similarity Search
- Find clinically similar notes using vector search
- Powered by pgvector extension in PostgreSQL

### 3. Billing Code Validation
- Validate if billing codes match clinical documentation
- Uses cosine similarity between code descriptions and notes

## Fallback Behavior

The notebook intelligently handles different scenarios:

1. **Best Case**: HF_TOKEN set + huggingface_hub installed
   - Uses authenticated InferenceClient
   - Highest rate limits

2. **Fallback 1**: No HF_TOKEN but huggingface_hub installed
   - Uses unauthenticated HTTP requests
   - Lower rate limits

3. **Fallback 2**: No huggingface_hub library
   - Uses raw HTTP requests with `requests` library
   - Still works, just with lower performance

## Advantages Over Previous Approach

| Feature | Old (PyTorch) | New (HF Serverless) |
|---------|---------------|---------------------|
| PyTorch installation | Required (~2GB) | Not required |
| Dependency conflicts | Common | None |
| Local compute | Required (CPU/GPU) | Serverless |
| Environment size | Large | Small |
| Model quality | ✅ | ✅ (identical) |
| Free tier | N/A | Available |
| Production scaling | Manual | Built-in |

## Usage Example

```python
# Generate embedding
embedding = get_embedding("Patient with diabetes")

# Search similar notes
results = search_similar_notes("diabetes insulin", top_k=5)

# Validate billing code
validation = validate_billing_code(claim_id=123, icd_code='E11.9')
```

## Troubleshooting

### Rate Limiting
If you see 429 errors, you're hitting rate limits:
- **Solution**: Set HF_TOKEN environment variable for higher limits
- Free tier: 1000 requests/month
- With token: Much higher limits

### Model Loading (503 Error)
First request may take 20 seconds as model loads:
- Notebook automatically retries with backoff
- Subsequent requests are instant

### No Internet Connection
This approach requires internet connectivity to reach Hugging Face API:
- For offline use, you'd need to use the old PyTorch approach with proper version compatibility

## Next Steps
After running Notebook 05:
- **Notebook 06**: LLM clustering with category-specific cache tables
- Use embeddings for advanced analytics
- Build production semantic search features

## Support
For issues or questions:
- Hugging Face Inference API: https://huggingface.co/docs/inference-providers
- Model documentation: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
