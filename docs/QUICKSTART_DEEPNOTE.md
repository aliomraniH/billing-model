# Quick Start Guide for Deepnote (Memory-Optimized)

## ⚡ NEW: API-First Architecture (Recommended)

**Memory usage: ~200MB** (vs ~2.5GB with local NLP models)

This guide uses HuggingFace Inference API + Claude instead of heavy local models.

**Benefits:**
- 💾 92% less memory (~2.3GB saved)
- ⚡ Faster startup (instant vs 30-60 seconds)
- 🔄 Automatic model updates via API
- 📦 Smaller install (1-2 min vs 5-10 min)

> **Note:** If you prefer local models, see the bottom of this guide.

---

## 1️⃣ Setup Environment Variables (One-Time)

1. Click **⚙️ (gear icon)** in left sidebar
2. Select **"Environment variables"**
3. Add the following variables:

### Required Variables
```
Name: VERCEL_POSTGRES_URL
Value: postgresql://...@....neon.tech/...?sslmode=require
```

```
Name: HF_TOKEN
Value: hf_...
```

### Optional Variables
```
Name: ANTHROPIC_API_KEY
Value: sk-ant-...
(Optional - for Claude-based context analysis, otherwise uses rule-based fallback)
```

```
Name: BLOB_READ_WRITE_TOKEN
Value: vercel_blob_...
(Optional - for production caching)
```

```
Name: KV_REST_API_URL
Value: https://....upstash.io
(Optional - for production caching)
```

```
Name: KV_REST_API_TOKEN
Value: ...
(Optional - for production caching)
```

4. **Restart machine**: ... menu → Restart machine

---

## 2️⃣ Install Lightweight Dependencies (1-2 minutes)

```python
# Install API-first dependencies (NO heavy spaCy/medspaCy models)
%pip install -q -r /work/requirements_lightweight.txt
```

**What gets installed:**
- ✅ Database clients (~50MB)
- ✅ API clients (~30MB)
- ✅ Data processing (~100MB)
- ❌ NO spaCy (~200MB saved)
- ❌ NO medspaCy (~150MB saved)
- ❌ NO scispaCy models (~400MB saved)
- ❌ NO sentence-transformers (~500MB saved)

**Total: ~200MB vs ~2.5GB**

---

## 3️⃣ Verify Setup (~1 minute)

```python
%run /work/notebooks/00_verify_setup_lightweight.py
```

**Expected output:**
```
✅ SETUP VERIFICATION COMPLETE
============================================================

📋 Architecture Summary:
   • Database: Vercel Postgres with pgvector ✅
   • NLP: HuggingFace Inference API ✅
   • Embeddings: PubMedBERT via API ✅
   • Context Analysis: Claude API or rule-based ✅
   • Memory: ~200 MB (vs ~2.5 GB with local models) ✅
```

---

## 4️⃣ Build Knowledge Base (10-20 minutes, one-time)

```python
%run /work/notebooks/07_nlp_knowledge_base_setup.py
```

This downloads ICD-10 and HCPCS codes and generates embeddings via HuggingFace Inference API.

---

## 5️⃣ Extract Clinical Codes

```python
%run /work/notebooks/08_nlp_code_extraction.py
```

Extract billing codes from clinical notes using NLP.

---

## 6️⃣ Run Gap Analysis

```python
%run /work/notebooks/09_gap_analysis_reporting.py
```

Identify revenue leakage and compliance risks.

---

## 🔧 Troubleshooting

### Module Import Errors

If you see `No module named 'src'` or `'config' is not a package`:

1. Ensure you're using the **correct path** when running notebooks:
   ```python
   # ✅ CORRECT - Deepnote path
   %run /work/notebooks/00_verify_setup.py

   # ❌ WRONG - Relative path won't work
   %run notebooks/00_verify_setup.py
   ```

2. The notebook automatically searches for the project in:
   - `/work` (Deepnote's primary location)
   - `/work/billing-model`
   - `/home/user/billing-model`

3. Check the output shows:
   ```
   📁 Project root: /work
   📁 sys.path updated: ✅
   ```

### HuggingFace API Errors

If you see 410 or 503 errors:
- 503: Model is loading (wait 20-30 seconds and retry)
- 410: See `docs/TROUBLESHOOTING_HF_API.md` for model alternatives

### Database Connection Errors

Verify your `VERCEL_POSTGRES_URL` is set correctly:
```python
import os
print(os.getenv("VERCEL_POSTGRES_URL")[:20])  # Should show "postgresql://..."
```

---

## 📚 Documentation

- `docs/ENVIRONMENT_VARIABLES.md` - Complete variable list
- `docs/TROUBLESHOOTING_HF_API.md` - HuggingFace API issues
- `docs/DEPLOYMENT.md` - Vercel deployment guide
- `VERCEL_INTEGRATION.md` - Vercel storage integration

---

## ⚡ Architecture (API-First)

- **Cloud-Native**: No Docker, no local servers, no local models
- **Managed Services**: Vercel Postgres (Neon), HuggingFace Inference API, Claude API
- **Memory Optimized**: ~200MB (vs ~2.5GB with local models)
- **Free Tier Available**: HuggingFace (30K requests/month), Postgres (free tier)
- **HIPAA Ready**: Microsoft Presidio for PII redaction
- **Embeddings**: Microsoft PubMedBERT via HF Inference API (NO local download)
- **NLP**: HuggingFace Biomedical NER API (d4data/biomedical-ner-all)
- **Context Analysis**: Claude API or lightweight rule-based fallback
- **Vector Search**: pgvector with HNSW indexing

---

## 🔄 Alternative: Local Model Setup (Heavy)

If you prefer local models (requires ~2.5GB memory):

### Install Heavy Dependencies
```python
%pip install spacy medspacy scispacy sentence-transformers
%pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### Verify Heavy Setup
```python
%run /work/notebooks/00_verify_setup.py  # Old verification with local models
```

**Trade-offs:**
- ✅ Works offline
- ✅ Lower latency (~50ms vs ~200ms)
- ❌ Requires ~2.5GB memory
- ❌ Slow install (5-10 minutes)
- ❌ Slow startup (30-60 seconds)
- ❌ Manual model updates

**Recommendation:** Use API-first for Deepnote free tier, local models only if you have memory to spare.
