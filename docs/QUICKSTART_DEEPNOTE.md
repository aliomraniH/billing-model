# Quick Start Guide for Deepnote

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

### Optional Variables (for caching)
```
Name: BLOB_READ_WRITE_TOKEN
Value: vercel_blob_...
```

```
Name: KV_REST_API_URL
Value: https://....upstash.io
```

```
Name: KV_REST_API_TOKEN
Value: ...
```

4. **Restart machine**: ... menu → Restart machine

---

## 2️⃣ First-Time Installation (5-10 minutes)

In a new Deepnote cell, run:

```python
%run /home/user/billing-model/notebooks/00_first_time_install.py
```

**Expected output:**
```
✅ INSTALLATION COMPLETE
  ✅ Core dependencies
  ✅ NLP packages
  ✅ spaCy models
  ✅ Database connection
```

---

## 3️⃣ Verify Setup (~1 minute)

```python
%run /home/user/billing-model/notebooks/00_verify_setup.py
```

**Expected output:**
```
✅ SETUP VERIFICATION COMPLETE
  ✅ Environment variables
  ✅ Python packages
  ✅ Database connection
  ✅ HuggingFace API
  ✅ Core modules
```

---

## 4️⃣ Build Knowledge Base (10-20 minutes, one-time)

```python
%run /home/user/billing-model/notebooks/07_nlp_knowledge_base_setup.py
```

This downloads ICD-10 and HCPCS codes and generates embeddings via HuggingFace Inference API.

---

## 5️⃣ Extract Clinical Codes

```python
%run /home/user/billing-model/notebooks/08_nlp_code_extraction.py
```

Extract billing codes from clinical notes using NLP.

---

## 6️⃣ Run Gap Analysis

```python
%run /home/user/billing-model/notebooks/09_gap_analysis_reporting.py
```

Identify revenue leakage and compliance risks.

---

## 🔧 Troubleshooting

### Module Import Errors

If you see `No module named 'src'` or `'config' is not a package`:

1. Ensure you're using the **full path** when running notebooks:
   ```python
   %run /home/user/billing-model/notebooks/00_verify_setup.py
   ```

2. The notebook automatically searches for the project in:
   - `/home/user/billing-model`
   - `/work/billing-model`
   - `/datasets/billing-model`

3. Check the output shows:
   ```
   📁 Project root: /home/user/billing-model
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

## ⚡ Architecture

- **Cloud-Native**: No Docker, no local servers
- **Managed Services**: Vercel Postgres (Neon), HuggingFace Inference API
- **Zero-Cost Stack**: Free tiers only
- **HIPAA Ready**: Microsoft Presidio for PII redaction
- **Embeddings**: Microsoft PubMedBERT via HF Inference API
- **NLP**: medspaCy + scispaCy for clinical entity extraction
- **Vector Search**: pgvector with HNSW indexing
