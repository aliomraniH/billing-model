# Deepnote Setup Guide

Complete guide for setting up the Medical Billing NLP system in Deepnote.

## 🚀 Quick Start (5 Steps)

### Step 1: Open Deepnote Setup Notebook

In Deepnote, open and run:
```
notebooks/00_deepnote_setup.py
```

This notebook will guide you through the complete setup process.

### Step 2: Add Environment Variables in Deepnote UI

**DO NOT run bash commands in Python cells!** Use the Deepnote UI instead:

1. **Click the gear icon** (⚙️) in the left sidebar
2. **Select "Environment variables"**
3. **Click "+ Add environment variable"**
4. Add each variable below:

#### Required: VERCEL_POSTGRES_URL

```
Name: VERCEL_POSTGRES_URL
Value: postgresql://user:password@host.neon.tech/dbname?sslmode=require
```

**Where to get it:**
1. Go to https://vercel.com/dashboard
2. Navigate to: **Storage → Postgres → Your Database**
3. Click the **".env.local"** tab
4. Copy the `POSTGRES_URL` value
5. Paste into Deepnote environment variables

**Example:**
```
postgresql://neondb_owner:xxxxx@ep-xxxxx.us-east-2.aws.neon.tech/neondb?sslmode=require
```

#### Optional: HF_TOKEN (Recommended)

```
Name: HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxxxx
```

**Where to get it:**
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `deepnote-billing-nlp`
4. Type: **Read**
5. Copy the token
6. Paste into Deepnote

**Benefits:**
- Faster model downloads
- Access to gated models (if needed)
- No rate limiting

#### Optional: Cloud Caching (Production Only)

For production deployments with Vercel Functions:

**On your LOCAL machine terminal** (not in Deepnote):
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Link project
vercel link

# Create Blob storage
vercel blob create nlp-models

# Create KV cache
vercel kv create nlp-cache

# Pull environment variables to .env file
vercel env pull

# View the values
cat .env
```

Then copy these values to Deepnote:
- `BLOB_READ_WRITE_TOKEN`
- `KV_REST_API_URL`
- `KV_REST_API_TOKEN`

**Note:** These are optional for Deepnote development. Only needed for production deployments.

### Step 3: Restart Deepnote Machine

After adding environment variables:

1. Click the **"..." menu** (three dots) in the top right
2. Select **"Restart machine"**
3. Wait for the machine to restart (~30 seconds)

**Why?** Deepnote needs to restart to load new environment variables.

### Step 4: Run Setup Notebook

Run all cells in `notebooks/00_deepnote_setup.py`:

1. Click **"Run all"** at the top
2. Wait for installation to complete (5-10 minutes first time)
3. Review output for any errors

Expected output:
```
✅ Environment variables configured
✅ Dependencies installed
✅ Database connection successful
✅ All health checks passed
```

### Step 5: Build Knowledge Base

Run the knowledge base setup (one-time, 10-20 minutes):

```python
%run notebooks/07_nlp_knowledge_base_setup.py
```

This will:
- Download ICD-10 and HCPCS codes from CMS
- Generate embeddings with BioClinical BERT
- Store in Postgres with pgvector
- Create HNSW index for fast search

## 🎯 Common Issues & Solutions

### Issue: "SyntaxError: invalid syntax" when running bash commands

**Problem:** You tried to run bash commands in a Python cell:
```python
npm install -g vercel  # ❌ WRONG - This is bash, not Python
```

**Solution:** Don't run bash commands in Deepnote notebooks. Use the UI instead:
- For environment variables: Use Deepnote UI (⚙️ → Environment variables)
- For Vercel setup: Use your **local terminal**, not Deepnote
- For Python scripts: Use `%run` or `!python`:
  ```python
  %run scripts/setup_nlp_system.py  # ✅ Correct
  ```

### Issue: "VERCEL_POSTGRES_URL not set"

**Solution:**
1. Go to Vercel dashboard: https://vercel.com/dashboard
2. Navigate to: Storage → Postgres
3. Copy the connection string from ".env.local" tab
4. In Deepnote: ⚙️ → Environment variables → Add variable
5. Restart machine: ... → Restart machine

### Issue: "Module not found: medspacy"

**Solution:**
```python
# Run the setup script
%run scripts/setup_nlp_system.py

# Then restart machine
# ... menu → Restart machine
```

### Issue: "Connection refused" or "Database connection failed"

**Checklist:**
1. ✅ VERCEL_POSTGRES_URL is set in Deepnote environment variables
2. ✅ Connection string includes `?sslmode=require` at the end
3. ✅ Database is not paused (check Vercel dashboard)
4. ✅ Machine has been restarted after adding variables

**Test connection:**
```python
import os
print(os.getenv("VERCEL_POSTGRES_URL"))  # Should show connection string
```

### Issue: Model download is very slow

**First time:** 5-10 minutes is normal (downloading 400MB+ models)

**Subsequent runs:** Should be <1 second (uses cache)

**Speed it up:**
1. Add HF_TOKEN to environment variables (see Step 2)
2. Wait for first download to complete
3. Future runs will use cached models

### Issue: "pgvector extension not found"

**Solution:**
```python
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
    print("✅ pgvector enabled")
```

## 📋 Verification Checklist

After setup, verify everything works:

```python
# 1. Check environment variables
import os
assert os.getenv("VERCEL_POSTGRES_URL"), "❌ VERCEL_POSTGRES_URL not set"
print("✅ Environment variables configured")

# 2. Check database connection
from sqlalchemy import create_engine, text
engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))
with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
print("✅ Database connection works")

# 3. Check NLP dependencies
import medspacy
import spacy
from sentence_transformers import SentenceTransformer
print("✅ NLP dependencies installed")

# 4. Check imports
from src.features.clinical_code_object import ClinicalCodeObject
from src.features.nlp_extractor import ClinicalCodeExtractor
from src.features.gap_analyzer import GapAnalyzer
print("✅ All modules importable")

# 5. Test model loading
model = SentenceTransformer("NeuML/bioclinical-modernbert-base-embeddings")
print("✅ Embedding model loads")

print("\n🎉 All systems ready!")
```

## 🔄 Development Workflow

### Daily workflow in Deepnote:

1. **Start a new notebook**
2. **Run your analysis:**
   ```python
   # Extract codes from claims
   %run notebooks/08_nlp_code_extraction.py

   # Run gap analysis
   %run notebooks/09_gap_analysis_reporting.py
   ```

3. **Use the API programmatically:**
   ```python
   from src.features.nlp_extractor import ClinicalCodeExtractor
   from sqlalchemy import create_engine
   import os

   engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))
   extractor = ClinicalCodeExtractor(db_connection=engine)

   # Process a single claim
   clinical_obj = extractor.process_claim(
       claim_id="TEST-001",
       clinical_text="Patient with type 2 diabetes and hypertension..."
   )

   # View extracted codes
   for diag in clinical_obj.billable_diagnoses:
       print(f"{diag.code}: {diag.term} (confidence: {diag.confidence:.2%})")
   ```

### When to restart machine:

- ✅ After adding/changing environment variables
- ✅ After installing new packages
- ✅ If imports suddenly fail
- ✅ Once a day (to clear memory)

**Don't restart:**
- ❌ In the middle of long-running processes
- ❌ While models are downloading

## 📚 Next Steps

After setup completes:

1. **Build knowledge base** (one-time):
   ```python
   %run notebooks/07_nlp_knowledge_base_setup.py
   ```

2. **Extract codes from claims:**
   ```python
   %run notebooks/08_nlp_code_extraction.py
   ```

3. **Run gap analysis:**
   ```python
   %run notebooks/09_gap_analysis_reporting.py
   ```

## 🆘 Getting Help

**Documentation:**
- [NLP System README](../NLP_SYSTEM_README.md)
- [Cloud Storage Guide](../docs/CLOUD_STORAGE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)

**Common Commands:**

```python
# Check environment
import os
print(os.getenv("VERCEL_POSTGRES_URL"))

# Test database
from sqlalchemy import create_engine, text
engine = create_engine(os.getenv("VERCEL_POSTGRES_URL"))
with engine.connect() as conn:
    print(conn.execute(text("SELECT version()")).fetchone())

# Run health checks
%run scripts/run_health_checks.py

# Verify dependencies
%run notebooks/00_verify_nlp_dependencies.py
```

**Still stuck?**
1. Check Deepnote console for errors (View → Console)
2. Review the setup notebook output
3. Ensure machine has been restarted
4. Verify environment variables in UI (⚙️)

---

**Remember:** No bash commands in Python cells! Use `%run` for scripts or the Deepnote UI for configuration.
