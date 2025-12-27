# 🚀 Quick Start for Deepnote (Updated)

**Clean 3-step setup - no bash errors!**

---

## Step 1: Add Environment Variables in Deepnote

### Click ⚙️ (gear icon) → Environment variables → + Add

You need to add **5 environment variables**:

### ✅ You Already Have (2/5):

1. **VERCEL_POSTGRES_URL**
   ```
   postgresql://user:pass@host.neon.tech/db?sslmode=require
   ```

2. **HF_TOKEN**
   ```
   hf_xxxxxxxxxxxxxxxxxxxx
   ```

### 📦 You Need to Get from Vercel (3/5):

**Run these commands on your LOCAL machine** (NOT in Deepnote):

```bash
# Install Vercel CLI (one-time)
npm install -g vercel

# Login and link project
vercel login
vercel link

# Create Blob storage
vercel blob create nlp-models

# Create KV cache
vercel kv create nlp-cache

# Get all tokens
vercel env pull
cat .env.local
```

**Then copy these 3 values to Deepnote:**

3. **BLOB_READ_WRITE_TOKEN**
   ```
   vercel_blob_rw_xxxxxxxxxxxx
   ```

4. **KV_REST_API_URL**
   ```
   https://xxxxx.kv.vercel-storage.com
   ```

5. **KV_REST_API_TOKEN**
   ```
   xxxxxxxxxxxxxxxxxx
   ```

**After adding all 5:** Click "..." menu → **Restart machine**

**Detailed guide:** [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)

---

## Step 2: Run Installation (5-10 minutes)

In Deepnote, create a new cell and run:

```python
%run notebooks/00_first_time_install.py
```

This will:
- ✅ Install all Python packages
- ✅ Install NLP models (spaCy, medspaCy)
- ✅ Test database connection
- ✅ Enable pgvector extension

**Expected output:**
```
✅ INSTALLATION COMPLETE
  ✅ Core dependencies
  ✅ NLP packages
  ✅ spaCy language models
  ✅ Database connection verified
```

**If you get errors:** Check that you added all environment variables and restarted the machine.

---

## Step 3: Verify Setup (1 minute)

Run the verification notebook:

```python
%run notebooks/00_verify_setup.py
```

This tests:
- ✅ All packages installed correctly
- ✅ spaCy models loaded
- ✅ Database connection works
- ✅ Cloud storage (if configured)

**Expected output:**
```
✅ SETUP VERIFICATION COMPLETE
  ✅ Environment variables
  ✅ Python packages
  ✅ spaCy models
  ✅ Database connection
  ✅ Core modules
```

---

## ✅ You're Ready!

Now run the NLP notebooks:

### Build Knowledge Base (one-time, 10-20 min)

```python
%run notebooks/07_nlp_knowledge_base_setup.py
```

Downloads ICD-10 and HCPCS codes, generates embeddings.

### Extract Codes from Claims

```python
%run notebooks/08_nlp_code_extraction.py
```

Extracts billing codes from clinical notes.

### Run Gap Analysis

```python
%run notebooks/09_gap_analysis_reporting.py
```

Identifies revenue leakage and compliance risks.

---

## 🆘 Troubleshooting

### "Missing required variables"

**Fix:**
1. Go to ⚙️ → Environment variables
2. Add all 5 variables listed in Step 1
3. Restart machine: ... → Restart machine
4. Re-run installation

### "Module not found"

**Fix:**
1. Restart machine: ... → Restart machine
2. Re-run installation: `%run notebooks/00_first_time_install.py`

### "Database connection failed"

**Fix:**
1. Check VERCEL_POSTGRES_URL in Deepnote env vars
2. Verify database is not paused in Vercel dashboard
3. Ensure connection string ends with `?sslmode=require`

### "I don't have Vercel CLI"

**Option 1: Skip Blob/KV for now**
- You can still use the system with just VERCEL_POSTGRES_URL and HF_TOKEN
- Add Blob/KV tokens later when ready for production

**Option 2: Install Vercel CLI**
- Run on your local machine: `npm install -g vercel`
- Not available in Deepnote (it's Python-only)

---

## 📚 Documentation

- **Complete env vars guide:** [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)
- **Deepnote setup:** [docs/DEEPNOTE_SETUP.md](docs/DEEPNOTE_SETUP.md)
- **NLP system:** [NLP_SYSTEM_README.md](NLP_SYSTEM_README.md)

---

## 📋 What Changed from Before

**OLD (had errors):**
- ❌ Mixed installation and testing in one notebook
- ❌ Bash commands in Python cells (syntax errors)
- ❌ Incomplete documentation on environment variables

**NEW (clean):**
- ✅ Separate installation notebook (`00_first_time_install.py`)
- ✅ Separate verification notebook (`00_verify_setup.py`)
- ✅ Complete list of all 5 required environment variables
- ✅ Clear instructions for getting Vercel tokens
- ✅ No bash commands in Python notebooks

---

## 🎯 Summary

**3 simple steps:**

1. **Add 5 environment variables in Deepnote UI** (⚙️ icon)
2. **Run installation:** `%run notebooks/00_first_time_install.py`
3. **Verify setup:** `%run notebooks/00_verify_setup.py`

**Then start using the system!**

Need help? See [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)
