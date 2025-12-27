# Complete Environment Variables Guide for Deepnote

This document lists **ALL** environment variables you need to add in Deepnote.

## 🎯 How to Add Environment Variables in Deepnote

1. **Click the ⚙️ (gear icon)** in the left sidebar
2. **Select "Environment variables"**
3. **Click "+ Add environment variable"**
4. Enter the name and value for each variable below
5. **Click "Save"** for each one
6. **After adding all variables: Restart machine** (... menu → Restart machine)

---

## ✅ Variables You Already Have

### 1. VERCEL_POSTGRES_URL
```
Name: VERCEL_POSTGRES_URL
Value: postgresql://user:pass@host.neon.tech/db?sslmode=require
```
**Status:** ✅ Already added
**Source:** Vercel Dashboard → Storage → Postgres → .env.local

---

### 2. HF_TOKEN
```
Name: HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxx
```
**Status:** ✅ Already added
**Source:** https://huggingface.co/settings/tokens

---

## 📦 Variables You Need to Add (Get from Vercel)

These 3 variables enable cloud caching for faster model loading in production.

### 3. BLOB_READ_WRITE_TOKEN

```
Name: BLOB_READ_WRITE_TOKEN
Value: vercel_blob_rw_xxxxxxxxxxxxxxxxx
```

**What it's for:** Stores large files (models, embeddings) in Vercel Blob storage

**How to get it:**

**On your LOCAL machine** (NOT in Deepnote):

```bash
# 1. Install Vercel CLI (one-time)
npm install -g vercel

# 2. Login to Vercel
vercel login

# 3. Link to your project
vercel link

# 4. Create Blob storage
vercel blob create nlp-models

# 5. Pull environment variables
vercel env pull

# 6. Get the token
cat .env.local | grep BLOB_READ_WRITE_TOKEN
```

**Example output:**
```
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_ABC123xyz..."
```

**Copy the value** (without quotes) and add to Deepnote environment variables.

---

### 4. KV_REST_API_URL

```
Name: KV_REST_API_URL
Value: https://xxxxxx-xxxxxx.kv.vercel-storage.com
```

**What it's for:** Fast key-value cache for model metadata

**How to get it:**

**On your LOCAL machine**:

```bash
# 1. Create KV cache
vercel kv create nlp-cache

# 2. Pull environment variables
vercel env pull

# 3. Get the URL
cat .env.local | grep KV_REST_API_URL
```

**Example output:**
```
KV_REST_API_URL="https://fluent-rabbit-12345.kv.vercel-storage.com"
```

**Copy the URL** and add to Deepnote.

---

### 5. KV_REST_API_TOKEN

```
Name: KV_REST_API_TOKEN
Value: xxxxxxxxxxxxxxxxxxxxxxxxxx
```

**What it's for:** Authentication for KV cache

**How to get it:**

**On your LOCAL machine**:

```bash
# Get the token (after running 'vercel kv create' above)
cat .env.local | grep KV_REST_API_TOKEN
```

**Example output:**
```
KV_REST_API_TOKEN="AbCdEf123456..."
```

**Copy the token** and add to Deepnote.

---

## 📋 Complete Checklist

After adding all variables, you should have:

- [x] ✅ **VERCEL_POSTGRES_URL** - Database connection (you have this)
- [x] ✅ **HF_TOKEN** - HuggingFace API (you have this)
- [ ] 📦 **BLOB_READ_WRITE_TOKEN** - Vercel Blob storage (get from Vercel)
- [ ] 📦 **KV_REST_API_URL** - KV cache URL (get from Vercel)
- [ ] 📦 **KV_REST_API_TOKEN** - KV cache auth (get from Vercel)

---

## 🚀 Quick Command Reference

**All commands run on LOCAL machine, not Deepnote:**

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

# Pull all environment variables
vercel env pull

# View all tokens
cat .env.local
```

---

## ⚠️ Important Notes

1. **Run Vercel CLI commands on your LOCAL machine**, not in Deepnote
   - Deepnote is for Python notebooks only
   - Vercel CLI needs Node.js/npm (not available in Deepnote)

2. **Production vs Development:**
   - **Required for development:** VERCEL_POSTGRES_URL, HF_TOKEN
   - **Optional for development:** BLOB, KV tokens
   - **Required for production:** All 5 variables

3. **Security:**
   - Never commit these tokens to git
   - Only add them in Deepnote UI (⚙️ → Environment variables)
   - Deepnote encrypts environment variables

4. **After adding variables:**
   - **Must restart machine** for Deepnote to load them
   - ... menu → Restart machine
   - Wait for restart to complete (~30 seconds)

---

## 🔧 Alternative: Skip Blob/KV for Now

If you don't want to set up Vercel CLI right now, you can skip the Blob/KV variables:

**What works without them:**
- ✅ Database queries (Postgres)
- ✅ Code extraction (NLP)
- ✅ Gap analysis
- ✅ All core functionality

**What you'll miss:**
- ⚠️ Slower model loading (downloads each time instead of cache)
- ⚠️ No cloud caching benefits

**You can add them later** when you're ready for production deployment.

---

## 🎯 Next Steps

After adding all environment variables:

1. **Restart machine** (... menu → Restart machine)

2. **Run first-time installation:**
   ```python
   %run notebooks/00_first_time_install.py
   ```

3. **Verify setup:**
   ```python
   %run notebooks/00_verify_setup.py
   ```

4. **Start using the system!**

---

## 📚 Documentation

- **Installation Guide:** [docs/DEEPNOTE_SETUP.md](DEEPNOTE_SETUP.md)
- **Vercel Integration:** [VERCEL_INTEGRATION.md](../VERCEL_INTEGRATION.md)
- **Cloud Storage:** [CLOUD_STORAGE.md](CLOUD_STORAGE.md)
