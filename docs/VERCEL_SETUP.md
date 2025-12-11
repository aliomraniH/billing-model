# Vercel Postgres (Neon) Setup Guide

## Overview

This guide walks you through setting up Vercel Postgres (powered by Neon) for the Medical Billing ML project. The database will store claims, diagnoses, procedures, clinical notes with embeddings, and model predictions.

## Prerequisites

- A Vercel account (free tier available)
- A web browser
- Google Colab account (for running notebooks)

## Estimated Time

- **Account setup:** 5 minutes
- **Database creation:** 2 minutes
- **Connection testing:** 3 minutes
- **Total:** ~10 minutes

---

## Step 1: Create Vercel Account

1. Go to https://vercel.com/signup
2. Sign up with GitHub, GitLab, Bitbucket, or email
3. Complete email verification if required
4. You'll be redirected to your Vercel dashboard

**Free Tier Includes:**
- 512 MB storage
- 60 hours compute per month
- Unlimited queries (within fair use)

---

## Step 2: Create Postgres Database

### 2.1 Navigate to Storage

1. From Vercel dashboard, click **Storage** in the left sidebar
2. Click **Create Database** button
3. Select **Postgres** (powered by Neon)

### 2.2 Configure Database

**Database Settings:**
- **Name:** `medical-billing-ml` (or your preferred name)
- **Region:** Choose closest to your location
  - US East (recommended for North America)
  - EU West (recommended for Europe)
  - Asia Pacific (recommended for Asia/Australia)
- **Primary Branch:** `main` (default)

**Why region matters:**
- Lower latency from Google Colab (US-based servers)
- Faster data loading and queries
- Recommended: **US East (N. Virginia)** for best Colab performance

### 2.3 Create Database

1. Click **Create** button
2. Wait 10-20 seconds for provisioning
3. You'll see "Database created successfully"

---

## Step 3: Get Connection Strings

### 3.1 Access Connection Details

1. Click on your newly created database
2. Navigate to **Settings** tab
3. Scroll to **Connection String** section

### 3.2 Copy Connection Strings

You'll see multiple connection strings:

**IMPORTANT:** You need the **DIRECT** connection string (not pooled)

```
┌─────────────────────────────────────────────────────────────┐
│ Connection Strings                                          │
├─────────────────────────────────────────────────────────────┤
│ POSTGRES_URL (Pooled)                                       │
│ postgresql://user:pass@aws-0-us-east-1.pooler.neon.tech/db │
│                                                              │
│ POSTGRES_URL_NON_POOLING (Direct) ✅ USE THIS              │
│ postgresql://user:pass@aws-0-us-east-1.neon.tech:5432/db   │
└─────────────────────────────────────────────────────────────┘
```

**Copy the DIRECT connection string** (POSTGRES_URL_NON_POOLING)
- Look for the one **without** `.pooler.` in the hostname
- Should have `:5432` port specified
- Format: `postgresql://username:password@host:5432/dbname?sslmode=require`

### 3.3 Why Direct Connection?

| Connection Type | When to Use | Why NOT for Our Project |
|-----------------|-------------|-------------------------|
| **Pooled** | High-traffic web apps | Connection limits, no pgvector support |
| **Direct** ✅ | Data pipelines, analytics | Full Postgres features, pgvector works |

---

## Step 4: Store Connection String Securely

### Option A: Google Colab Secrets (Recommended)

1. Open Google Colab: https://colab.research.google.com
2. Click 🔑 **Secrets** icon in left sidebar
3. Click **+ Add new secret**
4. Enter:
   - **Name:** `VERCEL_POSTGRES_URL`
   - **Value:** `postgresql://...` (paste your direct connection string)
5. Toggle **Notebook access** ON
6. Click **Save**

### Option B: Local `.env` File (For Local Development)

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and paste your connection string:
   ```bash
   VERCEL_POSTGRES_URL=postgresql://user:pass@host:5432/db?sslmode=require
   HF_TOKEN=your_huggingface_token_here
   ```

3. **NEVER commit `.env` to git** (already in `.gitignore`)

---

## Step 5: Verify Connection

### 5.1 Using Google Colab

1. Create new Colab notebook or use existing
2. Run this code:

```python
#@title 🔍 Test Database Connection
from google.colab import userdata
from sqlalchemy import create_engine, text
import sys

try:
    # Get connection string from secrets
    DATABASE_URL = userdata.get('VERCEL_POSTGRES_URL')

    if not DATABASE_URL:
        print("❌ ERROR: VERCEL_POSTGRES_URL not found in Colab secrets!")
        print("👉 Add it via the 🔑 Secrets sidebar")
        sys.exit(1)

    # Verify format
    if '.pooler.' in DATABASE_URL:
        print("⚠️  WARNING: You're using a POOLED connection string!")
        print("👉 Switch to POSTGRES_URL_NON_POOLING (direct connection)")

    # Create engine
    engine = create_engine(DATABASE_URL)

    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]

        print("✅ SUCCESS! Connected to Vercel Postgres")
        print(f"📊 Database version: {version[:60]}...")

        # Check pgvector availability
        result = conn.execute(text("""
            SELECT COUNT(*) FROM pg_available_extensions
            WHERE name = 'vector'
        """))
        has_vector = result.fetchone()[0] > 0

        if has_vector:
            print("✅ pgvector extension available")
        else:
            print("⚠️  pgvector not found (will be enabled in setup script)")

except Exception as e:
    print(f"❌ CONNECTION FAILED: {str(e)}")
    print("\n🔧 Troubleshooting:")
    print("1. Check connection string is correct")
    print("2. Ensure you used DIRECT connection (not pooled)")
    print("3. Verify database is running in Vercel dashboard")
    print("4. Check for typos in secret name (VERCEL_POSTGRES_URL)")
```

### 5.2 Expected Output

**Success:**
```
✅ SUCCESS! Connected to Vercel Postgres
📊 Database version: PostgreSQL 15.3 on x86_64-pc-linux-gnu...
✅ pgvector extension available
```

**If pooled connection detected:**
```
⚠️  WARNING: You're using a POOLED connection string!
👉 Switch to POSTGRES_URL_NON_POOLING (direct connection)
```

**If connection fails:**
```
❌ CONNECTION FAILED: connection timeout
🔧 Troubleshooting:
1. Check connection string is correct
2. Ensure you used DIRECT connection (not pooled)
3. Verify database is running in Vercel dashboard
4. Check for typos in secret name (VERCEL_POSTGRES_URL)
```

---

## Step 6: Enable pgvector Extension

Once connection is verified, run this in Colab:

```python
#@title 📦 Enable pgvector Extension
from google.colab import userdata
from sqlalchemy import create_engine, text

DATABASE_URL = userdata.get('VERCEL_POSTGRES_URL')
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Enable pgvector
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

    # Verify
    result = conn.execute(text("""
        SELECT installed_version FROM pg_available_extensions
        WHERE name = 'vector'
    """))
    version = result.fetchone()[0]

    print(f"✅ pgvector {version} enabled successfully!")
```

**Expected output:**
```
✅ pgvector 0.8.0 enabled successfully!
```

---

## Step 7: Create Schema

Now run the full setup notebook:

```python
# In Google Colab, copy-paste notebooks/01_setup_database.py
# This will:
# 1. Enable pgvector
# 2. Create all 5 tables (claims, diagnoses, procedures, clinical_notes, predictions)
# 3. Create indexes
# 4. Verify tables exist
```

**Expected output:**
```
✅ Dependencies installed!
✅ Connected to: PostgreSQL 15.3...
✅ Schema created successfully!
📋 Tables: claims, clinical_notes, diagnoses, predictions, procedures
```

---

## Troubleshooting

### Issue 1: "connection timeout"

**Cause:** Wrong connection string or network issue

**Fix:**
1. Verify you copied the **direct** connection string (not pooled)
2. Check connection string doesn't have extra spaces
3. Ensure database is "Active" in Vercel dashboard (Settings → Status)

### Issue 2: "password authentication failed"

**Cause:** Incorrect username/password in connection string

**Fix:**
1. Re-copy connection string from Vercel dashboard
2. Click "Reveal" to see full password
3. Ensure no URL encoding issues (special characters)

### Issue 3: "database does not exist"

**Cause:** Database name in connection string doesn't match

**Fix:**
1. Check database name in Vercel dashboard (Settings → General)
2. Verify connection string has correct `/dbname` at the end

### Issue 4: "too many clients already"

**Cause:** Connection limit reached (free tier: 20 connections)

**Fix:**
1. Close old Colab sessions
2. Wait 5 minutes for connections to close
3. Use connection pooling for production (but not for setup)

### Issue 5: "pgvector extension not available"

**Cause:** Using old Vercel Postgres version

**Fix:**
1. pgvector is built-in for all new Vercel Postgres databases
2. If not available, database may be too old (created before pgvector support)
3. Solution: Create new database (pgvector can't be added retroactively)

---

## Database Management

### View Database in Browser

1. Go to Vercel dashboard → Storage → Your database
2. Click **Query** tab
3. Run SQL directly:

```sql
-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Count records
SELECT
    (SELECT COUNT(*) FROM claims) as claims_count,
    (SELECT COUNT(*) FROM diagnoses) as diagnoses_count,
    (SELECT COUNT(*) FROM clinical_notes) as notes_count;
```

### Monitor Usage

1. Vercel dashboard → Storage → Your database
2. Click **Usage** tab
3. Monitor:
   - Storage used (512 MB limit on free tier)
   - Compute hours used (60 hours/month on free tier)
   - Active connections

### Upgrade to Pro (When Needed)

**Free Tier Limits:**
- 512 MB storage
- 60 compute hours/month
- 20 concurrent connections

**Pro Tier ($20/month):**
- 10 GB storage
- 100 compute hours/month
- 60 concurrent connections

**Upgrade when:**
- Storage > 400 MB (80% capacity)
- Compute hours > 50/month
- Need more than 20 connections

---

## Security Best Practices

### ✅ Do's

- ✅ Use Colab Secrets for connection strings
- ✅ Use direct (non-pooled) connections for data pipelines
- ✅ Enable SSL/TLS (included in connection string: `?sslmode=require`)
- ✅ Rotate credentials if accidentally exposed
- ✅ Use read-only users for analytics queries (future)

### ❌ Don'ts

- ❌ Never commit connection strings to git
- ❌ Don't share connection strings in Slack/Discord
- ❌ Don't use pooled connections for pgvector queries
- ❌ Don't store PHI (Protected Health Information) without proper compliance
- ❌ Don't expose database publicly (Vercel uses IP allowlisting)

---

## Next Steps

Once your database is set up:

1. ✅ **Run notebooks in sequence:**
   - `01_setup_database.py` (you just did this!)
   - `02_load_synthea_data.py` (load synthetic claims)
   - `03_outlier_detection.py` (train model)
   - `04_huggingface_autotrain.py` (XGBoost)
   - `05_embeddings_search.py` (similarity search)

2. ✅ **Get Hugging Face token:**
   - Sign up at https://huggingface.co
   - Create write token
   - Add to Colab secrets as `HF_TOKEN`

3. ✅ **Monitor database growth:**
   - Check storage usage weekly
   - Plan upgrade before hitting 512 MB limit

---

## Support & Resources

- **Vercel Docs:** https://vercel.com/docs/storage/vercel-postgres
- **Neon Docs:** https://neon.tech/docs/introduction
- **pgvector Docs:** https://github.com/pgvector/pgvector
- **Troubleshooting:** See [ROADMAP.md](ROADMAP.md) for common issues

---

**Status:** Ready to proceed to data loading (Week 2)
