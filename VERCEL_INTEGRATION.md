# 🚀 Vercel Integration Complete

Your medical billing NLP system now has **fully automated Vercel deployment** with comprehensive health monitoring!

## ✅ What's Been Added

### 1. Automated Service Provisioning

**File:** `scripts/vercel_setup.py`

Automatically creates and configures:
- ✅ Vercel Blob storage (`nlp-models`)
- ✅ Vercel KV cache (`nlp-cache`)
- ✅ Postgres pgvector extension
- ✅ Environment variable synchronization

**How it works:**
```bash
# Runs automatically during Vercel build
python scripts/vercel_setup.py

# Or run manually for local testing
vercel login
python scripts/vercel_setup.py
```

### 2. Comprehensive Health Checks

**File:** `scripts/run_health_checks.py`

Tests every data system at each deployment:

| System | What's Checked | Pass Criteria |
|--------|---------------|---------------|
| **Postgres** | Connection, version, tables | Can execute queries |
| **pgvector** | Extension version, HNSW index | Vector search works |
| **Blob Storage** | Read/write operations | Upload/download successful |
| **KV Cache** | Get/set/delete operations | Cache operations work |
| **HuggingFace** | Model access, API | Can access BioClinical BERT |

**Usage:**
```bash
# Standard health check
python scripts/run_health_checks.py

# Verbose output (shows all checks)
python scripts/run_health_checks.py --verbose

# JSON output (for CI/CD)
python scripts/run_health_checks.py --json

# Exit on first failure
python scripts/run_health_checks.py --fail-fast
```

**Example Output:**
```
🏥 HEALTH CHECK RESULTS
============================================================
✅ Vercel Postgres: PASS (4/4) [0.25s]
  ✓ Database connection - Connected successfully
  ✓ PostgreSQL version - 15.3
  ✓ Tables exist - 12 tables found

✅ pgvector Extension: PASS (4/4) [0.18s]
  ✓ Extension installed - version 0.5.1
  ✓ Embeddings loaded - 12,450 codes
  ✓ Vector search - Similarity search working

✅ ALL SYSTEMS HEALTHY (18/18 checks passed)
============================================================
```

### 3. CI/CD Pipeline (GitHub Actions)

**File:** `.github/workflows/test.yml`

Runs automatically on:
- 📝 Every push to `main` or `claude/**` branches
- 🔀 Every pull request
- ⏰ Every 6 hours (scheduled monitoring)
- 🔘 Manual trigger via GitHub UI

**Pipeline Jobs:**

```
┌─────────────────────────────────┐
│  Job 1: Health Checks           │
│  • Test all Vercel services     │
│  • Export JSON results          │
│  • Upload artifacts             │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│  Job 2: Dependency Checks       │
│  • Run setup_nlp_system.py      │
│  • Verify all NLP packages      │
│  • Test module imports          │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│  Job 3: Unit Tests              │
│  • Run pytest suite             │
│  • Generate coverage report     │
│  • Upload to codecov            │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│  Job 4: Deployment Readiness    │
│  • Verify required files        │
│  • Validate vercel.json         │
│  • Check env var docs           │
└─────────────────────────────────┘
```

### 4. Vercel Configuration

**File:** `vercel.json`

Complete deployment configuration:

```json
{
  "buildCommand": "bash scripts/vercel_predeploy.sh",
  "installCommand": "pip install -r requirements.txt",
  "functions": {
    "api/**/*.py": {
      "runtime": "python3.11",
      "maxDuration": 60
    }
  },
  "env": {
    "VERCEL_POSTGRES_URL": "@postgres-url",
    "BLOB_READ_WRITE_TOKEN": "@blob-token",
    "KV_REST_API_URL": "@kv-url",
    "KV_REST_API_TOKEN": "@kv-token"
  }
}
```

### 5. Unit Test Suite

**File:** `tests/test_storage_health.py`

Comprehensive tests for:
- CloudModelCache functionality
- BlobStorage operations
- Database connectivity
- ClinicalCodeObject serialization
- GapAnalyzer validation

**Run tests:**
```bash
pytest tests/ -v --cov=src
```

## 🎯 Quick Start

### For Deepnote Users

**You're in Deepnote!** Don't run bash commands. Instead:

1. **Open the Deepnote setup notebook:**
   ```python
   # In a Deepnote cell, run:
   %run notebooks/00_deepnote_setup.py
   ```

2. **Follow the interactive guide** - it will walk you through:
   - Adding environment variables in Deepnote UI (⚙️ → Environment variables)
   - Installing dependencies
   - Verifying setup
   - Testing database connection

**See detailed guide:** [docs/DEEPNOTE_SETUP.md](docs/DEEPNOTE_SETUP.md)

---

### For Local/Production Deployment

If you're deploying to Vercel (not using Deepnote), follow this:

1. **Link Vercel Project**
   ```bash
   npm install -g vercel
   vercel login
   vercel link
   ```

2. **Configure GitHub Secrets**

   Go to: `Repository → Settings → Secrets → Actions`

   Add these secrets:
   - `VERCEL_POSTGRES_URL`
   - `BLOB_READ_WRITE_TOKEN`
   - `KV_REST_API_URL`
   - `KV_REST_API_TOKEN`
   - `HF_TOKEN` (optional)

3. **Initial Deployment**
   ```bash
   # Setup services
   python scripts/vercel_setup.py

   # Run health checks
   python scripts/run_health_checks.py --verbose

   # Deploy
   vercel --prod
   ```

### Every Deployment

The system automatically:
1. ✅ Provisions Blob/KV if needed
2. ✅ Runs health checks on all systems
3. ✅ Warms model cache
4. ✅ Validates deployment readiness
5. ✅ Deploys to production

**You just need:**
```bash
git push origin main
```

That's it! GitHub Actions and Vercel handle the rest.

## 📊 Monitoring

### GitHub Actions

View test results:
```
GitHub → Actions → Latest workflow run
```

- See health check status for every push
- Download JSON artifacts for historical analysis
- Get notified on failures (configure in repo settings)

### Vercel Dashboard

Monitor deployments:
```
Vercel Dashboard → Your Project → Deployments
```

- View build logs with health check output
- See function invocation metrics
- Monitor storage usage (Blob, KV, Postgres)

### Scheduled Monitoring

Health checks run every 6 hours automatically:
- Detects issues proactively
- No manual intervention needed
- Alerts if systems become unhealthy

## 🔧 Troubleshooting

### "BLOB_READ_WRITE_TOKEN not set"

```bash
# Re-pull environment variables
vercel env pull

# Or create Blob storage manually
vercel blob create nlp-models
```

### "Health check failed: Postgres connection"

```bash
# Verify connection string
echo $VERCEL_POSTGRES_URL

# Test manually
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('$VERCEL_POSTGRES_URL')
with engine.connect() as conn:
    print(conn.execute(text('SELECT 1')).fetchone())
"
```

### "pgvector extension not found"

```bash
# Enable extension
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('$VERCEL_POSTGRES_URL')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()
    print('✅ pgvector enabled')
"
```

### GitHub Actions failing

1. Check secrets are configured:
   ```
   Settings → Secrets → Actions
   ```

2. Verify all required secrets exist:
   - VERCEL_POSTGRES_URL
   - BLOB_READ_WRITE_TOKEN
   - KV_REST_API_URL
   - KV_REST_API_TOKEN

3. Re-run failed jobs:
   ```
   Actions → Failed workflow → Re-run failed jobs
   ```

## 💰 Cost Analysis

All within free tiers:

| Service | Free Tier | Estimated Usage | Cost |
|---------|-----------|-----------------|------|
| Vercel Postgres | 500MB (Neon) | ~50MB | **$0** |
| Vercel Blob | 1GB | ~400MB (models) | **$0** |
| Vercel KV | 100K ops/day | ~5K ops/day | **$0** |
| GitHub Actions | 2000 min/month | ~20 min/month | **$0** |
| Vercel Build | 100 hrs/month | ~2 hrs/month | **$0** |

**Total: $0/month** for typical usage

Monitor usage:
- Vercel Dashboard → Usage tab
- GitHub → Settings → Billing → Actions usage

## 📚 Documentation

- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Complete deployment guide
- **[CLOUD_STORAGE.md](docs/CLOUD_STORAGE.md)** - Cloud caching architecture
- **[NLP_SYSTEM_README.md](NLP_SYSTEM_README.md)** - NLP system documentation

## 🎉 What This Enables

### Before
- ❌ Manual Blob/KV setup for each developer
- ❌ No automated health monitoring
- ❌ Deploy and hope everything works
- ❌ Manual cache warming
- ❌ No deployment validation

### After
- ✅ **Automatic** service provisioning
- ✅ **Comprehensive** health checks on every deploy
- ✅ **Proactive** monitoring (every 6 hours)
- ✅ **Automated** cache warming
- ✅ **Validated** deployments (fail fast on issues)
- ✅ **Zero-cost** infrastructure
- ✅ **Audit trail** of all health checks

## 🚦 Status Indicators

You can now add badges to your README:

```markdown
[![Health Checks](https://github.com/aliomraniH/billing-model/actions/workflows/test.yml/badge.svg)](https://github.com/aliomraniH/billing-model/actions/workflows/test.yml)
```

## 🔗 Quick Links

- [Vercel Dashboard](https://vercel.com/dashboard)
- [GitHub Actions](https://github.com/aliomraniH/billing-model/actions)
- [Health Check Workflow](.github/workflows/test.yml)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

**Next Steps:**
1. Configure GitHub secrets (see setup above)
2. Link Vercel project: `vercel link`
3. Deploy: `git push origin main`
4. Monitor: GitHub Actions + Vercel Dashboard

**Questions?** See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed troubleshooting.
