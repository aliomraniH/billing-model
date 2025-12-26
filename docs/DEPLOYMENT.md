# Vercel Deployment & Health Checks

This guide covers automated Vercel deployment with integrated health checks.

## Overview

The deployment pipeline automatically:
1. Provisions Vercel Blob and KV storage
2. Runs comprehensive health checks
3. Warms model cache for fast cold starts
4. Validates all systems before going live

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GitHub Push / PR                           │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│           GitHub Actions CI/CD                          │
│  • Health checks (all storage systems)                  │
│  • Dependency verification                              │
│  • Unit tests                                           │
│  • Deployment readiness validation                      │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│           Vercel Build Process                          │
│  1. Run vercel_setup.py (provision services)            │
│  2. Install dependencies                                │
│  3. Run health checks                                   │
│  4. Warm cache (if healthy)                             │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│           Production Deployment                         │
│  ✅ All systems verified and cached                     │
└─────────────────────────────────────────────────────────┘
```

## Setup

### 1. Configure GitHub Secrets

In your GitHub repository settings, add these secrets:

```
Settings → Secrets and variables → Actions → New repository secret
```

Required secrets:
- `VERCEL_POSTGRES_URL`: Your Vercel Postgres connection string
- `BLOB_READ_WRITE_TOKEN`: Vercel Blob storage token
- `KV_REST_API_URL`: Vercel KV REST API URL
- `KV_REST_API_TOKEN`: Vercel KV API token
- `HF_TOKEN`: HuggingFace API token (optional)

### 2. Link Vercel Project

```bash
# Install Vercel CLI
npm install -g vercel

# Login and link project
vercel login
vercel link

# Pull environment variables
vercel env pull
```

### 3. Enable GitHub Actions

The workflow file `.github/workflows/test.yml` will automatically run on:
- Push to `main` or `claude/**` branches
- Pull requests to `main`
- Every 6 hours (scheduled health checks)
- Manual trigger via GitHub UI

## Manual Deployment

### Local Pre-Deployment Checks

Before deploying, run locally:

```bash
# 1. Setup Vercel services
python scripts/vercel_setup.py

# 2. Run health checks
python scripts/run_health_checks.py --verbose

# 3. Warm cache
python scripts/deploy_warm_cache.py
```

### Deploy to Vercel

```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

## Health Check Details

### What's Checked

| System | Checks |
|--------|--------|
| **Vercel Postgres** | Connection, version, table existence |
| **pgvector** | Extension installed, version, vector search |
| **Vercel Blob** | Read/write operations, connectivity |
| **Vercel KV** | Get/set/delete operations |
| **HuggingFace** | Model access, API connectivity |

### Running Health Checks

**Locally:**
```bash
# Standard check
python scripts/run_health_checks.py

# Verbose output
python scripts/run_health_checks.py --verbose

# JSON output for CI/CD
python scripts/run_health_checks.py --json

# Fail fast (exit on first failure)
python scripts/run_health_checks.py --fail-fast
```

**In GitHub Actions:**
- Automatically runs on every push/PR
- Results uploaded as artifacts
- Fails the build if any critical system is down

**In Vercel:**
- Runs during build process
- Logged in Vercel dashboard → Deployments → Build Logs
- Non-critical failures logged as warnings

### Example Output

```
🚀 Starting comprehensive health checks...

🔍 Checking Postgres... ✅
🔍 Checking pgvector... ✅
🔍 Checking Blob... ✅
🔍 Checking KV... ✅
🔍 Checking HuggingFace... ✅

============================================================
🏥 HEALTH CHECK RESULTS
============================================================
Timestamp: 2025-12-26T20:00:00Z
============================================================

✅ Vercel Postgres: PASS (4/4) [0.25s]
  ✓ Environment variable - VERCEL_POSTGRES_URL found
  ✓ Database connection - Connected successfully
  ✓ PostgreSQL version - 15.3
  ✓ Tables exist - 12 tables found

✅ pgvector Extension: PASS (4/4) [0.18s]
  ✓ Extension installed - version 0.5.1
  ✓ code_embeddings table - Table exists
  ✓ Embeddings loaded - 12,450 codes
  ✓ Vector search - Similarity search working

✅ Vercel Blob Storage: PASS (2/2) [0.42s]
  ✓ Environment variable - Token found
  ✓ Write operation - Test upload successful
  ✓ Read operation - Test download successful

✅ Vercel KV Cache: PASS (2/2) [0.15s]
  ✓ Environment variables - Credentials found
  ✓ Write operation - Test set successful
  ✓ Read operation - Test get successful

✅ HuggingFace Hub: PASS (2/2) [1.23s]
  ✓ Environment variable - Token found
  ✓ Model access - NeuML/bioclinical-modernbert-base-embeddings

============================================================
✅ ALL SYSTEMS HEALTHY (18/18 checks passed)
============================================================
```

## Continuous Monitoring

### Scheduled Health Checks

GitHub Actions runs health checks every 6 hours to detect issues proactively:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
```

View results:
- GitHub → Actions → Latest "Health Checks & Tests" run
- Download artifacts for detailed JSON reports

### Vercel Cron Jobs

Configure in `vercel.json`:

```json
"crons": [
  {
    "path": "/api/health",
    "schedule": "0 */6 * * *"
  }
]
```

Create `api/health.py`:

```python
from http.server import BaseHTTPRequestHandler
import subprocess
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Run health checks
        result = subprocess.run(
            ["python", "scripts/run_health_checks.py", "--json"],
            capture_output=True,
            text=True
        )

        self.send_response(200 if result.returncode == 0 else 503)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(result.stdout.encode())
```

## Troubleshooting

### Health Check Failures

**Postgres connection failed:**
```bash
# Verify environment variable
echo $VERCEL_POSTGRES_URL

# Test connection manually
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('$VERCEL_POSTGRES_URL')
with engine.connect() as conn:
    print(conn.execute(text('SELECT version()')).fetchone())
"
```

**pgvector not found:**
```bash
# Enable extension
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('$VERCEL_POSTGRES_URL')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()
"
```

**Blob/KV authentication failed:**
```bash
# Re-pull environment variables
vercel env pull

# Verify tokens are set
grep -E 'BLOB_|KV_' .env
```

### Deployment Failures

**Build command failed:**
- Check Vercel dashboard → Deployments → Build Logs
- Run locally: `python scripts/vercel_setup.py`
- Verify all dependencies in requirements.txt

**Health checks timing out:**
- Increase timeout in `vercel.json`: `"maxDuration": 120`
- Run health checks with `--fail-fast` to identify slow operations

## Cost Monitoring

All services have generous free tiers:

| Service | Free Tier | Estimated Usage | Cost |
|---------|-----------|-----------------|------|
| Postgres | 500MB (Neon) | ~50MB | $0 |
| Blob Storage | 1GB | ~400MB models | $0 |
| KV Cache | 100K ops/day | ~5K ops/day | $0 |
| Vercel Build | 100 hrs/month | ~2 hrs/month | $0 |

**Total estimated cost: $0/month** for typical usage.

Monitor in Vercel dashboard → Usage tab.

## Best Practices

1. **Always run health checks before major deployments**
   ```bash
   python scripts/run_health_checks.py --verbose
   ```

2. **Review health check artifacts in GitHub Actions**
   - Download JSON reports for historical analysis
   - Track trends in check duration and failure rates

3. **Set up alerts for production**
   - Use Vercel integrations (Slack, PagerDuty, etc.)
   - Monitor health check cron endpoint

4. **Warm cache after code changes**
   ```bash
   vercel --prod
   python scripts/deploy_warm_cache.py
   ```

5. **Test locally before pushing**
   ```bash
   python scripts/run_health_checks.py --fail-fast
   pytest tests/ -v
   ```

## References

- [Vercel Build Configuration](https://vercel.com/docs/build-output-api/v3)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Vercel Cron Jobs](https://vercel.com/docs/cron-jobs)
