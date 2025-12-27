#!/bin/bash
# Vercel Pre-Deployment Hook
# Runs automatically before each Vercel deployment

set -e  # Exit on error

echo "="
echo "🔧 Vercel Pre-Deployment Setup"
echo "="

# 1. Setup Vercel services
echo "📦 Setting up Vercel services..."
python scripts/vercel_setup.py

# 2. Run health checks
echo "🏥 Running health checks..."
python scripts/run_health_checks.py --json > /tmp/health-check.json || {
    echo "⚠️  Health checks failed, but continuing deployment"
    echo "   Review logs in Vercel dashboard"
}

# 3. Warm cache (optional, only if all services healthy)
if [ -f /tmp/health-check.json ]; then
    HEALTH_STATUS=$(python -c "import json; data=json.load(open('/tmp/health-check.json')); print(data['overall_status'])")

    if [ "$HEALTH_STATUS" = "healthy" ]; then
        echo "✅ All services healthy, warming cache..."
        python scripts/deploy_warm_cache.py || {
            echo "⚠️  Cache warming failed, continuing deployment"
        }
    else
        echo "⚠️  Skipping cache warming (services not fully healthy)"
    fi
fi

echo "✅ Pre-deployment setup complete"
