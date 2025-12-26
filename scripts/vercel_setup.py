#!/usr/bin/env python3
"""
Vercel Automated Setup
=======================
Automatically provisions and configures all required Vercel services.
Runs during deployment build phase.

Creates:
- Vercel Blob storage (nlp-models)
- Vercel KV cache (nlp-cache)
- Vercel Postgres database (if not exists)
- pgvector extension

Environment Variables Required:
- VERCEL_TOKEN: Vercel API token for provisioning

Usage:
    python scripts/vercel_setup.py
"""

import os
import sys
import json
import subprocess
from typing import Optional, Dict

def run_vercel_command(cmd: str) -> Optional[Dict]:
    """Run a Vercel CLI command and return JSON output."""
    try:
        result = subprocess.run(
            f"vercel {cmd}",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        # Try to parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"output": result.stdout, "success": True}
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def check_vercel_cli():
    """Ensure Vercel CLI is installed."""
    try:
        result = subprocess.run(
            "vercel --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Vercel CLI: {result.stdout.strip()}")
            return True
    except:
        pass

    print("❌ Vercel CLI not found. Installing...")
    try:
        subprocess.run("npm install -g vercel", shell=True, check=True)
        print("✅ Vercel CLI installed")
        return True
    except:
        print("❌ Failed to install Vercel CLI")
        return False

def provision_blob_storage() -> bool:
    """Create Vercel Blob storage if not exists."""
    print("\n📦 Provisioning Vercel Blob storage...")

    # Check if already exists
    result = run_vercel_command("blob ls")
    if result and result.get("success"):
        print("♻️  Blob storage already exists")
        return True

    # Create new blob store
    result = run_vercel_command("blob create nlp-models --yes")
    if result:
        print("✅ Blob storage 'nlp-models' created")
        return True

    print("❌ Failed to create Blob storage")
    return False

def provision_kv_cache() -> bool:
    """Create Vercel KV cache if not exists."""
    print("\n📦 Provisioning Vercel KV cache...")

    # Check if already exists
    result = run_vercel_command("kv ls")
    if result and result.get("success"):
        print("♻️  KV cache already exists")
        return True

    # Create new KV store
    result = run_vercel_command("kv create nlp-cache --yes")
    if result:
        print("✅ KV cache 'nlp-cache' created")
        return True

    print("❌ Failed to create KV cache")
    return False

def setup_postgres_extensions() -> bool:
    """Enable required Postgres extensions."""
    print("\n📦 Setting up Postgres extensions...")

    from sqlalchemy import create_engine, text

    db_url = os.getenv("VERCEL_POSTGRES_URL") or os.getenv("POSTGRES_URL")
    if not db_url:
        print("⚠️  No Postgres URL found, skipping extension setup")
        return True  # Not a failure, just skipped

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Enable pgvector
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✅ pgvector extension enabled")

            # Verify
            result = conn.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ))
            row = result.fetchone()
            if row:
                print(f"✅ pgvector version: {row[0]}")
                return True
    except Exception as e:
        print(f"⚠️  Postgres setup warning: {e}")
        # Not critical, may run later
        return True

    return False

def pull_environment_variables() -> bool:
    """Pull all environment variables from Vercel."""
    print("\n📦 Pulling environment variables...")

    result = run_vercel_command("env pull --yes")
    if result:
        print("✅ Environment variables synchronized")
        return True

    print("⚠️  Failed to pull environment variables")
    return False

def verify_setup() -> Dict[str, bool]:
    """Verify all services are accessible."""
    print("\n🔍 Verifying setup...")

    status = {
        "postgres": False,
        "blob": False,
        "kv": False,
        "pgvector": False
    }

    # Check Postgres
    db_url = os.getenv("VERCEL_POSTGRES_URL") or os.getenv("POSTGRES_URL")
    if db_url:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                status["postgres"] = True
                print("✅ Postgres: Connected")

                # Check pgvector
                result = conn.execute(text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                ))
                if result.fetchone():
                    status["pgvector"] = True
                    print("✅ pgvector: Enabled")
        except Exception as e:
            print(f"❌ Postgres: {e}")
    else:
        print("⚠️  Postgres: No connection URL")

    # Check Blob
    blob_token = os.getenv("BLOB_READ_WRITE_TOKEN")
    if blob_token:
        status["blob"] = True
        print("✅ Blob: Token found")
    else:
        print("⚠️  Blob: No token found")

    # Check KV
    kv_url = os.getenv("KV_REST_API_URL")
    kv_token = os.getenv("KV_REST_API_TOKEN")
    if kv_url and kv_token:
        status["kv"] = True
        print("✅ KV: Credentials found")
    else:
        print("⚠️  KV: No credentials found")

    return status

def main():
    print("=" * 60)
    print("🚀 Vercel Automated Setup")
    print("=" * 60)

    # Check environment
    is_vercel_build = os.getenv("VERCEL") == "1"
    if is_vercel_build:
        print("📍 Running in Vercel build environment")
    else:
        print("📍 Running in local environment")

    # Check CLI
    if not check_vercel_cli():
        if is_vercel_build:
            # CLI should be pre-installed in Vercel
            print("⚠️  Continuing without CLI (may be pre-configured)")
        else:
            print("❌ Please install Vercel CLI: npm install -g vercel")
            sys.exit(1)

    # Provision services (skip in Vercel build, use pre-existing)
    if not is_vercel_build:
        provision_blob_storage()
        provision_kv_cache()
        pull_environment_variables()
    else:
        print("\n♻️  Using pre-configured Vercel services")

    # Always try to setup Postgres extensions
    setup_postgres_extensions()

    # Verify
    status = verify_setup()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)

    for service, healthy in status.items():
        icon = "✅" if healthy else "⚠️"
        print(f"{icon} {service.upper()}: {'Ready' if healthy else 'Needs configuration'}")

    all_healthy = all(status.values())

    if all_healthy:
        print("\n✅ All services ready!")
        print("\nNext steps:")
        print("  1. Run health checks: python scripts/run_health_checks.py")
        print("  2. Warm cache: python scripts/deploy_warm_cache.py")
    else:
        print("\n⚠️  Some services need configuration")
        print("\nRun manually:")
        if not status["blob"]:
            print("  vercel blob create nlp-models")
        if not status["kv"]:
            print("  vercel kv create nlp-cache")
        print("  vercel env pull")

    print("=" * 60)

    # Exit code: 0 even if not all healthy (allow deployment to continue)
    # Health checks will run separately
    return 0

if __name__ == "__main__":
    sys.exit(main())
