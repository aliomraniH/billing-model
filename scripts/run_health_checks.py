#!/usr/bin/env python3
"""
Comprehensive Health Checks for All Storage Systems
====================================================
Verifies connectivity and functionality of:
- Vercel Postgres + pgvector
- Vercel Blob Storage
- Vercel KV Cache
- HuggingFace Hub (optional)

Usage:
    python scripts/run_health_checks.py
    python scripts/run_health_checks.py --verbose
    python scripts/run_health_checks.py --json  # CI/CD output
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class HealthCheckResult:
    """Container for health check results."""
    def __init__(self, name: str):
        self.name = name
        self.checks: List[Tuple[str, bool, str]] = []
        self.start_time = time.time()
        self.end_time = None

    def add_check(self, description: str, passed: bool, details: str = ""):
        """Add a check result."""
        self.checks.append((description, passed, details))

    def finish(self):
        """Mark health check as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get check duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def passed(self) -> bool:
        """Check if all tests passed."""
        return all(passed for _, passed, _ in self.checks)

    @property
    def pass_count(self) -> int:
        """Count of passed checks."""
        return sum(1 for _, passed, _ in self.checks if passed)

    @property
    def total_count(self) -> int:
        """Total number of checks."""
        return len(self.checks)

def check_postgres_health() -> HealthCheckResult:
    """Test Vercel Postgres connectivity and basic operations."""
    result = HealthCheckResult("Vercel Postgres")

    db_url = os.getenv("VERCEL_POSTGRES_URL") or os.getenv("POSTGRES_URL")

    if not db_url:
        result.add_check("Environment variable", False, "VERCEL_POSTGRES_URL not set")
        result.finish()
        return result

    result.add_check("Environment variable", True, "VERCEL_POSTGRES_URL found")

    try:
        from sqlalchemy import create_engine, text

        # Test connection
        engine = create_engine(db_url, connect_args={'connect_timeout': 10})

        with engine.connect() as conn:
            # Basic connectivity
            conn.execute(text("SELECT 1"))
            result.add_check("Database connection", True, "Connected successfully")

            # Check version
            version_result = conn.execute(text("SELECT version()"))
            version = version_result.fetchone()[0]
            postgres_version = version.split()[1] if version else "unknown"
            result.add_check("PostgreSQL version", True, postgres_version)

            # Check if tables exist
            tables_result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            table_count = tables_result.fetchone()[0]
            result.add_check("Tables exist", table_count > 0, f"{table_count} tables found")

    except Exception as e:
        result.add_check("Database connection", False, str(e))

    result.finish()
    return result

def check_pgvector_health() -> HealthCheckResult:
    """Test pgvector extension and functionality."""
    result = HealthCheckResult("pgvector Extension")

    db_url = os.getenv("VERCEL_POSTGRES_URL") or os.getenv("POSTGRES_URL")

    if not db_url:
        result.add_check("Database URL", False, "No Postgres URL")
        result.finish()
        return result

    try:
        from sqlalchemy import create_engine, text
        import numpy as np

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # Check extension exists
            ext_result = conn.execute(text("""
                SELECT extversion FROM pg_extension WHERE extname = 'vector'
            """))
            row = ext_result.fetchone()

            if row:
                result.add_check("Extension installed", True, f"version {row[0]}")
            else:
                result.add_check("Extension installed", False, "pgvector not found")
                result.finish()
                return result

            # Check if code_embeddings table exists
            table_result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'code_embeddings'
                )
            """))
            table_exists = table_result.fetchone()[0]
            result.add_check("code_embeddings table", table_exists,
                           "Table exists" if table_exists else "Table not found")

            if table_exists:
                # Count embeddings
                count_result = conn.execute(text("SELECT COUNT(*) FROM code_embeddings"))
                count = count_result.fetchone()[0]
                result.add_check("Embeddings loaded", count > 0, f"{count:,} codes")

                # Test vector search (if embeddings exist)
                if count > 0:
                    test_vector = np.random.rand(768).tolist()
                    search_result = conn.execute(text("""
                        SELECT code_id
                        FROM code_embeddings
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT 1
                    """), {"vec": test_vector})

                    if search_result.fetchone():
                        result.add_check("Vector search", True, "Similarity search working")
                    else:
                        result.add_check("Vector search", False, "Search returned no results")

    except Exception as e:
        result.add_check("pgvector test", False, str(e))

    result.finish()
    return result

def check_blob_health() -> HealthCheckResult:
    """Test Vercel Blob storage connectivity."""
    result = HealthCheckResult("Vercel Blob Storage")

    blob_token = os.getenv("BLOB_READ_WRITE_TOKEN")

    if not blob_token:
        result.add_check("Environment variable", False, "BLOB_READ_WRITE_TOKEN not set")
        result.finish()
        return result

    result.add_check("Environment variable", True, "Token found")

    try:
        import requests

        # Test write
        test_key = f"healthcheck_{int(time.time())}.txt"
        test_data = b"Health check test data"

        url = f"https://blob.vercel-storage.com/{test_key}"
        headers = {"Authorization": f"Bearer {blob_token}"}

        # Upload test
        upload_response = requests.put(url, data=test_data, headers=headers, timeout=10)
        if upload_response.status_code == 200:
            result.add_check("Write operation", True, "Test upload successful")

            # Download test
            download_response = requests.get(url, headers=headers, timeout=10)
            if download_response.status_code == 200 and download_response.content == test_data:
                result.add_check("Read operation", True, "Test download successful")
            else:
                result.add_check("Read operation", False, f"Status: {download_response.status_code}")

            # Cleanup
            requests.delete(url, headers=headers, timeout=10)
        else:
            result.add_check("Write operation", False, f"Status: {upload_response.status_code}")

    except Exception as e:
        result.add_check("Blob operations", False, str(e))

    result.finish()
    return result

def check_kv_health() -> HealthCheckResult:
    """Test Vercel KV cache connectivity."""
    result = HealthCheckResult("Vercel KV Cache")

    kv_url = os.getenv("KV_REST_API_URL")
    kv_token = os.getenv("KV_REST_API_TOKEN")

    if not kv_url or not kv_token:
        result.add_check("Environment variables", False,
                        "KV_REST_API_URL or KV_REST_API_TOKEN not set")
        result.finish()
        return result

    result.add_check("Environment variables", True, "Credentials found")

    try:
        import requests

        headers = {"Authorization": f"Bearer {kv_token}"}
        test_key = f"healthcheck_{int(time.time())}"
        test_value = {"status": "healthy", "timestamp": time.time()}

        # Set test
        set_url = f"{kv_url}/set/{test_key}"
        set_response = requests.post(
            set_url,
            json={"value": json.dumps(test_value), "ex": 60},  # 60 second TTL
            headers=headers,
            timeout=10
        )

        if set_response.status_code == 200:
            result.add_check("Write operation", True, "Test set successful")

            # Get test
            get_url = f"{kv_url}/get/{test_key}"
            get_response = requests.get(get_url, headers=headers, timeout=10)

            if get_response.status_code == 200:
                retrieved = get_response.json().get("result")
                if retrieved:
                    result.add_check("Read operation", True, "Test get successful")
                else:
                    result.add_check("Read operation", False, "No data returned")
            else:
                result.add_check("Read operation", False, f"Status: {get_response.status_code}")

            # Delete test
            del_url = f"{kv_url}/del/{test_key}"
            requests.post(del_url, headers=headers, timeout=10)
        else:
            result.add_check("Write operation", False, f"Status: {set_response.status_code}")

    except Exception as e:
        result.add_check("KV operations", False, str(e))

    result.finish()
    return result

def check_huggingface_health() -> HealthCheckResult:
    """Test HuggingFace Hub connectivity (optional)."""
    result = HealthCheckResult("HuggingFace Hub")

    hf_token = os.getenv("HF_TOKEN")
    result.add_check("Environment variable", bool(hf_token),
                    "Token found" if hf_token else "Token not set (optional)")

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)

        # Test model info retrieval (lightweight operation)
        from config.settings import embedding_config
        model_info = api.model_info(embedding_config.model_name)

        if model_info:
            result.add_check("Model access", True, f"{embedding_config.model_name}")
            result.add_check("Model downloads", True, f"{model_info.downloads:,} downloads")

    except Exception as e:
        result.add_check("Hub connectivity", False, str(e))

    result.finish()
    return result

def print_results(results: List[HealthCheckResult], verbose: bool = False):
    """Print health check results in human-readable format."""
    print("\n" + "=" * 60)
    print("🏥 HEALTH CHECK RESULTS")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    overall_passed = True

    for result in results:
        icon = "✅" if result.passed else "❌"
        status = "PASS" if result.passed else "FAIL"
        print(f"\n{icon} {result.name}: {status} ({result.pass_count}/{result.total_count}) [{result.duration:.2f}s]")

        if verbose or not result.passed:
            for description, passed, details in result.checks:
                check_icon = "  ✓" if passed else "  ✗"
                detail_str = f" - {details}" if details else ""
                print(f"{check_icon} {description}{detail_str}")

        if not result.passed:
            overall_passed = False

    # Summary
    print("\n" + "=" * 60)
    total_checks = sum(r.total_count for r in results)
    passed_checks = sum(r.pass_count for r in results)

    if overall_passed:
        print(f"✅ ALL SYSTEMS HEALTHY ({passed_checks}/{total_checks} checks passed)")
    else:
        print(f"❌ SOME SYSTEMS UNHEALTHY ({passed_checks}/{total_checks} checks passed)")

    print("=" * 60)

    return overall_passed

def export_json(results: List[HealthCheckResult]) -> str:
    """Export results as JSON for CI/CD integration."""
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall_status": "healthy" if all(r.passed for r in results) else "unhealthy",
        "services": []
    }

    for result in results:
        service = {
            "name": result.name,
            "status": "healthy" if result.passed else "unhealthy",
            "duration_seconds": round(result.duration, 2),
            "checks_passed": result.pass_count,
            "checks_total": result.total_count,
            "checks": [
                {
                    "description": desc,
                    "passed": passed,
                    "details": details
                }
                for desc, passed, details in result.checks
            ]
        }
        output["services"].append(service)

    return json.dumps(output, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run health checks for all storage systems")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON for CI/CD")
    parser.add_argument("--fail-fast", action="store_true", help="Exit on first failure")
    args = parser.parse_args()

    if not args.json:
        print("🚀 Starting comprehensive health checks...")

    # Run all health checks
    results = []

    checks = [
        ("Postgres", check_postgres_health),
        ("pgvector", check_pgvector_health),
        ("Blob", check_blob_health),
        ("KV", check_kv_health),
        ("HuggingFace", check_huggingface_health),
    ]

    for name, check_func in checks:
        if not args.json:
            print(f"\n🔍 Checking {name}...", end="", flush=True)

        result = check_func()
        results.append(result)

        if not args.json:
            status = "✅" if result.passed else "❌"
            print(f" {status}")

        if args.fail_fast and not result.passed:
            if not args.json:
                print(f"\n❌ Health check failed for {name}, exiting (fail-fast mode)")
            sys.exit(1)

    # Output results
    if args.json:
        print(export_json(results))
    else:
        overall_passed = print_results(results, verbose=args.verbose)

    # Exit code
    sys.exit(0 if all(r.passed for r in results) else 1)

if __name__ == "__main__":
    main()
