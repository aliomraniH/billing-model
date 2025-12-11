#!/usr/bin/env python3
"""
Database Connection Verification Script
Run this script to verify your Vercel Postgres connection is properly configured.

Usage in Google Colab:
    1. Add VERCEL_POSTGRES_URL to Colab Secrets (🔑 sidebar)
    2. Copy-paste this entire script into a Colab cell
    3. Run the cell

Usage locally:
    1. Copy .env.example to .env
    2. Fill in your VERCEL_POSTGRES_URL
    3. Run: python scripts/verify_connection.py
"""

import sys
import os

# Try to import required packages
try:
    from sqlalchemy import create_engine, text
    import pandas as pd
except ImportError:
    print("❌ Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sqlalchemy", "pandas", "psycopg2-binary"])
    from sqlalchemy import create_engine, text
    import pandas as pd

# Detect environment (Colab vs local)
try:
    from google.colab import userdata
    IN_COLAB = True
    print("🔍 Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("🔍 Running locally")


def get_database_url():
    """Get database URL from environment"""
    if IN_COLAB:
        try:
            url = userdata.get('VERCEL_POSTGRES_URL')
            if not url:
                print("❌ ERROR: VERCEL_POSTGRES_URL not found in Colab secrets!")
                print("\n📝 To fix:")
                print("1. Click 🔑 Secrets icon in left sidebar")
                print("2. Click '+ Add new secret'")
                print("3. Name: VERCEL_POSTGRES_URL")
                print("4. Value: postgresql://...")
                print("5. Toggle 'Notebook access' ON")
                sys.exit(1)
            return url
        except Exception as e:
            print(f"❌ ERROR reading Colab secrets: {e}")
            sys.exit(1)
    else:
        # Try environment variable first
        url = os.getenv('VERCEL_POSTGRES_URL')
        if url:
            return url

        # Try loading from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            url = os.getenv('VERCEL_POSTGRES_URL')
            if url:
                return url
        except ImportError:
            print("💡 TIP: Install python-dotenv to load from .env file")
            print("   pip install python-dotenv")

        print("❌ ERROR: VERCEL_POSTGRES_URL not found!")
        print("\n📝 To fix:")
        print("1. Copy .env.example to .env")
        print("2. Fill in your VERCEL_POSTGRES_URL")
        print("3. Or set environment variable: export VERCEL_POSTGRES_URL='postgresql://...'")
        sys.exit(1)


def verify_connection():
    """Verify database connection and configuration"""
    print("\n" + "="*70)
    print("🔍 VERCEL POSTGRES CONNECTION VERIFICATION")
    print("="*70)

    # Get database URL
    DATABASE_URL = get_database_url()

    # Check for common issues
    print("\n1️⃣ Checking connection string format...")

    if '.pooler.' in DATABASE_URL:
        print("   ⚠️  WARNING: Pooled connection detected!")
        print("   👉 You should use POSTGRES_URL_NON_POOLING (direct connection)")
        print("   👉 Pooled connections may not support pgvector properly")
    else:
        print("   ✅ Direct connection (good!)")

    if ':5432' not in DATABASE_URL and '@' in DATABASE_URL:
        print("   ⚠️  WARNING: Port not specified in connection string")
        print("   👉 Add :5432 after hostname")
    else:
        print("   ✅ Port specified")

    if 'sslmode' not in DATABASE_URL:
        print("   ⚠️  WARNING: SSL mode not specified")
        print("   👉 Add ?sslmode=require to connection string")
    else:
        print("   ✅ SSL mode configured")

    # Test connection
    print("\n2️⃣ Testing database connection...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"   ✅ Connected successfully!")
            print(f"   📊 Database: {version[:70]}...")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Verify connection string is correct (check for typos)")
        print("   2. Ensure database is 'Active' in Vercel dashboard")
        print("   3. Try regenerating credentials in Vercel")
        print("   4. Check firewall/network settings")
        sys.exit(1)

    # Check pgvector availability
    print("\n3️⃣ Checking pgvector extension...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name, installed_version, default_version
                FROM pg_available_extensions
                WHERE name = 'vector'
            """))
            row = result.fetchone()

            if row:
                if row[1]:  # installed_version
                    print(f"   ✅ pgvector {row[1]} is installed")
                else:
                    print(f"   ⚠️  pgvector available but not enabled")
                    print(f"   👉 Run: CREATE EXTENSION IF NOT EXISTS vector;")
            else:
                print("   ❌ pgvector not available")
                print("   👉 This database may be too old")
                print("   👉 Create a new Vercel Postgres database")
    except Exception as e:
        print(f"   ⚠️  Could not check pgvector: {e}")

    # Check existing tables
    print("\n4️⃣ Checking database schema...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]

            if tables:
                print(f"   ✅ Found {len(tables)} table(s):")
                for table in tables:
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = count_result.fetchone()[0]
                    print(f"      • {table}: {count:,} rows")
            else:
                print("   📋 No tables found (this is normal for new database)")
                print("   👉 Run notebooks/01_setup_database.py to create schema")
    except Exception as e:
        print(f"   ⚠️  Could not check tables: {e}")

    # Test write permission
    print("\n5️⃣ Testing write permissions...")
    try:
        with engine.connect() as conn:
            # Try to create a test table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _connection_test (
                    id SERIAL PRIMARY KEY,
                    test_timestamp TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("INSERT INTO _connection_test DEFAULT VALUES"))
            conn.execute(text("DROP TABLE _connection_test"))
            conn.commit()
            print("   ✅ Write permissions verified")
    except Exception as e:
        print(f"   ❌ Write test failed: {e}")
        print("   👉 Check database user permissions")

    # Summary
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("   1. Run notebooks/01_setup_database.py to create schema")
    print("   2. Run notebooks/02_load_synthea_data.py to load sample data")
    print("   3. Run notebooks/03_outlier_detection.py to train model")
    print("\n📚 Documentation: docs/VERCEL_SETUP.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    verify_connection()
