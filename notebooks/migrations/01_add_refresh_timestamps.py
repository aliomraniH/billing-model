"""
Database Migration: Add Timestamp Columns for Refresh System

This migration adds timestamp tracking to enable automatic refresh of
embeddings and categories based on configurable intervals.

Tables Modified:
- clinical_notes: Add last_embedded_at, embedding_version
- claim_categories: Add last_refreshed_at, refresh_interval_hours
- New table: embedding_refresh_config for flexible refresh policies

Run this migration ONCE before using the refresh system.

Usage:
    python notebooks/migrations/01_add_refresh_timestamps.py
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text

print("=" * 70)
print("🔄 DATABASE MIGRATION: Add Refresh Timestamps")
print("=" * 70 + "\n")

# ============================================================
# CONFIGURATION
# ============================================================
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')

if not DATABASE_URL:
    print("❌ Error: VERCEL_POSTGRES_URL environment variable not set")
    sys.exit(1)

print(f"✅ Database URL loaded")

# ============================================================
# MIGRATION STEPS
# ============================================================
engine = create_engine(DATABASE_URL)

print("\n📋 Migration Steps:")
print("   1. Add timestamp columns to clinical_notes")
print("   2. Add refresh configuration to claim_categories")
print("   3. Create embedding_refresh_config table")
print("   4. Create indexes for efficient refresh queries")
print("   5. Populate initial timestamp values")
print("\n" + "=" * 70 + "\n")

try:
    # Use engine.begin() for automatic transaction management (SQLAlchemy 2.0 compatible)
    with engine.begin() as conn:
        # ============================================================
        # STEP 1: Add columns to clinical_notes
        # ============================================================
        print("STEP 1: Updating clinical_notes table...")

        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'clinical_notes'
            AND column_name IN ('last_embedded_at', 'embedding_version', 'embedding_model')
        """))
        existing_columns = [row[0] for row in result.fetchall()]

        if 'last_embedded_at' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE clinical_notes
                ADD COLUMN last_embedded_at TIMESTAMP WITH TIME ZONE
            """))
            print("   ✅ Added last_embedded_at column")
        else:
            print("   ⚠️  Column last_embedded_at already exists, skipping")

        if 'embedding_version' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE clinical_notes
                ADD COLUMN embedding_version VARCHAR(100)
            """))
            print("   ✅ Added embedding_version column")
        else:
            print("   ⚠️  Column embedding_version already exists, skipping")

        if 'embedding_model' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE clinical_notes
                ADD COLUMN embedding_model VARCHAR(255) DEFAULT 'BAAI/bge-small-en-v1.5'
            """))
            print("   ✅ Added embedding_model column")
        else:
            print("   ⚠️  Column embedding_model already exists, skipping")

        # ============================================================
        # STEP 2: Add columns to claim_categories
        # ============================================================
        print("\nSTEP 2: Updating claim_categories table...")

        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'claim_categories'
            AND column_name IN ('last_refreshed_at', 'refresh_interval_hours', 'cluster_version')
        """))
        existing_columns = [row[0] for row in result.fetchall()]

        if 'last_refreshed_at' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE claim_categories
                ADD COLUMN last_refreshed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """))
            print("   ✅ Added last_refreshed_at column")
        else:
            print("   ⚠️  Column last_refreshed_at already exists, skipping")

        if 'refresh_interval_hours' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE claim_categories
                ADD COLUMN refresh_interval_hours INTEGER DEFAULT 24
            """))
            print("   ✅ Added refresh_interval_hours column")
        else:
            print("   ⚠️  Column refresh_interval_hours already exists, skipping")

        if 'cluster_version' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE claim_categories
                ADD COLUMN cluster_version VARCHAR(50)
            """))
            print("   ✅ Added cluster_version column")
        else:
            print("   ⚠️  Column cluster_version already exists, skipping")

        # ============================================================
        # STEP 3: Create embedding_refresh_config table
        # ============================================================
        print("\nSTEP 3: Creating embedding_refresh_config table...")

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS embedding_refresh_config (
                config_id SERIAL PRIMARY KEY,
                config_name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                refresh_interval_hours INTEGER NOT NULL DEFAULT 12,
                enabled BOOLEAN DEFAULT TRUE,
                last_run_at TIMESTAMP WITH TIME ZONE,
                next_run_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        print("   ✅ Created embedding_refresh_config table")

        # Insert default configurations
        conn.execute(text("""
            INSERT INTO embedding_refresh_config
            (config_name, description, refresh_interval_hours)
            VALUES
            ('default_embeddings', 'Default refresh interval for clinical note embeddings', 12),
            ('category_clusters', 'Refresh interval for clustering and categories', 24),
            ('critical_notes', 'Fast refresh for high-priority notes', 6),
            ('archive_notes', 'Slow refresh for archived notes', 168)
            ON CONFLICT (config_name) DO NOTHING
        """))
        print("   ✅ Inserted default refresh configurations")

        # ============================================================
        # STEP 4: Create indexes
        # ============================================================
        print("\nSTEP 4: Creating indexes for efficient refresh queries...")

        # Index for finding stale embeddings
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_clinical_notes_last_embedded
            ON clinical_notes(last_embedded_at)
            WHERE embedding IS NOT NULL
        """))
        print("   ✅ Created index on clinical_notes.last_embedded_at")

        # Index for finding stale categories
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_categories_last_refreshed
            ON claim_categories(last_refreshed_at)
        """))
        print("   ✅ Created index on claim_categories.last_refreshed_at")

        # Index for refresh config lookup
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_refresh_config_next_run
            ON embedding_refresh_config(next_run_at)
            WHERE enabled = TRUE
        """))
        print("   ✅ Created index on embedding_refresh_config.next_run_at")

        # ============================================================
        # STEP 5: Populate initial timestamps
        # ============================================================
        print("\nSTEP 5: Populating initial timestamp values...")

        # Set last_embedded_at for existing embeddings
        result = conn.execute(text("""
            UPDATE clinical_notes
            SET last_embedded_at = created_at
            WHERE embedding IS NOT NULL
            AND last_embedded_at IS NULL
        """))
        updated_notes = result.rowcount
        print(f"   ✅ Set last_embedded_at for {updated_notes:,} existing notes")

        # Set last_refreshed_at for existing categories
        result = conn.execute(text("""
            UPDATE claim_categories
            SET last_refreshed_at = created_at
            WHERE last_refreshed_at IS NULL
        """))
        updated_categories = result.rowcount
        print(f"   ✅ Set last_refreshed_at for {updated_categories:,} categories")

        # ============================================================
        # VERIFICATION
        # ============================================================
        print("\n" + "=" * 70)
        print("✅ MIGRATION VERIFICATION")
        print("=" * 70)

        # Check clinical_notes columns
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_notes,
                COUNT(embedding) as embedded_notes,
                COUNT(last_embedded_at) as timestamped_notes
            FROM clinical_notes
        """))
        row = result.fetchone()
        print(f"\nclinical_notes:")
        print(f"   Total notes: {row[0]:,}")
        print(f"   With embeddings: {row[1]:,}")
        print(f"   With timestamps: {row[2]:,}")

        # Check claim_categories columns
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_categories,
                COUNT(last_refreshed_at) as timestamped_categories,
                AVG(refresh_interval_hours) as avg_interval
            FROM claim_categories
        """))
        row = result.fetchone()
        if row[0] > 0:
            print(f"\nclaim_categories:")
            print(f"   Total categories: {row[0]:,}")
            print(f"   With timestamps: {row[1]:,}")
            print(f"   Avg refresh interval: {row[2]:.1f} hours")

        # Check refresh config
        result = conn.execute(text("""
            SELECT config_name, refresh_interval_hours, enabled
            FROM embedding_refresh_config
            ORDER BY config_name
        """))
        configs = result.fetchall()
        print(f"\nembedding_refresh_config:")
        for config in configs:
            status = "✅ enabled" if config[2] else "⏸️  disabled"
            print(f"   {config[0]}: {config[1]}h ({status})")

        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETE")
        print("=" * 70)

        print("\n📋 Next Steps:")
        print("   1. Run notebooks with auto-refresh enabled")
        print("   2. Configure custom refresh intervals per category if needed")
        print("   3. Use refresh_manager.py to manually trigger refreshes")
        print("\n" + "=" * 70)

except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
