"""
Simple script to clear embedding timestamps - Notebook friendly!

This is a simplified version of fix_embeddings_refresh.py designed to run
easily in Jupyter/Deepnote notebooks without command-line arguments.

Usage in notebook:
    %run clear_embeddings.py

Or from command line:
    python notebooks/clear_embeddings.py
"""

import os
from sqlalchemy import create_engine, text

print("=" * 70)
print("🔧 CLEAR EMBEDDING TIMESTAMPS")
print("=" * 70)

# Check environment
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')

if not DATABASE_URL:
    print("\n❌ Error: VERCEL_POSTGRES_URL environment variable not set")
    print("   Please set it in your notebook or environment")
    exit(1)

print("\n✅ Database connection configured")

# Create engine
engine = create_engine(DATABASE_URL)

# Show current status
print("\n📊 Current Status:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_notes,
            COUNT(last_embedded_at) as with_timestamp,
            COUNT(CASE WHEN last_embedded_at > NOW() - INTERVAL '12 hours' THEN 1 END) as fresh,
            COUNT(CASE WHEN last_embedded_at <= NOW() - INTERVAL '12 hours' OR last_embedded_at IS NULL THEN 1 END) as stale_or_null
        FROM clinical_notes
    """))

    row = result.fetchone()
    print(f"   Total notes: {row[0]:,}")
    print(f"   With timestamps: {row[1]:,}")
    print(f"   Fresh embeddings (<12h old): {row[2]:,} (currently skipped)")
    print(f"   Stale/null embeddings: {row[3]:,} (currently processed)")

# Clear timestamps
print("\n⚠️  Clearing ALL embedding timestamps...")
print("   This will force re-processing of all claims on next run.")

with engine.begin() as conn:
    result = conn.execute(text("""
        UPDATE clinical_notes
        SET
            last_embedded_at = NULL,
            embedding_model = NULL,
            embedding_version = NULL
    """))

    print(f"\n✅ Cleared timestamps for {result.rowcount:,} notes")

# Show new status
print("\n📊 New Status:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_notes,
            COUNT(last_embedded_at) as with_timestamp,
            COUNT(CASE WHEN last_embedded_at IS NULL THEN 1 END) as will_process
        FROM clinical_notes
    """))

    row = result.fetchone()
    print(f"   Total notes: {row[0]:,}")
    print(f"   With timestamps: {row[1]:,}")
    print(f"   Will be processed: {row[2]:,}")

print("\n" + "=" * 70)
print("✅ DONE! All notes will be re-processed on next notebook run")
print("=" * 70)
print("\n💡 Next steps:")
print("   1. Run notebook 5 to re-process all claims")
print("   2. Or set AUTO_REFRESH_ENABLED=false to disable refresh system")
print("=" * 70)
