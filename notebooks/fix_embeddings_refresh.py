"""
Fix Script: Reset Embedding Timestamps to Force Re-processing

PROBLEM:
The auto-refresh system is skipping most claims because their embeddings are "fresh"
(less than 12 hours old). This results in low processing rates like 8.4%.

SOLUTION:
This script provides several options to fix the issue:
1. Clear ALL timestamps to force full re-processing
2. Clear timestamps for claims that need embeddings
3. Temporarily disable auto-refresh (modify config)

Choose the option that best fits your needs.

Usage:
    python notebooks/fix_embeddings_refresh.py --clear-all
    python notebooks/fix_embeddings_refresh.py --clear-missing
    python notebooks/fix_embeddings_refresh.py --status
"""

import os
import sys
import argparse
from sqlalchemy import create_engine, text
from datetime import datetime

# Check environment
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not DATABASE_URL:
    print("❌ Error: VERCEL_POSTGRES_URL not set")
    sys.exit(1)

print("=" * 70)
print("🔧 EMBEDDING REFRESH FIX SCRIPT")
print("=" * 70)


def show_status(engine):
    """Show current embedding status"""
    print("\n📊 Current Status:")

    with engine.connect() as conn:
        # Check if columns exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'clinical_notes'
            AND column_name IN ('last_embedded_at', 'embedding_model')
        """))
        columns = [row[0] for row in result]

        if not columns:
            print("   ⚠️  Refresh columns don't exist yet!")
            print("   ➡️  Run migration first: python notebooks/migrations/01_add_refresh_timestamps.py")
            return False

        # Get statistics
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_notes,
                COUNT(last_embedded_at) as with_timestamp,
                COUNT(CASE WHEN last_embedded_at > NOW() - INTERVAL '12 hours' THEN 1 END) as fresh,
                COUNT(CASE WHEN last_embedded_at <= NOW() - INTERVAL '12 hours' OR last_embedded_at IS NULL THEN 1 END) as stale_or_null
            FROM clinical_notes
        """))

        row = result.fetchone()
        print(f"   Total notes in database: {row[0]:,}")
        print(f"   Notes with timestamps: {row[1]:,}")
        print(f"   Fresh embeddings (<12h old): {row[2]:,} (will be SKIPPED)")
        print(f"   Stale/null embeddings: {row[3]:,} (will be PROCESSED)")

        # Check Pinecone if available
        if PINECONE_API_KEY:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index("medical-billing-notes")
                stats = index.describe_index_stats()
                print(f"\n   Vectors in Pinecone: {stats.total_vector_count:,}")
            except ImportError:
                print(f"\n   ⚠️  Pinecone module not installed (pip install pinecone)")
            except Exception as e:
                print(f"\n   ⚠️  Could not check Pinecone: {e}")

        return True


def clear_all_timestamps(engine, force=False):
    """Clear all embedding timestamps to force full re-processing"""
    print("\n⚠️  WARNING: This will clear ALL embedding timestamps!")
    print("   All claims will be re-processed on next run.")

    if not force:
        try:
            confirm = input("\n   Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("   Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            # Running in non-interactive environment (e.g., notebook)
            print("   ⚠️ Running in non-interactive mode. Use --yes flag to confirm.")
            return

    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE clinical_notes
            SET
                last_embedded_at = NULL,
                embedding_model = NULL,
                embedding_version = NULL
        """))

        print(f"\n   ✅ Cleared timestamps for {result.rowcount:,} notes")
        print(f"   Next run will process all claims fresh")


def clear_missing_embeddings(engine):
    """Clear timestamps only for claims without embeddings in Pinecone"""
    print("\n🔄 Checking Pinecone for missing embeddings...")

    if not PINECONE_API_KEY:
        print("   ❌ PINECONE_API_KEY not set, cannot check Pinecone")
        return

    try:
        from pinecone import Pinecone

        # Get all note IDs from database
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT claim_id, note_type
                FROM clinical_notes
                WHERE last_embedded_at IS NOT NULL
            """))
            db_notes = [(row[0], row[1]) for row in result]

        print(f"   Found {len(db_notes):,} notes with timestamps in database")

        # Check which ones are missing from Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("medical-billing-notes")

        missing = []
        print("   Checking Pinecone (this may take a moment)...")

        # Batch check
        batch_size = 100
        for i in range(0, len(db_notes), batch_size):
            batch = db_notes[i:i+batch_size]
            vector_ids = [f"claim_{cid}_{ntype}" for cid, ntype in batch]

            fetched = index.fetch(vector_ids)

            for (cid, ntype), vid in zip(batch, vector_ids):
                if vid not in fetched['vectors']:
                    missing.append((cid, ntype))

            if (i + batch_size) % 500 == 0:
                print(f"   Checked {i + batch_size:,}/{len(db_notes):,}...")

        print(f"\n   Found {len(missing):,} notes missing from Pinecone")

        if missing:
            # Clear timestamps for missing notes
            with engine.begin() as conn:
                for cid, ntype in missing:
                    conn.execute(text("""
                        UPDATE clinical_notes
                        SET
                            last_embedded_at = NULL,
                            embedding_model = NULL,
                            embedding_version = NULL
                        WHERE claim_id = :cid AND note_type = :ntype
                    """), {'cid': cid, 'ntype': ntype})

            print(f"   ✅ Cleared timestamps for {len(missing):,} notes")
            print(f"   These will be re-processed on next run")
        else:
            print(f"   ✅ All database notes exist in Pinecone")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def disable_autorefresh():
    """Show instructions for disabling auto-refresh"""
    print("\n📝 To disable auto-refresh temporarily:")
    print("\n   Option 1: Set environment variable")
    print("   export AUTO_REFRESH_ENABLED=false")
    print("\n   Option 2: Modify notebooks/config.py")
    print("   Change line 68 to: auto_refresh_enabled: bool = False")
    print("\n   Option 3: Create config.json in project root:")
    print('   {"refresh": {"auto_refresh_enabled": false}}')
    print("\n   Then re-run notebook 5 to process all claims")


def main():
    parser = argparse.ArgumentParser(description="Fix embedding refresh issues")
    parser.add_argument('--status', action='store_true', help="Show current status")
    parser.add_argument('--clear-all', action='store_true', help="Clear all timestamps")
    parser.add_argument('--clear-missing', action='store_true', help="Clear timestamps for missing embeddings")
    parser.add_argument('--disable-info', action='store_true', help="Show how to disable auto-refresh")
    parser.add_argument('--yes', '-y', action='store_true', help="Skip confirmation prompts (for automation)")

    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    if args.status or not any([args.clear_all, args.clear_missing, args.disable_info]):
        show_status(engine)

    if args.clear_all:
        clear_all_timestamps(engine, force=args.yes)
        show_status(engine)

    if args.clear_missing:
        clear_missing_embeddings(engine)
        show_status(engine)

    if args.disable_info:
        disable_autorefresh()

    if not any([args.status, args.clear_all, args.clear_missing, args.disable_info]):
        print("\n💡 Available actions:")
        print("   --status          Show current embedding status")
        print("   --clear-all       Clear all timestamps (forces full re-processing)")
        print("   --clear-missing   Clear timestamps only for notes missing from Pinecone")
        print("   --disable-info    Show how to disable auto-refresh")
        print("   --yes, -y         Skip confirmation prompts (use with --clear-all)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
