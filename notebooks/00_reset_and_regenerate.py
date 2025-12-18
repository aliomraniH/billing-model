"""
Reset Pinecone Index and Database for Fresh Start with New Templates

This script:
1. Deletes all vectors from Pinecone
2. Clears category tables in Postgres
3. Prepares for fresh embedding generation with new templates/fillers

Run this BEFORE running Notebook 5 to regenerate embeddings with:
- 10 templates per category (instead of 3)
- 12+ filler values each (instead of 5)
- Unique identifiers (dates + providers)

Expected improvement: 64.4% duplicates → <10% duplicates
"""

print("=" * 70)
print("🔄 RESET PINECONE & DATABASE FOR FRESH EMBEDDING GENERATION")
print("=" * 70)
print("⚠️  WARNING: This will delete ALL vectors and categories!")
print("=" * 70 + "\n")

import os
import time
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURATION
# ============================================================
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = "medical-billing-notes"

# Validate
if not DATABASE_URL:
    raise ValueError("Missing VERCEL_POSTGRES_URL")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

# ============================================================
# STEP 1: CLEAR PINECONE VECTORS
# ============================================================
print("🗑️  STEP 1: Clearing Pinecone vectors...")

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Get current stats
stats_before = index.describe_index_stats()
vectors_before = stats_before.total_vector_count

print(f"   Current vectors: {vectors_before:,}")

if vectors_before > 0:
    print(f"   Deleting all vectors from namespace ''...")

    # Delete all vectors in default namespace
    index.delete(delete_all=True, namespace='')

    # Wait for deletion to complete
    print(f"   Waiting for deletion to complete...", end='')
    for i in range(10):
        time.sleep(2)
        stats_now = index.describe_index_stats()
        if stats_now.total_vector_count == 0:
            print(" ✅")
            break
        print(".", end='', flush=True)
    else:
        print(" (may take longer)")

    stats_after = index.describe_index_stats()
    print(f"   ✅ Deleted {vectors_before:,} vectors")
    print(f"   Remaining: {stats_after.total_vector_count:,}")
else:
    print(f"   ✅ No vectors to delete")

# ============================================================
# STEP 2: CLEAR DATABASE TABLES
# ============================================================
print("\n🗑️  STEP 2: Clearing database tables...")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Count before
    result = conn.execute(text("SELECT COUNT(*) FROM claim_category_membership"))
    memberships_before = result.fetchone()[0]

    result = conn.execute(text("SELECT COUNT(*) FROM claim_categories"))
    categories_before = result.fetchone()[0]

    print(f"   Current memberships: {memberships_before:,}")
    print(f"   Current categories: {categories_before:,}")

    # Clear membership table first (foreign key constraint)
    conn.execute(text("DELETE FROM claim_category_membership"))
    conn.commit()
    print(f"   ✅ Deleted {memberships_before:,} claim-category memberships")

    # Clear categories table
    conn.execute(text("DELETE FROM claim_categories"))
    conn.commit()
    print(f"   ✅ Deleted {categories_before:,} categories")

    # Clear clinical_notes.embedding column to force regeneration
    conn.execute(text("UPDATE clinical_notes SET embedding = NULL"))
    conn.commit()

    result = conn.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE embedding IS NULL"))
    notes_cleared = result.fetchone()[0]
    print(f"   ✅ Cleared {notes_cleared:,} note embeddings (will regenerate)")

# ============================================================
# VERIFICATION
# ============================================================
print("\n✅ VERIFICATION")
print("=" * 70)

# Pinecone
stats_final = index.describe_index_stats()
print(f"Pinecone vectors: {stats_final.total_vector_count:,}")
assert stats_final.total_vector_count == 0, "Pinecone not empty!"

# Database
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM claim_category_membership"))
    assert result.fetchone()[0] == 0, "Memberships not empty!"

    result = conn.execute(text("SELECT COUNT(*) FROM claim_categories"))
    assert result.fetchone()[0] == 0, "Categories not empty!"

    result = conn.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE embedding IS NOT NULL"))
    embedded_notes = result.fetchone()[0]
    print(f"Database memberships: 0")
    print(f"Database categories: 0")
    print(f"Notes with embeddings: {embedded_notes:,}")

print("\n" + "=" * 70)
print("✅ RESET COMPLETE - Ready for fresh embedding generation!")
print("=" * 70)

print("\n📋 NEXT STEPS:")
print("=" * 70)
print()
print("1. Generate new embeddings with improved templates:")
print("   export MAX_CLAIMS_TO_PROCESS=1000  # Or -1 for all 15,000")
print("   python notebooks/05_embeddings_similarity_search.py")
print()
print("2. Run clustering with LLM labeling:")
print("   python notebooks/06_llm_clustering_cache.py")
print()
print("3. Expected improvements:")
print("   • Duplicate vectors: 64.4% → <10%")
print("   • Silhouette score: 0.249 → 0.35-0.45")
print("   • Unique combinations: 3,125 → 500,000+")
print()
print("4. New features in the updated notebooks:")
print("   • 10 templates per category (vs 3)")
print("   • 12+ filler values each (vs 5)")
print("   • Unique visit dates and providers")
print("   • Dynamic model selection (CLAUDE_MODEL, HF_EMBEDDING_MODEL)")
print()
print("=" * 70)
