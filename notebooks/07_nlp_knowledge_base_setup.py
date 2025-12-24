"""
Setup Medical Billing NLP Knowledge Base - Deepnote Compatible

This script downloads medical billing codes and builds the vector embedding knowledge base.
Designed to run in Deepnote with Vercel Postgres (Neon).

Steps:
1. Verify Vercel Postgres connection
2. Download ICD-10-CM codes from CMS
3. Download HCPCS Level II codes from CMS
4. Build embeddings using BioClinical ModernBERT
5. Populate pgvector database
6. Test semantic search

Usage:
    In Deepnote, simply run this notebook after setting environment variables.

Environment Variables Required (set in Deepnote Project Settings):
    - VERCEL_POSTGRES_URL: PostgreSQL connection string
    - HF_TOKEN (optional): Hugging Face token for model downloads
"""

import sys
import os

# Add project root to path
if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')
if os.path.exists('/work') and '/work' not in sys.path:
    sys.path.insert(0, '/work')

import pandas as pd
from src.data.download_codes import CodeDownloader
from src.data.build_embeddings import build_knowledge_base

print("=" * 70)
print(" " * 15 + "MEDICAL BILLING NLP KNOWLEDGE BASE SETUP")
print(" " * 20 + "Deepnote + Vercel Postgres Edition")
print("=" * 70)

# ============================================================================
# 1. ENVIRONMENT CHECK
# ============================================================================

print("\n📋 Step 1: Environment Check")
print("=" * 70)

import psycopg2
from config.settings import DATABASE_URL

if not DATABASE_URL or 'localhost' in DATABASE_URL:
    print("⚠️  WARNING: Using local database URL")
    print("   For Deepnote, set VERCEL_POSTGRES_URL in Project Settings")
    print("   Vercel Dashboard → Storage → Your Database → Settings")
else:
    print("✓ Vercel Postgres URL detected")

try:
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print("✓ Database connection successful")
        print(f"  PostgreSQL version: {version[:50]}...")

        # Check for pgvector extension
        cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
        has_pgvector = cur.fetchone()[0] > 0
        if has_pgvector:
            print("✓ pgvector extension is installed")
        else:
            print("⚠️  pgvector extension not found")
            print("   Installing pgvector...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            print("✓ pgvector extension installed")
    conn.close()
except Exception as e:
    print("✗ Database connection failed")
    print(f"  Error: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check VERCEL_POSTGRES_URL is set in Deepnote Project Settings")
    print("   2. Verify database is running in Vercel Dashboard")
    print("   3. Ensure you're using POSTGRES_URL_NON_POOLING (not pooled connection)")
    sys.exit(1)

# ============================================================================
# 2. DOWNLOAD ICD-10-CM CODES
# ============================================================================

print("\n📥 Step 2: Download ICD-10-CM Codes")
print("=" * 70)

downloader = CodeDownloader()

print("Downloading ICD-10-CM codes from CMS...")
icd10_df = downloader.download_icd10_codes()

print(f"\n✓ Downloaded {len(icd10_df)} ICD-10 codes")
print(f"  Billable codes: {icd10_df['is_billable'].sum()}")

print("\nSample ICD-10 codes:")
print(icd10_df.head(10).to_string(index=False))

# ============================================================================
# 3. DOWNLOAD HCPCS LEVEL II CODES
# ============================================================================

print("\n\n📥 Step 3: Download HCPCS Level II Codes")
print("=" * 70)

print("Downloading HCPCS codes from CMS...")
hcpcs_df = downloader.download_hcpcs_codes()

print(f"\n✓ Downloaded {len(hcpcs_df)} HCPCS codes")
print(f"  Categories: {hcpcs_df['category'].nunique()}")

print("\nCategory distribution:")
print(hcpcs_df['category'].value_counts().head(10).to_string())

print("\nSample HCPCS codes:")
print(hcpcs_df.head(10).to_string(index=False))

# ============================================================================
# 4. BUILD EMBEDDINGS AND POPULATE DATABASE
# ============================================================================

print("\n\n🤖 Step 4: Build Embeddings and Populate Database")
print("=" * 70)
print("This step:")
print("  1. Loads BioClinical ModernBERT model (768-dim, 8K context)")
print("  2. Generates embeddings for all code descriptions")
print("  3. Creates pgvector schema with HNSW index")
print("  4. Inserts embeddings into database")
print("\n⏱️  Estimated time: 10-20 minutes (first run)")
print("=" * 70)

# Build knowledge base
build_knowledge_base(rebuild=True)

# ============================================================================
# 5. VERIFY DATABASE
# ============================================================================

print("\n\n✅ Step 5: Verify Database")
print("=" * 70)

conn = psycopg2.connect(DATABASE_URL)

with conn.cursor() as cur:
    # Count total records
    cur.execute("SELECT COUNT(*) FROM code_embeddings;")
    total = cur.fetchone()[0]
    print(f"✓ Total embeddings: {total:,}")

    # Count by type
    cur.execute("""
        SELECT code_type, COUNT(*) as count
        FROM code_embeddings
        GROUP BY code_type;
    """)
    print("\nBy code type:")
    for code_type, count in cur.fetchall():
        print(f"  {code_type}: {count:,}")

    # Check index
    cur.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = 'code_embeddings';
    """)
    print("\nIndexes:")
    for idx in cur.fetchall():
        print(f"  ✓ {idx[0]}")

conn.close()

# ============================================================================
# 6. TEST SEMANTIC SEARCH
# ============================================================================

print("\n\n🔍 Step 6: Test Semantic Search")
print("=" * 70)

from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL

# Load model
print(f"Loading model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

def search_codes(query, code_type='ICD-10', top_k=5):
    """Search for medical codes using semantic similarity."""

    # Embed query
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    conn = psycopg2.connect(DATABASE_URL)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                code_id,
                long_description,
                1 - (embedding <=> %s::vector) as similarity
            FROM code_embeddings
            WHERE code_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, code_type, query_embedding, top_k))

        results = []
        for code_id, desc, similarity in cur.fetchall():
            results.append({
                'code': code_id,
                'description': desc,
                'similarity': similarity
            })

    conn.close()
    return pd.DataFrame(results)

# Test queries
test_queries = [
    "diabetes type 2",
    "high blood pressure",
    "chest pain",
    "heart failure",
    "pneumonia"
]

for query in test_queries:
    print(f"\n{'=' * 70}")
    print(f"Query: '{query}'")
    print(f"{'=' * 70}")
    results = search_codes(query, top_k=3)
    for _, row in results.iterrows():
        print(f"  {row['similarity']:.3f} | {row['code']}: {row['description'][:60]}...")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 70)
print(" " * 25 + "SETUP COMPLETE!")
print("=" * 70)

print("\n✅ Knowledge base successfully built!")
print("\nYou can now:")
print("  - Run NLP extraction pipeline")
print("  - Perform gap analysis")
print("  - Use semantic code search")

print("\n📝 Next steps:")
print("  1. Review search results above")
print("  2. Run notebooks/07_nlp_code_extraction.py for entity extraction")
print("  3. Run notebooks/08_gap_analysis.py for validation")

print("\n💡 Configuration:")
print(f"  Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'}")
print(f"  Model: {EMBEDDING_MODEL}")
print(f"  Total codes: {total:,}")

print("\n" + "=" * 70)
