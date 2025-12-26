"""
Notebook 07: NLP Knowledge Base Setup
======================================
One-time setup that downloads ICD-10/HCPCS codes and builds embeddings.
Stores everything in Vercel Postgres with pgvector.

Prerequisites:
- Run 00_verify_nlp_dependencies.py first
- VERCEL_POSTGRES_URL set in Deepnote Environment Variables

Time: 10-20 minutes (depends on code download speed)
"""

print("🔧 Medical Billing NLP - Knowledge Base Setup")
print("=" * 60)

# %% [markdown]
# ## Step 1: Verify Environment

# %%
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from config.settings import validate_environment, db_config, embedding_config

if not validate_environment():
    raise EnvironmentError("Please configure VERCEL_POSTGRES_URL in Deepnote Project Settings")

print(f"✅ Database URL: {db_config.url[:50]}...")
print(f"✅ Embedding Model: {embedding_config.model_name}")

# %% [markdown]
# ## Step 2: Test Database Connection

# %%
from sqlalchemy import create_engine, text

engine = create_engine(db_config.url)

with engine.connect() as conn:
    # Test connection
    result = conn.execute(text("SELECT version()"))
    version = result.fetchone()[0]
    print(f"✅ Connected to PostgreSQL: {version[:50]}...")

    # Check pgvector
    result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
    row = result.fetchone()
    if row:
        print(f"✅ pgvector extension: v{row[0]}")
    else:
        print("📦 Installing pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        print("✅ pgvector installed")

# %% [markdown]
# ## Step 3: Build Knowledge Base

# %%
from src.data.build_embeddings import build_knowledge_base

# This will:
# 1. Download ICD-10-CM codes from CMS
# 2. Download HCPCS Level II codes from CMS
# 3. Generate embeddings using BioClinical ModernBERT
# 4. Store in code_embeddings table with HNSW index

build_knowledge_base(database_url=db_config.url)

# %% [markdown]
# ## Step 4: Verify Knowledge Base

# %%
with engine.connect() as conn:
    # Count codes
    result = conn.execute(text("""
        SELECT code_type, COUNT(*) as count
        FROM code_embeddings
        GROUP BY code_type
    """))

    print("\n📊 Knowledge Base Statistics:")
    print("-" * 40)
    for row in result:
        print(f"  {row.code_type}: {row.count:,} codes")

    # Test similarity search
    print("\n🔍 Testing Similarity Search...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_config.model_name)

    test_query = "patient with diabetes and high blood sugar"
    query_embedding = model.encode(test_query)

    result = conn.execute(text("""
        SELECT code_id, short_description,
               1 - (embedding <=> CAST(:emb AS vector)) as similarity
        FROM code_embeddings
        WHERE code_type = 'ICD-10'
        ORDER BY embedding <=> CAST(:emb AS vector)
        LIMIT 5
    """), {'emb': query_embedding.tolist()})

    print(f"\n  Query: '{test_query}'")
    print("  Top matches:")
    for row in result:
        print(f"    {row.code_id}: {row.short_description} (similarity: {row.similarity:.3f})")

# %% [markdown]
# ## Step 5: Summary

# %%
print("\n" + "=" * 60)
print("✅ KNOWLEDGE BASE SETUP COMPLETE")
print("=" * 60)
print("""
Next steps:
1. Run notebook 08_nlp_code_extraction.py to extract codes from claims
2. Run notebook 09_gap_analysis_reporting.py for validation reports

The system is now ready to:
- Extract ICD-10 diagnosis codes from clinical notes
- Extract HCPCS procedure codes from clinical notes
- Perform semantic similarity search
- Generate code suggestions based on cluster consensus
- Flag missing or unsupported codes
""")
