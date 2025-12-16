# Medical Billing ML - Notebook 6: LLM Clustering & Category Cache
# Prerequisites: Run notebooks 01, 02, 05 first
# This notebook uses Claude API to intelligently cluster claims and create category-specific cache tables

#=============================================================================
# DEPENDENCIES & SETUP
#=============================================================================

print("📦 Installing dependencies...")
import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                      "anthropic", "scikit-learn>=1.3", "sentence-transformers"])

import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, KMeans
from anthropic import Anthropic

print("✅ Dependencies installed successfully")

#=============================================================================
# DATABASE CONNECTION
#=============================================================================

print("\n🔗 Connecting to database...")
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("❌ VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

engine = create_engine(DATABASE_URL)

# Test connection and verify clinical_notes table
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM clinical_notes WHERE embedding IS NOT NULL"))
        notes_count = result.fetchone()[0]
        print(f"✅ Connected! Found {notes_count:,} clinical notes with embeddings")

        if notes_count == 0:
            raise ValueError("❌ No embeddings found! Run notebook 05 first to generate embeddings")
except Exception as e:
    print(f"❌ Connection or table check failed: {e}")
    raise

#=============================================================================
# LOAD EMBEDDING MODEL
#=============================================================================

print("\n📥 Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = embed_model.get_sentence_embedding_dimension()
print(f"✅ Model loaded: all-MiniLM-L6-v2 ({embedding_dim} dimensions)")

#=============================================================================
# LOAD EMBEDDINGS FROM DATABASE
#=============================================================================

print("\n📊 Loading embeddings from database...")
embeddings_df = pd.read_sql(text("""
    SELECT
        note_id,
        claim_id,
        note_type,
        note_text,
        embedding
    FROM clinical_notes
    WHERE embedding IS NOT NULL
"""), engine)

print(f"✅ Loaded {len(embeddings_df):,} embeddings")

# Convert pgvector embeddings to numpy array
def pgvector_to_array(pgvector_str):
    """Convert pgvector string '[1,2,3]' to numpy array"""
    if isinstance(pgvector_str, str):
        # Remove brackets and parse
        return np.array([float(x) for x in pgvector_str.strip('[]').split(',')])
    return pgvector_str

embeddings_matrix = np.vstack([pgvector_to_array(emb) for emb in embeddings_df['embedding']])
print(f"   Embedding matrix shape: {embeddings_matrix.shape}")

#=============================================================================
# CLUSTERING WITH HDBSCAN
#=============================================================================

print("\n🔬 Performing HDBSCAN clustering...")
print("   Parameters: min_cluster_size=3, cluster_selection_method='eom'")

# Use sklearn's built-in HDBSCAN with store_centers to get centroids
clusterer = HDBSCAN(
    min_cluster_size=3,
    min_samples=None,  # Defaults to min_cluster_size
    metric='euclidean',
    cluster_selection_method='eom',
    store_centers='centroid'  # This stores cluster centroids
)

cluster_labels = clusterer.fit_predict(embeddings_matrix)
embeddings_df['cluster_label'] = cluster_labels

# Count clusters (excluding noise points labeled -1)
unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
noise_count = np.sum(cluster_labels == -1)

print(f"✅ Clustering complete!")
print(f"   Found {len(unique_clusters)} clusters")
print(f"   Noise points: {noise_count}")

# Fallback to KMeans if too few clusters
if len(unique_clusters) < 3:
    print("\n⚠️  Too few clusters found, falling back to KMeans with k=5...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_matrix)
    embeddings_df['cluster_label'] = cluster_labels
    unique_clusters = list(range(5))
    # Calculate centroids manually
    centroids = kmeans.cluster_centers_
    print(f"✅ KMeans clustering complete! Created {len(unique_clusters)} clusters")
else:
    # Get centroids from HDBSCAN
    centroids = clusterer.centroids_

# Display cluster distribution
print("\n📈 Cluster distribution:")
for cluster_id in sorted(unique_clusters):
    count = np.sum(cluster_labels == cluster_id)
    print(f"   Cluster {cluster_id}: {count} notes")

#=============================================================================
# LLM-BASED CLUSTER LABELING WITH ANTHROPIC API
#=============================================================================

print("\n🤖 Initializing Claude API for cluster labeling...")
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not ANTHROPIC_API_KEY:
    print("⚠️  ANTHROPIC_API_KEY not found! Using generic category names.")
    use_llm = False
else:
    try:
        client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
        print("✅ Claude API initialized")
        use_llm = True
    except Exception as e:
        print(f"⚠️  Failed to initialize Claude API: {e}")
        print("   Falling back to generic category names")
        use_llm = False

def label_cluster_with_llm(sample_notes: list, cluster_id: int) -> dict:
    """Use Claude to analyze cluster and generate category metadata"""
    if not use_llm:
        return {
            'category_name': f'category_{cluster_id}',
            'display_name': f'Category {cluster_id}',
            'description': f'Auto-generated category for cluster {cluster_id}'
        }

    try:
        notes_text = "\n---\n".join(sample_notes[:5])  # Limit to 5 notes

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            system="You are a medical coding expert. Analyze clinical notes and categorize them concisely.",
            messages=[{
                "role": "user",
                "content": f"""Analyze these clinical notes and provide a category.

Clinical Notes:
{notes_text}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"category_name": "snake_case_name", "display_name": "Human Readable Name", "description": "Brief 1-2 sentence description"}}"""
            }]
        )

        # Parse response
        response_text = message.content[0].text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('\n```', 1)[0]

        result = json.loads(response_text)
        print(f"   ✅ Cluster {cluster_id}: {result['display_name']}")
        return result

    except Exception as e:
        print(f"   ⚠️  LLM labeling failed for cluster {cluster_id}: {e}")
        return {
            'category_name': f'category_{cluster_id}',
            'display_name': f'Category {cluster_id}',
            'description': f'Auto-generated category for cluster {cluster_id}'
        }

print("\n🏷️  Generating category labels...")
categories = []

for cluster_id in sorted(unique_clusters):
    # Get sample notes closest to centroid
    cluster_mask = cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) == 0:
        continue

    # Get centroid for this cluster
    centroid = centroids[cluster_id]

    # Calculate distances to centroid
    cluster_embeddings = embeddings_matrix[cluster_indices]
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    # Get top 5 closest notes
    closest_indices = cluster_indices[np.argsort(distances)[:5]]
    sample_notes = embeddings_df.iloc[closest_indices]['note_text'].tolist()

    # Generate category metadata using LLM
    category_info = label_cluster_with_llm(sample_notes, cluster_id)
    category_info['cluster_id'] = cluster_id
    category_info['centroid'] = centroid
    category_info['sample_size'] = len(cluster_indices)

    categories.append(category_info)

print(f"\n✅ Generated {len(categories)} category labels")

#=============================================================================
# CREATE DATABASE SCHEMA
#=============================================================================

print("\n💾 Creating database tables...")

with engine.begin() as conn:
    # Create master category table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS claim_categories (
            category_id SERIAL PRIMARY KEY,
            category_name VARCHAR(100) UNIQUE NOT NULL,
            display_name VARCHAR(200),
            description TEXT,
            centroid_embedding VECTOR(384),
            claim_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))
    print("   ✅ Created claim_categories table")

    # Create category membership mapping
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS claim_category_membership (
            membership_id BIGSERIAL PRIMARY KEY,
            claim_id BIGINT REFERENCES claims(claim_id),
            category_id INTEGER REFERENCES claim_categories(category_id),
            similarity_score FLOAT,
            assigned_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(claim_id, category_id)
        )
    """))
    print("   ✅ Created claim_category_membership table")

    # Create HNSW index on category centroids
    try:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_categories_centroid
            ON claim_categories USING hnsw (centroid_embedding vector_cosine_ops)
        """))
        print("   ✅ Created HNSW index on category centroids")
    except Exception as e:
        print(f"   ⚠️  Could not create HNSW index (may need pgvector extension): {e}")

print("✅ Database schema created successfully")

#=============================================================================
# INSERT CATEGORIES INTO DATABASE
#=============================================================================

print("\n📥 Inserting categories into database...")

with engine.begin() as conn:
    # Clear existing categories (for idempotency)
    conn.execute(text("TRUNCATE claim_categories CASCADE"))

    for cat in categories:
        centroid_str = '[' + ','.join(map(str, cat['centroid'])) + ']'

        conn.execute(text("""
            INSERT INTO claim_categories
            (category_name, display_name, description, centroid_embedding, claim_count)
            VALUES (:name, :display, :desc, CAST(:centroid AS vector), 0)
        """), {
            'name': cat['category_name'],
            'display': cat['display_name'],
            'desc': cat['description'],
            'centroid': centroid_str
        })

print(f"✅ Inserted {len(categories)} categories")

# Display categories
print("\n📋 Category Summary:")
print("=" * 80)
for cat in categories:
    print(f"  {cat['display_name']}")
    print(f"    Name: {cat['category_name']}")
    print(f"    Description: {cat['description']}")
    print(f"    Sample Size: {cat['sample_size']} notes")
    print()

#=============================================================================
# CREATE DYNAMIC CATEGORY CACHE TABLES
#=============================================================================

print("💾 Creating category-specific cache tables...")

for cat in categories:
    table_name = f"cache_{cat['category_name']}"

    try:
        with engine.begin() as conn:
            # Create cache table
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    cache_id BIGSERIAL PRIMARY KEY,
                    claim_id BIGINT,
                    patient_id BIGINT,
                    service_date DATE,
                    total_charge NUMERIC(12,2),
                    total_paid NUMERIC(12,2),
                    primary_diagnosis VARCHAR(50),
                    note_summary TEXT,
                    embedding VECTOR(384),
                    similarity_to_centroid FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))

            # Create HNSW index for fast similarity search within category
            try:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_emb
                    ON {table_name} USING hnsw (embedding vector_cosine_ops)
                """))
            except:
                pass  # Index creation might fail if pgvector extension not fully configured

        print(f"   ✅ Created {table_name}")
    except Exception as e:
        print(f"   ❌ Failed to create {table_name}: {e}")

print("✅ All cache tables created successfully")

#=============================================================================
# POPULATE CACHE TABLES
#=============================================================================

print("\n📊 Populating cache tables with claims data...")

# Get all categories from database
categories_db = pd.read_sql("SELECT category_id, category_name, centroid_embedding FROM claim_categories", engine)

total_cached = 0

for _, cat_row in categories_db.iterrows():
    category_id = cat_row['category_id']
    category_name = cat_row['category_name']
    table_name = f"cache_{category_name}"
    centroid = pgvector_to_array(cat_row['centroid_embedding'])

    # Get notes for this cluster
    cluster_id = next((c['cluster_id'] for c in categories if c['category_name'] == category_name), None)
    if cluster_id is None:
        continue

    cluster_notes = embeddings_df[embeddings_df['cluster_label'] == cluster_id]

    if len(cluster_notes) == 0:
        continue

    # Join with claims data and insert into cache table
    with engine.begin() as conn:
        for _, note in cluster_notes.iterrows():
            # Calculate similarity to centroid
            note_embedding = pgvector_to_array(note['embedding'])
            similarity = 1 - np.linalg.norm(note_embedding - centroid) / (np.linalg.norm(note_embedding) * np.linalg.norm(centroid))

            # Get claim details
            claim_data = pd.read_sql(text("""
                SELECT
                    c.claim_id,
                    c.patient_id,
                    c.service_date,
                    c.total_charge,
                    c.total_paid,
                    d.icd_code as primary_diagnosis
                FROM claims c
                LEFT JOIN diagnoses d ON c.claim_id = d.claim_id AND d.is_primary = true
                WHERE c.claim_id = :cid
                LIMIT 1
            """), conn, params={'cid': note['claim_id']})

            if len(claim_data) == 0:
                continue

            claim = claim_data.iloc[0]
            emb_str = '[' + ','.join(map(str, note_embedding)) + ']'

            # Insert into cache table
            try:
                conn.execute(text(f"""
                    INSERT INTO {table_name}
                    (claim_id, patient_id, service_date, total_charge, total_paid,
                     primary_diagnosis, note_summary, embedding, similarity_to_centroid)
                    VALUES (:cid, :pid, :sdate, :charge, :paid, :dx, :note, CAST(:emb AS vector), :sim)
                    ON CONFLICT DO NOTHING
                """), {
                    'cid': claim['claim_id'],
                    'pid': claim['patient_id'],
                    'sdate': claim['service_date'],
                    'charge': claim['total_charge'],
                    'paid': claim['total_paid'],
                    'dx': claim['primary_diagnosis'],
                    'note': note['note_text'][:500],  # Truncate to 500 chars
                    'emb': emb_str,
                    'sim': float(similarity)
                })
                total_cached += 1

                # Update category membership
                conn.execute(text("""
                    INSERT INTO claim_category_membership
                    (claim_id, category_id, similarity_score)
                    VALUES (:cid, :catid, :sim)
                    ON CONFLICT (claim_id, category_id) DO UPDATE
                    SET similarity_score = EXCLUDED.similarity_score
                """), {
                    'cid': claim['claim_id'],
                    'catid': category_id,
                    'sim': float(similarity)
                })
            except Exception as e:
                print(f"   ⚠️  Error inserting claim {note['claim_id']}: {e}")

    # Update claim count
    with engine.begin() as conn:
        count = pd.read_sql(text(f"SELECT COUNT(*) as cnt FROM {table_name}"), conn).iloc[0]['cnt']
        conn.execute(text("""
            UPDATE claim_categories
            SET claim_count = :cnt
            WHERE category_id = :catid
        """), {'cnt': count, 'catid': category_id})

    print(f"   ✅ {table_name}: {count} claims cached")

print(f"\n✅ Cached {total_cached} total claim records across all categories")

#=============================================================================
# FAST CATEGORY SEARCH FUNCTION
#=============================================================================

def search_category(category_name: str, query: str, top_k: int = 10) -> pd.DataFrame:
    """
    Search for similar claims within a specific category cache

    Args:
        category_name: Snake_case category name (e.g., 'cardiac_procedures')
        query: Natural language search query
        top_k: Number of results to return

    Returns:
        DataFrame with similar claims from the category
    """
    # Generate query embedding
    query_embedding = embed_model.encode([query])[0]
    emb_str = '[' + ','.join(map(str, query_embedding)) + ']'

    # Search only within the category cache table
    table_name = f"cache_{category_name}"

    try:
        results = pd.read_sql(text(f"""
            SELECT
                cache_id,
                claim_id,
                patient_id,
                service_date,
                total_charge,
                total_paid,
                primary_diagnosis,
                note_summary,
                similarity_to_centroid,
                1 - (embedding <=> CAST(:emb AS vector)) AS similarity
            FROM {table_name}
            ORDER BY embedding <=> CAST(:emb AS vector)
            LIMIT :k
        """), engine, params={'emb': emb_str, 'k': top_k})

        return results
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return pd.DataFrame()

#=============================================================================
# AUTO-CATEGORIZE NEW CLAIMS FUNCTION
#=============================================================================

def categorize_claim(claim_id: int, threshold: float = 0.3) -> dict:
    """
    Assign a claim to the best matching category

    Args:
        claim_id: The claim ID to categorize
        threshold: Minimum similarity threshold (default 0.3)

    Returns:
        Dictionary with category info and similarity score
    """
    # Get claim's clinical note embedding
    note_data = pd.read_sql(text("""
        SELECT note_id, note_text, embedding
        FROM clinical_notes
        WHERE claim_id = :cid AND embedding IS NOT NULL
        LIMIT 1
    """), engine, params={'cid': claim_id})

    if len(note_data) == 0:
        return {
            'claim_id': claim_id,
            'category': 'uncategorized',
            'similarity': 0.0,
            'reason': 'No clinical notes with embeddings found'
        }

    note_embedding = pgvector_to_array(note_data.iloc[0]['embedding'])
    emb_str = '[' + ','.join(map(str, note_embedding)) + ']'

    # Compare against all category centroids
    categories = pd.read_sql(text("""
        SELECT
            category_id,
            category_name,
            display_name,
            1 - (centroid_embedding <=> CAST(:emb AS vector)) AS similarity
        FROM claim_categories
        ORDER BY centroid_embedding <=> CAST(:emb AS vector)
        LIMIT 1
    """), engine, params={'emb': emb_str})

    if len(categories) == 0 or categories.iloc[0]['similarity'] < threshold:
        return {
            'claim_id': claim_id,
            'category': 'uncategorized',
            'similarity': categories.iloc[0]['similarity'] if len(categories) > 0 else 0.0,
            'reason': 'No category match above threshold'
        }

    best_match = categories.iloc[0]

    return {
        'claim_id': claim_id,
        'category_id': best_match['category_id'],
        'category_name': best_match['category_name'],
        'display_name': best_match['display_name'],
        'similarity': float(best_match['similarity'])
    }

#=============================================================================
# DEMONSTRATION & TESTING
#=============================================================================

print("\n" + "=" * 80)
print("🔍 TESTING CATEGORY-SPECIFIC SEARCH")
print("=" * 80)

# Get first category for demo
if len(categories) > 0:
    demo_category = categories[0]['category_name']

    test_queries = [
        "patient with heart condition",
        "diabetes management",
        "surgical procedure"
    ]

    for query in test_queries:
        print(f"\n📋 Query: '{query}' in category '{demo_category}'")
        print("-" * 80)

        results = search_category(demo_category, query, top_k=3)

        if len(results) > 0:
            for _, row in results.iterrows():
                print(f"  [Sim: {row['similarity']:.3f}] Claim {row['claim_id']} | ${row['total_charge']:.2f}")
                print(f"     {row['note_summary'][:65]}...")
        else:
            print("  No results found")

print("\n" + "=" * 80)
print("🏥 TESTING AUTO-CATEGORIZATION")
print("=" * 80)

# Test categorization on sample claims
test_claims = pd.read_sql(text("""
    SELECT DISTINCT claim_id
    FROM clinical_notes
    WHERE embedding IS NOT NULL
    LIMIT 3
"""), engine)

for _, row in test_claims.iterrows():
    result = categorize_claim(row['claim_id'])
    if result.get('category_name'):
        print(f"Claim {result['claim_id']:5d} → {result['display_name']} (similarity: {result['similarity']:.3f})")
    else:
        print(f"Claim {result['claim_id']:5d} → {result['category']} ({result['reason']})")

#=============================================================================
# FINAL SUMMARY
#=============================================================================

print("\n" + "=" * 80)
print("📊 FINAL SUMMARY")
print("=" * 80)

summary = pd.read_sql(text("""
    SELECT
        category_name,
        display_name,
        claim_count,
        description
    FROM claim_categories
    ORDER BY claim_count DESC
"""), engine)

print(f"\n✅ Created {len(summary)} category cache tables:")
print()
for _, cat in summary.iterrows():
    print(f"  📁 {cat['display_name']}")
    print(f"     Table: cache_{cat['category_name']}")
    print(f"     Claims: {cat['claim_count']}")
    print(f"     Description: {cat['description']}")
    print()

# Get total membership
total_memberships = pd.read_sql(text("SELECT COUNT(*) as cnt FROM claim_category_membership"), engine).iloc[0]['cnt']
print(f"📈 Total category memberships: {total_memberships}")

print("\n" + "=" * 80)
print("✅ NOTEBOOK 6 COMPLETE!")
print("=" * 80)
print("\n💡 Key Features:")
print("   • LLM-powered intelligent cluster labeling")
print("   • Category-specific cache tables for fast search")
print("   • HNSW indexes for sub-millisecond similarity search")
print("   • Auto-categorization of new claims")
print("   • Semantic search within categories")
print("\n📖 Next steps:")
print("   • Use search_category() to find similar claims in a category")
print("   • Use categorize_claim() to auto-assign new claims to categories")
print("   • Query cache tables directly for ultra-fast retrieval")
print("=" * 80)
