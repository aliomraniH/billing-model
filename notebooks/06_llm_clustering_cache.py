"""
Medical Billing ML - Notebook 6: LLM-Powered Clustering & Category Cache
Architecture: Vercel Postgres (data) + Pinecone (vectors) + Claude (LLM)

This notebook:
1. Loads embeddings from Pinecone
2. Clusters using HDBSCAN (sklearn 1.3+)
3. Uses Claude to label clusters
4. Creates category tables for fast search
"""

print("=" * 70)
print("🤖 MEDICAL BILLING ML - LLM CLUSTERING & CATEGORY CACHE")
print("=" * 70)
print("✅ Using sklearn HDBSCAN (v1.3+)")
print("✅ Claude API: claude-sonnet-4-5-20250929")
print("=" * 70 + "\n")

# ============================================================
# INSTALL DEPENDENCIES (use new pinecone package)
# ============================================================
# ⚠️ Pinecone client rename: uninstall legacy `pinecone-client`
# !pip uninstall -y pinecone-client
# Then install required packages (no local torch needed)
# !pip install -q huggingface_hub pinecone anthropic numpy pandas sqlalchemy psycopg2-binary requests scikit-learn

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import time
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURATION
# ============================================================
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

MODEL_ID = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("HF_EMBEDDING_DIM", 384))
PINECONE_INDEX = "medical-billing-notes"

# Validate
missing = []
if not DATABASE_URL: missing.append("VERCEL_POSTGRES_URL")
if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
if not HF_TOKEN: missing.append("HF_TOKEN")
if missing:
    raise ValueError(f"Missing required: {', '.join(missing)}")

# ANTHROPIC_API_KEY is optional - will use generic names if missing
if not ANTHROPIC_API_KEY:
    print("⚠️ ANTHROPIC_API_KEY not set - will use generic category names")

print("✅ Environment validated")

# ============================================================
# INITIALIZE CLIENTS
# ============================================================
print("\n🔌 Initializing connections...")

# Database
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM claims"))
    print(f"   ✅ Postgres: {result.fetchone()[0]:,} claims")

# Pinecone
# Auto-install the renamed Pinecone client if it's missing (avoids ModuleNotFoundError)
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("pinecone") is None:
    print("   📦 Installing Pinecone client (renamed from `pinecone-client`)...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-client"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pinecone"])

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
stats = index.describe_index_stats()
print(f"   ✅ Pinecone: {stats.total_vector_count:,} vectors")

# HuggingFace
# Ensure huggingface_hub is present (avoids ModuleNotFoundError)
if importlib.util.find_spec("huggingface_hub") is None:
    print("   📦 Installing huggingface_hub (needed for InferenceClient)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])

from huggingface_hub import HfApi, InferenceClient

try:
    info = HfApi(token=HF_TOKEN).model_info(MODEL_ID)
    pipeline = getattr(info, "pipeline_tag", None)
    print(f"   ✅ HuggingFace: {MODEL_ID} (pipeline: {pipeline or 'unknown'})")
except Exception as exc:
    print(f"   ⚠️ Could not verify model availability ({exc}). Ensure the model supports feature extraction.")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
) if HF_TOKEN else None

# Anthropic
anthropic_client = None
if ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic()
    print("   ✅ Anthropic: claude-sonnet-4-5-20250929")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_embedding(text: str, model_id: Optional[str] = None) -> np.ndarray:
    """Generate embedding using HF Inference API"""
    if not hf_client:
        raise ValueError("HF_TOKEN required for embeddings")

    model_to_use = model_id or MODEL_ID
    result = hf_client.feature_extraction(text, model=model_to_use)
    embedding = np.array(result)
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)
    embedding = embedding.astype(np.float32)

    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension {embedding.shape[0]} does not match expected {EMBEDDING_DIM}. "
            "Update EMBEDDING_DIM/Pinecone index or choose a compatible model."
        )

    return embedding

# ============================================================
# LOAD ALL VECTORS FROM PINECONE
# ============================================================
print("\n📥 Loading vectors from Pinecone...")

# Fetch all vectors (for small datasets)
# For large datasets, use pagination with index.list()
all_vectors = []
all_metadata = []
all_ids = []

# Query with a random vector to get all results (hacky but works for small datasets)
# Better approach: iterate through known IDs
results = index.query(
    vector=[0.0] * EMBEDDING_DIM,  # Dummy vector
    top_k=10000,  # Max allowed
    include_values=True,
    include_metadata=True
)

for match in results.matches:
    all_ids.append(match.id)
    all_vectors.append(match.values)
    all_metadata.append(match.metadata)

X = np.array(all_vectors)
print(f"   ✅ Loaded {len(X)} vectors, shape: {X.shape}")

# ============================================================
# CLUSTERING WITH HDBSCAN (sklearn 1.3+)
# ============================================================
print("\n🔬 Clustering with HDBSCAN...")

from sklearn.cluster import HDBSCAN

# IMPORTANT: Use sklearn's HDBSCAN, not the old standalone package
clusterer = HDBSCAN(
    min_cluster_size=2,  # Small for demo data
    min_samples=1,
    metric='euclidean',
    cluster_selection_method='eom',
    store_centers='centroid',  # NEW in sklearn - stores cluster centroids!
)

cluster_labels = clusterer.fit_predict(X)
unique_labels = set(cluster_labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"   ✅ Found {n_clusters} clusters")
print(f"   ⚠️ Noise points: {n_noise}")

# Get centroids (only available with store_centers='centroid')
if hasattr(clusterer, 'centroids_') and clusterer.centroids_ is not None:
    centroids = clusterer.centroids_
    print(f"   ✅ Centroids shape: {centroids.shape}")
else:
    # Calculate manually if not available
    centroids = []
    for label in range(n_clusters):
        mask = cluster_labels == label
        centroid = X[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    print(f"   ✅ Calculated {len(centroids)} centroids")

# Fallback to KMeans if too few clusters
if n_clusters < 3:
    print("\n⚠️ HDBSCAN found too few clusters, falling back to KMeans...")
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    n_clusters = 5
    print(f"   ✅ KMeans: {n_clusters} clusters")

# ============================================================
# LLM CLUSTER LABELING
# ============================================================
print("\n🤖 Labeling clusters with LLM...")

def label_cluster_with_llm(sample_notes: list, cluster_idx: int) -> dict:
    """Use Claude to generate category name and description"""
    if not anthropic_client:
        return {
            "category_name": f"category_{cluster_idx}",
            "display_name": f"Category {cluster_idx}",
            "description": f"Auto-generated cluster {cluster_idx}"
        }

    notes_text = "\n---\n".join(sample_notes[:5])  # Max 5 samples

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            system="You are a medical coding expert. Analyze clinical notes and categorize them. Respond with ONLY valid JSON, no markdown.",
            messages=[{
                "role": "user",
                "content": f"""Analyze these clinical notes and provide a category.

Notes:
{notes_text}

Respond with ONLY this JSON structure:
{{"category_name": "snake_case_name", "display_name": "Human Readable Name", "description": "Brief 1-2 sentence description"}}"""
            }]
        )

        response_text = message.content[0].text.strip()
        # Clean potential markdown
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        return json.loads(response_text)

    except Exception as e:
        print(f"   ⚠️ LLM error: {e}")
        return {
            "category_name": f"category_{cluster_idx}",
            "display_name": f"Category {cluster_idx}",
            "description": f"Medical claims cluster {cluster_idx}"
        }

# Label each cluster
categories = []
for cluster_idx in range(n_clusters):
    print(f"   [{cluster_idx + 1}/{n_clusters}] Labeling cluster {cluster_idx}... ", end="", flush=True)

    # Get sample notes for this cluster
    mask = cluster_labels == cluster_idx
    cluster_indices = np.where(mask)[0]

    sample_notes = []
    for idx in cluster_indices[:5]:
        meta = all_metadata[idx]
        sample_notes.append(meta.get('text_preview', ''))

    # Get LLM label
    label_info = label_cluster_with_llm(sample_notes, cluster_idx)
    label_info['cluster_idx'] = cluster_idx
    label_info['size'] = int(mask.sum())
    label_info['centroid'] = centroids[cluster_idx].tolist()

    categories.append(label_info)
    print(f"'{label_info['display_name']}' ({label_info['size']} items)")

print(f"\n✅ Labeled {len(categories)} categories")

# ============================================================
# CREATE DATABASE TABLES
# ============================================================
print("\n📋 Creating database tables...")

with engine.begin() as conn:
    # Master category table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS claim_categories (
            category_id SERIAL PRIMARY KEY,
            category_name VARCHAR(100) UNIQUE NOT NULL,
            display_name VARCHAR(200),
            description TEXT,
            centroid_json TEXT,
            claim_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))
    print("   ✅ claim_categories table")

    # Category membership mapping
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS claim_category_membership (
            membership_id BIGSERIAL PRIMARY KEY,
            claim_id BIGINT,
            category_id INTEGER REFERENCES claim_categories(category_id),
            similarity_score FLOAT,
            assigned_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(claim_id, category_id)
        )
    """))
    print("   ✅ claim_category_membership table")

    # Clear existing categories (for re-runs)
    conn.execute(text("DELETE FROM claim_category_membership"))
    conn.execute(text("DELETE FROM claim_categories"))

# ============================================================
# POPULATE CATEGORIES
# ============================================================
print("\n📊 Populating categories...")

with engine.begin() as conn:
    for cat in categories:
        # Insert category
        result = conn.execute(text("""
            INSERT INTO claim_categories (category_name, display_name, description, centroid_json, claim_count)
            VALUES (:name, :display, :desc, :centroid, :count)
            RETURNING category_id
        """), {
            'name': cat['category_name'],
            'display': cat['display_name'],
            'desc': cat['description'],
            'centroid': json.dumps(cat['centroid']),
            'count': cat['size']
        })
        category_id = result.fetchone()[0]
        cat['category_id'] = category_id
        print(f"   ✅ {cat['display_name']}: {cat['size']} items")

# ============================================================
# ASSIGN CLAIMS TO CATEGORIES
# ============================================================
print("\n🔗 Assigning claims to categories...")

with engine.begin() as conn:
    for i, (vector_id, metadata, label) in enumerate(zip(all_ids, all_metadata, cluster_labels)):
        if label == -1:  # Skip noise
            continue

        claim_id = metadata.get('claim_id')
        if not claim_id:
            continue

        # Find category
        cat = next((c for c in categories if c['cluster_idx'] == label), None)
        if not cat:
            continue

        # Calculate similarity to centroid
        vec = np.array(all_vectors[i])
        centroid = np.array(cat['centroid'])
        similarity = float(np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid)))

        # Insert membership
        conn.execute(text("""
            INSERT INTO claim_category_membership (claim_id, category_id, similarity_score)
            VALUES (:cid, :cat_id, :sim)
            ON CONFLICT (claim_id, category_id) DO UPDATE SET
                similarity_score = EXCLUDED.similarity_score
        """), {
            'cid': claim_id,
            'cat_id': cat['category_id'],
            'sim': similarity
        })

print("   ✅ Claims assigned to categories")

# ============================================================
# CATEGORY SEARCH FUNCTION
# ============================================================
def search_by_category(
    query: str,
    category_name: Optional[str] = None,
    top_k: int = 5,
    model_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search for similar claims, optionally filtered by category.
    """
    query_emb = get_embedding(query, model_id=model_id)

    # Build filter
    filter_dict = None
    if category_name:
        filter_dict = {"category": {"$eq": category_name}}

    results = index.query(
        vector=query_emb.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    rows = []
    for match in results.matches:
        rows.append({
            'claim_id': match.metadata.get('claim_id'),
            'category': match.metadata.get('category'),
            'note_type': match.metadata.get('note_type'),
            'similarity': match.score,
            'preview': match.metadata.get('text_preview', '')[:80]
        })

    return pd.DataFrame(rows)

def auto_categorize_claim(
    claim_id: int,
    note_text: str,
    model_id: Optional[str] = None,
) -> dict:
    """
    Automatically categorize a new claim based on its clinical note.
    """
    note_emb = get_embedding(note_text, model_id=model_id)

    # Find best matching category by comparing to centroids
    best_cat = None
    best_sim = -1

    for cat in categories:
        centroid = np.array(cat['centroid'])
        sim = float(np.dot(note_emb, centroid) / (np.linalg.norm(note_emb) * np.linalg.norm(centroid)))
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    return {
        'claim_id': claim_id,
        'category': best_cat['category_name'] if best_cat else 'uncategorized',
        'display_name': best_cat['display_name'] if best_cat else 'Uncategorized',
        'similarity': best_sim
    }

# ============================================================
# DEMO: CATEGORY SEARCH
# ============================================================
print("\n" + "=" * 70)
print("🔍 DEMO: Category-Filtered Search")
print("=" * 70)

demo_queries = [
    ("heart attack chest pain", None),
    ("diabetes blood sugar", None),
]

for query, cat_filter in demo_queries:
    print(f"\nQuery: '{query}'" + (f" [filtered: {cat_filter}]" if cat_filter else ""))
    print("-" * 50)

    results = search_by_category(query, cat_filter, top_k=3)
    for _, row in results.iterrows():
        print(f"  [{row['similarity']:.3f}] {row['category']}: {row['preview']}...")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ NOTEBOOK 06 COMPLETE")
print("=" * 70)

# Final stats
with engine.connect() as conn:
    cat_count = conn.execute(text("SELECT COUNT(*) FROM claim_categories")).fetchone()[0]
    mem_count = conn.execute(text("SELECT COUNT(*) FROM claim_category_membership")).fetchone()[0]

print(f"""
📊 Summary:
   • Categories created: {cat_count}
   • Claims categorized: {mem_count}
   • Clustering: {'HDBSCAN' if n_clusters >= 3 else 'KMeans (fallback)'}
   • LLM labeling: {'Claude' if anthropic_client else 'Generic names'}

📋 New Tables:
   • claim_categories - Master category definitions
   • claim_category_membership - Claim-to-category mappings

🔍 Available Functions:
   • search_by_category(query, category_name, top_k)
   • auto_categorize_claim(claim_id, note_text)

🚀 Pipeline complete!
""")
print("=" * 70)
