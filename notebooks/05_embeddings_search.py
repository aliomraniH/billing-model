# Medical Billing ML - Notebook 5: Embeddings & Similarity Search
# Architecture: Vercel Postgres (notes) + Pinecone (vectors)

# %%
get_ipython().system('pip install -q sqlalchemy psycopg2-binary pandas numpy huggingface_hub pinecone-client')

# %%
import os
import time
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from pinecone import Pinecone, ServerlessSpec

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Get free key at pinecone.io

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
PINECONE_INDEX_NAME = "medical-billing-notes"

# Validate environment
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found!")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found! Get free key at https://www.pinecone.io/")

print("✅ Environment variables loaded")

# =============================================================================
# INITIALIZE CLIENTS
# =============================================================================

# Database
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM claims"))
    print(f"✅ Postgres connected! Found {result.fetchone()[0]:,} claims")

# HuggingFace
hf_client = InferenceClient(token=HF_API_KEY if HF_API_KEY else None)
print("✅ HuggingFace client ready")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if doesn't exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"📦 Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)  # Wait for index to be ready
    print("✅ Pinecone index created!")
else:
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' exists")

index = pc.Index(PINECONE_INDEX_NAME)
print(f"✅ Pinecone ready! Index stats: {index.describe_index_stats()}")

# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text: str, retries: int = 3) -> list:
    """Get embedding using HuggingFace InferenceClient"""
    for attempt in range(retries):
        try:
            embedding = hf_client.feature_extraction(text, model=EMBEDDING_MODEL)
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    return embedding[0]
            return list(embedding)
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise

# Test embedding
test_emb = get_embedding("Test sentence")
print(f"✅ Embeddings working! Dimension: {len(test_emb)}")

# =============================================================================
# DATABASE SETUP (Postgres - notes only, no vectors)
# =============================================================================

print("\n🔧 Setting up Postgres schema...")

with engine.begin() as conn:
    # Create clinical_notes table (no embedding column!)
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS clinical_notes (
            note_id SERIAL PRIMARY KEY,
            claim_id INTEGER,
            note_type VARCHAR(50),
            note_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    # Create unique constraint for upsert
    conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clinical_notes_claim_type
        ON clinical_notes (claim_id, note_type)
    """))

print("✅ Postgres schema ready!")

# =============================================================================
# SAMPLE DATA
# =============================================================================

sample_notes = [
    {"note_type": "discharge", "note_text": "Patient admitted with uncontrolled Type 2 diabetes mellitus. Blood glucose stabilized with insulin. HbA1c 9.2%. Discharged on adjusted metformin."},
    {"note_type": "discharge", "note_text": "Acute chest pain with ST elevation in V1-V4. Emergent PCI with drug-eluting stent to LAD. Post-MI protocol initiated."},
    {"note_type": "discharge", "note_text": "Community-acquired pneumonia with right lower lobe consolidation. IV antibiotics transitioned to oral azithromycin. Oxygen resolved day 3."},
    {"note_type": "operative", "note_text": "Right total knee arthroplasty for end-stage osteoarthritis. Cemented components. EBL 150mL. Tolerated well."},
    {"note_type": "procedure", "note_text": "Colonoscopy for CRC screening. Two 5mm tubular adenomas removed from ascending colon. No malignancy. Repeat in 5 years."},
]

# Get claim IDs
claim_ids = pd.read_sql("SELECT claim_id FROM claims LIMIT 5", engine)['claim_id'].tolist()
print(f"\n📝 Using {len(claim_ids)} claim IDs")

# =============================================================================
# STORE NOTES + EMBEDDINGS
# =============================================================================

print("\n🔄 Storing notes in Postgres and embeddings in Pinecone...")

vectors_to_upsert = []

with engine.begin() as conn:
    for i, note in enumerate(sample_notes):
        if i >= len(claim_ids):
            break

        print(f"   Processing note {i+1}/{len(sample_notes)}...", end=" ")

        # 1. Store note in Postgres
        result = conn.execute(text("""
            INSERT INTO clinical_notes (claim_id, note_type, note_text)
            VALUES (:cid, :ntype, :ntext)
            ON CONFLICT (claim_id, note_type)
            DO UPDATE SET note_text = EXCLUDED.note_text
            RETURNING note_id
        """), {'cid': claim_ids[i], 'ntype': note['note_type'], 'ntext': note['note_text']})

        note_id = result.fetchone()[0]

        # 2. Generate embedding
        embedding = get_embedding(note['note_text'])

        # 3. Prepare for Pinecone (batch upsert)
        vectors_to_upsert.append({
            "id": f"note_{note_id}",
            "values": embedding,
            "metadata": {
                "note_id": note_id,
                "claim_id": claim_ids[i],
                "note_type": note['note_type'],
                "model": EMBEDDING_MODEL
            }
        })

        print("✓")
        time.sleep(1)  # Rate limiting

# Batch upsert to Pinecone
index.upsert(vectors=vectors_to_upsert)
print(f"\n✅ Stored {len(vectors_to_upsert)} notes with embeddings!")

# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def search_similar_notes(query: str, top_k: int = 5):
    """Search for similar notes using Pinecone"""
    query_embedding = get_embedding(query)

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Fetch full note text from Postgres
    output = []
    for match in results.matches:
        note_id = match.metadata['note_id']
        note_data = pd.read_sql(
            text("SELECT * FROM clinical_notes WHERE note_id = :nid"),
            engine, params={'nid': note_id}
        )
        if len(note_data) > 0:
            row = note_data.iloc[0]
            output.append({
                'note_id': note_id,
                'claim_id': row['claim_id'],
                'note_type': row['note_type'],
                'note_text': row['note_text'],
                'similarity': match.score
            })

    return pd.DataFrame(output)

# =============================================================================
# TEST SEARCHES
# =============================================================================

print("\n" + "="*70)
print("🔍 TESTING SEMANTIC SIMILARITY SEARCH")
print("="*70)

queries = [
    "diabetes blood sugar insulin",
    "heart attack cardiac stent",
    "knee replacement surgery"
]

for query in queries:
    print(f"\n📋 Query: '{query}'")
    print("-" * 70)
    results = search_similar_notes(query, top_k=3)
    for _, row in results.iterrows():
        print(f"  [{row['similarity']:.3f}] {row['note_text'][:65]}...")

# =============================================================================
# BILLING CODE VALIDATION
# =============================================================================

CODE_DESCRIPTIONS = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'I21.0': 'ST elevation myocardial infarction anterior wall',
    'J18.9': 'Pneumonia unspecified organism',
    'M17.11': 'Primary osteoarthritis right knee',
}

def validate_billing_code(claim_id: int, icd_code: str):
    """Validate if billing code matches clinical documentation"""
    # Get notes for this claim from Postgres
    notes = pd.read_sql(
        text("SELECT note_id, note_text FROM clinical_notes WHERE claim_id = :cid"),
        engine, params={'cid': claim_id}
    )

    if len(notes) == 0:
        return {'valid': None, 'reason': 'No clinical notes found', 'similarity': 0.0}

    # Get code description embedding
    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = get_embedding(code_desc)

    # Query Pinecone for this claim's notes
    note_ids = [f"note_{nid}" for nid in notes['note_id'].tolist()]

    # Search with filter (if notes exist in Pinecone)
    results = index.query(
        vector=code_emb,
        top_k=1,
        filter={"claim_id": claim_id},
        include_metadata=True
    )

    max_sim = results.matches[0].score if results.matches else 0.0

    return {
        'code': icd_code,
        'description': code_desc,
        'similarity': float(max_sim),
        'valid': max_sim >= 0.4
    }

print("\n" + "="*70)
print("🏥 BILLING CODE VALIDATION")
print("="*70)

for claim_id in claim_ids[:3]:
    result = validate_billing_code(claim_id, 'E11.9')
    status = '✅ VALID' if result['valid'] else '❌ MISMATCH'
    print(f"Claim {claim_id:5d} | Code: E11.9 | Similarity: {result['similarity']:.3f} | {status}")

print("="*70)
print("\n✅ Setup complete!")
print("\n💡 Environment variables needed:")
print("   - VERCEL_POSTGRES_URL (you have this)")
print("   - PINECONE_API_KEY (free at https://www.pinecone.io/)")
print("   - HUGGINGFACE_API_KEY (optional, for higher rate limits)")
