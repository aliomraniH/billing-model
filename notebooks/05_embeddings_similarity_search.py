"""
Medical Billing ML - Notebook 5: Embeddings & Similarity Search
Architecture: Vercel Postgres (notes) + Pinecone (vectors)

FIXES APPLIED (December 2025):
- HF Inference API: router.huggingface.co/hf-inference
- Model: BAAI/bge-small-en-v1.5 (feature_extraction compatible)
- Hybrid storage: Pinecone for vectors, Postgres for text
"""

print("=" * 70)
print("📦 MEDICAL BILLING ML - EMBEDDINGS & SIMILARITY SEARCH")
print("=" * 70)
print("✅ Architecture: Vercel Postgres + Pinecone (hybrid)")
print("✅ HF API: router.huggingface.co (December 2025)")
print("=" * 70 + "\n")

# ============================================================
# IMPORTS
# ============================================================
import os
import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURATION
# ============================================================
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# CRITICAL: Use a model that works with feature_extraction on HF Inference
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
PINECONE_INDEX = "medical-billing-notes"

# Validate
missing = []
if not DATABASE_URL: missing.append("VERCEL_POSTGRES_URL")
if not HF_TOKEN: missing.append("HF_TOKEN")
if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")

if missing:
    print("❌ Missing environment variables:")
    for var in missing:
        print(f"   - {var}")
    print("\n📋 Setup instructions:")
    print("   HF_TOKEN: https://huggingface.co/settings/tokens (enable 'Inference Providers')")
    print("   PINECONE_API_KEY: https://www.pinecone.io/ (free signup)")
    raise ValueError(f"Missing: {', '.join(missing)}")

print("✅ All environment variables loaded")

# ============================================================
# DATABASE CONNECTION (Vercel Postgres)
# ============================================================
print("\n🔌 Connecting to Vercel Postgres...")

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM claims"))
    total_claims = result.fetchone()[0]
    print(f"   ✅ Connected! Found {total_claims:,} claims")

# ============================================================
# PINECONE INITIALIZATION
# ============================================================
print("\n🌲 Initializing Pinecone...")

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, create if not
existing_indexes = [idx.name for idx in pc.list_indexes()]

if PINECONE_INDEX not in existing_indexes:
    print(f"   Creating index '{PINECONE_INDEX}'...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Free tier region
        )
    )
    # Wait for index to be ready
    time.sleep(10)
    print(f"   ✅ Index created")
else:
    print(f"   ✅ Index '{PINECONE_INDEX}' exists")

index = pc.Index(PINECONE_INDEX)
stats = index.describe_index_stats()
print(f"   Vectors in index: {stats.total_vector_count:,}")

# ============================================================
# HUGGING FACE INFERENCE CLIENT (December 2025 API)
# ============================================================
print(f"\n🤗 Setting up HuggingFace embeddings...")
print(f"   Model: {MODEL_ID}")
print(f"   Dimensions: {EMBEDDING_DIM}")

from huggingface_hub import InferenceClient

# NEW 2025 API: Must use provider="hf-inference"
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding using HF Inference API (December 2025)"""
    result = hf_client.feature_extraction(
        text,
        model=MODEL_ID
    )
    embedding = np.array(result)
    # Mean pooling if token-level embeddings returned
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)
    return embedding.astype(np.float32)

# Test embedding
print("\n🧪 Testing embedding generation...")
try:
    test_emb = get_embedding("Patient with Type 2 diabetes mellitus")
    assert test_emb.shape == (EMBEDDING_DIM,), f"Expected ({EMBEDDING_DIM},), got {test_emb.shape}"
    print(f"   ✅ Shape: {test_emb.shape}")
    print(f"   ✅ Sample: [{test_emb[0]:.4f}, {test_emb[1]:.4f}, ...]")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    raise

# ============================================================
# ENSURE CLINICAL_NOTES TABLE (Postgres - text only, no vectors)
# ============================================================
print("\n📋 Checking clinical_notes table...")

with engine.begin() as conn:
    # Check if table exists
    result = conn.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'clinical_notes'
        )
    """))
    table_exists = result.scalar()

    if not table_exists:
        print("   Creating clinical_notes table (text storage only)...")
        conn.execute(text("""
            CREATE TABLE clinical_notes (
                note_id SERIAL PRIMARY KEY,
                claim_id INTEGER REFERENCES claims(claim_id),
                note_type VARCHAR(50),
                note_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(claim_id, note_type)
            )
        """))
        print("   ✅ Table created")
    else:
        # Check if old embedding column exists (migration from pgvector)
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'clinical_notes' AND column_name = 'embedding'
        """))
        if result.fetchone():
            print("   ⚠️ Found old 'embedding' column (pgvector)")
            print("   Migrating to Pinecone architecture (dropping column)...")
            conn.execute(text("ALTER TABLE clinical_notes DROP COLUMN IF EXISTS embedding"))
            print("   ✅ Migrated to hybrid architecture")
        else:
            print("   ✅ Table exists with correct schema")

# ============================================================
# SAMPLE CLINICAL NOTES
# ============================================================
print("\n📝 Preparing sample clinical notes...")

SAMPLE_NOTES = [
    {"note_type": "discharge", "note_text": "Patient admitted with uncontrolled Type 2 diabetes mellitus. Blood glucose stabilized with insulin. HbA1c 9.2%. Discharged on adjusted metformin.", "category": "diabetes"},
    {"note_type": "encounter", "note_text": "Diabetes follow-up. A1C improved to 7.1%. Mild peripheral neuropathy. Continue metformin and glipizide.", "category": "diabetes"},
    {"note_type": "discharge", "note_text": "Acute chest pain with ST elevation in V1-V4. Emergent PCI with drug-eluting stent to LAD. Post-MI protocol initiated.", "category": "cardiac"},
    {"note_type": "procedure", "note_text": "Cardiac catheterization for unstable angina. 80% stenosis RCA. Stent placed successfully.", "category": "cardiac"},
    {"note_type": "discharge", "note_text": "Community-acquired pneumonia with right lower lobe consolidation. IV antibiotics transitioned to oral azithromycin.", "category": "respiratory"},
    {"note_type": "discharge", "note_text": "COPD exacerbation with respiratory failure. BiPAP initiated. Steroids and bronchodilators given.", "category": "respiratory"},
    {"note_type": "operative", "note_text": "Right total knee arthroplasty for end-stage osteoarthritis. Cemented components. EBL 150mL.", "category": "orthopedic"},
    {"note_type": "operative", "note_text": "Left hip replacement for avascular necrosis. Uncemented prosthesis. PT started POD1.", "category": "orthopedic"},
    {"note_type": "procedure", "note_text": "Colonoscopy for CRC screening. Two 5mm tubular adenomas removed. No malignancy.", "category": "gi"},
    {"note_type": "operative", "note_text": "Laparoscopic cholecystectomy for symptomatic cholelithiasis. Multiple gallstones. No complications.", "category": "gi"},
]

# Get claim IDs
claim_ids = pd.read_sql(
    f"SELECT claim_id FROM claims LIMIT {len(SAMPLE_NOTES)}",
    engine
)['claim_id'].tolist()

print(f"   Found {len(claim_ids)} claim IDs for sample notes")

# ============================================================
# STORE NOTES IN POSTGRES + EMBEDDINGS IN PINECONE
# ============================================================
print("\n🔄 Processing notes (Postgres + Pinecone)...")

pinecone_vectors = []
notes_stored = 0

with engine.begin() as conn:
    for i, note in enumerate(SAMPLE_NOTES):
        if i >= len(claim_ids):
            break

        claim_id = claim_ids[i]
        print(f"   [{i+1}/{len(SAMPLE_NOTES)}] Processing claim {claim_id}... ", end="", flush=True)

        # 1. Store text in Postgres
        conn.execute(text("""
            INSERT INTO clinical_notes (claim_id, note_type, note_text)
            VALUES (:cid, :ntype, :ntext)
            ON CONFLICT (claim_id, note_type) DO UPDATE SET
                note_text = EXCLUDED.note_text
            RETURNING note_id
        """), {
            'cid': claim_id,
            'ntype': note['note_type'],
            'ntext': note['note_text']
        })

        # 2. Generate embedding
        embedding = get_embedding(note['note_text'])

        # 3. Prepare for Pinecone batch upsert
        vector_id = f"claim_{claim_id}_{note['note_type']}"
        pinecone_vectors.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": {
                "claim_id": claim_id,
                "note_type": note['note_type'],
                "category": note['category'],
                "text_preview": note['note_text'][:200]
            }
        })

        notes_stored += 1
        print("✓")

# Batch upsert to Pinecone
print(f"\n   Uploading {len(pinecone_vectors)} vectors to Pinecone...")
index.upsert(vectors=pinecone_vectors)
print(f"   ✅ Stored {notes_stored} notes in Postgres")
print(f"   ✅ Uploaded {len(pinecone_vectors)} vectors to Pinecone")

# ============================================================
# SIMILARITY SEARCH FUNCTION
# ============================================================
def search_similar_notes(query: str, top_k: int = 5, category_filter: str = None) -> pd.DataFrame:
    """
    Search for similar clinical notes using Pinecone.

    Args:
        query: Search query text
        top_k: Number of results to return
        category_filter: Optional category to filter by (diabetes, cardiac, etc.)

    Returns:
        DataFrame with matching notes and similarity scores
    """
    # Generate query embedding
    query_embedding = get_embedding(query)

    # Build filter
    filter_dict = None
    if category_filter:
        filter_dict = {"category": {"$eq": category_filter}}

    # Search Pinecone
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    # Format results
    rows = []
    for match in results.matches:
        rows.append({
            'id': match.id,
            'claim_id': match.metadata.get('claim_id'),
            'note_type': match.metadata.get('note_type'),
            'category': match.metadata.get('category'),
            'similarity': match.score,
            'text_preview': match.metadata.get('text_preview', '')
        })

    return pd.DataFrame(rows)

# ============================================================
# TEST SIMILARITY SEARCH
# ============================================================
print("\n" + "=" * 70)
print("🔍 TESTING SEMANTIC SIMILARITY SEARCH")
print("=" * 70)

test_queries = [
    ("diabetes blood sugar insulin", "diabetes"),
    ("heart attack cardiac chest pain", "cardiac"),
    ("knee hip joint replacement", "orthopedic"),
    ("pneumonia lung respiratory", "respiratory"),
    ("colonoscopy gallbladder", "gi"),
]

for query, expected in test_queries:
    print(f"\n📋 Query: '{query}'")
    print(f"   Expected category: {expected}")
    print("-" * 50)

    results = search_similar_notes(query, top_k=2)
    for _, row in results.iterrows():
        match_indicator = "✅" if row['category'] == expected else "⚠️"
        print(f"   {match_indicator} [{row['similarity']:.3f}] {row['category']}: {row['text_preview'][:50]}...")

# ============================================================
# BILLING CODE VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("🏥 BILLING CODE VALIDATION")
print("=" * 70)

CODE_DESCRIPTIONS = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'I21.0': 'ST elevation myocardial infarction anterior wall',
    'J18.9': 'Pneumonia unspecified organism',
    'M17.11': 'Primary osteoarthritis right knee',
    'K80.20': 'Calculus of gallbladder without cholecystitis',
}

def validate_billing_code(claim_id: int, icd_code: str, threshold: float = 0.35) -> dict:
    """
    Validate if a billing code matches the clinical documentation.

    Uses semantic similarity between code description and clinical notes.
    """
    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = get_embedding(code_desc)

    # Search Pinecone for this claim's notes
    results = index.query(
        vector=code_emb.tolist(),
        top_k=1,
        include_metadata=True,
        filter={"claim_id": {"$eq": claim_id}}
    )

    if not results.matches:
        return {'code': icd_code, 'valid': None, 'similarity': 0.0, 'reason': 'No notes found'}

    best_match = results.matches[0]
    similarity = best_match.score

    return {
        'code': icd_code,
        'description': code_desc,
        'similarity': float(similarity),
        'valid': similarity >= threshold,
        'matched_category': best_match.metadata.get('category', 'unknown')
    }

print("\nValidating billing codes against clinical documentation:\n")

# Test cases: (claim_id_index, icd_code, should_match)
test_cases = [
    (0, 'E11.9', True),   # Diabetes claim → diabetes code
    (2, 'I21.0', True),   # Cardiac claim → cardiac code
    (6, 'M17.11', True),  # Ortho claim → ortho code
    (0, 'I21.0', False),  # Diabetes claim → cardiac code (mismatch)
]

for idx, code, expected_valid in test_cases:
    if idx < len(claim_ids):
        claim_id = claim_ids[idx]
        result = validate_billing_code(claim_id, code)
        status = "✅ VALID" if result['valid'] else "❌ MISMATCH"
        expected_match = "✓" if (result['valid'] == expected_valid) else "⚠️"
        print(f"{expected_match} Claim {claim_id:5d} | {code} | Sim: {result['similarity']:.3f} | {status}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ NOTEBOOK 05 COMPLETE")
print("=" * 70)

final_stats = index.describe_index_stats()

print(f"""
📊 Summary:
   • Database: Vercel Postgres (text storage)
   • Vector DB: Pinecone (embeddings)
   • Model: {MODEL_ID}
   • Dimensions: {EMBEDDING_DIM}
   • Notes stored: {notes_stored}
   • Vectors in Pinecone: {final_stats.total_vector_count}

🔧 December 2025 Fixes Applied:
   • HF API: router.huggingface.co/hf-inference
   • Model: BAAI/bge-small-en-v1.5 (feature_extraction compatible)
   • Architecture: Hybrid (Postgres + Pinecone)

🚀 Ready for Notebook 06 (LLM Clustering)!
""")
print("=" * 70)
