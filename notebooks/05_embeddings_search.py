# Medical Billing ML - Notebook 5: Embeddings & Similarity Search
# Prerequisites:
# 1. Run notebooks 01-02 first to set up database and load data
# 2. Migration runs automatically (creates separate embeddings table)
# Note: This requires ~2GB of dependencies. For lighter option, use 05_embeddings_search_api.py

# Install only what we need for this notebook
# Using compatible versions to avoid dependency conflicts
get_ipython().system('pip install -q "torch>=2.0.0,<2.2.0" "transformers>=4.35.0,<5.0.0" "sentence-transformers>=2.2.0" sqlalchemy psycopg2-binary pandas numpy')

# Connect & Load Embedding Model
import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM claims"))
        total_claims = result.fetchone()[0]
        print(f"✅ Connected! Found {total_claims:,} claims in database")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# Run migration if needed (creates separate embeddings table)
print("\n🔧 Checking database schema...")
try:
    with engine.begin() as conn:
        # Check if migration is needed
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'clinical_note_embeddings'
            )
        """))
        migration_exists = result.fetchone()[0]

        if not migration_exists:
            print("⚠️  Running migration to create embeddings table...")
            # Read and execute migration
            migration_path = os.path.join(os.path.dirname(__file__), '..', 'sql', 'migrations', '001_separate_embeddings_table.sql')
            with open(migration_path, 'r') as f:
                migration_sql = f.read()
            conn.execute(text(migration_sql))
            print("✅ Migration completed successfully!")
        else:
            print("✅ Schema up to date - embeddings table exists")
except Exception as e:
    print(f"ℹ️  Note: If migration fails, run manually:")
    print(f"   psql $DATABASE_URL -f sql/migrations/001_separate_embeddings_table.sql")
    # Don't raise - migration might already be applied
    pass

# Load embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print("\n📥 Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = embed_model.get_sentence_embedding_dimension()
print(f"✅ Model loaded successfully")
print(f"   Model: {EMBEDDING_MODEL}")
print(f"   Embedding dimensions: {embedding_dim}")

# Create Sample Clinical Notes
print("\n📝 Creating sample clinical notes...")
sample_notes = [
    {
        "note_type": "discharge",
        "note_text": "Patient admitted with uncontrolled Type 2 diabetes mellitus. Blood glucose stabilized with insulin. HbA1c 9.2%. Discharged on adjusted metformin."
    },
    {
        "note_type": "discharge",
        "note_text": "Acute chest pain with ST elevation in V1-V4. Emergent PCI with drug-eluting stent to LAD. Post-MI protocol initiated."
    },
    {
        "note_type": "discharge",
        "note_text": "Community-acquired pneumonia with right lower lobe consolidation. IV antibiotics transitioned to oral azithromycin. Oxygen resolved day 3."
    },
    {
        "note_type": "operative",
        "note_text": "Right total knee arthroplasty for end-stage osteoarthritis. Cemented components. EBL 150mL. Tolerated well."
    },
    {
        "note_type": "procedure",
        "note_text": "Colonoscopy for CRC screening. Two 5mm tubular adenomas removed from ascending colon. No malignancy. Repeat in 5 years."
    },
]

# Get claim IDs to link notes to
claim_ids = pd.read_sql("SELECT claim_id FROM claims LIMIT 5", engine)['claim_id'].tolist()
print(f"   Using {len(claim_ids)} claim IDs")

# Generate embeddings and store
print("\n🔄 Generating embeddings and storing in database...")
with engine.begin() as conn:
    for i, note in enumerate(sample_notes):
        if i < len(claim_ids):
            # Generate embedding
            embedding = embed_model.encode([note['note_text']])[0]
            emb_str = '[' + ','.join(map(str, embedding)) + ']'

            # Step 1: Insert/update clinical note
            result = conn.execute(text("""
                INSERT INTO clinical_notes (claim_id, note_type, note_text)
                VALUES (:cid, :ntype, :ntext)
                ON CONFLICT (claim_id, note_type)
                DO UPDATE SET note_text = :ntext
                RETURNING note_id
            """), {
                'cid': claim_ids[i],
                'ntype': note['note_type'],
                'ntext': note['note_text']
            })
            note_id = result.fetchone()[0]

            # Step 2: Insert/update embedding in separate table
            conn.execute(text("""
                INSERT INTO clinical_note_embeddings (note_id, model_name, embedding)
                VALUES (:nid, :model, CAST(:emb AS vector))
                ON CONFLICT (note_id, model_name)
                DO UPDATE SET
                    embedding = CAST(:emb AS vector),
                    updated_at = NOW()
            """), {
                'nid': note_id,
                'model': EMBEDDING_MODEL,
                'emb': emb_str
            })
    # Auto-commits on context exit

print(f"✅ Stored {len(sample_notes)} clinical notes with embeddings!")

# Similarity Search Function
def search_similar_notes(query: str, top_k: int = 5, model: str = None):
    """Search for clinically similar notes using semantic similarity"""
    # Use default model if not specified
    if model is None:
        model = EMBEDDING_MODEL

    # Generate query embedding
    query_embedding = embed_model.encode([query])[0]
    emb_str = '[' + ','.join(map(str, query_embedding)) + ']'

    # Search using pgvector cosine distance with JOIN to embeddings table
    results = pd.read_sql(text("""
        SELECT
            cn.note_id,
            cn.claim_id,
            cn.note_type,
            cn.note_text,
            cne.model_name,
            1 - (cne.embedding <=> CAST(:emb AS vector)) AS similarity
        FROM clinical_notes cn
        INNER JOIN clinical_note_embeddings cne ON cn.note_id = cne.note_id
        WHERE cne.model_name = :model
        ORDER BY cne.embedding <=> CAST(:emb AS vector)
        LIMIT :k
    """), engine, params={'emb': emb_str, 'model': model, 'k': top_k})

    return results

# Test Searches
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

print("\n" + "="*70)

# Billing Code Validation Function
CODE_DESCRIPTIONS = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'I21.0': 'ST elevation myocardial infarction anterior wall',
    'J18.9': 'Pneumonia unspecified organism',
    'M17.11': 'Primary osteoarthritis right knee',
}

def validate_billing_code(claim_id: int, icd_code: str, model: str = None):
    """
    Validate if a billing code matches the clinical documentation
    Returns similarity score and validation result
    """
    # Use default model if not specified
    if model is None:
        model = EMBEDDING_MODEL

    # Get clinical notes with embeddings for this claim
    notes = pd.read_sql(text("""
        SELECT cn.note_text, cne.note_id
        FROM clinical_notes cn
        INNER JOIN clinical_note_embeddings cne ON cn.note_id = cne.note_id
        WHERE cn.claim_id = :cid AND cne.model_name = :model
    """), engine, params={'cid': claim_id, 'model': model})

    if len(notes) == 0:
        return {
            'valid': None,
            'reason': 'No clinical notes found',
            'similarity': 0.0
        }

    # Get code description
    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = embed_model.encode([code_desc])[0]

    # Calculate max similarity across all notes
    max_sim = 0
    for _, row in notes.iterrows():
        note_emb = embed_model.encode([row['note_text']])[0]
        sim = np.dot(code_emb, note_emb) / (np.linalg.norm(code_emb) * np.linalg.norm(note_emb))
        max_sim = max(max_sim, sim)

    return {
        'code': icd_code,
        'description': code_desc,
        'similarity': float(max_sim),
        'valid': max_sim >= 0.4  # Threshold for valid billing
    }

# Test Billing Code Validation
print("\n" + "="*70)
print("🏥 BILLING CODE VALIDATION")
print("="*70)

for claim_id in claim_ids[:3]:
    result = validate_billing_code(claim_id, 'E11.9')
    status = '✅ VALID' if result['valid'] else '❌ MISMATCH'
    print(f"Claim {claim_id:5d} | Code: E11.9 | Similarity: {result['similarity']:.3f} | {status}")

print("="*70)
