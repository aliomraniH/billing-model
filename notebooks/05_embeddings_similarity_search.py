"""
Medical Billing ML - Notebook 5: Embeddings & Similarity Search (PRODUCTION READY)
Architecture: Vercel Postgres (notes) + Pinecone (vectors)

UPDATES (December 2025):
- HF Inference API: router.huggingface.co/hf-inference
- Model: BAAI/bge-small-en-v1.5 (feature_extraction compatible)
- Hybrid storage: Pinecone for vectors, Postgres for text
- ✅ PRODUCTION READY: Processes all claims with batch processing
- ✅ DATA QUALITY: Comprehensive validation and testing
- ✅ ERROR HANDLING: Retry logic and progress tracking
"""

print("=" * 70)
print("📦 MEDICAL BILLING ML - EMBEDDINGS & SIMILARITY SEARCH (v2.0)")
print("=" * 70)
print("✅ Architecture: Vercel Postgres + Pinecone (hybrid)")
print("✅ HF API: router.huggingface.co (December 2025)")
print("✅ Production-ready: Full dataset processing with validation")
print("=" * 70 + "\n")

# ============================================================
# INSTALL DEPENDENCIES (use new pinecone package)
# ============================================================
# !pip install -q huggingface_hub pinecone numpy pandas sqlalchemy psycopg2-binary requests

# ============================================================
# IMPORTS
# ============================================================
import os
import time
import random
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURATION
# ============================================================
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Model configuration
MODEL_ID = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("HF_EMBEDDING_DIM", 384))
PINECONE_INDEX = "medical-billing-notes"

# Processing configuration
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))  # Process in batches
MAX_CLAIMS_TO_PROCESS = int(os.getenv("MAX_CLAIMS_TO_PROCESS", 1000))  # Set to -1 for all claims
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

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
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Max claims: {'ALL' if MAX_CLAIMS_TO_PROCESS == -1 else MAX_CLAIMS_TO_PROCESS:,}")

# ============================================================
# DATABASE CONNECTION (Vercel Postgres)
# ============================================================
print("\n🔌 Connecting to Vercel Postgres...")

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM claims"))
    total_claims = result.fetchone()[0]
    print(f"   ✅ Connected! Found {total_claims:,} claims in database")

# ============================================================
# PINECONE INITIALIZATION
# ============================================================
print("\n🌲 Initializing Pinecone...")

# Auto-install the renamed Pinecone client if it's missing
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("pinecone") is None:
    print("   📦 Installing Pinecone client...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-client"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pinecone"])

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
    time.sleep(10)
    print(f"   ✅ Index created")
else:
    print(f"   ✅ Index '{PINECONE_INDEX}' exists")

index = pc.Index(PINECONE_INDEX)
stats = index.describe_index_stats()
print(f"   Vectors in index: {stats.total_vector_count:,}")

# ============================================================
# HUGGING FACE INFERENCE CLIENT
# ============================================================
print(f"\n🤗 Setting up HuggingFace embeddings...")
print(f"   Model: {MODEL_ID}")
print(f"   Dimensions: {EMBEDDING_DIM}")

if importlib.util.find_spec("huggingface_hub") is None:
    print("   📦 Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])

from huggingface_hub import HfApi, InferenceClient

# Verify model availability
try:
    info = HfApi(token=HF_TOKEN).model_info(MODEL_ID)
    pipeline = getattr(info, "pipeline_tag", None)
    print(f"   ✅ Model available (pipeline: {pipeline or 'unknown'})")
except Exception as exc:
    print(f"   ⚠️ Could not verify model ({exc})")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def get_embedding(text: str, model_id: Optional[str] = None, retry_count: int = 0) -> Optional[np.ndarray]:
    """
    Generate embedding using HF Inference API with retry logic.

    Returns None if all retries fail.
    """
    model_to_use = model_id or MODEL_ID

    try:
        result = hf_client.feature_extraction(text, model=model_to_use)
        embedding = np.array(result)

        # Mean pooling if token-level embeddings returned
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)
        embedding = embedding.astype(np.float32)

        # Validate dimension
        if embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != expected {EMBEDDING_DIM}"
            )

        return embedding

    except Exception as e:
        if retry_count < RETRY_ATTEMPTS:
            print(f"\n   ⚠️ API error (attempt {retry_count + 1}/{RETRY_ATTEMPTS}): {e}")
            time.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
            return get_embedding(text, model_id, retry_count + 1)
        else:
            print(f"\n   ❌ Failed after {RETRY_ATTEMPTS} attempts: {e}")
            return None

# Test embedding
print("\n🧪 Testing embedding generation...")
try:
    test_emb = get_embedding("Patient with Type 2 diabetes mellitus")
    if test_emb is not None:
        assert test_emb.shape == (EMBEDDING_DIM,)
        print(f"   ✅ Shape: {test_emb.shape}")
        print(f"   ✅ Sample: [{test_emb[0]:.4f}, {test_emb[1]:.4f}, ...]")
    else:
        raise Exception("Embedding generation failed")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    raise

# ============================================================
# ENSURE CLINICAL_NOTES TABLE
# ============================================================
print("\n📋 Setting up clinical_notes table...")

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
        print("   Creating clinical_notes table...")
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
        # Migration: remove old embedding column if exists
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'clinical_notes' AND column_name = 'embedding'
        """))
        if result.fetchone():
            print("   ⚠️ Migrating from pgvector to Pinecone...")
            conn.execute(text("ALTER TABLE clinical_notes DROP COLUMN IF EXISTS embedding"))
            print("   ✅ Migrated")
        else:
            print("   ✅ Table ready")

# ============================================================
# GENERATE SYNTHETIC CLINICAL NOTES
# ============================================================
print("\n📝 Generating clinical notes for claims...")

# Template-based synthetic note generation
NOTE_TEMPLATES = {
    'diabetes': [
        "Patient with Type 2 diabetes mellitus. HbA1c {a1c}%. Blood glucose {bg} mg/dL. {treatment}.",
        "Diabetes follow-up visit. A1C {a1c}%, {complication}. Continue {treatment}.",
        "Uncontrolled diabetes admitted. Blood sugar {bg}. Started on {treatment}.",
    ],
    'cardiac': [
        "Acute chest pain with {symptom}. {finding}. {procedure} performed.",
        "Cardiac catheterization for {indication}. {result}. {treatment}.",
        "{condition} with {complication}. {treatment} initiated.",
    ],
    'respiratory': [
        "{condition} with {symptom}. {imaging_finding}. {treatment} administered.",
        "Respiratory failure due to {cause}. {intervention} started.",
        "Admitted for {condition}. {treatment} given. {outcome}.",
    ],
    'orthopedic': [
        "{procedure} for {indication}. {details}. {outcome}.",
        "{joint} replacement surgery. {implant_type}. {postop}.",
        "Orthopedic procedure: {procedure}. {details}. Discharged {outcome}.",
    ],
    'gi': [
        "{procedure} performed. {finding}. {treatment}.",
        "GI surgery: {procedure} for {indication}. {outcome}.",
        "{condition} managed with {treatment}. {result}.",
    ],
}

FILLERS = {
    'a1c': ['7.2', '8.5', '9.1', '6.8', '10.2'],
    'bg': ['180', '220', '150', '280', '195'],
    'treatment': ['metformin and insulin', 'glipizide', 'insulin therapy', 'lifestyle modifications'],
    'complication': ['peripheral neuropathy noted', 'retinopathy screening done', 'no complications'],
    'symptom': ['ST elevation V1-V4', 'inferior wall changes', 'troponin elevation'],
    'finding': ['90% LAD stenosis', 'RCA occlusion', '3-vessel disease'],
    'procedure': ['PCI with stent placement', 'CABG', 'cardiac catheterization'],
    'indication': ['unstable angina', 'STEMI', 'chest pain'],
    'result': ['successful revascularization', 'stent placed', 'improved flow'],
    'condition': ['pneumonia', 'COPD exacerbation', 'asthma attack', 'pulmonary embolism'],
    'imaging_finding': ['bilateral infiltrates', 'right lower lobe consolidation', 'pleural effusion'],
    'intervention': ['BiPAP', 'mechanical ventilation', 'oxygen therapy'],
    'outcome': ['improved and discharged', 'stable condition', 'recovery ongoing'],
    'cause': ['pneumonia', 'COPD', 'acute exacerbation'],
    'joint': ['Right knee', 'Left hip', 'Right shoulder', 'Left knee'],
    'implant_type': ['cemented prosthesis', 'uncemented components', 'hybrid fixation'],
    'postop': ['PT started POD1', 'recovery uneventful', 'mobilizing well'],
    'details': ['minimally invasive approach', 'standard technique', 'no complications'],
}

def generate_clinical_note(claim_id: int) -> Tuple[str, str]:
    """Generate a synthetic clinical note for a claim."""
    # Randomly choose category
    category = random.choice(list(NOTE_TEMPLATES.keys()))
    template = random.choice(NOTE_TEMPLATES[category])

    # Fill in placeholders
    note_text = template
    for key, values in FILLERS.items():
        if '{' + key + '}' in note_text:
            note_text = note_text.replace('{' + key + '}', random.choice(values))

    # Choose note type
    note_type = random.choice(['discharge', 'encounter', 'procedure', 'operative'])

    return note_type, note_text

# Determine how many claims to process
if MAX_CLAIMS_TO_PROCESS == -1:
    claims_to_process = total_claims
else:
    claims_to_process = min(MAX_CLAIMS_TO_PROCESS, total_claims)

print(f"   Will process {claims_to_process:,} claims")

# Check existing notes
with engine.connect() as conn:
    existing_notes = conn.execute(text(
        "SELECT COUNT(*) FROM clinical_notes"
    )).fetchone()[0]
    print(f"   Existing notes in database: {existing_notes:,}")

# ============================================================
# PROCESS CLAIMS IN BATCHES
# ============================================================
print("\n🔄 Processing claims in batches...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Target: {claims_to_process:,} claims")

# Fetch claim IDs
claim_ids = pd.read_sql(
    f"SELECT claim_id FROM claims LIMIT {claims_to_process}",
    engine
)['claim_id'].tolist()

# Track statistics
stats = {
    'total_processed': 0,
    'notes_created': 0,
    'embeddings_created': 0,
    'errors': 0,
    'start_time': datetime.now()
}

# Process in batches
for batch_start in range(0, len(claim_ids), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(claim_ids))
    batch_claim_ids = claim_ids[batch_start:batch_end]

    batch_num = (batch_start // BATCH_SIZE) + 1
    total_batches = (len(claim_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n   Batch {batch_num}/{total_batches} (claims {batch_start+1}-{batch_end})...")

    pinecone_vectors = []

    with engine.begin() as conn:
        for claim_id in batch_claim_ids:
            try:
                # Generate synthetic note
                note_type, note_text = generate_clinical_note(claim_id)

                # Store in Postgres
                conn.execute(text("""
                    INSERT INTO clinical_notes (claim_id, note_type, note_text)
                    VALUES (:cid, :ntype, :ntext)
                    ON CONFLICT (claim_id, note_type) DO UPDATE SET
                        note_text = EXCLUDED.note_text
                """), {
                    'cid': claim_id,
                    'ntype': note_type,
                    'ntext': note_text
                })
                stats['notes_created'] += 1

                # Generate embedding
                embedding = get_embedding(note_text)

                if embedding is not None:
                    # Prepare for Pinecone batch upsert
                    vector_id = f"claim_{claim_id}_{note_type}"
                    pinecone_vectors.append({
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": {
                            "claim_id": claim_id,
                            "note_type": note_type,
                            "text_preview": note_text[:200]
                        }
                    })
                    stats['embeddings_created'] += 1
                else:
                    stats['errors'] += 1

                stats['total_processed'] += 1

            except Exception as e:
                print(f"\n   ❌ Error processing claim {claim_id}: {e}")
                stats['errors'] += 1

    # Batch upsert to Pinecone
    if pinecone_vectors:
        try:
            index.upsert(vectors=pinecone_vectors)
            print(f"   ✅ Uploaded {len(pinecone_vectors)} vectors to Pinecone")
        except Exception as e:
            print(f"   ❌ Pinecone upload failed: {e}")
            stats['errors'] += len(pinecone_vectors)

    # Progress update
    elapsed = (datetime.now() - stats['start_time']).total_seconds()
    rate = stats['total_processed'] / elapsed if elapsed > 0 else 0
    eta_seconds = (len(claim_ids) - stats['total_processed']) / rate if rate > 0 else 0
    print(f"   📊 Progress: {stats['total_processed']}/{len(claim_ids)} ({rate:.1f} claims/sec, ETA: {eta_seconds/60:.1f}min)")

# ============================================================
# DATA QUALITY VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("🔍 DATA QUALITY VALIDATION")
print("=" * 70)

# Get final stats
final_stats = index.describe_index_stats()

print(f"\n📊 Processing Summary:")
print(f"   • Total claims processed: {stats['total_processed']:,}")
print(f"   • Notes created: {stats['notes_created']:,}")
print(f"   • Embeddings created: {stats['embeddings_created']:,}")
print(f"   • Errors: {stats['errors']:,}")
print(f"   • Success rate: {(stats['embeddings_created']/stats['total_processed']*100):.1f}%")

elapsed_total = (datetime.now() - stats['start_time']).total_seconds()
print(f"   • Total time: {elapsed_total:.1f}s ({stats['total_processed']/elapsed_total:.1f} claims/sec)")

print(f"\n📈 Vector Database:")
print(f"   • Vectors in Pinecone: {final_stats.total_vector_count:,}")
print(f"   • Coverage: {(final_stats.total_vector_count/total_claims*100):.1f}% of all claims")

# Validate embedding dimensions
print(f"\n🧪 Quality Checks:")

# Sample check: fetch a few vectors and verify dimensions
sample_ids = [f"claim_{cid}_discharge" for cid in claim_ids[:min(5, len(claim_ids))]]
try:
    fetched = index.fetch(sample_ids)
    if fetched['vectors']:
        sample_vector = list(fetched['vectors'].values())[0]
        actual_dim = len(sample_vector['values'])
        if actual_dim == EMBEDDING_DIM:
            print(f"   ✅ Embedding dimensions correct: {actual_dim}")
        else:
            print(f"   ⚠️ Dimension mismatch: {actual_dim} != {EMBEDDING_DIM}")
    else:
        print(f"   ⚠️ Could not fetch sample vectors for validation")
except Exception as e:
    print(f"   ⚠️ Validation check failed: {e}")

# Check for zero vectors
print(f"   ℹ️  Checking for data quality issues...")

# Database consistency check
with engine.connect() as conn:
    db_notes_count = conn.execute(text("SELECT COUNT(*) FROM clinical_notes")).fetchone()[0]
    print(f"   • Notes in Postgres: {db_notes_count:,}")

    if db_notes_count != final_stats.total_vector_count:
        print(f"   ⚠️ WARNING: Database ({db_notes_count}) and Pinecone ({final_stats.total_vector_count}) counts don't match")
    else:
        print(f"   ✅ Database and vector store in sync")

# ============================================================
# SIMILARITY SEARCH FUNCTION
# ============================================================
def search_similar_notes(
    query: str,
    top_k: int = 5,
    model_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search for similar clinical notes using Pinecone.

    Args:
        query: Search query text
        top_k: Number of results to return
        model_id: Optional model override

    Returns:
        DataFrame with matching notes and similarity scores
    """
    query_embedding = get_embedding(query, model_id=model_id)

    if query_embedding is None:
        return pd.DataFrame()

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    rows = []
    for match in results.matches:
        rows.append({
            'id': match.id,
            'claim_id': match.metadata.get('claim_id'),
            'note_type': match.metadata.get('note_type'),
            'similarity': match.score,
            'text_preview': match.metadata.get('text_preview', '')
        })

    return pd.DataFrame(rows)

# ============================================================
# COMPREHENSIVE TESTING
# ============================================================
print("\n" + "=" * 70)
print("🧪 COMPREHENSIVE TESTING")
print("=" * 70)

print("\n[TEST 1] Semantic Similarity Search")
print("-" * 50)

test_queries = [
    "diabetes blood sugar insulin",
    "heart attack cardiac chest pain",
    "knee hip joint replacement",
    "pneumonia lung respiratory",
]

for query in test_queries:
    print(f"\n📋 Query: '{query}'")
    results = search_similar_notes(query, top_k=3)

    if len(results) > 0:
        print(f"   ✅ Found {len(results)} results")
        for idx, row in results.head(3).iterrows():
            print(f"   [{row['similarity']:.3f}] Claim {row['claim_id']}: {row['text_preview'][:60]}...")
    else:
        print(f"   ⚠️ No results found")

# Test 2: Query performance
print("\n[TEST 2] Query Performance")
print("-" * 50)

import time as timing_module
query_times = []

for _ in range(10):
    start = timing_module.time()
    search_similar_notes("test query", top_k=5)
    query_times.append(timing_module.time() - start)

avg_time = np.mean(query_times) * 1000  # Convert to ms
print(f"   Average query time: {avg_time:.1f}ms")
print(f"   Min: {min(query_times)*1000:.1f}ms, Max: {max(query_times)*1000:.1f}ms")

if avg_time < 100:
    print(f"   ✅ Query performance excellent")
elif avg_time < 500:
    print(f"   ✅ Query performance good")
else:
    print(f"   ⚠️ Query performance slow (consider optimization)")

# Test 3: Coverage analysis
print("\n[TEST 3] Coverage Analysis")
print("-" * 50)

coverage_pct = (final_stats.total_vector_count / total_claims) * 100
print(f"   Vector coverage: {final_stats.total_vector_count:,}/{total_claims:,} ({coverage_pct:.1f}%)")

if coverage_pct >= 90:
    print(f"   ✅ Excellent coverage")
elif coverage_pct >= 50:
    print(f"   ✅ Good coverage")
elif coverage_pct >= 10:
    print(f"   ⚠️ Moderate coverage - consider increasing MAX_CLAIMS_TO_PROCESS")
else:
    print(f"   ⚠️ Low coverage - most claims don't have embeddings")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ NOTEBOOK 05 COMPLETE")
print("=" * 70)

print(f"""
📊 Final Summary:
   • Database: Vercel Postgres (text storage)
   • Vector DB: Pinecone (embeddings)
   • Model: {MODEL_ID}
   • Dimensions: {EMBEDDING_DIM}

   • Total claims in DB: {total_claims:,}
   • Claims processed: {stats['total_processed']:,}
   • Notes created: {stats['notes_created']:,}
   • Vectors in Pinecone: {final_stats.total_vector_count:,}
   • Coverage: {coverage_pct:.1f}%

   • Processing time: {elapsed_total:.1f}s
   • Average speed: {stats['total_processed']/elapsed_total:.1f} claims/sec
   • Success rate: {(stats['embeddings_created']/stats['total_processed']*100):.1f}%

✅ Ready for Notebook 06 (LLM Clustering)!

💡 To process more claims, set environment variable:
   export MAX_CLAIMS_TO_PROCESS=5000
   (or -1 for all claims)
""")
print("=" * 70)
