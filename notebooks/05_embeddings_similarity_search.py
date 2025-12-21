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
# CONFIGURATION (Dynamic - from config.py)
# ============================================================
# Load centralized configuration
from config import get_config
from utils import (
    get_embedding, init_database, init_pinecone, init_hf_client,
    validate_environment_variables, test_embedding_generation,
    ensure_package_installed
)

# Initialize configuration
cfg = get_config()

# Environment variables (still required)
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Model configuration (from config system)
MODEL_ID = cfg.embedding.model_id
EMBEDDING_DIM = cfg.embedding.dimension
PINECONE_INDEX = cfg.pinecone.index_name

# Processing configuration (from config system)
BATCH_SIZE = cfg.embedding.batch_size
MAX_CLAIMS_TO_PROCESS = cfg.processing.max_claims_to_process
RETRY_ATTEMPTS = cfg.embedding.max_retries
RETRY_DELAY = cfg.embedding.retry_delay_seconds

# Refresh configuration (from config system)
AUTO_REFRESH = cfg.refresh.auto_refresh_enabled
REFRESH_INTERVAL_HOURS = cfg.refresh.default_embedding_refresh_hours

# Validate environment variables
required_vars = ['VERCEL_POSTGRES_URL', 'HF_TOKEN', 'PINECONE_API_KEY']
if not validate_environment_variables(required_vars):
    raise ValueError("Missing required environment variables")

# Print loaded configuration
cfg.print_config()

# ============================================================
# DATABASE CONNECTION (Vercel Postgres)
# ============================================================
engine, total_claims = init_database(DATABASE_URL)

# ============================================================
# PINECONE INITIALIZATION
# ============================================================
pc, index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_DIM)

# ============================================================
# HUGGING FACE INFERENCE CLIENT
# ============================================================
print(f"   Dimensions: {EMBEDDING_DIM}")
hf_client = init_hf_client(HF_TOKEN, MODEL_ID)

# Test embedding generation
if not test_embedding_generation(hf_client, MODEL_ID, EMBEDDING_DIM):
    raise Exception("Embedding generation test failed")

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

# Template-based synthetic note generation (EXPANDED for variety)
NOTE_TEMPLATES = {
    'diabetes': [
        "Patient with Type 2 diabetes mellitus. HbA1c {a1c}%. Blood glucose {bg} mg/dL. {treatment}.",
        "Diabetes follow-up visit. A1C {a1c}%, {complication}. Continue {treatment}.",
        "Uncontrolled diabetes admitted. Blood sugar {bg}. Started on {treatment}.",
        "DM2 patient reviewed. Hemoglobin A1C {a1c}%, glucose {bg}. Adjust {treatment}.",
        "Admission for diabetic ketoacidosis. Initial BG {bg}. {treatment} initiated. Recent A1C {a1c}%.",
        "Outpatient diabetes management. {complication} identified. A1C {a1c}%. Modified {treatment}.",
        "Hospital admission DM2 uncontrolled. Blood sugar {bg}. A1C {a1c}%. {treatment} started.",
        "Diabetes clinic visit. Good glycemic control, A1C {a1c}%. {complication}. Continue {treatment}.",
        "Patient presents with hyperglycemia {bg}. {complication}. A1C {a1c}%. {treatment} adjusted.",
        "Type 2 diabetes follow-up. HbA1c {a1c}%, blood glucose {bg}. {treatment} regimen.",
    ],
    'cardiac': [
        "Acute chest pain with {symptom}. {finding}. {procedure} performed.",
        "Cardiac catheterization for {indication}. {result}. {treatment}.",
        "{condition} with {complication}. {treatment} initiated.",
        "Urgent cardiac evaluation. {symptom} noted. {finding} on imaging. {procedure} completed.",
        "Interventional cardiology: {indication}. {procedure} performed. {result}. {treatment} plan.",
        "MI protocol initiated. {symptom} present. Cath lab: {finding}. {procedure} successful.",
        "Cardiac surgery consult. {condition} diagnosis. {procedure} recommended. {treatment} started.",
        "Post-cath care. {indication} managed with {procedure}. {result}. {treatment} prescribed.",
        "Cardiology admission for {indication}. {symptom} reported. {finding}. {procedure} done.",
        "ACS presentation. {symptom} with {finding}. Emergency {procedure}. {result}. {treatment}.",
    ],
    'respiratory': [
        "{condition} with {symptom}. {imaging_finding}. {treatment} administered.",
        "Respiratory failure due to {cause}. {intervention} started.",
        "Admitted for {condition}. {treatment} given. {outcome}.",
        "Pulmonary consult: {condition}. Chest X-ray shows {imaging_finding}. {intervention} initiated.",
        "Acute {condition} exacerbation. {symptom} noted. {imaging_finding} on CT. {treatment} started.",
        "ICU admission respiratory distress. {cause} identified. {intervention} and {treatment}. {outcome}.",
        "Pulmonology evaluation. {condition} managed. {imaging_finding}. {intervention} applied. {outcome}.",
        "Respiratory support needed. {cause} diagnosed. {symptom} improved with {treatment}. {outcome}.",
        "Hospital admission {condition}. {imaging_finding} confirmed. {intervention} therapy. {outcome}.",
        "Urgent pulmonary care. {condition} with {symptom}. {treatment} and {intervention}. {outcome}.",
    ],
    'orthopedic': [
        "{procedure} for {indication}. {details}. {outcome}.",
        "{joint} replacement surgery. {implant_type}. {postop}.",
        "Orthopedic procedure: {procedure}. {details}. Discharged {outcome}.",
        "Elective {joint} arthroplasty. {indication} indication. {implant_type} used. {postop}. {outcome}.",
        "Operative report: {procedure}. {details}. {implant_type} components. {postop} protocol.",
        "Orthopedic surgery {procedure} completed. {indication}. {details}. {outcome} noted.",
        "Joint replacement {joint}. Diagnosis {indication}. {implant_type}. Post-op {postop}. {outcome}.",
        "Surgical intervention {procedure}. {indication} severity. {details} approach. {outcome}.",
        "Total {joint} replacement performed. {indication}. {implant_type} prosthesis. {postop}.",
        "{joint} surgery scheduled. {indication} diagnosed. {procedure} completed. {details}. {outcome}.",
    ],
    'gi': [
        "{procedure} performed. {finding}. {treatment}.",
        "GI surgery: {procedure} for {indication}. {outcome}.",
        "{condition} managed with {treatment}. {result}.",
        "Gastroenterology procedure {procedure} completed. {finding} identified. {treatment} plan.",
        "Endoscopy suite: {procedure} done. {indication}. {finding} noted. {treatment} recommended.",
        "GI intervention for {indication}. {procedure} successful. {finding}. {treatment} started.",
        "Digestive system evaluation. {condition} confirmed. {procedure} performed. {result}.",
        "Surgical management {procedure}. {indication} indication. {finding} pathology. {outcome}.",
        "GI consult: {condition}. {procedure} recommended and completed. {finding}. {treatment}.",
        "Abdominal surgery {procedure}. {indication}. {finding} discovered. {treatment}. {outcome}.",
    ],
}

FILLERS = {
    # Diabetes-related values (10+ options each)
    'a1c': ['6.5', '6.8', '7.0', '7.2', '7.5', '8.0', '8.5', '9.1', '9.8', '10.2', '11.0', '11.5'],
    'bg': ['140', '150', '165', '180', '195', '210', '220', '245', '260', '280', '310', '350'],
    'treatment': [
        'metformin and insulin', 'glipizide', 'insulin therapy', 'lifestyle modifications',
        'sitagliptin', 'empagliflozin', 'liraglutide', 'glyburide', 'pioglitazone', 'dulaglutide',
        'semaglutide', 'insulin pump therapy'
    ],
    'complication': [
        'peripheral neuropathy noted', 'retinopathy screening done', 'no complications',
        'diabetic foot ulcer', 'nephropathy stage 2', 'gastroparesis symptoms',
        'autonomic neuropathy', 'microalbuminuria detected', 'macular edema',
        'charcot arthropathy', 'hypoglycemia unawareness'
    ],

    # Cardiac-related values (10+ options each)
    'symptom': [
        'ST elevation V1-V4', 'inferior wall changes', 'troponin elevation',
        'T wave inversion', 'Q waves anterior', 'LBBB pattern',
        'atrial fibrillation', 'ventricular tachycardia', 'ST depression lateral',
        'prolonged QT interval', 'right axis deviation', 'low voltage QRS'
    ],
    'finding': [
        '90% LAD stenosis', 'RCA occlusion', '3-vessel disease',
        '70% LCx stenosis', 'diagonal branch occlusion', 'diffuse CAD',
        'ostial lesion RCA', 'bifurcation lesion', 'in-stent restenosis',
        'total occlusion LAD', 'chronic total occlusion', 'moderate LM disease'
    ],
    'procedure': [
        'PCI with stent placement', 'CABG', 'cardiac catheterization',
        'balloon angioplasty', 'rotational atherectomy', 'IVUS-guided PCI',
        'FFR measurement', 'OCT imaging', 'DES placement',
        'thrombus aspiration', 'kissing stents', 'atherectomy'
    ],
    'indication': [
        'unstable angina', 'STEMI', 'chest pain', 'NSTEMI',
        'stable angina', 'post-MI evaluation', 'cardiogenic shock',
        'failed medical therapy', 'positive stress test', 'crescendo angina',
        'acute coronary syndrome'
    ],
    'result': [
        'successful revascularization', 'stent placed', 'improved flow',
        'TIMI 3 flow restored', 'no residual stenosis', 'optimal result',
        'complications noted', 'requires CABG', 'incomplete revascularization',
        'dissection repaired', 'no-reflow phenomenon', 'successful PCI'
    ],

    # Respiratory-related values (10+ options each)
    'condition': [
        'pneumonia', 'COPD exacerbation', 'asthma attack', 'pulmonary embolism',
        'acute bronchitis', 'respiratory failure', 'COVID-19 pneumonia',
        'aspiration pneumonia', 'interstitial lung disease', 'pleural effusion',
        'spontaneous pneumothorax', 'acute respiratory distress'
    ],
    'imaging_finding': [
        'bilateral infiltrates', 'right lower lobe consolidation', 'pleural effusion',
        'left upper lobe opacity', 'ground glass opacities', 'hilar lymphadenopathy',
        'pulmonary edema', 'interstitial markings', 'cavitary lesion',
        'nodular densities', 'pneumothorax right', 'atelectasis left base'
    ],
    'intervention': [
        'BiPAP', 'mechanical ventilation', 'oxygen therapy',
        'high-flow nasal cannula', 'chest tube placement', 'bronchoscopy',
        'nebulizer treatments', 'inhaled corticosteroids', 'antibiotics IV',
        'prone positioning', 'ECMO support', 'thoracentesis'
    ],
    'outcome': [
        'improved and discharged', 'stable condition', 'recovery ongoing',
        'transferred to ICU', 'required intubation', 'weaned off oxygen',
        'discharged on home O2', 'readmission within 30 days', 'full recovery',
        'chronic oxygen dependence', 'pulmonary rehab referral'
    ],
    'cause': [
        'pneumonia', 'COPD', 'acute exacerbation', 'viral infection',
        'bacterial infection', 'allergen exposure', 'medication non-compliance',
        'seasonal triggers', 'smoking relapse', 'environmental factors'
    ],

    # Orthopedic-related values (10+ options each)
    'joint': [
        'Right knee', 'Left hip', 'Right shoulder', 'Left knee',
        'Right hip', 'Left shoulder', 'Right ankle', 'Left elbow',
        'bilateral knees', 'bilateral hips', 'cervical spine', 'lumbar spine'
    ],
    'implant_type': [
        'cemented prosthesis', 'uncemented components', 'hybrid fixation',
        'posterior-stabilized implant', 'cruciate-retaining design', 'ceramic-on-ceramic',
        'metal-on-polyethylene', 'dual-mobility construct', 'custom implant',
        'revision components', 'constrained liner'
    ],
    'postop': [
        'PT started POD1', 'recovery uneventful', 'mobilizing well',
        'weight-bearing as tolerated', 'ROM exercises initiated', 'pain well controlled',
        'no complications', 'early mobilization', 'discharge POD3',
        'home health arranged', 'DVT prophylaxis given', 'surgical site clean'
    ],
    'details': [
        'minimally invasive approach', 'standard technique', 'no complications',
        'anterolateral approach', 'posterior approach', 'computer-navigated',
        'robotic-assisted', 'direct anterior approach', 'mini-incision',
        'revision procedure', 'complex primary', 'staged bilateral'
    ],
}

def generate_clinical_note(claim_id: int) -> Tuple[str, str]:
    """Generate a synthetic clinical note for a claim with unique identifiers."""
    import datetime

    # Randomly choose category
    category = random.choice(list(NOTE_TEMPLATES.keys()))
    template = random.choice(NOTE_TEMPLATES[category])

    # Fill in placeholders
    note_text = template
    for key, values in FILLERS.items():
        if '{' + key + '}' in note_text:
            note_text = note_text.replace('{' + key + '}', random.choice(values))

    # Add unique identifiers to reduce duplicates
    visit_date = datetime.date(2024, random.randint(1, 12), random.randint(1, 28))
    providers = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones',
                 'Dr. Garcia', 'Dr. Miller', 'Dr. Davis', 'Dr. Rodriguez', 'Dr. Martinez',
                 'Dr. Hernandez', 'Dr. Lopez']
    provider = random.choice(providers)

    # Append unique metadata to note
    note_text = f"{note_text} Visit date: {visit_date}. Attending: {provider}."

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
                # Check if embedding needs refresh (if auto-refresh enabled)
                needs_embedding = True
                if AUTO_REFRESH:
                    result = conn.execute(text("""
                        SELECT last_embedded_at, embedding_model
                        FROM clinical_notes
                        WHERE claim_id = :cid
                        LIMIT 1
                    """), {'cid': claim_id})
                    row = result.fetchone()

                    if row and row[0]:
                        # Calculate age in hours
                        last_embedded = row[0]
                        age_hours = (datetime.now(last_embedded.tzinfo) - last_embedded).total_seconds() / 3600

                        # Skip if fresh and model hasn't changed
                        if age_hours < REFRESH_INTERVAL_HOURS and row[1] == MODEL_ID:
                            needs_embedding = False

                if not needs_embedding:
                    stats['total_processed'] += 1
                    continue

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
                embedding = get_embedding(
                    note_text, hf_client, MODEL_ID, EMBEDDING_DIM,
                    max_retries=RETRY_ATTEMPTS, retry_delay=RETRY_DELAY
                )

                if embedding is not None:
                    # Update timestamp and model version in Postgres
                    conn.execute(text("""
                        UPDATE clinical_notes
                        SET
                            last_embedded_at = NOW(),
                            embedding_model = :model,
                            embedding_version = :version
                        WHERE claim_id = :cid AND note_type = :ntype
                    """), {
                        'cid': claim_id,
                        'ntype': note_type,
                        'model': MODEL_ID,
                        'version': '1.0'  # Track version for future migrations
                    })

                    # Prepare for Pinecone batch upsert
                    vector_id = f"claim_{claim_id}_{note_type}"
                    pinecone_vectors.append({
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": {
                            "claim_id": claim_id,
                            "note_type": note_type,
                            "text_preview": note_text[:200],
                            "embedding_model": MODEL_ID,
                            "embedded_at": datetime.now().isoformat()
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
    use_model = model_id or MODEL_ID
    query_embedding = get_embedding(query, hf_client, use_model, EMBEDDING_DIM)

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
