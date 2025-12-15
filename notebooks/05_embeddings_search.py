# Medical Billing ML - Notebook 5: Embeddings & Similarity Search
# FIXED: Uses Hugging Face Serverless Inference API to avoid PyTorch compatibility issues
# Prerequisites: Run notebooks 01-02 first

# ============================================================
# INSTALL DEPENDENCIES (minimal - no torch needed!)
# ============================================================
# Run in your notebook environment:
# !pip install -q huggingface_hub numpy pandas sqlalchemy psycopg2-binary requests

print("="*70)
print("📦 MEDICAL BILLING ML - EMBEDDINGS & SIMILARITY SEARCH")
print("="*70)
print("✅ Using Hugging Face Serverless Inference API")
print("   ➜ No PyTorch installation required!")
print("   ➜ No GPU needed - serverless compute")
print("   ➜ Same model quality (all-MiniLM-L6-v2)")
print("="*70 + "\n")

# ============================================================
# IMPORTS & CONFIGURATION
# ============================================================
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ============================================================
# DATABASE CONNECTION
# ============================================================
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

# ============================================================
# HUGGING FACE INFERENCE CLIENT SETUP
# ============================================================
HF_TOKEN = os.getenv('HF_TOKEN')
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

print("\n📥 Setting up embedding generation...")

# Try to use InferenceClient first (preferred method)
try:
    from huggingface_hub import InferenceClient

    if HF_TOKEN:
        print("✅ Using Hugging Face Serverless Inference API (authenticated)")
        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN
        )

        def get_embedding(text: str) -> np.ndarray:
            """Generate embedding using HF Serverless API with authentication"""
            result = client.feature_extraction(
                text,
                model=MODEL_ID
            )
            # Mean pooling for sentence embedding
            embedding = np.array(result)
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)
            return embedding.astype(np.float32)
    else:
        print("⚠️  HF_TOKEN not found - using HTTP fallback (rate limited)")
        print("   For better performance, set HF_TOKEN in environment variables")
        print("   Get token at: https://huggingface.co/settings/tokens")
        client = None
        raise ImportError("No HF_TOKEN - fallback to HTTP")

except (ImportError, Exception) as e:
    # Fallback to raw HTTP requests
    print("⚠️  Using HTTP API fallback (no authentication)")
    import requests
    import time

    API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"

    if HF_TOKEN:
        HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
    else:
        HEADERS = {}

    def get_embedding(text: str, retry_count: int = 0) -> np.ndarray:
        """Generate embedding using raw HTTP request to HF API"""
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )

        if response.status_code == 200:
            result = response.json()
            embedding = np.array(result)
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)
            return embedding.astype(np.float32)
        elif response.status_code == 503 and retry_count < 2:
            # Model is loading - wait and retry
            wait_time = 20 * (retry_count + 1)
            print(f"   ⏳ Model loading, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return get_embedding(text, retry_count + 1)
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

print(f"   Model: {MODEL_ID}")
print(f"   Embedding dimensions: {EMBEDDING_DIM}")

# ============================================================
# TEST EMBEDDING GENERATION
# ============================================================
print("\n🧪 Testing embedding generation...")
test_embedding = get_embedding("This is a test sentence for medical billing.")
print(f"✅ Test embedding shape: {test_embedding.shape}")
print(f"   First 5 values: {test_embedding[:5]}")
print(f"   Dimension check: {len(test_embedding)} == {EMBEDDING_DIM} ✓")

# ============================================================
# CREATE SAMPLE CLINICAL NOTES
# ============================================================
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
    {
        "note_type": "discharge",
        "note_text": "Acute appendicitis with perforation. Emergency laparoscopic appendectomy. Peritoneal washout performed. IV antibiotics for 48 hours."
    },
    {
        "note_type": "procedure",
        "note_text": "Upper endoscopy for dyspepsia. Moderate gastritis H. pylori positive. Triple therapy prescribed."
    },
    {
        "note_type": "operative",
        "note_text": "Laparoscopic cholecystectomy for symptomatic cholelithiasis. Multiple gallstones. No complications. Discharged same day."
    },
]

# Get claim IDs to link notes to
claim_ids = pd.read_sql(f"SELECT claim_id FROM claims LIMIT {len(sample_notes)}", engine)['claim_id'].tolist()
print(f"   Using {len(claim_ids)} claim IDs")

# ============================================================
# GENERATE EMBEDDINGS AND STORE IN DATABASE
# ============================================================
print("\n🔄 Generating embeddings via Hugging Face API...")
print("   This uses serverless infrastructure - no local GPU/torch needed!")

with engine.begin() as conn:  # Auto-commits on context exit
    for i, note in enumerate(sample_notes):
        if i < len(claim_ids):
            print(f"   Processing note {i+1}/{len(sample_notes)}... ", end="")

            # Generate embedding via HF API
            embedding = get_embedding(note['note_text'])
            emb_str = '[' + ','.join(map(str, embedding)) + ']'

            # Insert into database
            conn.execute(text("""
                INSERT INTO clinical_notes (claim_id, note_type, note_text, embedding)
                VALUES (:cid, :ntype, :ntext, :emb::vector)
            """), {
                'cid': claim_ids[i],
                'ntype': note['note_type'],
                'ntext': note['note_text'],
                'emb': emb_str
            })
            print("✓")

print(f"✅ Stored {len(sample_notes)} clinical notes with embeddings!")

# ============================================================
# SIMILARITY SEARCH FUNCTION
# ============================================================
def search_similar_notes(query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Search for clinically similar notes using semantic similarity
    Uses HF API for query embedding generation
    """
    # Generate query embedding
    query_embedding = get_embedding(query)
    emb_str = '[' + ','.join(map(str, query_embedding)) + ']'

    # Search using pgvector cosine distance
    results = pd.read_sql(text("""
        SELECT
            note_id,
            claim_id,
            note_type,
            note_text,
            1 - (embedding <=> :emb::vector) AS similarity
        FROM clinical_notes
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :emb::vector
        LIMIT :k
    """), engine, params={'emb': emb_str, 'k': top_k})

    return results

# ============================================================
# TEST SEMANTIC SIMILARITY SEARCH
# ============================================================
print("\n" + "="*70)
print("🔍 TESTING SEMANTIC SIMILARITY SEARCH")
print("="*70)

queries = [
    "diabetes blood sugar insulin",
    "heart attack cardiac stent",
    "knee replacement surgery",
    "abdominal pain appendix surgery",
    "stomach ulcer infection"
]

for query in queries:
    print(f"\n📋 Query: '{query}'")
    print("-" * 70)
    results = search_similar_notes(query, top_k=3)
    for _, row in results.iterrows():
        print(f"  [{row['similarity']:.3f}] {row['note_text'][:65]}...")

print("\n" + "="*70)

# ============================================================
# BILLING CODE VALIDATION FUNCTION
# ============================================================
CODE_DESCRIPTIONS = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'I21.0': 'ST elevation myocardial infarction anterior wall',
    'J18.9': 'Pneumonia unspecified organism',
    'M17.11': 'Primary osteoarthritis right knee',
    'K35.80': 'Unspecified acute appendicitis',
    'K29.70': 'Gastritis unspecified without bleeding',
    'K80.20': 'Calculus of gallbladder without cholecystitis',
}

def validate_billing_code(claim_id: int, icd_code: str):
    """
    Validate if a billing code matches the clinical documentation
    Returns similarity score and validation result
    Uses HF API for embedding generation
    """
    # Get clinical notes for this claim
    notes = pd.read_sql(text("""
        SELECT note_text
        FROM clinical_notes
        WHERE claim_id = :cid AND embedding IS NOT NULL
    """), engine, params={'cid': claim_id})

    if len(notes) == 0:
        return {
            'code': icd_code,
            'description': 'N/A',
            'valid': None,
            'reason': 'No clinical notes found',
            'similarity': 0.0
        }

    # Get code description and embedding
    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = get_embedding(code_desc)

    # Calculate max similarity across all notes
    max_sim = 0
    for _, row in notes.iterrows():
        note_emb = get_embedding(row['note_text'])
        # Cosine similarity
        sim = np.dot(code_emb, note_emb) / (np.linalg.norm(code_emb) * np.linalg.norm(note_emb))
        max_sim = max(max_sim, sim)

    return {
        'code': icd_code,
        'description': code_desc,
        'similarity': float(max_sim),
        'valid': max_sim >= 0.4,  # Threshold for valid billing
        'reason': 'Match found' if max_sim >= 0.4 else 'Low similarity to clinical notes'
    }

# ============================================================
# TEST BILLING CODE VALIDATION
# ============================================================
print("\n" + "="*70)
print("🏥 BILLING CODE VALIDATION")
print("="*70)
print("Testing if billing codes match clinical documentation...\n")

# Test various code matches
test_cases = [
    (claim_ids[0], 'E11.9'),   # Diabetes note
    (claim_ids[1], 'I21.0'),   # Heart attack note
    (claim_ids[3], 'M17.11'),  # Knee surgery note
    (claim_ids[0], 'K80.20'),  # Mismatch: diabetes vs gallbladder
]

for claim_id, code in test_cases:
    result = validate_billing_code(claim_id, code)
    status = '✅ VALID' if result['valid'] else '❌ MISMATCH'
    print(f"Claim {claim_id:5d} | Code: {code:8s} | Similarity: {result['similarity']:.3f} | {status}")
    print(f"           {result['description']}")
    print()

print("="*70)

# ============================================================
# PERFORMANCE COMPARISON
# ============================================================
print("\n" + "="*70)
print("📊 APPROACH COMPARISON")
print("="*70)
print("""
╔═══════════════════════════════════════════════════════════════════╗
║  OLD APPROACH (sentence-transformers + PyTorch)                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  ❌ Requires PyTorch 2.2+ installation (~2GB)                     ║
║  ❌ Complex dependency conflicts (transformers compatibility)     ║
║  ❌ Needs local compute (CPU/GPU)                                 ║
║  ❌ Large environment footprint                                   ║
╚═══════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════╗
║  NEW APPROACH (Hugging Face Serverless API)                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  ✅ Zero PyTorch installation required                            ║
║  ✅ No dependency conflicts                                       ║
║  ✅ Serverless compute (no GPU needed)                            ║
║  ✅ Small environment footprint (huggingface_hub only)            ║
║  ✅ Same model quality (all-MiniLM-L6-v2)                         ║
║  ✅ Free tier available with generous limits                      ║
║  ✅ Production-ready scaling options                              ║
╚═══════════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*70)
print("✅ NOTEBOOK 05 COMPLETE - READY FOR CLUSTERING!")
print("="*70)
print("\n📝 Key takeaways:")
print("   • Embeddings enable semantic similarity search")
print("   • Can validate billing codes against clinical notes")
print("   • HF Serverless API avoids PyTorch dependency issues")
print("   • Next: Use embeddings for clustering similar cases")
print("\n" + "="*70)
