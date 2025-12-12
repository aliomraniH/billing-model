# Medical Billing ML - Notebook 5: Embeddings & Similarity Search (API Version)
# Prerequisites: Run notebooks 01-02 first
# This version uses HuggingFace Inference API - no heavy ML dependencies needed!

# Install only lightweight dependencies
get_ipython().system('pip install -q sqlalchemy psycopg2-binary pandas numpy requests')

import os
import requests
import json
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Configuration
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')  # Optional - works without it but with rate limits

if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

# HuggingFace Inference API endpoint (for embeddings, not chat)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"

# Setup headers for API
headers = {}
if HF_API_KEY:
    headers["Authorization"] = f"Bearer {HF_API_KEY}"
    print("✅ Using HuggingFace API key")
else:
    print("ℹ️  No HuggingFace API key - using rate-limited free tier")
    print("   To add key: Project Settings → Environment Variables → HUGGINGFACE_API_KEY")

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

# Embedding function using HuggingFace API
def get_embedding(text: str, retries: int = 3) -> list:
    """
    Get embedding for text using HuggingFace Inference API
    Returns 384-dimensional vector for all-MiniLM-L6-v2
    """
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                embedding = response.json()
                # API returns nested list, flatten it
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        return embedding[0]
                    return embedding
            elif response.status_code == 503:
                # Model is loading, wait and retry
                print(f"⏳ Model loading... (attempt {attempt + 1}/{retries})")
                import time
                time.sleep(5)
                continue
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                raise Exception(f"API request failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                import time
                time.sleep(2)
            else:
                raise

    raise Exception("Failed to get embedding after all retries")

# Test the API
print("\n📥 Testing HuggingFace Inference API...")
test_embedding = get_embedding("This is a test sentence")
print(f"✅ API working! Embedding dimension: {len(test_embedding)}")

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
print("\n🔄 Generating embeddings via API and storing in database...")
print("   (This may take ~30 seconds due to API rate limits)")

with engine.begin() as conn:
    for i, note in enumerate(sample_notes):
        if i < len(claim_ids):
            # Generate embedding via API
            print(f"   Processing note {i+1}/{len(sample_notes)}...", end=" ")
            embedding = get_embedding(note['note_text'])
            emb_str = '[' + ','.join(map(str, embedding)) + ']'

            # Insert into database
            conn.execute(text("""
                INSERT INTO clinical_notes (claim_id, note_type, note_text, embedding)
                VALUES (:cid, :ntype, :ntext, :emb::vector)
                ON CONFLICT (claim_id, note_type)
                DO UPDATE SET note_text = :ntext, embedding = :emb::vector
            """), {
                'cid': claim_ids[i],
                'ntype': note['note_type'],
                'ntext': note['note_text'],
                'emb': emb_str
            })
            print("✓")

            # Small delay to avoid rate limiting
            if i < len(sample_notes) - 1:
                import time
                time.sleep(1)

print(f"✅ Stored {len(sample_notes)} clinical notes with embeddings!")

# Similarity Search Function
def search_similar_notes(query: str, top_k: int = 5):
    """Search for clinically similar notes using semantic similarity"""
    # Generate query embedding via API
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

def validate_billing_code(claim_id: int, icd_code: str):
    """
    Validate if a billing code matches the clinical documentation
    Returns similarity score and validation result
    """
    # Get clinical notes for this claim
    notes = pd.read_sql(text("""
        SELECT note_text
        FROM clinical_notes
        WHERE claim_id = :cid AND embedding IS NOT NULL
    """), engine, params={'cid': claim_id})

    if len(notes) == 0:
        return {
            'valid': None,
            'reason': 'No clinical notes found',
            'similarity': 0.0
        }

    # Get code description
    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = get_embedding(code_desc)

    # Calculate max similarity across all notes
    max_sim = 0
    for _, row in notes.iterrows():
        note_emb = get_embedding(row['note_text'])
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
print("\n💡 TIP: For faster performance, get a free HuggingFace API key:")
print("   1. Sign up at https://huggingface.co")
print("   2. Go to Settings → Access Tokens")
print("   3. Create a new token")
print("   4. Add to Deepnote: Settings → Environment Variables → HUGGINGFACE_API_KEY")
