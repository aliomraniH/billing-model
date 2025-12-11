# Medical Billing ML - Notebook 5: Embeddings & Similarity Search
# Prerequisites: Run notebooks 01-02 first

1️⃣ Connect & Load Embedding Model
from Deepnote environment import userdata
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")
engine = create_engine(DATABASE_URL)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✅ Model loaded (dim={embed_model.get_sentence_embedding_dimension()})")

2️⃣ Create Sample Clinical Notes
sample_notes = [
    {"note_type": "discharge", "note_text": "Patient admitted with uncontrolled Type 2 diabetes mellitus. Blood glucose stabilized with insulin. HbA1c 9.2%. Discharged on adjusted metformin."},
    {"note_type": "discharge", "note_text": "Acute chest pain with ST elevation in V1-V4. Emergent PCI with drug-eluting stent to LAD. Post-MI protocol initiated."},
    {"note_type": "discharge", "note_text": "Community-acquired pneumonia with right lower lobe consolidation. IV antibiotics transitioned to oral azithromycin. Oxygen resolved day 3."},
    {"note_type": "operative", "note_text": "Right total knee arthroplasty for end-stage osteoarthritis. Cemented components. EBL 150mL. Tolerated well."},
    {"note_type": "procedure", "note_text": "Colonoscopy for CRC screening. Two 5mm tubular adenomas removed from ascending colon. No malignancy. Repeat in 5 years."},
]

claim_ids = pd.read_sql("SELECT claim_id FROM claims LIMIT 5", engine)['claim_id'].tolist()

with engine.connect() as conn:
    for i, note in enumerate(sample_notes):
        if i < len(claim_ids):
            embedding = embed_model.encode([note['note_text']])[0]
            emb_str = '[' + ','.join(map(str, embedding)) + ']'
            conn.execute(text("""
                INSERT INTO clinical_notes (claim_id, note_type, note_text, embedding)
                VALUES (:cid, :ntype, :ntext, :emb::vector)
            """), {'cid': claim_ids[i], 'ntype': note['note_type'], 'ntext': note['note_text'], 'emb': emb_str})
    conn.commit()
print(f"✅ Stored {len(sample_notes)} clinical notes with embeddings!")

3️⃣ Similarity Search Function
def search_similar_notes(query: str, top_k: int = 5):
    query_embedding = embed_model.encode([query])[0]
    emb_str = '[' + ','.join(map(str, query_embedding)) + ']'

    results = pd.read_sql(text("""
        SELECT note_id, claim_id, note_type, note_text,
               1 - (embedding <=> :emb::vector) AS similarity
        FROM clinical_notes WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :emb::vector LIMIT :k
    """), engine, params={'emb': emb_str, 'k': top_k})
    return results

4️⃣ Test Searches
print("🔍 Query: 'diabetes blood sugar insulin'")
for _, row in search_similar_notes("diabetes blood sugar insulin", 3).iterrows():
    print(f"  [{row['similarity']:.3f}] {row['note_text'][:80]}...")

print("\n🔍 Query: 'heart attack cardiac stent'")
for _, row in search_similar_notes("heart attack cardiac stent", 3).iterrows():
    print(f"  [{row['similarity']:.3f}] {row['note_text'][:80]}...")

print("\n🔍 Query: 'knee replacement surgery'")
for _, row in search_similar_notes("knee replacement surgery", 3).iterrows():
    print(f"  [{row['similarity']:.3f}] {row['note_text'][:80]}...")

5️⃣ Billing Code Validation Function
CODE_DESCRIPTIONS = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'I21.0': 'ST elevation myocardial infarction anterior wall',
    'J18.9': 'Pneumonia unspecified organism',
    'M17.11': 'Primary osteoarthritis right knee',
}

def validate_billing_code(claim_id: int, icd_code: str):
    notes = pd.read_sql(text("""
        SELECT note_text FROM clinical_notes
        WHERE claim_id = :cid AND embedding IS NOT NULL
    """), engine, params={'cid': claim_id})

    if len(notes) == 0:
        return {'valid': None, 'reason': 'No clinical notes found'}

    code_desc = CODE_DESCRIPTIONS.get(icd_code, f'ICD-10 code {icd_code}')
    code_emb = embed_model.encode([code_desc])[0]

    max_sim = 0
    for _, row in notes.iterrows():
        note_emb = embed_model.encode([row['note_text']])[0]
        sim = np.dot(code_emb, note_emb) / (np.linalg.norm(code_emb) * np.linalg.norm(note_emb))
        max_sim = max(max_sim, sim)

    return {'code': icd_code, 'description': code_desc, 'similarity': float(max_sim),
            'valid': max_sim >= 0.4}

print("\n🏥 BILLING CODE VALIDATION")
result = validate_billing_code(claim_ids[0], 'E11.9')
print(f"Claim {claim_ids[0]} | Code: E11.9 | Sim: {result['similarity']:.3f} | {'✅ VALID' if result['valid'] else '❌ MISMATCH'}")
