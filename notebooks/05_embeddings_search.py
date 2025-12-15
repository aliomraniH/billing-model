# Medical Billing ML - Notebook 5: Embeddings & Similarity Search
# Prerequisites: Run notebooks 01-02 first

# Import base dependencies
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

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

# Load embedding model (lazy import to avoid PyTorch initialization issues)
print("\n📥 Loading embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = embed_model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded successfully")
    print(f"   Model: all-MiniLM-L6-v2")
    print(f"   Embedding dimensions: {embedding_dim}")
except Exception as e:
    print(f"❌ Failed to load sentence_transformers: {e}")
    print("\n⚠️  PyTorch Compatibility Issue Detected!")
    print("   This is usually caused by version conflicts between torch and sentence-transformers.")
    print("\n   To fix this issue, run these commands in your terminal:")
    print("   pip uninstall -y torch torchvision torchaudio sentence-transformers")
    print("   pip install torch==2.0.1 sentence-transformers==2.2.2")
    print("\n   Then restart your notebook kernel and try again.")
    raise

# Create Comprehensive Clinical Notes for Clustering
print("\n📝 Creating comprehensive clinical notes dataset...")

# Diverse medical scenarios across multiple categories for meaningful clustering
sample_notes = [
    # CATEGORY 1: Diabetes & Endocrine (10 notes)
    {"note_type": "discharge", "note_text": "Patient admitted with uncontrolled Type 2 diabetes mellitus. Blood glucose stabilized with insulin. HbA1c 9.2%. Discharged on adjusted metformin."},
    {"note_type": "discharge", "note_text": "Type 1 diabetes with diabetic ketoacidosis. pH 7.21, glucose 450. IV insulin drip initiated. Transitioned to subcutaneous insulin regimen."},
    {"note_type": "discharge", "note_text": "Hyperosmolar hyperglycemic state in elderly diabetic. Glucose 850. Aggressive fluid resuscitation. Mental status improved. Home with sliding scale."},
    {"note_type": "encounter", "note_text": "Diabetes follow-up visit. A1C improved to 7.1% on current regimen. Mild peripheral neuropathy noted. Continue metformin and glipizide."},
    {"note_type": "encounter", "note_text": "New diagnosis Type 2 diabetes. Glucose 220 fasting. Started on metformin 500mg BID. Nutrition counseling provided. RTC 3 months."},
    {"note_type": "discharge", "note_text": "Diabetic foot ulcer with cellulitis. IV antibiotics, wound care. Vascular consult obtained. Discharged with home health wound care."},
    {"note_type": "encounter", "note_text": "Gestational diabetes screening positive. 3hr GTT confirmed diagnosis. Started on insulin therapy. MFM referral placed."},
    {"note_type": "discharge", "note_text": "Hypoglycemic episode in diabetic on insulin. Glucose 38. Dextrose given. Insulin regimen adjusted. Patient educated on hypoglycemia management."},
    {"note_type": "encounter", "note_text": "Diabetic retinopathy screening. Mild non-proliferative changes noted. Ophthalmology follow-up scheduled. Blood sugar control emphasized."},
    {"note_type": "discharge", "note_text": "Diabetic nephropathy with acute kidney injury. Creatinine 2.8. Metformin held. Started on ACE inhibitor. Nephrology consult completed."},

    # CATEGORY 2: Cardiac Conditions (10 notes)
    {"note_type": "discharge", "note_text": "Acute chest pain with ST elevation in V1-V4. Emergent PCI with drug-eluting stent to LAD. Post-MI protocol initiated."},
    {"note_type": "discharge", "note_text": "NSTEMI with elevated troponin. Cardiac catheterization showed 70% RCA stenosis. Stent placed. Started on dual antiplatelet therapy."},
    {"note_type": "discharge", "note_text": "Acute decompensated heart failure. BNP 2400. IV diuresis initiated. Echo shows EF 25%. Discharged on Lasix, carvedilol, lisinopril."},
    {"note_type": "encounter", "note_text": "Hypertension poorly controlled. BP 168/95. Medication non-compliance discussed. Added amlodipine to current regimen."},
    {"note_type": "discharge", "note_text": "Atrial fibrillation with rapid ventricular response. Rate controlled with diltiazem. Started on apixaban for stroke prevention."},
    {"note_type": "procedure", "note_text": "Elective coronary angiography. Three-vessel disease identified. CABG recommended. Cardiac surgery consult arranged."},
    {"note_type": "discharge", "note_text": "Acute pericarditis. Chest pain, pericardial friction rub. ECG changes consistent. Treated with NSAIDs and colchicine."},
    {"note_type": "encounter", "note_text": "Chest pain evaluation. Stress test positive for ischemia. Cardiac catheterization scheduled. Started on aspirin and statin."},
    {"note_type": "discharge", "note_text": "Hypertensive emergency. BP 210/120 with headache. Nicardipine drip initiated. Transitioned to oral agents. BP stable at discharge."},
    {"note_type": "procedure", "note_text": "Permanent pacemaker insertion for complete heart block. Dual chamber device placed. Post-op check normal. Device clinic follow-up."},

    # CATEGORY 3: Respiratory Conditions (10 notes)
    {"note_type": "discharge", "note_text": "Community-acquired pneumonia with right lower lobe consolidation. IV antibiotics transitioned to oral azithromycin. Oxygen resolved day 3."},
    {"note_type": "discharge", "note_text": "COPD exacerbation with acute respiratory failure. Bipap initiated. Steroids and bronchodilators. Improved, discharged on home oxygen."},
    {"note_type": "discharge", "note_text": "Asthma exacerbation requiring ICU admission. Status asthmaticus. Continuous albuterol, IV steroids. Stabilized and discharged with inhaler."},
    {"note_type": "encounter", "note_text": "Chronic bronchitis follow-up. Productive cough, wheezing. Spirometry shows obstruction. Advair and Spiriva initiated."},
    {"note_type": "discharge", "note_text": "COVID-19 pneumonia with hypoxemia. Oxygen requirements up to 6L NC. Dexamethasone and remdesivir given. Discharged after 8 days."},
    {"note_type": "procedure", "note_text": "Bronchoscopy for hemoptysis. Bleeding source identified in right middle lobe. No malignancy on biopsy. Anticoagulation held."},
    {"note_type": "discharge", "note_text": "Pulmonary embolism. D-dimer elevated, CTA positive. Started on apixaban. Risk stratification low. Home treatment appropriate."},
    {"note_type": "encounter", "note_text": "Lung nodule follow-up. 8mm nodule stable on CT. Continued surveillance recommended. Non-smoker, low malignancy risk."},
    {"note_type": "discharge", "note_text": "Pleural effusion requiring thoracentesis. 1.5L serosanguinous fluid removed. Cytology negative. Etiology likely CHF."},
    {"note_type": "discharge", "note_text": "Pneumothorax post central line placement. Small apical pneumothorax. Conservative management. Repeat CXR showed resolution."},

    # CATEGORY 4: Orthopedic/Surgical (10 notes)
    {"note_type": "operative", "note_text": "Right total knee arthroplasty for end-stage osteoarthritis. Cemented components. EBL 150mL. Tolerated well."},
    {"note_type": "operative", "note_text": "Left total hip replacement for avascular necrosis. Uncemented prosthesis. No complications. PT started POD1."},
    {"note_type": "operative", "note_text": "Open reduction internal fixation right distal radius fracture. Volar plate and screws placed. Neurovascular intact."},
    {"note_type": "operative", "note_text": "Arthroscopic rotator cuff repair. Full thickness supraspinatus tear. Anchors placed. Sling immobilization."},
    {"note_type": "operative", "note_text": "Lumbar laminectomy L4-L5 for spinal stenosis. Decompression achieved. No dural tear. Ambulating POD1."},
    {"note_type": "discharge", "note_text": "Vertebral compression fracture T12. Kyphoplasty performed. Pain improved significantly. Discharged with back brace."},
    {"note_type": "operative", "note_text": "ACL reconstruction with hamstring autograft. Arthroscopic technique. Stable post-op. Physical therapy protocol initiated."},
    {"note_type": "encounter", "note_text": "Post-op follow-up after knee replacement. Wound healing well. ROM improving. Continue PT three times weekly."},
    {"note_type": "discharge", "note_text": "Hip fracture post-fall. ORIF performed with dynamic hip screw. Weight bearing as tolerated. Rehab facility placement."},
    {"note_type": "operative", "note_text": "Carpal tunnel release bilateral. Endoscopic approach. Immediate symptom relief. Return to work in 2 weeks."},

    # CATEGORY 5: GI/General Surgery (10 notes)
    {"note_type": "procedure", "note_text": "Colonoscopy for CRC screening. Two 5mm tubular adenomas removed from ascending colon. No malignancy. Repeat in 5 years."},
    {"note_type": "operative", "note_text": "Laparoscopic cholecystectomy for acute cholecystitis. Gangrenous gallbladder. Converted to open. No bile duct injury."},
    {"note_type": "discharge", "note_text": "Acute appendicitis. Laparoscopic appendectomy performed. Pathology confirmed acute inflammation. Discharged POD2."},
    {"note_type": "procedure", "note_text": "Upper endoscopy for GERD. Esophagitis grade B. Multiple biopsies taken. Started on pantoprazole BID."},
    {"note_type": "discharge", "note_text": "Small bowel obstruction. CT showed adhesions. Conservative management with NGT decompression. Obstruction resolved."},
    {"note_type": "operative", "note_text": "Inguinal hernia repair with mesh. Indirect hernia reduced. Mesh fixation. Same day discharge. Light duty 2 weeks."},
    {"note_type": "discharge", "note_text": "GI bleed from duodenal ulcer. EGD with epinephrine injection and clipping. H. pylori positive. Triple therapy started."},
    {"note_type": "operative", "note_text": "Right hemicolectomy for colon cancer. Tumor resected with clear margins. Lymph nodes negative. Oncology referral."},
    {"note_type": "procedure", "note_text": "ERCP for choledocholithiasis. Sphincterotomy performed. Stone extraction successful. Stent not required."},
    {"note_type": "discharge", "note_text": "Diverticulitis with microperforation. IV antibiotics. CT-guided drainage of abscess. Improved, discharged on oral antibiotics."},

    # CATEGORY 6: Women's Health/OB (5 notes)
    {"note_type": "operative", "note_text": "Primary cesarean section for failure to progress. 8lb 2oz male infant. Apgars 8/9. Routine postpartum care."},
    {"note_type": "discharge", "note_text": "Preeclampsia with severe features at 36 weeks. Magnesium sulfate given. Induced labor, vaginal delivery."},
    {"note_type": "operative", "note_text": "Total abdominal hysterectomy for symptomatic fibroids. Uterus 16-week size. No complications. Home POD3."},
    {"note_type": "encounter", "note_text": "Routine prenatal visit at 28 weeks. Glucose tolerance test normal. Fundal height appropriate. Tdap vaccine given."},
    {"note_type": "procedure", "note_text": "Dilation and curettage for incomplete miscarriage. Products of conception removed. Minimal bleeding. Follow-up in 2 weeks."},

    # CATEGORY 7: Neurology (5 notes)
    {"note_type": "discharge", "note_text": "Acute ischemic stroke right MCA territory. tPA administered within window. NIH stroke scale improved from 12 to 4."},
    {"note_type": "encounter", "note_text": "Migraine with aura. Frequency 3-4 per month. Started on topiramate prophylaxis. Sumatriptan for acute episodes."},
    {"note_type": "discharge", "note_text": "New onset seizure disorder. Witnessed tonic-clonic seizure. MRI brain normal. Started on levetiracetam. Neurology follow-up."},
    {"note_type": "encounter", "note_text": "Parkinson disease management. Tremor and rigidity progressive. Carbidopa-levodopa dose increased. Gait stable."},
    {"note_type": "discharge", "note_text": "Multiple sclerosis relapse. Optic neuritis and weakness. IV methylprednisolone 1g daily x5 days. Improved at discharge."},

    # CATEGORY 8: Infectious Disease (5 notes)
    {"note_type": "discharge", "note_text": "Sepsis from urinary source. Blood cultures positive for E. coli. Broad spectrum antibiotics narrowed to ceftriaxone."},
    {"note_type": "discharge", "note_text": "Cellulitis right lower extremity. Erythema and edema improving on IV vancomycin. Transitioned to oral antibiotics."},
    {"note_type": "encounter", "note_text": "HIV positive patient. CD4 count 450. Viral load undetectable on ART. Continue current regimen. Excellent adherence."},
    {"note_type": "discharge", "note_text": "C. difficile colitis. Severe diarrhea with leukocytosis. Oral vancomycin initiated. Symptoms improved by day 5."},
    {"note_type": "encounter", "note_text": "Tuberculosis treatment monitoring. Month 3 of RIPE therapy. Chest X-ray improved. LFTs normal. Continue treatment."},
]

print(f"   Created {len(sample_notes)} diverse clinical notes across 8 medical categories")

# Get claim IDs to link notes to
claim_ids = pd.read_sql(f"SELECT claim_id FROM claims LIMIT {len(sample_notes)}", engine)['claim_id'].tolist()
print(f"   Using {len(claim_ids)} claim IDs for embedding generation")

# Generate embeddings and store
print("\n🔄 Generating embeddings and storing in database...")
print("   This may take a minute for large datasets...")

with engine.begin() as conn:  # Changed from engine.connect() to engine.begin()
    for i, note in enumerate(sample_notes):
        if i < len(claim_ids):
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{len(sample_notes)} embeddings generated...")

            # Generate embedding
            embedding = embed_model.encode([note['note_text']])[0]
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
    # Auto-commits on context exit

print(f"✅ Stored {len(sample_notes)} clinical notes with embeddings!")
print(f"   Dataset contains 8 medical categories for robust clustering:")
print(f"   • Diabetes & Endocrine (10 notes)")
print(f"   • Cardiac Conditions (10 notes)")
print(f"   • Respiratory (10 notes)")
print(f"   • Orthopedic/Surgical (10 notes)")
print(f"   • GI/General Surgery (10 notes)")
print(f"   • Women's Health/OB (5 notes)")
print(f"   • Neurology (5 notes)")
print(f"   • Infectious Disease (5 notes)")
print(f"\n   Ready for notebook 06 (LLM Clustering)!")

# Similarity Search Function
def search_similar_notes(query: str, top_k: int = 5):
    """Search for clinically similar notes using semantic similarity"""
    # Generate query embedding
    query_embedding = embed_model.encode([query])[0]
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
