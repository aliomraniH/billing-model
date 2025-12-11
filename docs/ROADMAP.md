# Implementation Roadmap

## Project Timeline Overview

```
Week 1-2: Foundation          │ Week 3-4: ML Pipeline       │ Week 5+: Advanced Features
──────────────────────────────┼─────────────────────────────┼────────────────────────────
• Database setup              │ • AutoTrain integration     │ • MIMIC-IV analysis
• Schema creation             │ • Embeddings + search       │ • medspaCy NLP
• Synthea data loading        │ • Model versioning          │ • ClinicalBERT fine-tuning
• Basic outlier detection     │ • Prediction storage        │ • Production API
```

---

## Week 1: Infrastructure & Connectivity

### Goals
- Set up all managed services
- Establish database connectivity from Google Colab
- Create schema and verify with sample data

### Tasks

#### 1.1 Create Vercel Postgres Database
**Status:** ⬜ Not Started

**Steps:**
1. Sign up at https://vercel.com
2. Navigate to Storage → Create Database → Postgres
3. Choose region closest to you (e.g., US East for lowest latency from Colab)
4. Copy **direct connection string** (not pooled)
   - Format: `postgresql://username:password@host:5432/dbname?sslmode=require`
5. Test connection locally:
   ```bash
   psql "postgresql://username:password@host:5432/dbname?sslmode=require"
   ```

**Expected Output:**
```
psql (14.5)
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_128_GCM_SHA256, bits: 128, compression: off)
Type "help" for help.

dbname=>
```

**Deliverables:**
- [ ] Vercel Postgres database created
- [ ] Connection string saved to Colab Secrets as `VERCEL_POSTGRES_URL`

---

#### 1.2 Test Connection from Google Colab
**Status:** ⬜ Not Started

**Steps:**
1. Open Google Colab: https://colab.research.google.com
2. Add secret in sidebar (🔑 icon): `VERCEL_POSTGRES_URL`
3. Run connection test:
   ```python
   from google.colab import userdata
   from sqlalchemy import create_engine, text

   DATABASE_URL = userdata.get('VERCEL_POSTGRES_URL')
   engine = create_engine(DATABASE_URL)

   with engine.connect() as conn:
       result = conn.execute(text("SELECT version();"))
       print(result.fetchone()[0])
   ```

**Expected Output:**
```
PostgreSQL 15.3 on x86_64-pc-linux-gnu, compiled by gcc...
```

**Deliverables:**
- [ ] Successful connection from Colab
- [ ] Connection test notebook saved

---

#### 1.3 Enable pgvector & Create Schema
**Status:** ⬜ Not Started

**Steps:**
1. Copy `sql/schema.sql` from this repository
2. Run in Colab using notebook `01_setup_database.py`
3. Verify tables created:
   ```sql
   SELECT table_name FROM information_schema.tables
   WHERE table_schema = 'public' ORDER BY table_name;
   ```

**Expected Output:**
```
    table_name
-----------------
 claims
 clinical_notes
 diagnoses
 predictions
 procedures
```

**Deliverables:**
- [ ] pgvector extension enabled
- [ ] All 5 tables created
- [ ] Indexes created

---

#### 1.4 Create Hugging Face Account
**Status:** ⬜ Not Started

**Steps:**
1. Sign up at https://huggingface.co/join
2. Navigate to Settings → Access Tokens
3. Create token with **write** permissions
4. Add to Colab Secrets as `HF_TOKEN`

**Deliverables:**
- [ ] Hugging Face account created
- [ ] Write-access token saved to Colab Secrets

---

## Week 2: Data Loading & Basic ML

### Goals
- Load Synthea synthetic data into Postgres
- Train initial outlier detection model
- Write predictions back to database

### Tasks

#### 2.1 Download Synthea Sample Dataset
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `02_load_synthea_data.py` (section 1)
2. Downloads 1.2K patient sample (~50 MB)
3. Extracts CSVs to `/content/synthea_data/csv/`

**Expected Files:**
- `patients.csv` (~1.2K rows)
- `encounters.csv` (~10K rows)
- `conditions.csv` (~15K rows)
- `procedures.csv` (~8K rows)
- `medications.csv` (~20K rows)

**Deliverables:**
- [ ] Synthea data downloaded
- [ ] CSVs extracted and verified

---

#### 2.2 Load Claims Data
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `02_load_synthea_data.py` (section 3)
2. Maps `encounters.csv` → `claims` table
3. Inserts 5,000 claims (batched for performance)

**Data Mapping:**
```python
encounters.PATIENT → claims.patient_id (hashed)
encounters.START → claims.service_date
encounters.ENCOUNTERCLASS → claims.claim_type
encounters.TOTAL_CLAIM_COST → claims.total_charge
encounters.PAYER_COVERAGE → claims.total_paid
```

**Expected Output:**
```
✅ Loaded 5,000 claims!
```

**Deliverables:**
- [ ] 5,000 claims inserted
- [ ] No duplicate claim IDs

---

#### 2.3 Load Diagnoses
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `02_load_synthea_data.py` (section 4)
2. Maps `conditions.csv` → `diagnoses` table
3. Joins with claims via patient + date

**Expected Output:**
```
✅ Loaded 10,000 diagnoses!

📊 DATABASE SUMMARY
Total Claims:    5,000
Total Diagnoses: 10,000
Avg Charge:      $1,234.56
Max Charge:      $45,678.90
```

**Deliverables:**
- [ ] Diagnoses linked to claims
- [ ] Foreign key constraints verified

---

#### 2.4 Train Outlier Detection Model
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `03_outlier_detection.py`
2. Extract features (charge amounts, payment ratios, diagnosis counts)
3. Train Isolation Forest (contamination=0.05)
4. Update `claims` table with `is_outlier` and `outlier_score`

**Expected Output:**
```
🔍 OUTLIER DETECTION RESULTS
Outliers Found: 250 (5.0%)

🚨 TOP 10 OUTLIERS
 claim_id  total_charge  num_diagnoses  num_procedures  outlier_score
     4521      45678.90              8              12           0.87
     3201      38234.12              1               2           0.82
      ...           ...            ...             ...            ...
```

**Deliverables:**
- [ ] Model trained and saved to `/content/outlier_model.pkl`
- [ ] 5,000 claims updated with outlier scores
- [ ] Top outliers analyzed

---

## Week 3: AutoTrain & Embeddings

### Goals
- Push dataset to Hugging Face Hub
- Train XGBoost classifier
- Generate embeddings for clinical notes

### Tasks

#### 3.1 Upload Dataset to Hugging Face
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `04_huggingface_autotrain.py` (section 2)
2. Extracts training data from Postgres
3. Splits into train/test (80/20)
4. Pushes to Hugging Face Hub

**Expected Output:**
```
✅ Dataset uploaded to: https://huggingface.co/datasets/YOUR_USERNAME/medical-billing-outliers
```

**Deliverables:**
- [ ] Dataset visible on Hugging Face
- [ ] Train/test split verified (80/20)

---

#### 3.2 Train XGBoost Classifier
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `04_huggingface_autotrain.py` (section 3)
2. Trains XGBoost on tabular features
3. Evaluates on test set

**Expected Output:**
```
📊 MODEL EVALUATION
              precision    recall  f1-score   support

      Normal       0.98      0.99      0.99       950
     Outlier       0.85      0.78      0.81        50

    accuracy                           0.97      1000
ROC-AUC: 0.9234
```

**Deliverables:**
- [ ] Model trained (ROC-AUC > 0.90)
- [ ] Model uploaded to Hugging Face

---

#### 3.3 Generate Embeddings for Clinical Notes
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `05_embeddings_search.py` (section 2)
2. Creates 5 sample clinical notes
3. Generates embeddings with all-MiniLM-L6-v2 (384-dim)
4. Stores in `clinical_notes` table

**Expected Output:**
```
✅ Stored 5 clinical notes with embeddings!
```

**Deliverables:**
- [ ] Sample notes inserted
- [ ] Embeddings stored as VECTOR(384)

---

#### 3.4 Implement Similarity Search
**Status:** ⬜ Not Started

**Steps:**
1. Run notebook `05_embeddings_search.py` (section 4)
2. Tests queries: "diabetes", "heart attack", "knee surgery"
3. Uses pgvector cosine similarity (<=> operator)

**Expected Output:**
```
🔍 Query: 'diabetes blood sugar insulin'
  [0.847] Patient admitted with uncontrolled Type 2 diabetes mellitus...
  [0.423] Community-acquired pneumonia with right lower lobe consolidation...
  [0.312] Acute chest pain with ST elevation in V1-V4...
```

**Deliverables:**
- [ ] Similarity search function working
- [ ] Top-k results validated manually

---

## Week 4+: Clinical Validation

### Goals
- Obtain MIMIC-IV access
- Implement medspaCy for entity extraction
- Fine-tune ClinicalBERT for code prediction

### Tasks

#### 4.1 Get MIMIC-IV Credentialing
**Status:** ⬜ Not Started

**Steps:**
1. Complete CITI training (2-3 hours): https://physionet.org/about/citi-course/
2. Upload certificate to PhysioNet profile
3. Sign MIMIC-IV Data Use Agreement
4. Wait for approval (usually < 24 hours)

**Deliverables:**
- [ ] CITI certificate obtained
- [ ] MIMIC-IV access approved

---

#### 4.2 Load MIMIC-IV Discharge Notes
**Status:** ⬜ Not Started

**Steps:**
1. Download `discharge.csv.gz` from MIMIC-IV
2. Create new Colab notebook: `06_mimic_notes.py`
3. Load sample of 1,000 discharge notes
4. Generate embeddings and store

**Expected Output:**
```
✅ Loaded 1,000 MIMIC-IV discharge notes
✅ Generated embeddings for all notes
```

**Deliverables:**
- [ ] 1,000 real clinical notes in database
- [ ] Embeddings generated

---

#### 4.3 Implement medspaCy Pipeline
**Status:** ⬜ Not Started

**Steps:**
1. Install medspaCy: `!pip install medspacy`
2. Load pre-trained model: `en_core_sci_md`
3. Extract entities (diseases, medications, procedures)
4. Detect negations ("no evidence of diabetes")

**Expected Output:**
```
Note: "Patient denies chest pain. History of diabetes."
Entities:
  - chest pain (PROBLEM, negated=True)
  - diabetes (PROBLEM, negated=False)
```

**Deliverables:**
- [ ] medspaCy pipeline working
- [ ] Negation detection validated

---

#### 4.4 Fine-tune ClinicalBERT
**Status:** ⬜ Not Started

**Steps:**
1. Prepare dataset: clinical notes → ICD-10 codes
2. Use Hugging Face AutoTrain for text classification
3. Fine-tune `emilyalsentzer/Bio_ClinicalBERT`
4. Evaluate on held-out test set

**Expected Output:**
```
📊 CODE PREDICTION RESULTS
Top-1 Accuracy: 67.3%
Top-5 Accuracy: 89.1%
```

**Deliverables:**
- [ ] Fine-tuned ClinicalBERT model
- [ ] Top-5 accuracy > 85%

---

## Milestones & Success Criteria

### Milestone 1: Database Pipeline ✅
- [ ] Vercel Postgres connected
- [ ] Schema created
- [ ] 5,000 Synthea claims loaded

### Milestone 2: Outlier Detection ✅
- [ ] Isolation Forest model trained
- [ ] Outlier scores written to database
- [ ] Top outliers manually reviewed

### Milestone 3: Embeddings & Search ✅
- [ ] Clinical notes embedded (384-dim)
- [ ] Similarity search working
- [ ] Query latency < 100ms for k=10

### Milestone 4: Clinical Validation ✅
- [ ] MIMIC-IV notes loaded
- [ ] medspaCy entity extraction
- [ ] Code suggestion model (top-5 accuracy > 85%)

---

## Open Questions & Decisions Needed

### Question 1: MIMIC-IV Priority
**Question:** Should we prioritize getting MIMIC-IV access early, or fully develop the pipeline with Synthea first?

**Options:**
- A) Start MIMIC-IV credentialing in Week 1 (parallel to Synthea work)
- B) Wait until Week 3 after Synthea pipeline is stable

**Recommendation:** **Option A** - Start credentialing early since approval takes 24+ hours

---

### Question 2: Vector DB Scaling
**Question:** At what point should we switch from pgvector to a dedicated vector database?

**Triggers:**
- Database size > 1 GB
- > 100K embeddings
- Query latency > 500ms for k=10

**Options:**
- Pinecone ($70/mo, 1M vectors)
- Qdrant Cloud ($25/mo, 1M vectors)
- Stay with pgvector + HNSW indexing

**Recommendation:** Stay with pgvector until latency becomes an issue (likely > 500K notes)

---

### Question 3: Code Granularity
**Question:** Should we predict ICD-10 at category level (3-char) or full code (5-7 char)?

**Trade-offs:**
- **3-char (e.g., E11):** Easier to predict, less useful
- **5-7 char (e.g., E11.65):** More precise, much harder (18,000+ codes)

**Recommendation:** Start with 3-char, expand to full codes after achieving > 80% accuracy

---

### Question 4: Ground Truth for Outliers
**Question:** How do we establish what constitutes a "true" outlier for evaluation?

**Options:**
- A) Manual review by domain expert (expensive, slow)
- B) Known fraud cases from CMS (hard to obtain)
- C) Synthetic injection (add artificial outliers to Synthea)
- D) Trust Isolation Forest scores as proxy

**Recommendation:** **Option C** - Inject synthetic outliers (e.g., $100K charge for flu visit) to validate detection

---

## Next Steps

1. **Choose a starting point** from the 6 options in the main README
2. **Run Week 1 tasks** (database setup) if not done yet
3. **Review open questions** and make architectural decisions
4. **Track progress** using this roadmap as a checklist

---

**Last Updated:** 2025-12-11
