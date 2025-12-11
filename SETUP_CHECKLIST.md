# Setup Checklist

Use this checklist to track your progress setting up the Medical Billing ML project.

## Phase 1: Account Setup

### Vercel Account & Database
- [ ] Create Vercel account at https://vercel.com/signup
- [ ] Navigate to Storage → Create Database → Postgres
- [ ] Choose database name: `medical-billing-ml` (or custom)
- [ ] Select region: **US East** (recommended for Colab)
- [ ] Copy **POSTGRES_URL_NON_POOLING** (direct connection)
- [ ] Verify connection string contains `:5432` and `?sslmode=require`

### Google Colab Setup
- [ ] Open Google Colab: https://colab.research.google.com
- [ ] Subscribe to Colab Pro ($9.99/mo) - recommended for GPU access
- [ ] Add secret: `VERCEL_POSTGRES_URL` (🔑 Secrets sidebar)
- [ ] Paste direct connection string
- [ ] Toggle "Notebook access" ON

### Hugging Face Account
- [ ] Create account at https://huggingface.co/join
- [ ] Navigate to Settings → Access Tokens
- [ ] Create new token with **Write** permissions
- [ ] Copy token (starts with `hf_`)
- [ ] Add to Colab Secrets: `HF_TOKEN`

---

## Phase 2: Database Setup

### Connection Verification
- [ ] Run verification script in Colab:
  ```python
  # Copy-paste scripts/verify_connection.py into Colab cell
  ```
- [ ] Verify output shows:
  - ✅ Connected successfully
  - ✅ pgvector available (or installed)
  - ✅ Write permissions verified

### Schema Creation
- [ ] Open notebook: `notebooks/01_setup_database.py`
- [ ] Copy-paste into new Colab notebook
- [ ] Run all cells sequentially
- [ ] Verify output:
  - ✅ Dependencies installed
  - ✅ Connected to PostgreSQL 15.x
  - ✅ Schema created successfully
  - 📋 Tables: claims, clinical_notes, diagnoses, predictions, procedures

### Schema Verification
Run this in Colab to verify tables:
```python
from google.colab import userdata
from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = userdata.get('VERCEL_POSTGRES_URL')
engine = create_engine(DATABASE_URL)

# List all tables
tables = pd.read_sql("""
    SELECT table_name,
           pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
""", engine)
print(tables)
```

Expected output:
```
       table_name     size
0          claims    8192 bytes
1  clinical_notes    8192 bytes
2       diagnoses    8192 bytes
3     predictions    8192 bytes
4      procedures    8192 bytes
```

---

## Phase 3: Data Loading

### Download Synthea Data
- [ ] Open notebook: `notebooks/02_load_synthea_data.py`
- [ ] Copy-paste into Colab
- [ ] Run section 1: Download Synthea Sample Data
- [ ] Verify files downloaded:
  - `/content/synthea_data/csv/encounters.csv`
  - `/content/synthea_data/csv/conditions.csv`
  - `/content/synthea_data/csv/patients.csv`

### Load Claims Data
- [ ] Run section 2: Connect to Database
- [ ] Run section 3: Load Claims Data
- [ ] Verify output: ✅ Loaded 5,000 claims
- [ ] Check database:
  ```python
  pd.read_sql("SELECT COUNT(*) FROM claims", engine)
  ```
  Expected: 5000

### Load Diagnoses
- [ ] Run section 4: Load Diagnoses
- [ ] Verify output: ✅ Loaded 10,000 diagnoses
- [ ] Run section 5: Verify Data Load
- [ ] Expected output:
  ```
  Total Claims:    5,000
  Total Diagnoses: 10,000
  Avg Charge:      $X,XXX.XX
  Max Charge:      $XX,XXX.XX
  ```

---

## Phase 4: Model Training

### Outlier Detection
- [ ] Open notebook: `notebooks/03_outlier_detection.py`
- [ ] Run section 1: Connect to Database
- [ ] Run section 2: Prepare Features
- [ ] Verify: ✅ Loaded X,XXX claims with features
- [ ] Run section 3: Feature Engineering & Training
- [ ] Verify output:
  ```
  🔍 OUTLIER DETECTION RESULTS
  Outliers Found: ~250 (5.0%)
  ```
- [ ] Run section 4: Write Predictions to Database
- [ ] Run section 5: Analyze Top Outliers
- [ ] Review top 10 outliers (do they look unusual?)
- [ ] Run section 6: Save Model
- [ ] Model saved: `/content/outlier_model.pkl`

### Verify Predictions Stored
```python
pd.read_sql("""
    SELECT is_outlier, COUNT(*) as count
    FROM claims
    WHERE outlier_score IS NOT NULL
    GROUP BY is_outlier
""", engine)
```

Expected output:
```
   is_outlier  count
0       False   4750
1        True    250
```

---

## Phase 5: Advanced Features (Optional)

### Hugging Face Integration
- [ ] Open notebook: `notebooks/04_huggingface_autotrain.py`
- [ ] Run section 1: Setup Hugging Face
- [ ] Verify: ✅ Logged in to Hugging Face!
- [ ] Run section 2: Prepare & Upload Dataset
- [ ] Dataset uploaded to: `https://huggingface.co/datasets/YOUR_USERNAME/...`
- [ ] Run section 3: Train XGBoost Locally
- [ ] Verify ROC-AUC > 0.90
- [ ] Run section 4: Upload Model to Hub

### Embeddings & Similarity Search
- [ ] Open notebook: `notebooks/05_embeddings_search.py`
- [ ] Run section 1: Load Embedding Model
- [ ] Verify: ✅ Model loaded (dim=384)
- [ ] Run section 2: Create Sample Clinical Notes
- [ ] Verify: ✅ Stored 5 clinical notes with embeddings!
- [ ] Run section 3: Define Search Function
- [ ] Run section 4: Test Searches
- [ ] Verify similarity scores look reasonable (0.3-0.9)
- [ ] Run section 5: Test Billing Code Validation

### MIMIC-IV Access (Future)
- [ ] Complete CITI training: https://physionet.org/about/citi-course/
- [ ] Upload certificate to PhysioNet profile
- [ ] Sign MIMIC-IV Data Use Agreement
- [ ] Wait for approval (usually < 24 hours)
- [ ] Download discharge notes sample
- [ ] Load into `clinical_notes` table

---

## Phase 6: Monitoring & Maintenance

### Database Health Check
Run this weekly to monitor database usage:

```python
# Database size
db_size = pd.read_sql("""
    SELECT pg_size_pretty(pg_database_size(current_database())) as size
""", engine)
print(f"Database size: {db_size['size'][0]}")

# Table sizes
table_sizes = pd.read_sql("""
    SELECT
        tablename,
        pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
        pg_total_relation_size(tablename::regclass) as bytes
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY bytes DESC
""", engine)
print(table_sizes)

# Record counts
counts = pd.read_sql("""
    SELECT
        (SELECT COUNT(*) FROM claims) as claims,
        (SELECT COUNT(*) FROM diagnoses) as diagnoses,
        (SELECT COUNT(*) FROM clinical_notes) as notes,
        (SELECT COUNT(*) FROM predictions) as predictions
""", engine)
print(counts)
```

### When to Upgrade Vercel Postgres

Upgrade from **Free** ($0/mo) to **Pro** ($20/mo) when:
- [ ] Database size > 400 MB (80% of 512 MB limit)
- [ ] Compute hours > 50/month (83% of 60 hours limit)
- [ ] Need more than 20 concurrent connections

### Model Retraining Schedule
- [ ] Week 1: Initial training with Synthea data
- [ ] Week 4: Retrain with MIMIC-IV data (after credentialing)
- [ ] Monthly: Retrain outlier detection as new data arrives
- [ ] Quarterly: Fine-tune code suggestion models

---

## Troubleshooting

### Common Issues

**Issue:** "connection timeout"
- [ ] Verify connection string is correct (no typos)
- [ ] Check database is "Active" in Vercel dashboard
- [ ] Try regenerating credentials in Vercel

**Issue:** "pgvector not found"
- [ ] Run: `CREATE EXTENSION IF NOT EXISTS vector;`
- [ ] If still not available, database is too old (create new one)

**Issue:** "out of memory" in Colab
- [ ] Reduce batch sizes in data loading scripts
- [ ] Upgrade to Colab Pro for more RAM
- [ ] Process data in smaller chunks

**Issue:** Slow query performance
- [ ] Check indexes are created: `\di` in psql
- [ ] Run `ANALYZE` to update query planner statistics
- [ ] Consider creating HNSW index for embeddings (if > 10K notes)

---

## Success Criteria

You've successfully completed setup when:

- [x] ✅ Can connect to Vercel Postgres from Colab
- [x] ✅ All 5 tables created with correct schema
- [x] ✅ 5,000+ claims loaded from Synthea
- [x] ✅ Outlier detection model trained (ROC-AUC > 0.85)
- [x] ✅ Predictions written back to database
- [x] ✅ Embeddings generated for clinical notes
- [x] ✅ Similarity search working (query latency < 500ms)

**Congratulations!** You're ready for production use cases.

---

## Next Steps After Setup

1. **Explore data:**
   - Query top outliers
   - Analyze diagnosis patterns
   - Review similarity search results

2. **Improve models:**
   - Add more features (provider patterns, temporal trends)
   - Try ensemble methods (XGBoost + Isolation Forest)
   - Fine-tune ClinicalBERT for code suggestions

3. **Scale up:**
   - Generate larger Synthea dataset (10K-100K patients)
   - Load MIMIC-IV discharge notes
   - Implement automated retraining pipeline

4. **Build dashboard:**
   - Streamlit app for reviewing outliers
   - Visualizations of billing patterns
   - Code suggestion interface

---

**Last Updated:** 2025-12-11
**Estimated Time to Complete:** 2-3 hours
