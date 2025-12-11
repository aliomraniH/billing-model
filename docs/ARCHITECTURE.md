# Architecture Documentation

## System Overview

The Medical Billing ML system is a browser-based machine learning pipeline designed for analyzing medical claims data. All components are managed services to minimize operational overhead and stay within a $50/month budget.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Google Colab Notebooks                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Data Loader │  │ Model Trainer│  │ Prediction Generator│   │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘   │
│         │                │                      │               │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vercel Postgres + pgvector                   │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Claims  │  │ Diagnoses  │  │ Procedures │  │  Clinical  │ │
│  │          │  │            │  │            │  │   Notes    │ │
│  └──────────┘  └────────────┘  └────────────┘  └────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Predictions Table (Model Outputs + Metadata)           │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │                                      │
          │                                      │
          ▼                                      ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Hugging Face Hub    │              │  Training Data       │
│  • Model Registry    │              │  • Synthea (free)    │
│  • Dataset Storage   │              │  • MIMIC-IV (cred)   │
│  • AutoTrain Jobs    │              │  • CMS DE-SynPUF     │
└──────────────────────┘              └──────────────────────┘
```

## Technology Selection Rationale

### Database: Vercel Postgres (Neon)

**Why Chosen:**
- Serverless Postgres with pgvector 0.8.0 pre-installed
- Free tier: 512 MB storage, 60 hours compute/month
- No IP allowlisting required (works from Google Colab)
- Direct connection support (not just pooled)
- Automatic backups and point-in-time recovery

**Alternatives Considered:**
- **Supabase**: Similar features but less generous free tier compute limits
- **AWS RDS**: Requires VPC setup, minimum $15/month, overkill for prototype
- **PlanetScale**: MySQL-based, no native vector support

**Limitations:**
- 512 MB storage limit on free tier (upgrade to Pro at $20/mo for 10 GB if needed)
- Connection limits (can hit with concurrent Colab sessions)

### Vector Store: pgvector

**Why Chosen:**
- Integrated with Postgres (single database for relational + embeddings)
- Supports up to 2000 dimensions (we use 384 for all-MiniLM-L6-v2)
- HNSW indexing for fast approximate nearest neighbor search
- Cosine, L2, and inner product distance metrics

**Alternatives Considered:**
- **Pinecone**: $70/month for 1M vectors, overkill for prototype scale
- **Qdrant**: Requires separate service deployment
- **Weaviate**: Complex setup for managed deployment

**Limitations:**
- Performance degrades beyond ~1M vectors without tuning
- HNSW index creation can be slow for large datasets

### Development Environment: Google Colab Pro

**Why Chosen:**
- $9.99/month for GPU access (T4, V100, A100 when available)
- 24-hour session limits (vs 12 hours on free tier)
- Built-in secrets management
- No local environment setup required
- Jupyter notebook interface familiar to data scientists

**Alternatives Considered:**
- **Kaggle Notebooks**: Free GPU but limited session time, no secrets management
- **Paperspace Gradient**: $8/month but requires more setup
- **Saturn Cloud**: Free tier too limited

**Limitations:**
- Sessions disconnect after 24 hours (need to re-run initialization)
- No persistent file storage (must save models to Hugging Face or Google Drive)
- Can't run background jobs

### AutoML: Hugging Face AutoTrain

**Why Chosen:**
- Pay-per-use compute (only pay when training)
- Supports tabular data + NLP tasks
- Integrates with Hugging Face Hub for dataset/model versioning
- No infrastructure management

**Alternatives Considered:**
- **Google AutoML**: $20/hour minimum, expensive for prototyping
- **Azure ML**: Complex setup, enterprise-focused pricing
- **AWS SageMaker Autopilot**: Minimum $0.17/hour, requires AWS setup

**Estimated Costs:**
- Tabular classification: ~$2-5 per training run
- Text classification (BERT fine-tuning): ~$10-15 per run
- Budget: 5-10 runs/month = $15-25

**Limitations:**
- Less control than custom training loops
- Limited hyperparameter tuning options

### Clinical NLP: medspaCy + scispaCy

**Why Chosen:**
- Free and open-source
- Built on spaCy (production-ready, fast)
- Medical-specific features:
  - Negation detection (e.g., "no evidence of diabetes")
  - UMLS entity linking
  - Section detection (History of Present Illness, Assessment, Plan)
- Pre-trained models available

**Alternatives Considered:**
- **Amazon Comprehend Medical**: $0.01 per 100 characters, too expensive at scale
- **Azure Health Text Analytics**: Similar pricing to AWS
- **ClinicalBERT**: Requires fine-tuning, no out-of-box negation detection

**Limitations:**
- Requires UMLS license for entity linking (free but needs registration)
- Pre-trained models are general (may need fine-tuning for specific specialties)

## Data Flow

### 1. Data Ingestion

```
Synthea CSVs → Pandas DataFrame → SQLAlchemy → Vercel Postgres
```

**Process:**
1. Download Synthea sample data (1.2K patients, ~10K encounters)
2. Map columns to schema (encounters → claims, conditions → diagnoses)
3. Hash patient/provider IDs for anonymization
4. Batch insert via SQLAlchemy (500 rows/batch)

### 2. Feature Engineering

```sql
SELECT
    c.claim_id,
    c.total_charge,
    c.total_paid,
    c.total_paid / NULLIF(c.total_charge, 0) AS payment_ratio,
    COUNT(DISTINCT d.diagnosis_id) AS num_diagnoses,
    COUNT(DISTINCT p.procedure_id) AS num_procedures
FROM claims c
LEFT JOIN diagnoses d ON c.claim_id = d.claim_id
LEFT JOIN procedures p ON c.claim_id = p.claim_id
GROUP BY c.claim_id
```

### 3. Model Training

**Outlier Detection (Isolation Forest):**
```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expect 5% outliers
    random_state=42
)
model.fit(X_scaled)
```

**Classification (XGBoost):**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=class_imbalance_ratio
)
model.fit(X_train, y_train)
```

### 4. Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(clinical_note_text)  # Returns 384-dim vector
```

### 5. Prediction Storage

```sql
UPDATE claims
SET is_outlier = TRUE, outlier_score = 0.87
WHERE claim_id = 12345;

INSERT INTO predictions (claim_id, model_name, prediction_type, prediction_value)
VALUES (12345, 'xgboost-v1.0', 'outlier', '{"probability": 0.87, "features": {...}}');
```

## Scalability Considerations

### Current Scale (Prototype)
- **Claims:** ~5,000
- **Clinical Notes:** ~100-500
- **Embeddings:** 500 × 384 = ~192K floats = 768 KB
- **Database Size:** < 50 MB

### Expected Production Scale
- **Claims:** 100K - 1M
- **Clinical Notes:** 50K - 500K
- **Embeddings:** 500K × 384 = ~768 MB
- **Database Size:** 5-50 GB

### Scaling Strategy

**When to upgrade Vercel Postgres:**
- Free tier limit: 512 MB storage
- Upgrade trigger: > 400 MB (80% capacity)
- Pro tier ($20/mo): 10 GB storage, 100 hours compute

**When to switch from pgvector to dedicated vector DB:**
- > 1M embeddings
- Query latency > 500ms for k=10 nearest neighbors
- Alternatives: Pinecone ($70/mo), Qdrant Cloud ($25/mo)

**When to move from Colab to dedicated compute:**
- Training jobs > 12 hours
- Need for scheduled/automated retraining
- Alternatives: Paperspace, Lambda Labs GPU instances

## Security & Compliance

### Data Protection
- **Synthetic Data Only (Phase 1)**: Synthea-generated, no HIPAA concerns
- **Real Data (Phase 2+)**: MIMIC-IV requires:
  - PhysioNet credentialing
  - Data use agreement (DUA)
  - No data export from Colab (keep in encrypted DB)

### Connection Security
- Vercel Postgres: TLS-encrypted connections (required)
- Hugging Face: Private repositories for datasets/models
- Colab Secrets: Encrypted credential storage

### Access Control
- Database: Single service account (principle of least privilege)
- Hugging Face: Write token for model uploads, read token for inference
- No production PHI in development environment

## Monitoring & Observability

### Metrics to Track
1. **Model Performance**
   - Outlier detection precision/recall
   - XGBoost ROC-AUC
   - Embedding similarity distributions

2. **Data Quality**
   - Claims with missing diagnoses
   - Negative charge amounts
   - Duplicate claim IDs

3. **System Health**
   - Database connection errors
   - Colab session disconnects
   - Model training failures

### Logging Strategy
- **Structured logs in predictions table:**
  ```json
  {
    "timestamp": "2025-12-11T10:30:00Z",
    "model_version": "xgboost-v1.2",
    "input_features": {...},
    "prediction": 0.87,
    "latency_ms": 45
  }
  ```

## Cost Breakdown

| Service | Tier | Monthly Cost | Notes |
|---------|------|--------------|-------|
| Vercel Postgres | Free → Pro | $0 → $20 | Upgrade when > 500 MB |
| Google Colab | Pro | $9.99 | Required for GPU |
| Hugging Face | Pro | $9.00 | Optional (better AutoTrain pricing) |
| AutoTrain Compute | Pay-per-use | $15-25 | 5-10 training runs |
| **Total** | | **$35-55** | Within $50 budget |

## Future Enhancements

1. **Real-time Inference API** (FastAPI on Render/Fly.io, $5-10/mo)
2. **Automated Retraining Pipeline** (GitHub Actions + scheduled Colab runs)
3. **Dashboard for Predictions** (Streamlit on Streamlit Cloud, free tier)
4. **Advanced NLP Models** (Fine-tune ClinicalBERT on MIMIC-IV notes)
