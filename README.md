# Medical Billing ML Project

> Browser-based machine learning system for medical billing analysis using managed services

[![Budget](https://img.shields.io/badge/Budget-Under%20$50%2Fmonth-green)]()
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)]()
[![Database](https://img.shields.io/badge/Database-Vercel%20Postgres-blue)]()
[![ML](https://img.shields.io/badge/ML-Hugging%20Face-yellow)]()

## Overview

An experimental ML system for detecting billing anomalies, validating clinical claims, and suggesting appropriate medical codes from clinical documentation. All development happens in browser-based environments (Google Colab) with managed services only.

### Primary Objectives

1. **Outlier Detection** - Identify claims with unusual cost patterns, rare procedure combinations, and statistical anomalies
2. **Clinical Validity Cross-Check** - Compare billed services against clinical documentation to verify medical necessity
3. **Code Suggestion** - Given clinical notes, suggest appropriate ICD-10, CPT, or HCPCS billing codes

### Success Criteria

- ✅ Working pipeline: Database → Model Training → Predictions stored back
- ✅ Semantic similarity search between clinical notes and billing codes
- ✅ Models trained via Hugging Face AutoTrain
- ✅ All code runnable via copy-paste in Google Colab

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vercel Postgres │◄──►│  Google Colab    │───►│ Hugging Face    │
│ + pgvector      │    │  Pro Notebook    │    │ AutoTrain       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        ▲                       │                      │
        └───────────────────────┴──────────────────────┘
                    Predictions written back
```

### Technology Stack

| Component | Tool | Cost | Why |
|-----------|------|------|-----|
| **Database** | Vercel Postgres (Neon) | Free | Managed serverless Postgres with pgvector 0.8.0 built-in |
| **Vector Store** | pgvector | Free | Single DB for relational + embeddings (up to 2000 dimensions) |
| **Development** | Google Colab Pro | $9.99/mo | T4/V100 GPU access, 24-hour sessions |
| **AutoML** | Hugging Face AutoTrain | $15-25/mo | No-code tabular + NLP, pay-per-use compute |
| **Clinical NLP** | medspaCy + scispaCy | Free | Negation detection, UMLS entity linking |
| **Training Data** | Synthea / MIMIC-IV | Free | Synthea: synthetic; MIMIC-IV: real (credentialed) |

**Total Monthly Cost:** $35-45

## Project Structure

```
billing-model/
├── README.md                          # This file
├── notebooks/                         # Google Colab notebooks (copy-paste ready)
│   ├── 01_setup_database.py          # Database connection & schema setup
│   ├── 02_load_synthea_data.py       # Load synthetic training data
│   ├── 03_outlier_detection.py       # Train Isolation Forest model
│   ├── 04_huggingface_autotrain.py   # XGBoost + AutoTrain workflow
│   └── 05_embeddings_search.py       # Clinical note embeddings & similarity
├── sql/
│   └── schema.sql                     # Database schema definitions
├── docs/
│   ├── ARCHITECTURE.md                # Detailed architecture decisions
│   ├── DATA_SOURCES.md                # Synthea, MIMIC-IV, CMS setup guides
│   └── ROADMAP.md                     # Implementation timeline
└── data/                              # Local data files (gitignored)
```

## Quick Start

### Prerequisites

1. **Vercel Postgres Database**
   - Create free account at [vercel.com](https://vercel.com)
   - Create Postgres database (Neon)
   - Copy direct connection string (NOT pooled)

2. **Hugging Face Account**
   - Create account at [huggingface.co](https://huggingface.co)
   - Generate access token with **write** permissions

3. **Google Colab Pro** (recommended)
   - Subscribe at [colab.research.google.com](https://colab.research.google.com)
   - $9.99/month for GPU access

### Setup Instructions

1. **Configure Colab Secrets** (🔑 sidebar in Colab)
   ```
   VERCEL_POSTGRES_URL = postgresql://user:pass@host/db
   HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxxx
   ```

2. **Run Notebooks in Order**
   - `01_setup_database.py` - Creates tables, enables pgvector
   - `02_load_synthea_data.py` - Downloads & loads synthetic claims
   - `03_outlier_detection.py` - Trains Isolation Forest, writes predictions
   - `04_huggingface_autotrain.py` - XGBoost classifier + HF upload
   - `05_embeddings_search.py` - Embedding generation + similarity search

3. **Verify Pipeline**
   - Check predictions table for outlier scores
   - Test similarity search with sample queries
   - Review model metrics in Hugging Face

## Data Sources

### Synthea (Primary - Free)
- **What:** Synthetic patient histories (encounters, conditions, procedures, claims)
- **Size:** ~1,200 patients in sample dataset
- **Use for:** Initial development, pipeline testing
- **Access:** Pre-generated datasets on AWS Open Data

### MIMIC-IV (Secondary - Credentialed)
- **What:** 546,028 real hospitalizations with ICD codes + clinical notes
- **Requires:** PhysioNet credentialing (~2-3 hours training)
- **Use for:** Clinical note analysis, code suggestion training
- **Demo:** 100-patient sample openly available

### CMS DE-SynPUF
- **What:** Medicare-like claims (~2.3M synthetic beneficiaries)
- **Limitation:** No clinical notes
- **Use for:** Schema testing, claim patterns

## Database Schema

Core tables:
- `claims` - Main billing claims (patient, provider, charges, outlier scores)
- `diagnoses` - ICD-10 diagnosis codes linked to claims
- `procedures` - CPT/HCPCS procedure codes
- `clinical_notes` - Text notes with vector embeddings (384-dim)
- `predictions` - Model outputs (outlier detection, code suggestions)

See [`sql/schema.sql`](sql/schema.sql) for full definitions.

## Implementation Roadmap

### ✅ Week 1: Infrastructure & Connectivity
- [x] Architecture decisions documented
- [ ] Create Vercel Postgres database
- [ ] Test connection from Colab
- [ ] Enable pgvector extension
- [ ] Create schema tables

### Week 2: Data Loading & Basic ML
- [ ] Download Synthea dataset
- [ ] Load data into Vercel Postgres
- [ ] Train outlier detection model (Isolation Forest)
- [ ] Write predictions back to database

### Week 3: AutoTrain & Embeddings
- [ ] Push dataset to Hugging Face Hub
- [ ] Run AutoTrain tabular classification
- [ ] Generate embeddings for clinical notes
- [ ] Implement similarity search

### Week 4+: Clinical Validation
- [ ] Get MIMIC-IV access (PhysioNet credentialing)
- [ ] Implement medspaCy pipeline
- [ ] Fine-tune ClinicalBERT for ICD prediction
- [ ] Build billing code validation pipeline

## Key Features

### 1. Outlier Detection
- Isolation Forest algorithm
- Features: charge amounts, payment ratios, diagnosis/procedure counts
- 5% contamination threshold
- Scores stored in `claims.outlier_score`

### 2. Clinical Validity Cross-Check
- Semantic similarity between clinical notes and billing codes
- all-MiniLM-L6-v2 embeddings (384 dimensions)
- Cosine similarity threshold: 0.4
- pgvector for fast nearest-neighbor search

### 3. Code Suggestion
- XGBoost classifier for tabular features
- Future: Fine-tuned ClinicalBERT for text-based prediction
- Outputs: Top-k ICD-10/CPT codes with confidence scores

## Open Questions

1. **MIMIC-IV Access:** Prioritize PhysioNet credentialing early, or fully develop pipeline with Synthea first?
2. **Vector DB Scaling:** At what point switch from pgvector to dedicated vector DB (Pinecone/Qdrant)?
3. **Code Granularity:** Predict ICD-10 at category level (3-char) or full code (5-7 char)?
4. **Ground Truth:** How to establish what constitutes a 'true' outlier for evaluation?

## Contributing

This is an experimental project. Current focus areas:

1. Improve outlier detection with better features
2. Implement clinical NLP with medspaCy
3. Build validation dashboard for predictions
4. Fine-tune transformer models for ICD code prediction

## Resources

- [Synthea Documentation](https://synthetichealth.github.io/synthea/)
- [MIMIC-IV Documentation](https://mimic.mit.edu/)
- [Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [medspaCy](https://github.com/medspacy/medspacy)

## License

MIT (for code) - Note: MIMIC-IV data has separate licensing requirements

---

**Status:** 🚧 Active Development | **Last Updated:** 2025-12-11
