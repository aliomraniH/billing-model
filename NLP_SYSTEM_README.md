# NLP Clinical Code Suggestion System

## Overview

The NLP Clinical Code Suggestion System adds intelligent medical billing code extraction and validation capabilities to the billing-model platform. It uses state-of-the-art clinical NLP models to automatically extract ICD-10 and HCPCS codes from clinical notes, validate billed codes against documentation, and identify revenue leakage or compliance risks.

## Architecture

### Cloud-Native Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Deepnote Notebooks                        │
│  00_verify_deps → 07_setup_kb → 08_extract → 09_analyze     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Core NLP Components                        │
│  • ClinicalCodeExtractor (medspaCy + BioClinical BERT)      │
│  • ClusterConsensus (weak supervision)                       │
│  • GapAnalyzer (revenue leakage detection)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Vercel Postgres (Neon) + pgvector              │
│  • code_embeddings (768-dim vectors, HNSW index)            │
│  • nlp_extractions (ClinicalCodeObject JSONs)                │
│  • gap_analysis_results (validation reports)                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **ClinicalCodeObject** (`src/features/clinical_code_object.py`)
   - Core data structure for NLP-derived billing intelligence
   - Contains extracted codes, suggestions, and validation flags
   - Serializable to JSON for database storage

2. **ClinicalCodeExtractor** (`src/features/nlp_extractor.py`)
   - Uses medspaCy for entity extraction
   - BioClinical ModernBERT (768-dim) for semantic similarity
   - Negation detection via ConText
   - Maps clinical terms → ICD-10/HCPCS codes via pgvector search

3. **ClusterConsensus** (`src/features/cluster_consensus.py`)
   - Weak supervision algorithm
   - Identifies consensus codes within patient clusters
   - Suggests missing codes based on similar claims

4. **GapAnalyzer** (`src/features/gap_analyzer.py`)
   - Compares NLP-extracted codes vs billed codes
   - Flags revenue leakage (documented but not billed)
   - Flags compliance risks (billed but not documented)

## Setup & Installation

### 1. Environment Variables

Configure in Deepnote Project Settings:

```bash
VERCEL_POSTGRES_URL=postgresql://...@...neon.tech/...  # Required
HF_TOKEN=hf_...  # Optional (for private models)
```

### 2. Install Dependencies

**Automated Setup (Recommended):**
```bash
python scripts/setup_nlp_system.py
```

**Manual Setup:**
```bash
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### 3. (Optional) Enable Cloud Caching

For production deployments on Vercel/Replit, enable model caching to reduce cold start times:

```bash
# Enable Vercel Blob and KV
vercel blob create nlp-models
vercel kv create nlp-cache

# Pull environment variables
vercel env pull

# Warm cache after deployment
python scripts/deploy_warm_cache.py
```

**See [docs/CLOUD_STORAGE.md](docs/CLOUD_STORAGE.md) for detailed cloud caching setup.**

### 4. Run Notebooks in Order

```bash
# Verify dependencies
python notebooks/00_verify_nlp_dependencies.py

# One-time setup (downloads codes, builds embeddings)
python notebooks/07_nlp_knowledge_base_setup.py

# Extract codes from claims
python notebooks/08_nlp_code_extraction.py

# Analyze gaps and generate reports
python notebooks/09_gap_analysis_reporting.py
```

## Usage Examples

### Extract Codes from Clinical Notes

```python
from src.features.nlp_extractor import ClinicalCodeExtractor
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")
extractor = ClinicalCodeExtractor(db_connection=engine)

# Process single claim
clinical_obj = extractor.process_claim(
    claim_id="CLM-12345",
    clinical_text="Patient with type 2 diabetes mellitus and hypertension..."
)

print(f"Extracted diagnoses: {len(clinical_obj.billable_diagnoses)}")
for diag in clinical_obj.billable_diagnoses:
    print(f"  {diag.code}: {diag.term} (confidence: {diag.confidence:.2%})")
```

### Cluster Consensus Suggestions

```python
from src.features.cluster_consensus import ClusterConsensus

consensus = ClusterConsensus(consensus_threshold=0.70)

# Build profiles from historical data
consensus.build_cluster_profiles(df, cluster_column='cluster_id')

# Get suggestions for a claim
suggestions = consensus.get_suggestions(
    claim_codes=['E11.9', 'I10'],
    cluster_id=3
)

for s in suggestions:
    print(f"{s['code']}: {s['rationale']}")
```

### Gap Analysis

```python
from src.features.gap_analyzer import GapAnalyzer

analyzer = GapAnalyzer(db_connection=engine)

result = analyzer.analyze_claim(
    clinical_object=clinical_obj,
    billed_codes=['E11.9']  # Only diabetes billed
)

print(f"Risk Score: {result.risk_score:.2f}")
print(f"Revenue Impact: ${result.revenue_impact:,.0f}")
for flag in result.flags:
    print(f"  {flag['type']}: {flag['message']}")
```

## Data Model

### code_embeddings Table

```sql
CREATE TABLE code_embeddings (
    code_id VARCHAR(20) PRIMARY KEY,
    code_type VARCHAR(10) NOT NULL,  -- 'ICD-10' or 'HCPCS'
    short_description TEXT,
    long_description TEXT,
    is_billable BOOLEAN DEFAULT TRUE,
    embedding vector(768)  -- BioClinical ModernBERT
);

CREATE INDEX idx_code_embeddings_hnsw
ON code_embeddings USING hnsw (embedding vector_cosine_ops);
```

### nlp_extractions Table

```sql
CREATE TABLE nlp_extractions (
    claim_id VARCHAR(50) PRIMARY KEY,
    extraction_timestamp TIMESTAMP DEFAULT NOW(),
    extraction_json JSONB,  -- ClinicalCodeObject
    diagnosis_count INT,
    procedure_count INT
);
```

### gap_analysis_results Table

```sql
CREATE TABLE gap_analysis_results (
    claim_id VARCHAR(50) PRIMARY KEY,
    analysis_timestamp TIMESTAMP DEFAULT NOW(),
    flag_count INT,
    revenue_leakage_flags INT,
    compliance_risk_flags INT,
    revenue_impact FLOAT,
    risk_score FLOAT,
    flags_json JSONB
);
```

## Performance

- **Code Embedding**: ~100 codes/second (BioClinical BERT)
- **Entity Extraction**: ~50 claims/minute (medspaCy)
- **Similarity Search**: <5ms per query (pgvector HNSW)
- **Batch Processing**: 1000 claims in ~2-5 minutes

## Compliance & Security

### HIPAA Compliance

- De-identification via Microsoft Presidio (optional, enable in config)
- No PHI in model training (pre-trained models only)
- Encrypted at rest (Vercel Postgres default)

### Code Sources

- **ICD-10-CM**: Public domain (CMS)
- **HCPCS Level II**: Public domain (CMS)
- **No CPT codes**: Avoided due to AMA licensing requirements

## Troubleshooting

### Issue: "No module named 'medspacy'"

```bash
# Run dependency verification
python notebooks/00_verify_nlp_dependencies.py

# Install missing packages
pip install medspacy scispacy
```

### Issue: "Database connection failed"

```bash
# Check environment variable
echo $VERCEL_POSTGRES_URL

# Set in Deepnote: Project Settings → Environment Variables
```

### Issue: "pgvector extension not found"

```sql
-- Run in Vercel Postgres console
CREATE EXTENSION IF NOT EXISTS vector;
```

### Issue: "Embedding model download slow"

- BioClinical BERT is ~400MB, first download takes ~2-5 minutes
- Uses HuggingFace cache for subsequent runs
- Optional: Set HF_TOKEN for faster downloads

## Limitations

1. **Context Window**: 8K tokens (BioClinical BERT) - adequate for most clinical notes
2. **Languages**: English only (scispaCy limitation)
3. **Code Systems**: ICD-10-CM and HCPCS only (no CPT, SNOMED-CT)
4. **Accuracy**: ~85-90% confidence threshold recommended
5. **Cold Start**: First run downloads models (~500MB total)

## Roadmap

- [ ] Add CPT code support (requires AMA licensing)
- [ ] Implement SNOMED-CT mapping
- [ ] Add multi-language support
- [ ] Real-time API endpoint
- [ ] Feedback loop for model improvement
- [ ] Integration with existing claim review workflow

## References

- **medspaCy**: https://github.com/medspacy/medspacy
- **scispaCy**: https://allenai.github.io/scispacy/
- **BioClinical BERT**: https://huggingface.co/NeuML/bioclinical-modernbert-base-embeddings
- **pgvector**: https://github.com/pgvector/pgvector
- **CMS ICD-10**: https://www.cms.gov/medicare/coding-billing/icd-10-codes
- **CMS HCPCS**: https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system

## Support

For issues or questions:
1. Check `notebooks/00_verify_nlp_dependencies.py` output
2. Review logs in Deepnote notebook outputs
3. Consult the troubleshooting section above
4. Open an issue in the repository
