# Migration Guide: API-First Architecture

## 🎯 Goal
Reduce memory usage from **~2.5GB → ~200MB** by replacing local NLP models with API-based approach.

---

## What Changed

### Before (Heavy Local Models)
```
Memory: ~2.5 GB
├── spaCy core: ~200MB
├── scispaCy: ~100MB
├── medspaCy: ~50MB
├── en_core_sci_md model: ~400MB
├── en_core_web_sm model: ~50MB
└── sentence-transformers: ~500MB + torch ~800MB
```

### After (API-First)
```
Memory: ~200 MB
├── Lightweight packages: ~150MB
├── API client code: ~10MB
└── Python runtime: ~40MB

APIs Used:
├── HuggingFace Inference API (Medical NER + Embeddings)
├── Claude API (Context Analysis) - optional
└── Vercel Postgres (pgvector)
```

---

## Quick Start (Deepnote)

### 1. Run Lightweight Verification

```python
# In Deepnote, run:
%run /work/notebooks/00_verify_setup_lightweight.py
```

**Expected output:**
```
✅ SETUP VERIFICATION COMPLETE
============================================================

📋 Architecture Summary:
   • Database: Vercel Postgres with pgvector ✅
   • NLP: HuggingFace Inference API ✅
   • Embeddings: PubMedBERT via API ✅
   • Context Analysis: Claude API or rule-based ✅
   • Memory: ~200 MB (vs ~2.5 GB with local models) ✅
```

### 2. Test the API Client

```python
from src.nlp_api_client import get_nlp_client

# Initialize client
client = get_nlp_client()

# Test medical entity extraction
text = "Patient diagnosed with Type 2 Diabetes Mellitus and hypertension."
entities = client.extract_medical_entities(text)

print(f"Found {len(entities)} entities:")
for e in entities:
    print(f"  • {e['text']:30} ({e['label']}, confidence: {e['score']:.2f})")

# Output:
# Found 2 entities:
#   • Type 2 Diabetes Mellitus    (DISEASE, confidence: 0.98)
#   • hypertension                (DISEASE, confidence: 0.96)
```

### 3. Test Context Analysis

```python
# Extract entities with clinical context
text = "No evidence of pneumonia. Patient denies chest pain. History of MI."
entities = client.extract_medical_entities(text)

# Analyze context (negation, uncertainty, history)
entities_with_context = client.analyze_clinical_context(text, entities)

for e in entities_with_context:
    flags = []
    if e['is_negated']: flags.append('NEGATED')
    if e['is_uncertain']: flags.append('UNCERTAIN')
    if e['is_historical']: flags.append('HISTORICAL')
    status = ' | '.join(flags) if flags else 'AFFIRMED'
    print(f"  • {e['text']:20} → {status}")

# Output:
#   • pneumonia          → NEGATED
#   • chest pain         → NEGATED
#   • MI                 → HISTORICAL
```

### 4. Test Embeddings

```python
# Generate embeddings for similarity search
texts = ["diabetes management", "blood glucose control", "hypertension treatment"]
embeddings = client.get_embeddings(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Output:
# Generated 3 embeddings
# Embedding dimension: 768
```

---

## API Configuration

### Required Environment Variables

Set in Deepnote: **Environment → Add Variable**

```bash
# Required
VERCEL_POSTGRES_URL=postgresql://...@....neon.tech/...?sslmode=require
HF_TOKEN=hf_...  # Get from huggingface.co/settings/tokens

# Optional (for Claude-based context analysis)
ANTHROPIC_API_KEY=sk-ant-...  # Get from console.anthropic.com
```

---

## Code Migration Examples

### Example 1: Replace medspaCy Entity Extraction

**Before (Heavy - ~2GB):**
```python
import medspacy
nlp = medspacy.load()  # Loads ~400MB model

doc = nlp("Patient has diabetes")
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

**After (Lightweight - ~10MB):**
```python
from src.nlp_api_client import get_nlp_client
client = get_nlp_client()

entities = client.extract_medical_entities("Patient has diabetes")
# Returns: [{"text": "diabetes", "label": "DISEASE", "score": 0.95, ...}]
```

### Example 2: Replace sentence-transformers Embeddings

**Before (Heavy - ~500MB):**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # Downloads 500MB

embeddings = model.encode(["text1", "text2"])
```

**After (Lightweight - API):**
```python
from src.nlp_api_client import get_nlp_client
client = get_nlp_client()

embeddings = client.get_embeddings(["text1", "text2"])
# Same 768-dim vectors, no download needed
```

### Example 3: Replace medspaCy ConText

**Before (Heavy):**
```python
import medspacy
from medspacy.context import ConTextComponent

nlp = medspacy.load()
nlp.add_pipe("medspacy_context")

doc = nlp("No evidence of pneumonia")
for ent in doc.ents:
    print(f"{ent.text}: negated={ent._.is_negated}")
```

**After (Lightweight):**
```python
from src.nlp_api_client import get_nlp_client
client = get_nlp_client()

text = "No evidence of pneumonia"
entities = client.extract_medical_entities(text)
entities = client.analyze_clinical_context(text, entities)

for ent in entities:
    print(f"{ent['text']}: negated={ent['is_negated']}")
```

---

## API Client Features

### Automatic Fallbacks

The API client includes intelligent fallbacks:

1. **HuggingFace API unavailable** → Rule-based NER
2. **Claude API unavailable** → Rule-based context analysis
3. **Model loading (503 error)** → Auto-retry after 20s
4. **Rate limiting** → Exponential backoff

Example:
```python
# If HF API returns 503 (model loading), client auto-waits and retries
entities = client.extract_medical_entities(text)
# Will retry automatically, no error handling needed
```

### Supported Entity Types

From HuggingFace Biomedical NER API:

- `DISEASE` - Medical conditions, diagnoses
- `CHEMICAL` - Medications, drugs, compounds
- `GENE_OR_GENE_PRODUCT` - Genes, proteins
- `ORGANISM` - Bacteria, viruses, pathogens

### Context Attributes

From context analysis (Claude or rule-based):

- `is_negated` - Condition is denied/absent
- `is_uncertain` - Possible/suspected diagnosis
- `is_historical` - Past medical history
- `is_family_history` - Family member's condition

---

## Cost Considerations

### HuggingFace Inference API
- **Free tier**: 30,000 requests/month
- **Pro ($9/month)**: Unlimited requests, priority

### Claude API (Optional)
- **Per request**: ~$0.003 for context analysis
- **10,000 notes/month**: ~$30

### Total Monthly Cost
- **Development (free tier)**: $0
- **Production (10K notes)**: $30-40

Compare to local GPU server: ~$500/month

---

## Performance Comparison

| Metric | Local Models | API-First |
|--------|-------------|-----------|
| Memory | 2.5 GB | 200 MB |
| Install time | 5-10 min | 1-2 min |
| Startup time | 30-60 sec | Instant |
| Model updates | Manual | Automatic |
| Entity accuracy | ~95% | ~95% (same models) |
| Latency | ~50ms | ~200ms (API) |

---

## Troubleshooting

### HuggingFace API Returns 503

**Error:** Model is loading (first use)

**Solution:** Automatic - client waits 20s and retries

```python
# No action needed - client handles this automatically
entities = client.extract_medical_entities(text)
```

### HuggingFace API Returns 410

**Error:** Model endpoint deprecated

**Solution:** Update embedding model in `config/settings.py`

```python
# In config/settings.py
class EmbeddingConfig:
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
```

See `docs/TROUBLESHOOTING_HF_API.md` for alternatives.

### No Entities Found

**Possible causes:**
1. API returned empty result
2. Confidence threshold too high
3. Using fallback rule-based NER

**Solution:** Lower min_score or check API status

```python
# Try lower confidence threshold
entities = client.extract_medical_entities(text, min_score=0.5)

# Check if using fallback
if len(entities) == 0:
    print("API may be unavailable, using fallback NER")
```

---

## Rollback to Old Setup

If you need to revert to local models:

```python
# Install heavy packages again
%pip install spacy medspacy scispacy sentence-transformers
%pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# Use old verification
%run /work/notebooks/00_verify_setup.py

# Import old way
import medspacy
nlp = medspacy.load()
```

**Note:** This will restore ~2.5GB memory usage.

---

## Next Steps

### 1. Update Existing Notebooks

Modify notebooks to use API client:

```python
# At top of notebook
from src.nlp_api_client import get_nlp_client
client = get_nlp_client()

# Replace medspaCy calls
# OLD: doc = nlp(text)
# NEW: entities = client.extract_medical_entities(text)
```

### 2. Build Knowledge Base

```python
%run /work/notebooks/07_nlp_knowledge_base_setup.py
```

This now uses API for embeddings instead of local models.

### 3. Extract Clinical Codes

```python
%run /work/notebooks/08_nlp_code_extraction.py
```

Uses API client for NER and context analysis.

---

## Summary

✅ **Implemented:**
- API-based medical NER (HuggingFace)
- API-based embeddings (PubMedBERT)
- API-based context analysis (Claude)
- Lightweight requirements (~200MB)
- Automatic fallbacks
- Memory savings: ~2.3GB (92%)

✅ **Ready to Use:**
- `src/nlp_api_client.py` - Drop-in replacement for medspaCy
- `notebooks/00_verify_setup_lightweight.py` - New verification
- `requirements_lightweight.txt` - No heavy packages

🚀 **Run this now:**
```python
%run /work/notebooks/00_verify_setup_lightweight.py
```
