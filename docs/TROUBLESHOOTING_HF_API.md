# HuggingFace Inference API Troubleshooting

## Issue: HTTP 410 (Gone) Error

### Symptom
When running verification, you see:
```
⚠️  API response: 410
```

### Root Cause
The `NeuML/bioclinical-modernbert-base-embeddings` model may not support the HuggingFace Inference API's feature extraction endpoint. This can happen because:

1. **Model Type Incompatibility**: Not all embedding models support the Inference API
2. **API Endpoint Changes**: Some models require different API endpoints
3. **Model Deprecation**: The model might have been moved or archived

### Why This Happens

HuggingFace Inference API has specific requirements:
- Model must have a supported task type (feature-extraction, sentence-similarity, etc.)
- Model must be public and actively maintained
- Model must not be gated or require special permissions

ModernBERT models (released recently) may not yet be fully integrated with the Inference API infrastructure.

### Solution Options

#### Option 1: Use Alternative Medical Embedding Models (Recommended)

Replace the model in `config/settings.py` with one of these Inference API-compatible medical models:

**Best Options:**
```python
# Option A: BioMedBERT (NCBI/NLM)
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
dimension = 768

# Option B: Clinical BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.2"
dimension = 768

# Option C: SapBERT (semantic search optimized)
model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
dimension = 768
```

**To change the model:**
1. Edit `config/settings.py`
2. Update the `embedding_config` dataclass:
   ```python
   @dataclass
   class EmbeddingConfig:
       model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
       dimension: int = 768
   ```
3. Re-run the verification notebook

#### Option 2: Use Sentence Transformers API Endpoint

Some models work better with the sentence-transformers API format:

Update `src/data/hf_embeddings.py`:
```python
# In HuggingFaceEmbeddings.__init__
# Change from feature-extraction endpoint to pipeline endpoint
self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
```

#### Option 3: Verify Model Availability

Check if the model supports Inference API:
```bash
curl https://api-inference.huggingface.co/models/NeuML/bioclinical-modernbert-base-embeddings \
  -H "Authorization: Bearer YOUR_HF_TOKEN"
```

Look for `"pipeline_tag": "feature-extraction"` in the response.

### Testing the Fix

After changing the model, test with:
```python
from src.data.hf_embeddings import get_embeddings_model

model = get_embeddings_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
embedding = model.encode("Patient with diabetes")
print(f"Embedding shape: {embedding.shape}")  # Should be (768,)
```

### Expected Behavior After Fix

```
🤖 HuggingFace Inference API:
------------------------------------------------------------
  ✅ HF_TOKEN configured
  ✅ HuggingFace API accessible
  ✅ Medical embedding model available
```

### Additional Resources

- [HF Inference API Docs](https://huggingface.co/docs/api-inference/index)
- [Supported Tasks](https://huggingface.co/docs/api-inference/detailed_parameters#feature-extraction-task)
- [Medical NLP Models](https://huggingface.co/models?pipeline_tag=feature-extraction&other=clinical)

### Performance Comparison

| Model | Dimensions | Medical Accuracy | Inference API Support |
|-------|-----------|------------------|----------------------|
| BioClinical ModernBERT | 768 | Excellent | ❌ Not yet |
| BiomedNLP-PubMedBERT | 768 | Excellent | ✅ Yes |
| BioLinkBERT | 768 | Very Good | ✅ Yes |
| SapBERT | 768 | Excellent | ✅ Yes |
| BioBERT v1.2 | 768 | Very Good | ✅ Yes |

**Recommendation**: Use `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` for best compatibility and medical accuracy.

### Why This Won't Break Your System

The 410 error only affects:
- Initial model testing
- First-time embedding generation

It does NOT affect:
- Database connectivity
- NLP entity extraction (medspaCy)
- Code matching (once embeddings are generated)
- Other system components

The system will work once you switch to a compatible model!
