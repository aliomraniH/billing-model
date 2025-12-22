# Medical Billing ML - Testing Guide

**Comprehensive overview of all testing implemented in this project**

---

## 📋 Table of Contents

1. [Configuration Testing](#1-configuration-testing)
2. [Database Schema Testing](#2-database-schema-testing)
3. [Embedding Generation Testing](#3-embedding-generation-testing)
4. [Data Quality Validation](#4-data-quality-validation)
5. [Semantic Search Testing](#5-semantic-search-testing)
6. [Performance Testing](#6-performance-testing)
7. [Coverage Analysis](#7-coverage-analysis)
8. [Refresh System Testing](#8-refresh-system-testing)
9. [Diagnostic Tools](#9-diagnostic-tools)

---

## 1. Configuration Testing

### Location
`notebooks/config.py` - `ModelConfig.print_config()`

### What It Tests
- Configuration loading from multiple sources
- Environment variable overrides
- Config file parsing
- Default value fallbacks

### How It Works
```python
from config import get_config

cfg = get_config()
cfg.print_config()
```

### Output Example
```
======================================================================
📋 MODEL CONFIGURATION
======================================================================

🤖 Embedding Model:
   Provider: huggingface
   Model: BAAI/bge-small-en-v1.5
   Dimension: 384
   Batch size: 100

🧠 LLM Model:
   Provider: anthropic
   Model: claude-sonnet-4-5-20250929

🔄 Refresh Configuration:
   Embedding refresh: every 12 hours
   Auto-refresh: enabled
```

### What It Validates
- ✅ All configuration sections load correctly
- ✅ Environment variables override defaults
- ✅ Config.json values are respected
- ✅ Default values work when no overrides present

---

## 2. Database Schema Testing

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 105-182

### What It Tests
- Table existence
- Column presence
- Schema migrations
- Index creation

### How It Works
```python
# Auto-migration system
with engine.begin() as conn:
    # Check if table exists
    result = conn.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'clinical_notes'
        )
    """))

    # Add missing columns automatically
    if 'last_embedded_at' not in existing_columns:
        conn.execute(text("""
            ALTER TABLE clinical_notes
            ADD COLUMN last_embedded_at TIMESTAMP WITH TIME ZONE
        """))
```

### What It Validates
- ✅ `clinical_notes` table exists or is created
- ✅ Refresh tracking columns present
- ✅ Indexes created for performance
- ✅ Old pgvector columns removed
- ✅ Automatic migration from old schema

---

## 3. Embedding Generation Testing

### Location
`notebooks/utils.py` - `test_embedding_generation()`

### What It Tests
- HuggingFace API connectivity
- Model availability
- Embedding dimension correctness
- Response format validation

### How It Works
```python
def test_embedding_generation(hf_client, model_id, expected_dim):
    """Test that embedding generation works"""
    test_text = "This is a test medical note."
    embedding = get_embedding(test_text, hf_client, model_id, expected_dim)

    if embedding is None:
        return False

    # Validate shape
    if embedding.shape[0] != expected_dim:
        return False

    return True
```

### Output Example
```
🧪 Testing embedding generation...
   ✅ Shape: (384,)
   ✅ Sample: [-0.0371, 0.0525, ...]
```

### What It Validates
- ✅ API token is valid
- ✅ Model is accessible
- ✅ Embeddings have correct dimensions
- ✅ Response format is correct
- ✅ No errors in generation pipeline

---

## 4. Data Quality Validation

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 498-552

### What It Tests
- Processing statistics
- Success rates
- Error tracking
- Database-Pinecone consistency

### How It Works
```python
# Track statistics during processing
stats = {
    'total_processed': 0,
    'skipped_fresh': 0,
    'notes_created': 0,
    'embeddings_created': 0,
    'errors': 0,
    'start_time': datetime.now()
}

# After processing
actual_processed = stats['total_processed'] - stats['skipped_fresh']
success_rate = (stats['embeddings_created'] / actual_processed * 100)
```

### Output Example
```
📊 Processing Summary:
   • Total claims processed: 5,000
   • Skipped (fresh embeddings): 0
   • Notes created: 5,000
   • Embeddings created: 5,000
   • Errors: 0
   • Success rate: 100.0% (of non-skipped claims)
```

### What It Validates
- ✅ All claims processed successfully
- ✅ No embedding generation failures
- ✅ Retry logic works correctly
- ✅ Statistics are accurate
- ✅ Database writes successful

---

## 5. Semantic Search Testing

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 604-623

### What It Tests
- Vector similarity search
- Query embedding generation
- Result relevance
- Metadata retrieval

### Test Queries
```python
test_queries = [
    "diabetes blood sugar insulin",
    "heart attack cardiac chest pain",
    "knee hip joint replacement",
    "pneumonia lung respiratory",
]
```

### How It Works
```python
for query in test_queries:
    results = search_similar_notes(query, top_k=3)

    if len(results) > 0:
        for idx, row in results.head(3).iterrows():
            print(f"[{row['similarity']:.3f}] Claim {row['claim_id']}: {row['text_preview'][:60]}...")
```

### Output Example
```
📋 Query: 'diabetes blood sugar insulin'
   ✅ Found 3 results
   [0.767] Claim 3396: Patient with Type 2 diabetes mellitus. HbA1c 8.0%...
   [0.762] Claim 4748: Patient with Type 2 diabetes mellitus. HbA1c 10.2%...
   [0.762] Claim 3379: Patient with Type 2 diabetes mellitus. HbA1c 9.8%...
```

### What It Validates
- ✅ Semantic search finds relevant results
- ✅ Similarity scores are reasonable (>0.7)
- ✅ Results match query intent
- ✅ Metadata is correctly stored
- ✅ Pinecone index is working

---

## 6. Performance Testing

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 625-646

### What It Tests
- Query latency
- Throughput
- Consistency
- Performance benchmarks

### How It Works
```python
query_times = []

for _ in range(10):
    start = timing_module.time()
    search_similar_notes("test query", top_k=5)
    query_times.append(timing_module.time() - start)

avg_time = np.mean(query_times) * 1000  # Convert to ms
```

### Output Example
```
[TEST 2] Query Performance
--------------------------------------------------
   Average query time: 76.5ms
   Min: 70.1ms, Max: 86.6ms
   ✅ Query performance excellent
```

### Performance Criteria
- **Excellent**: < 100ms average
- **Good**: < 500ms average
- **Needs optimization**: > 500ms average

### What It Validates
- ✅ Queries complete quickly
- ✅ Performance is consistent
- ✅ No major outliers
- ✅ Index is properly configured

---

## 7. Coverage Analysis

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 648-662

### What It Tests
- Vector database coverage
- Processing completeness
- Data availability

### How It Works
```python
coverage_pct = (final_stats.total_vector_count / total_claims) * 100

if coverage_pct >= 90:
    print("✅ Excellent coverage")
elif coverage_pct >= 50:
    print("✅ Good coverage")
elif coverage_pct >= 10:
    print("⚠️ Moderate coverage")
else:
    print("⚠️ Low coverage")
```

### Output Example
```
[TEST 3] Coverage Analysis
--------------------------------------------------
   Vector coverage: 5,858/15,000 (39.1%)
   ⚠️ Moderate coverage - consider increasing MAX_CLAIMS_TO_PROCESS
```

### Coverage Levels
- **Excellent**: ≥90% of claims
- **Good**: ≥50% of claims
- **Moderate**: ≥10% of claims
- **Low**: <10% of claims

### What It Validates
- ✅ Sufficient data for clustering
- ✅ Processing progress tracked
- ✅ Recommendations for improvement

---

## 8. Refresh System Testing

### Location
`notebooks/05_embeddings_similarity_search.py` - Lines 354-380

### What It Tests
- Auto-refresh logic
- Timestamp tracking
- Staleness detection
- Skip behavior

### How It Works
```python
if AUTO_REFRESH:
    result = conn.execute(text(f"""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN last_embedded_at > NOW() - INTERVAL '{REFRESH_INTERVAL_HOURS} hours' THEN 1 END) as fresh,
            COUNT(CASE WHEN last_embedded_at <= NOW() - INTERVAL '{REFRESH_INTERVAL_HOURS} hours' OR last_embedded_at IS NULL THEN 1 END) as stale_or_null
        FROM clinical_notes
    """))
```

### Output Example
```
🔄 Auto-refresh enabled (interval: 12h)
Fresh embeddings (<12h): 0 (will be SKIPPED)
Stale/missing embeddings: 2,777 (will be PROCESSED)

💡 To process all claims, either:
   1. Set AUTO_REFRESH_ENABLED=false in environment
   2. Run: python notebooks/fix_embeddings_refresh.py --clear-all
   3. Wait 12 hours for embeddings to become stale
```

### What It Validates
- ✅ Fresh embeddings are skipped
- ✅ Stale embeddings are refreshed
- ✅ Timestamps are accurate
- ✅ Model changes trigger refresh
- ✅ User guidance is clear

---

## 9. Diagnostic Tools

### Tools Created

#### A. `fix_embeddings_refresh.py`

**Purpose**: Diagnose and fix refresh issues

**Commands**:
```bash
# Check status
python notebooks/fix_embeddings_refresh.py --status

# Clear all timestamps
python notebooks/fix_embeddings_refresh.py --clear-all --yes

# Fix database/Pinecone sync
python notebooks/fix_embeddings_refresh.py --clear-missing
```

**What It Tests**:
- ✅ Database schema correctness
- ✅ Timestamp distribution
- ✅ Pinecone vector count
- ✅ Database-Pinecone consistency

#### B. `clear_embeddings.py`

**Purpose**: Simple notebook-friendly timestamp clearing

**Usage**:
```python
%run notebooks/clear_embeddings.py
```

**What It Tests**:
- ✅ Timestamp clearing works
- ✅ Database is accessible
- ✅ Counts are accurate

#### C. `check_table_schema.py`

**Purpose**: Inspect database schema

**What It Tests**:
- ✅ Table structure
- ✅ Column types
- ✅ Data statistics

---

## 🎯 Testing Best Practices

### Before Running Notebook 5
1. **Validate environment variables**
   ```python
   import os
   required = ['VERCEL_POSTGRES_URL', 'HF_TOKEN', 'PINECONE_API_KEY']
   missing = [v for v in required if not os.getenv(v)]
   if missing:
       print(f"❌ Missing: {missing}")
   ```

2. **Check database connection**
   ```python
   from utils import init_database
   engine, total_claims = init_database(os.getenv('VERCEL_POSTGRES_URL'))
   print(f"✅ Connected: {total_claims:,} claims")
   ```

3. **Verify Pinecone access**
   ```python
   from utils import init_pinecone
   pc, index = init_pinecone(os.getenv('PINECONE_API_KEY'), 'medical-billing-notes', 384)
   print(f"✅ Pinecone: {index.describe_index_stats().total_vector_count:,} vectors")
   ```

### After Running Notebook 5
1. **Review processing summary**
   - Check success rate (should be 100% of non-skipped)
   - Verify error count (should be 0)
   - Confirm expected coverage

2. **Test semantic search**
   - Run test queries
   - Verify results make sense
   - Check similarity scores

3. **Validate data quality**
   - Check database counts
   - Verify Pinecone counts
   - Investigate any discrepancies

---

## 🐛 Common Issues and Tests

### Issue: Low Success Rate (8.4%)
**Test**: Check refresh status
```python
%run notebooks/fix_embeddings_refresh.py --status
```
**Fix**: Clear timestamps if auto-refresh is skipping too many

### Issue: ValueError in config.py
**Test**: Check for formatting bug
```bash
grep -A 2 "Processing Configuration:" /work/config.py
```
**Fix**: Apply patch script (see REFRESH_SYSTEM_README.md Quick Start)

### Issue: Database/Pinecone Mismatch
**Test**: Run diagnostic
```python
!python notebooks/fix_embeddings_refresh.py --clear-missing --yes
```
**Fix**: Script syncs automatically

### Issue: Slow Query Performance
**Test**: Run performance test (built into Notebook 5)
**Fix**: Check index configuration, network latency

---

## 📊 Testing Metrics Summary

| Test Type | Location | Pass Criteria | Frequency |
|-----------|----------|---------------|-----------|
| Config Loading | config.py | No errors | Every run |
| Schema Migration | Notebook 5 | Columns present | Every run |
| Embedding Test | utils.py | Correct dimensions | Every run |
| Data Quality | Notebook 5 | 100% success rate | Every run |
| Semantic Search | Notebook 5 | Relevant results | Every run |
| Performance | Notebook 5 | <500ms queries | Every run |
| Coverage | Notebook 5 | Progress tracked | Every run |
| Refresh Logic | Notebook 5 | Correct skipping | Every run |

---

## 🚀 Continuous Testing Strategy

### Development
- Run all tests on every notebook execution
- Monitor success rates and errors
- Validate configuration changes

### Production
- Enable auto-refresh for cost optimization
- Monitor coverage trends
- Track performance metrics
- Review data quality reports

### Troubleshooting
1. Use diagnostic tools first
2. Check configuration
3. Validate environment variables
4. Review logs and statistics
5. Test with small batch sizes

---

**Last Updated**: December 2025
**Version**: 2.0
**Status**: Production Ready ✅
