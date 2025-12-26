# Notebook 6 Architecture Improvements
## Production-Ready Processing Strategy

## Current Bottlenecks Analysis

### 1. **Sequential Processing** (Current: ~5-10 minutes)
- ❌ Load 10k vectors → Cluster → Label → DB writes → Pinecone updates (sequential)
- ❌ Single DB connection held open for entire duration → timeout errors
- ❌ No checkpointing → must restart from beginning on failure
- ❌ 7,974 individual Pinecone updates (could be batched)
- ❌ 7,974 individual DB inserts (could be batched)

### 2. **Resource Inefficiency**
- ❌ DB connection open but idle during clustering (2-3 min)
- ❌ DB connection open but idle during LLM API calls
- ❌ Single-threaded LLM labeling (3 sequential API calls, ~9 seconds)

### 3. **Cost Inefficiency**
- ❌ Re-clusters all data even if only new vectors added
- ❌ No caching of LLM labels or cluster assignments
- ❌ Unnecessary API calls on re-runs

---

## Proposed Architecture: Stage-Based Processing

### **Stage 1: Data Loading & Validation** (~30s)
```
┌─────────────────────────────────────┐
│ 1. Load vectors from Pinecone       │
│ 2. Validate data quality            │
│ 3. Cache to disk: vectors.npz       │
│ 4. Close Pinecone connection        │
└─────────────────────────────────────┘
         ↓ Checkpoint saved
```

**Benefits:**
- ✅ Can resume from cached vectors
- ✅ No Pinecone connection held during clustering
- ✅ Fast re-runs during development

### **Stage 2: Clustering** (~2-3 min)
```
┌─────────────────────────────────────┐
│ 1. Load from cache: vectors.npz     │
│ 2. Run HDBSCAN clustering            │
│ 3. Calculate quality metrics         │
│ 4. Cache to disk: clusters.json     │
└─────────────────────────────────────┘
         ↓ Checkpoint saved
```

**Benefits:**
- ✅ CPU-intensive work isolated
- ✅ Can experiment with clustering parameters without re-loading vectors
- ✅ No external connections needed

### **Stage 3: LLM Labeling** (PARALLEL) (~3s with parallelization)
```
┌─────────────────────────────────────┐
│ 1. Load clusters.json                │
│ 2. PARALLEL: Label N clusters        │
│    ├─ Thread 1 → Claude API          │
│    ├─ Thread 2 → Claude API          │
│    └─ Thread 3 → Claude API          │
│ 3. Cache to disk: categories.json   │
└─────────────────────────────────────┘
         ↓ Checkpoint saved
```

**Benefits:**
- ✅ 3x faster with concurrent API calls (Anthropic allows concurrency)
- ✅ Cached labels → skip on re-runs if clusters unchanged
- ✅ Retry logic per cluster (not all-or-nothing)

### **Stage 4: Database Updates** (BATCHED) (~5s)
```
┌─────────────────────────────────────┐
│ 1. Fresh DB connection               │
│ 2. Batch INSERT categories (3 rows)  │
│ 3. Batch INSERT memberships (8k)     │
│    └─ Use executemany() for speed    │
│ 4. Close DB connection               │
└─────────────────────────────────────┘
         ↓ Checkpoint saved
```

**Benefits:**
- ✅ 10-100x faster with batch inserts
- ✅ Fresh connection → no timeout
- ✅ Transactional (rollback on error)

### **Stage 5: Pinecone Metadata Updates** (BATCHED) (~10s)
```
┌─────────────────────────────────────┐
│ 1. Fresh Pinecone connection         │
│ 2. Batch update 100 vectors at a time│
│ 3. Progress tracking & resume support│
│ 4. Close Pinecone connection         │
└─────────────────────────────────────┘
         ↓ Complete
```

**Benefits:**
- ✅ Pinecone batch API more efficient
- ✅ Progress bar for long operations
- ✅ Can resume from checkpoint if interrupted

### **Stage 6: Testing & Validation** (~5s)
```
┌─────────────────────────────────────┐
│ 1. Fresh connections as needed       │
│ 2. Run integration tests             │
│ 3. Generate report                   │
└─────────────────────────────────────┘
```

---

## Parallel Processing Strategy

### What Can Be Parallelized?

#### ✅ **LLM Labeling** (High Impact)
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def label_clusters_parallel(clusters, max_workers=3):
    """Label clusters in parallel (respects API rate limits)"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(label_cluster_with_llm, cluster): cluster
            for cluster in clusters
        }

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    return results
```

**Cost Analysis:**
- Current: 3 sequential calls × 3s = 9s
- Parallel: 3 concurrent calls = 3s
- **Savings: 67% faster, same API cost**

#### ✅ **Pinecone Batch Updates** (High Impact)
```python
def update_pinecone_batched(updates, batch_size=100):
    """Update Pinecone in batches instead of individual calls"""
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        index.upsert(vectors=batch)  # Single API call for 100 updates

        # Progress tracking
        print(f"Progress: {i+len(batch)}/{len(updates)}")
```

**Cost Analysis:**
- Current: 7,974 individual updates
- Batched: 80 batch calls (100 per batch)
- **Savings: 99% fewer API calls**

#### ✅ **Database Batch Inserts** (High Impact)
```python
def insert_memberships_batched(memberships, batch_size=1000):
    """Use executemany() for bulk inserts"""
    with engine.begin() as conn:
        # Single executemany() call for all rows
        conn.execute(
            text("""INSERT INTO claim_category_membership
                    (claim_id, category_id, similarity_score)
                    VALUES (:cid, :cat_id, :sim)
                    ON CONFLICT DO NOTHING"""),
            memberships  # List of dicts
        )
```

**Cost Analysis:**
- Current: 7,974 individual INSERT statements
- Batched: 1 executemany() call
- **Savings: 100x faster**

#### ❌ **Vector Loading** (Not Parallelizable)
- Pinecone API already optimized
- Network-bound, not CPU-bound

#### ❌ **Clustering** (Not Worth It)
- HDBSCAN already uses multiple cores internally
- Overhead of splitting data > benefits

---

## Resource Management: Sleep/Wake Strategy

### Connection Pooling
```python
from contextlib import contextmanager
from sqlalchemy.pool import NullPool

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # No persistent connections
    pool_pre_ping=True,  # Check connection before use
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
    }
)

@contextmanager
def get_db_connection():
    """Context manager for fresh DB connections"""
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()  # Always close when done

# Usage
with get_db_connection() as conn:
    # Do work
    pass
# Connection automatically closed
```

### Lazy Initialization
```python
class ResourceManager:
    """Lazy-load and auto-close resources"""

    def __init__(self):
        self._pinecone = None
        self._db = None
        self._anthropic = None

    @property
    def pinecone(self):
        if self._pinecone is None:
            self._pinecone = init_pinecone(...)
        return self._pinecone

    def close_all(self):
        """Close all open connections"""
        if self._db:
            self._db.close()
        self._pinecone = None
```

---

## Async/Webhook Notification System

### Option 1: Database-Based Job Queue (Simplest)
```python
# Job tracking table
CREATE TABLE processing_jobs (
    job_id SERIAL PRIMARY KEY,
    stage VARCHAR(50),
    status VARCHAR(20),  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_json TEXT,
    error_message TEXT
);

# Job processor
def process_stage(stage_name, stage_func):
    """Run stage and track in database"""
    job_id = create_job(stage_name)

    try:
        update_job(job_id, status='running')
        result = stage_func()
        update_job(job_id, status='completed', result=result)

        # Trigger next stage (or webhook notification)
        trigger_next_stage(stage_name)

    except Exception as e:
        update_job(job_id, status='failed', error=str(e))
        send_alert(f"Stage {stage_name} failed: {e}")
```

**Benefits:**
- ✅ Resume from last successful stage
- ✅ Monitor progress via database queries
- ✅ No external dependencies (Redis, Celery, etc.)

### Option 2: Webhook Notifications (Production)
```python
import requests

def send_webhook(event, data):
    """Send webhook on stage completion"""
    webhook_url = os.getenv('WEBHOOK_URL')
    if webhook_url:
        requests.post(webhook_url, json={
            'event': event,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })

# Usage
def run_stage_2_clustering():
    result = perform_clustering()
    save_checkpoint('clusters.json', result)
    send_webhook('clustering_complete', {
        'n_clusters': result['n_clusters'],
        'next_stage': 'llm_labeling'
    })
```

**Integration with External Systems:**
- Slack notifications
- Email alerts
- Trigger downstream pipelines
- Dashboard updates

---

## Checkpointing & Resume System

### Checkpoint Manager
```python
import json
import hashlib
from pathlib import Path

class CheckpointManager:
    """Manage checkpoints for resumable processing"""

    def __init__(self, cache_dir='.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def save(self, stage_name, data, metadata=None):
        """Save checkpoint with versioning"""
        checkpoint = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'metadata': metadata or {},
            'hash': self._compute_hash(data)
        }

        path = self.cache_dir / f"{stage_name}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f)

        print(f"✅ Checkpoint saved: {stage_name}")

    def load(self, stage_name):
        """Load checkpoint if exists"""
        path = self.cache_dir / f"{stage_name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def has_valid_checkpoint(self, stage_name, max_age_hours=24):
        """Check if valid checkpoint exists"""
        checkpoint = self.load(stage_name)
        if not checkpoint:
            return False

        timestamp = datetime.fromisoformat(checkpoint['timestamp'])
        age = datetime.now() - timestamp

        return age.total_seconds() < (max_age_hours * 3600)

# Usage
cp = CheckpointManager()

# Try to resume from checkpoint
if cp.has_valid_checkpoint('stage_2_clustering'):
    print("📦 Resuming from checkpoint...")
    clusters = cp.load('stage_2_clustering')['data']
else:
    print("🔄 Running clustering...")
    clusters = run_clustering()
    cp.save('stage_2_clustering', clusters, {'n_clusters': len(clusters)})
```

---

## Cost Optimization Strategy

### 1. **Incremental Processing**
```python
def get_new_vectors_since_last_run():
    """Only process new vectors, not entire dataset"""
    last_run = get_last_clustering_timestamp()

    # Query Pinecone for vectors added after last_run
    # (requires metadata field: created_at)
    new_vectors = index.query(
        filter={'created_at': {'$gte': last_run}},
        include_values=True
    )

    return new_vectors
```

**Savings:**
- Initial run: Process 10,000 vectors
- Incremental: Process only 100 new vectors (99% fewer)

### 2. **LLM Caching**
```python
def label_cluster_with_cache(sample_notes, cluster_idx):
    """Cache LLM responses to avoid re-labeling"""
    cache_key = hashlib.md5(''.join(sample_notes).encode()).hexdigest()

    # Check cache
    cached = cache.get(f"llm_label_{cache_key}")
    if cached:
        return cached

    # Call LLM
    result = label_cluster_with_llm(sample_notes, cluster_idx)

    # Save to cache
    cache.set(f"llm_label_{cache_key}", result, ttl=86400)

    return result
```

**Savings:**
- Re-runs: $0 (use cache)
- Only pay for new/changed clusters

### 3. **Smart Batch Sizing**
```python
def adaptive_batch_size(total_items, max_memory_mb=500):
    """Calculate optimal batch size based on available memory"""
    item_size_bytes = 1500  # Estimated size per vector
    items_per_mb = 1024 * 1024 / item_size_bytes

    optimal_size = int(max_memory_mb * items_per_mb)
    return min(optimal_size, 1000)  # Cap at 1000 for API limits
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Fix DB connection timeout** → Use fresh connections
2. ✅ **Batch database inserts** → 100x faster
3. ✅ **Batch Pinecone updates** → 99% fewer API calls

**Impact:** Fix timeouts, 10x faster, no architecture change

### Phase 2: Parallelization (2-3 hours)
1. ✅ **Parallel LLM labeling** → 3x faster
2. ✅ **Progress bars** → Better UX
3. ✅ **Error handling per-cluster** → More resilient

**Impact:** 3x faster LLM stage, better reliability

### Phase 3: Checkpointing (3-4 hours)
1. ✅ **Checkpoint manager** → Resume from failures
2. ✅ **Stage-based execution** → Independent stages
3. ✅ **Cache vector loading** → Fast re-runs

**Impact:** Resumable processing, fast iteration

### Phase 4: Production Ready (1-2 days)
1. ✅ **Job queue system** → Async processing
2. ✅ **Webhook notifications** → Integration
3. ✅ **Incremental updates** → Cost optimization
4. ✅ **Monitoring & alerting** → Observability

**Impact:** Production-grade system, cost efficient

---

## Example: Refactored Notebook Flow

```python
# ============================================================
# STAGE-BASED PROCESSING WITH CHECKPOINTS
# ============================================================

cp = CheckpointManager(cache_dir='notebooks/.cache')
rm = ResourceManager()  # Lazy resource initialization

# STAGE 1: Load vectors
if cp.has_valid_checkpoint('stage_1_vectors', max_age_hours=12):
    print("📦 Loading vectors from checkpoint...")
    vectors_data = cp.load('stage_1_vectors')
else:
    print("📥 Loading vectors from Pinecone...")
    vectors_data = load_and_validate_vectors(rm.pinecone)
    cp.save('stage_1_vectors', vectors_data)
    rm.close('pinecone')  # Close connection

# STAGE 2: Clustering
if cp.has_valid_checkpoint('stage_2_clusters', max_age_hours=24):
    print("📦 Loading clusters from checkpoint...")
    clusters = cp.load('stage_2_clusters')
else:
    print("🔬 Clustering vectors...")
    clusters = run_clustering(vectors_data)
    cp.save('stage_2_clusters', clusters)

# STAGE 3: LLM Labeling (PARALLEL)
if cp.has_valid_checkpoint('stage_3_categories', max_age_hours=24):
    print("📦 Loading categories from checkpoint...")
    categories = cp.load('stage_3_categories')
else:
    print("🤖 Labeling clusters (parallel)...")
    categories = label_clusters_parallel(
        clusters,
        max_workers=3,  # Anthropic allows concurrency
        anthropic_client=rm.anthropic
    )
    cp.save('stage_3_categories', categories)

# STAGE 4: Database Updates (BATCHED)
print("💾 Updating database (batched)...")
with get_db_connection() as conn:
    # Batch insert categories (3 rows)
    insert_categories_batch(conn, categories)

    # Batch insert memberships (8k rows)
    memberships = prepare_memberships(clusters, categories)
    insert_memberships_batch(conn, memberships)

# Connection auto-closed

# STAGE 5: Pinecone Updates (BATCHED)
print("🔄 Updating Pinecone metadata (batched)...")
update_pinecone_metadata_batched(
    rm.pinecone,
    clusters,
    categories,
    batch_size=100,
    show_progress=True
)

# STAGE 6: Testing
print("🧪 Running integration tests...")
run_integration_tests(rm)

print("✅ Complete! All stages finished successfully.")
```

---

## Monitoring & Observability

### Performance Metrics
```python
from time import time

class StageTimer:
    """Track stage execution times"""

    def __init__(self):
        self.times = {}

    def __enter__(self, stage_name):
        self.current_stage = stage_name
        self.start = time()
        return self

    def __exit__(self, *args):
        elapsed = time() - self.start
        self.times[self.current_stage] = elapsed
        print(f"⏱️  {self.current_stage}: {elapsed:.1f}s")

# Usage
timer = StageTimer()

with timer('stage_1_loading'):
    load_vectors()

with timer('stage_2_clustering'):
    run_clustering()

# Report at end
print(f"\n📊 Total time: {sum(timer.times.values()):.1f}s")
for stage, duration in timer.times.items():
    pct = (duration / sum(timer.times.values())) * 100
    print(f"   {stage}: {duration:.1f}s ({pct:.1f}%)")
```

---

## Summary

| Improvement | Current | Optimized | Savings |
|-------------|---------|-----------|---------|
| **Total Runtime** | ~8-10 min | ~1-2 min | **80% faster** |
| **DB Inserts** | 7,974 calls | 1 batch call | **100x faster** |
| **Pinecone Updates** | 7,974 calls | 80 batch calls | **99% fewer calls** |
| **LLM Labeling** | 9s sequential | 3s parallel | **67% faster** |
| **Reliability** | Timeout errors | No timeouts | **100% success** |
| **Resume Capability** | Start over | Resume from checkpoint | **Massive time savings** |
| **Cost on Re-runs** | Full cost | Near $0 (cached) | **99% savings** |

**Total Impact: 10x faster, 100x more reliable, 99% cheaper on re-runs**
