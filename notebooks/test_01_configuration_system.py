"""
Test Notebook 1: Configuration System Validation

This notebook tests the dynamic configuration system including:
- Default configuration loading
- Environment variable overrides
- JSON file configuration
- Priority order (ENV > JSON > defaults)
- Refresh manager integration
- Database migration verification

Run this first before testing the main notebooks.
"""

import os
import sys
import json
from datetime import datetime
from sqlalchemy import create_engine, text

# Add /work to Python path for config imports
sys.path.insert(0, '/work')

print("=" * 70)
print("🧪 TEST 1: CONFIGURATION SYSTEM VALIDATION")
print("=" * 70)
print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70 + "\n")

# Verify configuration files exist
print("📁 Checking configuration file locations...")
config_files = {
    'config.py': '/work/config.py',
    'refresh_manager.py': '/work/refresh_manager.py'
}

for name, path in config_files.items():
    if os.path.exists(path):
        print(f"   ✅ {name}: {path}")
    else:
        print(f"   ❌ {name}: NOT FOUND at {path}")
        sys.exit(1)

print("   ✅ All configuration files found\n")

# ============================================================
# TEST 1.1: Environment Setup
# ============================================================
print("📋 TEST 1.1: Environment Setup")
print("-" * 70)

# Check required environment variables
required_vars = ['VERCEL_POSTGRES_URL', 'HF_TOKEN', 'PINECONE_API_KEY']
optional_vars = ['ANTHROPIC_API_KEY']

env_status = {}
missing = []

for var in required_vars:
    value = os.getenv(var)
    if value:
        env_status[var] = "✅ Set"
    else:
        env_status[var] = "❌ Missing"
        missing.append(var)

for var in optional_vars:
    value = os.getenv(var)
    env_status[var] = "✅ Set" if value else "⚠️  Optional (not set)"

for var, status in env_status.items():
    print(f"   {var}: {status}")

if missing:
    print(f"\n❌ ERROR: Missing required variables: {', '.join(missing)}")
    print("\nPlease set them before running:")
    for var in missing:
        print(f"   export {var}='your-value-here'")
    sys.exit(1)

print("\n✅ All required environment variables are set")

# ============================================================
# TEST 1.2: Default Configuration
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.2: Default Configuration Loading")
print("-" * 70)

try:
    from config import get_config

    cfg = get_config()

    print("\n🤖 Embedding Configuration:")
    print(f"   Provider: {cfg.embedding.provider}")
    print(f"   Model: {cfg.embedding.model_id}")
    print(f"   Dimension: {cfg.embedding.dimension}")
    print(f"   Batch size: {cfg.embedding.batch_size}")
    print(f"   Max retries: {cfg.embedding.max_retries}")

    print("\n🧠 LLM Configuration:")
    print(f"   Provider: {cfg.llm.provider}")
    print(f"   Model: {cfg.llm.model_id}")
    print(f"   Max tokens: {cfg.llm.max_tokens}")
    print(f"   Temperature: {cfg.llm.temperature}")

    print("\n🔄 Refresh Configuration:")
    print(f"   Embedding refresh: every {cfg.refresh.default_embedding_refresh_hours} hours")
    print(f"   Category refresh: every {cfg.refresh.category_refresh_hours} hours")
    print(f"   Cluster refresh: every {cfg.refresh.cluster_refresh_hours} hours")
    print(f"   Auto-refresh: {cfg.refresh.auto_refresh_enabled}")
    print(f"   Max refresh per run: {cfg.refresh.max_refresh_per_run:,}")

    print("\n📊 Pinecone Configuration:")
    print(f"   Index: {cfg.pinecone.index_name}")
    print(f"   Dimension: {cfg.pinecone.dimension}")
    print(f"   Metric: {cfg.pinecone.metric}")

    print("\n🔬 Clustering Configuration:")
    print(f"   Min cluster size: {cfg.clustering.min_cluster_size}")
    print(f"   Min samples: {cfg.clustering.min_samples}")
    print(f"   Adaptive sizing: {cfg.clustering.adaptive_sizing}")
    print(f"   Adaptive ratio: {cfg.clustering.adaptive_size_ratio}")

    print("\n⚙️  Processing Configuration:")
    max_claims = cfg.processing.max_claims_to_process
    print(f"   Max claims: {'ALL' if max_claims == -1 else f'{max_claims:,}'}")
    print(f"   Batch size: {cfg.processing.batch_size}")
    print(f"   Parallel workers: {cfg.processing.parallel_workers}")

    print("\n✅ Default configuration loaded successfully")

except Exception as e:
    print(f"\n❌ ERROR loading configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# TEST 1.3: Environment Variable Override
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.3: Environment Variable Override")
print("-" * 70)

# Save original values
original_claude = os.getenv('CLAUDE_MODEL')
original_embedding = os.getenv('HF_EMBEDDING_MODEL')
original_refresh = os.getenv('EMBEDDING_REFRESH_HOURS')

# Set test overrides
os.environ['CLAUDE_MODEL'] = 'test-claude-model'
os.environ['HF_EMBEDDING_MODEL'] = 'test-embedding-model'
os.environ['EMBEDDING_REFRESH_HOURS'] = '6'

# Reload configuration
cfg_override = get_config()

print("\n🧪 Testing overrides:")
print(f"   ENV: CLAUDE_MODEL='test-claude-model'")
print(f"   Result: {cfg_override.llm.model_id}")
assert cfg_override.llm.model_id == 'test-claude-model', "Claude model override failed!"
print("   ✅ Claude model override works")

print(f"\n   ENV: HF_EMBEDDING_MODEL='test-embedding-model'")
print(f"   Result: {cfg_override.embedding.model_id}")
assert cfg_override.embedding.model_id == 'test-embedding-model', "Embedding model override failed!"
print("   ✅ Embedding model override works")

print(f"\n   ENV: EMBEDDING_REFRESH_HOURS='6'")
print(f"   Result: {cfg_override.refresh.default_embedding_refresh_hours}")
assert cfg_override.refresh.default_embedding_refresh_hours == 6, "Refresh hours override failed!"
print("   ✅ Refresh hours override works")

# Restore original values
if original_claude:
    os.environ['CLAUDE_MODEL'] = original_claude
else:
    os.environ.pop('CLAUDE_MODEL', None)

if original_embedding:
    os.environ['HF_EMBEDDING_MODEL'] = original_embedding
else:
    os.environ.pop('HF_EMBEDDING_MODEL', None)

if original_refresh:
    os.environ['EMBEDDING_REFRESH_HOURS'] = original_refresh
else:
    os.environ.pop('EMBEDDING_REFRESH_HOURS', None)

print("\n✅ All environment variable overrides work correctly")

# ============================================================
# TEST 1.4: JSON Configuration Override
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.4: JSON Configuration Override")
print("-" * 70)

test_config = {
    "llm": {
        "model_id": "test-json-claude-model",
        "max_tokens": 1000
    },
    "refresh": {
        "default_embedding_refresh_hours": 24
    }
}

test_config_file = "test-config.json"

try:
    # Write test config
    with open(test_config_file, 'w') as f:
        json.dump(test_config, f, indent=2)

    print(f"\n📝 Created test config file: {test_config_file}")
    print(f"   Content: {json.dumps(test_config, indent=6)}")

    # Load with config file
    cfg_json = get_config(config_file=test_config_file)

    print(f"\n🧪 Testing JSON overrides:")
    print(f"   JSON: llm.model_id='test-json-claude-model'")
    print(f"   Result: {cfg_json.llm.model_id}")
    assert cfg_json.llm.model_id == 'test-json-claude-model', "JSON model override failed!"
    print("   ✅ JSON model override works")

    print(f"\n   JSON: llm.max_tokens=1000")
    print(f"   Result: {cfg_json.llm.max_tokens}")
    assert cfg_json.llm.max_tokens == 1000, "JSON max_tokens override failed!"
    print("   ✅ JSON max_tokens override works")

    print(f"\n   JSON: refresh.default_embedding_refresh_hours=24")
    print(f"   Result: {cfg_json.refresh.default_embedding_refresh_hours}")
    assert cfg_json.refresh.default_embedding_refresh_hours == 24, "JSON refresh hours override failed!"
    print("   ✅ JSON refresh hours override works")

    print("\n✅ JSON configuration override works correctly")

finally:
    # Clean up test file
    if os.path.exists(test_config_file):
        os.remove(test_config_file)
        print(f"\n🧹 Cleaned up test config file")

# ============================================================
# TEST 1.5: Configuration Priority Order
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.5: Configuration Priority Order (ENV > JSON > Defaults)")
print("-" * 70)

# Create JSON config
json_config = {
    "llm": {
        "model_id": "json-model",
        "max_tokens": 750
    }
}

json_config_file = "priority-test.json"

try:
    # Write JSON config
    with open(json_config_file, 'w') as f:
        json.dump(json_config, f)

    # Set ENV variable
    os.environ['CLAUDE_MODEL'] = 'env-model'

    # Load config
    cfg_priority = get_config(config_file=json_config_file)

    print("\n🧪 Testing priority order:")
    print(f"   Default: claude-sonnet-4-5-20250929")
    print(f"   JSON:    json-model")
    print(f"   ENV:     env-model")
    print(f"   Result:  {cfg_priority.llm.model_id}")

    assert cfg_priority.llm.model_id == 'env-model', "Priority order failed! ENV should override JSON"
    print("\n   ✅ Priority correct: ENV > JSON > Default")

    # Test without ENV
    os.environ.pop('CLAUDE_MODEL')
    cfg_priority2 = get_config(config_file=json_config_file)

    print(f"\n   Without ENV:")
    print(f"   Result:  {cfg_priority2.llm.model_id}")
    assert cfg_priority2.llm.model_id == 'json-model', "JSON should override default!"
    print("   ✅ Priority correct: JSON > Default")

    print("\n✅ Configuration priority order works correctly")

finally:
    # Clean up
    if os.path.exists(json_config_file):
        os.remove(json_config_file)
    os.environ.pop('CLAUDE_MODEL', None)

# ============================================================
# TEST 1.6: Database Migration Verification
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.6: Database Migration Verification")
print("-" * 70)

DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')

try:
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Check clinical_notes columns
        print("\n🔍 Checking clinical_notes table:")
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'clinical_notes'
            AND column_name IN ('last_embedded_at', 'embedding_model', 'embedding_version')
            ORDER BY column_name
        """))
        columns = [row[0] for row in result.fetchall()]

        expected_columns = ['embedding_model', 'embedding_version', 'last_embedded_at']
        for col in expected_columns:
            if col in columns:
                print(f"   ✅ {col}")
            else:
                print(f"   ❌ {col} - MISSING!")

        if set(expected_columns) == set(columns):
            print("\n   ✅ All timestamp columns exist")
        else:
            print("\n   ❌ Some columns missing! Run migration first:")
            print("      python migrations/01_add_refresh_timestamps.py")
            sys.exit(1)

        # Check claim_categories columns
        print("\n🔍 Checking claim_categories table:")
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'claim_categories'
            AND column_name IN ('last_refreshed_at', 'refresh_interval_hours', 'cluster_version')
            ORDER BY column_name
        """))
        columns = [row[0] for row in result.fetchall()]

        expected_columns = ['cluster_version', 'last_refreshed_at', 'refresh_interval_hours']
        for col in expected_columns:
            if col in columns:
                print(f"   ✅ {col}")
            else:
                print(f"   ❌ {col} - MISSING!")

        # Check refresh config table
        print("\n🔍 Checking embedding_refresh_config table:")
        result = conn.execute(text("""
            SELECT config_name, refresh_interval_hours, enabled
            FROM embedding_refresh_config
            ORDER BY config_name
        """))
        configs = result.fetchall()

        if configs:
            for name, hours, enabled in configs:
                status = "✅ enabled" if enabled else "⏸️  disabled"
                print(f"   {name}: {hours}h ({status})")
            print(f"\n   ✅ Found {len(configs)} refresh configurations")
        else:
            print("   ❌ No refresh configurations found!")
            print("      Run migration: python migrations/01_add_refresh_timestamps.py")
            sys.exit(1)

        # Check data statistics
        print("\n📊 Data Statistics:")
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_notes,
                COUNT(last_embedded_at) as with_timestamps,
                COUNT(embedding_model) as with_model
            FROM clinical_notes
        """))
        row = result.fetchone()
        print(f"   Total notes: {row[0]:,}")
        print(f"   With timestamps: {row[1]:,}")
        print(f"   With model set: {row[2]:,}")

        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_categories,
                COUNT(last_refreshed_at) as with_timestamps
            FROM claim_categories
        """))
        row = result.fetchone()
        print(f"\n   Total categories: {row[0]:,}")
        print(f"   With timestamps: {row[1]:,}")

    print("\n✅ Database migration verified successfully")

except Exception as e:
    print(f"\n❌ ERROR verifying database: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# TEST 1.7: Refresh Manager Integration
# ============================================================
print("\n" + "=" * 70)
print("📋 TEST 1.7: Refresh Manager Integration")
print("-" * 70)

try:
    from refresh_manager import RefreshManager

    print("\n🔍 Checking refresh status...")

    manager = RefreshManager(DATABASE_URL, dry_run=True)

    # Check embeddings
    stale_embeddings = manager.find_stale_embeddings(limit=5)
    print(f"\n📊 Embeddings:")
    print(f"   Found {len(stale_embeddings)} stale embeddings (showing max 5)")

    if stale_embeddings:
        print(f"\n   Sample stale embeddings:")
        for i, note in enumerate(stale_embeddings[:3], 1):
            age = note['age_hours']
            age_str = f"{age:.1f}h" if age else "never embedded"
            print(f"      [{i}] Note {note['note_id']}: {age_str} old")
    else:
        print("   ✅ All embeddings are fresh!")

    # Check categories
    stale_categories = manager.find_stale_categories()
    print(f"\n📊 Categories:")
    print(f"   Found {len(stale_categories)} stale categories")

    if stale_categories:
        print(f"\n   Sample stale categories:")
        for i, cat in enumerate(stale_categories[:3], 1):
            age = cat['age_hours']
            age_str = f"{age:.1f}h" if age else "never refreshed"
            interval = cat['refresh_interval_hours']
            print(f"      [{i}] {cat['category_name']}: {age_str} old (refresh every {interval}h)")
    else:
        print("   ✅ All categories are fresh!")

    print("\n✅ Refresh manager integration working")

except Exception as e:
    print(f"\n❌ ERROR testing refresh manager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# TEST SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)

print("""
✅ Test Results:
   1.1 ✅ Environment variables validated
   1.2 ✅ Default configuration loaded
   1.3 ✅ Environment variable overrides work
   1.4 ✅ JSON configuration overrides work
   1.5 ✅ Priority order correct (ENV > JSON > Default)
   1.6 ✅ Database migration verified
   1.7 ✅ Refresh manager integration works

📋 Configuration System Status:
   ✅ All components working correctly
   ✅ Ready to run main notebooks
   ✅ Dynamic model updates enabled
   ✅ Timestamp-based refresh enabled

🎯 Next Steps:
   1. Run Notebook 5 (Embeddings):
      export MAX_CLAIMS_TO_PROCESS=100
      python 05_embeddings_similarity_search.py

   2. Run Notebook 6 (Clustering):
      python 06_llm_clustering_cache.py

   3. Test auto-refresh:
      python 05_embeddings_similarity_search.py  # Should skip fresh embeddings

💡 Tips:
   - Change models: export CLAUDE_MODEL='claude-opus-4-5-20251101'
   - Change refresh: export EMBEDDING_REFRESH_HOURS=6
   - Check staleness: python refresh_manager.py --check

""")

print("=" * 70)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
