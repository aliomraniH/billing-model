"""
Medical Billing NLP System - Dependency Verification

Run this script BEFORE notebooks/07_nlp_knowledge_base_setup.py to verify that:
1. All required Python packages are installed
2. Database connection works
3. Models are accessible
4. System is ready for NLP setup

This saves time by catching configuration issues early.

Usage:
    In Deepnote: %run notebooks/00_verify_nlp_dependencies.py
    In terminal: python notebooks/00_verify_nlp_dependencies.py
"""

import sys
import os
from typing import List, Tuple, Dict

# Add project root to path
if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')
if os.path.exists('/work') and '/work' not in sys.path:
    sys.path.insert(0, '/work')

print("=" * 80)
print(" " * 20 + "NLP SYSTEM DEPENDENCY VERIFICATION")
print(" " * 25 + "Medical Billing Code Extraction")
print("=" * 80)
print("\nThis script verifies that your environment is ready for NLP setup.")
print("Run this BEFORE notebooks/07_nlp_knowledge_base_setup.py\n")

# Track results
all_checks_passed = True
warnings = []
errors = []

# ============================================================================
# 1. PYTHON VERSION CHECK
# ============================================================================

print("\n" + "=" * 80)
print("📌 Step 1: Python Version Check")
print("=" * 80)

python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major == 3 and python_version.minor >= 8:
    print("✅ Python version is compatible (3.8+)")
else:
    print("❌ Python version must be 3.8 or higher")
    errors.append("Python version too old (need 3.8+)")
    all_checks_passed = False

# ============================================================================
# 2. CORE PACKAGE IMPORTS
# ============================================================================

print("\n" + "=" * 80)
print("📦 Step 2: Core Package Imports")
print("=" * 80)

core_packages = [
    ('pandas', 'Data manipulation'),
    ('numpy', 'Numerical operations'),
    ('sklearn', 'scikit-learn - Machine learning'),
    ('scipy', 'Scientific computing'),
    ('sqlalchemy', 'Database ORM'),
    ('psycopg2', 'PostgreSQL driver'),
    ('requests', 'HTTP requests'),
    ('tqdm', 'Progress bars'),
]

for package_name, description in core_packages:
    try:
        if package_name == 'sklearn':
            import sklearn
        else:
            __import__(package_name)
        print(f"✅ {package_name:20s} - {description}")
    except ImportError as e:
        print(f"❌ {package_name:20s} - MISSING: {description}")
        errors.append(f"Missing package: {package_name}")
        all_checks_passed = False

# ============================================================================
# 3. NLP PACKAGE IMPORTS
# ============================================================================

print("\n" + "=" * 80)
print("🧠 Step 3: NLP Package Imports")
print("=" * 80)

nlp_packages = [
    ('spacy', 'spaCy - NLP framework', 'pip install spacy>=3.7.0'),
    ('medspacy', 'medspaCy - Clinical NLP', 'pip install medspacy>=1.2.0'),
    ('scispacy', 'scispaCy - Scientific NLP', 'pip install scispacy>=0.5.4'),
    ('sentence_transformers', 'Sentence Transformers - Embeddings', 'pip install sentence-transformers>=2.2.2'),
    ('transformers', 'HuggingFace Transformers', 'pip install transformers>=4.48.0'),
    ('torch', 'PyTorch - Deep learning', 'pip install torch>=2.0.0'),
]

for package_name, description, install_cmd in nlp_packages:
    try:
        __import__(package_name)
        print(f"✅ {package_name:25s} - {description}")
    except ImportError:
        print(f"❌ {package_name:25s} - MISSING: {description}")
        print(f"   Install: {install_cmd}")
        errors.append(f"Missing NLP package: {package_name}")
        all_checks_passed = False

# ============================================================================
# 4. SPACY MODELS CHECK
# ============================================================================

print("\n" + "=" * 80)
print("🔤 Step 4: spaCy Models")
print("=" * 80)

spacy_models_ok = True
try:
    import spacy

    # Check for base English model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ en_core_web_sm - Base English model")
    except OSError:
        print("❌ en_core_web_sm - NOT INSTALLED")
        print("   Install: python -m spacy download en_core_web_sm")
        warnings.append("Missing spaCy model: en_core_web_sm")
        spacy_models_ok = False

    # Check for scispaCy model
    try:
        nlp = spacy.load("en_core_sci_md")
        print("✅ en_core_sci_md - Scientific/medical model")
    except OSError:
        print("❌ en_core_sci_md - NOT INSTALLED")
        print("   Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz")
        warnings.append("Missing scispaCy model: en_core_sci_md")
        spacy_models_ok = False

except ImportError:
    print("⚠️  spaCy not installed, skipping model check")
    spacy_models_ok = False

# ============================================================================
# 5. PGVECTOR EXTENSION CHECK
# ============================================================================

print("\n" + "=" * 80)
print("🗄️  Step 5: Database & pgvector Extension")
print("=" * 80)

try:
    import pgvector
    print("✅ pgvector Python package installed")
except ImportError:
    print("❌ pgvector Python package NOT INSTALLED")
    print("   Install: pip install pgvector>=0.2.4")
    errors.append("Missing package: pgvector")
    all_checks_passed = False

# ============================================================================
# 6. PRESIDIO (DE-IDENTIFICATION) CHECK
# ============================================================================

print("\n" + "=" * 80)
print("🔒 Step 6: Presidio (HIPAA De-identification)")
print("=" * 80)

presidio_packages = [
    ('presidio_analyzer', 'Presidio Analyzer - PHI detection', 'pip install presidio-analyzer>=2.2.0'),
    ('presidio_anonymizer', 'Presidio Anonymizer - PHI replacement', 'pip install presidio-anonymizer>=2.2.0'),
]

for package_name, description, install_cmd in presidio_packages:
    try:
        __import__(package_name)
        print(f"✅ {package_name:25s} - {description}")
    except ImportError:
        print(f"⚠️  {package_name:25s} - MISSING: {description}")
        print(f"   Install: {install_cmd}")
        warnings.append(f"Optional package missing: {package_name} (needed for de-identification)")

# ============================================================================
# 7. DATABASE CONNECTION TEST
# ============================================================================

print("\n" + "=" * 80)
print("🔌 Step 7: Database Connection Test")
print("=" * 80)

try:
    from config.settings import DATABASE_URL
    import psycopg2

    # Check if URL is configured
    if not DATABASE_URL or 'localhost' in DATABASE_URL:
        print("⚠️  WARNING: Using default/local database URL")
        print("   For Deepnote: Set VERCEL_POSTGRES_URL in Project Settings")
        print("   Current URL:", DATABASE_URL[:50] + "...")
        warnings.append("Database URL not configured for cloud (using default)")
    else:
        print("✅ Cloud database URL detected (Vercel Postgres)")
        print("   Host:", DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'unknown')

    # Test connection
    print("\nTesting database connection...")
    conn = psycopg2.connect(DATABASE_URL)

    with conn.cursor() as cur:
        # Get PostgreSQL version
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"✅ Connected to PostgreSQL")
        print(f"   Version: {version.split(',')[0]}")

        # Check for pgvector extension
        cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
        has_pgvector = cur.fetchone()[0] > 0

        if has_pgvector:
            print("✅ pgvector extension is installed")

            # Get pgvector version
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            pgvector_version = cur.fetchone()
            if pgvector_version:
                print(f"   Version: {pgvector_version[0]}")
        else:
            print("⚠️  pgvector extension NOT INSTALLED")
            print("   The setup script will attempt to install it automatically")
            print("   If this fails, run in PostgreSQL console: CREATE EXTENSION vector;")
            warnings.append("pgvector extension not yet installed (will be installed during setup)")

        # Check if code_embeddings table exists (from previous setup)
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'code_embeddings';
        """)
        has_table = cur.fetchone()[0] > 0

        if has_table:
            cur.execute("SELECT COUNT(*) FROM code_embeddings;")
            count = cur.fetchone()[0]
            print(f"ℹ️  Found existing code_embeddings table with {count:,} codes")
            print("   Setup will rebuild/update this table")
        else:
            print("ℹ️  code_embeddings table does not exist yet (will be created)")

    conn.close()
    print("✅ Database connection successful")

except Exception as e:
    print(f"❌ Database connection FAILED: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Verify VERCEL_POSTGRES_URL in Deepnote Project Settings")
    print("   2. Check database is running in Vercel Dashboard")
    print("   3. Ensure using POSTGRES_URL_NON_POOLING (not pooled)")
    print("   4. Verify network connectivity")
    errors.append(f"Database connection failed: {str(e)[:100]}")
    all_checks_passed = False

# ============================================================================
# 8. HUGGINGFACE MODEL ACCESS TEST
# ============================================================================

print("\n" + "=" * 80)
print("🤗 Step 8: HuggingFace Model Access Test")
print("=" * 80)

try:
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL

    print(f"Testing model access: {EMBEDDING_MODEL}")
    print("⏳ This may take a few minutes on first run (downloading ~1GB)...")
    print("   Subsequent runs will be instant (model cached)\n")

    # Try to load model (this will download if not cached)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"✅ Model loaded successfully")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   Max sequence length: {model.max_seq_length}")

    # Test encoding
    test_text = "diabetes type 2"
    embedding = model.encode(test_text, convert_to_numpy=True)
    print(f"   Test encoding: '{test_text}' → vector[{len(embedding)}]")

    print("✅ Model is fully functional")

except Exception as e:
    print(f"❌ Model access FAILED: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check internet connectivity")
    print("   2. Verify HF_TOKEN if using private models (optional for this model)")
    print("   3. Try: pip install --upgrade sentence-transformers transformers")
    errors.append(f"Model access failed: {str(e)[:100]}")
    all_checks_passed = False

# ============================================================================
# 9. MEDSPACY PIPELINE TEST
# ============================================================================

print("\n" + "=" * 80)
print("🏥 Step 9: medspaCy Pipeline Test")
print("=" * 80)

try:
    import medspacy
    from medspacy.ner import TargetRule

    print("Loading medspaCy pipeline...")
    nlp = medspacy.load(enable=["sentencizer"])

    print("✅ Base pipeline loaded")

    # Add components
    nlp.add_pipe("medspacy_context")
    print("✅ ConText component added")

    nlp.add_pipe("medspacy_target_matcher")
    print("✅ Target matcher added")

    # Test with sample text
    test_text = "Patient denies chest pain. History of diabetes."
    doc = nlp(test_text)

    print(f"✅ Pipeline functional")
    print(f"   Test text: '{test_text}'")
    print(f"   Sentences detected: {len(list(doc.sents))}")

except Exception as e:
    print(f"❌ medspaCy pipeline FAILED: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Verify medspacy is installed: pip install medspacy")
    print("   2. Check spaCy models are installed (see Step 4)")
    print("   3. Try: pip install --upgrade medspacy spacy")
    errors.append(f"medspaCy pipeline failed: {str(e)[:100]}")
    all_checks_passed = False

# ============================================================================
# 10. MEMORY CHECK
# ============================================================================

print("\n" + "=" * 80)
print("💾 Step 10: System Memory Check")
print("=" * 80)

try:
    import psutil

    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)

    print(f"Total memory: {memory_gb:.1f} GB")
    print(f"Available memory: {available_gb:.1f} GB")
    print(f"Memory usage: {memory.percent}%")

    if available_gb < 2:
        print("⚠️  WARNING: Low available memory (<2GB)")
        print("   Recommend closing other applications")
        print("   Model loading may be slow")
        warnings.append("Low available memory (<2GB)")
    else:
        print("✅ Sufficient memory available")

except ImportError:
    print("⚠️  psutil not installed, skipping memory check")
    print("   Install: pip install psutil")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📊 VERIFICATION SUMMARY")
print("=" * 80)

print(f"\n{'=' * 80}")
if all_checks_passed and len(errors) == 0:
    print("✅ ALL CRITICAL CHECKS PASSED!")
    print("=" * 80)
    print("\n🎉 Your environment is ready for NLP setup!")
    print("\nNext steps:")
    print("   1. Run: %run notebooks/07_nlp_knowledge_base_setup.py")
    print("   2. Wait 10-20 minutes for initial setup")
    print("   3. Start using NLP code extraction")
else:
    print("❌ SOME CHECKS FAILED")
    print("=" * 80)
    print("\n⚠️  Please fix the errors below before proceeding:")

if errors:
    print(f"\n🚫 ERRORS ({len(errors)}):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

if all_checks_passed and len(errors) == 0:
    if warnings:
        print(f"\nℹ️  {len(warnings)} warning(s) found, but you can proceed.")
        print("   Warnings indicate optional features or minor issues.")
    print("\n✅ Ready to run notebooks/07_nlp_knowledge_base_setup.py")
else:
    print("\n❌ NOT ready for setup. Fix errors above first.")
    sys.exit(1)

print("\n" + "=" * 80)
print("End of verification")
print("=" * 80)
