# Import Reference Guide

## File Structure

```
billing-model/
└── notebooks/
    ├── config.py                           # Configuration module
    ├── utils.py                            # Shared utilities (NEW!)
    ├── refresh_manager.py                  # Refresh manager
    ├── 05_embeddings_similarity_search.py  # Notebook 05
    ├── 06_llm_clustering_cache.py          # Notebook 06
    └── ... (other notebooks)
```

## Import Hierarchy

### Level 1: Configuration
```python
from config import get_config
```
- Used by: ALL notebooks and utils.py
- No dependencies

### Level 2: Utilities
```python
from utils import (
    get_embedding,
    init_database,
    init_pinecone,
    init_hf_client,
    init_anthropic_client,
    validate_environment_variables,
    test_embedding_generation,
    ensure_package_installed
)
```
- Used by: Notebooks 05, 06
- Depends on: config.py

### Level 3: Notebooks
```python
# In 05_embeddings_similarity_search.py and 06_llm_clustering_cache.py
from config import get_config
from utils import get_embedding, init_database, ...
```
- Imports from: config.py, utils.py
- Runs the ML pipeline

## How Imports Work

When you run a notebook from the notebooks directory:

```bash
cd /home/user/billing-model/notebooks
python 05_embeddings_similarity_search.py
```

Python's import resolution:
1. Looks in current directory (`/home/user/billing-model/notebooks/`)
2. Finds `config.py` ✅
3. Finds `utils.py` ✅
4. Imports work correctly ✅

## Verification

Test imports are working:

```bash
cd /home/user/billing-model/notebooks

# Test 1: Check config imports
python -c "from config import get_config; print('✅ Config OK')"

# Test 2: Check utils imports (requires numpy, sqlalchemy installed)
python -c "from utils import get_embedding; print('✅ Utils OK')"

# Test 3: Check notebook can import both
python -c "from config import get_config; from utils import init_database; print('✅ All OK')"
```

## Common Import Patterns

### Pattern 1: Notebook 05
```python
from config import get_config
from utils import (
    get_embedding, init_database, init_pinecone, init_hf_client,
    validate_environment_variables, test_embedding_generation
)

cfg = get_config()
engine, total_claims = init_database(DATABASE_URL)
hf_client = init_hf_client(HF_TOKEN, MODEL_ID)
embedding = get_embedding(text, hf_client, MODEL_ID, DIM)
```

### Pattern 2: Notebook 06
```python
from config import get_config
from utils import (
    get_embedding, init_database, init_pinecone, 
    init_hf_client, init_anthropic_client
)

cfg = get_config()
engine, total_claims = init_database(DATABASE_URL)
claude_client = init_anthropic_client(ANTHROPIC_API_KEY, CLAUDE_MODEL)
```

### Pattern 3: Refresh Manager
```python
from config import get_config

cfg = get_config()
# Uses config for refresh intervals
```

## No Circular Dependencies

```
config.py (independent)
    ↓
utils.py (depends on config)
    ↓
notebooks (depend on config + utils)
```

Clean dependency chain with no circular imports ✅

## File Locations Confirmed

All files are in `/home/user/billing-model/notebooks/`:
- ✅ config.py
- ✅ utils.py
- ✅ refresh_manager.py
- ✅ 05_embeddings_similarity_search.py
- ✅ 06_llm_clustering_cache.py

Import paths are correct and will work when notebooks are run! 🎉
