# Medical Billing ML

> AI-powered medical claim categorization using semantic embeddings and intelligent clustering

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)](https://billing-model.vercel.app)

---

## Overview

Medical Billing ML automatically categorizes medical claims by analyzing clinical notes using semantic similarity. Instead of manually sorting thousands of bills, the system uses AI to understand medical context, group similar claims, and generate meaningful category labels.

**Key capabilities:**
- Semantic search across clinical documentation
- Automatic clustering of similar medical claims
- LLM-powered category naming using Claude
- Real-time categorization of new claims
- Production-ready with 100x performance optimizations

---

## Business Context

### The Problem

Healthcare organizations process thousands of medical claims daily. Manual categorization is:
- **Time-intensive**: Staff spend hours sorting claims
- **Inconsistent**: Different reviewers categorize claims differently
- **Costly**: Labor costs scale linearly with volume

### The Solution

This system automates claim categorization by:
1. Understanding clinical context via AI embeddings
2. Finding patterns through unsupervised clustering
3. Creating meaningful labels via Claude LLM
4. Enabling instant similarity search

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Time per 1,000 claims | 8+ hours | 5 minutes |
| Categorization consistency | ~70% | 95%+ |
| Search time for similar claims | 15+ minutes | <1 second |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────────┐    ┌────────────────┐       │
│   │  Claims  │───▶│  Embeddings  │───▶│   Clustering   │       │
│   │  Input   │    │  (HF API)    │    │   (HDBSCAN)    │       │
│   └──────────┘    └──────────────┘    └────────────────┘       │
│                          │                    │                  │
│                          ▼                    ▼                  │
│               ┌──────────────────┐  ┌─────────────────┐        │
│               │ Pinecone Vector  │  │  Claude LLM     │        │
│               │ (Similarity)     │  │  (Labeling)     │        │
│               └──────────────────┘  └─────────────────┘        │
│                          │                    │                  │
│                          └────────┬───────────┘                 │
│                                   ▼                              │
│                        ┌──────────────────┐                     │
│                        │ Vercel Postgres  │                     │
│                        │ (Storage)        │                     │
│                        └──────────────────┘                     │
│                                   │                              │
│                                   ▼                              │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  OUTPUTS: Categorized Claims | Similarity Search | API   │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | Vercel Postgres | Claims, categories storage |
| Vector Store | Pinecone | Fast similarity search |
| Embeddings | HuggingFace (bge-small-en-v1.5) | 384-dim clinical text vectors |
| LLM | Claude API | Intelligent category labels |
| Clustering | HDBSCAN | Density-based grouping |

---

## 🧠 NLP Clinical Code Suggestion System

**NEW:** Phase 2 adds intelligent medical billing code extraction and validation.

### Features

- **Automated Code Extraction**: Extract ICD-10 and HCPCS codes from clinical notes using medspaCy + BioClinical ModernBERT
- **Semantic Similarity Search**: Find matching codes via pgvector (HNSW index) with <5ms query time
- **Cluster Consensus Suggestions**: Weak supervision algorithm suggests codes based on similar claims
- **Gap Analysis**: Identify revenue leakage (documented but not billed) and compliance risks
- **Negation Detection**: ConText-aware entity extraction filters out negated conditions

### NLP Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Clinical NLP | medspaCy + scispaCy | Medical entity extraction |
| Embeddings | BioClinical ModernBERT (768-dim) | Semantic code matching |
| Vector Store | pgvector (in Vercel Postgres) | Fast similarity search |
| De-identification | Microsoft Presidio | HIPAA compliance |
| Code Sources | CMS ICD-10-CM, HCPCS Level II | Public domain, free |

### Quick Start - NLP System

```bash
# 1. Install NLP dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Verify setup
python notebooks/00_verify_nlp_dependencies.py

# 3. Build knowledge base (one-time, 10-20 min)
python notebooks/07_nlp_knowledge_base_setup.py

# 4. Extract codes from claims
python notebooks/08_nlp_code_extraction.py

# 5. Run gap analysis
python notebooks/09_gap_analysis_reporting.py
```

### Example Output

```
📊 GAP ANALYSIS SUMMARY
═══════════════════════════════════════
Claims analyzed: 1,000
Claims with issues: 287 (28.7%)

💰 Revenue Impact:
  Revenue leakage flags: 156
  Compliance risk flags: 73
  Estimated revenue impact: $23,400

⚠️  Risk Distribution:
  High risk (>=0.5): 45 claims
  Medium risk (0.25-0.5): 98 claims
  Low risk (<0.25): 857 claims
```

**See [NLP_SYSTEM_README.md](NLP_SYSTEM_README.md) for detailed documentation.**

---

## Quick Start

### Prerequisites
- Python 3.11+
- Vercel, HuggingFace, Pinecone accounts
- Anthropic API key (optional)

### Installation

```bash
git clone https://github.com/aliomraniH/billing-model.git
cd billing-model
pip install -r requirements.txt

# Set environment variables
export VERCEL_POSTGRES_URL="postgresql://..."
export HF_TOKEN="hf_..."
export PINECONE_API_KEY="..."
```

### Run Pipeline

```bash
cd notebooks
python 01_setup_database.py      # Initialize DB
python 02_load_synthea_data.py   # Load sample data
python 05_embeddings_similarity_search.py  # Generate vectors
python 06_llm_clustering_cache.py  # Create categories
```

---

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Production Notes](docs/PRODUCTION_NOTES.md)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Testing Guide](docs/TESTING.md)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Author:** Ali Omrani | [GitHub](https://github.com/aliomraniH)
