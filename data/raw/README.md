# Raw Data Directory

This directory contains raw medical billing code data downloaded from public sources.

## Data Sources

### ICD-10-CM Codes
- **Source**: Centers for Medicare & Medicaid Services (CMS)
- **URL**: https://www.cms.gov/medicare/coding-billing/icd-10-codes
- **License**: Public Domain
- **File**: `icd10cm_codes_2025.csv`
- **Downloaded by**: `src/data/download_codes.py`

### HCPCS Level II Codes
- **Source**: Centers for Medicare & Medicaid Services (CMS)
- **URL**: https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system
- **License**: Public Domain
- **File**: `hcpcs_codes_2025.csv`
- **Downloaded by**: `src/data/download_codes.py`

## Usage

Run the setup notebook to download and process these files:
```bash
jupyter notebook notebooks/0_setup_knowledge_base.ipynb
```

## HIPAA Compliance

**IMPORTANT**: Do NOT place any files containing Protected Health Information (PHI) in this directory.

Only public medical code references and de-identified synthetic data should be stored here.

For real clinical data:
1. Always de-identify using `src/data/deidentify.py`
2. Store only in secure, encrypted locations
3. Never commit to git
