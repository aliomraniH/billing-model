# Data Sources Guide

## Overview

This document provides detailed information on obtaining and using the three primary data sources for the Medical Billing ML project.

---

## 1. Synthea (Primary - Free)

### What is Synthea?

Synthea is an open-source synthetic patient generator that creates realistic but fake electronic health records (EHRs). It's our primary data source for development and testing.

### Key Features

- **Fully synthetic:** No privacy concerns, freely redistributable
- **Comprehensive:** Complete patient histories from birth to death
- **Realistic:** Based on real clinical guidelines and disease prevalence
- **Structured:** Outputs in FHIR, C-CDA, CSV formats

### Data Included

| File | Description | Key Columns |
|------|-------------|-------------|
| `patients.csv` | Demographics | ID, BIRTHDATE, GENDER, RACE, ETHNICITY |
| `encounters.csv` | Hospital visits | ID, PATIENT, START, STOP, ENCOUNTERCLASS, TOTAL_CLAIM_COST |
| `conditions.csv` | Diagnoses | PATIENT, START, STOP, CODE (SNOMED-CT), DESCRIPTION |
| `procedures.csv` | Medical procedures | PATIENT, DATE, CODE (SNOMED-CT), DESCRIPTION, COST |
| `medications.csv` | Prescriptions | PATIENT, START, STOP, CODE (RxNorm), DESCRIPTION, COST |
| `observations.csv` | Lab results, vitals | PATIENT, DATE, CODE (LOINC), VALUE, UNITS |

### How to Obtain

**Option 1: Pre-generated Sample (Recommended for Quick Start)**
```bash
# Download 1.2K patient sample (April 2020)
wget https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip
unzip synthea_sample_data_csv_apr2020.zip
```

**Option 2: Generate Custom Dataset**
```bash
# Install Synthea
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build

# Generate 1000 patients for Massachusetts
./run_synthea -p 1000 Massachusetts

# Output will be in output/csv/
```

**Option 3: AWS Open Data (Large Datasets)**
- Bucket: `s3://synthetichealth-open-data/`
- 1M synthetic patients available
- Access via AWS CLI or browser: https://registry.opendata.aws/synthetichealth/

### Data Mapping to Our Schema

```python
# encounters.csv → claims table
claims_data = pd.DataFrame({
    'patient_id': encounters['PATIENT'].apply(lambda x: hash(x) % 10**9),
    'service_date': pd.to_datetime(encounters['START']).dt.date,
    'claim_type': encounters['ENCOUNTERCLASS'].map({
        'inpatient': 'inpatient',
        'outpatient': 'outpatient',
        'ambulatory': 'outpatient',
        'emergency': 'inpatient',
        'urgentcare': 'outpatient'
    }),
    'total_charge': encounters['TOTAL_CLAIM_COST'],
    'total_paid': encounters['PAYER_COVERAGE']
})

# conditions.csv → diagnoses table
# Note: Synthea uses SNOMED-CT codes, you may need to map to ICD-10
diagnoses_data = pd.DataFrame({
    'claim_id': matched_claim_ids,  # Join via patient + date
    'icd_code': conditions['CODE'],
    'icd_version': 10,
    'is_primary': True
})
```

### Limitations

- **No clinical notes:** Synthea doesn't generate free-text progress notes
- **SNOMED-CT codes:** Need mapping to ICD-10 for billing
- **Simplified pathways:** Disease progressions are modeled, may not capture rare presentations
- **Limited specialties:** Focused on primary care and common conditions

### Use Cases in This Project

- ✅ Initial database schema testing
- ✅ Outlier detection model training
- ✅ Pipeline development and debugging
- ❌ Clinical note analysis (use MIMIC-IV instead)

---

## 2. MIMIC-IV (Secondary - Credentialed Access)

### What is MIMIC-IV?

MIMIC-IV (Medical Information Mart for Intensive Care IV) contains **real** de-identified health data from 546,028 hospital admissions at Beth Israel Deaconess Medical Center (2008-2019).

### Key Features

- **Real clinical notes:** Discharge summaries, radiology reports, progress notes
- **Complete billing codes:** ICD-9 and ICD-10 diagnosis/procedure codes
- **ICU data:** Detailed vitals, labs, medications for critical care patients
- **Credentialed access:** Requires CITI training, approved use cases

### Data Included

| Module | Description | Size |
|--------|-------------|------|
| `hosp` | Hospital-wide data (billing, labs, microbiology) | ~70 GB |
| `icu` | ICU-specific data (vitals, medications, ventilation) | ~30 GB |
| `note` | Clinical notes (discharge, radiology, ECG) | ~40 GB |
| `cxr` | Chest X-ray images + reports | ~600 GB |

### How to Obtain

**Step 1: PhysioNet Credentialing (2-3 hours)**

1. Create account at https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training
   - Free, online, ~2-3 hours
   - Provides certificate valid for 3 years
3. Upload CITI certificate to PhysioNet profile
4. Sign Data Use Agreement (DUA) for MIMIC-IV
   - Agree to:
     - Not attempt re-identification
     - No data redistribution
     - Cite MIMIC-IV in publications

**Step 2: Download Dataset**

```bash
# Install GCP SDK (MIMIC-IV hosted on Google Cloud)
curl https://sdk.cloud.google.com | bash

# Request access at https://physionet.org/content/mimic-iv/2.2/
# After approval (usually < 24 hours):

# Download specific modules (NOT entire dataset)
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-iv/2.2/hosp/diagnoses_icd.csv.gz
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz
```

**Step 3: Access Demo (No Credentialing Required)**

- 100-patient sample: https://physionet.org/content/mimic-iv-demo/2.2/
- Includes all modules (hosp, icu, note)
- Great for testing queries before downloading full dataset

### Data Mapping to Our Schema

```python
# MIMIC-IV diagnoses_icd.csv → diagnoses table
mimic_diagnoses = pd.read_csv('diagnoses_icd.csv.gz')
diagnoses_data = pd.DataFrame({
    'claim_id': mimic_diagnoses['hadm_id'],  # Hospital admission ID
    'icd_code': mimic_diagnoses['icd_code'],
    'icd_version': mimic_diagnoses['icd_version'],  # 9 or 10
    'is_primary': mimic_diagnoses['seq_num'] == 1,
    'sequence_num': mimic_diagnoses['seq_num']
})

# MIMIC-IV discharge.csv → clinical_notes table
discharge_notes = pd.read_csv('discharge.csv.gz')
notes_data = pd.DataFrame({
    'claim_id': discharge_notes['hadm_id'],
    'note_type': 'discharge_summary',
    'note_text': discharge_notes['text']
    # embedding generated via all-MiniLM-L6-v2 in separate step
})
```

### Key Tables for Our Use Cases

| Table | Use Case | Key Columns |
|-------|----------|-------------|
| `diagnoses_icd` | Code suggestion training | hadm_id, icd_code, icd_version, seq_num |
| `procedures_icd` | Procedure analysis | hadm_id, icd_code, chartdate |
| `discharge` | Clinical note → code mapping | hadm_id, text |
| `radiology` | Radiology report analysis | hadm_id, text |

### Limitations

- **ICU-focused:** Sicker patients than general population
- **Single hospital:** May not generalize to other institutions
- **2008-2019:** Some billing codes/procedures outdated
- **De-identification artifacts:** Names replaced with [**Name**], dates shifted

### Use Cases in This Project

- ✅ Clinical note embeddings
- ✅ Code suggestion model training (notes → ICD-10)
- ✅ Validity cross-checking (do notes support billed codes?)
- ✅ NER/entity extraction with medspaCy

---

## 3. CMS DE-SynPUF (Tertiary - Public)

### What is DE-SynPUF?

The CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF) is a synthetic Medicare claims dataset created by CMS for public use.

### Key Features

- **Medicare-like structure:** Mimics real CMS claims format
- **2.3M synthetic beneficiaries:** Large-scale testing
- **No credentialing required:** Fully public
- **Linked across years:** 2008-2010 data available

### Data Included

| File | Description |
|------|-------------|
| `DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv` | Demographics, enrollment |
| `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv` | Hospital stays |
| `DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv` | Ambulatory visits |
| `DE1_0_2008_to_2010_Carrier_Claims_Sample_1.csv` | Physician services |
| `DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv` | Part D medications |

### How to Obtain

```bash
# Download from CMS website
wget https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip
unzip DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip

# Or use Kaggle datasets (easier)
# https://www.kaggle.com/cms/cms-desynpuf
```

### Limitations

- **No clinical notes:** Only structured claims data
- **Medicare population:** Skewed toward elderly (65+)
- **2008-2010 data:** ICD-9 codes only (need mapping to ICD-10)
- **Synthetic:** Created via complex statistical methods, may have artifacts

### Use Cases in This Project

- ✅ Schema validation (CMS claim structure is standard)
- ✅ Large-scale outlier detection testing
- ❌ Clinical note analysis (no notes available)
- ❌ Primary training data (use Synthea instead for ICD-10 codes)

---

## Data Source Comparison

| Feature | Synthea | MIMIC-IV | CMS DE-SynPUF |
|---------|---------|----------|---------------|
| **Access** | Free, instant | Free, credentialed | Free, instant |
| **Data Type** | Synthetic | Real (de-identified) | Synthetic |
| **Clinical Notes** | ❌ No | ✅ Yes | ❌ No |
| **Scale** | 1K-1M patients | 546K admissions | 2.3M beneficiaries |
| **ICD-10 Codes** | ✅ Yes (via mapping) | ✅ Yes | ❌ No (ICD-9 only) |
| **Best For** | Quick prototyping | NLP, code suggestion | Schema testing |

---

## Recommended Workflow

### Phase 1: Prototype (Weeks 1-2)
- Use **Synthea sample** (1.2K patients)
- Load into Vercel Postgres
- Train initial outlier detection models

### Phase 2: Clinical NLP (Weeks 3-4)
- Get **MIMIC-IV** credentialing
- Download discharge notes + diagnoses
- Train code suggestion models

### Phase 3: Scale Testing (Future)
- Generate larger **Synthea** dataset (10K-100K patients)
- OR download full **MIMIC-IV** (546K admissions)
- Benchmark query performance, vector search latency

---

## References

- **Synthea:** https://synthetichealth.github.io/synthea/
- **MIMIC-IV:** https://physionet.org/content/mimic-iv/2.2/
- **CMS DE-SynPUF:** https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/
- **PhysioNet CITI Training:** https://physionet.org/about/citi-course/
