"""
Download and parse medical billing codes from public CMS sources.

This module handles downloading ICD-10-CM and HCPCS Level II codes from
the Centers for Medicare & Medicaid Services (CMS) public domain data.
"""

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from tqdm import tqdm

from config.settings import (
    CMS_ICD10_URL,
    CMS_HCPCS_URL,
    NLM_ICD10_API,
    NLM_HCPCS_API,
    RAW_DATA_DIR
)


class CodeDownloader:
    """Download and parse medical billing codes from CMS."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded files (default: RAW_DATA_DIR)
        """
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Medical-Billing-Research/1.0'
        })

    def download_file(
        self,
        url: str,
        filename: str,
        max_retries: int = 3
    ) -> Path:
        """
        Download a file with retry logic.

        Args:
            url: URL to download from
            filename: Name to save file as
            max_retries: Maximum number of retry attempts

        Returns:
            Path to downloaded file

        Raises:
            requests.RequestException: If download fails after all retries
        """
        filepath = self.cache_dir / filename

        # Return cached file if exists
        if filepath.exists():
            print(f"Using cached file: {filepath}")
            return filepath

        print(f"Downloading {filename}...")

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(filepath, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filename
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                print(f"Downloaded: {filepath}")
                return filepath

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Download failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    raise

        raise requests.RequestException(f"Failed to download {url} after {max_retries} attempts")

    def download_icd10_codes(self) -> pd.DataFrame:
        """
        Download and parse ICD-10-CM codes from CMS.

        The ICD-10-CM codes are available as a ZIP file containing a text file
        with fixed-width format data.

        Returns:
            DataFrame with columns: code, short_description, long_description, is_billable

        Note:
            CMS URLs may change. If download fails, check CMS website for updated URL:
            https://www.cms.gov/medicare/coding-billing/icd-10-codes
        """
        print("=" * 60)
        print("Downloading ICD-10-CM Codes from CMS")
        print("=" * 60)

        # Note: CMS URLs change yearly. This is a fallback approach.
        # For 2025, we'll try to parse the CMS page or use NLM API

        try:
            # Try direct download (URL may need updating)
            zip_path = self.download_file(
                CMS_ICD10_URL,
                "icd10cm_2025.zip"
            )

            # Extract and parse
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Look for the order file (contains all codes)
                txt_files = [f for f in zf.namelist() if f.endswith('.txt')]

                if not txt_files:
                    raise FileNotFoundError("No .txt file found in ZIP")

                # Usually the largest file contains the codes
                txt_file = max(txt_files, key=lambda f: zf.getinfo(f).file_size)

                with zf.open(txt_file) as f:
                    content = f.read().decode('utf-8', errors='ignore')

            # Parse fixed-width format
            codes_data = []
            for line in content.split('\n'):
                if len(line) < 10:
                    continue

                # ICD-10 codes are typically in format:
                # Code (positions 0-7), Description (rest of line)
                code_match = re.match(r'^([A-Z][0-9][0-9A-Z]\.?[0-9A-Z]{0,4})\s+(.+)$', line.strip())

                if code_match:
                    code = code_match.group(1).replace('.', '')
                    description = code_match.group(2).strip()

                    # Billable codes are leaf nodes (most specific)
                    # Simplified: codes with 4+ characters after first 3
                    is_billable = len(code) >= 4

                    codes_data.append({
                        'code': code,
                        'short_description': description[:100],
                        'long_description': description,
                        'is_billable': is_billable
                    })

            df = pd.DataFrame(codes_data)
            print(f"Loaded {len(df)} ICD-10-CM codes")

        except Exception as e:
            print(f"CMS download failed: {e}")
            print("Falling back to NLM synthetic dataset...")

            # Create a minimal synthetic dataset for testing
            # In production, use NLM API or manual download
            df = self._create_synthetic_icd10()

        # Save to cache
        csv_path = self.cache_dir / "icd10cm_codes_2025.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")

        return df

    def download_hcpcs_codes(self) -> pd.DataFrame:
        """
        Download and parse HCPCS Level II codes from CMS.

        Returns:
            DataFrame with columns: code, long_description, short_description, category

        Note:
            HCPCS codes are free (public domain), unlike CPT codes which require AMA licensing.
        """
        print("=" * 60)
        print("Downloading HCPCS Level II Codes from CMS")
        print("=" * 60)

        try:
            # Try direct download
            zip_path = self.download_file(
                CMS_HCPCS_URL,
                "hcpcs_2025.zip"
            )

            # Extract and parse
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Look for Excel or CSV file
                data_files = [f for f in zf.namelist()
                            if f.endswith(('.xlsx', '.xls', '.csv', '.txt'))]

                if not data_files:
                    raise FileNotFoundError("No data file found in ZIP")

                data_file = data_files[0]

                with zf.open(data_file) as f:
                    if data_file.endswith('.csv'):
                        df = pd.read_csv(f)
                    elif data_file.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(f)
                    else:
                        # Try CSV parsing
                        df = pd.read_csv(f)

            # Standardize column names
            column_mapping = {
                'HCPCS Code': 'code',
                'Long Description': 'long_description',
                'Short Description': 'short_description',
                'Code': 'code',
                'Description': 'long_description'
            }

            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            if 'code' not in df.columns:
                raise ValueError("Could not find code column")

            if 'long_description' not in df.columns:
                if 'short_description' in df.columns:
                    df['long_description'] = df['short_description']
                else:
                    df['long_description'] = 'No description'

            if 'short_description' not in df.columns:
                df['short_description'] = df['long_description'].str[:100]

            # Add category based on first letter
            df['category'] = df['code'].str[0].map({
                'A': 'Transportation/Medical Supplies',
                'B': 'Enteral and Parenteral Therapy',
                'C': 'Outpatient PPS',
                'D': 'Dental Procedures',
                'E': 'Durable Medical Equipment',
                'G': 'Procedures/Services',
                'H': 'Behavioral Health',
                'J': 'Drugs',
                'K': 'Temporary Codes',
                'L': 'Orthotics/Prosthetics',
                'M': 'Medical Services',
                'P': 'Pathology/Laboratory',
                'Q': 'Temporary Codes',
                'R': 'Diagnostic Radiology',
                'S': 'Temporary National Codes',
                'T': 'State Medicaid',
                'V': 'Vision/Hearing'
            }).fillna('Other')

            print(f"Loaded {len(df)} HCPCS codes")

        except Exception as e:
            print(f"CMS download failed: {e}")
            print("Falling back to synthetic dataset...")
            df = self._create_synthetic_hcpcs()

        # Save to cache
        csv_path = self.cache_dir / "hcpcs_codes_2025.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")

        return df

    def search_icd10_api(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search ICD-10 codes using NLM Clinical Tables API.

        This is a real-time API that doesn't require authentication.

        Args:
            query: Search term
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with 'code' and 'description' keys
        """
        url = f"{NLM_ICD10_API}?terms={query}&maxList={max_results}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # NLM API returns: [count, [codes], null, [descriptions]]
            if len(data) >= 4 and data[1] and data[3]:
                results = []
                for code, desc in zip(data[1], data[3]):
                    results.append({
                        'code': code,
                        'description': desc
                    })
                return results

        except Exception as e:
            print(f"NLM API error: {e}")

        return []

    def search_hcpcs_api(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search HCPCS codes using NLM Clinical Tables API.

        Args:
            query: Search term
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with 'code' and 'description' keys
        """
        url = f"{NLM_HCPCS_API}?terms={query}&maxList={max_results}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) >= 4 and data[1] and data[3]:
                results = []
                for code, desc in zip(data[1], data[3]):
                    results.append({
                        'code': code,
                        'description': desc
                    })
                return results

        except Exception as e:
            print(f"NLM API error: {e}")

        return []

    def _create_synthetic_icd10(self) -> pd.DataFrame:
        """Create synthetic ICD-10 dataset for testing."""
        synthetic_codes = [
            # Diabetes
            ('E11', 'Type 2 diabetes mellitus', 'Type 2 DM', False),
            ('E119', 'Type 2 diabetes mellitus without complications', 'Type 2 DM w/o comp', True),
            ('E1165', 'Type 2 diabetes mellitus with hyperglycemia', 'Type 2 DM w/ hyperglycemia', True),
            ('E1122', 'Type 2 diabetes mellitus with diabetic chronic kidney disease', 'Type 2 DM w/ CKD', True),

            # Hypertension
            ('I10', 'Essential (primary) hypertension', 'HTN', True),
            ('I110', 'Hypertensive heart disease with heart failure', 'HTN heart disease w/ HF', True),

            # Heart failure
            ('I50', 'Heart failure', 'Heart failure', False),
            ('I5020', 'Unspecified systolic (congestive) heart failure', 'Systolic CHF', True),
            ('I5030', 'Unspecified diastolic (congestive) heart failure', 'Diastolic CHF', True),

            # Coronary artery disease
            ('I20', 'Angina pectoris', 'Angina', False),
            ('I200', 'Unstable angina', 'Unstable angina', True),
            ('I2589', 'Other forms of chronic ischemic heart disease', 'Other chronic IHD', True),

            # COPD
            ('J44', 'Other chronic obstructive pulmonary disease', 'COPD', False),
            ('J441', 'Chronic obstructive pulmonary disease with (acute) exacerbation', 'COPD exacerbation', True),

            # Pneumonia
            ('J189', 'Pneumonia, unspecified organism', 'Pneumonia NOS', True),

            # Atrial fibrillation
            ('I4891', 'Unspecified atrial fibrillation', 'A-fib', True),

            # CKD
            ('N18', 'Chronic kidney disease (CKD)', 'CKD', False),
            ('N183', 'Chronic kidney disease, stage 3', 'CKD stage 3', True),

            # Anemia
            ('D50', 'Iron deficiency anemia', 'Iron deficiency anemia', False),
            ('D509', 'Iron deficiency anemia, unspecified', 'IDA unspecified', True),
        ]

        df = pd.DataFrame(synthetic_codes, columns=[
            'code', 'long_description', 'short_description', 'is_billable'
        ])

        return df

    def _create_synthetic_hcpcs(self) -> pd.DataFrame:
        """Create synthetic HCPCS dataset for testing."""
        synthetic_codes = [
            ('99213', 'Office visit, established patient, low complexity', 'Office visit low', 'Medical Services'),
            ('99214', 'Office visit, established patient, moderate complexity', 'Office visit mod', 'Medical Services'),
            ('93000', 'Electrocardiogram, routine ECG with interpretation', 'ECG complete', 'Procedures/Services'),
            ('93005', 'Electrocardiogram, tracing only', 'ECG tracing', 'Procedures/Services'),
            ('80053', 'Comprehensive metabolic panel', 'CMP', 'Pathology/Laboratory'),
            ('85025', 'Complete blood count (CBC) with differential', 'CBC w/ diff', 'Pathology/Laboratory'),
            ('71046', 'Chest X-ray, 2 views', 'CXR 2 views', 'Diagnostic Radiology'),
            ('G0008', 'Administration of influenza virus vaccine', 'Flu vaccine admin', 'Procedures/Services'),
            ('J3301', 'Injection, triamcinolone acetonide, 10 mg', 'Triamcinolone inj', 'Drugs'),
        ]

        df = pd.DataFrame(synthetic_codes, columns=[
            'code', 'long_description', 'short_description', 'category'
        ])

        return df


def main():
    """Main function to download all codes."""
    downloader = CodeDownloader()

    print("\n" + "=" * 60)
    print("MEDICAL BILLING CODE DOWNLOADER")
    print("=" * 60 + "\n")

    # Download ICD-10 codes
    icd10_df = downloader.download_icd10_codes()
    print(f"\nICD-10 Summary:")
    print(f"  Total codes: {len(icd10_df)}")
    print(f"  Billable codes: {icd10_df['is_billable'].sum()}")

    # Download HCPCS codes
    hcpcs_df = downloader.download_hcpcs_codes()
    print(f"\nHCPCS Summary:")
    print(f"  Total codes: {len(hcpcs_df)}")
    print(f"  Categories: {hcpcs_df['category'].nunique()}")

    # Test NLM API
    print("\n" + "=" * 60)
    print("Testing NLM API Search")
    print("=" * 60)

    test_query = "diabetes"
    results = downloader.search_icd10_api(test_query)
    print(f"\nICD-10 search for '{test_query}':")
    for r in results[:5]:
        print(f"  {r['code']}: {r['description']}")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    return icd10_df, hcpcs_df


if __name__ == "__main__":
    main()
