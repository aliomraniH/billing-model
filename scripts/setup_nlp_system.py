#!/usr/bin/env python3
"""
NLP System Setup Script
========================
Automated installation of all NLP dependencies for cloud environments.
Safe to run multiple times (idempotent).

Usage:
    python scripts/setup_nlp_system.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Ensure Python 3.7+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required. You have {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

def main():
    print("=" * 60)
    print("🚀 Medical Billing NLP - Automated Setup")
    print("=" * 60)

    # Check Python version
    check_python_version()

    # Install core packages
    packages = [
        ("pip install --upgrade pip", "Upgrade pip"),
        ("pip install medspacy>=1.2.0", "Install medspacy"),
        ("pip install scispacy>=0.5.4", "Install scispacy"),
        ("pip install spacy>=3.7.0", "Install spacy"),
        ("pip install sentence-transformers>=2.2.2", "Install sentence-transformers"),
        ("pip install transformers>=4.48.0", "Install transformers"),
        ("pip install huggingface_hub>=0.20.0", "Install huggingface_hub"),
        ("pip install pgvector>=0.2.4", "Install pgvector"),
        ("pip install presidio-analyzer>=2.2.0", "Install presidio-analyzer"),
        ("pip install presidio-anonymizer>=2.2.0", "Install presidio-anonymizer"),
    ]

    results = []
    for cmd, desc in packages:
        results.append(run_command(cmd, desc))

    # Install spaCy models
    spacy_models = [
        ("python -m spacy download en_core_web_sm", "Download en_core_web_sm"),
        ("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz",
         "Download en_core_sci_md"),
    ]

    for cmd, desc in spacy_models:
        results.append(run_command(cmd, desc))

    # Summary
    print("\n" + "=" * 60)
    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print("✅ SETUP COMPLETE - All dependencies installed")
        print("\nNext steps:")
        print("  1. Verify: python notebooks/00_verify_nlp_dependencies.py")
        print("  2. Setup KB: python notebooks/07_nlp_knowledge_base_setup.py")
    else:
        print(f"⚠️  SETUP INCOMPLETE - {success_count}/{total_count} succeeded")
        print("\nPlease review errors above and retry failed steps manually.")

    print("=" * 60)

if __name__ == "__main__":
    main()
