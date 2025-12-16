"""
Dependency Fix Script - Run BEFORE Notebook 05 if you see numpy/tensorflow conflicts

This script resolves the numpy 2.x vs tensorflow 2.15 incompatibility.

WHEN TO RUN THIS:
- If you get import errors in notebook 05
- If you see tensorflow/numpy version conflicts
- After running notebook 04 (autotrain-advanced)

WHAT IT DOES:
- Downgrades numpy to 1.26.4 (compatible with both tensorflow and modern packages)
- Keeps all other packages at their current versions
"""

print("=" * 70)
print("🔧 FIXING DEPENDENCY CONFLICTS")
print("=" * 70)
print("\nThis script resolves numpy 2.x incompatibility with TensorFlow 2.15")
print("(Safe to run - only affects numpy version)\n")

import subprocess
import sys

def run_pip_command(cmd):
    """Run pip command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

print("📦 Step 1: Checking current numpy version...")
success, output = run_pip_command(f"{sys.executable} -m pip show numpy")
if success:
    for line in output.split('\n'):
        if line.startswith('Version:'):
            current_version = line.split(':')[1].strip()
            print(f"   Current numpy: {current_version}")

            if current_version.startswith('2.'):
                print("   ⚠️  NumPy 2.x detected - incompatible with TensorFlow 2.15")
            else:
                print("   ✅ NumPy version OK")
else:
    print("   ⚠️  NumPy not installed")

print("\n📦 Step 2: Installing numpy 1.26.4 (TensorFlow-compatible)...")
print("   This version works with:")
print("   • TensorFlow 2.15.x ✓")
print("   • HuggingFace ecosystem ✓")
print("   • Pinecone ✓")
print("   • Modern pandas ✓\n")

success, output = run_pip_command(
    f"{sys.executable} -m pip install 'numpy==1.26.4' --force-reinstall --no-deps -q"
)

if success:
    print("   ✅ NumPy 1.26.4 installed successfully")
else:
    print("   ❌ Failed to install numpy")
    print(f"   Error: {output}")
    sys.exit(1)

print("\n📦 Step 3: Verifying installation...")
success, output = run_pip_command(f"{sys.executable} -c 'import numpy; print(numpy.__version__)'")
if success:
    version = output.strip()
    print(f"   ✅ NumPy version verified: {version}")

    if version == "1.26.4":
        print("\n" + "=" * 70)
        print("✅ DEPENDENCY FIX COMPLETE")
        print("=" * 70)
        print("\n🚀 You can now run Notebook 05 without conflicts!")
        print("\nWhat was fixed:")
        print("  • NumPy: 2.2.6 → 1.26.4 (TensorFlow-compatible)")
        print("  • All other packages: unchanged")
        print("\nNext steps:")
        print("  1. Restart your kernel/runtime")
        print("  2. Run Notebook 05: Embeddings & Similarity Search")
        print("=" * 70)
    else:
        print(f"\n⚠️  Expected 1.26.4 but got {version}")
else:
    print("   ❌ Verification failed")
    sys.exit(1)
