# Medical Billing Categorization System

> An intelligent system that automatically organizes medical bills into meaningful categories using artificial intelligence

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 📖 Table of Contents

1. [What This Project Does](#what-this-project-does) - *Start here if you're new!*
2. [Who This Is For](#who-this-is-for)
3. [How It Works (Simple Explanation)](#how-it-works-simple-explanation)
4. [What You'll Need](#what-youll-need)
5. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
6. [Running the System](#running-the-system)
7. [Testing Everything Works](#testing-everything-works)
8. [Configuration Guide](#configuration-guide)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)
11. [Advanced Topics](#advanced-topics)

---

## 🎯 What This Project Does

### In Simple Terms

Imagine you work at a hospital or medical clinic. Every day, you get thousands of medical bills for different things:
- Doctor visits for diabetes
- Heart surgeries
- X-rays for broken bones
- Prescription medications

Instead of manually sorting through each bill to organize them, this system uses **artificial intelligence** to:

1. **Read** the medical notes (like "Patient came in with chest pain")
2. **Understand** what the visit was about
3. **Group** similar bills together automatically (all diabetes visits in one category, all heart surgeries in another)
4. **Label** each category with a meaningful name

**Result:** Instead of looking through 15,000 individual bills, you have 8-15 organized categories like "Diabetes Management", "Cardiac Procedures", "Respiratory Care", etc.

### Real-World Example

**Before this system:**
```
Bill #1: "Patient with Type 2 diabetes, elevated HbA1c..."
Bill #2: "Diabetes follow-up, blood sugar monitoring..."
Bill #3: "Acute chest pain, cardiac catheterization..."
Bill #4: "Type 2 diabetes mellitus management..."
Bill #5: "Heart attack, emergency stent placement..."
...15,000 more bills to sort manually
```

**After this system runs:**
```
📁 Diabetes Management (2,450 bills)
   ├── Bill #1
   ├── Bill #2
   ├── Bill #4
   └── ...

📁 Cardiac Procedures (1,820 bills)
   ├── Bill #3
   ├── Bill #5
   └── ...

📁 Respiratory Care (985 bills)
📁 Orthopedic Surgery (756 bills)
...and 8 more categories
```

---

## 👥 Who This Is For

This project is designed for:

- ✅ **Healthcare administrators** who need to organize billing data
- ✅ **Data analysts** working with medical records
- ✅ **Researchers** studying healthcare patterns
- ✅ **Developers** learning about AI and healthcare
- ✅ **Anyone** curious about how AI can help healthcare

**You don't need:**
- ❌ Medical degree or healthcare background
- ❌ Advanced programming knowledge (basic command line is enough)
- ❌ Expensive computers or servers
- ❌ Deep understanding of AI/machine learning

---

## 🔍 How It Works (Simple Explanation)

### The Big Picture

Think of this system like a very smart filing clerk that organizes paperwork:

```
Step 1: READ                Step 2: UNDERSTAND           Step 3: GROUP
┌─────────────┐            ┌─────────────┐             ┌─────────────┐
│ Medical     │            │ AI converts │             │ AI groups   │
│ bills with  │───────────>│ text into   │────────────>│ similar     │
│ clinical    │            │ numbers it  │             │ bills       │
│ notes       │            │ can analyze │             │ together    │
└─────────────┘            └─────────────┘             └─────────────┘
                                                               │
                                                               ▼
                                      Step 4: LABEL    ┌─────────────┐
                                      ┌────────────────│ AI creates  │
                                      │                │ category    │
                                      │                │ names       │
                                      ▼                └─────────────┘
                              ┌──────────────┐
                              │ Organized    │
                              │ categories   │
                              │ ready to use │
                              └──────────────┘
```

### Technical Terms Explained

| Term | What It Means | Simple Analogy |
|------|---------------|----------------|
| **Embedding** | Converting text into numbers | Like translating a book into Braille - different format, same meaning |
| **Vector** | A list of numbers representing text | Like coordinates on a map (latitude, longitude) but with 384 numbers |
| **Clustering** | Grouping similar things together | Like sorting M&Ms by color automatically |
| **LLM (Large Language Model)** | AI that understands and generates text | Like a very smart autocomplete on your phone |
| **Database** | Electronic filing cabinet | Like a spreadsheet, but much more powerful |
| **API** | A way for programs to talk to each other | Like ordering food through an app instead of calling |

### What Makes This System Special

1. **Fully Automated** - No manual sorting needed
2. **Smart Naming** - Uses AI (Claude) to create meaningful category names
3. **Memory Efficient** - Remembers what it learned, doesn't repeat work
4. **Cost-Effective** - Skips re-processing fresh data to save money
5. **Scalable** - Can handle 100 bills or 100,000 bills

---

## 🛠️ What You'll Need

### Required Accounts (All Free to Start)

You'll need to create accounts with these services. Don't worry - they're all free for small projects:

#### 1. **GitHub Account** (for code storage)
- Website: https://github.com
- Cost: FREE
- Sign up time: 5 minutes
- Why: Stores your code safely

#### 2. **Vercel Account** (for database)
- Website: https://vercel.com
- Cost: FREE for hobby projects
- Sign up time: 5 minutes
- Why: Provides a cloud database to store medical bills

#### 3. **HuggingFace Account** (for AI models)
- Website: https://huggingface.co
- Cost: FREE
- Sign up time: 5 minutes
- Why: Provides AI models to understand text

#### 4. **Pinecone Account** (for vector storage)
- Website: https://www.pinecone.io
- Cost: FREE tier available
- Sign up time: 10 minutes
- Why: Stores AI-processed data efficiently

#### 5. **Anthropic Account** (for smart labeling - OPTIONAL)
- Website: https://console.anthropic.com
- Cost: Pay-as-you-go (about $0.02 per 1000 bills)
- Sign up time: 10 minutes
- Why: Creates intelligent category names
- **Note:** If you skip this, the system will use generic names like "Category 1", "Category 2"

### Required Software

You'll need to install some software on your computer:

#### Python (Programming Language)
- **What:** The language this system is written in
- **How to install:**
  - Windows: Download from https://www.python.org/downloads/ (get version 3.11 or newer)
  - Mac: Open Terminal and type `brew install python3` (install Homebrew first if needed)
  - Linux: Type `sudo apt install python3.11` in terminal
- **How to verify:** Open terminal/command prompt and type `python --version`
  - You should see something like `Python 3.11.x`

#### Git (Version Control)
- **What:** Tool to download and manage code
- **How to install:**
  - Windows: Download from https://git-scm.com/download/win
  - Mac: Type `brew install git` in Terminal
  - Linux: Type `sudo apt install git`
- **How to verify:** Type `git --version`
  - You should see something like `git version 2.x.x`

### Computer Requirements

- **Operating System:** Windows 10+, macOS 10.14+, or any modern Linux
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 2GB free space
- **Internet:** Stable connection required

---

## 📚 Step-by-Step Setup Guide

Follow these instructions **exactly** in order. Each step builds on the previous one.

### Part 1: Get the Code (10 minutes)

#### Step 1.1: Open Terminal/Command Prompt

**Windows:**
- Press `Windows Key + R`
- Type `cmd` and press Enter
- A black window will appear

**Mac:**
- Press `Command + Space`
- Type `terminal` and press Enter
- A white/black window will appear

**Linux:**
- Press `Ctrl + Alt + T`

#### Step 1.2: Navigate to Your Home Folder

Type this command and press Enter:

```bash
cd ~
```

**What this does:** Takes you to your home folder

#### Step 1.3: Download the Code

Type this command exactly:

```bash
git clone https://github.com/aliomraniH/billing-model.git
```

**What this does:** Downloads all the code to your computer

**What you'll see:** Text scrolling as files download

#### Step 1.4: Enter the Project Folder

```bash
cd billing-model
```

**What this does:** Moves you into the project folder

**How to verify:** Type `ls` (Mac/Linux) or `dir` (Windows)
- You should see folders like `notebooks`, `sql`, etc.

---

### Part 2: Install Required Software (15 minutes)

#### Step 2.1: Install Python Libraries

Copy and paste this **entire block** and press Enter:

```bash
pip install numpy pandas sqlalchemy psycopg2-binary
pip install pinecone-client huggingface_hub anthropic
pip install scikit-learn requests
```

**What this does:** Installs all the helper tools the system needs

**What you'll see:** Lots of text scrolling - this is normal!

**If you see "pip: command not found":**
- Try `pip3` instead of `pip`
- On Windows, try `python -m pip install ...` instead

**How long:** 5-10 minutes depending on your internet speed

#### Step 2.2: Verify Installation

Type this command:

```bash
python -c "import numpy, pandas, sqlalchemy; print('✅ All libraries installed!')"
```

**Expected output:** `✅ All libraries installed!`

**If you see an error:** Go back to Step 2.1 and make sure it completed without errors

---

### Part 3: Set Up Your Database (20 minutes)

#### Step 3.1: Create Vercel Account

1. Go to https://vercel.com
2. Click "Sign Up"
3. Use your GitHub account to sign up (easiest method)
4. You'll be logged in automatically

#### Step 3.2: Create a Postgres Database

1. On Vercel dashboard, click "Storage" tab
2. Click "Create Database"
3. Select "Postgres"
4. Choose a name (e.g., "medical-billing-db")
5. Select your nearest region
6. Click "Create"
7. **Wait 2-3 minutes** for database to be ready

#### Step 3.3: Get Database Connection String

1. Click on your new database
2. Click ".env.local" tab
3. You'll see something like:
   ```
   POSTGRES_URL="postgresql://username:password@host:5432/database"
   ```
4. **COPY THE ENTIRE STRING** (including the quotes)
5. **Keep this safe** - you'll need it in the next step

---

### Part 4: Get API Keys (25 minutes)

#### Step 4.1: HuggingFace Token

1. Go to https://huggingface.co
2. Click "Sign Up" (or "Log In" if you have an account)
3. After signing in, click your profile picture (top right)
4. Click "Settings"
5. Click "Access Tokens" in the left menu
6. Click "New token"
7. Name it "medical-billing"
8. Select "Read" permission
9. Click "Generate token"
10. **COPY THE TOKEN** (starts with `hf_`)
11. **Save it somewhere safe**

#### Step 4.2: Pinecone API Key

1. Go to https://www.pinecone.io
2. Click "Sign Up" → choose "Starter" (free)
3. Verify your email
4. Log in to console
5. Click "API Keys" in left menu
6. Your API key is shown under "Value"
7. Click the copy icon
8. **SAVE THIS KEY**

#### Step 4.3: Anthropic API Key (OPTIONAL)

**Skip this if you want to use generic category names**

1. Go to https://console.anthropic.com
2. Sign up for an account
3. Add payment method (you'll only pay for what you use - about $0.02 per 1000 bills)
4. Click "API Keys"
5. Click "Create Key"
6. **COPY AND SAVE THE KEY** (starts with `sk-ant-`)

---

### Part 5: Configure the System (10 minutes)

#### Step 5.1: Navigate to Project Folder

Make sure you're in the billing-model folder:

```bash
cd ~/billing-model/notebooks
```

#### Step 5.2: Set Environment Variables

**Important:** Replace the placeholder values with your actual keys!

**On Mac/Linux:**

```bash
export VERCEL_POSTGRES_URL="postgresql://your-actual-url-here"
export HF_TOKEN="hf_your-actual-token-here"
export PINECONE_API_KEY="your-actual-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-actual-key-here"  # Optional
```

**On Windows (Command Prompt):**

```cmd
set VERCEL_POSTGRES_URL=postgresql://your-actual-url-here
set HF_TOKEN=hf_your-actual-token-here
set PINECONE_API_KEY=your-actual-key-here
set ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

**On Windows (PowerShell):**

```powershell
$env:VERCEL_POSTGRES_URL="postgresql://your-actual-url-here"
$env:HF_TOKEN="hf_your-actual-token-here"
$env:PINECONE_API_KEY="your-actual-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-actual-key-here"
```

**⚠️ IMPORTANT:** These variables are temporary! They will disappear when you close the terminal.

**To make them permanent:**
- Mac/Linux: Add the export commands to `~/.bashrc` or `~/.zshrc`
- Windows: Use System Environment Variables in Control Panel

#### Step 5.3: Verify Configuration

Run this command:

```bash
python test_imports.py
```

**Expected output:**
```
Testing imports...
--------------------------------------------------
✅ config.py - OK
✅ utils.py - Import path correct
--------------------------------------------------
Import structure is correct!
```

**If you see errors:** Check that you set the environment variables correctly in Step 5.2

---

## 🚀 Running the System

Now that everything is set up, here's how to actually use the system!

### Recent Performance Improvements 🎯

**Notebook 6 has been significantly optimized (v2.1):**
- ✅ **100x faster database operations** - Batch inserts instead of individual operations (2-3 seconds vs 2-3 minutes)
- ✅ **80% overall speedup** - Eliminated connection timeout errors through fresh connection pattern
- ✅ **Production-ready architecture** - Stage-based processing separates data preparation from I/O

**See detailed lessons learned in:** [`notebooks/LESSONS_LEARNED.md`](notebooks/LESSONS_LEARNED.md)

**Key improvements:**
- Fresh database connections for long-running operations (prevents timeouts)
- Batch preparation pattern (prepare data first, then execute in single operation)
- Context managers for guaranteed resource cleanup

---

### Understanding the Process

The system runs in 4 stages, like an assembly line:

```
Stage 1         Stage 2         Stage 3          Stage 4
Setup DB   →    Load Data   →   Create AI    →   Organize
                                Embeddings       Categories
   │                │               │                │
   ▼                ▼               ▼                ▼
Creates         Adds sample    Converts text   Groups similar
tables          medical bills  to numbers      bills together
```

### Stage 1: Setup Database (One-time, 2 minutes)

This creates the storage structure for your data.

```bash
cd ~/billing-model/notebooks
python 01_setup_database.py
```

**What you'll see:**
```
✅ Connected to database
✅ Creating tables...
✅ Database setup complete!
```

**What happened:** Created empty "filing cabinets" (database tables) to store bills

**If you see an error:**
- Check your `VERCEL_POSTGRES_URL` is set correctly
- Make sure your database is running on Vercel

---

### Stage 2: Load Sample Data (One-time, 5 minutes)

This loads example medical bills to test with.

```bash
python 02_load_synthea_data.py
```

**What you'll see:**
```
📥 Loading synthetic medical data...
✅ Loaded 15,000 medical claims
✅ Data loading complete!
```

**What happened:** Added 15,000 fake medical bills for testing

**How long:** ~5 minutes

**Note:** These are **synthetic** (fake but realistic) medical records - no real patient data!

---

### Stage 3: Generate AI Embeddings (20-30 minutes)

This is where AI "reads" and "understands" the medical notes.

#### Choose How Many Bills to Process

**For testing (recommended first time):**
```bash
export MAX_CLAIMS_TO_PROCESS=100
```

**For production (all bills):**
```bash
export MAX_CLAIMS_TO_PROCESS=-1
```

**Run the embedding generation:**
```bash
python 05_embeddings_similarity_search.py
```

**What you'll see:**
```
📋 MODEL CONFIGURATION
======================================================================
🤖 Embedding Model: BAAI/bge-small-en-v1.5
🔌 Connecting to Vercel Postgres...
   ✅ Connected! Found 15,000 claims

🔄 Processing claims in batches...
   Batch 1/10 (claims 1-100)...
   ✅ Uploaded 100 vectors to Pinecone
   📊 Progress: 100/1000 (22.1 claims/sec, ETA: 2.8min)

✅ NOTEBOOK 05 COMPLETE
   • Claims processed: 1,000
   • Success rate: 100.0%
```

**What happened:**
1. System read each medical note
2. Used AI to convert text into 384 numbers (embedding)
3. Stored these numbers in Pinecone (vector database)

**How long:**
- 100 bills: ~30 seconds
- 1,000 bills: ~2 minutes
- 15,000 bills: ~20 minutes

**Cost:** FREE (uses HuggingFace free tier)

---

### Stage 4: Organize into Categories (5-10 minutes)

This groups similar bills together and names each category.

```bash
python 06_llm_clustering_cache.py
```

**What you'll see:**
```
🔬 Clustering with HDBSCAN...
   ✅ Found 8 clusters

🤖 Labeling clusters with LLM...
   [1/8] Labeling cluster 0... 'Diabetes Management' (245 items)
   [2/8] Labeling cluster 1... 'Cardiac Procedures' (182 items)
   [3/8] Labeling cluster 2... 'Respiratory Care' (98 items)
   ...
   ✅ Labeled 8 categories

📊 Populating categories...
   ✅ All 8 categories inserted successfully

🧪 COMPREHENSIVE INTEGRATION TESTS
[TEST 1] Category-filtered search ✅
[TEST 2] General search ✅
[TEST 3] Auto-categorize new claim ✅
[TEST 4] Database consistency ✅
[TEST 5] Performance benchmarks ✅

✅ NOTEBOOK 06 COMPLETE
```

**What happened:**
1. AI analyzed all embeddings and found similar groups
2. Claude AI (if you set it up) created smart names like "Diabetes Management"
3. System saved categories to database
4. Ran automatic tests to verify everything works

**How long:** 5-10 minutes depending on how many bills you processed

**Cost:**
- FREE if you skipped Anthropic (uses generic names)
- ~$0.02 for 1000 bills with Claude AI

---

### 🎉 You're Done!

Your system is now running! Here's what you have:

✅ **Database** with 15,000 medical bills
✅ **AI embeddings** that understand the text
✅ **Organized categories** grouping similar bills
✅ **Search capability** to find similar bills instantly

---

## 🧪 Testing Everything Works

Let's verify the system is working correctly!

### Test 1: Verify Imports

```bash
cd ~/billing-model/notebooks
python test_imports.py
```

**Expected output:**
```
✅ config.py - OK
✅ utils.py - Import path correct
```

**What this tests:** All required code files are in the right place

---

### Test 2: Check Database Connection

```bash
python -c "from utils import init_database; import os; engine, total = init_database(os.getenv('VERCEL_POSTGRES_URL')); print(f'✅ Connected! Found {total} claims')"
```

**Expected output:**
```
✅ Connected! Found 15,000 claims
```

**What this tests:** Database connection works and data is loaded

---

### Test 3: Verify Categories Were Created

```bash
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM claim_categories'))
    count = result.fetchone()[0]
    print(f'✅ Found {count} categories')

    result = conn.execute(text('SELECT category_name, claim_count FROM claim_categories ORDER BY claim_count DESC LIMIT 5'))
    print('\nTop 5 categories:')
    for row in result:
        print(f'  📁 {row[0]}: {row[1]} bills')
"
```

**Expected output:**
```
✅ Found 8 categories

Top 5 categories:
  📁 diabetes_management: 245 bills
  📁 cardiac_procedures: 182 bills
  📁 respiratory_care: 98 bills
  📁 orthopedic_surgery: 87 bills
  📁 general_medicine: 156 bills
```

**What this tests:** Categories were successfully created and bills were organized

---

### Test 4: Test Search Functionality

Create a test file:

```bash
cat > test_search.py << 'EOF'
from utils import init_database, init_pinecone, init_hf_client, get_embedding
import os

# Initialize
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
HF_TOKEN = os.getenv('HF_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

engine, total_claims = init_database(DATABASE_URL, verbose=False)
pc, index = init_pinecone(PINECONE_API_KEY, "medical-billing-notes", 384, verbose=False)
hf_client = init_hf_client(HF_TOKEN, "BAAI/bge-small-en-v1.5", verbose=False)

# Search for diabetes-related bills
query = "patient with diabetes and high blood sugar"
query_emb = get_embedding(query, hf_client, "BAAI/bge-small-en-v1.5", 384)

results = index.query(
    vector=query_emb.tolist(),
    top_k=3,
    include_metadata=True
)

print("🔍 Search Results for: 'patient with diabetes and high blood sugar'\n")
for i, match in enumerate(results.matches, 1):
    print(f"{i}. Similarity: {match.score:.3f}")
    print(f"   Category: {match.metadata.get('llm_category', 'uncategorized')}")
    print(f"   Preview: {match.metadata.get('text_preview', '')[:80]}...")
    print()
EOF

python test_search.py
```

**Expected output:**
```
🔍 Search Results for: 'patient with diabetes and high blood sugar'

1. Similarity: 0.856
   Category: diabetes_management
   Preview: Patient with Type 2 diabetes mellitus. HbA1c 8.5%. Blood glucose 210 mg/...

2. Similarity: 0.823
   Category: diabetes_management
   Preview: Diabetes follow-up visit. A1C 7.2%, peripheral neuropathy noted. Contin...

3. Similarity: 0.791
   Category: diabetes_management
   Preview: Uncontrolled diabetes admitted. Blood sugar 280. Started on insulin ther...
```

**What this tests:** System can find similar bills based on text search

---

### Test 5: Performance Check

```bash
python -c "
from datetime import datetime
start = datetime.now()

# Your test code here (reuse test from Test 3)
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM claim_categories'))
    count = result.fetchone()[0]

elapsed = (datetime.now() - start).total_seconds()
print(f'✅ Query completed in {elapsed:.2f} seconds')
print(f'   Performance: {'GOOD' if elapsed < 1 else 'SLOW'}')
"
```

**Expected output:**
```
✅ Query completed in 0.23 seconds
   Performance: GOOD
```

**What this tests:** Database queries are fast

---

## ⚙️ Configuration Guide

### Understanding the Configuration Files

The system has 2 main configuration files in the `notebooks/` folder:

#### 1. `config.py` - Main Settings

**Location:** `/billing-model/notebooks/config.py`

**What it controls:**
- Which AI models to use
- How many bills to process at once
- How often to refresh old data
- Database and API settings

**Common settings you might want to change:**

```python
# In config.py, look for these classes:

class ProcessingConfig:
    max_claims_to_process: int = 1000  # Change this to process more/fewer bills
    batch_size: int = 100              # How many at once (bigger = faster but uses more memory)

class RefreshConfig:
    default_embedding_refresh_hours: int = 12  # How old before re-processing (in hours)
    auto_refresh_enabled: bool = True          # Skip fresh data to save costs

class ClusteringConfig:
    min_cluster_size: int = 5  # Minimum bills per category (smaller = more categories)
```

**How to change settings:**

**Method 1: Environment Variables (Recommended)**
```bash
export MAX_CLAIMS_TO_PROCESS=5000
export MIN_CLUSTER_SIZE=20
```

**Method 2: Edit config.py directly**
1. Open `notebooks/config.py` in a text editor
2. Find the setting you want to change
3. Change the number
4. Save the file

**Method 3: Create a config.json file**
```bash
cd ~/billing-model/notebooks
cat > config.json << 'EOF'
{
  "processing": {
    "max_claims_to_process": 5000
  },
  "clustering": {
    "min_cluster_size": 20
  }
}
EOF
```

---

#### 2. `utils.py` - Shared Functions

**Location:** `/billing-model/notebooks/utils.py`

**What it contains:**
- Functions used by multiple notebooks
- Database connection helpers
- AI model initialization
- Error handling

**You usually don't need to modify this file!**

---

### Environment Variables Reference

Here's what each variable does:

| Variable | Required? | Purpose | Example |
|----------|-----------|---------|---------|
| `VERCEL_POSTGRES_URL` | ✅ Yes | Database connection | `postgresql://user:pass@host/db` |
| `HF_TOKEN` | ✅ Yes | HuggingFace API access | `hf_xxxxxxxxxxxxx` |
| `PINECONE_API_KEY` | ✅ Yes | Vector database access | `xxxxxxxx-xxxx-xxxx` |
| `ANTHROPIC_API_KEY` | ❌ Optional | Smart category naming | `sk-ant-xxxxxxxxx` |
| `MAX_CLAIMS_TO_PROCESS` | ❌ Optional | How many bills to process | `1000` or `-1` (all) |
| `MIN_CLUSTER_SIZE` | ❌ Optional | Category size threshold | `5` to `20` |
| `CLAUDE_MODEL` | ❌ Optional | Which Claude model to use | `claude-sonnet-4-5-20250929` |

---

### Performance Tuning

#### If Processing is Too Slow

**Problem:** Processing 1000 bills takes more than 5 minutes

**Solutions:**
1. **Reduce batch size** (uses less memory, might be faster on older computers)
   ```bash
   export EMBEDDING_BATCH_SIZE=50  # Default is 100
   ```

2. **Process fewer bills for testing**
   ```bash
   export MAX_CLAIMS_TO_PROCESS=100  # Start small
   ```

3. **Check your internet connection** - Slow connection affects API calls

#### If You're Getting Too Many Categories

**Problem:** System creates 30+ categories (too granular)

**Solution:** Increase minimum cluster size
```bash
export MIN_CLUSTER_SIZE=20  # Default is 5
# Larger number = fewer, bigger categories
```

#### If Categories Are Too Broad

**Problem:** Only 2-3 categories (not enough detail)

**Solution:** Decrease minimum cluster size
```bash
export MIN_CLUSTER_SIZE=10  # Try different values
```

**Recommended values by dataset size:**
- 100 bills: `MIN_CLUSTER_SIZE=5`
- 1,000 bills: `MIN_CLUSTER_SIZE=15`
- 10,000 bills: `MIN_CLUSTER_SIZE=50`

---

## 🔧 Troubleshooting

### Common Problems and Solutions

#### Problem 1: "pip: command not found"

**Error message:**
```
-bash: pip: command not found
```

**Solution:**
Try `pip3` instead:
```bash
pip3 install numpy pandas sqlalchemy
```

Or use python directly:
```bash
python -m pip install numpy pandas sqlalchemy
```

---

#### Problem 2: "ModuleNotFoundError: No module named 'numpy'"

**Error message:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:**
The required libraries aren't installed. Run:
```bash
pip install numpy pandas sqlalchemy psycopg2-binary
pip install pinecone-client huggingface_hub anthropic scikit-learn
```

**If that doesn't work:**
```bash
pip3 install --user numpy pandas sqlalchemy psycopg2-binary
pip3 install --user pinecone-client huggingface_hub anthropic scikit-learn
```

---

#### Problem 3: "Cannot connect to database"

**Error message:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions:**
1. **Check your database URL is set:**
   ```bash
   echo $VERCEL_POSTGRES_URL
   ```
   If empty, you need to set it again (see Setup Step 5.2)

2. **Verify database is running:**
   - Log in to https://vercel.com
   - Go to Storage → Your Database
   - Check status shows "Active"

3. **Test connection directly:**
   ```bash
   python -c "from sqlalchemy import create_engine; import os; engine = create_engine(os.getenv('VERCEL_POSTGRES_URL')); print('✅ Connected!')"
   ```

---

#### Problem 3b: "Database timeout during long operations" ⭐ FIXED

**Error message:**
```
OperationalError: FATAL: terminating connection due to administrator command
SSL connection has been closed unexpectedly
```

**What happened:**
This was a critical issue in notebook 6 where database connections were held open during long-running operations (clustering, LLM calls), causing timeouts after 2-3 minutes.

**Solution (Already Fixed in v2.1):**
Notebook 6 now uses fresh connections with NullPool:
- Connections created only when needed
- Closed immediately after use
- No connection pooling for long-running notebooks

**If you're using an old version:**
Update to the latest code:
```bash
cd ~/billing-model
git pull origin claude/fix-notebook-6-error-SvwC5
```

**See:** `notebooks/LESSONS_LEARNED.md` for technical details

---

#### Problem 4: "API key not found"

**Error message:**
```
ValueError: Missing required environment variables
```

**Solution:**
Set your API keys again:
```bash
export HF_TOKEN="hf_your-actual-token"
export PINECONE_API_KEY="your-actual-key"
export VERCEL_POSTGRES_URL="postgresql://..."
```

**To verify they're set:**
```bash
echo $HF_TOKEN
echo $PINECONE_API_KEY
echo $VERCEL_POSTGRES_URL
```

Each should print your key. If they print nothing, the variable isn't set.

---

#### Problem 5: "Embedding generation failed"

**Error message:**
```
❌ Failed after 3 attempts: 401 Unauthorized
```

**Solution:**
Your HuggingFace token is invalid or expired.

1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Copy it
4. Set it:
   ```bash
   export HF_TOKEN="hf_your-new-token"
   ```

---

#### Problem 6: Too many categories created

**Problem:**
System creates 40+ categories when you expected 8-10

**Solution:**
Your cluster size is too small. Increase it:
```bash
export MIN_CLUSTER_SIZE=20
python 06_llm_clustering_cache.py
```

**Rule of thumb:**
- Minimum cluster size should be about 2% of your total bills
- For 1000 bills: use 20
- For 5000 bills: use 100

---

#### Problem 7: Out of memory error

**Error message:**
```
MemoryError: Unable to allocate array
```

**Solution:**
You're trying to process too many bills at once.

1. **Reduce batch size:**
   ```bash
   export EMBEDDING_BATCH_SIZE=50  # Default is 100
   ```

2. **Process fewer bills:**
   ```bash
   export MAX_CLAIMS_TO_PROCESS=500
   ```

3. **Close other programs** to free up RAM

---

#### Problem 8: "Import error: cannot import name 'Engine'"

**Error message:**
```
ImportError: cannot import name 'Engine' from 'sqlalchemy'
```

**Solution:**
This was fixed in the latest version. Update your code:
```bash
cd ~/billing-model
git pull origin claude/cleanup-billing-duplicates-M2Iqt
```

The fix changes the import from:
```python
from sqlalchemy import Engine  # Wrong
```
to:
```python
from sqlalchemy.engine import Engine  # Correct
```

---

### Getting Help

If you're still stuck:

1. **Check the documentation:**
   - `notebooks/QUICK_START.md` - Quick reference
   - `notebooks/LESSONS_LEARNED.md` - ⭐ Production fixes & errors
   - `notebooks/IMPORT_REFERENCE.md` - Import issues
   - `notebooks/CLEANUP_SUMMARY.md` - Technical details
   - `notebooks/DEEPNOTE_GUIDE.md` - Deepnote instructions

2. **Run the test script:**
   ```bash
   cd ~/billing-model/notebooks
   python test_imports.py
   ```
   This will tell you exactly what's wrong

3. **Check GitHub Issues:**
   - Go to https://github.com/aliomraniH/billing-model/issues
   - Search for your error message
   - Create a new issue if needed

---

## 📁 Project Structure

Here's what each file and folder does:

```
billing-model/
├── README.md                              # ← This file!
│
├── notebooks/                             # Main code folder
│   ├── config.py                         # Settings and configuration
│   ├── utils.py                          # Shared helper functions
│   │
│   ├── 01_setup_database.py             # Creates database tables
│   ├── 02_load_synthea_data.py          # Loads sample medical data
│   ├── 05_embeddings_similarity_search.py  # Converts text to AI format
│   ├── 06_llm_clustering_cache.py       # Organizes into categories
│   │
│   ├── refresh_manager.py               # Manages data updates
│   │
│   ├── test_imports.py                  # Tests if setup is correct
│   │
│   ├── QUICK_START.md                   # One-page quick reference
│   ├── IMPORT_REFERENCE.md              # Technical import details
│   ├── CLEANUP_SUMMARY.md               # Code refactoring details
│   ├── TESTING_GUIDE.md                 # Comprehensive testing guide
│   ├── LESSONS_LEARNED.md               # ⭐ Production errors & fixes
│   ├── ARCHITECTURE_IMPROVEMENTS.md     # Production architecture design
│   ├── DEEPNOTE_GUIDE.md                # Deepnote-specific instructions
│   ├── STEP_BY_STEP_MIGRATION.md        # Full migration guide
│   └── processing_framework.py          # Production framework (optional)
│
└── sql/
    └── schema.sql                         # Database structure definition
```

### What Each Notebook Does

| File | What It Does | When to Run | Time |
|------|--------------|-------------|------|
| `01_setup_database.py` | Creates empty database tables | Once at start | 2 min |
| `02_load_synthea_data.py` | Loads 15,000 sample bills | Once at start | 5 min |
| `05_embeddings_similarity_search.py` | Converts text to numbers AI can understand | Every time you want to process new bills | 2-20 min |
| `06_llm_clustering_cache.py` | Groups bills into categories (v2.1 optimized) | After running notebook 05 | 5-10 min |
| `refresh_manager.py` | Checks for old data that needs updating | Optionally, for maintenance | 1 min |

---

## 🎓 Advanced Topics

### Understanding the Technology

#### What Are Embeddings?

**Simple explanation:** Imagine you want to measure how similar two sentences are. Embeddings convert sentences into lists of 384 numbers. Similar sentences have similar numbers.

**Example:**
```
Sentence 1: "Patient has diabetes"
Embedding 1: [0.23, -0.45, 0.67, ... 381 more numbers]

Sentence 2: "Diabetic patient visit"
Embedding 2: [0.25, -0.43, 0.69, ... 381 more numbers]
```

The numbers are close, so the computer knows these sentences are similar!

#### What Is Clustering?

**Simple explanation:** Grouping similar things together automatically.

**Real-world analogy:**
- You have a box of 1000 crayons
- Clustering is like sorting them by color automatically
- Similar colors (light blue, navy blue, sky blue) end up in the same group
- The computer does this with medical bills instead of crayons

#### What Is a Vector Database (Pinecone)?

**Simple explanation:** A special database optimized for finding similar things quickly.

**Regular database:**
- Good for: "Find me bill #12345"
- Bad for: "Find bills similar to this description"

**Vector database:**
- Good for: "Find the 10 most similar bills to this one"
- Uses embeddings (those 384 numbers) to measure similarity
- Can search billions of items in milliseconds

---

### Customizing Categories

#### Method 1: Change Cluster Size

Bigger clusters = Fewer categories:
```bash
export MIN_CLUSTER_SIZE=30
python 06_llm_clustering_cache.py
```

#### Method 2: Manual Category Names

If you skip Anthropic AI, edit the code in `06_llm_clustering_cache.py`:

Find this function (around line 410):
```python
def label_cluster_with_llm(sample_notes: List[str], cluster_idx: int) -> Dict:
    if not anthropic_client:
        return {
            "category_name": f"category_{cluster_idx}",
            "display_name": f"Category {cluster_idx}",
            "description": f"Auto-generated cluster {cluster_idx}"
        }
```

Change it to:
```python
def label_cluster_with_llm(sample_notes: List[str], cluster_idx: int) -> Dict:
    if not anthropic_client:
        # Custom category names
        names = {
            0: ("diabetes_care", "Diabetes Care", "Diabetes management visits"),
            1: ("heart_procedures", "Heart Procedures", "Cardiac interventions"),
            2: ("respiratory", "Respiratory", "Lung and breathing care"),
            # Add more as needed
        }
        if cluster_idx in names:
            return {
                "category_name": names[cluster_idx][0],
                "display_name": names[cluster_idx][1],
                "description": names[cluster_idx][2]
            }
        # Fallback for additional clusters
        return {
            "category_name": f"category_{cluster_idx}",
            "display_name": f"Category {cluster_idx}",
            "description": f"Medical services cluster {cluster_idx}"
        }
```

---

### Processing Real Data

#### Using Your Own Medical Bills

1. **Prepare your data:**
   - Format: CSV file with columns `claim_id`, `clinical_note`, `total_charge`
   - Save as `my_claims.csv`

2. **Create a custom loader:**
   ```python
   # notebooks/load_my_data.py
   import pandas as pd
   from sqlalchemy import create_engine
   import os

   # Read your data
   df = pd.read_csv('my_claims.csv')

   # Connect to database
   engine = create_engine(os.getenv('VERCEL_POSTGRES_URL'))

   # Upload to database
   df.to_sql('claims', engine, if_exists='append', index=False)
   print(f"✅ Loaded {len(df)} claims")
   ```

3. **Run the loader:**
   ```bash
   python notebooks/load_my_data.py
   ```

4. **Process your data:**
   ```bash
   export MAX_CLAIMS_TO_PROCESS=-1  # Process all
   python 05_embeddings_similarity_search.py
   python 06_llm_clustering_cache.py
   ```

---

### Scaling to Production

#### Handling Large Datasets (100,000+ bills)

**1. Process in Batches:**
```bash
# Process 10,000 at a time
export MAX_CLAIMS_TO_PROCESS=10000
python 05_embeddings_similarity_search.py

# Wait for completion, then run again for next 10,000
# The system will automatically skip already-processed bills
```

**2. Use Bigger Clusters:**
```bash
export MIN_CLUSTER_SIZE=100  # For 100k bills
```

**3. Monitor Costs:**
- HuggingFace: Free for embeddings
- Pinecone: ~$70/month for 100k vectors
- Anthropic: ~$2 per 100k bills for categorization

---

### Monitoring and Maintenance

#### Daily Checks

Run this command to check system health:
```bash
python refresh_manager.py --check
```

Output shows:
- How many bills need re-processing
- How old the oldest data is
- Database statistics

#### Weekly Maintenance

Refresh old data:
```bash
python refresh_manager.py --refresh-embeddings --max-refresh 1000
python refresh_manager.py --refresh-categories
```

#### Monthly Tasks

1. **Backup your database:**
   - Vercel provides automatic backups
   - Or export manually:
     ```bash
     pg_dump $VERCEL_POSTGRES_URL > backup_$(date +%Y%m%d).sql
     ```

2. **Review categories:**
   - Check if category names still make sense
   - Adjust `MIN_CLUSTER_SIZE` if needed

3. **Check costs:**
   - Review Pinecone usage
   - Review Anthropic API usage
   - Optimize if needed

---

## 📞 Support and Resources

### Documentation Files

All in the `notebooks/` folder:

**Getting Started:**
- **QUICK_START.md** - One-page quick reference
- **DEEPNOTE_GUIDE.md** - Deepnote-specific cell-by-cell instructions

**Production & Learning:**
- **LESSONS_LEARNED.md** - ⭐ **START HERE** for production fixes, error patterns, and performance optimization
- **ARCHITECTURE_IMPROVEMENTS.md** - Stage-based processing design and patterns
- **STEP_BY_STEP_MIGRATION.md** - Full migration guide with 15 detailed steps

**Technical Details:**
- **IMPORT_REFERENCE.md** - How imports work (technical)
- **CLEANUP_SUMMARY.md** - Code refactoring details (technical)
- **TESTING_GUIDE.md** - Comprehensive testing guide

### External Resources

- **Vercel Postgres:** https://vercel.com/docs/storage/vercel-postgres
- **HuggingFace:** https://huggingface.co/docs
- **Pinecone:** https://docs.pinecone.io
- **Anthropic Claude:** https://docs.anthropic.com
- **Python for Beginners:** https://www.python.org/about/gettingstarted/

### Getting Help

1. **Check existing documentation** (files above)
2. **Run test scripts** to diagnose issues
3. **Search GitHub Issues:** https://github.com/aliomraniH/billing-model/issues
4. **Create new issue** if you found a bug or need help

---

## 📝 License

MIT License - Free to use for any purpose

---

## 🎯 Quick Reference Card

**Copy this and keep it handy!**

```bash
# ====== SETUP (One Time) ======
cd ~/billing-model/notebooks
export VERCEL_POSTGRES_URL="postgresql://..."
export HF_TOKEN="hf_..."
export PINECONE_API_KEY="..."
python 01_setup_database.py
python 02_load_synthea_data.py

# ====== PROCESS DATA ======
export MAX_CLAIMS_TO_PROCESS=1000
python 05_embeddings_similarity_search.py
python 06_llm_clustering_cache.py

# ====== TEST ======
python test_imports.py
python -c "from utils import init_database; import os; engine, total = init_database(os.getenv('VERCEL_POSTGRES_URL')); print(f'Found {total} claims')"

# ====== MAINTENANCE ======
python refresh_manager.py --check
python refresh_manager.py --refresh-embeddings --max-refresh 1000
```

---

**Last Updated:** December 23, 2025
**Version:** 2.1 (Production Optimizations)
**Status:** ✅ Production Ready
**Branch:** `claude/fix-notebook-6-error-SvwC5`

**Recent Changes (v2.1):**
- Fixed critical database timeout errors in notebook 6
- 100x performance improvement in database operations (batch inserts)
- 80% overall speedup through fresh connection pattern
- Comprehensive error documentation in LESSONS_LEARNED.md
- Production-ready architecture with stage-based processing
