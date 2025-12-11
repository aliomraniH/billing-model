# Medical Billing ML - Notebook 4: Hugging Face AutoTrain Workflow
# Prerequisites: Run notebooks 01-03 first

# Install & Setup Hugging Face
!pip install -q autotrain-advanced

import os
from sqlalchemy import create_engine, text
from huggingface_hub import login, HfApi
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

# Get Hugging Face token
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("""
    ❌ HF_TOKEN not found!

    📝 To fix:
    1. Go to https://huggingface.co/settings/tokens
    2. Create new token with WRITE permissions
    3. In Deepnote: Settings → Environment Variables
    4. Add: HF_TOKEN = hf_xxxxxxxxxxxxx
    """)

# Login to Hugging Face
login(token=HF_TOKEN)
print("✅ Logged in to Hugging Face!")

# Connect to database
DATABASE_URL = os.getenv('VERCEL_POSTGRES_URL')
if not DATABASE_URL:
    raise ValueError("VERCEL_POSTGRES_URL not found! Add it to Project Settings → Environment Variables")

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM claims WHERE outlier_score IS NOT NULL"))
        labeled_claims = result.fetchone()[0]
        print(f"✅ Connected! Found {labeled_claims:,} labeled claims")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# Prepare & Upload Dataset
print("\n📊 Preparing dataset for training...")
training_query = """
SELECT
    total_charge,
    total_paid,
    CASE WHEN total_charge > 0 THEN total_paid / total_charge ELSE 0 END as payment_ratio,
    claim_type,
    COALESCE(is_outlier, FALSE)::int as label
FROM claims
WHERE outlier_score IS NOT NULL AND total_charge > 0
"""

df = pd.read_sql(training_query, engine)
print(f"   Loaded {len(df):,} samples")
print(f"   Class distribution: {df['label'].value_counts().to_dict()}")

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['claim_type'], prefix='type')

# Split train/test
train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.2,
    random_state=42,
    stratify=df_encoded['label']
)
print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")

# Create HuggingFace dataset
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df.reset_index(drop=True))
})

# Upload to HuggingFace Hub
HF_USERNAME = HfApi().whoami()['name']
DATASET_NAME = f"{HF_USERNAME}/medical-billing-outliers"

print(f"\n📤 Uploading dataset to Hugging Face...")
dataset_dict.push_to_hub(DATASET_NAME, private=True)
print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{DATASET_NAME}")

# Train XGBoost Locally
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

print("\n🤖 Training XGBoost model...")

feature_cols = [c for c in train_df.columns if c != 'label']
X_train, y_train = train_df[feature_cols], train_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

# Calculate class weight for imbalance
scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
print(f"   Features: {len(feature_cols)}")
print(f"   Class weight: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Evaluate model
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("📊 MODEL EVALUATION")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Outlier']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("="*60)

# Upload Model to Hub
import json

print("\n📤 Uploading model to Hugging Face...")

model_dir = '/work/medical_outlier_model'
os.makedirs(model_dir, exist_ok=True)

# Save model
xgb_model.save_model(f'{model_dir}/model.json')

# Save config
config = {
    'model_type': 'xgboost',
    'task': 'binary_classification',
    'features': feature_cols,
    'roc_auc': float(roc_auc_score(y_test, y_prob)),
    'n_features': len(feature_cols),
    'class_distribution': df['label'].value_counts().to_dict()
}

with open(f'{model_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Upload to HuggingFace
api = HfApi()
MODEL_REPO = f"{HF_USERNAME}/medical-outlier-detector"
api.create_repo(repo_id=MODEL_REPO, exist_ok=True, private=True)
api.upload_folder(folder_path=model_dir, repo_id=MODEL_REPO, repo_type="model")

print(f"✅ Model uploaded to: https://huggingface.co/{MODEL_REPO}")
print(f"📦 Model size: {sum(os.path.getsize(f'{model_dir}/{f}') for f in os.listdir(model_dir)) / 1024:.1f} KB")
