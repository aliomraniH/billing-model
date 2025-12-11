# Medical Billing ML - Notebook 4: Hugging Face AutoTrain Workflow
# Prerequisites: Run notebooks 01-03 first

#@title 1️⃣ Install & Setup Hugging Face
!pip install -q autotrain-advanced
from google.colab import userdata
from sqlalchemy import create_engine
from huggingface_hub import login, HfApi
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN)
print("✅ Logged in to Hugging Face!")

DATABASE_URL = userdata.get('VERCEL_POSTGRES_URL')
engine = create_engine(DATABASE_URL)

#@title 2️⃣ Prepare & Upload Dataset
training_query = """
SELECT total_charge, total_paid,
       CASE WHEN total_charge > 0 THEN total_paid / total_charge ELSE 0 END as payment_ratio,
       claim_type,
       COALESCE(is_outlier, FALSE)::int as label
FROM claims WHERE outlier_score IS NOT NULL AND total_charge > 0
"""

df = pd.read_sql(training_query, engine)
df_encoded = pd.get_dummies(df, columns=['claim_type'], prefix='type')

train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_encoded['label'])

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df.reset_index(drop=True))
})

HF_USERNAME = HfApi().whoami()['name']
DATASET_NAME = f"{HF_USERNAME}/medical-billing-outliers"
dataset_dict.push_to_hub(DATASET_NAME, private=True)
print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{DATASET_NAME}")

#@title 3️⃣ Train XGBoost Locally (Quick)
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

feature_cols = [c for c in train_df.columns if c != 'label']
X_train, y_train = train_df[feature_cols], train_df['label']
X_test, y_test = test_df[feature_cols], test_df['label']

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\n📊 MODEL EVALUATION")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Outlier']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

#@title 4️⃣ Upload Model to Hub
import os, json

model_dir = '/content/medical_outlier_model'
os.makedirs(model_dir, exist_ok=True)
xgb_model.save_model(f'{model_dir}/model.json')

with open(f'{model_dir}/config.json', 'w') as f:
    json.dump({'model_type': 'xgboost', 'task': 'binary_classification',
               'features': feature_cols, 'roc_auc': float(roc_auc_score(y_test, y_prob))}, f)

api = HfApi()
MODEL_REPO = f"{HF_USERNAME}/medical-outlier-detector"
api.create_repo(repo_id=MODEL_REPO, exist_ok=True, private=True)
api.upload_folder(folder_path=model_dir, repo_id=MODEL_REPO, repo_type="model")
print(f"✅ Model uploaded to: https://huggingface.co/{MODEL_REPO}")
