# ML Assignment 2 - Training Script
# Student Performance Dataset

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# load the dataset
print("Loading dataset...")
df = pd.read_csv('data/student_performance.csv')
print(f"Shape: {df.shape}")

# separate features and target
X = df.drop('G3', axis=1)
y = df['G3']

print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")

# need to encode target for xgboost (it needs 0,1,2,3... not actual grades)
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# find categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

# encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# scale numerical columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("Preprocessing done!")

# split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
_, _, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# define all 6 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

model_files = {
    'Logistic Regression': 'logistic_model.pkl',
    'Decision Tree': 'dt_model.pkl',
    'KNN': 'knn_model.pkl',
    'Naive Bayes': 'nb_model.pkl',
    'Random Forest': 'rf_model.pkl',
    'XGBoost': 'xgb_model.pkl'
}

# create models folder
os.makedirs('models', exist_ok=True)

# train and evaluate each model
results = []
classes = np.unique(y_train)

print("\n" + "="*60)
print("Training Models...")
print("="*60)

for name, model in models.items():
    print(f"\n{name}:")
    
    # train
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    y_pred_orig = target_le.inverse_transform(y_pred)
    
    # calculate metrics
    acc = accuracy_score(y_test_orig, y_pred_orig)
    
    # auc score (ovo for multiclass)
    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro', labels=classes)
    except:
        auc = np.nan
    
    prec = precision_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
    rec = recall_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
    f1 = f1_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test_orig, y_pred_orig)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc
    })
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}" if not np.isnan(auc) else "  AUC: N/A")
    print(f"  F1: {f1:.4f}")
    
    # save model
    with open(f'models/{model_files[name]}', 'wb') as f:
        pickle.dump(model, f)

# save preprocessors for streamlit app
preprocessors = {
    'target_encoder': target_le,
    'label_encoders': label_encoders,
    'scaler': scaler
}
with open('models/preprocessors.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

print("\nModels saved to models/ folder")

# create comparison table
print("\n" + "="*60)
print("Model Comparison Table")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')
print(results_df.round(4))

# save to csv
results_df.to_csv('models/model_comparison.csv')
print("\nResults saved to models/model_comparison.csv")

# find best models
print("\nBest Models:")
print(f"  Accuracy: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})")
print(f"  AUC: {results_df['AUC'].idxmax()} ({results_df['AUC'].max():.4f})")
print(f"  F1: {results_df['F1'].idxmax()} ({results_df['F1'].max():.4f})")

print("\nDone!")
