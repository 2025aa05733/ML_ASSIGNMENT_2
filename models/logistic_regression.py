# Logistic Regression Model
# Student Performance Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# load data
df = pd.read_csv('data/student_performance.csv')
X = df.drop('G3', axis=1)
y = df['G3']

# encode target
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# scale numerical columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
_, _, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
y_pred_orig = target_le.inverse_transform(y_pred)

# calculate metrics
acc = accuracy_score(y_test_orig, y_pred_orig)
y_proba = model.predict_proba(X_test)
classes = np.unique(y_train)
auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro', labels=classes)
prec = precision_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
rec = recall_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
f1 = f1_score(y_test_orig, y_pred_orig, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test_orig, y_pred_orig)

print("Logistic Regression Results:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  AUC:       {auc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  MCC:       {mcc:.4f}")
