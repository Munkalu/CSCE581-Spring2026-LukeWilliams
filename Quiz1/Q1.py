import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df = pd.read_csv(url)

print("Original shape:", df.shape)
df = df.dropna(subset=["days_b_screening_arrest"])

df = df[
    (df.days_b_screening_arrest <= 30) &
    (df.days_b_screening_arrest >= -30) &
    (df.is_recid != -1) &
    (df.c_charge_degree != 'O') &
    (df.score_text != 'N/A')
]

df.reset_index(drop=True, inplace=True)
print("Filtered shape:", df.shape)
features = ['age', 'priors_count', 'juv_fel_count',
            'juv_misd_count', 'juv_other_count']

X = df[features]
y = df['two_year_recid']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:,1]

print("Logistic Regression")
cm = confusion_matrix(y_test, lr_pred)
TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix:")
print(cm)
print("Accuracy :", (TN + TP) / (TN + TP + FN + FP))
print("PPV :", TP / (TP + FP))
print("FPR:", FP / (FP + TN))
print("FNR:", FN / (FN + TP))
print()

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print("\nRandom Forest")
cm = confusion_matrix(y_test, rf_pred)
TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix:")
print(cm)
print("Accuracy :", (TN + TP) / (TN + TP + FN + FP))
print("PPV :", TP / (TP + FP))
print("FPR:", FP / (FP + TN))
print("FNR:", FN / (FN + TP))
print()