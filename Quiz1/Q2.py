#Q2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the csv to access data
df = pd.read_csv("water-treatment.csv", header=None)

# Drop date column if present
if pd.to_numeric(df.iloc[:,0], errors="coerce").isna().mean() > 0.8:
    df = df.drop(columns=[0]).reset_index(drop=True)

df.columns = [f"c{i}" for i in range(df.shape[1])]
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

# PH columns (documented indices)
PH_E = "c2"
PH_S = "c22"

# Correlation
corr = df[PH_E].corr(df[PH_S])
print("PH-E vs PH-S correlation:", corr)

# Create SAFE-PH-S
df["SAFE-PH-S"] = df[PH_S].apply(
    lambda x: "yes" if 6.5 <= x <= 8.5 else "no"
)

# Prepare data
features = [f"c{i}" for i in range(9)]
data = df[features + ["SAFE-PH-S"]].dropna()

X = data[features]
y = data["SAFE-PH-S"].map({"yes":1, "no":0})

# Split 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

min_class_size = y.value_counts().min()
cv_folds = min(10, min_class_size)

if cv_folds >= 2:
    scores = cross_val_score(lr, X, y, cv=cv_folds)
    print(f"{cv_folds}-fold CV Accuracy:", scores.mean())
else:
    print("Not enough samples for cross-validation.")
print("10-fold CV Accuracy:", scores.mean())

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))