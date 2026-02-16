# Q3.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

path = r"C:\Users\green\Desktop\Documents\TrustedAi\CSCE581-Spring2026-LukeWilliams\Quiz1\WaterAtlas-ManySites.csv"


print("Running from:", os.getcwd())
print("Reading:", path)

# Force comma separator for this dataset and skip malformed lines (so script continues)
df = pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip", encoding="utf-8")

# Collect text-like columns to search for "ph"
text_cols = []
for t in ("string", "object"):
    try:
        text_cols = df.select_dtypes(include=[t]).columns.tolist()
        if text_cols:
            break
    except Exception:
        continue

# Try some common parameter/characteristic column names first
param_col = None
param_mask = None
for guess in ["Characteristic", "CharacteristicName", "Parameter", "ParameterName", "CharacteristicName"]:
    if guess in df.columns:
        s = df[guess].astype(str)
        if s.str.contains(r"\bph\b", case=False, na=False).any():
            param_col = guess
            param_mask = s.str.contains(r"\bph\b", case=False, na=False)
            break

# If not found, search any text columns for 'ph'
if param_mask is None:
    if text_cols:
        text_df = df[text_cols].fillna("")
        param_mask = text_df.apply(lambda col: col.astype(str).str.contains("ph", case=False, na=False)).any(axis=1)
    else:
        param_mask = pd.Series(False, index=df.index)

# Detect the numeric column likely to be the measured value (pH)
value_col = None
for c in df.columns:
    if any(k in c.lower() for k in ["value", "result", "measurement", "value_numeric", "resultvalue", "value"]):
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            value_col = c
            break

# fallback: pick column with most numeric parses
if value_col is None:
    best_col = None
    best_count = -1
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        nnum = int(coerced.notna().sum())
        if nnum > best_count:
            best_count = nnum
            best_col = c
    value_col = best_col

if value_col is None:
    raise RuntimeError("Could not auto-detect a numeric value column. Inspect CSV columns manually.")

print("Detected parameter column (text):", param_col)
print("Detected value column (numeric):", value_col)

ph_df = df[param_mask].copy()
ph_df[value_col] = pd.to_numeric(ph_df[value_col], errors="coerce")

# SAFE-PH: yes if 6.5 <= pH <= 8.5; else no
ph_df["SAFE-PH"] = ph_df[value_col].apply(lambda x: "yes" if pd.notna(x) and 6.5 <= x <= 8.5 else "no")

# Prefer numeric columns (excluding the pH value itself)
numeric_features = [c for c in ph_df.select_dtypes(include=[np.number]).columns.tolist() if c != value_col]

# If no numeric features, try lat/lon candidates
if len(numeric_features) == 0:
    for latname in ["Latitude_DD", "Latitude", "Lat", "latitude", "lat"]:
        for lonname in ["Longitude_DD", "Longitude", "Lon", "Lng", "longitude", "lon"]:
            if latname in ph_df.columns and lonname in ph_df.columns:
                numeric_features = [latname, lonname]
                break
        if numeric_features:
            break

# If still no numeric features, one-hot a few top categorical columns
if len(numeric_features) == 0:
    obj_cols = ph_df.select_dtypes(include=["object", "string"]).columns.difference(["SAFE-PH"]).tolist()
    obj_cols_sorted = sorted(obj_cols, key=lambda c: ph_df[c].notna().sum(), reverse=True)[:3]
    if len(obj_cols_sorted) == 0:
        raise ValueError("No usable features found in the pH subset to train classifiers.")
    ph_df = pd.get_dummies(ph_df, columns=obj_cols_sorted, dummy_na=False, drop_first=True)
    numeric_features = [c for c in ph_df.select_dtypes(include=[np.number]).columns.tolist() if c != value_col]

if value_col in numeric_features:
    numeric_features.remove(value_col)

features = numeric_features[:9]
if len(features) == 0:
    raise ValueError("No numeric features left after removing pH value column. Inspect dataset/feature selection.")

print("Using features:", features)

#build and train the models of the data
data = ph_df[features + ["SAFE-PH"]].dropna().copy()
X = data[features]
y = data["SAFE-PH"].map({"yes": 1, "no": 0})

if y.nunique() < 2:
    raise ValueError("Not enough class variety in SAFE-PH after filtering. Check your pH selection and data.")

# train/test split 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(solver="liblinear", max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))

# cross-val (cap folds by min class size and 10)
min_class_size = int(y.value_counts().min())
cv_folds = int(min(10, min_class_size)) if min_class_size is not None else 2
if cv_folds >= 2:
    X_scaled_full = scaler.transform(X)
    scores_lr = cross_val_score(lr, X_scaled_full, y, cv=cv_folds)
    print(f"{cv_folds}-fold CV Accuracy (LR):", scores_lr.mean())
else:
    print("Not enough samples for cross-validation (LR).")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))

if cv_folds >= 2:
    scores_rf = cross_val_score(rf, X_scaled_full, y, cv=cv_folds)
    print(f"{cv_folds}-fold CV Accuracy (RF):", scores_rf.mean())
else:
    print("Not enough samples for cross-validation (RF).")

# Determine lat/lon column names (common candidates)
lat_col = None
lon_col = None
for c in ["Latitude_DD", "Latitude", "Lat", "latitude", "lat"]:
    if c in ph_df.columns:
        lat_col = c
        break
for c in ["Longitude_DD", "Longitude", "Lon", "Lng", "longitude", "lon"]:
    if c in ph_df.columns:
        lon_col = c
        break

# station identifier (if exists)
station_col = None
for name in ["StationID", "Station_Name", "StationName", "ActualStationID", "MonitoringLocationName", "Name"]:
    if name in ph_df.columns:
        station_col = name
        break

# if lat/lon missing, try to detect numeric columns that look like coords. If there are at least 2 numeric columns then pick the 2 with medians in a plausible range
if lat_col is None or lon_col is None:
    num_cols = ph_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        medians = {c: ph_df[c].median() for c in num_cols}
        lat_candidates = [c for c,v in medians.items() if -90 <= v <= 90]
        lon_candidates = [c for c,v in medians.items() if -180 <= v <= 180]
        if lat_candidates and lon_candidates:
            lat_col = lat_candidates[0]
            lon_col = lon_candidates[0]

if lat_col is None or lon_col is None:
    raise RuntimeError("Could not find latitude/longitude columns. Inspect CSV column names and update the script accordingly.")

ph_df[lat_col] = pd.to_numeric(ph_df[lat_col], errors="coerce")
ph_df[lon_col] = pd.to_numeric(ph_df[lon_col], errors="coerce")
ph_df = ph_df.dropna(subset=[value_col, lat_col, lon_col]).copy()


if station_col is None:
    ph_df["station_lat_round"] = ph_df[lat_col].round(4)
    ph_df["station_lon_round"] = ph_df[lon_col].round(4)
    ph_df["StationID_auto"] = ph_df["station_lat_round"].astype(str) + "_" + ph_df["station_lon_round"].astype(str)
    station_col = "StationID_auto"

ph_df["UNSAFE"] = ph_df["SAFE-PH"].map({"yes": 0, "no": 1})

grp = ph_df.groupby(station_col).agg(
    total_measurements = (value_col, "count"),
    unsafe_count = ("UNSAFE", "sum"),
    mean_ph = (value_col, "mean"),
    lat = (lat_col, "median"),
    lon = (lon_col, "median")
).reset_index()
grp["unsafe_rate"] = grp["unsafe_count"] / grp["total_measurements"]

print("\nTop 10 places with most UNSAFE pH (by absolute count):")
print(grp.sort_values("unsafe_count", ascending=False).head(10)[[station_col, "total_measurements", "unsafe_count", "unsafe_rate", "mean_ph", "lat", "lon"]].to_string(index=False))

print("\nTop 10 places with least UNSAFE pH (by absolute count):")
print(grp.sort_values("unsafe_count", ascending=True).head(10)[[station_col, "total_measurements", "unsafe_count", "unsafe_rate", "mean_ph", "lat", "lon"]].to_string(index=False))

# export data for Google My Maps / Google Earth
unsafe_points = ph_df[ph_df["UNSAFE"]==1].copy()
unsafe_points.to_csv("unsafe_points.csv", index=False, columns=[station_col, value_col, lat_col, lon_col, "SAFE-PH"])
