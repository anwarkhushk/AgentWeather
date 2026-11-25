
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from feature_store_data import load_multimetric_training_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, average_precision_score
)

Path("multimetric_models").mkdir(exist_ok=True)
Path("multimetric_results").mkdir(exist_ok=True)

train, test = load_multimetric_training_data()

for df in (train, test):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

target_columns = [
    "target_temperature_24h",
    "target_humidity_24h",
    "target_precipitation_24h",
    "target_uv_24h",
    "target_wind_speed_24h",
    "target_wind_direction_sin_24h",
    "target_wind_direction_cos_24h",
    "target_rain_probability_24h"
]

feature_columns = [
    c for c in train.select_dtypes(include=[np.number]).columns
    if c not in target_columns
]

X = train[feature_columns].replace([np.inf, -np.inf], np.nan)
X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median()).fillna(0)
X_test = X_test.fillna(X.median()).fillna(0)

# Chronological validation split with 24-hour purge
split = int(len(train) * 0.8)
purge = 24
fit_end = split - purge

X_fit = X.iloc[:fit_end]
X_val = X.iloc[split:]

print(f"Training rows: {len(X_fit)}")
print(f"Purged rows: {purge}")
print(f"Validation rows: {len(X_val)}")

results = []
regression_targets = target_columns[:-1]

for target in regression_targets:
    print(f"\n========== {target} ==========")

    y = train[target]
    y_fit = y.iloc[:fit_end]
    y_val = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train and validate without overlapping target windows
    model.fit(X_fit, y_fit)

    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    # Refit on complete training set after validation
    model.fit(X, y)

    test_pred = model.predict(X_test)
    test_y = test[target]

    mae = mean_absolute_error(test_y, test_pred)
    rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    r2 = r2_score(test_y, test_pred)

    print(
        f"Validation MAE: {val_mae:.4f} | "
        f"Validation RMSE: {val_rmse:.4f}"
    )
    print(f"Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    filename = target.replace("target_", "").replace("_24h", "")
    joblib.dump(model, f"multimetric_models/{filename}_rf.pkl")

    results.append({
        "target": target,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2
    })

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        f"multimetric_results/{filename}_importance.csv",
        index=False
    )

# Rain occurrence classifier
print("\n========== RAIN OCCURRENCE CLASSIFIER ==========")

rain_target = "target_rain_probability_24h"
y = train[rain_target].astype(int)
y_test = test[rain_target].astype(int)

classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Apply the same 24-row purge to classifier validation
y_rain_fit = y.iloc[:fit_end]
y_rain_val = y.iloc[split:]

classifier.fit(X_fit, y_rain_fit)
val_prob = classifier.predict_proba(X_val)[:, 1]

# Select threshold using validation F1
thresholds = np.arange(0.05, 0.96, 0.05)
best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    score = f1_score(y_rain_val, pred, zero_division=0)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

# Refit classifier on all training data
classifier.fit(X, y)

test_prob = classifier.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
pr_auc = average_precision_score(y_test, test_prob)

print(f"Selected validation threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")

joblib.dump(classifier, "multimetric_models/rain_classifier_rf.pkl")

results.append({
    "target": rain_target,
    "validation_mae": None,
    "validation_rmse": None,
    "test_mae": None,
    "test_rmse": None,
    "test_r2": None,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1": f1,
    "test_pr_auc": pr_auc,
    "decision_threshold": best_threshold
})

pd.DataFrame(results).to_csv(
    "multimetric_results/rf_metrics_comparison.csv",
    index=False
)

with open("multimetric_results/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

print("\n========================================")
print("MULTI-METRIC RANDOM FOREST TRAINING DONE")
print("Models: multimetric_models/")
print("Results: multimetric_results/")
print("========================================")
