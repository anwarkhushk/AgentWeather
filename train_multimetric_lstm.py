import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from feature_store_data import load_multimetric_training_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

Path("multimetric_models").mkdir(exist_ok=True)
Path("multimetric_results").mkdir(exist_ok=True)

# 1. Load data from Hopsworks Feature Store
train, test = load_multimetric_training_data()

# 2. Create time features
for df in (train, test):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

targets = [
    "target_temperature_24h",
    "target_humidity_24h",
    "target_precipitation_24h",
    "target_uv_24h",
    "target_wind_speed_24h",
    "target_wind_direction_sin_24h",
    "target_wind_direction_cos_24h"
]

feature_columns = [
    c for c in train.select_dtypes(include=[np.number]).columns
    if c not in targets + ["target_rain_probability_24h"]
]

X = train[feature_columns].replace([np.inf, -np.inf], np.nan)
X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan)

# Chronological split with 24-row purge
split = int(len(train) * 0.8)
purge = 24
fit_end = split - purge

X_fit_raw = X.iloc[:fit_end].copy()
X_val_raw = X.iloc[split:].copy()

y_fit_raw = train[targets].iloc[:fit_end].copy()
y_val_raw = train[targets].iloc[split:].copy()

print("Training rows:", len(X_fit_raw))
print("Purged rows:", purge)
print("Validation rows:", len(X_val_raw))
print("Test rows:", len(X_test))

# 3. Scaling - fit only on training portion
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_fit = x_scaler.fit_transform(
    X_fit_raw.fillna(X_fit_raw.median()).fillna(0)
)
medians = X_fit_raw.median()

X_val = x_scaler.transform(
    X_val_raw.fillna(medians).fillna(0)
)

y_fit = y_scaler.fit_transform(y_fit_raw)
y_val = y_scaler.transform(y_val_raw)

# 4. Create independent sequences for each period
SEQUENCE_LENGTH = 24

def create_sequences(X_data, y_data, sequence_length):
    X_seq, y_seq = [], []

    for i in range(sequence_length - 1, len(X_data)):
        X_seq.append(X_data[i-sequence_length+1:i+1])
        y_seq.append(y_data[i])

    return np.asarray(X_seq), np.asarray(y_seq)

X_fit_seq, y_fit_seq = create_sequences(
    X_fit, y_fit, SEQUENCE_LENGTH
)

X_val_seq, y_val_seq = create_sequences(
    X_val, y_val, SEQUENCE_LENGTH
)

print("\nTraining sequences:", X_fit_seq.shape)
print("Validation sequences:", X_val_seq.shape)

assert len(X_val_seq) == len(y_val_seq)

# 5. Build multi-output LSTM
def build_model():
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, len(feature_columns))),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(16, activation="relu"),
        Dense(len(targets))
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model

# 6. Validation training
model = build_model()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n========== LSTM VALIDATION TRAINING ==========")

history = model.fit(
    X_fit_seq,
    y_fit_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=15,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print("Best epoch:", best_epoch)

val_pred_scaled = model.predict(X_val_seq, verbose=0)
val_pred = y_scaler.inverse_transform(val_pred_scaled)

val_actual = y_val_raw.iloc[SEQUENCE_LENGTH-1:].to_numpy()

# 7. Validation metrics
validation_results = []

print("\n========== VALIDATION RESULTS ==========")

for i, target in enumerate(targets):
    mae = mean_absolute_error(val_actual[:, i], val_pred[:, i])
    rmse = np.sqrt(mean_squared_error(val_actual[:, i], val_pred[:, i]))
    r2 = r2_score(val_actual[:, i], val_pred[:, i])

    print(f"{target}: MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    validation_results.append({
        "target": target,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "validation_r2": r2
    })

# 8. Refit scalers and final model on all training data
print("\n========== FINAL MODEL TRAINING ==========")

X_all_raw = X.copy()
X_all = X_all_raw.fillna(medians).fillna(0)

final_x_scaler = StandardScaler()
final_y_scaler = StandardScaler()

X_all_scaled = final_x_scaler.fit_transform(X_all)
y_all_scaled = final_y_scaler.fit_transform(train[targets])

X_test_scaled = final_x_scaler.transform(
    X_test.fillna(medians).fillna(0)
)

X_all_seq, y_all_seq = create_sequences(
    X_all_scaled, y_all_scaled, SEQUENCE_LENGTH
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled,
    test[targets].to_numpy(),
    SEQUENCE_LENGTH
)

final_model = build_model()

final_model.fit(
    X_all_seq,
    y_all_scaled[SEQUENCE_LENGTH-1:],
    epochs=best_epoch,
    batch_size=64,
    verbose=1
)

# 9. Test predictions
test_pred_scaled = final_model.predict(X_test_seq, verbose=0)
test_pred = final_y_scaler.inverse_transform(test_pred_scaled)

test_actual = test[targets].iloc[SEQUENCE_LENGTH-1:].to_numpy()

print("\n========== FINAL TEST RESULTS ==========")

test_results = []

for i, target in enumerate(targets):
    mae = mean_absolute_error(test_actual[:, i], test_pred[:, i])
    rmse = np.sqrt(mean_squared_error(test_actual[:, i], test_pred[:, i]))
    r2 = r2_score(test_actual[:, i], test_pred[:, i])

    print(f"{target}: MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    test_results.append({
        "target": target,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2
    })

# 10. Wind direction evaluation in degrees
sin_idx = targets.index("target_wind_direction_sin_24h")
cos_idx = targets.index("target_wind_direction_cos_24h")

pred_angle = np.degrees(
    np.arctan2(test_pred[:, sin_idx], test_pred[:, cos_idx])
) % 360

actual_angle = np.degrees(
    np.arctan2(test_actual[:, sin_idx], test_actual[:, cos_idx])
) % 360

circular_error = np.abs(
    (actual_angle - pred_angle + 180) % 360 - 180
)

print(
    "\nLSTM wind direction circular MAE:",
    round(circular_error.mean(), 4),
    "degrees"
)

# 11. Save artifacts and results
final_model.save("multimetric_models/weather_multimetric_lstm.keras")

joblib.dump(
    {
        "x_scaler": final_x_scaler,
        "y_scaler": final_y_scaler,
        "feature_medians": medians,
        "features": feature_columns,
        "targets": targets,
        "sequence_length": SEQUENCE_LENGTH
    },
    "multimetric_models/weather_lstm_preprocessing.pkl"
)

pd.DataFrame(validation_results).to_csv(
    "multimetric_results/lstm_validation_metrics.csv",
    index=False
)

pd.DataFrame(test_results).to_csv(
    "multimetric_results/lstm_test_metrics.csv",
    index=False
)

predictions_df = pd.DataFrame(
    test_pred,
    columns=[t.replace("target_", "predicted_") for t in targets]
)

actual_df = pd.DataFrame(
    test_actual,
    columns=[t.replace("target_", "actual_") for t in targets]
)

pd.concat([actual_df, predictions_df], axis=1).to_csv(
    "multimetric_results/lstm_test_predictions.csv",
    index=False
)

with open("multimetric_results/lstm_feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

print("\n========================================")
print("MULTI-METRIC LSTM TRAINING COMPLETED")
print("Model: multimetric_models/weather_multimetric_lstm.keras")
print("Results: multimetric_results/")
print("========================================")
