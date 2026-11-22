
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load data
train = pd.read_csv("training_data.csv")

target = "target_aqi_24h"

features = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "aqi_lag_24",
    "aqi_rolling_mean_24h"
]

X = train[features].copy()
y = train[target].copy()

# 2. Chronological validation split
split_index = int(len(train) * 0.8)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

print("Training records:", len(y_train))
print("Validation records:", len(y_val))

# 3. Scale features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
).ravel()

# 4. Create 24-hour sequences
SEQUENCE_LENGTH = 24


def create_sequences(X_data, y_data, sequence_length):
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length - 1, len(X_data)):
        X_sequences.append(
            X_data[i - sequence_length + 1:i + 1]
        )
        y_sequences.append(y_data[i])

    return np.array(X_sequences), np.array(y_sequences)


# Training sequences
X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train_scaled,
    SEQUENCE_LENGTH
)

# Use only the previous 23 training observations as validation context.
# Each validation sequence will contain:
# 23 previous observations + current validation observation.
X_val_with_context = np.vstack([
    X_train_scaled[-(SEQUENCE_LENGTH - 1):],
    X_val_scaled
])

y_val_with_context = np.concatenate([
    y_train_scaled[-(SEQUENCE_LENGTH - 1):],
    y_val_scaled
])

X_val_seq, y_val_seq = create_sequences(
    X_val_with_context,
    y_val_with_context,
    SEQUENCE_LENGTH
)

print("\nTraining sequences:", X_train_seq.shape)
print("Validation sequences:", X_val_seq.shape)

# Verify sequence alignment
assert len(X_train_seq) == len(y_train_seq)
assert len(X_val_seq) == len(y_val)
assert len(y_val_seq) == len(y_val)

print("\nSequence alignment verified!")

# 5. Build LSTM model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(features))),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# 6. Train model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining LSTM...")

history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=15,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

print("\nLSTM training completed!")

# 7. Predict validation data
predictions_scaled = model.predict(
    X_val_seq,
    verbose=0
).ravel()

predictions = y_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).ravel()

actual = y_val.to_numpy()

# 8. Final length verification
print("\nActual values:", len(actual))
print("Predictions:", len(predictions))

assert len(actual) == len(predictions), \
    "Actual and prediction lengths do not match!"

# 9. Evaluate model
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\n========== LSTM RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# 10. Save predictions and model
results = pd.DataFrame({
    "actual": actual,
    "predicted": predictions
})

results.to_csv("lstm_validation_predictions.csv", index=False)

model.save("lstm_aqi_model.keras")

print("\nPredictions saved!")
print("Model saved as lstm_aqi_model.keras")
