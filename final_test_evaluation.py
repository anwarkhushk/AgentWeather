
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv("training_data.csv")
test = pd.read_csv("testing_data.csv")

target = "target_aqi_24h"

y_train = train[target].to_numpy()
y_test = test[target].to_numpy()

print("Training records:", len(train))
print("Testing records:", len(test))

all_predictions = {}

# --------------------------------------------------
# 2. Naive Baseline
# --------------------------------------------------

print("\nEvaluating Naive Baseline...")

all_predictions["Naive Baseline"] = test["aqi_lag_24"].to_numpy()

# --------------------------------------------------
# 3. Random Forest
# --------------------------------------------------

print("\nTraining Random Forest on full training data...")

X_train_rf = train.drop(columns=["timestamp", target])
X_test_rf = test.drop(columns=["timestamp", target])

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_rf, y_train)

# Save trained Random Forest model
joblib.dump(rf, "random_forest_aqi_final.pkl")

all_predictions["Random Forest"] = rf.predict(X_test_rf)

print("Random Forest model saved!")

# --------------------------------------------------
# 4. SARIMAX
# --------------------------------------------------

print("\nTraining SARIMAX on full training data...")

sarimax_features = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "aqi_lag_24",
    "aqi_rolling_mean_24h"
]

X_train_sarimax = train[sarimax_features]
X_test_sarimax = test[sarimax_features]

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train_sarimax)
X_test_scaled = X_scaler.transform(X_test_sarimax)

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

sarimax = SARIMAX(
    endog=y_train_scaled,
    exog=X_train_scaled,
    order=(1, 0, 0),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
)

sarimax_fitted = sarimax.fit(
    method="lbfgs",
    maxiter=200,
    disp=False
)

print(
    "SARIMAX convergence:",
    sarimax_fitted.mle_retvals.get("converged", False)
)

sarimax_forecast_scaled = sarimax_fitted.get_forecast(
    steps=len(test),
    exog=X_test_scaled
).predicted_mean

sarimax_predictions = y_scaler.inverse_transform(
    np.asarray(sarimax_forecast_scaled).reshape(-1, 1)
).ravel()

all_predictions["SARIMAX"] = sarimax_predictions

# Save SARIMAX model and scalers
sarimax_fitted.save("sarimax_aqi_final.pkl")
joblib.dump(X_scaler, "sarimax_feature_scaler.pkl")
joblib.dump(y_scaler, "sarimax_target_scaler.pkl")

print("SARIMAX model and scalers saved!")

# --------------------------------------------------
# 5. LSTM
# --------------------------------------------------

print("\nTraining LSTM on full training data...")

lstm_features = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "aqi_lag_24",
    "aqi_rolling_mean_24h"
]

X_train_lstm = train[lstm_features].to_numpy()
X_test_lstm = test[lstm_features].to_numpy()

X_lstm_scaler = StandardScaler()
y_lstm_scaler = StandardScaler()

X_train_lstm_scaled = X_lstm_scaler.fit_transform(X_train_lstm)
X_test_lstm_scaled = X_lstm_scaler.transform(X_test_lstm)

y_train_lstm_scaled = y_lstm_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

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


X_train_seq, y_train_seq = create_sequences(
    X_train_lstm_scaled,
    y_train_lstm_scaled,
    SEQUENCE_LENGTH
)

# 23 previous training rows provide context for first test prediction
X_test_with_context = np.vstack([
    X_train_lstm_scaled[-(SEQUENCE_LENGTH - 1):],
    X_test_lstm_scaled
])

X_test_seq, _ = create_sequences(
    X_test_with_context,
    np.zeros(len(X_test_with_context)),
    SEQUENCE_LENGTH
)

print("LSTM training sequences:", X_train_seq.shape)
print("LSTM test sequences:", X_test_seq.shape)

if len(X_test_seq) != len(test):
    raise ValueError("LSTM test sequence count does not match test records!")

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(lstm_features))),
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

model.fit(
    X_train_seq,
    y_train_seq,
    epochs=1,
    batch_size=64,
    verbose=1
)

lstm_predictions_scaled = model.predict(
    X_test_seq,
    verbose=0
)

lstm_predictions = y_lstm_scaler.inverse_transform(
    lstm_predictions_scaled
).ravel()

all_predictions["LSTM"] = lstm_predictions

# Save LSTM model and scalers
model.save("lstm_aqi_final.keras")

joblib.dump(X_lstm_scaler, "lstm_feature_scaler.pkl")
joblib.dump(y_lstm_scaler, "lstm_target_scaler.pkl")

print("LSTM model and scalers saved!")

# --------------------------------------------------
# 6. Evaluate all models on untouched test data
# --------------------------------------------------

results = []

for model_name, predictions in all_predictions.items():

    if len(predictions) != len(y_test):
        raise ValueError(
            f"{model_name}: prediction count does not match test data"
        )

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

    filename = model_name.lower().replace(" ", "_")

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    }).to_csv(
        f"{filename}_test_predictions.csv",
        index=False
    )

comparison = pd.DataFrame(results).sort_values("MAE")

print("\n========== FINAL TEST COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv("final_test_comparison.csv", index=False)

print("\n========== SAVED MODEL ARTIFACTS ==========")
print("random_forest_aqi_final.pkl")
print("sarimax_aqi_final.pkl")
print("sarimax_feature_scaler.pkl")
print("sarimax_target_scaler.pkl")
print("lstm_aqi_final.keras")
print("lstm_feature_scaler.pkl")
print("lstm_target_scaler.pkl")

print("\nFinal evaluation completed!")
