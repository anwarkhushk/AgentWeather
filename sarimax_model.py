
import pandas as pd
import numpy as np
import warnings

from feature_store_data import load_aqi_training_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load training data from Hopsworks Feature Store
train, _ = load_aqi_training_data()

target = "target_aqi_24h"

# Keep the feature set relatively small
features = [
    "us_aqi",
    "pm2_5",
    "pm10",
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

# 4. Build a simpler SARIMAX model
model = SARIMAX(
    endog=y_train_scaled,
    exog=X_train_scaled,
    order=(1, 0, 0),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
)

# 5. Train
print("\nTraining SARIMAX...")

fitted_model = model.fit(
    method="lbfgs",
    maxiter=200,
    disp=False
)

converged = fitted_model.mle_retvals.get("converged", False)

print("\nTraining completed!")
print("Model convergence:", converged)

# 6. Forecast validation data
forecast_scaled = fitted_model.get_forecast(
    steps=len(y_val),
    exog=X_val_scaled
).predicted_mean

# Convert predictions back to original AQI scale
predictions = y_scaler.inverse_transform(
    np.asarray(forecast_scaled).reshape(-1, 1)
).ravel()

# 7. Evaluate
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\n========== SARIMAX RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

print("\n========== PREDICTION DIAGNOSTICS ==========")
print(f"Actual AQI mean:      {y_val.mean():.2f}")
print(f"Predicted AQI mean:   {predictions.mean():.2f}")
print(f"Actual AQI range:     {y_val.min():.2f} - {y_val.max():.2f}")
print(f"Predicted AQI range:  {predictions.min():.2f} - {predictions.max():.2f}")

# 8. Save predictions
results = pd.DataFrame({
    "actual": y_val.to_numpy(),
    "predicted": predictions
})

results.to_csv("sarimax_validation_predictions.csv", index=False)

print("\nValidation predictions saved!")
