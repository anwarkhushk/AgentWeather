import os
import warnings
import joblib
import numpy as np
import pandas as pd

from feature_store_data import load_multimetric_training_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

os.makedirs("multimetric_models", exist_ok=True)
os.makedirs("multimetric_results", exist_ok=True)

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

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
    "uv_index",
    "wind_direction_sin",
    "wind_direction_cos",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]

missing = [c for c in features if c not in train.columns or c not in test.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

targets = {
    "Temperature": "target_temperature_24h",
    "Humidity": "target_humidity_24h",
    "Precipitation": "target_precipitation_24h",
    "UV": "target_uv_24h",
    "Wind speed": "target_wind_speed_24h"
}

X_train = train[features].replace([np.inf, -np.inf], np.nan)
X_test = test[features].replace([np.inf, -np.inf], np.nan)

medians = X_train.median()
X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

# Same chronological validation strategy: 80/20 with 24-row purge.
fit_end = int(len(train) * 0.8)
fit_end = fit_end - 24

X_fit = X_train.iloc[:fit_end]
X_val = X_train.iloc[fit_end + 24:]
X_full = X_train

results = []

for name, target in targets.items():
    print(f"\n========== {name} ==========", flush=True)

    y_train = train[target].astype(float)
    y_test = test[target].astype(float)

    y_fit = y_train.iloc[:fit_end]
    y_val = y_train.iloc[fit_end + 24:]

    # Validation model: fit only on the training portion.
    x_scaler_val = StandardScaler()
    y_scaler_val = StandardScaler()

    X_fit_scaled = x_scaler_val.fit_transform(X_fit)
    X_val_scaled = x_scaler_val.transform(X_train[features].iloc[fit_end + 24:])

    y_fit_scaled = y_scaler_val.fit_transform(y_fit.to_numpy().reshape(-1, 1)).ravel()

    val_model = SARIMAX(
        y_fit_scaled,
        exog=X_fit_scaled,
        order=(0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    val_fit = val_model.fit(disp=False, maxiter=50, method="lbfgs")
    val_pred_scaled = val_fit.get_forecast(
        steps=len(y_val),
        exog=X_val_scaled
    ).predicted_mean

    val_pred = y_scaler_val.inverse_transform(
        np.asarray(val_pred_scaled).reshape(-1, 1)
    ).ravel()

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    # Final model: refit on the complete training set before test evaluation.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_full_scaled = x_scaler.fit_transform(X_full)
    X_test_scaled = x_scaler.transform(X_test)

    y_full_scaled = y_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    final_model = SARIMAX(
        y_full_scaled,
        exog=X_full_scaled,
        order=(0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    final_fit = final_model.fit(disp=False, maxiter=50, method="lbfgs")

    test_pred_scaled = final_fit.get_forecast(
        steps=len(y_test),
        exog=X_test_scaled
    ).predicted_mean

    test_pred = y_scaler.inverse_transform(
        np.asarray(test_pred_scaled).reshape(-1, 1)
    ).ravel()

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"Validation MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
    print(f"Test MAE:       {test_mae:.4f} | RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")

    results.append({
        "Metric": name,
        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2
    })

    artifact = {
        "model": final_fit,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "features": features,
        "target": target
    }

    filename = f"multimetric_models/{name.lower().replace(' ', '_')}_sarimax.pkl"
    joblib.dump(artifact, filename)
    print(f"Saved: {filename}", flush=True)

result_df = pd.DataFrame(results)
result_df.to_csv("multimetric_results/sarimax_weather_metrics.csv", index=False)

print("\n========== SARIMAX SUMMARY ==========")
print(result_df.round(4).to_string(index=False))
print("\nSaved: multimetric_results/sarimax_weather_metrics.csv")
