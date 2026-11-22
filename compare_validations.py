
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load training data and reproduce the exact 80/20 chronological split
data = pd.read_csv("training_data.csv")

split_index = int(len(data) * 0.8)

train = data.iloc[:split_index].copy()
val = data.iloc[split_index:].copy()

y_actual = val["target_aqi_24h"].to_numpy()

def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

results = []

# 1. Naive baseline: AQI from 24 hours ago
results.append(
    evaluate("Naive Baseline", y_actual, val["aqi_lag_24"].to_numpy())
)

# 2. Random Forest
X_train = train.drop(columns=["timestamp", "target_aqi_24h"])
X_val = val.drop(columns=["timestamp", "target_aqi_24h"])

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest on validation split...")
rf.fit(X_train, train["target_aqi_24h"])

rf_predictions = rf.predict(X_val)

results.append(
    evaluate("Random Forest", y_actual, rf_predictions)
)

pd.DataFrame({
    "actual": y_actual,
    "predicted": rf_predictions
}).to_csv("rf_validation_predictions.csv", index=False)

# 3. Load existing LSTM and SARIMAX validation predictions
for model_name, filename in [
    ("LSTM", "lstm_validation_predictions.csv"),
    ("SARIMAX", "sarimax_validation_predictions.csv")
]:
    predictions_df = pd.read_csv(filename)

    if len(predictions_df) != len(y_actual):
        raise ValueError(
            f"{model_name}: Expected {len(y_actual)} predictions, "
            f"got {len(predictions_df)}"
        )

    if not np.allclose(
        predictions_df["actual"].to_numpy(),
        y_actual,
        atol=1e-5
    ):
        raise ValueError(
            f"{model_name}: Actual values do not match validation split!"
        )

    results.append(
        evaluate(
            model_name,
            y_actual,
            predictions_df["predicted"].to_numpy()
        )
    )

# 4. Display comparison
comparison = pd.DataFrame(results)
comparison = comparison.sort_values("MAE")

print("\n========== VALIDATION MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv("validation_model_comparison.csv", index=False)

print("\nComparison saved: validation_model_comparison.csv")
