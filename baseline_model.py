
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

test = pd.read_csv("testing_data.csv")

# Actual future AQI
y_actual = test["target_aqi_24h"]

# Official naive baseline: AQI from 24 hours ago
y_pred = test["aqi_lag_24"]

# Evaluation
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\n========== OFFICIAL NAIVE BASELINE ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

results = pd.DataFrame({
    "timestamp": test["timestamp"],
    "actual_aqi": y_actual,
    "predicted_aqi": y_pred
})

results.to_csv("baseline_predictions.csv", index=False)

print("\nOfficial baseline saved!")
