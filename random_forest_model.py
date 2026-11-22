
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load datasets
train = pd.read_csv("training_data.csv")
test = pd.read_csv("testing_data.csv")

print("Datasets loaded successfully!")

# 2. Separate features (X) and target (y)
X_train = train.drop(columns=["timestamp", "target_aqi_24h"])
y_train = train["target_aqi_24h"]

X_test = test.drop(columns=["timestamp", "target_aqi_24h"])
y_test = test["target_aqi_24h"]

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

# 3. Initialize Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 4. Train the model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Model training completed!")

# 5. Generate predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# 7. Official naive baseline comparison
baseline_mae = 13.6621
baseline_rmse = 19.8503

mae_improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse
) * 100

print("\n========== OFFICIAL BASELINE COMPARISON ==========")
print(f"Baseline MAE:       {baseline_mae:.4f}")
print(f"Random Forest MAE:  {mae:.4f}")

print(f"\nBaseline RMSE:      {baseline_rmse:.4f}")
print(f"Random Forest RMSE: {rmse:.4f}")

print("\n========== IMPROVEMENT ==========")
print(f"MAE Improvement:  {mae_improvement:.2f}%")
print(f"RMSE Improvement: {rmse_improvement:.2f}%")

# 8. Feature Importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(importance.head(10).to_string(index=False))

importance.to_csv("feature_importance.csv", index=False)

print("\nFeature importance saved successfully!")

# 9. Save predictions
results = pd.DataFrame({
    "timestamp": test["timestamp"],
    "actual_aqi": y_test,
    "predicted_aqi": y_pred
})

results.to_csv("random_forest_predictions.csv", index=False)

print("\nPredictions saved successfully!")
print("\nRandom Forest evaluation completed!")
