import pandas as pd

# 1. Load merged historical dataset
df = pd.read_csv("historical_merged_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# 2. Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Verify hourly continuity
expected_timestamps = pd.date_range(
    start=df["timestamp"].iloc[0],
    periods=len(df),
    freq="h"
)

if not df["timestamp"].equals(pd.Series(expected_timestamps)):
    raise ValueError("Timestamps are not continuous hourly!")

print("Hourly continuity verified!")

# 4. Create 24-hour-ahead AQI target
df["target_aqi_24h"] = df["us_aqi"].shift(-24)

# 5. Remove last 24 rows (they have no future target)
df = df.dropna(subset=["target_aqi_24h"]).reset_index(drop=True)

# 6. Validation
print("\n========== TARGET VALIDATION ==========")
print("Total training records:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing target values:")
print(df["target_aqi_24h"].isnull().sum())

print("\nTarget statistics:")
print(df["target_aqi_24h"].describe())

print("\nSample records:")
print(df[["timestamp", "us_aqi", "target_aqi_24h"]].head(10))

# 7. Save training dataset
df.to_csv("historical_training_data.csv", index=False)

print("\nTraining dataset saved successfully!")