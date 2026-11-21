import pandas as pd
import numpy as np

# 1. Load training dataset
df = pd.read_csv("historical_training_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 3. Lag features
for lag in [1, 3, 6, 12, 24, 48, 168]:
    df[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)

# 4. Rolling average features
for window in [3, 6, 24, 168]:
    df[f"aqi_rolling_mean_{window}h"] = (
        df["us_aqi"].rolling(window=window).mean()
    )

# 5. Wind direction cyclical encoding
df["wind_direction_sin"] = np.sin(
    np.radians(df["wind_direction_10m"])
)

df["wind_direction_cos"] = np.cos(
    np.radians(df["wind_direction_10m"])
)

# Remove original wind direction
df = df.drop(columns=["wind_direction_10m"])

# 6. Remove rows with missing lag/rolling values
df = df.dropna().reset_index(drop=True)

# 7. Validation
print("\n========== FEATURE ENGINEERING ==========")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate timestamps:")
print(df["timestamp"].duplicated().sum())

print("\nNew features:")
print(df.columns.tolist())

print("\nSample records:")
print(df.head())

# 8. Save engineered dataset
df.to_csv("historical_featured_data.csv", index=False)

print("\nFeature engineering completed!")
print("Saved: historical_featured_data.csv")