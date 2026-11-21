import pandas as pd

# 1. Load historical datasets
aqi_df = pd.read_csv("historical_aqi.csv")
weather_df = pd.read_csv("historical_weather.csv")

print("AQI records:", len(aqi_df))
print("Weather records:", len(weather_df))

# 2. Convert timestamps to datetime
aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

# 3. Check duplicate timestamps
print("\nAQI duplicates:", aqi_df["timestamp"].duplicated().sum())
print("Weather duplicates:", weather_df["timestamp"].duplicated().sum())

# 4. Sort datasets chronologically
aqi_df = aqi_df.sort_values("timestamp")
weather_df = weather_df.sort_values("timestamp")

# 5. Merge datasets using timestamp
merged_df = pd.merge(
    aqi_df,
    weather_df,
    on="timestamp",
    how="inner",
    validate="one_to_one"
)

# 6. Sort final dataset
merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

# 7. Validate merged dataset
print("\n========== VALIDATION ==========")

print("Total records:", len(merged_df))
print("Total columns:", len(merged_df.columns))

print("\nDate range:")
print("Start:", merged_df["timestamp"].min())
print("End:", merged_df["timestamp"].max())

print("\nMissing values:")
print(merged_df.isnull().sum())

print("\nDuplicate timestamps:",
    merged_df["timestamp"].duplicated().sum())

print("\nDataset columns:")
print(merged_df.columns.tolist())

# 8. Save historical dataset
merged_df.to_csv("historical_merged_data.csv", index=False)

print("\nHistorical dataset saved successfully!")
print("File: historical_merged_data.csv")

print("\nFirst 5 records:")
print(merged_df.head())