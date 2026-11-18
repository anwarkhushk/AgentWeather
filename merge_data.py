
import pandas as pd

# 1. Load raw datasets
aqi_df = pd.read_csv("aqi_data.csv")
weather_df = pd.read_csv("weather_data.csv")

# 2. Convert timestamps into datetime format
aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

# 3. Set timestamps as indexes
aqi_df = aqi_df.set_index("timestamp")
weather_df = weather_df.set_index("timestamp")

# 4. Align weather data with hourly AQI timestamps
weather_aligned = weather_df.reindex(
    weather_df.index.union(aqi_df.index)
).sort_index()

# 5. Interpolate numerical weather values
numeric_columns = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed"
]

weather_aligned[numeric_columns] = (
    weather_aligned[numeric_columns]
    .interpolate(method="time")
)

# 6. Fill missing categorical weather descriptions
weather_aligned["weather"] = (
    weather_aligned["weather"].ffill()
)

# 7. Keep only timestamps available in AQI dataset
weather_aligned = weather_aligned.reindex(aqi_df.index)

# 8. Merge AQI and aligned weather data
merged_df = aqi_df.join(weather_aligned)

# 9. Reset timestamp index
merged_df = merged_df.reset_index()

# 10. Display results
print("Total merged records:", len(merged_df))

print("\nMerged Dataset:")
print(merged_df.head(10))

# 11. Check missing values
print("\nMissing Values:")
print(merged_df.isnull().sum())

# 12. Save merged dataset
merged_df.to_csv("merged_data.csv", index=False)

print("\nMerged data saved successfully!")
