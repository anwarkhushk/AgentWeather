
import pandas as pd
import numpy as np

# 1. Load datasets
aqi_df = pd.read_csv("aqi_data.csv")
weather_df = pd.read_csv("weather_data.csv")
uv_df = pd.read_csv("uv_data.csv")

# 2. Convert timestamps to datetime
aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
uv_df["timestamp"] = pd.to_datetime(uv_df["timestamp"])

# 3. Check duplicate timestamps
print("Duplicate AQI:", aqi_df["timestamp"].duplicated().sum())
print("Duplicate Weather:", weather_df["timestamp"].duplicated().sum())
print("Duplicate UV:", uv_df["timestamp"].duplicated().sum())

# 4. Set timestamps as indexes
aqi_df = aqi_df.set_index("timestamp").sort_index()
weather_df = weather_df.set_index("timestamp").sort_index()
uv_df = uv_df.set_index("timestamp").sort_index()

# 5. Find common time range
start_time = max(
    aqi_df.index.min(),
    weather_df.index.min(),
    uv_df.index.min()
)

end_time = min(
    aqi_df.index.max(),
    weather_df.index.max(),
    uv_df.index.max()
)

print("\nCommon time range:", start_time, "to", end_time)

# 6. Create hourly timestamps
hourly_index = pd.date_range(
    start=start_time,
    end=end_time,
    freq="h"
)

# 7. Align AQI and UV data
aqi_aligned = aqi_df.reindex(hourly_index)
uv_aligned = uv_df.reindex(hourly_index)

# 8. Align weather data
weather_aligned = weather_df.reindex(
    weather_df.index.union(hourly_index)
).sort_index()

# 9. Interpolate continuous numerical weather features
numeric_columns = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rain_probability"
]

weather_aligned[numeric_columns] = (
    weather_aligned[numeric_columns]
    .interpolate(
        method="time",
        limit=2,
        limit_area="inside"
    )
)

# 10. Circular interpolation for wind direction
radians = np.deg2rad(weather_aligned["wind_direction"])

sin_values = pd.Series(
    np.sin(radians),
    index=weather_aligned.index
)

cos_values = pd.Series(
    np.cos(radians),
    index=weather_aligned.index
)

sin_values = sin_values.interpolate(
    method="time",
    limit=2,
    limit_area="inside"
)

cos_values = cos_values.interpolate(
    method="time",
    limit=2,
    limit_area="inside"
)

weather_aligned["wind_direction"] = (
    np.rad2deg(np.arctan2(sin_values, cos_values)) % 360
)

# 11. Handle precipitation and weather description
weather_aligned["precipitation"] = (
    weather_aligned["precipitation"].ffill(limit=2)
)

weather_aligned["weather"] = (
    weather_aligned["weather"].ffill(limit=2)
)

# 12. Keep only required hourly timestamps
weather_aligned = weather_aligned.reindex(hourly_index)

# 13. Merge all three datasets
merged_df = (
    aqi_aligned
    .join(weather_aligned)
    .join(uv_aligned)
)

# 14. Reset timestamp index
merged_df = merged_df.reset_index()
merged_df = merged_df.rename(columns={"index": "timestamp"})

# 15. Extract time-based features
merged_df["hour"] = merged_df["timestamp"].dt.hour
merged_df["day"] = merged_df["timestamp"].dt.day
merged_df["month"] = merged_df["timestamp"].dt.month

# 16. Calculate rate of change
merged_df["aqi_rate_change"] = merged_df["us_aqi"].diff()
merged_df["pm25_rate_change"] = merged_df["pm2_5"].diff()
merged_df["temperature_rate_change"] = merged_df["temperature"].diff()

# 17. Calculate 3-hour rolling averages
merged_df["aqi_rolling_mean_3h"] = (
    merged_df["us_aqi"].rolling(window=3).mean()
)

merged_df["pm25_rolling_mean_3h"] = (
    merged_df["pm2_5"].rolling(window=3).mean()
)

merged_df["temperature_rolling_mean_3h"] = (
    merged_df["temperature"].rolling(window=3).mean()
)

# 18. Validate merged dataset
print("\nTotal merged records:", len(merged_df))
print("\nColumns:", merged_df.columns.tolist())

print("\nMerged Dataset Preview:")
print(merged_df.head(10))

print("\nMissing Values:")
print(merged_df.isnull().sum())

print("\nDuplicate timestamps:",
    merged_df["timestamp"].duplicated().sum())

# 19. Save merged dataset
merged_df.to_csv("merged_data.csv", index=False)

print("\nMerged data saved successfully!")
print("File: merged_data.csv")
