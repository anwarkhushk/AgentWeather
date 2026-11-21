
import requests
import pandas as pd

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011

URL = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "uv_index",
    "timezone": "Asia/Karachi",
    "forecast_days": 5
}

# Fetch UV data
response = requests.get(URL, params=params, timeout=30)

print("Status Code:", response.status_code)
response.raise_for_status()

data = response.json()

# Extract hourly UV data
uv_df = pd.DataFrame({
    "timestamp": data["hourly"]["time"],
    "uv_index": data["hourly"]["uv_index"]
})

uv_df["timestamp"] = pd.to_datetime(uv_df["timestamp"])

# Validate
print("\nTotal UV records:", len(uv_df))
print("\nFirst 5 records:")
print(uv_df.head())

print("\nMissing UV values:")
print(uv_df["uv_index"].isnull().sum())

# Save dataset
uv_df.to_csv("uv_data.csv", index=False)

print("\nUV data saved successfully!")
print("File: uv_data.csv")
