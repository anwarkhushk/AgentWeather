import requests
import pandas as pd

LATITUDE = 24.8607
LONGITUDE = 67.0011

weather = pd.read_csv("historical_merged_data.csv")
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

start_date = weather["timestamp"].min().strftime("%Y-%m-%d")
end_date = weather["timestamp"].max().strftime("%Y-%m-%d")

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "uv_index,precipitation_probability",
    "timezone": "Asia/Karachi"
}

print(f"Fetching {start_date} to {end_date}...")

response = requests.get(url, params=params, timeout=180)
print("Status:", response.status_code)
response.raise_for_status()

hourly = response.json()["hourly"]

df = pd.DataFrame({
    "timestamp": pd.to_datetime(hourly["time"]),
    "uv_index": hourly["uv_index"],
    "precipitation_probability": hourly["precipitation_probability"]
})

df.to_csv("historical_auxiliary_data.csv", index=False)

print("\nSaved historical_auxiliary_data.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("\nSample:")
print(df.head())
print("\nDate range:", df.timestamp.min(), "to", df.timestamp.max())
