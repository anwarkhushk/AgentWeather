
import requests
import pandas as pd

# Karachi coordinates
latitude = 24.8607
longitude = 67.0011

# Open-Meteo API endpoint
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Parameters for our API request
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "us_aqi,pm2_5,pm10",
    "timezone": "Asia/Karachi",
    "forecast_days": 1
}

# Send request to API
response = requests.get(url, params=params, timeout=30)

# Check response
print("Status Code:", response.status_code)

if response.status_code == 200:

    data = response.json()
    hourly = data["hourly"]

    # Convert parallel arrays into individual records
    records = []

    for i in range(len(hourly["time"])):

        record = {
            "timestamp": hourly["time"][i],
            "us_aqi": hourly["us_aqi"][i],
            "pm2_5": hourly["pm2_5"][i],
            "pm10": hourly["pm10"][i]
        }

        records.append(record)

    print("\nTotal records:", len(records))
    print("\nFirst 5 records:")

    for record in records[:5]:
        print(record)

    # Convert records into Pandas DataFrame
    df = pd.DataFrame(records)

    # Save DataFrame into CSV
    df.to_csv("aqi_data.csv", index=False)

    print("\nAQI data saved successfully!")
    print("File: aqi_data.csv")

    print("\nDataFrame Preview:")
    print(df.head())

else:
    print("Request failed:", response.text)
