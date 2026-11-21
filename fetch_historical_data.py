import requests
import pandas as pd

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011

# Historical training period
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# API configurations
APIS = {
    "weather": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "pressure_msl,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "precipitation"
            
        ),
        "output": "historical_weather.csv"
    },

    "aqi": {
        "url": "https://air-quality-api.open-meteo.com/v1/air-quality",
        "hourly": "us_aqi,pm2_5,pm10",
        "output": "historical_aqi.csv"
    }
}


def fetch_historical_data(name, config):

    print(f"\nFetching historical {name.upper()} data...")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": config["hourly"],
        "timezone": "Asia/Karachi"
    }

    response = requests.get(
        config["url"],
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    # Standardize timestamp column
    df = df.rename(columns={"time": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate dataset
    print("Total records:", len(df))
    print("Date range:", df["timestamp"].min(),
        "to", df["timestamp"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate timestamps:",
        df["timestamp"].duplicated().sum())

    # Save CSV
    df.to_csv(config["output"], index=False)

    print(f"\nSaved successfully: {config['output']}")

    return df


if __name__ == "__main__":

    weather_df = fetch_historical_data(
        "weather", APIS["weather"]
    )

    aqi_df = fetch_historical_data(
        "aqi", APIS["aqi"]
    )

    print("\n" + "=" * 50)
    print("HISTORICAL DATA COLLECTION COMPLETED")
    print("=" * 50)

    print("Weather records:", len(weather_df))
    print("AQI records:", len(aqi_df))