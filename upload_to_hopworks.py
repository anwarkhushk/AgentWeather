
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Connect to Hopsworks
project = hopsworks.login(engine="python")
fs = project.get_feature_store()

print("\nConnected to Hopsworks!")
print("Feature Store:", fs.name)


# 3. Load datasets
aqi_df = pd.read_csv("aqi_data.csv")
weather_df = pd.read_csv("weather_data.csv")
features_df = pd.read_csv("merged_data.csv")


# 4. Prepare timestamps
aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])


# 5. Add location identifier
aqi_df["location"] = "karachi"
weather_df["location"] = "karachi"
features_df["location"] = "karachi"


# 6. Define upload function
def upload_feature_group(df, name, description):

    print(f"\nPreparing Feature Group: {name}")

    feature_group = fs.get_or_create_feature_group(
        name=name,
        version=1,
        description=description,
        primary_key=["location"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
        stream=True
    )

    print(f"Uploading {len(df)} records...")

    feature_group.insert(
        df,
        write_options={"wait_for_job": True}
    )

    print(f"Successfully uploaded: {name}")


# 7. Upload raw AQI data
upload_feature_group(
    aqi_df,
    "raw_aqi",
    "Raw air quality data from Open-Meteo API"
)


# 8. Upload raw weather data
upload_feature_group(
    weather_df,
    "raw_weather",
    "Raw weather forecast data from OpenWeather API"
)


# 9. Upload engineered weather features
upload_feature_group(
    features_df,
    "weather_features",
    "Merged and engineered weather and air quality features"
)


print("\nAll datasets uploaded successfully!")
