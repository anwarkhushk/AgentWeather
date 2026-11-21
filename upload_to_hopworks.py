
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

# 3. Load all four datasets
aqi_df = pd.read_csv("aqi_data.csv")
weather_df = pd.read_csv("weather_data.csv")
uv_df = pd.read_csv("uv_data.csv")
features_df = pd.read_csv("merged_data.csv")

# 4. Convert Karachi local timestamps to UTC
for df in [aqi_df, weather_df, uv_df, features_df]:
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize("Asia/Karachi")
        .dt.tz_convert("UTC")
    )

# 5. Add location identifier
for df in [aqi_df, weather_df, uv_df, features_df]:
    df["location"] = "karachi"

# 6. Define upload function
def upload_feature_group(df, name, version, description):

    print(f"\nPreparing Feature Group: {name} v{version}")

    feature_group = fs.get_or_create_feature_group(
        name=name,
        version=version,
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

    print(f"Successfully uploaded: {name} v{version}")


# 7. Upload raw AQI
upload_feature_group(
    aqi_df,
    "raw_aqi",
    2,
    "Raw AQI observations from Open-Meteo Air Quality API"
)

# 8. Upload raw weather
upload_feature_group(
    weather_df,
    "raw_weather",
    2,
    "Raw weather forecasts from OpenWeather API"
)

# 9. Upload raw UV
upload_feature_group(
    uv_df,
    "raw_uv",
    1,
    "Raw hourly UV Index forecasts from Open-Meteo API"
)

# 10. Upload engineered features
upload_feature_group(
    features_df,
    "weather_features",
    2,
    "Merged AQI, weather and UV features with feature engineering"
)

print("\nAll datasets uploaded successfully!")
