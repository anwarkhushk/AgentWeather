import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

EXPECTED_RECORDS = 26112
FEATURE_GROUP_NAME = "historical_multimetric_features"
FEATURE_GROUP_VERSION = 1


def build_multimetric_historical_dataframe() -> pd.DataFrame:
    """
    Build the complete 26,112-row historical multi-metric feature and target
    dataset aligned with historical_aqi_features v1 (2023-01-08 00:00:00 to
    2025-12-30 23:00:00 Asia/Karachi).
    """
    weather = pd.read_csv("historical_merged_data.csv")
    aux = pd.read_csv("historical_auxiliary_data.csv")
    aqi_featured = pd.read_csv("historical_featured_data.csv", usecols=["timestamp"])

    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    aux["timestamp"] = pd.to_datetime(aux["timestamp"])
    aqi_featured["timestamp"] = pd.to_datetime(aqi_featured["timestamp"])

    df = weather.merge(
        aux[["timestamp", "uv_index"]],
        on="timestamp",
        how="left",
        validate="one_to_one"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    radians = np.deg2rad(df["wind_direction_10m"])
    df["wind_direction_sin"] = np.sin(radians)
    df["wind_direction_cos"] = np.cos(radians)

    targets = {
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "precipitation": "precipitation",
        "uv": "uv_index",
        "wind_speed": "wind_speed_10m",
        "wind_direction_sin": "wind_direction_sin",
        "wind_direction_cos": "wind_direction_cos",
    }

    for name, column in targets.items():
        df[f"target_{name}_24h"] = df[column].shift(-24)

    df["target_rain_probability_24h"] = (
        df["precipitation"].shift(-24) > 0.1
    ).astype(float)
    df.loc[
        df["precipitation"].shift(-24).isna(),
        "target_rain_probability_24h"
    ] = np.nan

    valid_timestamps = set(aqi_featured["timestamp"])
    df = (
        df[df["timestamp"].isin(valid_timestamps)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(df) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, found {len(df)}"
        )

    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains missing values!")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found!")

    return df


if __name__ == "__main__":
    df = build_multimetric_historical_dataframe()
    print(f"Prepared historical multi-metric dataset: {len(df)} records, {len(df.columns)} columns")

    if df["timestamp"].dt.tz is not None:
        raise ValueError("Expected timezone-naive Karachi timestamps before conversion!")

    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize("Asia/Karachi")
        .dt.tz_convert("UTC")
    )

    project = hopsworks.login(engine="python")
    fs = project.get_feature_store()
    print("Connected to Hopsworks Feature Store:", fs.name)

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Historical AQI, weather, and UV features with 24-hour-ahead "
            "multi-metric forecast targets"
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI"
    )

    if feature_group.id is not None:
        try:
            existing_df = feature_group.read()
            if len(existing_df) == EXPECTED_RECORDS:
                print(
                    f"Feature Group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} "
                    f"already contains {len(existing_df)} records. Skipping insert."
                )
                raise SystemExit(0)
        except Exception:
            pass

    print(f"Uploading {len(df)} records to {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    feature_group.insert(
        df,
        write_options={"wait_for_job": True}
    )

    uploaded_df = feature_group.read()
    print("Expected records:", EXPECTED_RECORDS)
    print("Uploaded records:", len(uploaded_df))
    if len(uploaded_df) != EXPECTED_RECORDS:
        raise ValueError("Upload verification failed!")
    print("HISTORICAL MULTIMETRIC FEATURE GROUP VERIFIED!")
