import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

AQI_FEATURE_GROUP_NAME = "historical_aqi_features"
AQI_FEATURE_GROUP_VERSION = 1

MULTIMETRIC_FEATURE_GROUP_NAME = "historical_multimetric_features"
MULTIMETRIC_FEATURE_GROUP_VERSION = 1

EXPECTED_TOTAL_ROWS = 26112
EXPECTED_TRAIN_ROWS = 20865
EXPECTED_PURGE_ROWS = 24
EXPECTED_TEST_ROWS = 5223

AQI_CANONICAL_COLUMNS = [
    "timestamp",
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
    "target_aqi_24h",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_168",
    "aqi_rolling_mean_3h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_mean_168h",
    "wind_direction_sin",
    "wind_direction_cos",
]

MULTIMETRIC_CANONICAL_COLUMNS = [
    "timestamp",
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "uv_index",
    "wind_direction_sin",
    "wind_direction_cos",
    "target_temperature_24h",
    "target_humidity_24h",
    "target_precipitation_24h",
    "target_uv_24h",
    "target_wind_speed_24h",
    "target_wind_direction_sin_24h",
    "target_wind_direction_cos_24h",
    "target_rain_probability_24h",
]


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UTC timestamps returned by Hopsworks back to timezone-naive
    Asia/Karachi timestamps and sort chronologically.
    """
    if "timestamp" not in df.columns:
        raise RuntimeError("Retrieved Feature Group is missing 'timestamp' column.")

    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Karachi").dt.tz_localize(None)

    df = df.copy()
    df["timestamp"] = ts
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _validate_and_split(
    df: pd.DataFrame,
    canonical_columns: list[str],
    fg_name: str,
    fg_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate schema, row count, timestamp monotonicity, and split
    chronologically (80/20 with 24-hour purge gap).
    """
    missing_cols = [c for c in canonical_columns if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"Feature Group '{fg_name}' v{fg_version} is missing required columns: "
            f"{missing_cols}"
        )

    df = _normalize_timestamps(df)
    df = df[canonical_columns].copy()

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            f"Duplicate timestamps detected in Feature Group '{fg_name}' v{fg_version}."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError(
            f"Timestamps are not monotonic increasing in '{fg_name}' v{fg_version}."
        )

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Unexpected row count from '{fg_name}' v{fg_version}: "
            f"expected {EXPECTED_TOTAL_ROWS}, got {len(df)}."
        )

    split_index = int(len(df) * 0.80)
    train_end = split_index - EXPECTED_PURGE_ROWS

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)

    if (
        train_df["timestamp"].max() + pd.Timedelta(hours=EXPECTED_PURGE_ROWS)
        >= test_df["timestamp"].min()
    ):
        raise RuntimeError("24-hour purge gap validation failed!")

    train_df.attrs["source"] = "hopsworks_feature_store"
    train_df.attrs["feature_group"] = fg_name
    train_df.attrs["feature_group_version"] = fg_version
    train_df.attrs["retrieved_total_rows"] = len(df)

    test_df.attrs["source"] = "hopsworks_feature_store"
    test_df.attrs["feature_group"] = fg_name
    test_df.attrs["feature_group_version"] = fg_version
    test_df.attrs["retrieved_total_rows"] = len(df)

    print(
        f"[Hopsworks Feature Store] Successfully retrieved {len(df)} rows "
        f"from Feature Group '{fg_name}' (version {fg_version})."
    )
    print(
        f"[Hopsworks Feature Store] Chronological split with {EXPECTED_PURGE_ROWS}h purge: "
        f"train={len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()}), "
        f"test={len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})."
    )

    return train_df, test_df


def fetch_feature_group_dataframe(
    fg_name: str,
    fg_version: int,
    feature_store=None,
) -> pd.DataFrame:
    """
    Connect to Hopsworks Feature Store and read a Feature Group directly.
    Never falls back to local CSV files.
    """
    try:
        if feature_store is None:
            project = hopsworks.login(engine="python")
            feature_store = project.get_feature_store()

        fg = feature_store.get_feature_group(name=fg_name, version=fg_version)
        if fg is None:
            raise RuntimeError(
                f"Feature Group '{fg_name}' (version {fg_version}) not found in Hopsworks."
            )

        df = fg.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve training data from Hopsworks Feature Store "
            f"(Feature Group '{fg_name}' v{fg_version}): {exc}"
        ) from exc

    if df is None or len(df) == 0:
        raise RuntimeError(
            f"Hopsworks Feature Group '{fg_name}' v{fg_version} returned 0 rows."
        )

    return df


def load_multimetric_training_data(feature_store=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch multi-metric features and 24h targets directly from Hopsworks Feature Store
    ('historical_multimetric_features' v1) and return chronological (train_df, test_df).
    """
    df = fetch_feature_group_dataframe(
        MULTIMETRIC_FEATURE_GROUP_NAME,
        MULTIMETRIC_FEATURE_GROUP_VERSION,
        feature_store=feature_store,
    )
    return _validate_and_split(
        df,
        MULTIMETRIC_CANONICAL_COLUMNS,
        MULTIMETRIC_FEATURE_GROUP_NAME,
        MULTIMETRIC_FEATURE_GROUP_VERSION,
    )


def load_aqi_training_data(feature_store=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch AQI engineered features and 24h target directly from Hopsworks Feature Store
    ('historical_aqi_features' v1) and return chronological (train_df, test_df).
    """
    df = fetch_feature_group_dataframe(
        AQI_FEATURE_GROUP_NAME,
        AQI_FEATURE_GROUP_VERSION,
        feature_store=feature_store,
    )
    return _validate_and_split(
        df,
        AQI_CANONICAL_COLUMNS,
        AQI_FEATURE_GROUP_NAME,
        AQI_FEATURE_GROUP_VERSION,
    )
