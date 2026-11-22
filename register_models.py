
import os
import json
import shutil
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(engine="python")
mr = project.get_model_registry()

# Load feature names
train = pd.read_csv("training_data.csv")

rf_features = [
    col for col in train.columns
    if col not in ["timestamp", "target_aqi_24h"]
]

lstm_features = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "aqi_lag_24",
    "aqi_rolling_mean_24h"
]

sarimax_features = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "aqi_lag_24",
    "aqi_rolling_mean_24h"
]

# Validation metrics: model selection is based on validation,
# NOT final test performance.
models = [
    {
        "name": "aqi_random_forest",
        "framework": "sklearn",
        "artifact_files": [
            "random_forest_aqi_final.pkl"
        ],
        "features": rf_features,
        "metrics": {
            "MAE": 13.3464,
            "RMSE": 17.6929,
            "R2": 0.5328
        },
        "description": "Random Forest AQI 24-hour forecasting model."
    },
    {
        "name": "aqi_lstm",
        "framework": "tensorflow",
        "artifact_files": [
            "lstm_aqi_final.keras",
            "lstm_feature_scaler.pkl",
            "lstm_target_scaler.pkl"
        ],
        "features": lstm_features,
        "metrics": {
            "MAE": 13.1765,
            "RMSE": 17.6329,
            "R2": 0.5360
        },
        "description": "LSTM AQI 24-hour forecasting model. Best validation MAE."
    },
    {
        "name": "aqi_sarimax",
        "framework": "python",
        "artifact_files": [
            "sarimax_aqi_final.pkl",
            "sarimax_feature_scaler.pkl",
            "sarimax_target_scaler.pkl"
        ],
        "features": sarimax_features,
        "metrics": {
            "MAE": 20.5010,
            "RMSE": 25.8575,
            "R2": 0.0022
        },
        "description": "SARIMAX AQI 24-hour forecasting model."
    }
]

artifact_root = "registry_artifacts"
os.makedirs(artifact_root, exist_ok=True)

for config in models:

    name = config["name"]
    artifact_dir = os.path.join(artifact_root, name)

    print(f"\n========== {name} ==========")

    # Prevent accidental duplicate registration on reruns
    existing = mr.get_models(name=name)

    if existing:
        print(f"Already registered: {name}")
        print("Skipping to avoid creating an unintended new version.")
        continue

    os.makedirs(artifact_dir, exist_ok=True)

    # Copy model and supporting files
    for filename in config["artifact_files"]:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Missing model artifact: {filename}"
            )

        shutil.copy2(
            filename,
            os.path.join(artifact_dir, filename)
        )

    # Save feature metadata
    metadata = {
        "model_name": name,
        "target": "target_aqi_24h",
        "forecast_horizon_hours": 24,
        "features": config["features"],
        "validation_metrics": config["metrics"]
    }

    with open(
        os.path.join(artifact_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    # Create appropriate model object
    model_module = getattr(mr, config["framework"])

    model = model_module.create_model(
        name=name,
        metrics=config["metrics"],
        description=config["description"],
        tags=[
            {"name": "project", "value": "AgentWeather"},
            {"name": "target", "value": "AQI"},
            {"name": "forecast_horizon", "value": "24h"},
            {
                "name": "selection",
                "value": (
                    "best_validation_candidate"
                    if name == "aqi_lstm"
                    else "comparison_model"
                )
            }
        ]
    )

    # Register artifacts and metadata
    registered_model = model.save(artifact_dir)

    print("Registered successfully!")
    print("Name:", registered_model.name)
    print("Version:", registered_model.version)

print("\nModel registration process completed.")
