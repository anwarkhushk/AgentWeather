import os
import json
import shutil
from pathlib import Path
import hopsworks

project = hopsworks.login(engine="python")
mr = project.get_model_registry()
api = project.get_dataset_api()

models = [
    {
        "name": "aqi_lstm",
        "framework": "tensorflow",
        "files": [
            "lstm_aqi_final.keras",
            "lstm_feature_scaler.pkl",
            "lstm_target_scaler.pkl"
        ],
        "metrics": {
            "MAE": 13.1765,
            "RMSE": 17.6329,
            "R2": 0.5360
        },
        "description": "LSTM AQI 24-hour forecasting model. "
                       "Best AQI validation candidate."
    },
    {
        "name": "aqi_sarimax",
        "framework": "python",
        "files": [
            "sarimax_aqi_final.pkl",
            "sarimax_feature_scaler.pkl",
            "sarimax_target_scaler.pkl"
        ],
        "metrics": {
            "MAE": 20.5010,
            "RMSE": 25.8575,
            "R2": 0.0022
        },
        "description": "SARIMAX AQI 24-hour forecasting comparison model."
    }
]

features = {
    "aqi_lstm": [
        "us_aqi", "pm2_5", "pm10", "temperature_2m",
        "relative_humidity_2m", "wind_speed_10m",
        "aqi_lag_24", "aqi_rolling_mean_24h"
    ],
    "aqi_sarimax": [
        "us_aqi", "pm2_5", "pm10",
        "aqi_lag_24", "aqi_rolling_mean_24h"
    ]
}

for config in models:
    name = config["name"]
    version = 1
    artifact_dir = Path("registry_artifacts") / name

    print(f"\n========== {name} ==========")

    existing = mr.get_models(name=name) or []
    existing_versions = [int(m.version) for m in existing]

    if version in existing_versions:
        print(f"Version {version} already exists. Skipping.")
        continue

    artifact_dir.mkdir(parents=True, exist_ok=True)

    for filename in config["files"]:
        source = Path(filename)

        if not source.exists():
            raise FileNotFoundError(f"Missing local artifact: {source}")

        shutil.copy2(source, artifact_dir / filename)

    metadata = {
        "model_name": name,
        "target": "target_aqi_24h",
        "forecast_horizon_hours": 24,
        "features": features[name],
        "validation_metrics": config["metrics"]
    }

    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    module = getattr(mr, config["framework"])

    model = module.create_model(
        name=name,
        version=version,
        metrics=config["metrics"],
        description=config["description"]
    )

    print("Uploading artifacts...")
    registered = model.save(
        str(artifact_dir),
        keep_original_files=True
    )

    print("Registered:", registered.name)
    print("Version:", registered.version)

    path = registered.model_files_path
    print("Artifact path:", path)

    if not api.path_exists(path):
        raise RuntimeError(f"Artifact directory missing: {path}")

    print("Uploaded files:", api.list(path))
    print("Local artifacts preserved:",
          all((artifact_dir / f).exists() for f in config["files"]))

print("\nRegistration process finished.")
