import json
import shutil
from pathlib import Path
import joblib
import hopsworks

ROOT = Path(".")
ARTIFACT_ROOT = ROOT / "registry_artifacts"

project = hopsworks.login(engine="python")
mr = project.get_model_registry()
api = project.get_dataset_api()

# Validation metrics only. Test metrics are not used for selection.
sarimax_configs = {
    "temperature_sarimax": {
        "target": "target_temperature_24h",
        "MAE": 0.824047,
        "RMSE": 1.117399,
    },
    "humidity_sarimax": {
        "target": "target_humidity_24h",
        "MAE": 10.260593,
        "RMSE": 13.906493,
    },
    "precipitation_sarimax": {
        "target": "target_precipitation_24h",
        "MAE": 0.020835,
        "RMSE": 0.026722,
    },
    "uv_sarimax": {
        "target": "target_uv_24h",
        "MAE": 0.129344,
        "RMSE": 0.178056,
    },
    "wind_speed_sarimax": {
        "target": "target_wind_speed_24h",
        "MAE": 3.250698,
        "RMSE": 4.062173,
    },
}

def register(name, framework, source_files, metadata, metrics):
    print(f"\n========== {name} ==========")

    existing = mr.get_models(name=name) or []

    for old_model in existing:
        old_path = old_model.model_files_path
        try:
            complete = api.path_exists(old_path)
        except Exception:
            complete = False

        if complete:
            print(
                f"Complete registration already exists: "
                f"version {old_model.version}. Skipping."
            )
            return

        print(
            f"Version {old_model.version} has no remote artifact folder. "
            "Preserving it and creating a new version."
        )

    artifact_dir = ARTIFACT_ROOT / name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for source in source_files:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, artifact_dir / source.name)

    metadata["model_name"] = name
    metadata["forecast_horizon_hours"] = 24
    metadata["validation_metrics"] = metrics

    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    module = getattr(mr, framework)
    model = module.create_model(
        name=name,
        metrics=metrics,
        description=metadata["description"],
    )

    registered = model.save(
        str(artifact_dir),
        await_registration=480,
        keep_original_files=True,
    )

    print("Registered:", registered.name)
    print("Version:", registered.version)

    remote_path = registered.model_files_path
    if not api.path_exists(remote_path):
        raise RuntimeError(f"Remote artifacts not found: {remote_path}")

    print("Remote artifact verification: PASSED")
    print("Files:", api.list(remote_path))


# 1. Five SARIMAX models
for name, config in sarimax_configs.items():
    obj = joblib.load(ROOT / "multimetric_models" / f"{name}.pkl")

    metadata = {
        "target": config["target"],
        "features": obj["features"],
        "framework": "statsmodels SARIMAX",
        "selection": "best_validation_rmse_candidate",
        "description": (
            f"SARIMAX 24-hour forecast for {config['target']}."
            " Selected using validation RMSE."
        ),
    }

    register(
        name=name,
        framework="python",
        source_files=[
            ROOT / "multimetric_models" / f"{name}.pkl"
        ],
        metadata=metadata,
        metrics={
            "MAE": config["MAE"],
            "RMSE": config["RMSE"],
        },
    )


# 2. Rain probability classifier
rain_name = "rain_probability_rf"
rain_source = ROOT / "multimetric_models" / "rain_classifier_rf.pkl"

rain_model = joblib.load(rain_source)
rain_features = list(rain_model.feature_names_in_)

register(
    name=rain_name,
    framework="sklearn",
    source_files=[rain_source],
    metadata={
        "target": "rain_probability_24h",
        "features": rain_features,
        "framework": "RandomForestClassifier",
        "probability_threshold": 0.05,
        "selection": "validation_selected_threshold",
        "description": (
            "Random Forest rain probability classifier. "
            "Probability threshold selected on validation data."
        ),
    },
    metrics={},
)


# 3. Multi-target weather LSTM
lstm_name = "weather_multimetric_lstm"

prep_path = ROOT / "multimetric_models" / "weather_lstm_preprocessing.pkl"
prep = joblib.load(prep_path)

lstm_validation = {
    "temperature": {"MAE": 1.145853, "RMSE": 1.514902, "R2": 0.920648},
    "humidity": {"MAE": 11.740242, "RMSE": 15.703355, "R2": 0.535117},
    "precipitation": {"MAE": 0.022027, "RMSE": 0.030959, "R2": -131.685489},
    "uv": {"MAE": 0.330673, "RMSE": 0.466329, "R2": 0.960745},
    "wind_speed": {"MAE": 4.624373, "RMSE": 5.464776, "R2": -1.092073},
    "wind_direction_sin": {"MAE": 0.320500, "RMSE": 0.451392, "R2": 0.565419},
    "wind_direction_cos": {"MAE": 0.305353, "RMSE": 0.392187, "R2": 0.570896},
}

lstm_metrics = {}
for target, values in lstm_validation.items():
    for metric, value in values.items():
        lstm_metrics[f"{target}_{metric}"] = value

register(
    name=lstm_name,
    framework="tensorflow",
    source_files=[
        ROOT / "multimetric_models" / "weather_multimetric_lstm.keras",
        prep_path,
    ],
    metadata={
        "targets": prep["targets"],
        "features": prep["features"],
        "sequence_length": prep["sequence_length"],
        "framework": "TensorFlow Keras",
        "target_validation_metrics": lstm_validation,
        "selection": "multi_target_comparison_candidate",
        "description": (
            "Multi-target 24-hour weather LSTM. "
            "Predicts seven weather output components. "
            "Registered as a comparison candidate, not the overall best model."
        ),
    },
    metrics=lstm_metrics,
)

print("\n========== REGISTRATION SCRIPT FINISHED ==========")
