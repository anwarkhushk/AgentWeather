import ast
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv

from feature_store_data import (
    AQI_CANONICAL_COLUMNS,
    AQI_FEATURE_GROUP_NAME,
    AQI_FEATURE_GROUP_VERSION,
    EXPECTED_PURGE_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    MULTIMETRIC_CANONICAL_COLUMNS,
    MULTIMETRIC_FEATURE_GROUP_NAME,
    MULTIMETRIC_FEATURE_GROUP_VERSION,
    load_aqi_training_data,
    load_multimetric_training_data,
)

load_dotenv()

TRAINING_SCRIPTS = [
    "train_multimetric_rf.py",
    "train_multimetric_lstm.py",
    "train_weather_sarimax.py",
    "evaluate_rain_classifier_candidate.py",
    "random_forest_model.py",
    "sarimax_model.py",
    "lstm_model.py",
    "baseline_model.py",
    "compare_validations.py",
    "final_test_evaluation.py",
]

FORBIDDEN_CSV_INPUTS = {
    "training_data.csv",
    "testing_data.csv",
    "multi_metric_training_data.csv",
    "multi_metric_testing_data.csv",
    "historical_featured_data.csv",
    "historical_merged_data.csv",
}


def verify_scripts_use_feature_store() -> dict:
    """
    Parse the AST of each training script to verify it imports and calls
    load_multimetric_training_data or load_aqi_training_data from
    feature_store_data and does not read local training/testing CSV files.
    """
    report = {}
    for script_name in TRAINING_SCRIPTS:
        path = Path(script_name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=script_name)

        imported_loaders = set()
        called_loaders = set()
        forbidden_csv_reads = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "feature_store_data":
                for alias in node.names:
                    imported_loaders.add(alias.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in (
                    "load_multimetric_training_data",
                    "load_aqi_training_data",
                ):
                    called_loaders.add(node.func.id)
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read_csv"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in FORBIDDEN_CSV_INPUTS
                ):
                    forbidden_csv_reads.append(node.args[0].value)

        if not imported_loaders or not called_loaders or forbidden_csv_reads:
            raise AssertionError(
                f"{script_name} failed Feature Store AST check: "
                f"imported={imported_loaders}, called={called_loaders}, "
                f"forbidden_csv_reads={forbidden_csv_reads}"
            )

        report[script_name] = {
            "loader": sorted(called_loaders)[0],
            "uses_feature_store": True,
            "reads_local_training_csv": False,
        }
    return report


def main():
    print("=" * 70)
    print("VERIFYING HOPSWORKS FEATURE STORE TRAINING PIPELINE")
    print("=" * 70)

    # 1. Connect to Hopsworks Feature Store
    project = hopsworks.login(engine="python")
    fs = project.get_feature_store()
    print(f"\n1. Feature Store Connection Status: CONNECTED (Project: {project.name}, Feature Store: {fs.name})")

    # 2. Retrieve AQI and Multi-Metric Feature Groups through actual training loaders
    print("\n2. Feature Groups Used & Live Retrieval:")
    aqi_train, aqi_test = load_aqi_training_data(feature_store=fs)
    mm_train, mm_test = load_multimetric_training_data(feature_store=fs)

    for label, fg_name, fg_ver, tr_df, te_df, expected_cols in [
        (
            "AQI Training Pipeline",
            AQI_FEATURE_GROUP_NAME,
            AQI_FEATURE_GROUP_VERSION,
            aqi_train,
            aqi_test,
            AQI_CANONICAL_COLUMNS,
        ),
        (
            "Multi-Metric Weather Training Pipeline",
            MULTIMETRIC_FEATURE_GROUP_NAME,
            MULTIMETRIC_FEATURE_GROUP_VERSION,
            mm_train,
            mm_test,
            MULTIMETRIC_CANONICAL_COLUMNS,
        ),
    ]:
        assert tr_df.attrs.get("source") == "hopsworks_feature_store"
        assert te_df.attrs.get("source") == "hopsworks_feature_store"
        assert tr_df.attrs.get("feature_group") == fg_name
        assert tr_df.attrs.get("retrieved_total_rows") == EXPECTED_TOTAL_ROWS
        assert len(tr_df) == EXPECTED_TRAIN_ROWS
        assert len(te_df) == EXPECTED_TEST_ROWS
        assert list(tr_df.columns) == expected_cols
        assert list(te_df.columns) == expected_cols
        assert tr_df.isna().sum().sum() == 0
        assert te_df.isna().sum().sum() == 0
        assert tr_df["timestamp"].is_monotonic_increasing
        assert te_df["timestamp"].is_monotonic_increasing
        assert not tr_df["timestamp"].duplicated().any()
        assert not te_df["timestamp"].duplicated().any()

        gap_hours = (
            te_df["timestamp"].min() - tr_df["timestamp"].max()
        ) / pd.Timedelta(hours=1)
        assert gap_hours == EXPECTED_PURGE_ROWS + 1, (
            f"Expected {EXPECTED_PURGE_ROWS} purged hours (25h difference), got {gap_hours}"
        )

        print(f"   - [{label}]")
        print(f"     * Feature Group:             {fg_name} (v{fg_ver})")
        print("     * Retrieval Status:          SUCCESS (Hopsworks Feature Store)")
        print(f"     * Retrieved Total Row Count: {EXPECTED_TOTAL_ROWS}")
        print(
            f"     * Chronological Split:       train={len(tr_df)} rows, "
            f"purged={EXPECTED_PURGE_ROWS} rows, test={len(te_df)} rows"
        )
        print(f"     * Columns Verified:          {len(expected_cols)}/{len(expected_cols)} required columns present (0 NaNs)")
        print(
            f"     * Timestamp Ordering:        Monotonic increasing "
            f"(train {tr_df['timestamp'].min()} -> {tr_df['timestamp'].max()}; "
            f"test {te_df['timestamp'].min()} -> {te_df['timestamp'].max()})"
        )

    # 3. Smoke test of training-pipeline feature preparation & model inference on Hopsworks data
    print("\n3. Safe Training Pipeline Smoke Test (No Registry Mutation):")
    df_smoke = mm_train.copy()
    df_smoke["hour"] = df_smoke["timestamp"].dt.hour
    df_smoke["day_of_week"] = df_smoke["timestamp"].dt.dayofweek
    df_smoke["month"] = df_smoke["timestamp"].dt.month
    df_smoke["hour_sin"] = np.sin(2 * np.pi * df_smoke["hour"] / 24)
    df_smoke["hour_cos"] = np.cos(2 * np.pi * df_smoke["hour"] / 24)
    df_smoke["month_sin"] = np.sin(2 * np.pi * df_smoke["month"] / 12)
    df_smoke["month_cos"] = np.cos(2 * np.pi * df_smoke["month"] / 12)

    split = int(len(df_smoke) * 0.80)
    fit_end = split - EXPECTED_PURGE_ROWS
    assert fit_end == 16668 and len(df_smoke.iloc[split:]) == 4173

    temp_rf = joblib.load("multimetric_models/temperature_rf.pkl")
    feature_cols = list(temp_rf.feature_names_in_)
    sample_preds = temp_rf.predict(df_smoke[feature_cols].iloc[split:split + 5])
    assert len(sample_preds) == 5 and np.isfinite(sample_preds).all()
    print(
        f"   - Chronological validation split verified: fit={fit_end} rows, "
        f"purge={EXPECTED_PURGE_ROWS} rows, val={len(df_smoke.iloc[split:])} rows"
    )
    print(
        f"   - Feature matrix ({len(feature_cols)} features) & model smoke test passed on "
        "Feature Store data."
    )

    # 4. Verify all training scripts use Feature Store loaders rather than local CSVs
    print("\n4. Training Script Source Verification (Feature Store vs Local CSV):")
    script_report = verify_scripts_use_feature_store()
    for script_name, info in script_report.items():
        print(
            f"   - {script_name}: loader={info['loader']}() | "
            f"uses_feature_store={info['uses_feature_store']} | "
            f"reads_local_training_csv={info['reads_local_training_csv']}"
        )

    print("\n" + "=" * 70)
    print("ALL HOPSWORKS FEATURE STORE TRAINING VERIFICATIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
