import json
from model_promotion import should_promote

with open("multimetric_results/final_model_selection.json") as f:
    selection = json.load(f)

with open("multimetric_results/rain_classifier_monsoon_candidate.json") as f:
    rain = json.load(f)

registered = {
    "AQI": ("aqi_lstm", 1),
    "Temperature": ("temperature_sarimax", 2),
    "Humidity": ("humidity_sarimax", 1),
    "Precipitation": ("precipitation_sarimax", 1),
    "UV": ("uv_sarimax", 1),
    "Wind speed": ("wind_speed_sarimax", 1),
    "Wind direction": ("weather_multimetric_lstm", 1),
    "Rain probability": ("rain_probability_rf", 1),
}

review = {}

for metric, result in selection.items():
    if metric == "Rain probability":
        cand_val = rain["validation_metrics"]
        inc_val = rain["incumbent_validation_metrics"]
        naive_val = rain["naive_validation_metrics"]

        promote_f1, reason_f1 = should_promote(
            cand_val["f1"], inc_val["f1"], naive_val["f1"], direction="higher"
        )
        promote_rmse, reason_rmse = should_promote(
            cand_val["rmse"], inc_val["rmse"], naive_val["rmse"], direction="lower"
        )

        model_name, version = registered[metric]
        review[metric] = {
            "champion": "Random Forest (probability) / Naive persistence (binary F1)",
            "registry_model": model_name,
            "registry_version": version,
            "candidate": "Random Forest (monsoon threshold 0.15)",
            "same_protocol_validation": {
                "candidate_threshold_0.15": {
                    "F1": cand_val["f1"],
                    "PR_AUC": cand_val["pr_auc"],
                    "RMSE": cand_val["rmse"],
                    "MAE": cand_val["mae"],
                    "R2": cand_val["r2"]
                },
                "incumbent_threshold_0.05": {
                    "F1": inc_val["f1"],
                    "PR_AUC": inc_val["pr_auc"],
                    "RMSE": inc_val["rmse"],
                    "MAE": inc_val["mae"],
                    "R2": inc_val["r2"]
                },
                "naive_baseline_24h": {
                    "F1": naive_val["f1"],
                    "PR_AUC": naive_val["pr_auc"],
                    "RMSE": naive_val["rmse"],
                    "MAE": naive_val["mae"],
                    "R2": naive_val["r2"]
                }
            },
            "should_promote_by_f1": promote_f1,
            "should_promote_by_f1_reason": reason_f1,
            "should_promote_by_probability_rmse": promote_rmse,
            "should_promote_by_probability_rmse_reason": reason_rmse,
            "status": "EXISTING_CHAMPION",
            "reason": (
                "Same-protocol monsoon validation comparison completed. "
                "Candidate and registered incumbent (rain_probability_rf v1) share identical "
                "RandomForestClassifier weights (PR-AUC=0.2083, RMSE=0.2148 < Naive 0.2718). "
                "Threshold 0.15 improves F1 over incumbent threshold 0.05 (0.3002 > 0.2178) "
                "but does not exceed Naive binary F1 (0.3057); no automatic registry replacement executed."
            )
        }
    elif result["model"] == "Naive":
        model_name, version = registered[metric]
        review[metric] = {
            "champion": "Naive baseline",
            "registered_comparison_model": model_name,
            "registered_comparison_version": version,
            "status": "RETAIN_BASELINE",
            "reason": "No ML model beats the validation baseline."
        }
    else:
        model_name, version = registered[metric]
        review[metric] = {
            "champion": result["model"],
            "registry_model": model_name,
            "registry_version": version,
            "status": "EXISTING_CHAMPION",
            "reason": "Selected validation winner is already registered."
        }

with open("multimetric_results/promotion_review.json", "w") as f:
    json.dump(review, f, indent=4)

print(json.dumps(review, indent=4))
print("\nRead-only promotion review saved.")
