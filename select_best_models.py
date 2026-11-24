import json
import pandas as pd

# AQI validation comparison
aqi = pd.read_csv("validation_model_comparison.csv")
aqi_best = aqi.loc[aqi["RMSE"].idxmin()]

# Weather validation comparison (includes Wind direction and R2)
weather = pd.read_csv("multimetric_results/final_validation_comparison.csv")

selection = {
    "AQI": {
        "model": aqi_best["Model"],
        "validation_MAE": float(aqi_best["MAE"]),
        "validation_RMSE": float(aqi_best["RMSE"]),
        "validation_R2": float(aqi_best["R2"]),
        "promotion_candidate": bool(aqi_best["Model"] != "Naive Baseline")
    }
}

for metric, group in weather.groupby("Metric", sort=False):
    best = group.loc[group["RMSE"].idxmin()]
    entry = {
        "model": best["Model"],
        "validation_MAE": float(best["MAE"]),
        "validation_RMSE": float(best["RMSE"]),
        "validation_R2": float(best["R2"]),
        "promotion_candidate": bool(best["Model"] != "Naive")
    }
    selection[metric] = entry

# Rain classifier uses its monsoon holdout validation and same-protocol comparison
with open(
    "multimetric_results/rain_classifier_monsoon_candidate.json"
) as f:
    rain = json.load(f)

selection["Rain probability"] = {
    "model": "Random Forest",
    "validation_F1": rain["validation_metrics"]["f1"],
    "validation_PR_AUC": rain["validation_metrics"]["pr_auc"],
    "validation_MAE": rain["validation_metrics"]["mae"],
    "validation_RMSE": rain["validation_metrics"]["rmse"],
    "validation_R2": rain["validation_metrics"]["r2"],
    "incumbent_threshold_0.05_F1": rain["incumbent_validation_metrics"]["f1"],
    "naive_validation_F1": rain["naive_validation_metrics"]["f1"],
    "naive_validation_PR_AUC": rain["naive_validation_metrics"]["pr_auc"],
    "naive_validation_RMSE": rain["naive_validation_metrics"]["rmse"],
    "threshold": rain["threshold"],
    "promotion_candidate": False,
    "selection_note": (
        "Random Forest beats Naive baseline on probability RMSE (0.2148 vs 0.2718) "
        "and PR-AUC (0.2083 vs 0.1304) and is already registered as rain_probability_rf v1 "
        "with identical model weights; candidate threshold 0.15 improves F1 over incumbent "
        "threshold 0.05 (0.3002 vs 0.2178) but does not exceed Naive binary persistence F1 (0.3057)."
    )
}

with open("multimetric_results/final_model_selection.json", "w") as f:
    json.dump(selection, f, indent=4)

for target, result in selection.items():
    print(
        f"{target}: {result['model']} | "
        f"Promotion candidate: {result['promotion_candidate']}"
    )

print("\nSelection report saved.")
