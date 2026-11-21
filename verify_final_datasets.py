
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD ALL REQUIRED DATASETS
# ==========================================

raw = pd.read_csv("historical_merged_data.csv")
target_df = pd.read_csv("historical_training_data.csv")
source = pd.read_csv("historical_featured_data.csv")
train = pd.read_csv("training_data.csv")
test = pd.read_csv("testing_data.csv")

datasets = [raw, target_df, source, train, test]

for df in datasets:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

target = "target_aqi_24h"

# ==========================================
# 2. BASIC DATA VALIDATION
# ==========================================

for name, df in [
    ("Raw", raw),
    ("Target", target_df),
    ("Featured", source),
    ("Training", train),
    ("Testing", test)
]:
    assert df.isnull().sum().sum() == 0, \
        f"Missing values in {name} dataset!"

    assert not df["timestamp"].duplicated().any(), \
        f"Duplicate timestamps in {name} dataset!"

    assert df["timestamp"].is_monotonic_increasing, \
        f"Unsorted timestamps in {name} dataset!"

print("Basic dataset validation: PASSED")

# ==========================================
# 3. VERIFY 24-HOUR TARGET ALIGNMENT
# ==========================================

# Create a timestamp-indexed original AQI series
raw_aqi = raw.set_index("timestamp")["us_aqi"]

# Find the AQI exactly 24 hours after each timestamp
future_timestamps = source["timestamp"] + pd.Timedelta(hours=24)

expected_target = raw_aqi.reindex(future_timestamps)

# Every featured record must have a valid future AQI
assert expected_target.notna().all(), \
    "Future AQI observations are missing!"

# Compare actual labels with original future AQI
actual_target = source[target].to_numpy()
expected_values = expected_target.to_numpy()

assert np.array_equal(
    actual_target,
    expected_values
), "24-hour target alignment FAILED!"

print("24-hour target alignment: VERIFIED")
print("Validated target labels:", len(source))

# ==========================================
# 4. VERIFY FEATURE / TARGET SEPARATION
# ==========================================

features = [
    col for col in source.columns
    if col != "timestamp" and not col.startswith("target_")
]

assert target not in features
assert len(features) == 28

assert not any(
    col.startswith("target_") for col in features
)

print("Input features:", len(features))
print("Target column:", target)
print("Feature/target separation: VERIFIED")

# ==========================================
# 5. VERIFY CHRONOLOGICAL SPLIT
# ==========================================

assert train["timestamp"].max() < test["timestamp"].min()

gap = (
    test["timestamp"].min() -
    train["timestamp"].max()
)

assert gap == pd.Timedelta(hours=25)

assert len(train) == 20865
assert len(test) == 5223

print("Training records:", len(train))
print("Testing records:", len(test))
print("Purged records: 24")
print("Timestamp gap:", gap)
print("Chronological split: VERIFIED")

# ==========================================
# 6. VERIFY DATASET SCHEMAS
# ==========================================

assert list(train.columns) == list(source.columns)
assert list(test.columns) == list(source.columns)

print("Dataset schemas: VERIFIED")

# ==========================================
# 7. FINAL REPORT
# ==========================================

print("\n========== FINAL DATASET VALIDATION ==========")

print("Raw records:", len(raw))
print("Target-engineered records:", len(target_df))
print("Feature-engineered records:", len(source))
print("Training records:", len(train))
print("Testing records:", len(test))

print("\nMissing values: 0")
print("Duplicate timestamps: 0")

print("\n24-hour target alignment: VERIFIED")
print("Feature/target separation: VERIFIED")
print("Chronological ordering: VERIFIED")
print("24-hour purge gap: VERIFIED")
print("Dataset schemas: VERIFIED")

print("\nALL VALIDATIONS PASSED!")
