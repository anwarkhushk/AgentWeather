
import pandas as pd

# 1. Load feature-engineered dataset
df = pd.read_csv("historical_featured_data.csv")

# 2. Sort chronologically
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Define target
target = "target_aqi_24h"

# 4. Validate dataset
assert target in df.columns, "Target column missing!"
assert df.isnull().sum().sum() == 0, "Missing values detected!"
assert df["timestamp"].is_monotonic_increasing
assert not df["timestamp"].duplicated().any()

# 5. Separate features and target
feature_columns = [
    col for col in df.columns
    if col != "timestamp" and not col.startswith("target_")
]

X = df[feature_columns]
y = df[target]

# 6. Chronological split with 24-hour purge gap
split_index = int(len(df) * 0.80)

forecast_horizon = 24
train_end = split_index - forecast_horizon

X_train = X.iloc[:train_end]
X_test = X.iloc[split_index:]

y_train = y.iloc[:train_end]
y_test = y.iloc[split_index:]

# 7. Save datasets with timestamps
train_df = df.iloc[:train_end].copy()
test_df = df.iloc[split_index:].copy()

train_df.to_csv("training_data.csv", index=False)
test_df.to_csv("testing_data.csv", index=False)

# 8. Display results
print("\n========== CHRONOLOGICAL DATA SPLIT ==========")

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Purged records:", split_index - train_end)

print("\nTraining period:")
print(train_df["timestamp"].min(), "to",
      train_df["timestamp"].max())

print("\nTesting period:")
print(test_df["timestamp"].min(), "to",
      test_df["timestamp"].max())

print("\nFeature columns:", len(feature_columns))
print("Target column:", target)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# 9. Verify 24-hour target separation
assert len(train_df) == len(X_train)
assert len(test_df) == len(X_test)

assert (
    train_df["timestamp"].max()
    + pd.Timedelta(hours=24)
    < test_df["timestamp"].min()
)

print("\n24-hour purge gap verified!")
print("Chronological split verified!")
print("Saved: training_data.csv")
print("Saved: testing_data.csv")
