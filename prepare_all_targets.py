import pandas as pd
import numpy as np

weather = pd.read_csv("historical_merged_data.csv")
aux = pd.read_csv("historical_auxiliary_data.csv")

weather["timestamp"] = pd.to_datetime(weather["timestamp"])
aux["timestamp"] = pd.to_datetime(aux["timestamp"])

df = weather.merge(
    aux[["timestamp", "uv_index"]],
    on="timestamp",
    how="left",
    validate="one_to_one"
)

df = df.sort_values("timestamp").reset_index(drop=True)

# Circular wind direction representation
radians = np.deg2rad(df["wind_direction_10m"])
df["wind_direction_sin"] = np.sin(radians)
df["wind_direction_cos"] = np.cos(radians)

# 24-hour-ahead targets
targets = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "precipitation": "precipitation",
    "uv": "uv_index",
    "wind_speed": "wind_speed_10m",
    "wind_direction_sin": "wind_direction_sin",
    "wind_direction_cos": "wind_direction_cos"
}

for name, column in targets.items():
    df[f"target_{name}_24h"] = df[column].shift(-24)

# Rain occurrence at the target hour
df["target_rain_probability_24h"] = (
    df["precipitation"].shift(-24) > 0.1
).astype(float)

df.loc[df["precipitation"].shift(-24).isna(),
       "target_rain_probability_24h"] = np.nan

# Keep timestamps and features; remove final 24 rows without future targets
df = df.iloc[:-24].copy()

# Exclude rows with missing UV (should be zero, but validate)
required_targets = [f"target_{x}_24h" for x in targets]
required_targets.append("target_rain_probability_24h")
df = df.dropna(subset=required_targets)

# Reuse exact existing Phase 2 split timestamps
train_old = pd.read_csv("training_data.csv", usecols=["timestamp"])
test_old = pd.read_csv("testing_data.csv", usecols=["timestamp"])

train_times = set(pd.to_datetime(train_old["timestamp"]))
test_times = set(pd.to_datetime(test_old["timestamp"]))

train = df[df["timestamp"].isin(train_times)].copy()
test = df[df["timestamp"].isin(test_times)].copy()

train.to_csv("multi_metric_training_data.csv", index=False)
test.to_csv("multi_metric_testing_data.csv", index=False)

print("TRAIN:", train.shape)
print("TEST:", test.shape)
print("\nTraining date range:")
print(train.timestamp.min(), "to", train.timestamp.max())
print("\nTesting date range:")
print(test.timestamp.min(), "to", test.timestamp.max())

print("\nTarget missing values:")
print(df[required_targets].isna().sum())

print("\nRain occurrence distribution:")
print(train["target_rain_probability_24h"].value_counts(normalize=True))

print("\nUV missing values:", df["uv_index"].isna().sum())
print("\nSaved multi_metric_training_data.csv")
print("Saved multi_metric_testing_data.csv")
