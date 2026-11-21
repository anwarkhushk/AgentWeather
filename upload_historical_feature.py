
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Load historical featured dataset
df = pd.read_csv("historical_featured_data.csv")

print("Dataset loaded:", len(df), "records")

# 3. Validate dataset before upload
EXPECTED_RECORDS = 26112

if len(df) != EXPECTED_RECORDS:
    raise ValueError(
        f"Expected {EXPECTED_RECORDS} records, found {len(df)}"
    )

if df.isnull().sum().sum() != 0:
    raise ValueError("Dataset contains missing values!")

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found!")

# 4. Convert Karachi local timestamps to UTC
df["timestamp"] = pd.to_datetime(df["timestamp"])

if df["timestamp"].dt.tz is not None:
    raise ValueError("Expected timezone-naive Karachi timestamps!")

df["timestamp"] = (
    df["timestamp"]
    .dt.tz_localize("Asia/Karachi")
    .dt.tz_convert("UTC")
)

print("\nTimezone conversion completed!")
print("Timestamp timezone:", df["timestamp"].dt.tz)

# 5. Connect to Hopsworks
project = hopsworks.login(engine="python")
fs = project.get_feature_store()

print("\nConnected to Hopsworks!")
print("Feature Store:", fs.name)

# 6. Create or retrieve historical feature group
feature_group = fs.get_or_create_feature_group(
    name="historical_aqi_features",
    version=1,
    description=(
        "Historical AQI and weather features with "
        "24-hour future AQI prediction target"
    ),
    primary_key=["timestamp"],
    event_time="timestamp",
    online_enabled=False,
    time_travel_format="HUDI"
)

# 7. Check existing feature group metadata
if feature_group.id is not None:
    existing_df = feature_group.read()

    if len(existing_df) > 0:
        raise RuntimeError(
            "Feature Group already contains records. "
            "Upload stopped to prevent duplicates."
        )

# 8. Upload historical data
print("\nUploading", len(df), "records...")

feature_group.insert(
    df,
    write_options={"wait_for_job": True}
)

print("\nUpload completed!")

# 9. Verify uploaded records
uploaded_df = feature_group.read()

print("\n========== UPLOAD VERIFICATION ==========")
print("Expected records:", EXPECTED_RECORDS)
print("Uploaded records:", len(uploaded_df))

if len(uploaded_df) != EXPECTED_RECORDS:
    raise ValueError("Upload verification failed!")

print("\nHISTORICAL FEATURE GROUP VERIFIED!")
