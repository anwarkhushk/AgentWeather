
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(engine="python")
fs = project.get_feature_store()

groups = [
    ("raw_aqi", 2, 120),
    ("raw_weather", 2, 40),
    ("raw_uv", 1, 120),
    ("weather_features", 2, 103)
]

for name, version, expected_records in groups:

    print("\n" + "=" * 50)
    print(f"Feature Group: {name} v{version}")

    fg = fs.get_feature_group(
        name=name,
        version=version
    )

    df = fg.read()

    print("Records:", len(df))
    print("Expected:", expected_records)
    print("Columns:", df.columns.tolist())

    print("\nFirst 3 timestamps:")
    print(df["timestamp"].head(3))

    print("\nTimezone:", df["timestamp"].dtype)

    assert len(df) == expected_records, (
        f"Record count mismatch in {name}"
    )

    print("\nVerification successful!")

print("\nAll Feature Groups verified!")
