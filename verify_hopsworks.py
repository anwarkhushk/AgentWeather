
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(engine="python")
fs = project.get_feature_store()

# Feature Groups to verify
groups = [
    ("raw_aqi", 1),
    ("raw_weather", 1),
    ("weather_features", 1)
]

for name, version in groups:

    print(f"\nChecking: {name}")

    fg = fs.get_feature_group(
        name=name,
        version=version
    )

    df = fg.read()

    print("Records:", len(df))
    print("Columns:", df.columns.tolist())
    print("First 3 records:")
    print(df.head(3))

print("\nFeature Store verification completed!")
