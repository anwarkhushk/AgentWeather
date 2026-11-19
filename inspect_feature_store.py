
import hopsworks
from dotenv import load_dotenv

# Load credentials
load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(engine="python")
fs = project.get_feature_store()

# Retrieve existing Feature Groups
feature_groups = fs.get_feature_groups()

print("\nExisting Feature Groups:")
print("------------------------")

if not feature_groups:
    print("No Feature Groups found.")

else:
    for fg in feature_groups:
        print("Name:", fg.name)
        print("Version:", fg.version)
        print("------------------------")
