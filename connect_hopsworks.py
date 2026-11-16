from dotenv import load_dotenv
import hopsworks

# Load environment variables
load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(engine="python")

# Access Feature Store
fs = project.get_feature_store()

print("Successfully connected to Hopsworks!")
print("Feature Store:", fs.name)