import os
from dotenv import load_dotenv

load_dotenv()

for name in ["HOPSWORKS_API_KEY", "OPENWEATHER_API_KEY", "AQICN_TOKEN"]:
    value = os.getenv(name)
    print(name, "->", "OK" if value else "MISSING")