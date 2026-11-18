
import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Karachi location
CITY = "Karachi"
COUNTRY = "PK"

# OpenWeather 5-day / 3-hour forecast API
URL = "https://api.openweathermap.org/data/2.5/forecast"

params = {
    "q": f"{CITY},{COUNTRY}",
    "appid": API_KEY,
    "units": "metric"
}

# Fetch weather data
response = requests.get(URL, params=params, timeout=30)

print("Status Code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    # Display location information
    print("\nLocation Information:")
    print("City:", data["city"]["name"])
    print("Country:", data["city"]["country"])
    print("Coordinates:", data["city"]["coord"])

    weather_records = []

    for item in data["list"]:

        record = {
            "timestamp": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind_speed": item["wind"]["speed"],
            "weather": item["weather"][0]["description"]
        }

        weather_records.append(record)

    print("\nTotal records:", len(weather_records))

    print("\nFirst 5 records:")

    for record in weather_records[:5]:
        print(record)

    # Convert records into Pandas DataFrame
    df = pd.DataFrame(weather_records)

    # Save DataFrame into CSV
    df.to_csv("weather_data.csv", index=False)

    print("\nWeather data saved successfully!")
    print("File: weather_data.csv")

    print("\nDataFrame Preview:")
    print(df.head())

else:
    print("Error:", response.text)
