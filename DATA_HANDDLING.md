# AgentWeather - Data Handling Documentation

## 1. Data Sources

The project currently uses three data sources:

- **OpenWeather API:** Weather forecast data, including temperature, humidity, pressure, wind speed, wind direction, precipitation, rain probability, and weather description.
- **Open-Meteo Air Quality API:** Hourly air quality data, including US AQI, PM2.5, and PM10.
- **Open-Meteo Forecast API:** Hourly UV Index data.

All APIs use Karachi coordinates:

- Latitude: 24.8607
- Longitude: 67.0011
- Local timezone: Asia/Karachi

The current pipeline collects forecast data, not historical weather observations.

## 2. Data Frequency

- AQI data: Hourly records.
- UV Index data: Hourly records.
- Weather data: Every 3 hours.
- Weather data is aligned with the hourly AQI and UV timestamps.

The weather dataset is interpolated to an hourly frequency during the merging process.

## 3. Common Time Window

The merged dataset uses timestamps within the overlapping period available across all three datasets.

For the current dataset:

- Start: 2026-11-21 17:00 (Asia/Karachi)
- End: 2026-11-25 23:00 (Asia/Karachi)
- Total records: 103 hourly records.

This ensures that the merged dataset contains aligned records from all three sources.

## 4. Missing Value Handling

### AQI Data

- AQI values are not interpolated.
- Original hourly AQI observations from Open-Meteo are retained.

### UV Index

- UV Index values are not interpolated.
- Original hourly UV Index forecast values from Open-Meteo are retained.

### Numerical Weather Features

The following features use time-based linear interpolation:

- Temperature
- Humidity
- Pressure
- Wind Speed
- Rain Probability

Interpolation is limited to two consecutive missing rows and is only applied between available observations.

### Wind Direction

Wind direction is circular data.

Circular interpolation uses sine and cosine transformations to avoid incorrect interpolation around 0° and 360°.

### Precipitation

Precipitation is obtained from OpenWeather's 3-hour accumulated rainfall field.

Forward filling is limited to two consecutive missing rows.

The values retain their original 3-hour accumulation meaning and are not treated as independent hourly rainfall measurements.

### Weather Description

Missing weather descriptions are forward-filled, with a maximum limit of two consecutive rows.

## 5. Feature Engineering

The merged dataset contains the following engineered time and statistical features:

### Time Features

- Hour
- Day
- Month

### Rate of Change Features

- AQI rate of change
- PM2.5 rate of change
- Temperature rate of change

### Rolling Mean Features

Three-hour rolling averages are calculated for:

- AQI
- PM2.5
- Temperature

The first record of each rate-of-change feature and the first two records of each three-hour rolling mean feature contain NaN values because sufficient previous observations are not available.

These mathematically undefined values are not replaced with zero.

## 6. Final Dataset Structure

The merged dataset contains **22 columns**:

| Category | Features |
|---|---|
| AQI | us_aqi, pm2_5, pm10 |
| Weather | temperature, humidity, pressure, wind_speed, weather |
| Precipitation | precipitation, rain_probability |
| Wind | wind_direction |
| UV | uv_index |
| Time | hour, day, month |
| Rate of Change | aqi_rate_change, pm25_rate_change, temperature_rate_change |
| Rolling Means | aqi_rolling_mean_3h, pm25_rolling_mean_3h, temperature_rolling_mean_3h |
| Timestamp | timestamp |

Total: 22 columns.

## 7. Data Validation

The following validation checks are performed:

- Duplicate timestamps are checked in the source datasets.
- Original numerical weather observations are compared against merged values.
- Original weather descriptions are compared.
- Missing values are checked after merging.
- Duplicate timestamps in the final dataset are checked.
- UV Index values are checked for missing values and reasonable ranges.
- Engineered feature NaN values are checked.

## 8. Current Validation Results

The latest pipeline execution produced:

- 120 AQI records fetched.
- 40 weather records fetched.
- 120 UV Index records fetched.
- 103 hourly records successfully merged.
- 22 columns in the merged dataset.
- Zero duplicate timestamps.
- Zero missing values in the original source features after merging.
- Expected NaN values in the initial rate-of-change and rolling mean calculations.
- Wind direction circular interpolation implemented.
- Original 3-hour precipitation accumulation meaning preserved.

## 9. Hopsworks Feature Store

The raw datasets and engineered features are stored in Hopsworks.

The following Feature Groups have been successfully created and verified:

| Feature Group | Version | Records |
|---|---:|---:|
| raw_aqi | 2 | 120 |
| raw_weather | 2 | 40 |
| raw_uv | 1 | 120 |
| weather_features | 2 | 103 |

All four Feature Groups were successfully read back from Hopsworks.

Timestamps are converted from Asia/Karachi local time to UTC before uploading to Hopsworks.

The stored timestamps were verified as UTC-aware during readback.

## 10. Current Pipeline Status

**Completed:**

- AQI data fetching.
- Weather data fetching.
- UV Index integration.
- Dataset merging and time alignment.
- Missing value handling.
- Feature engineering.
- Data validation.
- Raw data storage in Hopsworks.
- Engineered feature storage in Hopsworks.
- Hopsworks readback verification.

**Pending:**

- Historical data backfill.
- Creation of 24-hour-ahead prediction targets.
- Model training and evaluation.
- Agent development and integration.
- Dashboard and final project deliverables.

## 11. Important Note

The current dataset contains forecast data collected from the APIs. It must not be treated as historical observed data for model training.

Historical observations will be collected separately during Phase 2 using an appropriate historical data source.

The UV Index source is Open-Meteo, as decided during supervisor discussion. This source will also be documented in the final project report.