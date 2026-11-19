
# AgentWeather - Data Handling Documentation

## 1. Data Sources

The project uses two data sources:

- OpenWeather API: Weather forecast data.
- Open-Meteo Air Quality API: Air quality data.

Both datasets use Karachi coordinates:
- Latitude: 24.8607
- Longitude: 67.0011

All timestamps are aligned to Asia/Karachi local time.

## 2. Data Frequency

- AQI data: Hourly records.
- Weather data: Every 3 hours.
- Weather data is aligned with hourly AQI timestamps.

## 3. Common Time Window

The merged dataset uses only timestamps available within
the overlapping period of both datasets.

This prevents creating records outside the available
observation window.

## 4. Missing Value Handling

### AQI Data
- AQI values are not interpolated.
- Original hourly AQI values are retained.

### Numerical Weather Features
The following features use time-based linear interpolation:

- Temperature
- Humidity
- Pressure
- Wind Speed
- Rain Probability

Interpolation is limited to two consecutive missing rows
and only applied between available observations.

### Wind Direction
Wind direction is circular data.

Circular interpolation uses sine and cosine transformations
to avoid incorrect interpolation around 0° and 360°.

### Precipitation
Precipitation is a 3-hour accumulated rainfall measurement.

Forward filling is limited to two consecutive missing rows.
The values retain their original 3-hour accumulation meaning.

### Weather Description
Missing weather descriptions are forward-filled,
with a maximum limit of two consecutive rows.

## 5. Data Validation

The following checks are performed:

- Duplicate timestamps are checked.
- Original numerical weather observations are compared
  against merged values.
- Original weather descriptions are compared.
- Missing values are checked after merging.
- Duplicate timestamps in the final dataset are checked.

## 6. Validation Results

- 100 hourly records successfully merged.
- 12 columns in the merged dataset.
- Zero duplicate timestamps.
- Zero missing values in the final dataset.
- Original weather numerical observations preserved.
- Original weather descriptions preserved.
- Circular wind direction interpolation validated.

## 7. Pending Items

- UV Index integration.
- Further feature engineering.
- Raw and engineered feature storage in Hopsworks.
