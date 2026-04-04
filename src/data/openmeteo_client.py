"""Open-Meteo API client for enhanced weather variables across 5 cities."""

import requests
import pandas as pd
import numpy as np
from typing import List, Union, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Cities relevant to PJM SOUTH and WHub
CITIES = {
    "Richmond VA":    {"lat": 37.54, "lon": -77.44},
    "Norfolk VA":     {"lat": 36.85, "lon": -76.29},
    "Raleigh NC":     {"lat": 35.78, "lon": -78.64},
    "Pittsburgh PA":  {"lat": 40.44, "lon": -79.99},
    "Columbus OH":    {"lat": 39.96, "lon": -82.99},
}

HOURLY_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "apparent_temperature",
    "windgusts_10m",
    "precipitation",
    "pressure_msl",
]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    """Client for Open-Meteo free weather API (no authentication required).

    Fetches 6 enhanced weather variables for all 5 cities:
      - shortwave_radiation  (W/m²) — solar GHI
      - direct_radiation     (W/m²) — direct-beam radiation
      - apparent_temperature (°C)   — 'feels like' temperature
      - windgusts_10m        (m/s)  — wind gusts at 10m
      - precipitation        (mm)   — hourly precipitation
      - pressure_msl         (hPa)  — mean sea-level pressure

    Falls back to mock data if the API request fails.
    """

    def __init__(self, cities: Dict[str, Dict] = None):
        self.cities = cities or CITIES
        self._mock = OpenMeteoMockData()

    def fetch_forecast(
        self,
        target_date: Union[str, pd.Timestamp],
        cities: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch Open-Meteo forecast data for target_date across cities.

        Returns a DataFrame with columns:
            datetime, city, shortwave_radiation, direct_radiation,
            apparent_temperature, windgusts_10m, precipitation, pressure_msl
        """
        cities = cities or list(self.cities.keys())
        target_date = pd.Timestamp(target_date)
        frames = []

        for city in cities:
            coords = self.cities.get(city)
            if coords is None:
                logger.warning(f"Unknown city: {city}, skipping")
                continue
            try:
                params = {
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "hourly": ",".join(HOURLY_VARS),
                    "timezone": "America/New_York",
                    "start_date": target_date.strftime("%Y-%m-%d"),
                    "end_date": target_date.strftime("%Y-%m-%d"),
                }
                resp = requests.get(FORECAST_URL, params=params, timeout=10)
                resp.raise_for_status()
                df = self._parse_response(resp.json(), city)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Open-Meteo forecast failed for {city}: {e} — using mock data")
                frames.append(self._mock.generate(target_date, target_date, city))

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_historical(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        cities: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch Open-Meteo historical archive data across cities.

        Returns a DataFrame with columns:
            datetime, city, shortwave_radiation, direct_radiation,
            apparent_temperature, windgusts_10m, precipitation, pressure_msl
        """
        cities = cities or list(self.cities.keys())
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        frames = []

        for city in cities:
            coords = self.cities.get(city)
            if coords is None:
                logger.warning(f"Unknown city: {city}, skipping")
                continue
            try:
                params = {
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "hourly": ",".join(HOURLY_VARS),
                    "timezone": "America/New_York",
                }
                resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
                resp.raise_for_status()
                df = self._parse_response(resp.json(), city)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Open-Meteo historical failed for {city}: {e} — using mock data")
                frames.append(self._mock.generate(start_date, end_date, city))

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _parse_response(self, data: dict, city: str) -> pd.DataFrame:
        """Parse Open-Meteo JSON response into a flat DataFrame."""
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        df = pd.DataFrame({"datetime": times})
        for var in HOURLY_VARS:
            df[var] = hourly.get(var, np.nan)
        df["city"] = city
        return df


class OpenMeteoMockData:
    """Mock data generator for Open-Meteo variables with realistic ranges.

    Ranges:
      - shortwave_radiation:  0–1000 W/m² (solar GHI, zero at night)
      - direct_radiation:     0–800  W/m² (subset of GHI)
      - apparent_temperature: -20–42 °C   (seasonal with diurnal variation)
      - windgusts_10m:        0–40   m/s
      - precipitation:        0–10   mm/hr
      - pressure_msl:         990–1040 hPa
    """

    RNG_SEED = 42

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed or self.RNG_SEED)

    def generate(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        city: str = "Richmond VA",
    ) -> pd.DataFrame:
        """Generate realistic hourly mock data for a single city."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        n = len(idx)

        # Seasonal base temperature (°C), peaks in summer
        doy = idx.dayofyear
        seasonal_c = 15.0 + 15.0 * np.sin(2 * np.pi * (doy - 80) / 365)
        diurnal_c = 5.0 * np.sin(np.pi * (idx.hour - 6) / 12)
        apparent_temp = seasonal_c + diurnal_c + self.rng.normal(0, 2, n)

        # Solar radiation — only during daytime (hours 6–20)
        solar_shape = np.maximum(0.0, np.sin(np.pi * (idx.hour - 6) / 14))
        seasonal_solar = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365)
        ghi = np.maximum(0.0, 900.0 * solar_shape * seasonal_solar + self.rng.normal(0, 50, n))
        direct = np.minimum(ghi, np.maximum(0.0, ghi * 0.8 + self.rng.normal(0, 30, n)))

        # Wind gusts
        wind_gusts = np.maximum(0.0, self.rng.gamma(shape=2, scale=4, size=n))

        # Precipitation (sparse — most hours zero)
        precip_prob = 0.08
        precip_mask = self.rng.random(n) < precip_prob
        precipitation = np.where(precip_mask, np.abs(self.rng.exponential(1.5, n)), 0.0)

        # Pressure (slow-moving, mean-reverting around 1013 hPa)
        pressure = 1013.0 + np.cumsum(self.rng.normal(0, 0.3, n))
        pressure = np.clip(pressure - (pressure - 1013.0) * 0.002, 990.0, 1040.0)

        return pd.DataFrame({
            "datetime": idx,
            "city": city,
            "shortwave_radiation": np.round(ghi, 2),
            "direct_radiation": np.round(direct, 2),
            "apparent_temperature": np.round(apparent_temp, 2),
            "windgusts_10m": np.round(wind_gusts, 2),
            "precipitation": np.round(precipitation, 3),
            "pressure_msl": np.round(pressure, 2),
        })

    def generate_all_cities(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        cities: List[str] = None,
    ) -> pd.DataFrame:
        """Generate mock data for all cities and return concatenated DataFrame."""
        cities = cities or list(CITIES.keys())
        frames = [self.generate(start_date, end_date, city) for city in cities]
        return pd.concat(frames, ignore_index=True)
