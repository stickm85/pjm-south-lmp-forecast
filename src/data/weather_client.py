"""Legacy weather data client — temperature, humidity, wind/cloud. See openmeteo_client.py for primary weather source (Open-Meteo API)."""

import pandas as pd
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class WeatherClient:
    """Legacy client for temperature, humidity, and wind/cloud data.

    Note: The primary weather data source is now Open-Meteo (see openmeteo_client.py).
    This client provides temperature_f, dew_point, humidity, wind_speed, and cloud_cover
    using mock data fallback.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.noaa_base = cfg.get("weather", {}).get("noaa_api_base", "https://api.weather.gov")
        self._mock = MockFallback()

    def fetch_temperature(self, start_date, end_date, city: str) -> pd.DataFrame:
        """Fetch hourly temperature forecasts for the given city.

        Returns DataFrame with columns: datetime, temperature_f, city
        """
        try:
            logger.debug(f"Fetching temperature for {city} (using mock fallback)")
            return self._mock.generate_weather_forecast(start_date, end_date, city)
        except Exception as e:
            logger.warning(f"Weather fetch failed for {city}: {e} — using mock")
            return self._mock.generate_weather_forecast(start_date, end_date, city)

    def fetch_humidity(self, start_date, end_date, city: str) -> pd.DataFrame:
        """Fetch hourly dew point and relative humidity for the given city.

        Returns DataFrame with columns: datetime, dew_point_f, rh_pct, city
        """
        try:
            return self._mock.generate_humidity_forecast(start_date, end_date, city)
        except Exception as e:
            logger.warning(f"Humidity fetch failed for {city}: {e} — using mock")
            return self._mock.generate_humidity_forecast(start_date, end_date, city)

    def fetch_wind_cloud(self, start_date, end_date, city: str) -> pd.DataFrame:
        """Fetch hourly wind speed and cloud cover for the given city.

        Returns DataFrame with columns: datetime, wind_speed_mph, cloud_cover_pct, city
        """
        return self._mock.generate_wind_cloud_forecast(start_date, end_date, city)


from .mock_data import MockDataGenerator as MockFallback
