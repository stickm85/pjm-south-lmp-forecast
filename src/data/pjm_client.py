"""PJM DataMiner2 API client for fetching LMP, load, and generation data."""

import requests
import pandas as pd
from datetime import datetime, date
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class PJMClient:
    """Client for PJM DataMiner2 API.

    Fetches LMP, load forecasts, metered load, and generation data.
    Falls back to mock data when API key is not configured.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.api_base = cfg["pjm"]["api_base_url"]
        self.api_key = cfg["pjm"]["api_key"]
        self.node = cfg["pjm"]["node"]
        self._mock = MockFallback()

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fetch_da_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Day-Ahead LMP for SOUTH node with energy/congestion/loss components."""
        if not self._has_api_key():
            logger.warning("No PJM API key configured — using mock data")
            return self._mock.generate_pjm_da_lmp(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_whub_da_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Western Hub Day-Ahead LMP."""
        if not self._has_api_key():
            return self._mock.generate_whub_da_lmp(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_rt_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Real-Time LMP for SOUTH node."""
        if not self._has_api_key():
            return self._mock.generate_rt_lmp(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_load_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Fetch D+1 zonal load forecast (SOUTH + PJM total)."""
        if not self._has_api_key():
            return self._mock.generate_load_forecast(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_metered_load(self, start_date, end_date) -> pd.DataFrame:
        """Fetch actual metered load (SOUTH zone)."""
        if not self._has_api_key():
            return self._mock.generate_metered_load(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_solar_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Fetch PJM D+1 solar generation forecast."""
        if not self._has_api_key():
            return self._mock.generate_solar_forecast(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_wind_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Fetch PJM D+1 wind generation forecast."""
        if not self._has_api_key():
            return self._mock.generate_wind_forecast(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_solar_actuals(self, start_date, end_date) -> pd.DataFrame:
        """Fetch actual solar generation."""
        if not self._has_api_key():
            return self._mock.generate_solar_actuals(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_wind_actuals(self, start_date, end_date) -> pd.DataFrame:
        """Fetch actual wind generation."""
        if not self._has_api_key():
            return self._mock.generate_wind_actuals(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_interchange(self, start_date, end_date) -> pd.DataFrame:
        """Fetch PJM net scheduled interchange flows."""
        if not self._has_api_key():
            return self._mock.generate_interchange(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_fuel_mix(self, start_date, end_date) -> pd.DataFrame:
        """Fetch generation fuel mix."""
        if not self._has_api_key():
            return self._mock.generate_fuel_mix(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_virtual_bids(self, start_date, end_date) -> pd.DataFrame:
        """Fetch virtual bid volume at SOUTH node."""
        if not self._has_api_key():
            return self._mock.generate_virtual_bids(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")


from .mock_data import MockDataGenerator as MockFallback
