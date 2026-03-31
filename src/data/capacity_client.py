"""Installed capacity client for solar, wind, and BESS capacity data."""

import pandas as pd
from typing import Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CapacityClient:
    """Client for installed generation capacity data."""

    def __init__(self, config_path: Union[str, Path] = None):
        self._mock = MockFallback()

    def fetch_installed_solar(self, start_date, end_date) -> pd.DataFrame:
        """Fetch monthly installed solar capacity (MW).

        Returns DataFrame with columns: date, solar_capacity_mw
        """
        logger.debug("Fetching installed solar capacity (using mock)")
        df = self._mock.generate_installed_capacity(start_date, end_date)
        return df[["date", "solar_capacity_mw"]]

    def fetch_installed_wind(self, start_date, end_date) -> pd.DataFrame:
        """Fetch monthly installed wind capacity (MW).

        Returns DataFrame with columns: date, wind_capacity_mw
        """
        logger.debug("Fetching installed wind capacity (using mock)")
        df = self._mock.generate_installed_capacity(start_date, end_date)
        return df[["date", "wind_capacity_mw"]]

    def fetch_installed_bess(self, start_date, end_date) -> pd.DataFrame:
        """Fetch quarterly BESS capacity (MW).

        Returns DataFrame with columns: date, bess_capacity_mw
        """
        logger.debug("Fetching BESS capacity (using mock)")
        return self._mock.generate_bess_capacity(start_date, end_date)


from .mock_data import MockDataGenerator as MockFallback
