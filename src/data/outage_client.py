"""Outage data client for transmission and generator outages."""

import pandas as pd
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class OutageClient:
    """Client for transmission and generator outage data."""

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        self._mock = MockFallback()

    def fetch_transmission_outages(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily transmission outage flags.

        Returns DataFrame with columns: date, transmission_outage_flag
        """
        logger.debug("Fetching transmission outages (using mock)")
        return self._mock.generate_transmission_outages(start_date, end_date)

    def fetch_generator_outages(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily generator outages in MW offline.

        Returns DataFrame with columns: date, mw_offline
        """
        logger.debug("Fetching generator outages (using mock)")
        return self._mock.generate_generator_outages(start_date, end_date)


from .mock_data import MockDataGenerator as MockFallback
