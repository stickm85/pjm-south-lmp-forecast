"""ISO interface price client for MISO and NYISO border prices."""

import pandas as pd
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ISOClient:
    """Client for neighboring ISO interface prices."""

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        self._mock = MockFallback()

    def fetch_miso_prices(self, start_date, end_date) -> pd.DataFrame:
        """Fetch hourly MISO interface prices.

        Returns DataFrame with columns: datetime, miso_price
        """
        logger.debug("Fetching MISO prices (using mock)")
        df = self._mock.generate_iso_prices(start_date, end_date)
        return df[["datetime", "miso_price"]]

    def fetch_nyiso_prices(self, start_date, end_date) -> pd.DataFrame:
        """Fetch hourly NYISO interface prices.

        Returns DataFrame with columns: datetime, nyiso_price
        """
        logger.debug("Fetching NYISO prices (using mock)")
        df = self._mock.generate_iso_prices(start_date, end_date)
        return df[["datetime", "nyiso_price"]]

    def fetch_iso_prices(self, start_date, end_date) -> pd.DataFrame:
        """Fetch all ISO interface prices combined."""
        return self._mock.generate_iso_prices(start_date, end_date)


from .mock_data import MockDataGenerator as MockFallback
