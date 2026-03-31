"""Market data client for RGGI, coal, emergency events, and demand response."""

import pandas as pd
from typing import Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketClient:
    """Client for market/environmental/compliance data."""

    def __init__(self, config_path: Union[str, Path] = None):
        self._mock = MockFallback()

    def fetch_rggi_price(self, start_date, end_date) -> pd.DataFrame:
        """Fetch quarterly RGGI allowance price ($/ton CO2).

        Returns DataFrame with columns: date, rggi_price
        """
        logger.debug("Fetching RGGI price (using mock)")
        return self._mock.generate_rggi_price(start_date, end_date)

    def fetch_coal_price(self, start_date, end_date) -> pd.DataFrame:
        """Fetch weekly coal spot price ($/short ton).

        Returns DataFrame with columns: date, coal_price_per_ton
        """
        logger.debug("Fetching coal price (using mock)")
        return self._mock.generate_coal_price(start_date, end_date)

    def fetch_emergency_logs(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Emergency Event Alert (EEA) flags.

        Returns DataFrame with columns: date, eea_flag
        """
        logger.debug("Fetching emergency logs (using mock)")
        return self._mock.generate_emergency_logs(start_date, end_date)

    def fetch_demand_response(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily demand response event flags and MW.

        Returns DataFrame with columns: date, dr_event_flag, dr_mw
        """
        logger.debug("Fetching demand response (using mock)")
        return self._mock.generate_demand_response(start_date, end_date)


from .mock_data import MockDataGenerator as MockFallback
