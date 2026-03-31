"""Gas price API client for Transco Zone 6 NNY and Henry Hub prices."""

import pandas as pd
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class GasClient:
    """Client for gas price data (ICE, Platts, NGI).

    Falls back to mock data if no API key is configured.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.api_key = cfg.get("gas", {}).get("api_key", "")
        self.source = cfg.get("gas", {}).get("source", "ice")
        self._mock = MockFallback()

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fetch_transco_z6_nny(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Transco Zone 6 NNY gas price ($/MMBtu).

        Returns DataFrame with columns: date, gas_price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data")
            return self._mock.generate_gas_price(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_henry_hub(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Henry Hub futures price ($/MMBtu).

        Returns DataFrame with columns: date, henry_hub_price
        """
        if not self._has_api_key():
            return self._mock.generate_henry_hub(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")


from .mock_data import MockDataGenerator as MockFallback
