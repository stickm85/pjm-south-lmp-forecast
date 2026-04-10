"""Gas price API client for Transco Zone 5, Transco Zone 6 NNY, Henry Hub, and regional gas prices."""

import pandas as pd
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class GasClient:
    """Client for gas price data (Morningstar Commodities).

    Fetches Transco Zone 5 (primary), Zone 6 NNY (secondary feature),
    Henry Hub, Columbia Gas, Dominion South, TETCO M3, and forward curves.
    Falls back to mock data if no API key is configured.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.api_key = cfg.get("gas", {}).get("api_key", "")
        self._mock = MockFallback()

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fetch_transco_z5(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Transco Zone 5 gas price ($/MMBtu).

        This is the PRIMARY gas price input for PJM SOUTH forecasting.
        Zone 5 covers Virginia/North Carolina where SOUTH-zone generators
        take delivery.

        Returns DataFrame with columns: date, gas_price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data")
            return self._mock.generate_gas_price(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_transco_z6_nny(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Transco Zone 6 NNY gas price ($/MMBtu).

        Currently unused — available if Zone 6 spread feature is added later.

        Returns DataFrame with columns: date, gas_price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data")
            return self._mock.generate_gas_price(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_henry_hub(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Henry Hub futures price ($/MMBtu).

        # Deprecated: use EIAClient.fetch_henry_hub_spot() instead (free, no auth)

        Returns DataFrame with columns: date, henry_hub_price
        """
        if not self._has_api_key():
            return self._mock.generate_henry_hub(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_columbia_gas(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Columbia Gas (TCO) daily spot price ($/MMBtu).

        Columbia Gas is a secondary gas delivery point for some Dominion
        generators in the SOUTH zone. Adding this captures a second gas
        delivery cost signal relevant to SOUTH LMP.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data for Columbia Gas")
            return self._mock.generate_columbia_gas(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_whub_forward(self, start_date, end_date) -> pd.DataFrame:
        """Fetch PJM Western Hub prompt-month forward curve price ($/MWh).

        The premium of the prompt-month forward over spot DA signals
        market expectations of near-term power price tightness.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data for WHub forward")
            return self._mock.generate_whub_forward(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_z5_gas_forward(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Transco Zone 5 prompt-month gas forward price ($/MMBtu).

        The spread between the prompt-month forward and the spot price
        (gas contango/backwardation) signals supply/demand expectations.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data for Z5 forward")
            return self._mock.generate_z5_gas_forward(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_dominion_south(self, start_date, end_date) -> pd.DataFrame:
        """Fetch Dominion South Point daily gas spot price ($/MMBtu).

        Dominion South is the Appalachian production-area index.
        Some generators in western Virginia and the Allegheny region
        take gas delivery here. It typically trades at a discount to
        Transco Z5 due to pipeline basis and Appalachian production surplus.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data for Dominion South")
            return self._mock.generate_dominion_south(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")

    def fetch_tetco_m3(self, start_date, end_date) -> pd.DataFrame:
        """Fetch TETCO M3 daily gas spot price ($/MMBtu).

        Texas Eastern M3 serves the Philadelphia/New Jersey corridor.
        TETCO M3 vs Transco Z5 spread widens during cold weather
        when pipeline constraints create regional pricing divergence.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No gas API key configured — using mock data for TETCO M3")
            return self._mock.generate_tetco_m3(start_date, end_date)
        raise NotImplementedError("Set gas.api_key in config/settings.yaml")


from .mock_data import MockDataGenerator as MockFallback
