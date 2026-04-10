"""Morningstar Commodities data client for gas and power forward curves."""

import pandas as pd
import numpy as np
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class MorningstarClient:
    """Client for Morningstar Commodities daily price data.

    Provides:
      - Columbia Gas (TCO) daily spot price
      - PJM Western Hub forward curve (prompt-month)
      - Transco Zone 5 gas forward curve (prompt-month)

    Falls back to mock data if no API key is configured.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.api_key = cfg.get("morningstar", {}).get("api_key", "")
        self.api_base = cfg.get("morningstar", {}).get("api_base_url",
                                                       "https://api.morningstar.com/v2/datasets")
        self._mock = MorningstarMockData()

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fetch_columbia_gas(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch Columbia Gas (TCO) daily spot price ($/MMBtu).

        Columbia Gas is a secondary gas delivery point for some Dominion
        generators in the SOUTH zone that do not take delivery on Transco.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No Morningstar API key configured — using mock data for Columbia Gas")
            return self._mock.generate_columbia_gas(start_date, end_date)
        raise NotImplementedError("Set morningstar.api_key in config/settings.yaml")

    def fetch_whub_forward(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch PJM Western Hub prompt-month forward price ($/MWh).

        The premium of the prompt-month forward over the spot DA price signals
        market expectations of near-term power price tightness.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No Morningstar API key configured — using mock data for WHub forward")
            return self._mock.generate_whub_forward(start_date, end_date)
        raise NotImplementedError("Set morningstar.api_key in config/settings.yaml")

    def fetch_z5_gas_forward(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch Transco Zone 5 prompt-month gas forward price ($/MMBtu).

        The spread between the prompt-month forward and the spot price
        indicates gas market contango (positive) or backwardation (negative),
        signaling supply/demand expectations.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No Morningstar API key configured — using mock data for Z5 forward")
            return self._mock.generate_z5_gas_forward(start_date, end_date)
        raise NotImplementedError("Set morningstar.api_key in config/settings.yaml")

    def fetch_dominion_south(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch Dominion South Point daily gas spot price ($/MMBtu).

        Dominion South is the Appalachian production-area index.
        Some generators in western Virginia and the Allegheny region
        take gas delivery here. It typically trades at a discount to
        Transco Z5 due to pipeline basis and Appalachian production surplus.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No Morningstar API key configured — using mock data for Dominion South")
            return self._mock.generate_dominion_south(start_date, end_date)
        raise NotImplementedError("Set morningstar.api_key in config/settings.yaml")

    def fetch_tetco_m3(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Fetch TETCO M3 daily gas spot price ($/MMBtu).

        Texas Eastern M3 serves the Philadelphia/New Jersey corridor.
        TETCO M3 vs Transco Z5 spread widens during cold weather
        when pipeline constraints create regional pricing divergence.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No Morningstar API key configured — using mock data for TETCO M3")
            return self._mock.generate_tetco_m3(start_date, end_date)
        raise NotImplementedError("Set morningstar.api_key in config/settings.yaml")


class MorningstarMockData:
    """Mock data generator for Morningstar price datasets.

    Realistic ranges:
      - Columbia Gas:   $2–8 /MMBtu  (tracks Transco Z5 with ±$0.50 spread)
      - WHub forward:   $25–80 /MWh  (prompt-month power forward)
      - Z5 gas forward: $2–8 /MMBtu  (prompt-month gas forward)
    """

    RNG_SEED = 42

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed or self.RNG_SEED)

    def _daily_index(self, start_date, end_date) -> pd.DatetimeIndex:
        return pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")

    def _gas_base(self, idx: pd.DatetimeIndex, base: float = 3.5) -> np.ndarray:
        """Seasonal gas price base (higher in winter)."""
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        noise = self.rng.normal(0, 0.2, len(idx))
        return np.maximum(1.5, base + seasonal + noise)

    def generate_columbia_gas(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate Columbia Gas (TCO) daily spot price ($/MMBtu).

        Tracks Transco Z5 with a ±$0.50 regional basis spread.
        """
        idx = self._daily_index(start_date, end_date)
        z5_base = self._gas_base(idx, base=3.5)
        spread = self.rng.uniform(-0.5, 0.5, len(idx))
        price = np.maximum(2.0, z5_base + spread)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_whub_forward(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate PJM WHub prompt-month forward price ($/MWh).

        Seasonal pattern with winter and summer peaks, realistic $25–80 range.
        """
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        # Seasonal: peaks in winter (Jan) and summer (Aug)
        summer_peak = 15.0 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        winter_peak = 8.0 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        base = 45.0
        noise = self.rng.normal(0, 3.0, n)
        price = np.clip(base + summer_peak + winter_peak + noise, 25.0, 80.0)
        return pd.DataFrame({"date": idx, "price": np.round(price, 2)})

    def generate_z5_gas_forward(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate Transco Z5 prompt-month gas forward price ($/MMBtu).

        Closely tracks spot with a small contango/backwardation premium.
        """
        idx = self._daily_index(start_date, end_date)
        z5_base = self._gas_base(idx, base=3.5)
        # Prompt-month is usually slightly above spot (contango)
        contango = np.abs(self.rng.normal(0.15, 0.10, len(idx)))
        price = np.maximum(2.0, z5_base + contango)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_dominion_south(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate Dominion South Point daily gas spot price ($/MMBtu).

        Tracks Transco Z5 with a -$0.30 to -$0.80 Appalachian discount.
        Range: $1.50-$7.50/MMBtu.
        """
        idx = self._daily_index(start_date, end_date)
        z5_base = self._gas_base(idx, base=3.5)
        discount = self.rng.uniform(-0.80, -0.30, len(idx))
        price = np.clip(z5_base + discount, 1.50, 7.50)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_tetco_m3(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate TETCO M3 daily gas spot price ($/MMBtu).

        Tracks Transco Z5 with ±$0.40 spread, wider in winter.
        Range: $2.00-$8.50/MMBtu.
        """
        idx = self._daily_index(start_date, end_date)
        z5_base = self._gas_base(idx, base=3.5)
        winter_factor = np.maximum(0, np.sin(2 * np.pi * (idx.dayofyear - 355) / 365))
        spread = self.rng.uniform(-0.40, 0.40, len(idx)) * (1 + winter_factor)
        price = np.clip(z5_base + spread, 2.00, 8.50)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})
