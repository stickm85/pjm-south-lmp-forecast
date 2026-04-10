"""U.S. Energy Information Administration (EIA) API v2 client."""

import requests
import pandas as pd
import numpy as np
from typing import Union
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EIAClient:
    """U.S. Energy Information Administration API v2 client.

    Free data source (requires free API key from https://www.eia.gov/opendata/register.php).
    Provides Henry Hub gas prices, wholesale electricity prices, and gas storage data.
    Falls back to mock data if no API key configured.

    Configure in config/settings.yaml:
        eia:
          api_key: ""  # Free — register at https://www.eia.gov/opendata/register.php
          api_base_url: "https://api.eia.gov/v2"
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        eia_cfg = cfg.get("eia", {})
        self.api_key = eia_cfg.get("api_key", "")
        self.api_base = eia_cfg.get("api_base_url", "https://api.eia.gov/v2")
        self._mock = EIAMockData()

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fetch_henry_hub_spot(self, start_date, end_date) -> pd.DataFrame:
        """Fetch daily Henry Hub natural gas spot price ($/MMBtu).

        Endpoint: https://api.eia.gov/v2/natural-gas/pri/fut/data/
        Free, 1-day lag. Daily data.

        Returns DataFrame with columns: date, price
        """
        if not self._has_api_key():
            logger.warning("No EIA API key configured — using mock data for Henry Hub spot")
            return self._mock.generate_henry_hub_spot(start_date, end_date)
        try:
            params = {
                "api_key": self.api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": "RNGC1",
                "start": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            url = f"{self.api_base}/natural-gas/pri/fut/data/"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("response", {}).get("data", [])
            if not data:
                logger.warning("EIA returned no data for Henry Hub — using mock")
                return self._mock.generate_henry_hub_spot(start_date, end_date)
            df = pd.DataFrame(data)
            df = df.rename(columns={"period": "date", "value": "price"})
            df["date"] = pd.to_datetime(df["date"])
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df[["date", "price"]].dropna().reset_index(drop=True)
        except Exception as e:
            logger.warning(f"EIA Henry Hub fetch failed: {e} — using mock")
            return self._mock.generate_henry_hub_spot(start_date, end_date)

    def fetch_wholesale_power(self, start_date, end_date, hub: str = "PJM") -> pd.DataFrame:
        """Fetch daily wholesale electricity prices for PJM region.

        Not used in the forecast pipeline — SOUTH/WHub LMPs from PJM DataMiner are
        more granular and node-specific.

        Endpoint: https://api.eia.gov/v2/electricity/rto/daily-region-data/data/
        Includes PJM region day-ahead and real-time average prices. Free, daily.

        Returns DataFrame with columns: date, da_price, rt_price
        """
        if not self._has_api_key():
            logger.warning("No EIA API key configured — using mock data for wholesale power")
            return self._mock.generate_wholesale_power(start_date, end_date, hub)
        try:
            params = {
                "api_key": self.api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[respondent][]": hub,
                "start": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            url = f"{self.api_base}/electricity/rto/daily-region-data/data/"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("response", {}).get("data", [])
            if not data:
                logger.warning("EIA returned no data for wholesale power — using mock")
                return self._mock.generate_wholesale_power(start_date, end_date, hub)
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["period"])
            df["price"] = pd.to_numeric(df.get("value", np.nan), errors="coerce")
            # Approximate da/rt split from single price series
            result = df.groupby("date")["price"].mean().reset_index()
            result = result.rename(columns={"price": "da_price"})
            result["rt_price"] = result["da_price"]
            return result[["date", "da_price", "rt_price"]].dropna().reset_index(drop=True)
        except Exception as e:
            logger.warning(f"EIA wholesale power fetch failed: {e} — using mock")
            return self._mock.generate_wholesale_power(start_date, end_date, hub)

    def fetch_gas_storage(self, start_date, end_date) -> pd.DataFrame:
        """Fetch weekly working gas in underground storage (Lower 48, Bcf).

        Endpoint: https://api.eia.gov/v2/natural-gas/stor/wkly/data/
        Weekly working gas in storage (Lower 48). Critical for gas price
        forecasting — storage vs 5-year average signals supply tightness.

        Returns DataFrame with columns: date, storage_bcf, storage_delta_bcf
        """
        if not self._has_api_key():
            logger.warning("No EIA API key configured — using mock data for gas storage")
            return self._mock.generate_gas_storage(start_date, end_date)
        try:
            params = {
                "api_key": self.api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
                "start": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            url = f"{self.api_base}/natural-gas/stor/wkly/data/"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("response", {}).get("data", [])
            if not data:
                logger.warning("EIA returned no data for gas storage — using mock")
                return self._mock.generate_gas_storage(start_date, end_date)
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["period"])
            df["storage_bcf"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "storage_bcf"]].dropna().sort_values("date").reset_index(drop=True)
            df["storage_delta_bcf"] = df["storage_bcf"].diff().fillna(0)
            return df
        except Exception as e:
            logger.warning(f"EIA gas storage fetch failed: {e} — using mock")
            return self._mock.generate_gas_storage(start_date, end_date)


class EIAMockData:
    """Mock data generator for EIA data sources."""

    RNG_SEED = 42

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed or self.RNG_SEED)

    def generate_henry_hub_spot(self, start_date, end_date) -> pd.DataFrame:
        """Daily Henry Hub natural gas spot price ($/MMBtu).

        Seasonal pattern: higher in winter (heating demand), lower in summer.
        Range: $1.50-$8.00/MMBtu.
        """
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
        n = len(idx)
        base = 3.0
        seasonal = 0.4 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        price = np.maximum(1.50, base + seasonal + self.rng.normal(0, 0.2, n))
        price = np.minimum(8.00, price)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_wholesale_power(self, start_date, end_date, hub: str = "PJM") -> pd.DataFrame:
        """Daily wholesale electricity prices for PJM region ($/MWh).

        Seasonal pattern with summer and winter peaks.
        Returns DataFrame with columns: date, da_price, rt_price
        """
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
        n = len(idx)
        base = 40.0
        summer_peak = 12.0 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        winter_peak = 6.0 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        da_price = np.clip(base + summer_peak + winter_peak + self.rng.normal(0, 3.0, n), 20.0, 90.0)
        rt_price = da_price + self.rng.normal(0, 2.0, n)
        rt_price = np.clip(rt_price, 15.0, 100.0)
        return pd.DataFrame({
            "date": idx,
            "da_price": np.round(da_price, 2),
            "rt_price": np.round(rt_price, 2),
        })

    def generate_gas_storage(self, start_date, end_date) -> pd.DataFrame:
        """Weekly working gas in underground storage (Lower 48, Bcf).

        Seasonal pattern:
        - Injection season (Apr-Oct): storage builds
        - Withdrawal season (Nov-Mar): storage draws down
        Range: 1,500-4,000 Bcf.
        """
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="W-FRI")
        n = len(idx)
        doy = idx.dayofyear
        # Phase offset 209 gives sin peak at doy 300 (late October)
        storage = 2750.0 + 1050.0 * np.sin(2 * np.pi * (doy - 209) / 365)
        storage = np.clip(storage + self.rng.normal(0, 80, n), 1500.0, 4000.0)
        delta = np.diff(storage, prepend=storage[0])
        return pd.DataFrame({
            "date": idx,
            "storage_bcf": np.round(storage, 1),
            "storage_delta_bcf": np.round(delta, 1),
        })
