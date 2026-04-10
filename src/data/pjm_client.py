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

    def fetch_ancillary_prices(self, start_date, end_date) -> pd.DataFrame:
        """Fetch ancillary service market clearing prices (RegA, RegD, Sync Reserve).

        Sources: reg_market_results and sr_market_results PJM feeds.
        When regulation or reserve prices spike, DA LMPs often co-move via ORDC.

        Returns DataFrame with columns:
            datetime, reg_a_price, reg_d_price, sync_reserve_price
        """
        if not self._has_api_key():
            logger.warning("No PJM API key configured — using mock data for ancillary prices")
            return self._mock.generate_ancillary_prices(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_emission_rates(self, start_date, end_date) -> pd.DataFrame:
        """Fetch 5-minute marginal CO2 emission rates, aggregated to hourly.

        Source: fivemin_marginal_emission_rates PJM feed.
        Higher emission rate signals dirtier/more expensive unit on the margin
        (gas peaker or coal), directly correlated with higher LMPs.

        Returns DataFrame with columns:
            datetime, marginal_emission_rate_lbs_mwh
        """
        if not self._has_api_key():
            logger.warning("No PJM API key configured — using mock data for emission rates")
            return self._mock.generate_emission_rates(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_instantaneous_load(self, start_date, end_date) -> pd.DataFrame:
        """Fetch PJM instantaneous load (5-min), aggregated to hourly.

        Source: inst_load PJM feed.
        Allows computing real-time load forecast error for intra-day accuracy
        assessment and model recalibration.

        Returns DataFrame with columns:
            datetime, instantaneous_load_mw
        """
        if not self._has_api_key():
            logger.warning("No PJM API key configured — using mock data for instantaneous load")
            return self._mock.generate_instantaneous_load(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")

    def fetch_transmission_constraints(self, start_date, end_date) -> pd.DataFrame:
        """Fetch active binding transmission constraints with shadow prices.

        Source: PJM DataMiner 2 `transmission_constraints` feed.
        Binding constraints into SOUTH increase the congestion component of LMP.
        Shadow prices quantify the $/MWh impact of each constraint.
        This is the highest-value missing data source for predicting the congestion
        component of SOUTH LMP (~30% of the SOUTH-WHub basis).

        Returns DataFrame with columns:
            datetime, n_binding_constraints, max_shadow_price, total_shadow_price
        """
        if not self._has_api_key():
            logger.warning("No PJM API key configured — using mock data for transmission constraints")
            return self._mock.generate_transmission_constraints(start_date, end_date)
        raise NotImplementedError("Set pjm.api_key in config/settings.yaml")


from .mock_data import MockDataGenerator as MockFallback
