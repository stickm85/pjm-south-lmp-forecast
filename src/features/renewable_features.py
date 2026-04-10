"""Renewable energy feature builder: capacity factors, ramps, penetration."""

import pandas as pd
import numpy as np
from typing import Optional


class RenewableFeatureBuilder:
    """Builds renewable energy features for LMP forecasting.

    Features:
      - Solar_Capacity_Factor_D1: Solar_Actual / Installed_Solar_Capacity (D-1)
      - Wind_Capacity_Factor_D1: Wind_Actual / Installed_Wind_Capacity (D-1)
      - Solar_Ramp_D1: change in solar actuals from D-2 to D-1
      - Renewable_Penetration_D1: (Solar+Wind)/System_Load (D-1)
      - Solar_Forecast_vs_Actual_Ratio: D-1 solar forecast accuracy
    """

    def build(
        self,
        target_date: pd.Timestamp,
        solar_actuals: pd.DataFrame,
        wind_actuals: pd.DataFrame,
        solar_forecast_d1: pd.DataFrame,
        capacity: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build renewable features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        # Get D-1 totals
        solar_actual_d1 = self._daily_sum(solar_actuals, d1, "solar_mw")
        wind_actual_d1 = self._daily_sum(wind_actuals, d1, "wind_mw")
        solar_forecast_d1_val = self._daily_sum(solar_forecast_d1, d1, "solar_mw")

        # Get D-2 totals for ramp
        d2 = target_date - pd.Timedelta(days=2)
        solar_actual_d2 = self._daily_sum(solar_actuals, d2, "solar_mw")

        # Get latest capacity
        solar_cap = self._get_latest_capacity(capacity, "solar_capacity_mw", default=15000.0)
        wind_cap = self._get_latest_capacity(capacity, "wind_capacity_mw", default=25000.0)

        solar_cf = solar_actual_d1 / (solar_cap * 24) if solar_cap > 0 else 0.0
        wind_cf = wind_actual_d1 / (wind_cap * 24) if wind_cap > 0 else 0.0
        solar_ramp = solar_actual_d1 - solar_actual_d2

        # Rough load estimate for penetration
        renewable_total = solar_actual_d1 + wind_actual_d1
        estimated_load = 7000 * 24  # rough system load in MWh/day
        renewable_penetration = renewable_total / estimated_load if estimated_load > 0 else 0.0

        # Forecast vs actual ratio for D-1
        fa_ratio = solar_actual_d1 / solar_forecast_d1_val if solar_forecast_d1_val > 0 else 1.0

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "Solar_Capacity_Factor_D1": float(solar_cf),
                "Wind_Capacity_Factor_D1": float(wind_cf),
                "Solar_Ramp_D1": float(solar_ramp),
                "Renewable_Penetration_D1": float(renewable_penetration),
                "Solar_Forecast_vs_Actual_Ratio": float(fa_ratio),
            })

        return pd.DataFrame(rows)

    def _daily_sum(self, df: pd.DataFrame, target_day: pd.Timestamp, col: str) -> float:
        """Sum a column for a specific day."""
        if df is None or col not in df.columns:
            return 0.0
        if "datetime" not in df.columns:
            return 0.0
        mask = df["datetime"].dt.date == target_day.date()
        if mask.any():
            return float(df.loc[mask, col].sum())
        return 0.0

    def _get_latest_capacity(self, df: pd.DataFrame, col: str, default: float) -> float:
        """Get the most recent capacity value."""
        if df is None or col not in df.columns or len(df) == 0:
            return default
        return float(df[col].iloc[-1])
