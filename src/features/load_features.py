"""Load feature builder for net load, ramp, and ratio features."""

import pandas as pd
import numpy as np
from typing import Optional


class LoadFeatureBuilder:
    """Builds load-related features for LMP forecasting.

    Features:
      - Net_Load_h: Load_Forecast - Solar_Forecast - Wind_Forecast
      - Net_Load_Ramp: Net_Load[h] - Net_Load[h-1]
      - Load_Ratio: SOUTH_Load / PJM_Total_Load
      - Rolling_12mo_peak_load_trend: rolling 365-day peak
      - YoY_load_growth_rate: year-over-year load growth
    """

    def build(
        self,
        target_date: pd.Timestamp,
        load_forecast: pd.DataFrame,
        solar_forecast: pd.DataFrame,
        wind_forecast: pd.DataFrame,
        metered_load: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build load features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)

        # Filter to target_date
        day_load = load_forecast[load_forecast["datetime"].dt.date == target_date.date()].copy()
        day_solar = solar_forecast[solar_forecast["datetime"].dt.date == target_date.date()].copy()
        day_wind = wind_forecast[wind_forecast["datetime"].dt.date == target_date.date()].copy()

        # Compute rolling peak load from historical metered data
        rolling_peak = metered_load["south_load_mw"].max() if len(metered_load) > 0 else 10000.0

        # Year-over-year growth rate
        yoy_growth = self._compute_yoy_growth(metered_load)

        rows = []
        for he in range(1, 25):
            if he == 24:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, 0, 0) + pd.Timedelta(days=1)
            else:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, he, 0)

            south_load = self._get_value(day_load, dt, "south_load_mw", default=7000.0)
            pjm_load = self._get_value(day_load, dt, "pjm_load_mw", default=80000.0)
            solar = self._get_value(day_solar, dt, "solar_mw", default=0.0)
            wind = self._get_value(day_wind, dt, "wind_mw", default=5000.0)

            net_load = south_load - solar - wind
            load_ratio = south_load / pjm_load if pjm_load > 0 else 0.0

            rows.append({
                "hour_ending": he,
                "South_Load_Forecast": south_load,
                "PJM_Load_Forecast": pjm_load,
                "Solar_Forecast": solar,
                "Wind_Forecast": wind,
                "Net_Load_h": net_load,
                "Load_Ratio": load_ratio,
                "Rolling_12mo_Peak_Load": rolling_peak,
                "YoY_Load_Growth_Rate": yoy_growth,
            })

        df = pd.DataFrame(rows)
        # Net load ramp: difference from previous hour
        df["Net_Load_Ramp"] = df["Net_Load_h"].diff().fillna(0)
        return df

    def _get_value(self, df: pd.DataFrame, dt: pd.Timestamp, col: str, default: float = 0.0) -> float:
        """Extract scalar value from DataFrame for a specific datetime."""
        if df is None or len(df) == 0:
            return default
        if "datetime" in df.columns:
            mask = df["datetime"] == dt
            if mask.any():
                val = df.loc[mask, col].values[0]
                return float(val)
        # Fallback: use positional index based on hour
        hour_idx = dt.hour if dt.hour > 0 else 0
        if hour_idx < len(df):
            return float(df[col].iloc[hour_idx])
        return default

    def _compute_yoy_growth(self, metered_load: pd.DataFrame) -> float:
        """Compute year-over-year load growth rate from historical data."""
        if metered_load is None or len(metered_load) < 2:
            return 0.01
        try:
            metered_load = metered_load.copy()
            metered_load["year"] = metered_load["datetime"].dt.year
            annual = metered_load.groupby("year")["south_load_mw"].mean()
            if len(annual) >= 2:
                years = sorted(annual.index)
                growth = (annual[years[-1]] / annual[years[-2]]) - 1
                return float(growth)
        except Exception:
            pass
        return 0.01
