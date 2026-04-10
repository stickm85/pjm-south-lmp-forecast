"""Regime features: reserve margin signal."""

import pandas as pd
import numpy as np
from typing import Optional


class RegimeFeatureBuilder:
    """Builds market regime features.

    Features:
      - Reserve_Margin_D1_pct: available capacity margin percentage
    """

    def build(
        self,
        target_date: pd.Timestamp,
        metered_load: pd.DataFrame,
        generator_outages: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build regime features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        reserve_margin = self._compute_reserve_margin(metered_load, generator_outages, d1)

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "Reserve_Margin_D1_pct": float(reserve_margin),
            })

        return pd.DataFrame(rows)

    def _compute_reserve_margin(
        self,
        metered_load: pd.DataFrame,
        generator_outages: pd.DataFrame,
        day: pd.Timestamp,
        installed_capacity_mw: float = 180000.0,
    ) -> float:
        """Estimate reserve margin as (capacity - outages - peak_load) / capacity.

        Uses pjm_load_mw if available (more accurate PJM-system total);
        falls back to south_load_mw * 12 if pjm_load_mw is not present.
        """
        try:
            # Get peak load for the day
            if metered_load is not None and len(metered_load) > 0 and "datetime" in metered_load.columns:
                day_load = metered_load[metered_load["datetime"].dt.date == day.date()]
                if len(day_load) > 0:
                    if "pjm_load_mw" in metered_load.columns:
                        peak_load = float(day_load["pjm_load_mw"].max())
                    elif "south_load_mw" in metered_load.columns:
                        peak_load = float(day_load["south_load_mw"].max()) * 12
                    else:
                        peak_load = 80000.0
                else:
                    peak_load = 80000.0
            else:
                peak_load = 80000.0

            # Get outages
            if generator_outages is not None and len(generator_outages) > 0:
                date_col = "date" if "date" in generator_outages.columns else "datetime"
                mask = pd.to_datetime(generator_outages[date_col]).dt.date == day.date()
                mw_offline = float(generator_outages.loc[mask, "mw_offline"].values[0]) if mask.any() else 2000.0
            else:
                mw_offline = 2000.0

            available_cap = installed_capacity_mw - mw_offline
            if peak_load > 0:
                return (available_cap - peak_load) / available_cap * 100
            return 20.0
        except Exception:
            return 20.0
