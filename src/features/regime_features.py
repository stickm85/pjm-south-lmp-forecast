"""Regime features: EEA flags, reserve margins, scarcity indicators."""

import pandas as pd
import numpy as np
from typing import Optional


class RegimeFeatureBuilder:
    """Builds market regime features.

    Features:
      - EEA_Flag_D1: Emergency Event Alert flag from D-1
      - Reserve_Margin_D1_pct: available capacity margin percentage
      - Consecutive_Scarcity_Days: number of consecutive prior days with EEA events
    """

    def build(
        self,
        target_date: pd.Timestamp,
        emergency_logs: pd.DataFrame,
        metered_load: pd.DataFrame,
        generator_outages: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build regime features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        eea_flag_d1 = self._get_eea_flag(emergency_logs, d1)
        consecutive_scarcity = self._consecutive_scarcity(emergency_logs, target_date)
        reserve_margin = self._compute_reserve_margin(metered_load, generator_outages, d1)

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "EEA_Flag_D1": int(eea_flag_d1),
                "Consecutive_Scarcity_Days": int(consecutive_scarcity),
                "Reserve_Margin_D1_pct": float(reserve_margin),
            })

        return pd.DataFrame(rows)

    def _get_eea_flag(self, emergency_logs: pd.DataFrame, day: pd.Timestamp) -> int:
        """Get EEA flag for a specific day."""
        if emergency_logs is None or len(emergency_logs) == 0:
            return 0
        try:
            date_col = "date" if "date" in emergency_logs.columns else "datetime"
            mask = pd.to_datetime(emergency_logs[date_col]).dt.date == day.date()
            if not mask.any():
                return 0
            return int(emergency_logs.loc[mask, "eea_flag"].values[0])
        except Exception:
            return 0

    def _consecutive_scarcity(self, emergency_logs: pd.DataFrame, target_date: pd.Timestamp) -> int:
        """Count consecutive days with EEA events prior to target_date."""
        count = 0
        for lag in range(1, 15):
            day = target_date - pd.Timedelta(days=lag)
            if self._get_eea_flag(emergency_logs, day):
                count += 1
            else:
                break
        return count

    def _compute_reserve_margin(
        self,
        metered_load: pd.DataFrame,
        generator_outages: pd.DataFrame,
        day: pd.Timestamp,
        installed_capacity_mw: float = 180000.0,
    ) -> float:
        """Estimate reserve margin as (capacity - outages - peak_load) / capacity."""
        try:
            # Get peak load for the day
            if metered_load is not None and len(metered_load) > 0 and "datetime" in metered_load.columns:
                day_load = metered_load[metered_load["datetime"].dt.date == day.date()]
                peak_load = float(day_load["south_load_mw"].max()) * 12 if len(day_load) > 0 else 80000.0
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
