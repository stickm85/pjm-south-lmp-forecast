"""Market features: peak/off-peak spread."""

import pandas as pd
import numpy as np
from typing import Optional


class MarketFeatureBuilder:
    """Builds market structure features.

    Features:
      - Peak_OffPeak_Spread_D1: DA on-peak avg - off-peak avg (D-1)
    """

    def build(
        self,
        target_date: pd.Timestamp,
        south_da: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build market features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        # Peak/off-peak spread D-1
        peak_offpeak_spread = self._peak_offpeak_spread(south_da, d1)

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "Peak_OffPeak_Spread_D1": float(peak_offpeak_spread),
            })

        return pd.DataFrame(rows)

    def _peak_offpeak_spread(self, south_da: pd.DataFrame, day: pd.Timestamp) -> float:
        """Compute on-peak minus off-peak DA LMP spread for D-1.

        PJM on-peak definition: HE08–HE23 (hour >= 8 and hour <= 23).
        """
        if south_da is None or len(south_da) == 0:
            return 8.0
        try:
            if "datetime" not in south_da.columns:
                return 8.0
            day_data = south_da[south_da["datetime"].dt.date == day.date()]
            if len(day_data) == 0:
                return 8.0
            hours = day_data["datetime"].dt.hour
            onpeak_mask = (hours >= 8) & (hours <= 23)
            peak_avg = day_data.loc[onpeak_mask, "lmp"].mean()
            offpeak_avg = day_data.loc[~onpeak_mask, "lmp"].mean()
            if pd.isna(peak_avg) or pd.isna(offpeak_avg):
                return 8.0
            return float(peak_avg - offpeak_avg)
        except Exception:
            return 8.0
