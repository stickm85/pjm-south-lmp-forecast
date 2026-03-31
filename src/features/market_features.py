"""Market features: virtual bids, BESS capacity, peak/off-peak spread."""

import pandas as pd
import numpy as np
from typing import Optional


class MarketFeatureBuilder:
    """Builds market structure features.

    Features:
      - Net_Virtual_Volume_D1: INCs - DECs at SOUTH node (D-1)
      - Virtual_Volume_Trend_7d: rolling 7-day net virtual volume trend
      - Cumulative_BESS_MW: installed battery storage capacity
      - Peak_OffPeak_Spread_D1: DA on-peak avg - off-peak avg (D-1)
    """

    def build(
        self,
        target_date: pd.Timestamp,
        virtual_bids: pd.DataFrame,
        bess_capacity: pd.DataFrame,
        south_da: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build market features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        # Net virtual volume D-1
        net_virtual_d1 = self._get_net_virtual(virtual_bids, d1)

        # 7-day trend in net virtual volume
        virtual_trend = self._virtual_trend_7d(virtual_bids, target_date)

        # Latest BESS capacity
        bess_mw = self._get_bess_capacity(bess_capacity)

        # Peak/off-peak spread D-1
        peak_offpeak_spread = self._peak_offpeak_spread(south_da, d1)

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "Net_Virtual_Volume_D1": float(net_virtual_d1),
                "Virtual_Volume_Trend_7d": float(virtual_trend),
                "Cumulative_BESS_MW": float(bess_mw),
                "Peak_OffPeak_Spread_D1": float(peak_offpeak_spread),
            })

        return pd.DataFrame(rows)

    def _get_net_virtual(self, virtual_bids: pd.DataFrame, day: pd.Timestamp) -> float:
        """Compute net virtual volume (INC - DEC) for a given day."""
        if virtual_bids is None or len(virtual_bids) == 0:
            return 0.0
        try:
            date_col = "date" if "date" in virtual_bids.columns else "datetime"
            mask = pd.to_datetime(virtual_bids[date_col]).dt.date == day.date()
            if not mask.any():
                return 0.0
            row = virtual_bids[mask].iloc[0]
            return float(row.get("inc_volume_mw", 0)) - float(row.get("dec_volume_mw", 0))
        except Exception:
            return 0.0

    def _virtual_trend_7d(self, virtual_bids: pd.DataFrame, target_date: pd.Timestamp) -> float:
        """Rolling 7-day trend in net virtual volume."""
        values = []
        for lag in range(1, 8):
            day = target_date - pd.Timedelta(days=lag)
            values.append(self._get_net_virtual(virtual_bids, day))
        return float(np.mean(values))

    def _get_bess_capacity(self, bess_capacity: pd.DataFrame) -> float:
        """Get latest BESS capacity."""
        if bess_capacity is None or len(bess_capacity) == 0:
            return 500.0
        if "bess_capacity_mw" in bess_capacity.columns:
            return float(bess_capacity["bess_capacity_mw"].iloc[-1])
        return 500.0

    def _peak_offpeak_spread(self, south_da: pd.DataFrame, day: pd.Timestamp) -> float:
        """Compute on-peak minus off-peak DA LMP spread for D-1."""
        if south_da is None or len(south_da) == 0:
            return 8.0
        try:
            if "datetime" not in south_da.columns:
                return 8.0
            day_data = south_da[south_da["datetime"].dt.date == day.date()]
            if len(day_data) == 0:
                return 8.0
            hours = day_data["datetime"].dt.hour
            onpeak_mask = (hours >= 7) & (hours <= 22)
            peak_avg = day_data.loc[onpeak_mask, "lmp"].mean()
            offpeak_avg = day_data.loc[~onpeak_mask, "lmp"].mean()
            if pd.isna(peak_avg) or pd.isna(offpeak_avg):
                return 8.0
            return float(peak_avg - offpeak_avg)
        except Exception:
            return 8.0
