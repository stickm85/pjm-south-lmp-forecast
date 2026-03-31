"""Temporal and cyclical features for PJM SOUTH LMP forecasting."""

import numpy as np
import pandas as pd
from datetime import date, datetime
from ..data.calendar_utils import CalendarUtils


class TemporalFeatureBuilder:
    """Builds temporal and cyclical features.

    Features:
      - hour_sin, hour_cos: cyclical encoding of HE01–HE24
      - month_sin, month_cos: cyclical encoding of month 1–12
      - day_of_week: 0=Mon, 6=Sun
      - is_holiday: NERC holiday flag
      - is_weekend: Saturday/Sunday flag
      - is_onpeak: on-peak hour flag
      - DST_flag: Daylight Saving Time flag
      - days_since_epoch: time trend (captures non-stationarity / load growth)
    """

    EPOCH = pd.Timestamp("2020-01-01")

    def __init__(self, config_path=None):
        self.cal = CalendarUtils(config_path)

    def build(self, target_date) -> pd.DataFrame:
        """Build temporal features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        rows = []
        for he in range(1, 25):
            if he == 24:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, 0, 0) + pd.Timedelta(days=1)
            else:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, he, 0)

            month = target_date.month
            days_epoch = (target_date - self.EPOCH).days

            rows.append({
                "hour_ending": he,
                "hour_sin": np.sin(2 * np.pi * he / 24),
                "hour_cos": np.cos(2 * np.pi * he / 24),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "day_of_week": target_date.dayofweek,
                "is_holiday": int(self.cal.is_nerc_holiday(target_date)),
                "is_weekend": int(target_date.dayofweek >= 5),
                "is_onpeak": int(self.cal.is_onpeak(dt) if he < 24 else False),
                "DST_flag": int(self.cal.is_dst(dt)),
                "days_since_epoch": days_epoch,
            })

        return pd.DataFrame(rows)
