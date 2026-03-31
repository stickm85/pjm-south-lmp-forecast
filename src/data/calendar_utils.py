"""Calendar utilities for PJM on-peak/off-peak classification and NERC holidays."""

import yaml
from datetime import date, datetime
from pathlib import Path
from typing import Union
import pandas as pd


class CalendarUtils:
    """Manages NERC holiday calendar and on-peak/off-peak classification.

    PJM On-Peak definition:
      - Hours Ending 07–23 (HE07–HE23) Eastern Prevailing Time
      - Monday through Friday
      - Excluding NERC holidays

    All other hours are Off-Peak.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "nerc_holidays.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        self._holidays = set()
        for year_holidays in data.get("holidays", {}).values():
            for d in year_holidays:
                self._holidays.add(pd.Timestamp(d).date())

    def is_nerc_holiday(self, dt: Union[date, datetime, pd.Timestamp]) -> bool:
        """Return True if the given date is a NERC holiday."""
        if isinstance(dt, pd.Timestamp):
            d = dt.date()
        elif isinstance(dt, datetime):
            d = dt.date()
        else:
            d = dt
        return d in self._holidays

    def is_onpeak(self, dt: Union[datetime, pd.Timestamp]) -> bool:
        """Return True if the given datetime (EPT) is an on-peak hour.

        On-Peak: HE07–HE23, Monday–Friday, non-NERC-holiday.
        HE07 means the hour ending at 07:00 (i.e., 06:00–07:00).
        We use hour-ending convention: a datetime with hour=7 means HE07.
        """
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        he = dt.hour  # 0 = midnight (HE24 prior day), 1 = HE01, ..., 23 = HE23
        if he < 7 or he > 23:
            return False
        if dt.weekday() >= 5:
            return False
        if self.is_nerc_holiday(dt):
            return False
        return True

    def is_dst(self, dt: Union[datetime, pd.Timestamp]) -> bool:
        """Return True if the datetime falls in Daylight Saving Time (EPT = EDT)."""
        import pytz
        eastern = pytz.timezone("America/New_York")
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        try:
            localized = eastern.localize(dt, is_dst=None)
        except Exception:
            localized = eastern.localize(dt, is_dst=True)
        return bool(localized.dst().total_seconds() > 0)

    def hour_ending_label(self, he: int) -> str:
        """Convert integer hour-ending (1–24) to string label 'HE01'–'HE24'."""
        return f"HE{he:02d}"

    def get_onpeak_hours(self, target_date: Union[date, datetime, pd.Timestamp]) -> list:
        """Return list of hour-ending ints (1–24) that are on-peak for the given date."""
        if isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()

        onpeak = []
        for he in range(1, 25):
            if he == 24:
                pass  # midnight of next day — always off-peak
            else:
                dt = datetime(target_date.year, target_date.month, target_date.day, he, 0)
                if self.is_onpeak(dt):
                    onpeak.append(he)
        return onpeak

    def add_calendar_features(self, df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
        """Add is_onpeak, is_holiday, is_weekend, DST_flag columns to a DataFrame."""
        df = df.copy()
        df["is_onpeak"] = df[datetime_col].apply(self.is_onpeak)
        df["is_holiday"] = df[datetime_col].apply(self.is_nerc_holiday)
        df["is_weekend"] = df[datetime_col].apply(lambda x: x.weekday() >= 5)
        df["DST_flag"] = df[datetime_col].apply(self.is_dst)
        return df
