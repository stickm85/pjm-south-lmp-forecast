"""Expand user inputs (3 scalars) into 24 hourly feature vectors."""

import numpy as np
import pandas as pd
from datetime import date
from typing import Union
from ..data.calendar_utils import CalendarUtils


class UserInputExpander:
    """Expands three user-supplied scalars into hourly feature vectors.

    User provides:
      - whub_onpeak: Western Hub DA On-Peak price ($/MWh)
      - whub_offpeak: Western Hub DA Off-Peak price ($/MWh)
      - gas_price: Transco Zone 5 gas price ($/MMBtu)

    Outputs 24 rows (HE01–HE24) with derived features.
    """

    HEAT_RATE_GAS = 7.0   # Efficient CCGT heat rate (MMBtu/MWh)
    HEAT_RATE_OLD = 10.0  # Older/peaker gas unit heat rate

    def __init__(self, config_path=None):
        self.cal = CalendarUtils(config_path)

    def expand(
        self,
        target_date: Union[date, pd.Timestamp, str],
        whub_onpeak: float,
        whub_offpeak: float,
        gas_price: float,
    ) -> pd.DataFrame:
        """Build a 24-row DataFrame with one row per hour-ending for target_date."""
        if isinstance(target_date, str):
            target_date = pd.Timestamp(target_date).date()
        elif isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()

        rows = []
        for he in range(1, 25):
            if he == 24:
                dt = pd.Timestamp(
                    target_date.year, target_date.month, target_date.day, 0, 0
                ) + pd.Timedelta(days=1)
                onpeak = False  # HE24 is always off-peak
                # WHub price assignment: HE24 is off-peak (outside HE08-HE23 block)
                whub_onpeak_hour = False
            else:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, he, 0)
                onpeak = self.cal.is_onpeak(dt)
                # WHub price assignment uses hour block only (HE08-HE23), not NERC definition.
                # Weekends and holidays also get on-peak WHub price for HE08-HE23.
                whub_onpeak_hour = (8 <= he <= 23)

            whub = whub_onpeak if whub_onpeak_hour else whub_offpeak
            implied_hr = whub / gas_price if gas_price > 0 else 0.0
            spark_gas = whub - (self.HEAT_RATE_GAS * gas_price)
            spark_old = whub - (self.HEAT_RATE_OLD * gas_price)
            gas_power_ratio = gas_price / whub_onpeak if whub_onpeak > 0 else 0.0

            rows.append({
                "datetime": dt,
                "hour_ending": he,
                "WHub_DA": whub,
                "Gas_Price": gas_price,
                "Implied_Heat_Rate": implied_hr,
                "Spark_Spread_Gas": spark_gas,
                "Spark_Spread_Old": spark_old,
                "Gas_Power_Ratio": gas_power_ratio,
                "is_onpeak": int(onpeak),
            })

        return pd.DataFrame(rows)
