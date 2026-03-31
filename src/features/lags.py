"""Lag feature construction for basis and DA-RT spread."""

import pandas as pd
import numpy as np
from typing import Optional


class LagFeatureBuilder:
    """Constructs lagged features from historical LMP data.

    Features computed:
      - Basis_D1: SOUTH_DA - WHub_DA (D-1 lag)
      - Basis_D7: SOUTH_DA - WHub_DA (D-7 lag)
      - DA_RT_Spread_D1: DA_LMP - RT_LMP (D-1 lag)
      - DA_RT_Spread_D7: DA_LMP - RT_LMP (D-7 lag)
      - DA_RT_Spread_7d_mean: rolling 7-day mean
      - DA_RT_Spread_7d_std: rolling 7-day std
      - DA_RT_Spread_persist: persistence flag (spread > 0 for 3+ consecutive days)
      - Congestion_D1, Congestion_D7: congestion component lags
      - Loss_D1, Loss_D7: loss component lags
    """

    def build(
        self,
        target_date: pd.Timestamp,
        south_da: pd.DataFrame,
        whub_da: pd.DataFrame,
        south_rt: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build lag features for each of 24 hours on target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)
        d7 = target_date - pd.Timedelta(days=7)

        south_da_idx = south_da.set_index("datetime") if "datetime" in south_da.columns else south_da.copy()
        whub_da_idx = whub_da.set_index("datetime") if "datetime" in whub_da.columns else whub_da.copy()
        south_rt_idx = south_rt.set_index("datetime") if "datetime" in south_rt.columns else south_rt.copy()

        rows = []
        for he in range(1, 25):
            if he == 24:
                dt_target = pd.Timestamp(target_date.year, target_date.month, target_date.day, 0, 0) + pd.Timedelta(days=1)
                dt_d1 = dt_target - pd.Timedelta(days=1)
                dt_d7 = dt_target - pd.Timedelta(days=7)
            else:
                dt_target = pd.Timestamp(target_date.year, target_date.month, target_date.day, he, 0)
                dt_d1 = pd.Timestamp(d1.year, d1.month, d1.day, he, 0)
                dt_d7 = pd.Timestamp(d7.year, d7.month, d7.day, he, 0)

            def safe_get(df_idx, dt, col, default=0.0):
                try:
                    val = df_idx.loc[dt, col]
                    return float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                except Exception:
                    return default

            south_d1 = safe_get(south_da_idx, dt_d1, "lmp")
            whub_d1 = safe_get(whub_da_idx, dt_d1, "lmp")
            rt_d1 = safe_get(south_rt_idx, dt_d1, "lmp")
            south_d7 = safe_get(south_da_idx, dt_d7, "lmp")
            whub_d7 = safe_get(whub_da_idx, dt_d7, "lmp")
            rt_d7 = safe_get(south_rt_idx, dt_d7, "lmp")

            congestion_d1 = safe_get(south_da_idx, dt_d1, "congestion")
            congestion_d7 = safe_get(south_da_idx, dt_d7, "congestion")
            loss_d1 = safe_get(south_da_idx, dt_d1, "loss")
            loss_d7 = safe_get(south_da_idx, dt_d7, "loss")

            basis_d1 = south_d1 - whub_d1
            basis_d7 = south_d7 - whub_d7
            spread_d1 = south_d1 - rt_d1
            spread_d7 = south_d7 - rt_d7

            spreads_7d = []
            for lag in range(1, 8):
                dt_lag = dt_target - pd.Timedelta(days=lag)
                s = safe_get(south_da_idx, dt_lag, "lmp")
                r = safe_get(south_rt_idx, dt_lag, "lmp")
                spreads_7d.append(s - r)

            spread_mean = float(np.mean(spreads_7d))
            spread_std = float(np.std(spreads_7d))
            spread_persist = int(sum(1 for x in spreads_7d[:3] if x > 0) >= 3)

            rows.append({
                "hour_ending": he,
                "Basis_D1": basis_d1,
                "Basis_D7": basis_d7,
                "DA_RT_Spread_D1": spread_d1,
                "DA_RT_Spread_D7": spread_d7,
                "DA_RT_Spread_7d_mean": spread_mean,
                "DA_RT_Spread_7d_std": spread_std,
                "DA_RT_Spread_persist": spread_persist,
                "Congestion_D1": congestion_d1,
                "Congestion_D7": congestion_d7,
                "Loss_D1": loss_d1,
                "Loss_D7": loss_d7,
            })

        return pd.DataFrame(rows)
