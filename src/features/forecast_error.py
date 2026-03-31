"""Forecast error feature builder: load, solar, wind forecast accuracy."""

import pandas as pd
import numpy as np
from typing import Optional


class ForecastErrorBuilder:
    """Computes rolling forecast error features.

    Features:
      - Load_Forecast_Error_D1: Load_Actual - Load_Forecast (D-1)
      - Solar_Forecast_Error_D1: Solar_Actual - Solar_Forecast (D-1)
      - Wind_Forecast_Error_D1: Wind_Actual - Wind_Forecast (D-1)
      - Load_MAPE_7d: rolling 7-day MAPE for load forecasts
      - Solar_MAPE_7d: rolling 7-day MAPE for solar forecasts
      - Wind_MAPE_7d: rolling 7-day MAPE for wind forecasts
    """

    def build(
        self,
        target_date: pd.Timestamp,
        metered_load: pd.DataFrame,
        load_forecast: pd.DataFrame,
        solar_actuals: pd.DataFrame,
        solar_forecast: pd.DataFrame,
        wind_actuals: pd.DataFrame,
        wind_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build forecast error features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        d1 = target_date - pd.Timedelta(days=1)

        # D-1 errors
        load_err_d1 = self._daily_error(metered_load, load_forecast, d1, "south_load_mw")
        solar_err_d1 = self._daily_error(solar_actuals, solar_forecast, d1, "solar_mw")
        wind_err_d1 = self._daily_error(wind_actuals, wind_forecast, d1, "wind_mw")

        # 7-day rolling MAPE
        load_mape = self._rolling_mape(metered_load, load_forecast, target_date, 7, "south_load_mw")
        solar_mape = self._rolling_mape(solar_actuals, solar_forecast, target_date, 7, "solar_mw")
        wind_mape = self._rolling_mape(wind_actuals, wind_forecast, target_date, 7, "wind_mw")

        rows = []
        for he in range(1, 25):
            rows.append({
                "hour_ending": he,
                "Load_Forecast_Error_D1": float(load_err_d1),
                "Solar_Forecast_Error_D1": float(solar_err_d1),
                "Wind_Forecast_Error_D1": float(wind_err_d1),
                "Load_MAPE_7d": float(load_mape),
                "Solar_MAPE_7d": float(solar_mape),
                "Wind_MAPE_7d": float(wind_mape),
            })

        return pd.DataFrame(rows)

    def _daily_error(
        self,
        actuals: pd.DataFrame,
        forecast: pd.DataFrame,
        day: pd.Timestamp,
        col: str,
    ) -> float:
        """Compute average daily error (actual - forecast) for a given day."""
        try:
            act_day = actuals[actuals["datetime"].dt.date == day.date()][col].mean()
            fc_day = forecast[forecast["datetime"].dt.date == day.date()][col].mean()
            if pd.isna(act_day) or pd.isna(fc_day):
                return 0.0
            return float(act_day - fc_day)
        except Exception:
            return 0.0

    def _rolling_mape(
        self,
        actuals: pd.DataFrame,
        forecast: pd.DataFrame,
        target_date: pd.Timestamp,
        window: int,
        col: str,
        eps: float = 1e-6,
    ) -> float:
        """Compute rolling MAPE over the last `window` days."""
        mapes = []
        for lag in range(1, window + 1):
            day = target_date - pd.Timedelta(days=lag)
            try:
                act = actuals[actuals["datetime"].dt.date == day.date()][col].values
                fc = forecast[forecast["datetime"].dt.date == day.date()][col].values
                if len(act) == 0 or len(fc) == 0:
                    continue
                min_len = min(len(act), len(fc))
                ape = np.abs((act[:min_len] - fc[:min_len]) / (np.abs(act[:min_len]) + eps))
                mapes.append(float(np.mean(ape) * 100))
            except Exception:
                continue
        return float(np.mean(mapes)) if mapes else 5.0
