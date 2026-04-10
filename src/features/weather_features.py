"""Weather feature builder: heat index, wet bulb, HDD/CDD, temp differentials."""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class WeatherFeatureBuilder:
    """Builds weather-derived features for LMP forecasting.

    Features:
      - Heat_Index: apparent temperature from temp + humidity
      - Wet_Bulb_Temp: wet bulb temperature
      - Temp_South_Avg: average temperature across SOUTH cities
      - Temp_WHub_Avg: average temperature across WHub cities
      - Temp_Differential: SOUTH avg - WHub avg
      - HDD: Heating Degree Days (base 65°F)
      - CDD: Cooling Degree Days (base 65°F)
      - HDD_sq: HDD squared
      - CDD_sq: CDD squared
      - RH_Avg: Average relative humidity
    """

    BASE_TEMP = 65.0  # °F

    def build(
        self,
        target_date: pd.Timestamp,
        weather_south: Dict[str, pd.DataFrame],
        weather_whub: Dict[str, pd.DataFrame],
        humidity: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build weather features for all 24 hours of target_date."""
        target_date = pd.Timestamp(target_date)
        rows = []

        for he in range(1, 25):
            if he == 24:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, 0, 0) + pd.Timedelta(days=1)
            else:
                dt = pd.Timestamp(target_date.year, target_date.month, target_date.day, he, 0)

            # Average temperature across SOUTH cities
            south_temps = []
            for city, df in weather_south.items():
                t = self._get_temp(df, dt)
                if t is not None:
                    south_temps.append(t)
            temp_south_avg = float(np.mean(south_temps)) if south_temps else 65.0

            # Average temperature across WHub cities
            whub_temps = []
            for city, df in weather_whub.items():
                t = self._get_temp(df, dt)
                if t is not None:
                    whub_temps.append(t)
            temp_whub_avg = float(np.mean(whub_temps)) if whub_temps else 55.0

            # Average humidity (RH) across SOUTH cities
            rh_values = []
            for city, df in humidity.items():
                rh = self._get_col(df, dt, "rh_pct")
                if rh is not None:
                    rh_values.append(rh)

            rh_avg = float(np.mean(rh_values)) if rh_values else 60.0

            heat_index = self._heat_index(temp_south_avg, rh_avg)
            wet_bulb = self._wet_bulb(temp_south_avg, rh_avg)
            hdd = max(0.0, self.BASE_TEMP - temp_south_avg)
            cdd = max(0.0, temp_south_avg - self.BASE_TEMP)

            rows.append({
                "hour_ending": he,
                "Temp_South_Avg": temp_south_avg,
                "Temp_WHub_Avg": temp_whub_avg,
                "Temp_Differential": temp_south_avg - temp_whub_avg,
                "Heat_Index": heat_index,
                "Wet_Bulb_Temp": wet_bulb,
                "RH_Avg": rh_avg,
                "HDD": hdd,
                "CDD": cdd,
                "HDD_sq": hdd ** 2,
                "CDD_sq": cdd ** 2,
            })

        return pd.DataFrame(rows)

    def _get_temp(self, df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
        """Extract temperature value from weather DataFrame for given datetime."""
        return self._get_col(df, dt, "temperature_f")

    def _get_col(self, df: pd.DataFrame, dt: pd.Timestamp, col: str) -> Optional[float]:
        """Extract a column value from DataFrame for given datetime."""
        if df is None or col not in df.columns:
            return None
        if "datetime" in df.columns:
            mask = df["datetime"] == dt
            if mask.any():
                return float(df.loc[mask, col].values[0])
            # Try nearest hour
            hour_df = df[df["datetime"].dt.date == dt.date()]
            if len(hour_df) > 0:
                hour_mask = hour_df["datetime"].dt.hour == dt.hour
                if hour_mask.any():
                    return float(hour_df.loc[hour_mask, col].values[0])
                return float(hour_df[col].mean())
        return float(df[col].mean()) if len(df) > 0 else None

    @staticmethod
    def _heat_index(temp_f: float, rh: float) -> float:
        """Compute apparent temperature (heat index) using Rothfusz equation."""
        if temp_f < 80:
            return temp_f
        hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh
              - 0.22475541 * temp_f * rh - 0.00683783 * temp_f ** 2
              - 0.05481717 * rh ** 2 + 0.00122874 * temp_f ** 2 * rh
              + 0.00085282 * temp_f * rh ** 2 - 0.00000199 * temp_f ** 2 * rh ** 2)
        return hi

    @staticmethod
    def _wet_bulb(temp_f: float, rh: float) -> float:
        """Approximate wet bulb temperature (Stull 2011 empirical formula)."""
        temp_c = (temp_f - 32) * 5 / 9
        wb_c = (temp_c * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
                + np.arctan(temp_c + rh)
                - np.arctan(rh - 1.676331)
                + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
                - 4.686035)
        return wb_c * 9 / 5 + 32
