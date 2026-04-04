"""Enhanced feature engineering using new Open-Meteo, Morningstar, and PJM data sources."""

import pandas as pd
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# Approximate installed solar capacity in PJM SOUTH (MW) — updated quarterly from PJM
SOUTH_INSTALLED_SOLAR_MW = 4500.0


class EnhancedFeatureBuilder:
    """Builds 11 new derived features from Open-Meteo, Morningstar, and PJM data.

    Feature list:
      1.  ghi_solar_estimate_h      — avg GHI × installed solar MW / 1000
      2.  clear_sky_fraction_h      — direct / shortwave (clipped 0–1)
      3.  gust_curtailment_flag     — 1 if max(windgusts) > 25 m/s else 0
      4.  columbia_z5_spread        — columbia_gas_price − transco_z5_price
      5.  gas_contango              — z5_prompt_month − z5_spot
      6.  power_contango            — whub_prompt_month − whub_spot_da
      7.  reg_price_d1              — regulation market clearing price (D-1 avg)
      8.  reserve_scarcity_signal   — 1 if sync_reserve_price > 50 else 0
      9.  marginal_emission_rate_d1 — avg hourly emission rate D-1 (lb CO2/MWh)
      10. pressure_gradient_12h     — pressure_h − pressure_{h-12}
      11. precip_flag               — 1 if precipitation > 0 else 0
    """

    def build(
        self,
        target_date: Union[str, pd.Timestamp],
        openmeteo_data: Optional[pd.DataFrame] = None,
        columbia_gas: Optional[pd.DataFrame] = None,
        z5_spot: Optional[Union[pd.Series, pd.DataFrame, float]] = None,
        z5_forward: Optional[pd.DataFrame] = None,
        whub_forward: Optional[pd.DataFrame] = None,
        whub_spot_da: Optional[float] = None,
        ancillary_prices: Optional[pd.DataFrame] = None,
        emission_rates: Optional[pd.DataFrame] = None,
        installed_solar_mw: float = SOUTH_INSTALLED_SOLAR_MW,
    ) -> pd.DataFrame:
        """Build 24-row enhanced feature DataFrame for target_date.

        Args:
            target_date:       Date to build features for.
            openmeteo_data:    DataFrame from OpenMeteoClient (all cities, hourly).
            columbia_gas:      DataFrame with columns [date, price] from Morningstar.
            z5_spot:           Series, DataFrame, or float — Transco Z5 spot price D-1 ($/MMBtu).
            z5_forward:        DataFrame with columns [date, price] — Z5 prompt-month.
            whub_forward:      DataFrame with columns [date, price] — WHub prompt-month.
            whub_spot_da:      WHub DA on-peak price (user input, $/MWh).
            ancillary_prices:  DataFrame from PJMClient.fetch_ancillary_prices().
            emission_rates:    DataFrame from PJMClient.fetch_emission_rates().
            installed_solar_mw: Installed solar capacity in SOUTH zone (MW).

        Returns:
            DataFrame with 24 rows (one per hour_ending 1–24) and 11 new feature columns.
        """
        target_date = pd.Timestamp(target_date)
        hours = list(range(1, 25))
        result = pd.DataFrame({"hour_ending": hours})

        # ------------------------------------------------------------------
        # Features 1, 2, 3, 10, 11 — from Open-Meteo
        # ------------------------------------------------------------------
        result = self._add_openmeteo_features(result, target_date, openmeteo_data, installed_solar_mw)

        # ------------------------------------------------------------------
        # Features 4, 5, 6 — from Morningstar / gas forward curves
        # ------------------------------------------------------------------
        result = self._add_gas_power_features(result, target_date, columbia_gas, z5_spot,
                                              z5_forward, whub_forward, whub_spot_da)

        # ------------------------------------------------------------------
        # Features 7, 8, 9 — from PJM ancillary / emission rates
        # ------------------------------------------------------------------
        result = self._add_pjm_ancillary_features(result, target_date, ancillary_prices,
                                                   emission_rates)

        return result

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _add_openmeteo_features(
        self,
        result: pd.DataFrame,
        target_date: pd.Timestamp,
        openmeteo_data: Optional[pd.DataFrame],
        installed_solar_mw: float,
    ) -> pd.DataFrame:
        """Add features 1, 2, 3, 10, 11 from Open-Meteo data."""

        if openmeteo_data is None or openmeteo_data.empty:
            logger.warning("No Open-Meteo data — setting enhanced weather features to NaN")
            result["ghi_solar_estimate_h"] = np.nan
            result["clear_sky_fraction_h"] = np.nan
            result["gust_curtailment_flag"] = np.nan
            result["pressure_gradient_12h"] = np.nan
            result["precip_flag"] = np.nan
            return result

        # Filter to target date
        df = openmeteo_data.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        day_data = df[df["datetime"].dt.date == target_date.date()].copy()

        if day_data.empty:
            logger.warning(f"No Open-Meteo data for {target_date.date()}")
            result["ghi_solar_estimate_h"] = np.nan
            result["clear_sky_fraction_h"] = np.nan
            result["gust_curtailment_flag"] = np.nan
            result["pressure_gradient_12h"] = np.nan
            result["precip_flag"] = np.nan
            return result

        # Aggregate across all cities per hour
        day_data["hour_ending"] = day_data["datetime"].dt.hour
        day_data.loc[day_data["hour_ending"] == 0, "hour_ending"] = 24

        hourly = day_data.groupby("hour_ending").agg(
            avg_ghi=("shortwave_radiation", "mean"),
            avg_direct=("direct_radiation", "mean"),
            max_gusts=("windgusts_10m", "max"),
            sum_precip=("precipitation", "sum"),
            avg_pressure=("pressure_msl", "mean"),
        ).reset_index()

        result = result.merge(hourly, on="hour_ending", how="left")

        # Feature 1: GHI solar estimate
        result["ghi_solar_estimate_h"] = (
            result["avg_ghi"] * installed_solar_mw / 1000.0
        ).round(2)

        # Feature 2: Clear-sky fraction (direct / shortwave, clipped 0-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(
                result["avg_ghi"] > 0,
                np.clip(result["avg_direct"] / result["avg_ghi"], 0.0, 1.0),
                0.0,
            )
        result["clear_sky_fraction_h"] = np.round(fraction, 3)

        # Feature 3: Gust curtailment flag (max gusts > 25 m/s)
        result["gust_curtailment_flag"] = (result["max_gusts"] > 25.0).astype(int)

        # Feature 11: Precipitation flag
        result["precip_flag"] = (result["sum_precip"] > 0).astype(int)

        # Feature 10: Pressure gradient (pressure_h - pressure_{h-12})
        # Build a full pressure series indexed by hour_ending
        pressure_map = result.set_index("hour_ending")["avg_pressure"].to_dict()
        gradients = []
        for he in result["hour_ending"]:
            prev_he = he - 12
            if prev_he <= 0:
                prev_he += 24
            cur_p = pressure_map.get(he, np.nan)
            prev_p = pressure_map.get(prev_he, np.nan)
            gradients.append(round(cur_p - prev_p, 2) if not (np.isnan(cur_p) or np.isnan(prev_p)) else np.nan)
        result["pressure_gradient_12h"] = gradients

        # Drop intermediate columns
        result.drop(columns=["avg_ghi", "avg_direct", "max_gusts", "sum_precip", "avg_pressure"],
                    errors="ignore", inplace=True)
        return result

    def _add_gas_power_features(
        self,
        result: pd.DataFrame,
        target_date: pd.Timestamp,
        columbia_gas: Optional[pd.DataFrame],
        z5_spot: Optional[Union[pd.Series, float]],
        z5_forward: Optional[pd.DataFrame],
        whub_forward: Optional[pd.DataFrame],
        whub_spot_da: Optional[float],
    ) -> pd.DataFrame:
        """Add features 4, 5, 6 from Morningstar gas and power forward data."""

        # Resolve scalar prices for D-1 (use latest available price before target_date)
        columbia_price = self._latest_price(columbia_gas, target_date)
        z5_spot_price = (
            float(z5_spot) if isinstance(z5_spot, (int, float))
            else self._latest_price(z5_spot if isinstance(z5_spot, pd.DataFrame) else None,
                                    target_date)
        )
        z5_fwd_price = self._latest_price(z5_forward, target_date)
        whub_fwd_price = self._latest_price(whub_forward, target_date)

        # Feature 4: Columbia Gas − Transco Z5 spread
        if columbia_price is not None and z5_spot_price is not None:
            result["columbia_z5_spread"] = round(columbia_price - z5_spot_price, 3)
        else:
            result["columbia_z5_spread"] = np.nan

        # Feature 5: Gas contango (Z5 prompt-month − Z5 spot)
        if z5_fwd_price is not None and z5_spot_price is not None:
            result["gas_contango"] = round(z5_fwd_price - z5_spot_price, 3)
        else:
            result["gas_contango"] = np.nan

        # Feature 6: Power contango (WHub prompt-month − WHub spot DA)
        if whub_fwd_price is not None and whub_spot_da is not None:
            result["power_contango"] = round(whub_fwd_price - float(whub_spot_da), 2)
        else:
            result["power_contango"] = np.nan

        return result

    def _add_pjm_ancillary_features(
        self,
        result: pd.DataFrame,
        target_date: pd.Timestamp,
        ancillary_prices: Optional[pd.DataFrame],
        emission_rates: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Add features 7, 8, 9 from PJM ancillary service and emission rate data."""

        d1_date = target_date - pd.Timedelta(days=1)

        # --- Features 7 and 8 from ancillary_prices ---
        if ancillary_prices is not None and not ancillary_prices.empty:
            ap = ancillary_prices.copy()
            ap["datetime"] = pd.to_datetime(ap["datetime"])
            d1_data = ap[ap["datetime"].dt.date == d1_date.date()]
            if not d1_data.empty:
                avg_reg = d1_data["reg_a_price"].mean() if "reg_a_price" in d1_data else np.nan
                max_sync = d1_data["sync_reserve_price"].max() if "sync_reserve_price" in d1_data else np.nan
                result["reg_price_d1"] = round(float(avg_reg), 2) if pd.notna(avg_reg) else np.nan
                result["reserve_scarcity_signal"] = int(float(max_sync) > 50) if pd.notna(max_sync) else np.nan
            else:
                result["reg_price_d1"] = np.nan
                result["reserve_scarcity_signal"] = np.nan
        else:
            result["reg_price_d1"] = np.nan
            result["reserve_scarcity_signal"] = np.nan

        # --- Feature 9 from emission_rates ---
        if emission_rates is not None and not emission_rates.empty:
            er = emission_rates.copy()
            er["datetime"] = pd.to_datetime(er["datetime"])
            d1_data = er[er["datetime"].dt.date == d1_date.date()]
            if not d1_data.empty:
                col = "marginal_emission_rate_lbs_mwh"
                avg_er = d1_data[col].mean() if col in d1_data else np.nan
                result["marginal_emission_rate_d1"] = round(float(avg_er), 1) if pd.notna(avg_er) else np.nan
            else:
                result["marginal_emission_rate_d1"] = np.nan
        else:
            result["marginal_emission_rate_d1"] = np.nan

        return result

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_price(
        df: Optional[pd.DataFrame],
        before_date: pd.Timestamp,
    ) -> Optional[float]:
        """Return the most recent price in df strictly before before_date."""
        if df is None or df.empty:
            return None
        df = df.copy()
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        price_col = "price" if "price" in df.columns else df.columns[1]
        mask = df[date_col] < before_date
        subset = df[mask]
        if subset.empty:
            return None
        return float(subset.iloc[-1][price_col])
