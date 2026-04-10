"""Mock data generator for all 41 data sources used in PJM SOUTH LMP forecasting."""

import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Union


class MockDataGenerator:
    """Generates realistic mock data for all data sources.

    All generated data uses seeded random number generation for reproducibility
    and realistic domain-specific patterns (seasonal temperature, daily load shape, etc.).
    """

    RNG_SEED = 42

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed or self.RNG_SEED)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hourly_index(self, start_date, end_date) -> pd.DatetimeIndex:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return pd.date_range(start, end, freq="h", inclusive="left")

    def _daily_index(self, start_date, end_date) -> pd.DatetimeIndex:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        return pd.date_range(start, end, freq="D")

    def _hour_shape(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Typical hourly load/price shape: low overnight, peaks mid-afternoon."""
        hours = idx.hour
        shape = 1.0 + 0.3 * np.sin(np.pi * (hours - 6) / 18) + 0.15 * np.sin(2 * np.pi * hours / 24)
        return shape

    def _seasonal_temp(self, idx: pd.DatetimeIndex, base: float = 55.0, amp: float = 25.0) -> np.ndarray:
        """Seasonal temperature in °F (peaks in summer)."""
        day_of_year = idx.dayofyear
        return base + amp * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # ------------------------------------------------------------------
    # LMP / price data
    # ------------------------------------------------------------------

    def generate_pjm_da_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Hourly SOUTH DA LMP with energy, congestion, loss components."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        base_price = 40.0
        shape = self._hour_shape(idx)
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        noise = self.rng.normal(0, 3, n)
        energy = base_price * shape * seasonal + noise
        congestion = self.rng.normal(2.5, 1.5, n)
        loss = self.rng.normal(0.8, 0.3, n)
        lmp = energy + congestion + loss
        return pd.DataFrame({"datetime": idx, "lmp": lmp, "energy": energy,
                              "congestion": congestion, "loss": loss})

    def generate_whub_da_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Western Hub DA LMP hourly."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        base_price = 38.0
        shape = self._hour_shape(idx)
        seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        noise = self.rng.normal(0, 2.5, n)
        lmp = base_price * shape * seasonal + noise
        return pd.DataFrame({"datetime": idx, "lmp": lmp})

    def generate_rt_lmp(self, start_date, end_date) -> pd.DataFrame:
        """Hourly SOUTH RT LMP."""
        da = self.generate_pjm_da_lmp(start_date, end_date)
        noise = self.rng.normal(0, 4, len(da))
        da["lmp"] = da["lmp"] + noise
        return da[["datetime", "lmp"]]

    def generate_gas_price(self, start_date, end_date) -> pd.DataFrame:
        """Daily Transco Zone 5 gas price ($/MMBtu)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        base = 3.5
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        noise = self.rng.normal(0, 0.25, n)
        price = np.maximum(1.5, base + seasonal + noise)
        return pd.DataFrame({"date": idx, "gas_price": price})

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def generate_load_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Hourly D+1 load forecast (SOUTH + PJM total)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        shape = self._hour_shape(idx)
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        south_base = 7000.0
        pjm_base = 80000.0
        noise_s = self.rng.normal(0, 150, n)
        noise_p = self.rng.normal(0, 1500, n)
        south_load = south_base * shape * seasonal + noise_s
        pjm_load = pjm_base * shape * seasonal + noise_p
        return pd.DataFrame({"datetime": idx, "south_load_mw": south_load, "pjm_load_mw": pjm_load})

    def generate_metered_load(self, start_date, end_date) -> pd.DataFrame:
        """Hourly actual metered load."""
        df = self.generate_load_forecast(start_date, end_date)
        noise = self.rng.normal(0, 100, len(df))
        df["south_load_mw"] += noise
        return df

    # ------------------------------------------------------------------
    # Weather data
    # ------------------------------------------------------------------

    def generate_weather_forecast(self, start_date, end_date, city: str = "Richmond VA") -> pd.DataFrame:
        """Hourly D+1 temperature forecasts (°F)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        base_temp = self._seasonal_temp(idx)
        diurnal = 5.0 * np.sin(np.pi * (idx.hour - 6) / 12)
        noise = self.rng.normal(0, 2, n)
        temp = base_temp + diurnal + noise
        return pd.DataFrame({"datetime": idx, "temperature_f": temp, "city": city})

    def generate_humidity_forecast(self, start_date, end_date, city: str = "Richmond VA") -> pd.DataFrame:
        """Hourly dew point + relative humidity."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        dew_point = self._seasonal_temp(idx, base=45.0, amp=15.0) + self.rng.normal(0, 3, n)
        rh = np.clip(60 + self.rng.normal(0, 10, n), 20, 100)
        return pd.DataFrame({"datetime": idx, "dew_point_f": dew_point, "rh_pct": rh, "city": city})

    def generate_wind_cloud_forecast(self, start_date, end_date, city: str = "Richmond VA") -> pd.DataFrame:
        """Hourly wind speed + cloud cover."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        wind_speed = np.abs(self.rng.normal(10, 5, n))
        cloud_cover = np.clip(self.rng.normal(50, 30, n), 0, 100)
        return pd.DataFrame({"datetime": idx, "wind_speed_mph": wind_speed,
                              "cloud_cover_pct": cloud_cover, "city": city})

    # ------------------------------------------------------------------
    # Renewable generation
    # ------------------------------------------------------------------

    def generate_solar_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Hourly PJM solar forecast (MW)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        solar_shape = np.maximum(0, np.sin(np.pi * (idx.hour - 6) / 12))
        base = 8000.0
        noise = self.rng.normal(0, 200, n)
        solar = np.maximum(0, base * solar_shape * seasonal + noise)
        return pd.DataFrame({"datetime": idx, "solar_mw": solar})

    def generate_wind_forecast(self, start_date, end_date) -> pd.DataFrame:
        """Hourly PJM wind forecast (MW)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        base = 10000.0
        seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (idx.dayofyear - 80) / 365)
        noise = self.rng.normal(0, 500, n)
        wind = np.maximum(0, base * seasonal + noise)
        return pd.DataFrame({"datetime": idx, "wind_mw": wind})

    def generate_solar_actuals(self, start_date, end_date) -> pd.DataFrame:
        """Hourly solar generation actuals."""
        df = self.generate_solar_forecast(start_date, end_date)
        df["solar_mw"] += self.rng.normal(0, 300, len(df))
        df["solar_mw"] = np.maximum(0, df["solar_mw"])
        return df

    def generate_wind_actuals(self, start_date, end_date) -> pd.DataFrame:
        """Hourly wind generation actuals."""
        df = self.generate_wind_forecast(start_date, end_date)
        df["wind_mw"] += self.rng.normal(0, 400, len(df))
        df["wind_mw"] = np.maximum(0, df["wind_mw"])
        return df

    # ------------------------------------------------------------------
    # Outages
    # ------------------------------------------------------------------

    def generate_transmission_outages(self, start_date, end_date) -> pd.DataFrame:
        """Daily transmission outage flags."""
        idx = self._daily_index(start_date, end_date)
        outage_flag = (self.rng.random(len(idx)) < 0.1).astype(int)
        return pd.DataFrame({"date": idx, "transmission_outage_flag": outage_flag})

    def generate_generator_outages(self, start_date, end_date) -> pd.DataFrame:
        """Daily generator outages (MW offline)."""
        idx = self._daily_index(start_date, end_date)
        mw_offline = np.abs(self.rng.normal(2000, 800, len(idx)))
        return pd.DataFrame({"date": idx, "mw_offline": mw_offline})

    # ------------------------------------------------------------------
    # ISO interface / interchange
    # ------------------------------------------------------------------

    def generate_iso_prices(self, start_date, end_date) -> pd.DataFrame:
        """Hourly MISO/NYISO interface prices."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        miso = 35.0 + self._hour_shape(idx) * 5 + self.rng.normal(0, 3, n)
        nyiso = 42.0 + self._hour_shape(idx) * 6 + self.rng.normal(0, 4, n)
        return pd.DataFrame({"datetime": idx, "miso_price": miso, "nyiso_price": nyiso})

    def generate_interchange(self, start_date, end_date) -> pd.DataFrame:
        """Hourly net scheduled interchange flows (MW)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        net_flow = self.rng.normal(500, 800, n)
        return pd.DataFrame({"datetime": idx, "net_scheduled_mw": net_flow})

    # ------------------------------------------------------------------
    # Fuel / market
    # ------------------------------------------------------------------

    def generate_henry_hub(self, start_date, end_date) -> pd.DataFrame:
        """Daily Henry Hub futures ($/MMBtu)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        base = 3.0
        seasonal = 0.4 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        price = np.maximum(1.5, base + seasonal + self.rng.normal(0, 0.2, n))
        return pd.DataFrame({"date": idx, "henry_hub_price": price})

    def generate_fuel_mix(self, start_date, end_date) -> pd.DataFrame:
        """Hourly generation by fuel (MW)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        gas = np.maximum(0, 20000 + self.rng.normal(0, 2000, n))
        coal = np.maximum(0, 30000 + self.rng.normal(0, 1500, n))
        nuclear = np.full(n, 30000.0) + self.rng.normal(0, 200, n)
        hydro = np.maximum(0, 3000 + self.rng.normal(0, 300, n))
        return pd.DataFrame({"datetime": idx, "gas_mw": gas, "coal_mw": coal,
                              "nuclear_mw": nuclear, "hydro_mw": hydro})

    def generate_coal_price(self, start_date, end_date) -> pd.DataFrame:
        """Weekly coal spot price ($/short ton)."""
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="W")
        n = len(idx)
        price = np.maximum(50, 70 + self.rng.normal(0, 5, n))
        return pd.DataFrame({"date": idx, "coal_price_per_ton": price})

    # ------------------------------------------------------------------
    # Capacity / BESS
    # ------------------------------------------------------------------

    def generate_installed_capacity(self, start_date, end_date) -> pd.DataFrame:
        """Monthly solar/wind capacity (MW)."""
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="MS")
        n = len(idx)
        solar_cap = 15000 + np.arange(n) * 50 + self.rng.normal(0, 100, n)
        wind_cap = 25000 + np.arange(n) * 30 + self.rng.normal(0, 80, n)
        return pd.DataFrame({"date": idx, "solar_capacity_mw": solar_cap,
                              "wind_capacity_mw": wind_cap})

    def generate_bess_capacity(self, start_date, end_date) -> pd.DataFrame:
        """Quarterly BESS capacity (MW)."""
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="QS")
        n = len(idx)
        bess = 500 + np.arange(n) * 100 + self.rng.normal(0, 20, n)
        return pd.DataFrame({"date": idx, "bess_capacity_mw": np.maximum(0, bess)})

    # ------------------------------------------------------------------
    # Market / virtual / DR / environmental
    # ------------------------------------------------------------------

    def generate_virtual_bids(self, start_date, end_date) -> pd.DataFrame:
        """Daily virtual bid volume (INCs and DECs, MW)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        incs = np.abs(self.rng.normal(500, 200, n))
        decs = np.abs(self.rng.normal(450, 180, n))
        return pd.DataFrame({"date": idx, "inc_volume_mw": incs, "dec_volume_mw": decs})

    def generate_emergency_logs(self, start_date, end_date) -> pd.DataFrame:
        """Daily EEA flags."""
        idx = self._daily_index(start_date, end_date)
        eea_flag = (self.rng.random(len(idx)) < 0.03).astype(int)
        return pd.DataFrame({"date": idx, "eea_flag": eea_flag})

    def generate_rggi_price(self, start_date, end_date) -> pd.DataFrame:
        """Quarterly RGGI allowance price ($/ton CO2)."""
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="QS")
        n = len(idx)
        price = np.maximum(5, 14 + self.rng.normal(0, 1.5, n))
        return pd.DataFrame({"date": idx, "rggi_price": price})

    def generate_demand_response(self, start_date, end_date) -> pd.DataFrame:
        """Daily DR event flags."""
        idx = self._daily_index(start_date, end_date)
        dr_flag = (self.rng.random(len(idx)) < 0.05).astype(int)
        dr_mw = dr_flag * np.abs(self.rng.normal(800, 200, len(idx)))
        return pd.DataFrame({"date": idx, "dr_event_flag": dr_flag, "dr_mw": dr_mw})

    # ------------------------------------------------------------------
    # New PJM API mock generators
    # ------------------------------------------------------------------

    def generate_ancillary_prices(self, start_date, end_date) -> pd.DataFrame:
        """Hourly ancillary service prices (RegA, RegD, Sync Reserve) in $/MW."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        reg_a = np.maximum(0, self.rng.normal(12.0, 5.0, n))
        reg_d = np.maximum(0, self.rng.normal(18.0, 7.0, n))
        sync_reserve = np.maximum(0, self.rng.normal(8.0, 15.0, n))
        return pd.DataFrame({
            "datetime": idx,
            "reg_a_price": np.round(reg_a, 2),
            "reg_d_price": np.round(reg_d, 2),
            "sync_reserve_price": np.round(sync_reserve, 2),
        })

    def generate_emission_rates(self, start_date, end_date) -> pd.DataFrame:
        """Hourly marginal CO2 emission rates (lb CO2/MWh)."""
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        # Gas peaker ~1100 lb/MWh, coal ~2200 lb/MWh; mix varies by hour
        base = 1100.0 + self._hour_shape(idx) * 200.0
        noise = self.rng.normal(0, 100, n)
        rate = np.clip(base + noise, 500.0, 2500.0)
        return pd.DataFrame({
            "datetime": idx,
            "marginal_emission_rate_lbs_mwh": np.round(rate, 1),
        })

    def generate_tx_ratings(self, start_date, end_date) -> pd.DataFrame:
        """Daily transmission de-rate flags and derated MW."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        derate_flag = (self.rng.random(n) < 0.08).astype(int)
        derated_mw = derate_flag * np.abs(self.rng.normal(500, 200, n))
        return pd.DataFrame({
            "date": idx,
            "tx_derate_flag": derate_flag,
            "derated_mw": np.round(derated_mw, 0),
        })

    def generate_instantaneous_load(self, start_date, end_date) -> pd.DataFrame:
        """Hourly PJM instantaneous load (aggregated from 5-min, MW)."""
        df = self.generate_metered_load(start_date, end_date)
        # Instantaneous is slightly different from metered (5-min snaps)
        df["instantaneous_load_mw"] = df["south_load_mw"] + self.rng.normal(0, 50, len(df))
        return df[["datetime", "instantaneous_load_mw"]]

    def generate_transmission_constraints(self, start_date, end_date) -> pd.DataFrame:
        """Hourly binding transmission constraint data with shadow prices.

        Realistic patterns:
        - ~15% of hours have at least one binding constraint
        - Shadow prices $0-50/MWh, higher during peak/hot weather
        - More constraints bind during on-peak hours (HE08-HE23)

        Returns DataFrame with columns:
            datetime, n_binding_constraints, max_shadow_price, total_shadow_price
        """
        idx = self._hourly_index(start_date, end_date)
        n = len(idx)
        # More binding constraints likely during on-peak hours
        is_onpeak = ((idx.hour >= 7) & (idx.hour <= 22)).astype(float)
        binding_prob = 0.10 + 0.10 * is_onpeak  # 10% off-peak, 20% on-peak
        has_binding = self.rng.random(n) < binding_prob
        n_binding = np.where(has_binding, self.rng.integers(1, 6, n), 0)
        # Shadow prices higher on-peak and in summer
        seasonal = 0.3 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        shadow_base = 15.0 + 10.0 * is_onpeak + 8.0 * seasonal
        max_shadow = np.where(
            n_binding > 0,
            np.clip(shadow_base + self.rng.normal(0, 5.0, n), 1.0, 50.0),
            0.0,
        )
        total_shadow = np.where(n_binding > 0, max_shadow * n_binding * 0.7, 0.0)
        return pd.DataFrame({
            "datetime": idx,
            "n_binding_constraints": n_binding.astype(int),
            "max_shadow_price": np.round(max_shadow, 2),
            "total_shadow_price": np.round(total_shadow, 2),
        })

    # ------------------------------------------------------------------
    # New Morningstar mock generators
    # ------------------------------------------------------------------

    def generate_columbia_gas(self, start_date, end_date) -> pd.DataFrame:
        """Daily Columbia Gas (TCO) spot price ($/MMBtu)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        spread = self.rng.uniform(-0.5, 0.5, n)
        price = np.maximum(2.0, 3.5 + seasonal + spread + self.rng.normal(0, 0.2, n))
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_whub_forward(self, start_date, end_date) -> pd.DataFrame:
        """Daily PJM WHub prompt-month forward price ($/MWh)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        summer_peak = 15.0 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
        winter_peak = 8.0 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        price = np.clip(45.0 + summer_peak + winter_peak + self.rng.normal(0, 3.0, n), 25.0, 80.0)
        return pd.DataFrame({"date": idx, "price": np.round(price, 2)})

    def generate_z5_gas_forward(self, start_date, end_date) -> pd.DataFrame:
        """Daily Transco Z5 prompt-month gas forward price ($/MMBtu)."""
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        contango = np.abs(self.rng.normal(0.15, 0.10, n))
        price = np.maximum(2.0, 3.5 + seasonal + contango + self.rng.normal(0, 0.2, n))
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_gas_storage(self, start_date, end_date) -> pd.DataFrame:
        """Weekly working gas in underground storage (Lower 48, Bcf).

        Seasonal pattern:
        - Injection season (Apr-Oct): storage builds ~1,500 Bcf total
        - Withdrawal season (Nov-Mar): storage draws down ~1,500 Bcf total
        - Range: 1,500-4,000 Bcf

        Returns DataFrame with columns: date, storage_bcf, storage_delta_bcf
        """
        idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="W-FRI")
        n = len(idx)
        # Seasonal storage level: peaks in late Oct (~doy 300, ~3,800 Bcf),
        # troughs in late Mar (~doy 90, ~1,700 Bcf).
        # Phase offset 209 gives sin peak at doy 300.
        doy = idx.dayofyear
        storage = 2750.0 + 1050.0 * np.sin(2 * np.pi * (doy - 209) / 365)
        storage = np.clip(storage + self.rng.normal(0, 80, n), 1500.0, 4000.0)
        # Week-over-week change (positive = injection/bearish, negative = withdrawal/bullish)
        delta = np.diff(storage, prepend=storage[0])
        # Inject ~Apr-Oct (doy 90-300), withdraw Nov-Mar
        return pd.DataFrame({
            "date": idx,
            "storage_bcf": np.round(storage, 1),
            "storage_delta_bcf": np.round(delta, 1),
        })

    def generate_dominion_south(self, start_date, end_date) -> pd.DataFrame:
        """Daily Dominion South Point gas spot price ($/MMBtu).

        Tracks Transco Z5 with a -$0.30 to -$0.80 discount reflecting
        Appalachian production surplus depressing local prices.
        Range: $1.50-$7.50/MMBtu.
        """
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        z5_base = 3.5 + seasonal + self.rng.normal(0, 0.2, n)
        # Appalachian discount: -$0.30 to -$0.80 below Z5
        discount = self.rng.uniform(-0.80, -0.30, n)
        price = np.clip(z5_base + discount, 1.50, 7.50)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    def generate_tetco_m3(self, start_date, end_date) -> pd.DataFrame:
        """Daily TETCO M3 gas spot price ($/MMBtu).

        Tracks Transco Z5 with ±$0.40 spread, wider in winter when
        pipeline constraints create regional pricing divergence.
        Range: $2.00-$8.50/MMBtu.
        """
        idx = self._daily_index(start_date, end_date)
        n = len(idx)
        seasonal = 0.5 * np.sin(2 * np.pi * (idx.dayofyear - 355) / 365)
        z5_base = 3.5 + seasonal + self.rng.normal(0, 0.2, n)
        # Winter spread widens (cold weather, pipeline constraints)
        winter_factor = np.maximum(0, np.sin(2 * np.pi * (idx.dayofyear - 355) / 365))
        spread = self.rng.uniform(-0.40, 0.40, n) * (1 + winter_factor)
        price = np.clip(z5_base + spread, 2.00, 8.50)
        return pd.DataFrame({"date": idx, "price": np.round(price, 3)})

    # ------------------------------------------------------------------
    # generate_all
    # ------------------------------------------------------------------

    def generate_all(self, start_date, end_date) -> dict:
        """Generate all datasets and return as a dict."""
        return {
            "south_da": self.generate_pjm_da_lmp(start_date, end_date),
            "whub_da": self.generate_whub_da_lmp(start_date, end_date),
            "south_rt": self.generate_rt_lmp(start_date, end_date),
            "gas_price": self.generate_gas_price(start_date, end_date),
            "load_forecast": self.generate_load_forecast(start_date, end_date),
            "load_forecast_hist": self.generate_load_forecast(start_date, end_date),
            "metered_load": self.generate_metered_load(start_date, end_date),
            "weather_south": {
                city: self.generate_weather_forecast(start_date, end_date, city)
                for city in ["Richmond VA", "Norfolk VA", "Raleigh NC"]
            },
            "weather_whub": {
                city: self.generate_weather_forecast(start_date, end_date, city)
                for city in ["Pittsburgh PA", "Columbus OH"]
            },
            "humidity": {
                city: self.generate_humidity_forecast(start_date, end_date, city)
                for city in ["Richmond VA", "Norfolk VA", "Raleigh NC"]
            },
            "solar_forecast": self.generate_solar_forecast(start_date, end_date),
            "solar_forecast_d1": self.generate_solar_forecast(start_date, end_date),
            "solar_forecast_hist": self.generate_solar_forecast(start_date, end_date),
            "wind_forecast": self.generate_wind_forecast(start_date, end_date),
            "wind_forecast_hist": self.generate_wind_forecast(start_date, end_date),
            "solar_actuals": self.generate_solar_actuals(start_date, end_date),
            "wind_actuals": self.generate_wind_actuals(start_date, end_date),
            "transmission_outages": self.generate_transmission_outages(start_date, end_date),
            "generator_outages": self.generate_generator_outages(start_date, end_date),
            "iso_prices": self.generate_iso_prices(start_date, end_date),
            "interchange": self.generate_interchange(start_date, end_date),
            "henry_hub": self.generate_henry_hub(start_date, end_date),
            "fuel_mix": self.generate_fuel_mix(start_date, end_date),
            "capacity": self.generate_installed_capacity(start_date, end_date),
            "bess_capacity": self.generate_bess_capacity(start_date, end_date),
            "virtual_bids": self.generate_virtual_bids(start_date, end_date),
            "emergency_logs": self.generate_emergency_logs(start_date, end_date),
            "rggi_price": self.generate_rggi_price(start_date, end_date),
            "demand_response": self.generate_demand_response(start_date, end_date),
            "coal_price": self.generate_coal_price(start_date, end_date),
            # New PJM feeds
            "ancillary_prices": self.generate_ancillary_prices(start_date, end_date),
            "emission_rates": self.generate_emission_rates(start_date, end_date),
            "tx_ratings": self.generate_tx_ratings(start_date, end_date),
            "instantaneous_load": self.generate_instantaneous_load(start_date, end_date),
            "transmission_constraints": self.generate_transmission_constraints(start_date, end_date),
            # New Morningstar feeds
            "columbia_gas": self.generate_columbia_gas(start_date, end_date),
            "whub_forward": self.generate_whub_forward(start_date, end_date),
            "z5_gas_forward": self.generate_z5_gas_forward(start_date, end_date),
            "dominion_south": self.generate_dominion_south(start_date, end_date),
            "tetco_m3": self.generate_tetco_m3(start_date, end_date),
            # EIA feeds
            "gas_storage": self.generate_gas_storage(start_date, end_date),
        }
