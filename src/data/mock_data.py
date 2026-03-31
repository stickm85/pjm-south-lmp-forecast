"""Mock data generator for all 28 data sources used in PJM SOUTH LMP forecasting."""

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
        """Daily Transco Z6 NNY gas price ($/MMBtu)."""
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
            "capacity_installed": self.generate_generator_outages(start_date, end_date),
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
        }
