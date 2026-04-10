"""Daily data ingestion pipeline for ~30 data sources."""

import logging
import json
from datetime import date
from pathlib import Path
from typing import Optional, Union, Dict
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DailyIngestPipeline:
    """Orchestrates daily data fetching and caching for all data sources."""

    SOURCES = [
        "south_da_lmp", "whub_da_lmp", "south_rt_lmp",
        "gas_price", "henry_hub",
        "load_forecast", "metered_load",
        "weather_south", "weather_whub", "humidity",
        "solar_forecast", "wind_forecast",
        "solar_actuals", "wind_actuals",
        "transmission_outages", "generator_outages",
        "iso_prices", "interchange",
        "fuel_mix", "installed_capacity",
        "columbia_gas", "whub_forward", "z5_gas_forward",
        "ancillary_prices", "emission_rates",
        "instantaneous_load", "transmission_constraints",
        "dominion_south", "tetco_m3",
        "gas_storage",
    ]

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.cache_dir = Path(self.config.get("data", {}).get("cache_dir", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        from ..data.pjm_client import PJMClient
        from ..data.gas_client import GasClient
        from ..data.weather_client import WeatherClient
        from ..data.outage_client import OutageClient
        from ..data.iso_client import ISOClient
        from ..data.capacity_client import CapacityClient

        self.pjm = PJMClient(config_path)
        self.gas = GasClient(config_path)
        self.weather = WeatherClient(config_path)
        self.outage = OutageClient(config_path)
        self.iso = ISOClient(config_path)
        self.capacity = CapacityClient(config_path)

        from ..data.openmeteo_client import OpenMeteoClient
        from ..data.eia_client import EIAClient
        self.openmeteo = OpenMeteoClient()
        self.eia = EIAClient(config_path)

    def run(
        self,
        target_date: Union[str, pd.Timestamp, date] = None,
        force_refresh: bool = False,
    ) -> Dict:
        """Fetch and cache all data sources for target_date.

        Args:
            target_date: Date to ingest data for (default: today)
            force_refresh: Re-fetch even if cached

        Returns:
            Summary dict with sources_fetched, errors, cache_dir
        """
        if target_date is None:
            target_date = pd.Timestamp.now()
        target_date = pd.Timestamp(target_date)

        start = target_date - pd.Timedelta(days=7)
        end = target_date

        sources_fetched = 0
        errors = 0
        fetched_sources = []

        fetch_map = {
            "south_da_lmp": lambda: self.pjm.fetch_da_lmp(start, end),
            "whub_da_lmp": lambda: self.pjm.fetch_whub_da_lmp(start, end),
            "south_rt_lmp": lambda: self.pjm.fetch_rt_lmp(start, end),
            "load_forecast": lambda: self.pjm.fetch_load_forecast(start, end),
            "metered_load": lambda: self.pjm.fetch_metered_load(start, end),
            "solar_forecast": lambda: self.pjm.fetch_solar_forecast(start, end),
            "wind_forecast": lambda: self.pjm.fetch_wind_forecast(start, end),
            "solar_actuals": lambda: self.pjm.fetch_solar_actuals(start, end),
            "wind_actuals": lambda: self.pjm.fetch_wind_actuals(start, end),
            "interchange": lambda: self.pjm.fetch_interchange(start, end),
            "fuel_mix": lambda: self.pjm.fetch_fuel_mix(start, end),
            "gas_price": lambda: self.gas.fetch_transco_z5(start, end),
            "henry_hub": lambda: self.eia.fetch_henry_hub_spot(start, end),
            "columbia_gas": lambda: self.gas.fetch_columbia_gas(start, end),
            "whub_forward": lambda: self.gas.fetch_whub_forward(start, end),
            "z5_gas_forward": lambda: self.gas.fetch_z5_gas_forward(start, end),
            "transmission_outages": lambda: self.outage.fetch_transmission_outages(start, end),
            "generator_outages": lambda: self.outage.fetch_generator_outages(start, end),
            "iso_prices": lambda: self.iso.fetch_iso_prices(start, end),
            "installed_capacity": lambda: self.capacity.fetch_installed_solar(start, end),
            # New PJM feeds
            "ancillary_prices": lambda: self.pjm.fetch_ancillary_prices(start, end),
            "emission_rates": lambda: self.pjm.fetch_emission_rates(start, end),
            "instantaneous_load": lambda: self.pjm.fetch_instantaneous_load(start, end),
            "transmission_constraints": lambda: self.pjm.fetch_transmission_constraints(start, end),
            # Regional gas price feeds (now via GasClient)
            "dominion_south": lambda: self.gas.fetch_dominion_south(start, end),
            "tetco_m3": lambda: self.gas.fetch_tetco_m3(start, end),
            # EIA feeds
            "gas_storage": lambda: self.eia.fetch_gas_storage(start, end),
        }

        for source_name, fetch_fn in fetch_map.items():
            cache_path = self.cache_dir / f"{source_name}_{target_date.date()}.parquet"

            if cache_path.exists() and not force_refresh:
                logger.debug(f"Cache hit: {source_name}")
                sources_fetched += 1
                fetched_sources.append(source_name)
                continue

            try:
                df = fetch_fn()
                if len(df) > 0:
                    df.to_parquet(cache_path, index=False)
                    sources_fetched += 1
                    fetched_sources.append(source_name)
                    logger.debug(f"Fetched and cached: {source_name} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Failed to fetch {source_name}: {e}")
                errors += 1

        # Weather data (per city)
        south_cities = ["Richmond VA", "Norfolk VA", "Raleigh NC"]
        whub_cities = ["Pittsburgh PA", "Columbus OH"]

        for city in south_cities + whub_cities:
            city_key = city.replace(" ", "_").lower()
            cache_path = self.cache_dir / f"weather_{city_key}_{target_date.date()}.parquet"
            if not cache_path.exists() or force_refresh:
                try:
                    df = self.weather.fetch_temperature(start, end, city)
                    df.to_parquet(cache_path, index=False)
                    sources_fetched += 1
                    fetched_sources.append(f"weather_{city_key}")
                except Exception as e:
                    logger.warning(f"Weather fetch failed for {city}: {e}")
                    errors += 1
            else:
                sources_fetched += 1

        # Open-Meteo data (all 6 variables, all cities combined)
        all_cities = south_cities + whub_cities
        openmeteo_cache = self.cache_dir / f"openmeteo_all_{target_date.date()}.parquet"
        if not openmeteo_cache.exists() or force_refresh:
            try:
                df = self.openmeteo.fetch_forecast(target_date, all_cities)
                if not df.empty:
                    df.to_parquet(openmeteo_cache, index=False)
                    sources_fetched += 1
                    fetched_sources.append("openmeteo_all")
                    logger.debug(f"Fetched and cached: openmeteo_all ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Open-Meteo fetch failed: {e} — falling back to mock data")
                from ..data.openmeteo_client import OpenMeteoMockData
                mock_om = OpenMeteoMockData()
                df = mock_om.generate_all_cities(start, end)
                df.to_parquet(openmeteo_cache, index=False)
                sources_fetched += 1
                fetched_sources.append("openmeteo_all_mock")
                errors += 1
        else:
            sources_fetched += 1

        return {
            "sources_fetched": sources_fetched,
            "errors": errors,
            "cache_dir": str(self.cache_dir),
            "target_date": str(target_date.date()),
            "fetched_sources": fetched_sources,
        }
