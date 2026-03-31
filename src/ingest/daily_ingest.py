"""Daily data ingestion pipeline for all 28 data sources."""

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
        "fuel_mix", "installed_capacity", "bess_capacity",
        "virtual_bids", "emergency_logs", "rggi_price",
        "demand_response", "coal_price",
        "wind_cloud", "rggi_price",
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
        from ..data.market_client import MarketClient

        self.pjm = PJMClient(config_path)
        self.gas = GasClient(config_path)
        self.weather = WeatherClient(config_path)
        self.outage = OutageClient(config_path)
        self.iso = ISOClient(config_path)
        self.capacity = CapacityClient(config_path)
        self.market = MarketClient(config_path)

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
            "virtual_bids": lambda: self.pjm.fetch_virtual_bids(start, end),
            "gas_price": lambda: self.gas.fetch_transco_z6_nny(start, end),
            "henry_hub": lambda: self.gas.fetch_henry_hub(start, end),
            "transmission_outages": lambda: self.outage.fetch_transmission_outages(start, end),
            "generator_outages": lambda: self.outage.fetch_generator_outages(start, end),
            "iso_prices": lambda: self.iso.fetch_iso_prices(start, end),
            "installed_capacity": lambda: self.capacity.fetch_installed_solar(start, end),
            "bess_capacity": lambda: self.capacity.fetch_installed_bess(start, end),
            "rggi_price": lambda: self.market.fetch_rggi_price(start, end),
            "coal_price": lambda: self.market.fetch_coal_price(start, end),
            "emergency_logs": lambda: self.market.fetch_emergency_logs(start, end),
            "demand_response": lambda: self.market.fetch_demand_response(start, end),
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

        return {
            "sources_fetched": sources_fetched,
            "errors": errors,
            "cache_dir": str(self.cache_dir),
            "target_date": str(target_date.date()),
            "fetched_sources": fetched_sources,
        }
