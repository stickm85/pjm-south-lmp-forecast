"""Main feature pipeline that assembles all feature modules into a single DataFrame."""

import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any, Optional, Union

from .user_inputs import UserInputExpander
from .lags import LagFeatureBuilder
from .load_features import LoadFeatureBuilder
from .weather_features import WeatherFeatureBuilder
from .renewable_features import RenewableFeatureBuilder
from .forecast_error import ForecastErrorBuilder
from .market_features import MarketFeatureBuilder
from .regime_features import RegimeFeatureBuilder
from .temporal_features import TemporalFeatureBuilder
from .enhanced_features import EnhancedFeatureBuilder
from ..data.mock_data import MockDataGenerator


class FeaturePipeline:
    """Assembles all features for 24-hour LMP forecast (51 original + 7 new = ~58+ features).

    Usage:
        pipeline = FeaturePipeline()
        features = pipeline.build(
            target_date="2026-01-15",
            whub_onpeak=45.0,
            whub_offpeak=30.0,
            gas_price=3.5,
            historical_data=historical_dict,  # or None for mock
        )
        # Returns 24-row DataFrame, one row per hour-ending
    """

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.user_input_expander = UserInputExpander(config_path)
        self.lag_builder = LagFeatureBuilder()
        self.load_builder = LoadFeatureBuilder()
        self.weather_builder = WeatherFeatureBuilder()
        self.renewable_builder = RenewableFeatureBuilder()
        self.error_builder = ForecastErrorBuilder()
        self.market_builder = MarketFeatureBuilder()
        self.regime_builder = RegimeFeatureBuilder()
        self.temporal_builder = TemporalFeatureBuilder(config_path)
        self.enhanced_builder = EnhancedFeatureBuilder()
        self.mock = MockDataGenerator()

    def build(
        self,
        target_date: Union[str, date, pd.Timestamp],
        whub_onpeak: float,
        whub_offpeak: float,
        gas_price: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build complete 24-row feature DataFrame for target_date."""
        target_date = pd.Timestamp(target_date)

        if historical_data is None:
            historical_data = self._get_mock_historical(target_date)

        # 1. User input features
        user_feats = self.user_input_expander.expand(
            target_date, whub_onpeak, whub_offpeak, gas_price
        )

        # 2. Temporal features
        temporal_feats = self.temporal_builder.build(target_date)

        # 3. Lag features
        lag_feats = self.lag_builder.build(
            target_date,
            historical_data.get("south_da", self.mock.generate_pjm_da_lmp(
                target_date - pd.Timedelta(days=30), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("whub_da", self.mock.generate_whub_da_lmp(
                target_date - pd.Timedelta(days=30), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("south_rt", self.mock.generate_rt_lmp(
                target_date - pd.Timedelta(days=30), target_date - pd.Timedelta(days=1)
            )),
        )

        # 4. Load features
        load_feats = self.load_builder.build(
            target_date,
            historical_data.get("load_forecast", self.mock.generate_load_forecast(
                target_date, target_date
            )),
            historical_data.get("solar_forecast", self.mock.generate_solar_forecast(
                target_date, target_date
            )),
            historical_data.get("wind_forecast", self.mock.generate_wind_forecast(
                target_date, target_date
            )),
            historical_data.get("metered_load", self.mock.generate_metered_load(
                target_date - pd.Timedelta(days=365), target_date - pd.Timedelta(days=1)
            )),
        )

        # 5. Weather features
        weather_feats = self.weather_builder.build(
            target_date,
            historical_data.get("weather_south", {
                city: self.mock.generate_weather_forecast(target_date, target_date, city)
                for city in ["Richmond VA", "Norfolk VA", "Raleigh NC"]
            }),
            historical_data.get("weather_whub", {
                city: self.mock.generate_weather_forecast(target_date, target_date, city)
                for city in ["Pittsburgh PA", "Columbus OH"]
            }),
            historical_data.get("humidity", {
                city: self.mock.generate_humidity_forecast(target_date, target_date, city)
                for city in ["Richmond VA", "Norfolk VA", "Raleigh NC"]
            }),
        )

        # 6. Renewable features
        renewable_feats = self.renewable_builder.build(
            target_date,
            historical_data.get("solar_actuals", self.mock.generate_solar_actuals(
                target_date - pd.Timedelta(days=2), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("wind_actuals", self.mock.generate_wind_actuals(
                target_date - pd.Timedelta(days=2), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("solar_forecast_d1", self.mock.generate_solar_forecast(
                target_date, target_date
            )),
            historical_data.get("capacity", self.mock.generate_installed_capacity(
                target_date - pd.Timedelta(days=30), target_date
            )),
        )

        # 7. Forecast error features
        error_feats = self.error_builder.build(
            target_date,
            historical_data.get("metered_load", self.mock.generate_metered_load(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("load_forecast_hist", self.mock.generate_load_forecast(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("solar_actuals", self.mock.generate_solar_actuals(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("solar_forecast_hist", self.mock.generate_solar_forecast(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("wind_actuals", self.mock.generate_wind_actuals(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("wind_forecast_hist", self.mock.generate_wind_forecast(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
        )

        # 8. Market features
        market_feats = self.market_builder.build(
            target_date,
            historical_data.get("virtual_bids", self.mock.generate_virtual_bids(
                target_date - pd.Timedelta(days=10), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("bess_capacity", self.mock.generate_bess_capacity(
                target_date - pd.Timedelta(days=90), target_date
            )),
            historical_data.get("south_da", self.mock.generate_pjm_da_lmp(
                target_date - pd.Timedelta(days=2), target_date - pd.Timedelta(days=1)
            )),
        )

        # 9. Regime features
        regime_feats = self.regime_builder.build(
            target_date,
            historical_data.get("emergency_logs", self.mock.generate_emergency_logs(
                target_date - pd.Timedelta(days=5), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("metered_load", self.mock.generate_metered_load(
                target_date - pd.Timedelta(days=5), target_date - pd.Timedelta(days=1)
            )),
            historical_data.get("capacity_installed", self.mock.generate_generator_outages(
                target_date - pd.Timedelta(days=5), target_date - pd.Timedelta(days=1)
            )),
        )

        # 10. Enhanced features (Open-Meteo, Morningstar, PJM ancillary/emissions,
        #     transmission constraints, EIA gas storage, regional gas spreads)
        from ..data.openmeteo_client import OpenMeteoMockData
        openmeteo_mock = OpenMeteoMockData()
        openmeteo_data = historical_data.get(
            "openmeteo",
            openmeteo_mock.generate_all_cities(target_date, target_date),
        )
        enhanced_feats = self.enhanced_builder.build(
            target_date,
            openmeteo_data=openmeteo_data,
            columbia_gas=historical_data.get(
                "columbia_gas",
                self.mock.generate_columbia_gas(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            z5_spot=historical_data.get(
                "gas_price",
                self.mock.generate_gas_price(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            z5_forward=historical_data.get(
                "z5_gas_forward",
                self.mock.generate_z5_gas_forward(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            whub_forward=historical_data.get(
                "whub_forward",
                self.mock.generate_whub_forward(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            whub_spot_da=whub_onpeak,
            ancillary_prices=historical_data.get(
                "ancillary_prices",
                self.mock.generate_ancillary_prices(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            emission_rates=historical_data.get(
                "emission_rates",
                self.mock.generate_emission_rates(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            transmission_constraints=historical_data.get(
                "transmission_constraints",
                self.mock.generate_transmission_constraints(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            gas_storage=historical_data.get(
                "gas_storage",
                self.mock.generate_gas_storage(
                    target_date - pd.Timedelta(days=90), target_date - pd.Timedelta(days=1)
                ),
            ),
            dominion_south=historical_data.get(
                "dominion_south",
                self.mock.generate_dominion_south(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
            tetco_m3=historical_data.get(
                "tetco_m3",
                self.mock.generate_tetco_m3(
                    target_date - pd.Timedelta(days=7), target_date - pd.Timedelta(days=1)
                ),
            ),
        )

        # Merge all feature sets on hour_ending
        result = user_feats.copy()
        for df in [temporal_feats, lag_feats, load_feats, weather_feats,
                   renewable_feats, error_feats, market_feats, regime_feats,
                   enhanced_feats]:
            if "hour_ending" in df.columns:
                merge_cols = [c for c in df.columns if c != "datetime"]
                # Avoid duplicate columns
                new_cols = [c for c in merge_cols if c == "hour_ending" or c not in result.columns]
                result = result.merge(df[new_cols], on="hour_ending", how="left")

        return result

    def _get_mock_historical(self, target_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Generate mock historical data for all required datasets."""
        start = target_date - pd.Timedelta(days=400)
        end = target_date - pd.Timedelta(days=1)
        return self.mock.generate_all(start, end)

    def get_feature_names(self) -> list:
        """Return list of all feature column names (excluding datetime, hour_ending)."""
        sample = self.build("2024-06-15", 45.0, 30.0, 3.5)
        exclude = {"datetime", "hour_ending"}
        return [c for c in sample.columns if c not in exclude]
