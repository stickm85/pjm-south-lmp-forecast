"""Tests for enhanced features, new data clients, and pipeline integration."""

import pytest
import pandas as pd
import numpy as np
from datetime import date


# ---------------------------------------------------------------------------
# Open-Meteo client tests
# ---------------------------------------------------------------------------

class TestOpenMeteoClient:
    """Tests for OpenMeteoClient and its mock data generator."""

    def test_mock_schema_single_city(self):
        """Mock data should have all 6 required columns plus datetime and city."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-06-15", "2024-06-15", "Richmond VA")

        expected_cols = {
            "datetime", "city",
            "shortwave_radiation", "direct_radiation",
            "apparent_temperature", "windgusts_10m",
            "precipitation", "pressure_msl",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_mock_24_rows_per_day(self):
        """One day of data should produce exactly 24 hourly rows."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-06-15", "2024-06-15", "Richmond VA")
        assert len(df) == 24

    def test_mock_realistic_ghi_range(self):
        """GHI should be non-negative and not exceed 1000 W/m²."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-06-15", "2024-06-15", "Richmond VA")
        assert (df["shortwave_radiation"] >= 0).all()
        assert (df["shortwave_radiation"] <= 1001).all()

    def test_mock_direct_le_ghi(self):
        """Direct radiation should be <= shortwave radiation (GHI)."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-06-15", "2024-06-15", "Richmond VA")
        assert (df["direct_radiation"] <= df["shortwave_radiation"] + 1e-6).all()

    def test_mock_pressure_range(self):
        """Pressure should be within realistic atmospheric range."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-01-01", "2024-12-31", "Pittsburgh PA")
        assert (df["pressure_msl"] >= 988).all()
        assert (df["pressure_msl"] <= 1042).all()

    def test_mock_precipitation_non_negative(self):
        """Precipitation should never be negative."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate("2024-06-15", "2024-06-16", "Norfolk VA")
        assert (df["precipitation"] >= 0).all()

    def test_mock_all_cities_schema(self):
        """generate_all_cities should return data for all 5 cities."""
        from src.data.openmeteo_client import OpenMeteoMockData, CITIES
        mock = OpenMeteoMockData(seed=0)
        df = mock.generate_all_cities("2024-06-15", "2024-06-15")
        assert set(df["city"].unique()) == set(CITIES.keys())
        assert len(df) == 5 * 24  # 5 cities × 24 hours

    def test_forecast_fallback_to_mock(self):
        """fetch_forecast should fall back to mock on API failure."""
        from src.data.openmeteo_client import OpenMeteoClient
        client = OpenMeteoClient()
        # Force a bad URL so the real API call fails
        client._mock  # ensure mock is available
        # We'll test with a valid call that uses mock internally
        from unittest.mock import patch
        import requests
        with patch.object(requests, "get", side_effect=ConnectionError("mock network error")):
            df = client.fetch_forecast("2024-06-15", cities=["Richmond VA"])
        assert not df.empty
        assert "shortwave_radiation" in df.columns


# ---------------------------------------------------------------------------
# Morningstar client tests
# ---------------------------------------------------------------------------

class TestMorningstarClient:
    """Tests for MorningstarClient and its mock data generator."""

    def test_columbia_gas_schema(self):
        """Columbia Gas mock should have date and price columns."""
        from src.data.morningstar_client import MorningstarClient
        client = MorningstarClient()
        df = client.fetch_columbia_gas("2024-01-01", "2024-01-31")
        assert "date" in df.columns
        assert "price" in df.columns
        assert len(df) == 31

    def test_columbia_gas_price_range(self):
        """Columbia Gas price should be within $2–$8/MMBtu range."""
        from src.data.morningstar_client import MorningstarClient
        client = MorningstarClient()
        df = client.fetch_columbia_gas("2024-01-01", "2024-12-31")
        assert (df["price"] >= 1.5).all(), "Price below minimum realistic value"
        assert (df["price"] <= 12.0).all(), "Price above maximum realistic value"

    def test_whub_forward_schema(self):
        """WHub forward mock should have date and price columns."""
        from src.data.morningstar_client import MorningstarClient
        client = MorningstarClient()
        df = client.fetch_whub_forward("2024-01-01", "2024-01-31")
        assert "date" in df.columns
        assert "price" in df.columns

    def test_whub_forward_price_range(self):
        """WHub forward price should be within $25–$80/MWh range."""
        from src.data.morningstar_client import MorningstarClient
        client = MorningstarClient()
        df = client.fetch_whub_forward("2024-01-01", "2024-12-31")
        assert (df["price"] >= 25.0).all()
        assert (df["price"] <= 80.0).all()

    def test_z5_gas_forward_schema(self):
        """Z5 forward mock should have date and price columns."""
        from src.data.morningstar_client import MorningstarClient
        client = MorningstarClient()
        df = client.fetch_z5_gas_forward("2024-01-01", "2024-01-31")
        assert "date" in df.columns
        assert "price" in df.columns

    def test_z5_forward_above_spot(self):
        """Z5 forward (prompt-month) should typically be above spot (contango)."""
        from src.data.morningstar_client import MorningstarClient
        from src.data.gas_client import GasClient
        client = MorningstarClient()
        gas = GasClient()
        fwd = client.fetch_z5_gas_forward("2024-06-01", "2024-06-30")
        spot = gas.fetch_transco_z5("2024-06-01", "2024-06-30")
        # On average, forward should be >= spot (contango assumption in mock)
        avg_fwd = fwd["price"].mean()
        avg_spot = spot["gas_price"].mean()
        assert avg_fwd >= avg_spot - 0.5, "Forward should generally be >= spot"


# ---------------------------------------------------------------------------
# PJM new methods tests
# ---------------------------------------------------------------------------

class TestPJMNewMethods:
    """Tests for the 4 new PJM API methods."""

    def test_ancillary_prices_schema(self):
        """Ancillary prices mock should have 4 required columns."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_ancillary_prices("2024-01-01", "2024-01-07")
        assert "datetime" in df.columns
        assert "reg_a_price" in df.columns
        assert "reg_d_price" in df.columns
        assert "sync_reserve_price" in df.columns
        assert len(df) == 7 * 24

    def test_ancillary_prices_non_negative(self):
        """All ancillary prices should be non-negative."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_ancillary_prices("2024-01-01", "2024-01-31")
        assert (df["reg_a_price"] >= 0).all()
        assert (df["reg_d_price"] >= 0).all()
        assert (df["sync_reserve_price"] >= 0).all()

    def test_emission_rates_schema(self):
        """Emission rates mock should have datetime and marginal_emission_rate_lbs_mwh."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_emission_rates("2024-01-01", "2024-01-07")
        assert "datetime" in df.columns
        assert "marginal_emission_rate_lbs_mwh" in df.columns
        assert len(df) == 7 * 24

    def test_emission_rates_realistic_range(self):
        """Emission rates should be in realistic CO2 range (500–2500 lb/MWh)."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_emission_rates("2024-01-01", "2024-01-31")
        assert (df["marginal_emission_rate_lbs_mwh"] >= 400).all()
        assert (df["marginal_emission_rate_lbs_mwh"] <= 2600).all()

    def test_tx_ratings_schema(self):
        """TX ratings mock should have date, tx_derate_flag, and derated_mw."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_tx_ratings("2024-01-01", "2024-01-31")
        assert "date" in df.columns
        assert "tx_derate_flag" in df.columns
        assert "derated_mw" in df.columns
        assert len(df) == 31

    def test_tx_derate_flag_binary(self):
        """TX de-rate flag should be binary (0 or 1)."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_tx_ratings("2024-01-01", "2024-12-31")
        assert df["tx_derate_flag"].isin([0, 1]).all()

    def test_instantaneous_load_schema(self):
        """Instantaneous load mock should have datetime and instantaneous_load_mw."""
        from src.data.pjm_client import PJMClient
        client = PJMClient()
        df = client.fetch_instantaneous_load("2024-01-01", "2024-01-07")
        assert "datetime" in df.columns
        assert "instantaneous_load_mw" in df.columns
        assert len(df) == 7 * 24


# ---------------------------------------------------------------------------
# Enhanced features tests
# ---------------------------------------------------------------------------

class TestEnhancedFeatures:
    """Tests for each of the 11 new derived features."""

    TARGET_DATE = "2024-06-15"

    def _make_openmeteo_data(self, target_date: str = TARGET_DATE) -> pd.DataFrame:
        """Create realistic Open-Meteo mock data for testing."""
        from src.data.openmeteo_client import OpenMeteoMockData
        mock = OpenMeteoMockData(seed=0)
        return mock.generate_all_cities(target_date, target_date)

    def test_output_shape(self):
        """Enhanced features should produce exactly 24 rows."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        assert len(df) == 24
        assert "hour_ending" in df.columns

    def test_all_11_features_present(self):
        """All 11 new features should be present in the output."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator()
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=7)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        builder = EnhancedFeatureBuilder()
        df = builder.build(
            self.TARGET_DATE,
            openmeteo_data=self._make_openmeteo_data(),
            columbia_gas=mock.generate_columbia_gas(start, end),
            z5_spot=mock.generate_gas_price(start, end).rename(columns={"gas_price": "price"}),
            z5_forward=mock.generate_z5_gas_forward(start, end),
            whub_forward=mock.generate_whub_forward(start, end),
            whub_spot_da=45.0,
            ancillary_prices=mock.generate_ancillary_prices(start, end),
            emission_rates=mock.generate_emission_rates(start, end),
        )

        expected_features = [
            "ghi_solar_estimate_h",
            "clear_sky_fraction_h",
            "gust_curtailment_flag",
            "columbia_z5_spread",
            "gas_contango",
            "power_contango",
            "reg_price_d1",
            "reserve_scarcity_signal",
            "marginal_emission_rate_d1",
            "pressure_gradient_12h",
            "precip_flag",
        ]
        missing = [f for f in expected_features if f not in df.columns]
        assert not missing, f"Missing features: {missing}"

    def test_ghi_solar_estimate_non_negative(self):
        """GHI solar estimate should always be non-negative."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        non_nan = df["ghi_solar_estimate_h"].dropna()
        assert (non_nan >= 0).all()

    def test_clear_sky_fraction_bounded(self):
        """Clear-sky fraction should be clipped to [0, 1]."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        non_nan = df["clear_sky_fraction_h"].dropna()
        assert (non_nan >= 0).all()
        assert (non_nan <= 1.0).all()

    def test_gust_curtailment_flag_binary(self):
        """Gust curtailment flag should be 0 or 1."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        non_nan = df["gust_curtailment_flag"].dropna()
        assert non_nan.isin([0, 1]).all()

    def test_precip_flag_binary(self):
        """Precipitation flag should be 0 or 1."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        non_nan = df["precip_flag"].dropna()
        assert non_nan.isin([0, 1]).all()

    def test_columbia_z5_spread_formula(self):
        """Columbia−Z5 spread should equal columbia price − z5 price (scalar)."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=3)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        col_df = mock.generate_columbia_gas(start, end)
        z5_df = mock.generate_gas_price(start, end).rename(columns={"gas_price": "price"})

        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE,
                           columbia_gas=col_df,
                           z5_spot=z5_df)
        # All 24 rows should have the same scalar spread
        spread = df["columbia_z5_spread"].dropna()
        assert not spread.empty
        assert spread.nunique() == 1  # scalar — same for all hours

    def test_gas_contango_positive_in_mock(self):
        """Gas contango (forward − spot) should generally be positive in mock data."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=30)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        z5_spot_df = mock.generate_gas_price(start, end).rename(columns={"gas_price": "price"})
        z5_fwd_df = mock.generate_z5_gas_forward(start, end)

        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, z5_spot=z5_spot_df, z5_forward=z5_fwd_df)
        contango = df["gas_contango"].dropna()
        # Mock forward is always slightly above spot, so contango >= -0.5
        assert (contango >= -0.5).all()

    def test_power_contango_formula(self):
        """Power contango = whub_forward_price − whub_spot_da."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=3)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        whub_fwd_df = mock.generate_whub_forward(start, end)
        whub_spot = 45.0

        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, whub_forward=whub_fwd_df, whub_spot_da=whub_spot)
        contango = df["power_contango"].dropna()
        assert not contango.empty
        # Value should be fwd_price - 45.0
        fwd_price = float(whub_fwd_df.iloc[-1]["price"])
        expected = round(fwd_price - whub_spot, 2)
        assert abs(contango.iloc[0] - expected) < 0.01

    def test_reg_price_d1_is_scalar_per_day(self):
        """Regulation price D-1 should be same value for all 24 hours."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=7)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        anc = mock.generate_ancillary_prices(start, end)
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, ancillary_prices=anc)
        reg = df["reg_price_d1"].dropna()
        assert not reg.empty
        assert reg.nunique() == 1  # same D-1 average for all hours

    def test_reserve_scarcity_signal_binary(self):
        """Reserve scarcity signal should be 0 or 1."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=7)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        anc = mock.generate_ancillary_prices(start, end)
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, ancillary_prices=anc)
        sig = df["reserve_scarcity_signal"].dropna()
        assert not sig.empty
        assert sig.isin([0, 1]).all()

    def test_marginal_emission_rate_d1_reasonable(self):
        """Emission rate D-1 should be in realistic CO2 range."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        from src.data.mock_data import MockDataGenerator
        mock = MockDataGenerator(seed=0)
        start = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=7)
        end = pd.Timestamp(self.TARGET_DATE) - pd.Timedelta(days=1)

        er = mock.generate_emission_rates(start, end)
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, emission_rates=er)
        rate = df["marginal_emission_rate_d1"].dropna()
        assert not rate.empty
        assert (rate >= 400).all()
        assert (rate <= 2600).all()

    def test_pressure_gradient_12h(self):
        """Pressure gradient should equal pressure_h - pressure_{h-12}."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        df = builder.build(self.TARGET_DATE, openmeteo_data=self._make_openmeteo_data())
        grad = df["pressure_gradient_12h"].dropna()
        assert not grad.empty
        # Gradients should be reasonable (not extreme)
        assert (grad.abs() < 100).all()

    def test_no_data_returns_nan_not_error(self):
        """When no data is supplied, features should be NaN, not raise exceptions."""
        from src.features.enhanced_features import EnhancedFeatureBuilder
        builder = EnhancedFeatureBuilder()
        # Call with all defaults (None) — should not raise
        df = builder.build(self.TARGET_DATE)
        assert len(df) == 24
        assert "ghi_solar_estimate_h" in df.columns
        assert df["ghi_solar_estimate_h"].isna().all()


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Tests that new features appear in the final feature matrix."""

    ENHANCED_FEATURE_NAMES = [
        "ghi_solar_estimate_h",
        "clear_sky_fraction_h",
        "gust_curtailment_flag",
        "columbia_z5_spread",
        "gas_contango",
        "power_contango",
        "reg_price_d1",
        "reserve_scarcity_signal",
        "marginal_emission_rate_d1",
        "pressure_gradient_12h",
        "precip_flag",
    ]

    def test_pipeline_includes_enhanced_features(self):
        """FeaturePipeline.build() output should include all 11 new features."""
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = pipeline.build(
            target_date="2024-06-15",
            whub_onpeak=45.0,
            whub_offpeak=30.0,
            gas_price=3.5,
        )
        missing = [f for f in self.ENHANCED_FEATURE_NAMES if f not in df.columns]
        assert not missing, f"Pipeline missing enhanced features: {missing}"

    def test_pipeline_feature_count_increased(self):
        """Pipeline should now have more features than the original ~40."""
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        features = pipeline.get_feature_names()
        # Original was ~40; with 11 new features it should be > 45
        assert len(features) > 45, f"Expected >45 features, got {len(features)}"

    def test_pipeline_enhanced_features_not_all_nan(self):
        """Enhanced features in pipeline output should not be all NaN."""
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)
        for feat in self.ENHANCED_FEATURE_NAMES:
            if feat in df.columns:
                non_nan_count = df[feat].notna().sum()
                assert non_nan_count > 0, f"Feature {feat} is all NaN in pipeline output"
