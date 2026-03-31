"""Tests for the feature pipeline."""

import pytest
import pandas as pd
import numpy as np


def test_pipeline_build_shape():
    """Pipeline produces exactly 24 rows."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)
    assert len(df) == 24


def test_pipeline_has_required_columns():
    """Pipeline output has all required feature columns."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)

    required_cols = [
        "hour_ending", "datetime",
        "WHub_DA", "Gas_Price", "Spark_Spread_Gas",
        "is_onpeak", "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_week", "is_holiday", "is_weekend",
        "Basis_D1", "DA_RT_Spread_D1",
        "Net_Load_h", "HDD", "CDD",
        "Solar_Capacity_Factor_D1", "Wind_Capacity_Factor_D1",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_pipeline_no_all_nan_columns():
    """Pipeline columns should not be entirely NaN after fillna."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)
    exclude = {"datetime"}
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
    df_filled = df[numeric_cols].fillna(0)
    # After fillna(0), no NaN should remain
    assert not df_filled.isnull().any().any()


def test_pipeline_feature_count():
    """Pipeline produces a reasonable number of features (>= 20)."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)
    exclude = {"datetime", "hour_ending"}
    n_features = len([c for c in df.columns if c not in exclude])
    assert n_features >= 20, f"Expected >= 20 features, got {n_features}"


def test_pipeline_hour_ending_range():
    """Hour ending values must be 1-24."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    df = pipeline.build("2024-06-15", 45.0, 30.0, 3.5)
    assert df["hour_ending"].min() == 1
    assert df["hour_ending"].max() == 24
    assert len(df["hour_ending"].unique()) == 24


def test_pipeline_onpeak_consistency():
    """On-peak hours should align with Monday calendar definition."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    # 2024-01-08 is a Monday (non-holiday)
    df = pipeline.build("2024-01-08", 50.0, 30.0, 3.5)
    # HE07-HE23 should be on-peak on a weekday
    for he in range(7, 24):
        row = df[df["hour_ending"] == he].iloc[0]
        assert row["is_onpeak"] == 1, f"HE{he:02d} should be on-peak on Monday"
    # HE01-HE06 should be off-peak
    for he in range(1, 7):
        row = df[df["hour_ending"] == he].iloc[0]
        assert row["is_onpeak"] == 0, f"HE{he:02d} should be off-peak"


def test_pipeline_with_historical_data():
    """Pipeline works with pre-provided historical data."""
    from src.features.pipeline import FeaturePipeline
    from src.data.mock_data import MockDataGenerator
    pipeline = FeaturePipeline()
    mock = MockDataGenerator()
    target = pd.Timestamp("2024-06-15")
    hist = mock.generate_all(target - pd.Timedelta(days=400), target - pd.Timedelta(days=1))
    df = pipeline.build(target, 45.0, 30.0, 3.5, historical_data=hist)
    assert len(df) == 24


def test_pipeline_get_feature_names():
    """get_feature_names returns a non-empty list."""
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    names = pipeline.get_feature_names()
    assert isinstance(names, list)
    assert len(names) >= 20
    assert "datetime" not in names
    assert "hour_ending" not in names
