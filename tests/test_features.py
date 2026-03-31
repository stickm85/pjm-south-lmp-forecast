"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import date


def test_calendar_utils_onpeak():
    """Verify on-peak classification for weekday HE07-HE23."""
    from src.data.calendar_utils import CalendarUtils
    cal = CalendarUtils()

    # Monday 2024-01-08, HE10 → on-peak
    dt_onpeak = pd.Timestamp("2024-01-08 10:00")
    assert cal.is_onpeak(dt_onpeak) is True

    # Monday 2024-01-08, HE02 → off-peak (early morning)
    dt_offpeak_early = pd.Timestamp("2024-01-08 02:00")
    assert cal.is_onpeak(dt_offpeak_early) is False

    # Saturday 2024-01-06, HE10 → off-peak (weekend)
    dt_weekend = pd.Timestamp("2024-01-06 10:00")
    assert cal.is_onpeak(dt_weekend) is False

    # Monday HE07 boundary
    dt_he07 = pd.Timestamp("2024-01-08 07:00")
    assert cal.is_onpeak(dt_he07) is True

    # Monday HE23 boundary
    dt_he23 = pd.Timestamp("2024-01-08 23:00")
    assert cal.is_onpeak(dt_he23) is True

    # Monday HE24 (hour=0 next day) → off-peak
    dt_he24 = pd.Timestamp("2024-01-09 00:00")
    assert cal.is_onpeak(dt_he24) is False


def test_calendar_utils_nerc_holiday():
    """Verify NERC holiday detection."""
    from src.data.calendar_utils import CalendarUtils
    cal = CalendarUtils()

    # 2024 Labor Day (Monday) — should be NERC holiday
    assert cal.is_nerc_holiday(date(2024, 9, 2)) is True

    # Normal Monday — not a holiday
    assert cal.is_nerc_holiday(date(2024, 9, 9)) is False

    # 2024 Thanksgiving
    assert cal.is_nerc_holiday(date(2024, 11, 28)) is True

    # 2024 Christmas
    assert cal.is_nerc_holiday(date(2024, 12, 25)) is True


def test_calendar_utils_holiday_offpeak():
    """NERC holiday weekday should be classified as off-peak."""
    from src.data.calendar_utils import CalendarUtils
    cal = CalendarUtils()

    # 2024 Labor Day Monday HE10 — holiday, so off-peak
    dt_holiday = pd.Timestamp("2024-09-02 10:00")
    assert cal.is_onpeak(dt_holiday) is False


def test_user_input_expander_shape():
    """Verify UserInputExpander produces 24 rows."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    df = expander.expand("2024-06-15", 45.0, 30.0, 3.5)
    assert len(df) == 24
    assert "hour_ending" in df.columns
    assert "WHub_DA" in df.columns
    assert "Gas_Price" in df.columns
    assert "Spark_Spread_Gas" in df.columns
    assert "is_onpeak" in df.columns


def test_user_input_expander_onpeak_assignment():
    """On-peak hours use whub_onpeak, off-peak hours use whub_offpeak."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    # 2024-01-08 is a Monday (non-holiday)
    df = expander.expand("2024-01-08", whub_onpeak=50.0, whub_offpeak=30.0, gas_price=3.5)

    # HE10 should be on-peak → WHub_DA = 50.0
    he10 = df[df["hour_ending"] == 10].iloc[0]
    assert he10["is_onpeak"] == 1
    assert abs(he10["WHub_DA"] - 50.0) < 1e-6

    # HE02 should be off-peak → WHub_DA = 30.0
    he02 = df[df["hour_ending"] == 2].iloc[0]
    assert he02["is_onpeak"] == 0
    assert abs(he02["WHub_DA"] - 30.0) < 1e-6


def test_user_input_expander_spark_spread():
    """Spark spread = WHub - heat_rate * gas."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    df = expander.expand("2024-01-08", 50.0, 30.0, 4.0)

    # HE10 on-peak: Spark_Spread_Gas = 50 - 7*4 = 22
    he10 = df[df["hour_ending"] == 10].iloc[0]
    assert abs(he10["Spark_Spread_Gas"] - 22.0) < 1e-6

    # HE02 off-peak: Spark_Spread_Gas = 30 - 7*4 = 2
    he02 = df[df["hour_ending"] == 2].iloc[0]
    assert abs(he02["Spark_Spread_Gas"] - 2.0) < 1e-6


def test_temporal_features_shape():
    """Verify temporal features have 24 rows and correct columns."""
    from src.features.temporal_features import TemporalFeatureBuilder
    builder = TemporalFeatureBuilder()
    df = builder.build("2024-06-15")
    assert len(df) == 24
    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns
    assert "month_sin" in df.columns
    assert "month_cos" in df.columns
    assert "day_of_week" in df.columns
    assert "days_since_epoch" in df.columns


def test_temporal_features_cyclical_encoding():
    """Verify cyclical encoding is in [-1, 1] range."""
    from src.features.temporal_features import TemporalFeatureBuilder
    builder = TemporalFeatureBuilder()
    df = builder.build("2024-06-15")
    assert df["hour_sin"].between(-1.01, 1.01).all()
    assert df["hour_cos"].between(-1.01, 1.01).all()
    assert df["month_sin"].between(-1.01, 1.01).all()
    assert df["month_cos"].between(-1.01, 1.01).all()


def test_lag_features_shape():
    """Verify lag features produce 24 rows."""
    from src.features.lags import LagFeatureBuilder
    from src.data.mock_data import MockDataGenerator
    mock = MockDataGenerator()
    target = pd.Timestamp("2024-06-15")
    start = target - pd.Timedelta(days=14)
    end = target - pd.Timedelta(days=1)
    builder = LagFeatureBuilder()
    df = builder.build(
        target,
        mock.generate_pjm_da_lmp(start, end),
        mock.generate_whub_da_lmp(start, end),
        mock.generate_rt_lmp(start, end),
    )
    assert len(df) == 24
    assert "Basis_D1" in df.columns
    assert "DA_RT_Spread_D1" in df.columns
    assert "Congestion_D1" in df.columns


def test_mock_data_generator():
    """Verify mock data generator produces correct shapes."""
    from src.data.mock_data import MockDataGenerator
    mock = MockDataGenerator()
    start = "2024-01-01"
    end = "2024-01-07"

    da = mock.generate_pjm_da_lmp(start, end)
    assert "datetime" in da.columns
    assert "lmp" in da.columns
    assert "congestion" in da.columns
    assert "loss" in da.columns
    assert len(da) == 7 * 24  # 7 days * 24 hours

    gas = mock.generate_gas_price(start, end)
    assert "date" in gas.columns
    assert "gas_price" in gas.columns
    assert (gas["gas_price"] > 0).all()
