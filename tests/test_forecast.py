"""Tests for the ForecastEngine."""

import pytest
import pandas as pd
import numpy as np


def test_forecast_engine_rule_based():
    """ForecastEngine without trained model uses rule-based fallback."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()  # No model_path → untrained model
    result = engine.forecast(
        target_date="2024-06-15",
        whub_onpeak=45.0,
        whub_offpeak=30.0,
        gas_price=3.5,
    )
    assert result is not None
    assert len(result) == 24


def test_forecast_output_columns():
    """Forecast output has all required columns."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    required = ["Hour_EPT", "Forecast_LMP", "Lower_90", "Upper_90",
                "Spike_Risk", "WHub_DA", "Is_OnPeak"]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_forecast_24_hours():
    """Forecast produces exactly 24 hours HE01-HE24."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    assert len(result) == 24
    hour_labels = result["Hour_EPT"].tolist()
    expected = [f"HE{he:02d}" for he in range(1, 25)]
    assert hour_labels == expected


def test_forecast_lmp_reasonable():
    """Forecast LMP values should be in a reasonable range."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    # All prices should be positive and below some extreme threshold
    assert (result["Forecast_LMP"] > 0).all()
    assert (result["Forecast_LMP"] < 500).all()  # No extreme spikes in rule-based


def test_forecast_ci_valid():
    """Lower_90 <= Forecast_LMP <= Upper_90 for all hours."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    assert (result["Lower_90"] <= result["Forecast_LMP"]).all(), \
        "Lower_90 should be <= Forecast_LMP"
    assert (result["Forecast_LMP"] <= result["Upper_90"]).all(), \
        "Forecast_LMP should be <= Upper_90"


def test_forecast_spike_risk_labels():
    """Spike risk labels are valid."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    valid_labels = {"Low", "Moderate", "High", "Very High"}
    assert set(result["Spike_Risk"].unique()).issubset(valid_labels)


def test_forecast_onpeak_labels():
    """Is_OnPeak labels are either 'On-Peak' or 'Off-Peak'."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-06-15", 45.0, 30.0, 3.5)
    valid = {"On-Peak", "Off-Peak"}
    assert set(result["Is_OnPeak"].unique()).issubset(valid)


def test_forecast_whub_da_matches_inputs():
    """WHub_DA column reflects the on-peak/off-peak inputs provided."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast("2024-01-08", 50.0, 30.0, 3.5)  # Monday

    # On-peak hours should show WHub_DA = 50.0
    onpeak_rows = result[result["Is_OnPeak"] == "On-Peak"]
    assert not onpeak_rows.empty
    # Allow small floating point difference
    assert (abs(onpeak_rows["WHub_DA"] - 50.0) < 0.01).all()


def test_forecast_metrics():
    """Evaluation metrics functions work correctly."""
    from src.evaluation.metrics import mae, rmse, mape, bias, coverage, summary_metrics

    y_true = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
    y_pred = np.array([42.0, 43.0, 52.0, 54.0, 58.0])
    lower = y_pred - 5.0
    upper = y_pred + 5.0

    assert mae(y_true, y_pred) > 0
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)  # RMSE >= MAE
    assert mape(y_true, y_pred) > 0
    assert isinstance(bias(y_true, y_pred), float)

    cov = coverage(y_true, lower, upper)
    assert 0 <= cov <= 1

    metrics = summary_metrics(y_true, y_pred, lower=lower, upper=upper)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "bias" in metrics
    assert "coverage_90" in metrics
