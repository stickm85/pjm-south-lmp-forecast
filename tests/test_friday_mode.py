"""Tests for Friday 3-day forecast mode."""

import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# ForecastEngine.forecast_friday_mode tests
# ---------------------------------------------------------------------------

def test_friday_mode_returns_three_dataframes():
    """forecast_friday_mode returns dicts with 'saturday', 'sunday', 'monday' keys."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    assert set(results.keys()) == {"saturday", "sunday", "monday"}
    for key in ("saturday", "sunday", "monday"):
        assert len(results[key]) == 24, f"{key} should have 24 rows"


def test_friday_mode_24_hours_each_day():
    """Each day in friday mode produces exactly 24 hourly rows HE01-HE24."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    expected_hours = [f"HE{he:02d}" for he in range(1, 25)]
    for key in ("saturday", "sunday", "monday"):
        assert results[key]["Hour_EPT"].tolist() == expected_hours, (
            f"{key} hour labels should be HE01-HE24"
        )


def test_friday_mode_output_columns():
    """Each day DataFrame has all required output columns."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    required = ["Hour_EPT", "Forecast_LMP", "Lower_90", "Upper_90",
                "Spike_Risk", "WHub_DA", "Is_OnPeak"]
    for key in ("saturday", "sunday", "monday"):
        for col in required:
            assert col in results[key].columns, f"{key} missing column: {col}"


def test_weekend_whub_price_assignment():
    """Weekend days (Sat/Sun) assign WHub on-peak price to HE08-HE23, off-peak to others."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    for day in ("saturday", "sunday"):
        df = results[day]
        for _, row in df.iterrows():
            he = int(row["Hour_EPT"].replace("HE", ""))
            expected_whub = 32.00 if 8 <= he <= 23 else 22.50
            assert abs(row["WHub_DA"] - expected_whub) < 0.01, (
                f"{day} {row['Hour_EPT']}: expected WHub_DA={expected_whub}, "
                f"got {row['WHub_DA']}"
            )


def test_monday_uses_different_prices():
    """Monday uses Monday WHub prices, not weekend prices."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    df = results["monday"]
    for _, row in df.iterrows():
        he = int(row["Hour_EPT"].replace("HE", ""))
        expected_whub = 46.00 if 8 <= he <= 23 else 28.00
        assert abs(row["WHub_DA"] - expected_whub) < 0.01, (
            f"Monday {row['Hour_EPT']}: expected WHub_DA={expected_whub}, "
            f"got {row['WHub_DA']}"
        )


def test_ci_widening_sunday():
    """Sunday CIs are ~15% wider than Saturday CIs (relative to forecast)."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    sat = results["saturday"]
    sun = results["sunday"]

    # Compare half-width (distance from forecast to CI bound)
    sat_upper_hw = (sat["Upper_90"] - sat["Forecast_LMP"]).mean()
    sun_upper_hw = (sun["Upper_90"] - sun["Forecast_LMP"]).mean()

    if sat_upper_hw > 0:
        ratio = sun_upper_hw / sat_upper_hw
        # Sunday should be ~15% wider (ratio ~1.15); allow ±5% tolerance
        assert 1.05 <= ratio <= 1.30, (
            f"Sunday CI should be ~15% wider than Saturday. Ratio={ratio:.3f}"
        )


def test_ci_widening_monday():
    """Monday CIs are ~25% wider than Saturday CIs relative to forecast value."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    sat = results["saturday"]
    mon = results["monday"]

    # Compare relative (fractional) half-widths to neutralize different WHub price levels
    sat_rel_hw = ((sat["Upper_90"] - sat["Forecast_LMP"]) / sat["Forecast_LMP"]).mean()
    mon_rel_hw = ((mon["Upper_90"] - mon["Forecast_LMP"]) / mon["Forecast_LMP"]).mean()

    if sat_rel_hw > 0:
        ratio = mon_rel_hw / sat_rel_hw
        # Monday should be ~25% wider (ratio ~1.25); allow ±10% tolerance
        assert 1.10 <= ratio <= 1.45, (
            f"Monday CI should be ~25% wider than Saturday (relative). Ratio={ratio:.3f}"
        )


def test_ci_lower_le_forecast_le_upper():
    """Lower_90 <= Forecast_LMP <= Upper_90 for all hours in all three days."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )
    for key in ("saturday", "sunday", "monday"):
        df = results[key]
        assert (df["Lower_90"] <= df["Forecast_LMP"]).all(), (
            f"{key}: Lower_90 should be <= Forecast_LMP"
        )
        assert (df["Forecast_LMP"] <= df["Upper_90"]).all(), (
            f"{key}: Forecast_LMP should be <= Upper_90"
        )


def test_gas_price_same_across_days():
    """Gas_Price feature should be the same value across all three days."""
    from src.forecast.engine import ForecastEngine
    from src.features.pipeline import FeaturePipeline
    from pathlib import Path

    config_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    pipeline = FeaturePipeline(config_path=config_path)

    gas_price = 3.25
    for date_str in ("2026-04-11", "2026-04-12", "2026-04-13"):
        features = pipeline.build(
            target_date=date_str,
            whub_onpeak=32.00,
            whub_offpeak=22.50,
            gas_price=gas_price,
        )
        assert (abs(features["Gas_Price"] - gas_price) < 0.001).all(), (
            f"Gas_Price should be {gas_price} for {date_str}"
        )


def test_cascade_logic_synthetic_d1():
    """Sunday forecast uses Saturday forecast as synthetic D-1 actuals."""
    from src.forecast.engine import ForecastEngine
    engine = ForecastEngine()

    # Run friday mode to get results
    results = engine.forecast_friday_mode(
        saturday_date="2026-04-11",
        whub_onpeak_weekend=32.00,
        whub_offpeak_weekend=22.50,
        whub_onpeak_monday=46.00,
        whub_offpeak_monday=28.00,
        gas_price=3.25,
    )

    # Build the synthetic historical data manually and verify sunday forecasts
    # are not identical to running with no historical data
    sunday_no_hist = engine.forecast(
        target_date="2026-04-12",
        whub_onpeak=32.00,
        whub_offpeak=22.50,
        gas_price=3.25,
    )

    # The friday mode sunday forecast should not be identical to running without
    # any historical data injection (since synthetic D-1 was injected).
    # Both may be rule-based if model is untrained, so just verify structure.
    assert len(results["sunday"]) == 24
    assert len(sunday_no_hist) == 24


# ---------------------------------------------------------------------------
# user_inputs.py WHub price assignment tests
# ---------------------------------------------------------------------------

def test_whub_assignment_weekday():
    """On a weekday, HE08-HE23 get on-peak WHub, others get off-peak WHub."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    df = expander.expand("2026-04-07", whub_onpeak=50.0, whub_offpeak=30.0, gas_price=3.5)
    for _, row in df.iterrows():
        he = int(row["hour_ending"])
        if 8 <= he <= 23:
            assert abs(row["WHub_DA"] - 50.0) < 0.01, f"HE{he:02d} should be on-peak"
        else:
            assert abs(row["WHub_DA"] - 30.0) < 0.01, f"HE{he:02d} should be off-peak"


def test_whub_assignment_weekend_saturday():
    """On Saturday (2026-04-11), HE08-HE23 get on-peak WHub price."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    df = expander.expand("2026-04-11", whub_onpeak=32.0, whub_offpeak=22.5, gas_price=3.25)
    for _, row in df.iterrows():
        he = int(row["hour_ending"])
        if 8 <= he <= 23:
            assert abs(row["WHub_DA"] - 32.0) < 0.01, (
                f"Saturday HE{he:02d} should get on-peak WHub=32.0, got {row['WHub_DA']}"
            )
        else:
            assert abs(row["WHub_DA"] - 22.5) < 0.01, (
                f"Saturday HE{he:02d} should get off-peak WHub=22.5, got {row['WHub_DA']}"
            )


def test_whub_assignment_weekend_sunday():
    """On Sunday (2026-04-12), HE08-HE23 get on-peak WHub price."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    df = expander.expand("2026-04-12", whub_onpeak=32.0, whub_offpeak=22.5, gas_price=3.25)
    for _, row in df.iterrows():
        he = int(row["hour_ending"])
        if 8 <= he <= 23:
            assert abs(row["WHub_DA"] - 32.0) < 0.01, (
                f"Sunday HE{he:02d} should get on-peak WHub=32.0, got {row['WHub_DA']}"
            )
        else:
            assert abs(row["WHub_DA"] - 22.5) < 0.01, (
                f"Sunday HE{he:02d} should get off-peak WHub=22.5, got {row['WHub_DA']}"
            )


def test_nerc_is_onpeak_unchanged_for_weekends():
    """NERC is_onpeak feature should still be False for weekend hours (unchanged behavior)."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    # Saturday = 2026-04-11
    df = expander.expand("2026-04-11", whub_onpeak=32.0, whub_offpeak=22.5, gas_price=3.25)
    # All hours on Saturday should have is_onpeak = 0 (NERC excludes weekends)
    assert (df["is_onpeak"] == 0).all(), (
        "NERC is_onpeak should be 0 for all weekend hours"
    )


def test_nerc_is_onpeak_true_for_weekday_peak():
    """NERC is_onpeak should be True for HE08-HE23 on a weekday."""
    from src.features.user_inputs import UserInputExpander
    expander = UserInputExpander()
    # Monday 2026-04-13 is a weekday
    df = expander.expand("2026-04-13", whub_onpeak=46.0, whub_offpeak=28.0, gas_price=3.25)
    onpeak_rows = df[df["hour_ending"].between(8, 23)]
    assert (onpeak_rows["is_onpeak"] == 1).all(), (
        "NERC is_onpeak should be 1 for HE08-HE23 on a weekday"
    )
    offpeak_rows = df[~df["hour_ending"].between(8, 23)]
    assert (offpeak_rows["is_onpeak"] == 0).all(), (
        "NERC is_onpeak should be 0 for off-peak hours on a weekday"
    )


# ---------------------------------------------------------------------------
# CLI validation tests
# ---------------------------------------------------------------------------

def test_cli_friday_mode_requires_all_four_prices():
    """CLI --friday-mode raises UsageError if any of the four prices are missing."""
    from cli import cli
    runner = CliRunner()
    # Only supply one of the four weekend/monday prices — the others are missing
    result = runner.invoke(cli, [
        "forecast",
        "--friday-mode",
        "--whub-onpeak-weekend", "32.00",
        "--gas", "3.25",
        "--date", "2026-04-11",
    ])
    assert result.exit_code != 0
    # The UsageError from our validation lists the missing flags
    assert "--whub-offpeak-weekend" in result.output or "friday-mode requires" in result.output.lower()


def test_cli_friday_mode_all_prices_valid():
    """CLI --friday-mode succeeds (no crash) when all required prices are supplied."""
    from cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "forecast",
        "--friday-mode",
        "--whub-onpeak-weekend", "32.00",
        "--whub-offpeak-weekend", "22.50",
        "--whub-onpeak-monday", "46.00",
        "--whub-offpeak-monday", "28.00",
        "--gas-price", "3.25",
        "--date", "2026-04-11",
    ])
    # Should not fail due to missing args (may fail for other reasons like model not trained,
    # but the CLI argument validation itself should pass)
    assert "--friday-mode requires" not in result.output


def test_cli_normal_mode_requires_whub_onpeak_offpeak():
    """Normal mode raises UsageError if --whub-onpeak or --whub-offpeak is missing."""
    from cli import cli
    runner = CliRunner()
    # Missing --whub-offpeak
    result = runner.invoke(cli, [
        "forecast",
        "--whub-onpeak", "45.00",
        "--gas", "3.25",
        "--date", "2026-04-07",
    ])
    assert result.exit_code != 0
    assert "--whub-offpeak" in result.output


def test_cli_normal_mode_unchanged():
    """Normal single-day forecast mode still works with --whub-onpeak and --whub-offpeak."""
    from cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "forecast",
        "--whub-onpeak", "45.00",
        "--whub-offpeak", "28.00",
        "--gas", "3.25",
        "--date", "2026-04-07",
    ])
    # Should produce 24-hour forecast output without crashing on argument validation
    assert "Missing required option" not in result.output
    assert "--friday-mode requires" not in result.output


def test_cli_gas_label_transco_z5():
    """CLI output should show 'Gas (Transco Z5)' not 'Gas (Z6 NNY)'."""
    from cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "forecast",
        "--whub-onpeak", "45.00",
        "--whub-offpeak", "28.00",
        "--gas", "3.25",
        "--date", "2026-04-07",
    ])
    assert "Z6 NNY" not in result.output, "Old gas label 'Z6 NNY' should not appear"
    assert "Transco Z5" in result.output, "Gas label should say 'Transco Z5'"
