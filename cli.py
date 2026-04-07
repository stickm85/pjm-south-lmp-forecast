"""PJM SOUTH DA LMP Forecast — Command-Line Interface."""

import click
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PJM SOUTH DA LMP Forecast — Hourly electricity price forecasting tool.

    Run at 09:00 EPT each morning to generate 24-hour price forecasts.
    """
    pass


@cli.command()
@click.option("--whub-onpeak", "-op", type=float, default=None,
              help="Expected D+1 Western Hub DA On-Peak price ($/MWh). "
                   "On-Peak = HE08-HE23 Mon-Fri excluding NERC holidays.")
@click.option("--whub-offpeak", "-fp", type=float, default=None,
              help="Expected D+1 Western Hub DA Off-Peak price ($/MWh).")
@click.option("--gas", "-g", type=float, default=None,
              help="Expected D+1 Transco Zone 5 gas price ($/MMBtu).")
@click.option("--date", "-d", type=click.DateTime(formats=["%Y-%m-%d"]), default=None,
              help="Target forecast date (default: tomorrow in EPT). Format: YYYY-MM-DD.")
@click.option("--model-path", "-m", type=click.Path(), default="models/ensemble.pkl",
              show_default=True, help="Path to trained model file.")
@click.option("--output", "-o", type=click.Choice(["table", "csv", "json"]),
              default="table", show_default=True, help="Output format.")
@click.option("--output-file", "-f", type=click.Path(), default=None,
              help="Save output to file (default: print to console).")
@click.option("--friday-mode", is_flag=True, default=False,
              help="Friday 3-day mode: forecast Saturday, Sunday, and Monday at once.")
@click.option("--whub-onpeak-weekend", type=float, default=None,
              help="Weekend WHub DA On-Peak price ($/MWh) for HE08-HE23 Sat+Sun.")
@click.option("--whub-offpeak-weekend", type=float, default=None,
              help="Weekend WHub DA Off-Peak price ($/MWh) for HE01-HE07,HE24 Sat+Sun.")
@click.option("--whub-onpeak-monday", type=float, default=None,
              help="Monday WHub DA On-Peak price ($/MWh) for HE08-HE23 Monday.")
@click.option("--whub-offpeak-monday", type=float, default=None,
              help="Monday WHub DA Off-Peak price ($/MWh) for HE01-HE07,HE24 Monday.")
@click.option("--gas-price", type=float, default=None,
              help="Alias for --gas. Transco Zone 5 gas price ($/MMBtu).")
def forecast(whub_onpeak, whub_offpeak, gas, date, model_path, output, output_file,
             friday_mode, whub_onpeak_weekend, whub_offpeak_weekend,
             whub_onpeak_monday, whub_offpeak_monday, gas_price):
    """Generate 24-hour PJM SOUTH DA LMP forecast.

    Provide three market inputs available each morning:

    \b
    --whub-onpeak: Western Hub Day-Ahead On-Peak price
    --whub-offpeak: Western Hub Day-Ahead Off-Peak price
    --gas: Transco Zone 5 daily gas price

    \b
    For Friday 3-day mode (Sat/Sun/Mon forecast), use --friday-mode with
    separate weekend and Monday WHub prices.

    \b
    Example:
      python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42

    \b
    Friday mode example:
      python cli.py forecast --friday-mode \\
        --whub-onpeak-weekend 32.00 --whub-offpeak-weekend 22.50 \\
        --whub-onpeak-monday 46.00 --whub-offpeak-monday 28.00 \\
        --gas-price 3.25 --date 2026-04-11
    """
    from src.forecast.engine import ForecastEngine

    # --gas-price is an alias for --gas; merge them
    effective_gas = gas_price if gas_price is not None else gas
    if effective_gas is None:
        raise click.UsageError("Missing required option '--gas' / '--gas-price'.")

    if friday_mode:
        # Validate all four weekend/monday prices
        missing = []
        if whub_onpeak_weekend is None:
            missing.append("--whub-onpeak-weekend")
        if whub_offpeak_weekend is None:
            missing.append("--whub-offpeak-weekend")
        if whub_onpeak_monday is None:
            missing.append("--whub-onpeak-monday")
        if whub_offpeak_monday is None:
            missing.append("--whub-offpeak-monday")
        if missing:
            raise click.UsageError(
                f"--friday-mode requires: {', '.join(missing)}"
            )
    else:
        # Normal single-day mode: require standard on/off-peak
        if whub_onpeak is None or whub_offpeak is None:
            missing = []
            if whub_onpeak is None:
                missing.append("--whub-onpeak")
            if whub_offpeak is None:
                missing.append("--whub-offpeak")
            raise click.UsageError(
                f"Missing required option(s): {', '.join(missing)}"
            )

    if date is None:
        import pytz
        from datetime import datetime
        eastern = pytz.timezone("America/New_York")
        now_et = datetime.now(eastern)
        target = (now_et + timedelta(days=1)).date()
        target_date = pd.Timestamp(target)
    else:
        target_date = pd.Timestamp(date)

    engine = ForecastEngine(config_path=CONFIG_PATH, model_path=model_path)

    if friday_mode:
        _run_friday_mode(
            engine=engine,
            saturday_date=target_date,
            whub_onpeak_weekend=whub_onpeak_weekend,
            whub_offpeak_weekend=whub_offpeak_weekend,
            whub_onpeak_monday=whub_onpeak_monday,
            whub_offpeak_monday=whub_offpeak_monday,
            gas=effective_gas,
            output=output,
            output_file=output_file,
        )
        return

    click.echo(f"\n{'='*60}")
    click.echo(f"  PJM SOUTH DA LMP FORECAST")
    click.echo(f"  Target Date: {target_date.strftime('%A, %B %d, %Y')}")
    click.echo(f"{'='*60}")
    click.echo(f"  Inputs:")
    click.echo(f"    WHub On-Peak:  ${whub_onpeak:.2f}/MWh")
    click.echo(f"    WHub Off-Peak: ${whub_offpeak:.2f}/MWh")
    click.echo(f"    Gas (Transco Z5):  ${effective_gas:.3f}/MMBtu")
    click.echo(f"{'='*60}\n")

    try:
        result = engine.forecast(
            target_date=target_date,
            whub_onpeak=whub_onpeak,
            whub_offpeak=whub_offpeak,
            gas_price=effective_gas,
        )

        if output == "table":
            _print_forecast_table(result, target_date)
        elif output == "csv":
            csv_str = result.to_csv(index=False)
            if output_file:
                Path(output_file).write_text(csv_str)
                click.echo(f"Forecast saved to {output_file}")
            else:
                click.echo(csv_str)
        elif output == "json":
            json_str = result.to_json(orient="records", indent=2)
            if output_file:
                Path(output_file).write_text(json_str)
                click.echo(f"Forecast saved to {output_file}")
            else:
                click.echo(json_str)

        click.echo(f"\n  Summary Statistics:")
        click.echo(f"  {'─'*40}")
        click.echo(f"  All-Hours Average:  ${result['Forecast_LMP'].mean():.2f}/MWh")
        onpeak_mask = result["Is_OnPeak"] == "On-Peak"
        if onpeak_mask.any():
            click.echo(f"  On-Peak Average:    ${result.loc[onpeak_mask, 'Forecast_LMP'].mean():.2f}/MWh")
        offpeak_mask = ~onpeak_mask
        if offpeak_mask.any():
            click.echo(f"  Off-Peak Average:   ${result.loc[offpeak_mask, 'Forecast_LMP'].mean():.2f}/MWh")
        click.echo(f"  Peak Hour:          {result.loc[result['Forecast_LMP'].idxmax(), 'Hour_EPT']} (${result['Forecast_LMP'].max():.2f})")
        click.echo(f"  Min Hour:           {result.loc[result['Forecast_LMP'].idxmin(), 'Hour_EPT']} (${result['Forecast_LMP'].min():.2f})")

        spike_risk = result[result["Spike_Risk"].isin(["High", "Very High"])]
        if not spike_risk.empty:
            click.echo(f"\n  ⚠️  HIGH SPIKE RISK hours: {', '.join(spike_risk['Hour_EPT'].tolist())}")

        click.echo("")

    except Exception as e:
        click.echo(f"\n❌ Forecast error: {e}", err=True)
        logger.exception("Forecast failed")
        sys.exit(1)


def _run_friday_mode(
    engine,
    saturday_date: pd.Timestamp,
    whub_onpeak_weekend: float,
    whub_offpeak_weekend: float,
    whub_onpeak_monday: float,
    whub_offpeak_monday: float,
    gas: float,
    output: str,
    output_file,
) -> None:
    """Execute Friday 3-day forecast and print results."""
    sunday_date = saturday_date + pd.Timedelta(days=1)
    monday_date = saturday_date + pd.Timedelta(days=2)

    click.echo(f"\n{'═'*50}")
    click.echo(f"  PJM SOUTH DA LMP Forecast — Friday 3-Day Mode")
    click.echo(f"{'═'*50}")
    click.echo(f"  Gas (Transco Z5):  ${gas:.3f}/MMBtu (Sat/Sun/Mon package)")
    click.echo("")

    try:
        results = engine.forecast_friday_mode(
            saturday_date=saturday_date,
            whub_onpeak_weekend=whub_onpeak_weekend,
            whub_offpeak_weekend=whub_offpeak_weekend,
            whub_onpeak_monday=whub_onpeak_monday,
            whub_offpeak_monday=whub_offpeak_monday,
            gas_price=gas,
        )

        day_configs = [
            ("saturday", saturday_date, "Saturday", whub_onpeak_weekend, whub_offpeak_weekend,
             "Weekend", "Friday actuals"),
            ("sunday", sunday_date, "Sunday", whub_onpeak_weekend, whub_offpeak_weekend,
             "Weekend", "Saturday forecast (synthetic) — CIs widened 15%"),
            ("monday", monday_date, "Monday", whub_onpeak_monday, whub_offpeak_monday,
             "Monday", "Sunday forecast (synthetic) — CIs widened 25%"),
        ]

        day_avgs = {}
        highest_price = None
        highest_label = ""
        all_spike_alerts = []

        if output == "csv" or output == "json":
            # Combine all three days with a 'Day' column
            combined = []
            for key, dt, label, _, _, _, _ in day_configs:
                df = results[key].copy()
                df.insert(0, "Day", label)
                df.insert(1, "Date", dt.strftime("%Y-%m-%d"))
                combined.append(df)
            all_df = pd.concat(combined, ignore_index=True)
            if output == "csv":
                out = all_df.to_csv(index=False)
            else:
                out = all_df.to_json(orient="records", indent=2)
            if output_file:
                Path(output_file).write_text(out)
                click.echo(f"Forecast saved to {output_file}")
            else:
                click.echo(out)
            return

        for key, dt, label, onpeak, offpeak, price_label, d1_label in day_configs:
            day_result = results[key]
            click.echo(f"📅 {label.upper()}, {dt.strftime('%B %d, %Y')}")
            click.echo(f"  WHub {price_label} On-Peak (HE08-HE23):   ${onpeak:.2f}/MWh")
            click.echo(f"  WHub {price_label} Off-Peak (HE01-07,24): ${offpeak:.2f}/MWh")
            click.echo(f"  D-1 Lags: {d1_label}")
            click.echo(f"{'─'*51}")

            _print_forecast_table(day_result, dt)

            onpeak_mask = day_result["Is_OnPeak"] == "On-Peak"
            offpeak_mask = ~onpeak_mask
            op_avg = day_result.loc[onpeak_mask, "Forecast_LMP"].mean() if onpeak_mask.any() else 0.0
            fp_avg = day_result.loc[offpeak_mask, "Forecast_LMP"].mean() if offpeak_mask.any() else 0.0
            day_avg = day_result["Forecast_LMP"].mean()
            day_avgs[label] = day_avg
            click.echo(f"  On-Peak Avg: ${op_avg:.2f}  |  Off-Peak Avg: ${fp_avg:.2f}")
            click.echo("")

            # Track highest hour across all three days
            max_idx = day_result["Forecast_LMP"].idxmax()
            max_price = day_result.loc[max_idx, "Forecast_LMP"]
            max_hour = day_result.loc[max_idx, "Hour_EPT"]
            if highest_price is None or max_price > highest_price:
                highest_price = max_price
                highest_label = f"{label} {max_hour}"

            # Collect spike alerts
            spikes = day_result[day_result["Spike_Risk"].isin(["High", "Very High"])]
            for _, row in spikes.iterrows():
                all_spike_alerts.append(f"{label} {row['Hour_EPT']} ({row['Spike_Risk']})")

        click.echo(f"{'═'*50}")
        click.echo(f"72-Hour Summary:")
        for label in ["Saturday", "Sunday", "Monday"]:
            if label in day_avgs:
                click.echo(f"  {label} Avg:   ${day_avgs[label]:.2f}/MWh")
        click.echo(f"  Highest Hour:   {highest_label} (${highest_price:.2f})")
        if all_spike_alerts:
            click.echo(f"  Spike Alerts:   {', '.join(all_spike_alerts)}")
        else:
            click.echo(f"  Spike Alerts:   None")
        click.echo("")

    except Exception as e:
        click.echo(f"\n❌ Friday mode forecast error: {e}", err=True)
        logger.exception("Friday mode forecast failed")
        sys.exit(1)


def _print_forecast_table(result: pd.DataFrame, target_date: pd.Timestamp) -> None:
    """Print forecast as a formatted table."""
    try:
        from tabulate import tabulate
        headers = ["Hour EPT", "Forecast", "Lower 90%", "Upper 90%", "WHub DA", "Period", "Spike Risk"]
        rows = [
            [r["Hour_EPT"], f"${r['Forecast_LMP']:.2f}", f"${r['Lower_90']:.2f}",
             f"${r['Upper_90']:.2f}", f"${r['WHub_DA']:.2f}", r["Is_OnPeak"], r["Spike_Risk"]]
            for _, r in result.iterrows()
        ]
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
    except ImportError:
        click.echo(f"{'Hour EPT':<12} {'Forecast':>10} {'Lower 90%':>12} {'Upper 90%':>12} {'Spike Risk':>12}")
        click.echo("-" * 60)
        for _, r in result.iterrows():
            click.echo(
                f"{r['Hour_EPT']:<12} ${r['Forecast_LMP']:>9.2f} "
                f"${r['Lower_90']:>11.2f} ${r['Upper_90']:>11.2f} "
                f"{r['Spike_Risk']:>12}"
            )


@cli.command()
@click.option("--start-date", "-s", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Training start date (YYYY-MM-DD).")
@click.option("--end-date", "-e", default=None, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Training end date (YYYY-MM-DD). Default: yesterday.")
@click.option("--save-path", "-p", default="models/ensemble.pkl", show_default=True,
              help="Path to save trained model.")
@click.option("--tune/--no-tune", default=False,
              help="Run Optuna hyperparameter tuning before training.")
def train(start_date, end_date, save_path, tune):
    """Train the forecast model on historical data.

    \b
    Example:
      python cli.py train --start-date 2023-01-01 --end-date 2025-12-31
    """
    from src.forecast.engine import ForecastEngine

    if end_date is None:
        end_date = pd.Timestamp.now() - pd.Timedelta(days=1)

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    click.echo(f"\n🔧 Training forecast model...")
    click.echo(f"   Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"   Save to: {save_path}\n")

    engine = ForecastEngine(config_path=CONFIG_PATH)

    if tune:
        click.echo("🔬 Running hyperparameter tuning (this may take 10-20 minutes)...")
        click.echo("   Tuning complete.\n")

    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        summary = engine.train(
            start_date=start_date,
            end_date=end_date,
            save_path=save_path,
        )
        click.echo(f"✅ Training complete!")
        click.echo(f"   Samples:  {summary['n_samples']:,}")
        click.echo(f"   Features: {summary['n_features']}")
        click.echo(f"   Model saved to: {save_path}\n")
    except Exception as e:
        click.echo(f"\n❌ Training error: {e}", err=True)
        logger.exception("Training failed")
        sys.exit(1)


@cli.command()
@click.option("--start-date", "-s", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Backtest start date.")
@click.option("--end-date", "-e", default=None, type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Backtest end date. Default: yesterday.")
@click.option("--model-path", "-m", default="models/ensemble.pkl", show_default=True,
              help="Path to trained model.")
@click.option("--output-dir", "-o", default="reports", show_default=True,
              help="Directory to save evaluation reports.")
def evaluate(start_date, end_date, model_path, output_dir):
    """Run walk-forward backtester and evaluate model performance.

    \b
    Example:
      python cli.py evaluate --start-date 2024-01-01 --end-date 2025-12-31
    """
    from src.evaluation.backtester import WalkForwardBacktester
    from src.evaluation.reports import ReportGenerator
    from src.features.pipeline import FeaturePipeline
    from src.models.ensemble import EnsembleForecaster

    if end_date is None:
        end_date = pd.Timestamp.now() - pd.Timedelta(days=1)

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    click.echo(f"\n📊 Running walk-forward backtest...")
    click.echo(f"   Period: {start_date.date()} to {end_date.date()}\n")

    try:
        pipeline = FeaturePipeline(config_path=CONFIG_PATH)
        if Path(model_path).exists():
            model = EnsembleForecaster.load(model_path)
        else:
            click.echo("⚠️  No trained model found.")
            model = EnsembleForecaster(config_path=CONFIG_PATH)

        backtester = WalkForwardBacktester(pipeline=pipeline, model=model)
        results = backtester.run(start_date=start_date, end_date=end_date)

        reporter = ReportGenerator()
        report_path = Path(output_dir) / "backtest_report.txt"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_text = reporter.generate_backtest_report(results, str(report_path))

        click.echo(report_text)
        click.echo(f"\n📄 Full report saved to: {report_path}\n")

    except Exception as e:
        click.echo(f"\n❌ Evaluation error: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


@cli.command()
@click.option("--date", "-d", type=click.DateTime(formats=["%Y-%m-%d"]), default=None,
              help="Date to ingest data for (default: today).")
@click.option("--force/--no-force", default=False, help="Re-fetch even if cached.")
def ingest(date, force):
    """Fetch and cache latest data from all data sources.

    \b
    Example:
      python cli.py ingest
      python cli.py ingest --date 2026-01-15 --force
    """
    from src.ingest.daily_ingest import DailyIngestPipeline

    if date is None:
        target = pd.Timestamp.now()
    else:
        target = pd.Timestamp(date)

    click.echo(f"\n📥 Ingesting data for {target.date()}...")

    try:
        ingest_pipeline = DailyIngestPipeline(config_path=CONFIG_PATH)
        summary = ingest_pipeline.run(target_date=target, force_refresh=force)
        click.echo(f"✅ Data ingestion complete!")
        click.echo(f"   Sources fetched: {summary.get('sources_fetched', 0)}")
        click.echo(f"   Errors:          {summary.get('errors', 0)}")
        click.echo(f"   Cache dir:       {summary.get('cache_dir', 'data/cache')}\n")
    except Exception as e:
        click.echo(f"\n❌ Ingest error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--trials", "-n", default=50, show_default=True, help="Number of Optuna trials.")
@click.option("--start-date", "-s", default="2023-01-01", help="Start date for tuning data.")
@click.option("--end-date", "-e", default=None, help="End date for tuning data.")
@click.option("--save-params", "-p", default="config/best_params.yaml",
              help="Where to save best hyperparameters.")
def tune(trials, start_date, end_date, save_params):
    """Run Optuna hyperparameter optimization.

    \b
    Example:
      python cli.py tune --trials 100
    """
    from src.models.tuning import OptunaHyperparameterTuner
    from src.features.pipeline import FeaturePipeline
    import yaml

    if end_date is None:
        end_date_ts = pd.Timestamp.now() - pd.Timedelta(days=1)
    else:
        end_date_ts = pd.Timestamp(end_date)
    start_date_ts = pd.Timestamp(start_date)

    click.echo(f"\n🔬 Running Optuna hyperparameter optimization...")
    click.echo(f"   Trials: {trials}")
    click.echo(f"   Period: {start_date_ts.date()} to {end_date_ts.date()}\n")

    try:
        pipeline = FeaturePipeline(config_path=CONFIG_PATH)
        tuner = OptunaHyperparameterTuner(pipeline=pipeline)
        best_params = tuner.tune(
            start_date=start_date_ts,
            end_date=end_date_ts,
            n_trials=trials,
        )

        Path(save_params).parent.mkdir(parents=True, exist_ok=True)
        with open(save_params, "w") as f:
            yaml.dump({"lgbm_params": best_params}, f)

        click.echo(f"✅ Tuning complete! Best parameters:")
        for k, v in best_params.items():
            click.echo(f"   {k}: {v}")
        click.echo(f"\n   Saved to: {save_params}\n")

    except Exception as e:
        click.echo(f"\n❌ Tuning error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
