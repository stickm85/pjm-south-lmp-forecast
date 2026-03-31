"""Report generation for forecast and backtest results."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Dict, Optional

from .metrics import summary_metrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates text reports for forecast and backtest results."""

    def generate_forecast_report(
        self,
        forecast_df: pd.DataFrame,
        target_date,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted forecast report.

        Args:
            forecast_df: Output from ForecastEngine.forecast()
            target_date: The forecast date
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        try:
            from tabulate import tabulate
            _has_tabulate = True
        except ImportError:
            _has_tabulate = False

        target_date = pd.Timestamp(target_date)
        lines = [
            "=" * 70,
            f"  PJM SOUTH DA LMP FORECAST REPORT",
            f"  Date: {target_date.strftime('%A, %B %d, %Y')}",
            "=" * 70,
            "",
        ]

        # Hourly table
        if _has_tabulate and len(forecast_df) > 0:
            cols = ["Hour_EPT", "Forecast_LMP", "Lower_90", "Upper_90",
                    "WHub_DA", "Is_OnPeak", "Spike_Risk"]
            available_cols = [c for c in cols if c in forecast_df.columns]
            table_data = forecast_df[available_cols].values.tolist()
            lines.append(tabulate(table_data, headers=available_cols, tablefmt="simple",
                                   floatfmt=".2f"))
        else:
            lines.append(forecast_df.to_string(index=False))

        lines.append("")
        lines.append("  Summary Statistics:")
        lines.append("  " + "-" * 40)

        if "Forecast_LMP" in forecast_df.columns:
            lines.append(f"  All-Hours Average:  ${forecast_df['Forecast_LMP'].mean():.2f}/MWh")
            onpeak_mask = forecast_df.get("Is_OnPeak", pd.Series(dtype=str)) == "On-Peak"
            if onpeak_mask.any():
                lines.append(f"  On-Peak Average:    ${forecast_df.loc[onpeak_mask, 'Forecast_LMP'].mean():.2f}/MWh")
            offpeak_mask = ~onpeak_mask
            if offpeak_mask.any():
                lines.append(f"  Off-Peak Average:   ${forecast_df.loc[offpeak_mask, 'Forecast_LMP'].mean():.2f}/MWh")
            peak_idx = forecast_df["Forecast_LMP"].idxmax()
            min_idx = forecast_df["Forecast_LMP"].idxmin()
            lines.append(f"  Peak Hour:          {forecast_df.loc[peak_idx, 'Hour_EPT']} (${forecast_df['Forecast_LMP'].max():.2f})")
            lines.append(f"  Min Hour:           {forecast_df.loc[min_idx, 'Hour_EPT']} (${forecast_df['Forecast_LMP'].min():.2f})")

        report = "\n".join(lines) + "\n"

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report)
            logger.info(f"Forecast report saved to {output_path}")

        return report

    def generate_backtest_report(
        self,
        backtest_results: Union[pd.DataFrame, Dict],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted backtest performance report.

        Args:
            backtest_results: DataFrame from WalkForwardBacktester.run()
                              or Dict from generate_report()
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        try:
            from tabulate import tabulate
            _has_tabulate = True
        except ImportError:
            _has_tabulate = False

        lines = [
            "=" * 70,
            "  PJM SOUTH DA LMP FORECAST — BACKTEST PERFORMANCE REPORT",
            "=" * 70,
            "",
        ]

        if isinstance(backtest_results, pd.DataFrame):
            df = backtest_results
            if len(df) == 0:
                lines.append("  No backtest results available.")
                report = "\n".join(lines) + "\n"
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_text(report)
                return report

            overall = summary_metrics(
                df["y_true"], df["y_pred"],
                lower=df.get("lower_90"), upper=df.get("upper_90")
            )
            n_days = df["date"].nunique() if "date" in df.columns else len(df) // 24

        elif isinstance(backtest_results, dict):
            overall = backtest_results.get("overall", {})
            n_days = backtest_results.get("n_days", 0)
            df = backtest_results.get("results_df", pd.DataFrame())
        else:
            lines.append("  Invalid backtest results format.")
            report = "\n".join(lines) + "\n"
            return report

        lines.append("  Overall Performance:")
        lines.append("  " + "-" * 40)
        for metric, value in overall.items():
            lines.append(f"  {metric.upper():<20} {value:.4f}")
        lines.append(f"  {'DAYS_EVALUATED':<20} {n_days}")
        lines.append("")

        # Hourly breakdown if results DataFrame available
        if isinstance(df, pd.DataFrame) and len(df) > 0 and "hour_ending" in df.columns:
            hourly_data = []
            for he in range(1, 25):
                mask = df["hour_ending"] == he
                if mask.sum() == 0:
                    continue
                he_mae = float(np.mean(np.abs(df.loc[mask, "y_true"] - df.loc[mask, "y_pred"])))
                he_rmse = float(np.sqrt(np.mean((df.loc[mask, "y_true"] - df.loc[mask, "y_pred"]) ** 2)))
                hourly_data.append([f"HE{he:02d}", f"{he_mae:.2f}", f"{he_rmse:.2f}"])

            lines.append("  Hourly Performance:")
            if _has_tabulate and hourly_data:
                lines.append(tabulate(hourly_data, headers=["Hour", "MAE", "RMSE"], tablefmt="simple"))
            else:
                for row in hourly_data:
                    lines.append(f"  {row[0]:<8} MAE={row[1]}  RMSE={row[2]}")

        report = "\n".join(lines) + "\n"

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report)
            logger.info(f"Backtest report saved to {output_path}")

        return report
