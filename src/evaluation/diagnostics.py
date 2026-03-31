"""Model diagnostics: SHAP plots, residual analysis, hourly MAE plots."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    """Generates diagnostic plots and statistics for model evaluation."""

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_shap_plots(self, model, X: pd.DataFrame, output_dir: Optional[str] = None) -> str:
        """Generate SHAP feature importance plots.

        Args:
            model: Trained LightGBMForecaster (must have .model attribute)
            X: Feature DataFrame to explain
            output_dir: Override output directory

        Returns:
            Path to saved plot
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        plot_path = str(out / "shap_importance.png")

        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X.fillna(0))

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X.fillna(0), show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

        return plot_path

    def plot_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        output_dir: Optional[str] = None,
    ) -> str:
        """Plot residuals (y_true - y_pred).

        Returns:
            Path to saved plot
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        plot_path = str(out / "residuals.png")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            residuals = y_true - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
            axes[0].axhline(0, color="red", linestyle="--")
            axes[0].set_xlabel("Predicted LMP ($/MWh)")
            axes[0].set_ylabel("Residual ($/MWh)")
            axes[0].set_title("Residuals vs Predicted")

            axes[1].hist(residuals, bins=50, edgecolor="black")
            axes[1].axvline(0, color="red", linestyle="--")
            axes[1].set_xlabel("Residual ($/MWh)")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Residual Distribution (MAE = {np.mean(np.abs(residuals)):.2f})")

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Residual plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Residual plot failed: {e}")

        return plot_path

    def plot_hourly_mae(
        self,
        results: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> str:
        """Plot MAE by hour of day.

        Args:
            results: DataFrame with columns: hour_ending, y_true, y_pred

        Returns:
            Path to saved plot
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        plot_path = str(out / "hourly_mae.png")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            hourly_maes = []
            for he in range(1, 25):
                mask = results["hour_ending"] == he
                if mask.sum() > 0:
                    m = float(np.mean(np.abs(
                        results.loc[mask, "y_true"].values -
                        results.loc[mask, "y_pred"].values
                    )))
                    hourly_maes.append({"hour_ending": he, "mae": m})

            if not hourly_maes:
                return plot_path

            df_mae = pd.DataFrame(hourly_maes)
            plt.figure(figsize=(12, 5))
            plt.bar(df_mae["hour_ending"], df_mae["mae"], color="steelblue", edgecolor="black")
            plt.xlabel("Hour Ending (EPT)")
            plt.ylabel("MAE ($/MWh)")
            plt.title("Forecast MAE by Hour of Day — PJM SOUTH DA LMP")
            plt.xticks(range(1, 25))
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Hourly MAE plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Hourly MAE plot failed: {e}")

        return plot_path
