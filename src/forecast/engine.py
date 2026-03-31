"""Main forecast engine: orchestrates data collection, feature building, and prediction."""

import logging
from datetime import date
from pathlib import Path
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np
import yaml

from ..features.pipeline import FeaturePipeline
from ..models.ensemble import EnsembleForecaster
from ..data.mock_data import MockDataGenerator

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main engine for producing 24-hour PJM SOUTH DA LMP forecasts."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
    ):
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
        self.config_path = Path(config_path)

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.pipeline = FeaturePipeline(config_path=self.config_path)
        self.mock = MockDataGenerator()

        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            self.model = EnsembleForecaster.load(str(model_path))
        else:
            logger.info("No trained model found — initializing untrained model")
            self.model = EnsembleForecaster(config_path=self.config_path)

    def forecast(
        self,
        target_date: Union[str, date, pd.Timestamp],
        whub_onpeak: float,
        whub_offpeak: float,
        gas_price: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Generate 24-hour LMP forecast for target_date."""
        target_date = pd.Timestamp(target_date)
        logger.info(
            f"Running forecast for {target_date.date()} | "
            f"WHub On-Peak=${whub_onpeak:.2f} Off-Peak=${whub_offpeak:.2f} "
            f"Gas=${gas_price:.3f}"
        )

        features = self.pipeline.build(
            target_date, whub_onpeak, whub_offpeak, gas_price, historical_data
        )

        exclude_cols = {"datetime", "hour_ending"}
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        X = features[feature_cols].fillna(0)

        if self.model.lgbm.model is not None:
            result_df = self.model.predict(X, target_date)
            output = pd.DataFrame({
                "Hour_EPT": [f"HE{he:02d}" for he in range(1, 25)],
                "Forecast_LMP": result_df["forecast"].round(2).values,
                "Lower_90": result_df["lower_90"].round(2).values,
                "Upper_90": result_df["upper_90"].round(2).values,
                "Spike_Risk": self._spike_risk_label(result_df["spike_prob"]).values,
                "WHub_DA": features["WHub_DA"].round(2).values,
                "Is_OnPeak": features["is_onpeak"].map({1: "On-Peak", 0: "Off-Peak"}).values,
                "Spike_Prob_Pct": (result_df["spike_prob"] * 100).round(1).values,
            })
        else:
            logger.warning("Model not trained. Generating rule-based placeholder forecast.")
            output = self._rule_based_forecast(features, whub_onpeak, whub_offpeak, gas_price)

        return output

    def _spike_risk_label(self, spike_probs: pd.Series) -> pd.Series:
        """Convert spike probability to human-readable risk label."""
        def label(p):
            if p < 0.10:
                return "Low"
            elif p < 0.35:
                return "Moderate"
            elif p < 0.65:
                return "High"
            else:
                return "Very High"
        return spike_probs.apply(label)

    def _rule_based_forecast(
        self,
        features: pd.DataFrame,
        whub_onpeak: float,
        whub_offpeak: float,
        gas_price: float,
    ) -> pd.DataFrame:
        """Simple rule-based forecast when model is not trained."""
        rows = []
        for _, row in features.iterrows():
            he = int(row["hour_ending"])
            is_onpeak = bool(row["is_onpeak"])
            whub = row["WHub_DA"]
            basis = 3.0 if is_onpeak else 2.0
            forecast = whub + basis
            lower = forecast * 0.85
            upper = forecast * 1.15
            spark = whub - 7.0 * gas_price
            spike_prob = max(0.0, min(1.0, (1 / (1 + np.exp(-(spark - 20) / 10)))))
            rows.append({
                "Hour_EPT": f"HE{he:02d}",
                "Forecast_LMP": round(forecast, 2),
                "Lower_90": round(lower, 2),
                "Upper_90": round(upper, 2),
                "Spike_Risk": self._spike_risk_label(pd.Series([spike_prob]))[0],
                "WHub_DA": round(whub, 2),
                "Is_OnPeak": "On-Peak" if is_onpeak else "Off-Peak",
                "Spike_Prob_Pct": round(spike_prob * 100, 1),
            })
        return pd.DataFrame(rows)

    def train(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        save_path: Optional[str] = None,
    ) -> Dict:
        """Train the ensemble model on historical data."""
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        logger.info(f"Training on data from {start_date.date()} to {end_date.date()}")

        hist_data = self.mock.generate_all(start_date, end_date)
        south_da = hist_data["south_da"]

        all_features = []
        all_targets = []

        current = start_date + pd.Timedelta(days=30)
        while current <= end_date:
            try:
                day_actuals = south_da[south_da["datetime"].dt.date == current.date()]
                if len(day_actuals) < 24:
                    current += pd.Timedelta(days=1)
                    continue

                whub_onpeak = float(np.random.normal(45, 10))
                whub_offpeak = float(np.random.normal(30, 7))
                gas_price = max(1.0, float(np.random.normal(3.5, 0.5)))

                feats = self.pipeline.build(
                    current, whub_onpeak, whub_offpeak, gas_price, hist_data
                )
                exclude_cols = {"datetime", "hour_ending"}
                feature_cols = [c for c in feats.columns if c not in exclude_cols]
                X_day = feats[feature_cols].fillna(0)
                y_day = day_actuals.sort_values("datetime")["lmp"].values[:24]

                if len(y_day) == 24 and len(X_day) == 24:
                    all_features.append(X_day)
                    all_targets.extend(y_day)

            except Exception as e:
                logger.debug(f"Skipping {current.date()}: {e}")

            current += pd.Timedelta(days=1)

        if not all_features:
            raise ValueError("No training samples could be built")

        X_train = pd.concat(all_features, ignore_index=True)
        y_train = pd.Series(all_targets)

        logger.info(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
        self.model.fit(X_train, y_train)

        if save_path:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")

        return {
            "n_samples": len(X_train),
            "n_features": len(X_train.columns),
            "features": list(X_train.columns),
            "train_start": str(start_date.date()),
            "train_end": str(end_date.date()),
        }
