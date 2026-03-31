"""Ensemble model combining LightGBM + Ridge forecasts.

Final_LMP_h = alpha_h * LightGBM_h + (1 - alpha_h) * Ridge_h
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

from .lightgbm_model import LightGBMForecaster
from .ridge_model import RidgeForecaster
from .spike_classifier import SpikeClassifier
from ..data.calendar_utils import CalendarUtils

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """Ensemble of LightGBM + Ridge with spike classifier overlay."""

    DEFAULT_ALPHAS = {
        "onpeak_weekday": 0.75,
        "offpeak_weekday": 0.70,
        "weekend_holiday": 0.65,
    }

    def __init__(
        self,
        lgbm_forecaster: Optional[LightGBMForecaster] = None,
        ridge_forecaster: Optional[RidgeForecaster] = None,
        spike_classifier: Optional[SpikeClassifier] = None,
        alphas: Optional[Dict[str, float]] = None,
        config_path=None,
    ):
        self.lgbm = lgbm_forecaster or LightGBMForecaster(config_path=config_path)
        self.ridge = ridge_forecaster or RidgeForecaster(config_path=config_path)
        self.spike_clf = spike_classifier or SpikeClassifier()
        self.alphas = alphas or self.DEFAULT_ALPHAS.copy()
        self.cal = CalendarUtils(config_path)
        self.config_path = config_path

    def _get_alpha(self, is_onpeak: bool, is_weekend_or_holiday: bool) -> float:
        """Get blending alpha for a given hour type."""
        if is_weekend_or_holiday:
            return self.alphas["weekend_holiday"]
        elif is_onpeak:
            return self.alphas["onpeak_weekday"]
        else:
            return self.alphas["offpeak_weekday"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleForecaster":
        """Train all sub-models."""
        logger.info("Training LightGBM forecaster...")
        self.lgbm.fit(X, y)

        logger.info("Training Ridge forecaster...")
        self.ridge.fit(X, y)

        y_spike = (y > self.spike_clf.spike_threshold).astype(int)
        if y_spike.sum() > 0:
            logger.info(f"Training spike classifier ({y_spike.sum()} spike events)...")
            self.spike_clf.fit(X, y_spike)

        return self

    def predict(
        self,
        X: pd.DataFrame,
        target_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """Generate ensemble forecast for 24 hours."""
        if self.lgbm.model is None:
            raise RuntimeError("Models not trained. Call fit() first.")

        lgbm_pred, lower, upper = self.lgbm.predict_with_ci(X)
        ridge_pred = self.ridge.predict(X)

        try:
            spike_probs = self.spike_clf.predict_proba(X)
        except Exception:
            spike_probs = np.zeros(len(X))

        forecasts = []
        alphas_used = []

        for i, row in enumerate(X.itertuples()):
            is_onpeak = bool(getattr(row, "is_onpeak", 0))
            is_weekend = bool(getattr(row, "is_weekend", 0))
            is_holiday = bool(getattr(row, "is_holiday", 0))
            is_we_hol = is_weekend or is_holiday

            alpha = self._get_alpha(is_onpeak, is_we_hol)
            blended = alpha * lgbm_pred[i] + (1 - alpha) * ridge_pred[i]

            if spike_probs[i] > 0.6:
                adjustment = 1.0 + 0.3 * (spike_probs[i] - 0.6) / 0.4
                blended *= adjustment

            forecasts.append(blended)
            alphas_used.append(alpha)

        hour_endings = list(range(1, len(forecasts) + 1))
        return pd.DataFrame({
            "hour_ending": hour_endings,
            "forecast": forecasts,
            "lgbm_forecast": lgbm_pred,
            "ridge_forecast": ridge_pred,
            "lower_90": lower,
            "upper_90": upper,
            "spike_prob": spike_probs,
            "alpha": alphas_used,
        })

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "EnsembleForecaster":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
