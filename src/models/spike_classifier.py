"""LightGBM binary classifier for LMP spike events."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

DEFAULT_SPIKE_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
    "scale_pos_weight": 10,  # Account for class imbalance
}


class SpikeClassifier:
    """Binary classifier predicting LMP spike events (> threshold $/MWh).

    Returns per-hour spike probability which is used by the ensemble
    to adjust forecasts upward in high-risk hours.
    """

    def __init__(
        self,
        spike_threshold: float = 100.0,
        params: Optional[Dict] = None,
    ):
        self.spike_threshold = spike_threshold
        self.params = params or DEFAULT_SPIKE_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: Optional[list] = None
        self._trained = False

    def fit(self, X: pd.DataFrame, y_binary: pd.Series) -> "SpikeClassifier":
        """Train the spike classifier.

        Args:
            X: Feature DataFrame
            y_binary: Binary target (1 if LMP > spike_threshold, else 0)
        """
        self.feature_names_ = list(X.columns)

        if y_binary.sum() == 0:
            logger.warning("No spike events in training data — classifier will predict all zeros")
            self._trained = False
            return self

        logger.info(f"Training spike classifier on {len(X)} samples ({y_binary.sum()} spikes)")
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X.fillna(0), y_binary)
        self._trained = True
        logger.info("Spike classifier training complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of spike for each sample.

        Returns array of shape (n_samples,) with values in [0, 1].
        """
        if not self._trained or self.model is None:
            return np.zeros(len(X))
        proba = self.model.predict_proba(X.fillna(0))
        return proba[:, 1]  # Probability of class 1 (spike)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary spike predictions."""
        if not self._trained or self.model is None:
            return np.zeros(len(X), dtype=int)
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Save classifier to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SpikeClassifier":
        """Load classifier from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
