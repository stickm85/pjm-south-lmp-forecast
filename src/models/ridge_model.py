"""Ridge regression model for PJM SOUTH DA LMP forecasting."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional
import logging
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RidgeForecaster:
    """Ridge regression-based DA LMP forecaster.

    Uses sklearn Ridge with StandardScaler preprocessing.
    Best suited for capturing linear relationships between
    gas prices, hub prices, and load.
    """

    def __init__(self, alpha: float = 1.0, config_path=None):
        if config_path is not None:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                alpha = cfg.get("model", {}).get("ridge_alpha", alpha)
            except Exception:
                pass
        self.alpha = alpha
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        """Train the Ridge regression model.

        Args:
            X: Feature DataFrame
            y: Target Series (SOUTH DA LMP values)
        """
        self.feature_names_ = list(X.columns)
        logger.info(f"Training Ridge model (alpha={self.alpha}) on {len(X)} samples")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.pipeline.fit(X.fillna(0), y)
        logger.info("Ridge training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.pipeline.predict(X.fillna(0))

    def get_coefficients(self) -> pd.DataFrame:
        """Return feature coefficients."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        coefs = self.pipeline.named_steps["ridge"].coef_
        return pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": coefs,
        }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ridge model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RidgeForecaster":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
