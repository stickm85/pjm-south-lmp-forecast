"""LightGBM model for PJM SOUTH DA LMP forecasting."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


class LightGBMForecaster:
    """LightGBM-based DA LMP forecaster with confidence intervals."""

    def __init__(self, params: Optional[Dict] = None, config_path=None):
        if config_path is not None:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                loaded = cfg.get("model", {}).get("lgbm_params", {})
                self.params = {**DEFAULT_PARAMS, **loaded}
            except Exception:
                self.params = params or DEFAULT_PARAMS.copy()
        else:
            self.params = params or DEFAULT_PARAMS.copy()

        self.model: Optional[lgb.LGBMRegressor] = None
        self.model_q05: Optional[lgb.LGBMRegressor] = None
        self.model_q95: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMForecaster":
        """Train the main MAE model and quantile models."""
        self.feature_names_ = list(X.columns)
        logger.info(f"Training LightGBM MAE model on {len(X)} samples, {len(X.columns)} features")

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

        q_params_lo = {**self.params, "objective": "quantile", "alpha": 0.05}
        self.model_q05 = lgb.LGBMRegressor(**q_params_lo)
        self.model_q05.fit(X, y)

        q_params_hi = {**self.params, "objective": "quantile", "alpha": 0.95}
        self.model_q95 = lgb.LGBMRegressor(**q_params_hi)
        self.model_q95.fit(X, y)

        logger.info("LightGBM training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_with_ci(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate point forecasts + 90% confidence interval."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        forecast = self.model.predict(X)
        lower = self.model_q05.predict(X)
        upper = self.model_q95.predict(X)
        return forecast, lower, upper

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMForecaster":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
