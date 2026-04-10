"""Optuna hyperparameter tuning for LightGBM model."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available — hyperparameter tuning disabled")


class OptunaHyperparameterTuner:
    """Optuna-based hyperparameter tuning for LightGBM.

    Uses walk-forward cross-validation to avoid data leakage.
    Optimizes for out-of-sample MAE.
    """

    def __init__(self, pipeline=None, n_trials: int = 50):
        self.pipeline = pipeline
        self.n_trials = n_trials
        self.best_params: Optional[Dict] = None
        self.study = None

    def tune(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        n_trials: Optional[int] = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ) -> Dict:
        """Run Optuna hyperparameter optimization.

        Args:
            start_date: Training period start (used with pipeline to generate data)
            end_date: Training period end
            n_trials: Number of optimization trials (overrides constructor value)
            X_train: Pre-built feature matrix (optional, uses pipeline if None)
            y_train: Target series (optional)

        Returns:
            Dict of best hyperparameters
        """
        if not _OPTUNA_AVAILABLE:
            logger.warning("Optuna not available — returning default parameters")
            return self._default_params()

        n_trials = n_trials or self.n_trials
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Build training data if not provided
        if X_train is None or y_train is None:
            X_train, y_train = self._build_training_data(start_date, end_date)

        if len(X_train) == 0:
            logger.warning("No training data available — returning defaults")
            return self._default_params()

        def objective(trial):
            params = {
                "objective": "mae",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42,
            }
            return self._walk_forward_cv(X_train, y_train, params)

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = self.study.best_params
        self.best_params["objective"] = "mae"
        self.best_params["n_jobs"] = -1
        self.best_params["verbose"] = -1
        self.best_params["random_state"] = 42

        logger.info(f"Tuning complete. Best MAE: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        return self.best_params

    def _walk_forward_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict,
        n_folds: int = 3,
    ) -> float:
        """Walk-forward cross-validation returning mean OOS MAE."""
        import lightgbm as lgb
        n = len(X)
        fold_size = n // (n_folds + 1)

        if fold_size < 24:
            # Not enough data for proper CV — use a simple train/test split
            split = int(n * 0.8)
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[:split], y.iloc[:split])
            pred = model.predict(X.iloc[split:])
            return float(np.mean(np.abs(y.iloc[split:].values - pred)))

        maes = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, n)
            X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
            X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

            if len(X_te) == 0:
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            maes.append(float(np.mean(np.abs(y_te.values - pred))))

        return float(np.mean(maes)) if maes else 1e9

    def _build_training_data(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        """Build training data using the pipeline."""
        if self.pipeline is None:
            return pd.DataFrame(), pd.Series(dtype=float)

        from ..data.mock_data import MockDataGenerator
        mock = MockDataGenerator()
        hist = mock.generate_all(start_date, end_date)
        south_da = hist["south_da"]

        features_list = []
        targets = []
        rng = np.random.default_rng(42)
        current = start_date + pd.Timedelta(days=30)

        while current <= end_date:
            try:
                day_actuals = south_da[south_da["datetime"].dt.date == current.date()]
                if len(day_actuals) < 24:
                    current += pd.Timedelta(days=1)
                    continue

                feats = self.pipeline.build(
                    current,
                    whub_onpeak=float(rng.normal(45, 10)),
                    whub_offpeak=float(rng.normal(30, 7)),
                    gas_price=max(1.5, float(rng.normal(3.5, 0.5))),
                    historical_data=hist,
                )
                exclude = {"datetime", "hour_ending"}
                feat_cols = [c for c in feats.columns if c not in exclude]
                X_day = feats[feat_cols].fillna(0)
                y_day = day_actuals.sort_values("datetime")["lmp"].values[:24]

                if len(y_day) == 24 and len(X_day) == 24:
                    features_list.append(X_day)
                    targets.extend(y_day)
            except Exception as e:
                logger.debug(f"Skipping {current.date()} in tuning: {e}")
            current += pd.Timedelta(days=1)

        if not features_list:
            return pd.DataFrame(), pd.Series(dtype=float)

        return pd.concat(features_list, ignore_index=True), pd.Series(targets)

    def _default_params(self) -> Dict:
        return {
            "objective": "mae",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
