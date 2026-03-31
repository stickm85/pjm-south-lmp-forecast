"""Tests for model classes."""

import pytest
import pandas as pd
import numpy as np


def _make_training_data(n_days: int = 30, n_features: int = 10):
    """Generate simple training data for model tests."""
    np.random.seed(42)
    n = n_days * 24
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Add is_onpeak and is_weekend columns
    X["is_onpeak"] = np.tile([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], n_days)
    X["is_weekend"] = 0
    X["is_holiday"] = 0
    y = pd.Series(40.0 + 5 * np.random.randn(n))
    return X, y


def test_lightgbm_fit_predict():
    """LightGBM model fit and predict produce correct shapes."""
    from src.models.lightgbm_model import LightGBMForecaster

    X, y = _make_training_data(30)
    # Use fast params for testing
    params = {
        "objective": "mae", "n_estimators": 10, "learning_rate": 0.1,
        "num_leaves": 7, "n_jobs": 1, "verbose": -1, "random_state": 42,
    }
    model = LightGBMForecaster(params=params)
    model.fit(X, y)

    X_pred = X.iloc[:24]
    pred = model.predict(X_pred)
    assert pred.shape == (24,)
    assert not np.isnan(pred).any()


def test_lightgbm_predict_with_ci():
    """LightGBM predict_with_ci returns three arrays of same shape."""
    from src.models.lightgbm_model import LightGBMForecaster

    X, y = _make_training_data(30)
    params = {
        "objective": "mae", "n_estimators": 10, "learning_rate": 0.1,
        "num_leaves": 7, "n_jobs": 1, "verbose": -1, "random_state": 42,
    }
    model = LightGBMForecaster(params=params)
    model.fit(X, y)

    X_pred = X.iloc[:24]
    forecast, lower, upper = model.predict_with_ci(X_pred)
    assert forecast.shape == (24,)
    assert lower.shape == (24,)
    assert upper.shape == (24,)


def test_lightgbm_feature_importance():
    """Feature importance returns a DataFrame with correct columns."""
    from src.models.lightgbm_model import LightGBMForecaster

    X, y = _make_training_data(15)
    params = {"objective": "mae", "n_estimators": 5, "num_leaves": 7,
              "n_jobs": 1, "verbose": -1, "random_state": 42}
    model = LightGBMForecaster(params=params)
    model.fit(X, y)

    imp = model.get_feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == X.shape[1]


def test_ridge_fit_predict():
    """Ridge model fit and predict produce correct shapes."""
    from src.models.ridge_model import RidgeForecaster

    X, y = _make_training_data(30)
    model = RidgeForecaster(alpha=1.0)
    model.fit(X, y)

    X_pred = X.iloc[:24]
    pred = model.predict(X_pred)
    assert pred.shape == (24,)
    assert not np.isnan(pred).any()


def test_ridge_coefficients():
    """Ridge coefficients DataFrame has correct shape."""
    from src.models.ridge_model import RidgeForecaster

    X, y = _make_training_data(20)
    model = RidgeForecaster(alpha=1.0)
    model.fit(X, y)

    coefs = model.get_coefficients()
    assert "feature" in coefs.columns
    assert "coefficient" in coefs.columns
    assert len(coefs) == X.shape[1]


def test_spike_classifier_no_spikes():
    """Spike classifier with no spike events returns zeros."""
    from src.models.spike_classifier import SpikeClassifier

    X, y = _make_training_data(30)
    clf = SpikeClassifier(spike_threshold=100.0)
    y_binary = (y > 100.0).astype(int)
    clf.fit(X, y_binary)

    probs = clf.predict_proba(X.iloc[:24])
    assert probs.shape == (24,)
    assert (probs == 0.0).all()  # No spikes, should return all zeros


def test_spike_classifier_with_spikes():
    """Spike classifier trained with spikes returns valid probabilities."""
    from src.models.spike_classifier import SpikeClassifier

    np.random.seed(42)
    n = 500
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"feat_{i}" for i in range(5)])
    y_binary = pd.Series(np.random.randint(0, 2, n))

    clf = SpikeClassifier(spike_threshold=100.0)
    clf.fit(X, y_binary)

    probs = clf.predict_proba(X.iloc[:24])
    assert probs.shape == (24,)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_ensemble_predict_shape():
    """Ensemble model returns 24-row DataFrame."""
    from src.models.ensemble import EnsembleForecaster

    X, y = _make_training_data(30)
    lgbm_params = {"objective": "mae", "n_estimators": 10, "num_leaves": 7,
                   "n_jobs": 1, "verbose": -1, "random_state": 42}

    from src.models.lightgbm_model import LightGBMForecaster
    from src.models.ridge_model import RidgeForecaster

    lgbm = LightGBMForecaster(params=lgbm_params)
    ridge = RidgeForecaster(alpha=1.0)

    ensemble = EnsembleForecaster(lgbm_forecaster=lgbm, ridge_forecaster=ridge)
    ensemble.fit(X, y)

    X_pred = X.iloc[:24].reset_index(drop=True)
    result = ensemble.predict(X_pred)

    assert len(result) == 24
    assert "forecast" in result.columns
    assert "lower_90" in result.columns
    assert "upper_90" in result.columns
    assert "spike_prob" in result.columns
    assert "alpha" in result.columns


def test_ensemble_not_trained_raises():
    """Ensemble raises RuntimeError when not trained."""
    from src.models.ensemble import EnsembleForecaster
    ensemble = EnsembleForecaster()
    X = pd.DataFrame({"feat": [1.0, 2.0]})
    with pytest.raises(RuntimeError):
        ensemble.predict(X)
