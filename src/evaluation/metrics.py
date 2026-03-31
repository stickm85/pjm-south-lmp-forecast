"""Evaluation metrics for LMP forecasting."""

import numpy as np
import pandas as pd
from typing import Union


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred, eps=1e-6) -> float:
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def bias(y_true, y_pred) -> float:
    """Mean Bias (positive = over-forecast)."""
    return float(np.mean(np.array(y_pred) - np.array(y_true)))


def pinball_loss(y_true, y_pred, alpha: float) -> float:
    """Pinball (quantile) loss for confidence interval evaluation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


def coverage(y_true, lower, upper) -> float:
    """Fraction of true values within [lower, upper]."""
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def hourly_mae(y_true: pd.DataFrame, y_pred: pd.DataFrame, hour_col="hour_ending") -> pd.DataFrame:
    """Compute MAE separately for each of 24 hour-endings."""
    results = []
    for he in range(1, 25):
        mask_t = y_true[hour_col] == he
        mask_p = y_pred[hour_col] == he
        if mask_t.sum() == 0:
            continue
        m = mae(y_true.loc[mask_t, "lmp"], y_pred.loc[mask_p, "forecast"])
        results.append({"hour_ending": he, "mae": m})
    return pd.DataFrame(results)


def summary_metrics(y_true, y_pred, lower=None, upper=None) -> dict:
    """Compute a full suite of metrics."""
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if lower is not None and upper is not None:
        result["coverage_90"] = coverage(y_true, lower, upper)
        result["pinball_05"] = pinball_loss(y_true, lower, 0.05)
        result["pinball_95"] = pinball_loss(y_true, upper, 0.95)
    return result
