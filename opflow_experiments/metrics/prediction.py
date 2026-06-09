"""Raw count prediction metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Compute mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def poisson_deviance(y_true, lambda_pred, eps: float = 1e-8) -> float:
    """Compute mean Poisson deviance."""
    y_true = np.asarray(y_true, dtype=float)
    lambda_pred = np.maximum(np.asarray(lambda_pred, dtype=float), eps)
    safe_ratio = np.where(y_true > 0, y_true / lambda_pred, 1.0)
    term = np.where(y_true > 0, y_true * np.log(safe_ratio), 0.0)
    deviance = 2.0 * np.mean(term - (y_true - lambda_pred))
    return float(deviance)
