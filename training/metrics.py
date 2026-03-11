"""Metric helpers for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics
import torch


def sigmoid(x):
  """Apply a numerically stable sigmoid to numpy-like values."""
  x = np.asarray(x)
  return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def rmse(y_true, y_pred) -> float:
  """Return root mean squared error."""
  return float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))


def calculate_metric(y_true, predictions, regression: bool = True) -> float:
  """Compute the project evaluation metric."""
  if regression:
    return rmse(y_true, predictions)
  return float(sk_metrics.roc_auc_score(y_true, sigmoid(predictions)))


def detach_to_numpy(value: torch.Tensor) -> np.ndarray:
  """Detach a tensor and convert it to numpy."""
  return value.detach().cpu().numpy()
