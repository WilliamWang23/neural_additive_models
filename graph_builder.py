"""Compatibility helpers built on top of the PyTorch training stack."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from neural_additive_models.training.losses import calculate_loss
from neural_additive_models.training.metrics import calculate_metric
from neural_additive_models.training.metrics import rmse
from neural_additive_models.training.metrics import sigmoid
from neural_additive_models.training.trainer import TrainingConfig
from neural_additive_models.training.trainer import create_model
from neural_additive_models.training.trainer import create_nam_model
from neural_additive_models.training.trainer import evaluate_model


GraphOpsAndTensors = Dict[str, Any]


def build_graph(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate,
    batch_size,
    output_regularization,
    dropout,
    decay_rate,
    shallow,
    l2_regularization=0.0,
    feature_dropout=0.0,
    num_basis_functions=1000,
    units_multiplier=2,
    activation="exu",
    regression=False,
    use_dnn=False,
    **_,
) -> Tuple[GraphOpsAndTensors, Dict[str, Any]]:
  """Build PyTorch components analogous to the legacy graph builder."""
  config = TrainingConfig(
      training_epochs=1,
      learning_rate=learning_rate,
      batch_size=batch_size,
      output_regularization=output_regularization,
      dropout=dropout,
      decay_rate=decay_rate,
      shallow=shallow,
      l2_regularization=l2_regularization,
      feature_dropout=feature_dropout,
      num_basis_functions=num_basis_functions,
      units_multiplier=units_multiplier,
      activation=activation,
      regression=regression,
      use_dnn=use_dnn,
  )
  model = create_model(config, x_train)
  graph_tensors = {
      "nn_model": model,
      "config": config,
      "x_train": x_train,
      "y_train": y_train,
      "x_test": x_test,
      "y_test": y_test,
  }
  metric_scores = {
      "train": lambda _: evaluate_model(
          model=model,
          features=x_train,
          targets=y_train,
          regression=regression,
          batch_size=batch_size,
          device=torch.device("cpu"),
      ),
      "test": lambda _: evaluate_model(
          model=model,
          features=x_test,
          targets=y_test,
          regression=regression,
          batch_size=batch_size,
          device=torch.device("cpu"),
      ),
  }
  return graph_tensors, metric_scores


__all__ = [
    "GraphOpsAndTensors",
    "TrainingConfig",
    "build_graph",
    "calculate_loss",
    "calculate_metric",
    "create_nam_model",
    "evaluate_model",
    "rmse",
    "sigmoid",
]
