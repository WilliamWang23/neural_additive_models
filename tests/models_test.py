# coding=utf-8
"""Tests model construction and forward passes."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_additive_models import models


class LoadModelsTest(unittest.TestCase):
  """Tests whether neural net models can run without error."""

  def test_nam_and_dnn_forward(self):
    """Smoke-test NAM and DNN forward passes."""
    x = torch.rand(5, 10, dtype=torch.float32)
    architectures = {
        "exu_nam": models.NAM(num_inputs=x.shape[1], num_units=32, shallow=True, activation="exu"),
        "relu_nam": models.NAM(num_inputs=x.shape[1], num_units=16, shallow=False, activation="relu"),
        "dnn": models.DNN(input_dim=x.shape[1]),
    }
    for name, model in architectures.items():
      with self.subTest(architecture=name):
        out = model(x)
        self.assertIsInstance(out.detach().cpu().numpy(), np.ndarray)
        self.assertEqual(out.shape, (5,))

  def test_calc_outputs_matches_forward_without_feature_dropout(self):
    """Ensure feature contributions sum to the full prediction up to bias."""
    x = torch.rand(4, 6, dtype=torch.float32)
    model = models.NAM(num_inputs=x.shape[1], num_units=8, shallow=True, activation="exu")
    model.eval()
    outputs = model.calc_outputs(x, training=False)
    stacked = torch.stack(outputs, dim=-1).sum(dim=-1) + model.bias
    prediction = model(x, training=False)
    self.assertTrue(torch.allclose(stacked, prediction))


if __name__ == "__main__":
  unittest.main()
