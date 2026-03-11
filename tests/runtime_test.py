# coding=utf-8
"""Tests runtime helpers and checkpoint behavior."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_additive_models import models
from neural_additive_models.runtime import find_checkpoint_path
from neural_additive_models.runtime import load_checkpoint
from neural_additive_models.runtime import resolve_device
from neural_additive_models.runtime import save_checkpoint


class RuntimeTest(unittest.TestCase):
  """Tests runtime utilities."""

  def test_resolve_device_auto_priority(self):
    """Prefer CUDA over MPS, then CPU."""
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.backends.mps.is_available", return_value=True
    ):
      self.assertEqual(resolve_device("auto").type, "cuda")
    with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
        "torch.backends.mps.is_available", return_value=True
    ):
      self.assertEqual(resolve_device("auto").type, "mps")
    with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
        "torch.backends.mps.is_available", return_value=False
    ):
      self.assertEqual(resolve_device("auto").type, "cpu")

  def test_checkpoint_roundtrip(self):
    """Save and load a checkpoint payload."""
    model = models.NAM(num_inputs=4, num_units=8)
    with tempfile.TemporaryDirectory() as tempdir:
      checkpoint_path = os.path.join(tempdir, "best_checkpoint", "model.pt")
      save_checkpoint({"state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)
      resolved_path = find_checkpoint_path(os.path.join(tempdir))
      payload = load_checkpoint(resolved_path, map_location="cpu")
    self.assertIn("state_dict", payload)
    self.assertEqual(payload["epoch"], 1)


if __name__ == "__main__":
  unittest.main()
