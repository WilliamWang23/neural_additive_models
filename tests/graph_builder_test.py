# coding=utf-8
"""Tests the PyTorch training stack."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_additive_models import data_utils
from neural_additive_models.training.trainer import TrainingConfig
from neural_additive_models.training.trainer import create_model
from neural_additive_models.training.trainer import evaluate_model
from neural_additive_models.training.trainer import train_ensemble


class TrainerTest(unittest.TestCase):
  """Tests whether training utilities can be run without error."""

  def test_create_model_and_evaluate_smoke(self):
    """Verify that the training helpers expose usable objects."""
    data_x, data_y, _ = data_utils.load_dataset("BreastCancer")
    data_gen = data_utils.split_training_dataset(data_x, data_y, n_splits=2, stratified=True)
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)
    config = TrainingConfig(
        training_epochs=1,
        activation="exu",
        learning_rate=1e-3,
        batch_size=256,
        shallow=True,
        regression=False,
        output_regularization=0.1,
        dropout=0.1,
        decay_rate=0.999,
        l2_regularization=0.1,
    )
    model = create_model(config, x_train)
    metric_value = evaluate_model(
        model=model,
        features=x_validation,
        targets=y_validation,
        regression=False,
        batch_size=256,
        device="cpu",
    )
    self.assertIsNotNone(model)
    self.assertIsInstance(metric_value, float)

  def test_train_ensemble_smoke(self):
    """Run a short end-to-end training loop."""
    data_x, data_y, _ = data_utils.load_dataset("BreastCancer")
    data_gen = data_utils.split_training_dataset(data_x, data_y, n_splits=2, stratified=True)
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)
    config = TrainingConfig(
        training_epochs=2,
        batch_size=128,
        save_checkpoint_every_n_epochs=1,
        early_stopping_epochs=2,
        num_basis_functions=16,
        n_models=1,
        device="cpu",
    )
    with tempfile.TemporaryDirectory() as tempdir:
      train_metric, validation_metric = train_ensemble(
          x_train=x_train,
          y_train=y_train,
          x_validation=x_validation,
          y_validation=y_validation,
          logdir=tempdir,
          config=config,
      )
    self.assertIsInstance(train_metric, float)
    self.assertIsInstance(validation_metric, float)


if __name__ == "__main__":
  unittest.main()
