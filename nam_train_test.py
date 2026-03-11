# coding=utf-8
"""Tests functionality of training NAM models."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_additive_models import data_utils
from neural_additive_models import nam_train


class NAMTrainingTest(unittest.TestCase):
  """Tests whether NAM training runs without error."""

  def test_nam_classification(self):
    """Run a short classification training pipeline."""
    data_x, data_y, _ = data_utils.load_dataset("BreastCancer")
    (x_train_all, y_train_all), _ = data_utils.get_train_test_fold(
        data_x, data_y, fold_num=1, num_folds=5, stratified=True)
    data_gen = data_utils.split_training_dataset(
        x_train_all, y_train_all, n_splits=3, stratified=True)
    parser = nam_train.build_parser()
    args = parser.parse_args([
        "--training_epochs=4",
        "--save_checkpoint_every_n_epochs=2",
        "--early_stopping_epochs=2",
        "--dataset_name=BreastCancer",
        "--num_basis_functions=16",
        f"--logdir={tempfile.mkdtemp()}",
    ])
    nam_train.single_split_training(data_gen, args.logdir, args)


if __name__ == "__main__":
  unittest.main()
