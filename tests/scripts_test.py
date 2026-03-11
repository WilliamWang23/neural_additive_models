# coding=utf-8
"""Smoke tests for evaluation and plotting scripts."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_additive_models import data_utils
from neural_additive_models import nam_test
from neural_additive_models import nam_train
from neural_additive_models import plot_nam_ensemble


class ScriptSmokeTest(unittest.TestCase):
  """Tests script entrypoints with a short local training run."""

  def test_eval_and_plot_scripts(self):
    """Train one small model, then run evaluation and plotting."""
    with tempfile.TemporaryDirectory() as tempdir:
      parser = nam_train.build_parser()
      args = parser.parse_args([
          "--training_epochs=2",
          "--save_checkpoint_every_n_epochs=1",
          "--early_stopping_epochs=2",
          "--dataset_name=BreastCancer",
          "--num_basis_functions=8",
          "--n_models=1",
          "--batch_size=128",
          "--device=cpu",
          f"--logdir={os.path.join(tempdir, 'training')}",
      ])
      data_x, data_y, _ = data_utils.load_dataset(args.dataset_name)
      (x_train_all, y_train_all), _ = data_utils.get_train_test_fold(
          data_x, data_y, fold_num=args.fold_num, num_folds=5, stratified=True)
      data_gen = data_utils.split_training_dataset(
          x_train_all, y_train_all, n_splits=args.num_splits, stratified=True)
      nam_train.single_split_training(data_gen, args.logdir, args)

      run_dir = os.path.join(args.logdir, "fold_1")
      nam_test.main([
          f"--run_dir={run_dir}",
          "--device=cpu",
      ])
      output_dir = os.path.join(tempdir, "plots")
      plot_nam_ensemble.main([
          f"--run_dir={run_dir}",
          "--device=cpu",
          f"--output_dir={output_dir}",
      ])
      self.assertTrue(os.path.exists(os.path.join(run_dir, "test_outputs", "breastcancer_test_results.txt")))
      self.assertTrue(os.path.exists(os.path.join(output_dir, "breastcancer_shape_plots_ensemble.png")))


if __name__ == "__main__":
  unittest.main()
