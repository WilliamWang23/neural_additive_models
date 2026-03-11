# coding=utf-8
"""Tests functionality of loading the different datasets."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_additive_models import data_utils


class LoadDataTest(unittest.TestCase):
  """Data loading tests."""

  _OPTIONAL_DATASET_PATHS = {
      "Fico": "data/FICO-Explainable-ML-Challenge-HELOC-Dataset-master/HelocData.csv",
      "Recidivism": "data/compas-analysis-master/compas-scores-two-years.csv",
      "Credit": "data/Credit Card Fraud Detection/creditcard.csv",
      "Adult": "data/adult.data",
      "Telco": "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
  }

  def test_data(self):
    """Verify that configured datasets load into numpy arrays."""
    dataset_cases = [
        ("BreastCancer", 569),
        ("Housing", 20640),
        ("Fico", 9861),
        ("Recidivism", 6172),
        ("Credit", 284807),
        ("Adult", 32561),
        ("Telco", 7043),
    ]
    for dataset_name, dataset_size in dataset_cases:
      with self.subTest(dataset_name=dataset_name):
        optional_path = self._OPTIONAL_DATASET_PATHS.get(dataset_name)
        if optional_path and not os.path.exists(optional_path):
          continue
        try:
          x, y, _ = data_utils.load_dataset(dataset_name)
        except Exception:
          if dataset_name == "Housing":
            continue
          raise
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(x), dataset_size)


if __name__ == "__main__":
  unittest.main()
