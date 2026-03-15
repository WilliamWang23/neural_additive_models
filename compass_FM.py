#!/usr/bin/env python
# coding=utf-8
"""COMPAS NAM vs NAM+FM experiment on overall AUROC."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _repo_root() -> str:
  return osp.dirname(osp.abspath(__file__))


def _add_repo_parent_to_path(repo_root: str) -> None:
  parent = osp.dirname(repo_root)
  if parent not in sys.path:
    sys.path.insert(0, parent)


_add_repo_parent_to_path(_repo_root())

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from neural_additive_models import data_utils
from neural_additive_models import models
from neural_additive_models.nam_train import str2bool
from neural_additive_models.runtime import resolve_device
from neural_additive_models.runtime import save_checkpoint
from neural_additive_models.training.data import create_eval_loader
from neural_additive_models.training.data import create_train_loader
from neural_additive_models.training.trainer import infer_num_units


TASK_VALUE_TO_INDEX = {"Female": 0, "Male": 1}
COMPAS_FEATURE_COLUMNS = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "race",
    "sex",
]


@dataclass
class ExperimentConfig:
  """Hyperparameters for the COMPASS_FM experiment."""

  training_epochs: int
  learning_rate: float
  batch_size: int
  decay_rate: float
  dropout: float
  feature_dropout: float
  num_basis_functions: int
  units_multiplier: int
  fm_rank: int
  activation: str
  shallow: bool
  output_regularization: float
  l2_regularization: float
  early_stopping_epochs: int
  n_models: int
  n_folds: int
  seed: int
  device: str
  output_dir: str


def build_parser() -> argparse.ArgumentParser:
  """Create the CLI parser."""
  parser = argparse.ArgumentParser(description="Run the compass_FM COMPAS AUROC experiment.")
  parser.add_argument("--training_epochs", type=int, default=80)
  parser.add_argument("--learning_rate", type=float, default=1e-2)
  parser.add_argument("--batch_size", type=int, default=512)
  parser.add_argument("--decay_rate", type=float, default=0.995)
  parser.add_argument("--dropout", type=float, default=0.15)
  parser.add_argument("--feature_dropout", type=float, default=0.0)
  parser.add_argument("--num_basis_functions", type=int, default=64)
  parser.add_argument("--units_multiplier", type=int, default=2)
  parser.add_argument("--fm_rank", type=int, default=8)
  parser.add_argument("--activation", type=str, default="exu")
  parser.add_argument("--shallow", type=str2bool, nargs="?", const=True, default=True)
  parser.add_argument("--output_regularization", type=float, default=0.0)
  parser.add_argument("--l2_regularization", type=float, default=0.0)
  parser.add_argument("--early_stopping_epochs", type=int, default=20)
  parser.add_argument("--n_models", type=int, default=10)
  parser.add_argument("--n_folds", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--output_dir", type=str, default=osp.join(_repo_root(), "output", "compass_FM"))
  return parser


def make_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
  """Convert CLI args into a typed config."""
  return ExperimentConfig(
      training_epochs=args.training_epochs,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
      decay_rate=args.decay_rate,
      dropout=args.dropout,
      feature_dropout=args.feature_dropout,
      num_basis_functions=args.num_basis_functions,
      units_multiplier=args.units_multiplier,
      fm_rank=args.fm_rank,
      activation=args.activation,
      shallow=args.shallow,
      output_regularization=args.output_regularization,
      l2_regularization=args.l2_regularization,
      early_stopping_epochs=args.early_stopping_epochs,
      n_models=args.n_models,
      n_folds=args.n_folds,
      seed=args.seed,
      device=args.device,
      output_dir=args.output_dir,
  )


def ensure_dir(path: str) -> str:
  """Create a directory if needed and return it."""
  os.makedirs(path, exist_ok=True)
  return path


def set_torch_seed(seed: int) -> None:
  """Seed numpy and torch RNGs."""
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def sigmoid(values: np.ndarray) -> np.ndarray:
  """Apply a numerically stable sigmoid to logits."""
  values = np.asarray(values, dtype=np.float64)
  return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def prepare_compass_data() -> Dict[str, object]:
  """Load COMPAS data for overall recidivism prediction."""
  df = data_utils.load_recidivism_dataframe()
  x_df = df[COMPAS_FEATURE_COLUMNS].copy()
  features, column_names = data_utils.transform_data(x_df)
  labels = df["two_year_recid"].to_numpy(dtype=np.float32)
  task_index = np.array([TASK_VALUE_TO_INDEX[value] for value in df["sex"].to_numpy()], dtype=np.int64)
  return {
      "features": features.astype(np.float32),
      "labels": labels,
      "task_index": task_index,
      "column_names": column_names,
  }


def build_group_labels(labels: np.ndarray, task_index: np.ndarray) -> np.ndarray:
  """Preserve sex/label balance across folds while reporting only overall AUROC."""
  return np.array([f"{int(task)}_{int(label)}" for task, label in zip(task_index, labels)], dtype=object)


def feature_output_regularization(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
  """Penalize the mean squared additive NAM contributions."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  penalties = [torch.mean(outputs.square()) for outputs in per_feature_outputs]
  return torch.stack(penalties).mean()


def weight_decay(model: nn.Module, num_networks: int) -> torch.Tensor:
  """Compute L2 regularization across trainable parameters."""
  penalties = [0.5 * parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad]
  if not penalties:
    return torch.tensor(0.0)
  return torch.stack(penalties).sum() / max(num_networks, 1)


def build_nam_model(x_train: np.ndarray, config: ExperimentConfig) -> models.NAM:
  """Create the NAM baseline."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
  )
  return models.NAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
      dropout=config.dropout,
      activation=config.activation,
  )


def build_factorized_nam_model(x_train: np.ndarray, config: ExperimentConfig) -> models.FactorizedNAM:
  """Create the NAM + FM interaction model."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
  )
  return models.FactorizedNAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      fm_rank=config.fm_rank,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
      dropout=config.dropout,
      activation=config.activation,
  )


def predict_logits(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
  """Predict logits in batches."""
  loader = create_eval_loader(features, batch_size=batch_size)
  outputs = []
  model.eval()
  with torch.no_grad():
    for batch_features, _ in loader:
      outputs.append(model(batch_features.to(device), training=False).detach().cpu().numpy())
  return np.concatenate(outputs, axis=0)


def _checkpoint_payload(model: nn.Module, epoch: int, metric_value: float) -> Dict[str, object]:
  """Build a minimal checkpoint payload."""
  return {
      "epoch": epoch,
      "metric_value": metric_value,
      "state_dict": model.state_dict(),
      "model_type": model.__class__.__name__,
  }


def train_binary_model(
    model_kind: str,
    model_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    model_dir: str,
):
  """Train one binary classifier model and return the best checkpointed model."""
  set_torch_seed(config.seed + model_index)
  if model_kind == "nam":
    model = build_nam_model(x_train, config).to(device)
  elif model_kind == "factorized_nam":
    model = build_factorized_nam_model(x_train, config).to(device)
  else:
    raise ValueError(f"Unsupported model kind: {model_kind}")

  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
  train_loader = create_train_loader(
      x_train=x_train,
      y_train=y_train,
      batch_size=min(config.batch_size, x_train.shape[0]),
      regression=False,
  )

  best_metric = -np.inf
  best_epoch = 0
  best_state = None
  patience = 0
  best_dir = ensure_dir(osp.join(model_dir, "best_checkpoint"))

  for epoch in range(1, config.training_epochs + 1):
    model.train()
    for batch_features, batch_targets in train_loader:
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      optimizer.zero_grad()
      logits = model(batch_features, training=True)
      loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
      if config.output_regularization > 0:
        loss = loss + config.output_regularization * feature_output_regularization(model, batch_features)
      if config.l2_regularization > 0:
        loss = loss + config.l2_regularization * weight_decay(model, num_networks=len(model.feature_nns))
      loss.backward()
      optimizer.step()
    scheduler.step()

    val_probabilities = sigmoid(predict_logits(model, x_val, config.batch_size, device))
    validation_metric = float(roc_auc_score(y_val, val_probabilities))
    if validation_metric > best_metric:
      best_metric = validation_metric
      best_epoch = epoch
      patience = 0
      best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
      save_checkpoint(_checkpoint_payload(model, epoch, validation_metric), osp.join(best_dir, "model.pt"))
    else:
      patience += 1
      if patience >= config.early_stopping_epochs:
        break

  if best_state is None:
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_checkpoint(_checkpoint_payload(model, config.training_epochs, best_metric), osp.join(best_dir, "model.pt"))
  model.load_state_dict(best_state)
  model.eval()
  return model, {"best_epoch": best_epoch or config.training_epochs, "best_validation_auc": float(best_metric)}


def ensemble_mean(predictions: Sequence[np.ndarray]) -> np.ndarray:
  """Average a list of per-model probability arrays."""
  return np.mean(np.stack(predictions, axis=0), axis=0)


def summarize_metric_list(values: Iterable[float]) -> Dict[str, float]:
  """Return mean and std summary for a list of metrics."""
  array = np.asarray(list(values), dtype=np.float64)
  return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1) if len(array) > 1 else 0.0)}


def run_cross_validation(
    dataset: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
  """Run 5-fold NAM vs NAM+FM COMPAS comparison on overall AUROC."""
  features = np.asarray(dataset["features"], dtype=np.float32)
  labels = np.asarray(dataset["labels"], dtype=np.float32)
  task_index = np.asarray(dataset["task_index"], dtype=np.int64)
  strata = build_group_labels(labels, task_index)
  splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
  cv_results = {"nam": [], "factorized_nam": []}

  for fold_index, (train_val_index, test_index) in enumerate(splitter.split(features, strata), start=1):
    fold_dir = ensure_dir(osp.join(config.output_dir, "training", f"fold_{fold_index}"))
    train_val_strata = strata[train_val_index]
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=config.seed + fold_index,
    )
    train_rel_index, val_rel_index = next(validation_splitter.split(features[train_val_index], train_val_strata))
    train_index = train_val_index[train_rel_index]
    val_index = train_val_index[val_rel_index]

    for model_kind in ("nam", "factorized_nam"):
      model_probabilities = []
      model_metadata = []
      model_fold_dir = ensure_dir(osp.join(fold_dir, model_kind))
      for model_index in tqdm(range(config.n_models), desc=f"Fold {fold_index} {model_kind}"):
        model, model_info = train_binary_model(
            model_kind=model_kind,
            model_index=model_index,
            x_train=features[train_index],
            y_train=labels[train_index],
            x_val=features[val_index],
            y_val=labels[val_index],
            config=config,
            device=device,
            model_dir=ensure_dir(osp.join(model_fold_dir, f"model_{model_index}")),
        )
        probabilities = sigmoid(predict_logits(model, features[test_index], config.batch_size, device))
        model_probabilities.append(probabilities)
        model_metadata.append(model_info)

      ensemble_probabilities = ensemble_mean(model_probabilities)
      fold_auc = float(roc_auc_score(labels[test_index], ensemble_probabilities))
      cv_results[model_kind].append({
          "fold": fold_index,
          "combined_auc": fold_auc,
          "models": model_metadata,
      })

  summary = {
      model_kind: {
          "combined_auc": summarize_metric_list([fold["combined_auc"] for fold in cv_results[model_kind]])
      }
      for model_kind in ("nam", "factorized_nam")
  }
  summary["delta_auc"] = (
      summary["factorized_nam"]["combined_auc"]["mean"] -
      summary["nam"]["combined_auc"]["mean"]
  )
  return {"folds": cv_results, "summary": summary}


def save_json(payload: Dict[str, object], path: str) -> None:
  """Persist a JSON payload."""
  ensure_dir(osp.dirname(path))
  with open(path, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2)


def save_summary_text(result: Dict[str, object], path: str) -> None:
  """Persist a human-readable summary."""
  summary = result["summary"]
  lines = [
      "compass_FM Experiment Summary",
      f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
      "",
      "NAM",
      "  Combined AUROC: "
      f"{summary['nam']['combined_auc']['mean']:.3f} +/- {summary['nam']['combined_auc']['std']:.3f}",
      "",
      "NAM + FactorizedMachine",
      "  Combined AUROC: "
      f"{summary['factorized_nam']['combined_auc']['mean']:.3f} +/- "
      f"{summary['factorized_nam']['combined_auc']['std']:.3f}",
      "",
      f"Delta AUROC (NAM+FM - NAM): {summary['delta_auc']:.4f}",
  ]
  with open(path, "w", encoding="utf-8") as file_obj:
    file_obj.write("\n".join(lines).rstrip() + "\n")


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint."""
  args = build_parser().parse_args(argv)
  config = make_experiment_config(args)
  ensure_dir(config.output_dir)
  device = resolve_device(config.device)
  dataset = prepare_compass_data()
  cv_result = run_cross_validation(dataset, config, device)
  result = {
      "experiment_name": "compass_FM",
      "config": {
          **config.__dict__,
          "resolved_device": str(device),
      },
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      **cv_result,
  }
  json_path = osp.join(config.output_dir, "compass_FM_results.json")
  text_path = osp.join(config.output_dir, "compass_FM_summary.txt")
  save_json(result, json_path)
  save_summary_text(result, text_path)
  print(f"Saved: {json_path}")
  print(f"Saved: {text_path}")


if __name__ == "__main__":
  main()
