#!/usr/bin/env python
# coding=utf-8
"""Standalone NAM ensemble testing script (no plotting)."""

import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _repo_root() -> str:
  return osp.dirname(osp.abspath(__file__))


def _add_repo_parent_to_path(repo_root: str) -> None:
  parent = osp.dirname(repo_root)
  if parent not in sys.path:
    sys.path.insert(0, parent)


def resolve_checkpoint_path(model_dir: str) -> str:
  best_dir = osp.join(model_dir, "best_checkpoint")
  ckpt = tf.train.latest_checkpoint(best_dir) if osp.isdir(best_dir) else None
  if ckpt:
    return ckpt
  ckpt = tf.train.latest_checkpoint(model_dir)
  if ckpt:
    return ckpt
  index_files = tf.io.gfile.glob(osp.join(model_dir, "*.index"))
  if index_files:
    return sorted(index_files)[-1].replace(".index", "")
  raise FileNotFoundError(f"No checkpoint found under: {model_dir}")


def build_restore_var_map(nn_model, checkpoint_path: str, model_idx: int) -> Dict[str, tf.Variable]:
  ckpt_vars = {name for name, _ in tf.train.list_variables(checkpoint_path)}
  var_map = {}
  for var in nn_model.variables:
    base_name = var.name.split(":", 1)[0]
    candidates = [
        base_name,
        base_name.replace("/nam/", f"/nam_{model_idx}/"),
        base_name.replace(f"/nam_{model_idx}/", "/nam/"),
    ]
    chosen = next((c for c in candidates if c in ckpt_vars), None)
    if chosen is not None:
      var_map[chosen] = var
  return var_map


def batched_predict(sess, pred_tensor, pred_input_ph, x_data: np.ndarray, batch_size: int = 512) -> np.ndarray:
  batch_size = min(batch_size, x_data.shape[0])
  preds = []
  for start in range(0, x_data.shape[0], batch_size):
    preds.append(sess.run(pred_tensor, feed_dict={pred_input_ph: x_data[start:start + batch_size]}))
  return np.concatenate(preds, axis=0)


def _load_training_params(run_dir: str) -> Dict[str, Any]:
  params_path = osp.join(run_dir, "training_params.json")
  if not osp.exists(params_path):
    # Backward/alternate layout compatibility:
    # run_dir = .../repro_runs/<run_name>/fold_<k>
    # params   = .../repro_runs/<run_name>/training/fold_<k>/training_params.json
    fold_name = osp.basename(osp.normpath(run_dir))
    alt_params_path = osp.join(osp.dirname(run_dir), "training", fold_name, "training_params.json")
    if osp.exists(alt_params_path):
      params_path = alt_params_path
    else:
      raise FileNotFoundError(
          f"training_params.json not found in run_dir: {run_dir}. "
          "Pass explicit args or run training with updated nam_train.py."
      )
  with open(params_path, "r", encoding="utf-8") as f:
    return json.load(f)


def _infer_model_logdir(repo_root: str, run_dir: str, fold_num: int, split_idx: int) -> str:
  # run_dir can be either:
  #   .../repro_runs/<run_name>/fold_<k>        -> logdir_root = .../run_name/training
  #   .../repro_runs/<run_name>/training/fold_<k> -> logdir_root = .../run_name/training (already)
  run_dir = osp.normpath(run_dir)
  parent = osp.dirname(run_dir)
  if osp.basename(parent) == "training":
    logdir_root = parent
  else:
    logdir_root = osp.join(parent, "training")
  return osp.join(logdir_root, f"fold_{fold_num}", f"split_{split_idx}")


def main():
  repo_root = _repo_root()

  parser = argparse.ArgumentParser(description="Evaluate trained NAM ensemble on held-out test split.")
  parser.add_argument("--repo_root", default=repo_root)
  parser.add_argument("--run_dir", default=None, help="Fold run dir containing training_params.json")
  parser.add_argument("--model_logdir", default=None, help="Path like .../training/fold_x/split_y")
  parser.add_argument("--dataset_name", default=None)
  parser.add_argument("--n_models", type=int, default=None)
  parser.add_argument("--fold_num", type=int, default=None)
  parser.add_argument("--num_folds", type=int, default=5)
  parser.add_argument("--num_splits", type=int, default=None)
  parser.add_argument("--split_idx", type=int, default=1, help="1-based split index")
  parser.add_argument("--activation", default=None)
  parser.add_argument("--num_basis_functions", type=int, default=None)
  parser.add_argument("--units_multiplier", type=int, default=None)
  parser.add_argument("--dropout", type=float, default=None)
  parser.add_argument("--feature_dropout", type=float, default=None)
  parser.add_argument("--shallow", action="store_true")
  parser.add_argument("--regression", action="store_true", default=None)
  parser.add_argument("--output_dir", default=None)
  args = parser.parse_args()

  _add_repo_parent_to_path(args.repo_root)
  from neural_additive_models import data_utils  # pylint: disable=import-error
  from neural_additive_models import graph_builder  # pylint: disable=import-error

  cfg = {}
  if args.run_dir:
    cfg = _load_training_params(args.run_dir)

  dataset_name = args.dataset_name or cfg.get("dataset_name")
  n_models = args.n_models if args.n_models is not None else cfg.get("n_models")
  fold_num = args.fold_num if args.fold_num is not None else cfg.get("fold_num")
  num_splits = args.num_splits if args.num_splits is not None else cfg.get("num_splits")
  activation = args.activation or cfg.get("activation")
  num_basis_functions = (
      args.num_basis_functions if args.num_basis_functions is not None else cfg.get("num_basis_functions")
  )
  units_multiplier = args.units_multiplier if args.units_multiplier is not None else cfg.get("units_multiplier", 2)
  dropout = args.dropout if args.dropout is not None else cfg.get("dropout", 0.0)
  feature_dropout = (
      args.feature_dropout if args.feature_dropout is not None else cfg.get("feature_dropout", 0.0)
  )
  if args.regression is not None:
    is_regression = bool(args.regression)
  else:
    is_regression = bool(cfg.get("regression")) if "regression" in cfg else (dataset_name in ["Housing", "Fico"])
  shallow = bool(args.shallow) if args.shallow else bool(cfg.get("shallow", False))

  if not dataset_name:
    raise ValueError("Missing dataset_name. Pass --dataset_name or --run_dir with training_params.json.")
  for key, val in {
      "n_models": n_models,
      "fold_num": fold_num,
      "num_splits": num_splits,
      "activation": activation,
      "num_basis_functions": num_basis_functions,
  }.items():
    if val is None:
      raise ValueError(f"Missing required parameter: {key}. Pass explicitly or via --run_dir.")

  if args.model_logdir:
    model_logdir = args.model_logdir
  elif args.run_dir:
    model_logdir = _infer_model_logdir(args.repo_root, args.run_dir, int(fold_num), int(args.split_idx))
  else:
    raise ValueError("Provide --model_logdir or --run_dir.")

  if args.output_dir:
    output_dir = args.output_dir
  elif args.run_dir:
    output_dir = osp.join(args.run_dir, "test_outputs")
  else:
    output_dir = osp.join(args.repo_root, "repro_runs", f"{dataset_name.lower()}_test_outputs")

  os.makedirs(output_dir, exist_ok=True)
  data_x, data_y, _ = data_utils.load_dataset(dataset_name)
  (x_train_all, y_train_all), test_dataset = data_utils.get_train_test_fold(
      data_x, data_y, fold_num=int(fold_num), num_folds=args.num_folds, stratified=not is_regression
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all, y_train_all, n_splits=int(num_splits), stratified=not is_regression
  )
  split = None
  for _ in range(args.split_idx):
    split = next(data_gen)
  (x_train, _), _ = split
  x_test, y_test = test_dataset

  per_model_metric = []
  for model_idx in range(int(n_models)):
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    with tf.Graph().as_default():
      nn_model = graph_builder.create_nam_model(
          x_train=x_train,
          dropout=float(dropout),
          feature_dropout=float(feature_dropout),
          num_basis_functions=int(num_basis_functions),
          units_multiplier=int(units_multiplier),
          activation=activation,
          trainable=False,
          shallow=shallow,
          name_scope=f"model_{model_idx}",
      )
      pred_input_ph = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
      pred_tensor = nn_model(pred_input_ph, training=False)
      if not is_regression:
        pred_tensor = tf.nn.sigmoid(pred_tensor)

      ckpt = resolve_checkpoint_path(osp.join(model_logdir, f"model_{model_idx}"))
      saver = tf.train.Saver(var_list=build_restore_var_map(nn_model, ckpt, model_idx))
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        y_pred = batched_predict(sess, pred_tensor, pred_input_ph, x_test)
      metric_val = float(graph_builder.calculate_metric(y_test, y_pred, regression=is_regression))
      per_model_metric.append(metric_val)

  metric_name = "RMSE" if is_regression else "AUROC"
  mean_metric = float(np.mean(per_model_metric))
  prefix = dataset_name.lower()
  details_path = osp.join(output_dir, f"{prefix}_test_details.json")
  text_path = osp.join(output_dir, f"{prefix}_test_results.txt")
  payload = {
      "dataset_name": dataset_name,
      "metric_name": metric_name,
      "n_models": int(n_models),
      "fold_num": int(fold_num),
      "num_folds": args.num_folds,
      "num_splits": int(num_splits),
      "split_idx": args.split_idx,
      "per_model_metric": per_model_metric,
      "mean_metric": mean_metric,
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "model_logdir": model_logdir,
      "resolved_from_run_dir": bool(args.run_dir),
  }
  with open(details_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  summary = [
      f"Dataset: {dataset_name}",
      f"Fold: {fold_num}/{args.num_folds}",
      f"Models: {n_models}",
      f"Metric: {metric_name}",
      f"Mean test {metric_name}: {mean_metric:.6f}",
      "Per-model test: " + ", ".join([f"{x:.6f}" for x in per_model_metric]),
  ]
  with open(text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary) + "\n")

  print(f"Per-model test {metric_name}: {per_model_metric}")
  print(f"Mean test {metric_name}: {mean_metric:.6f}")
  print(f"Saved: {details_path}")
  print(f"Saved: {text_path}")


if __name__ == "__main__":
  main()
