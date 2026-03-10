#!/usr/bin/env python
# coding=utf-8
"""Standalone NAM ensemble visualization for trained checkpoints."""

import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


def _repo_root() -> str:
  return osp.dirname(osp.abspath(__file__))


def _add_repo_parent_to_path(repo_root: str) -> None:
  parent = osp.dirname(repo_root)
  if parent not in sys.path:
    sys.path.insert(0, parent)


def inverse_min_max_scaler(x, min_val, max_val):
  return (x + 1) / 2 * (max_val - min_val) + min_val


def load_col_min_max(data_utils, dataset_name: str) -> Dict[str, Tuple[float, float]]:
  if dataset_name == "Housing":
    dataset = data_utils.load_california_housing_data()
  elif dataset_name == "BreastCancer":
    dataset = data_utils.load_breast_data()
  elif dataset_name == "Recidivism":
    dataset = data_utils.load_recidivism_data()
  elif dataset_name == "Fico":
    dataset = data_utils.load_fico_score_data()
  elif dataset_name == "Credit":
    dataset = data_utils.load_credit_data()
  else:
    raise ValueError(f"{dataset_name} not found!")

  if "full" in dataset:
    dataset = dataset["full"]
  x_df = dataset["X"].copy()
  # Recidivism (and Adult, etc.) use one-hot encoding; col names must match load_dataset output.
  if dataset_name == "Recidivism":
    is_cat = np.array([dt.kind == "O" for dt in x_df.dtypes])
    cat_cols = x_df.columns.values[is_cat].tolist()
    num_cols = x_df.columns.values[~is_cat].tolist()
    if cat_cols:
      x_df = x_df.astype({c: str for c in cat_cols})
      transformed = pd.get_dummies(x_df, columns=cat_cols, prefix_sep=": ", dtype=np.float32)
    else:
      transformed = x_df.copy()
    if num_cols:
      transformed[num_cols] = transformed[num_cols].astype(np.float32)
    col_min_max = {}
    for col in transformed.columns:
      vals = transformed[col].values
      col_min_max[col] = (float(np.min(vals)), float(np.max(vals)))
    return col_min_max
  x = x_df
  col_min_max = {}
  for col in x:
    unique_vals = x[col].unique()
    col_min_max[col] = (float(np.min(unique_vals)), float(np.max(unique_vals)))
  return col_min_max


def compute_all_indices(
    data_x: np.ndarray, unique_features: List[np.ndarray], column_names: List[str]
) -> Dict[str, np.ndarray]:
  all_indices = {}
  for i, col in enumerate(column_names):
    x_i = data_x[:, i]
    all_indices[col] = np.searchsorted(unique_features[i][:, 0], x_i, "left")
  return all_indices


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


def build_restore_var_map(nn_model, checkpoint_path: str, model_idx: int):
  ckpt_vars = {name for name, _ in tf.train.list_variables(checkpoint_path)}
  var_map = {}
  missing_vars = []
  for var in nn_model.variables:
    base_name = var.name.split(":", 1)[0]
    candidates = [base_name]
    candidates.append(base_name.replace(f"/nam/", f"/nam_{model_idx}/"))
    candidates.append(base_name.replace(f"/nam_{model_idx}/", "/nam/"))
    chosen = next((c for c in candidates if c in ckpt_vars), None)
    if chosen is None:
      missing_vars.append(base_name)
      continue
    var_map[chosen] = var

  if missing_vars:
    preview = ", ".join(missing_vars[:5])
    raise KeyError(f"Unmatched model vars ({len(missing_vars)}). Examples: {preview}")
  return var_map


def get_test_predictions(sess, pred_tensor, pred_input_ph, x_test: np.ndarray, batch_size: int = 256):
  batch_size = min(batch_size, x_test.shape[0])
  preds = []
  for start in range(0, x_test.shape[0], batch_size):
    pred = sess.run(pred_tensor, feed_dict={pred_input_ph: x_test[start : start + batch_size]})
    preds.append(pred)
  return np.concatenate(preds, axis=0)


def get_feature_predictions(sess, feature_tensors, feature_input_phs, unique_features: List[np.ndarray]):
  feature_predictions = []
  for i, feat_vals in enumerate(unique_features):
    f_preds = sess.run(feature_tensors[i], feed_dict={feature_input_phs[i]: feat_vals})
    feature_predictions.append(np.squeeze(f_preds))
  return feature_predictions


def compute_model_mean_pred(model_hist_data, all_indices, column_names):
  return {col: float(np.mean(model_hist_data[col][all_indices[col]])) for col in column_names}


def compute_mean_feature_importance(avg_hist_data, mean_pred):
  mean_abs_score = {}
  for k in avg_hist_data:
    mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
  x1, x2 = zip(*mean_abs_score.items())
  return x1, x2


def plot_mean_feature_importance(dataset_name, cols, x2, output_path):
  fig = plt.figure(figsize=(7, 4.5))
  ind = np.arange(len(cols))
  x_order = np.argsort(x2)
  cols_here = [cols[i] for i in x_order]
  x2_here = [x2[i] for i in x_order]

  plt.bar(ind, x2_here, width=0.6, label="NAM Ensemble")
  plt.xticks(ind, cols_here, rotation=75, fontsize=10, ha="right")
  plt.ylabel("Mean Absolute Score", fontsize=12)
  plt.legend(loc="upper right", fontsize=10)
  plt.title(f"Overall Importance: {dataset_name}", fontsize=13)
  plt.tight_layout()
  fig.savefig(output_path, dpi=180)


def shade_by_density_blocks(
    hist_data,
    num_rows,
    num_cols,
    unique_features_original,
    single_features_original,
    categorical_names,
    n_blocks=20,
    color=(0.9, 0.5, 0.5),
):
  hist_data_pairs = sorted(list(hist_data.items()), key=lambda x: x[0])
  min_y = np.min([np.min(a[1]) for a in hist_data_pairs])
  max_y = np.max([np.max(a[1]) for a in hist_data_pairs])
  span = max_y - min_y
  min_y = min_y - 0.01 * span
  max_y = max_y + 0.01 * span

  for i, (name, _) in enumerate(hist_data_pairs):
    unique_x_data = unique_features_original[name]
    single_feature_data = single_features_original[name]
    ax = plt.subplot(num_rows, num_cols, i + 1)
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    x_n_blocks = min(n_blocks, len(unique_x_data))
    if name in categorical_names:
      min_x -= 0.5
      max_x += 0.5
    segments = (max_x - min_x) / x_n_blocks
    density = np.histogram(single_feature_data, bins=x_n_blocks)
    normed_density = density[0] / np.maximum(np.max(density[0]), 1e-12)

    for p in range(x_n_blocks):
      start_x = min_x + segments * p
      end_x = min_x + segments * (p + 1)
      alpha = min(1.0, 0.01 + normed_density[p])
      rect = patches.Rectangle(
          (start_x, min_y - 1),
          end_x - start_x,
          max_y - min_y + 1,
          linewidth=0.01,
          edgecolor=color,
          facecolor=color,
          alpha=alpha,
      )
      ax.add_patch(rect)


def plot_all_hist(
    dataset_name,
    hist_data,
    num_rows,
    num_cols,
    color_base,
    col_names_map,
    categorical_names,
    mean_pred,
    unique_features_original,
    linewidth=3.0,
    min_y=None,
    max_y=None,
    alpha=1.0,
):
  hist_data_pairs = sorted(list(hist_data.items()), key=lambda x: x[0])
  if min_y is None:
    min_y = np.min([np.min(a) for _, a in hist_data_pairs])
  if max_y is None:
    max_y = np.max([np.max(a) for _, a in hist_data_pairs])
  span = max_y - min_y
  min_y = min_y - 0.01 * span
  max_y = max_y + 0.01 * span

  col_mapping = col_names_map[dataset_name]

  for i, (name, pred) in enumerate(hist_data_pairs):
    center = mean_pred[name]
    unique_x_data = unique_features_original[name]
    plt.subplot(num_rows, num_cols, i + 1)
    if name in categorical_names:
      unique_x_data = np.round(unique_x_data, decimals=1)
      step_loc = "mid" if len(unique_x_data) <= 2 else "post"
      unique_plot_data = np.array(unique_x_data) - 0.5
      unique_plot_data[-1] += 1
      plt.step(unique_plot_data, pred - center, color=color_base, linewidth=linewidth, where=step_loc, alpha=alpha)
      plt.xticks(unique_x_data, labels=unique_x_data, fontsize=10)
    else:
      plt.plot(unique_x_data, pred - center, color=color_base, linewidth=linewidth, alpha=alpha)
      plt.xticks(fontsize=10)

    plt.ylim(min_y, max_y)
    plt.yticks(fontsize=10)
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    if name in categorical_names:
      min_x -= 0.5
      max_x += 0.5
    plt.xlim(min_x, max_x)
    if i % num_cols == 0:
      plt.ylabel("House Price Contribution", fontsize=11)
    plt.xlabel(col_mapping[name], fontsize=11)
  return min_y, max_y


def main():
  repo_root = _repo_root()
  # model_logdir must point to .../training/fold_X/split_Y (contains model_0, model_1, ...)
  default_model_logdir = osp.join(
      repo_root, "repro_runs", "housing_nmodels20", "training", "fold_5", "split_1"
  )
  default_output_dir = osp.join(repo_root, "repro_runs", "housing_nmodels20", "fold_5", "visualization_outputs")

  parser = argparse.ArgumentParser(description="Plot NAM ensemble shape functions.")
  parser.add_argument("--repo_root", default=repo_root)
  parser.add_argument("--model_logdir", default=default_model_logdir)
  parser.add_argument("--dataset_name", default="Housing")
  parser.add_argument("--n_models", type=int, default=20)
  parser.add_argument("--fold_num", type=int, default=5)
  parser.add_argument("--num_folds", type=int, default=5)
  parser.add_argument("--num_splits", type=int, default=3)
  parser.add_argument("--split_idx", type=int, default=1, help="1-based split index")
  parser.add_argument("--activation", default="relu")
  parser.add_argument("--num_basis_functions", type=int, default=64)
  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--shallow", action="store_true")
  parser.add_argument(
      "--run_test_metrics",
      action="store_true",
      help="If set, also compute and save held-out test metrics.",
  )
  parser.add_argument("--output_dir", default=default_output_dir)
  args = parser.parse_args()

  _add_repo_parent_to_path(args.repo_root)
  from neural_additive_models import data_utils  # pylint: disable=import-error
  from neural_additive_models import graph_builder  # pylint: disable=import-error

  os.makedirs(args.output_dir, exist_ok=True)

  is_regression = args.dataset_name in ["Housing", "Fico"]
  data_x, data_y, column_names = data_utils.load_dataset(args.dataset_name)
  col_min_max = load_col_min_max(data_utils, args.dataset_name)

  (x_train_all, y_train_all), test_dataset = data_utils.get_train_test_fold(
      data_x, data_y, fold_num=args.fold_num, num_folds=args.num_folds, stratified=not is_regression
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all, y_train_all, n_splits=args.num_splits, stratified=not is_regression
  )
  split = None
  for _ in range(args.split_idx):
    split = next(data_gen)
  (x_train, _), _ = split

  num_features = data_x.shape[1]
  single_features = np.split(data_x, num_features, axis=1)
  unique_features = [np.unique(x, axis=0) for x in single_features]

  unique_features_original = {}
  single_features_original = {}
  for i, col in enumerate(column_names):
    min_val, max_val = col_min_max[col]
    unique_features_original[col] = inverse_min_max_scaler(unique_features[i][:, 0], min_val, max_val)
    single_features_original[col] = inverse_min_max_scaler(single_features[i][:, 0], min_val, max_val)
  all_indices = compute_all_indices(data_x, unique_features, column_names)

  col_names_map = {
      "Housing": {
          "MedInc": "Median Income",
          "HouseAge": "Median House Age",
          "AveRooms": "# Avg Rooms",
          "AveBedrms": "# Avg Bedrooms",
          "Population": "Block Population",
          "AveOccup": "# Avg Occupancy",
          "Latitude": "Latitude",
          "Longitude": "Longitude",
      }
  }
  if args.dataset_name not in col_names_map:
    # Fallback: use raw feature names when no pretty mapping is defined.
    col_names_map[args.dataset_name] = {c: c for c in column_names}
  categorical_names = []

  per_model_hist_data = []
  per_model_mean_pred = []
  per_model_test_metric = []

  for model_idx in range(args.n_models):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default():
      nn_model = graph_builder.create_nam_model(
          x_train=x_train,
          dropout=args.dropout,
          num_basis_functions=args.num_basis_functions,
          activation=args.activation,
          trainable=False,
          shallow=args.shallow,
          name_scope=f"model_{model_idx}",
      )
      pred_input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
      pred_tensor = nn_model(pred_input_ph, training=False)
      feature_input_phs = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(num_features)]
      feature_tensors = [nn_model.feature_nns[i](feature_input_phs[i], training=nn_model._false) for i in range(num_features)]

      ckpt = resolve_checkpoint_path(osp.join(args.model_logdir, f"model_{model_idx}"))
      saver = tf.compat.v1.train.Saver(var_list=build_restore_var_map(nn_model, ckpt, model_idx))

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, ckpt)
        if args.run_test_metrics:
          test_predictions = get_test_predictions(sess, pred_tensor, pred_input_ph, test_dataset[0])
          per_model_test_metric.append(
              float(graph_builder.calculate_metric(test_dataset[1], test_predictions, regression=is_regression))
          )

        feature_predictions = get_feature_predictions(sess, feature_tensors, feature_input_phs, unique_features)
        model_hist_data = {col: pred for col, pred in zip(column_names, feature_predictions)}
        per_model_hist_data.append(model_hist_data)
        per_model_mean_pred.append(compute_model_mean_pred(model_hist_data, all_indices, column_names))

  avg_hist_data = {col: np.mean(np.stack([m[col] for m in per_model_hist_data], axis=0), axis=0) for col in column_names}
  mean_pred_ensemble = compute_model_mean_pred(avg_hist_data, all_indices, column_names)

  x1, x2 = compute_mean_feature_importance(avg_hist_data, mean_pred_ensemble)
  cols = [col_names_map[args.dataset_name][x] for x in x1]

  prefix = args.dataset_name.lower()
  importance_path = osp.join(args.output_dir, f"{prefix}_feature_importance.png")
  shape_path = osp.join(args.output_dir, f"{prefix}_shape_plots_ensemble.png")
  details_path = osp.join(args.output_dir, f"{prefix}_ensemble_plot_run_details.json")
  text_summary_path = osp.join(args.output_dir, f"{prefix}_test_results.txt")

  plot_mean_feature_importance(args.dataset_name, cols, x2, importance_path)

  num_cols = 4
  num_rows = int(np.ceil(num_features / num_cols))
  fig = plt.figure(figsize=(num_cols * 4.5, num_rows * 4.5), facecolor="w", edgecolor="k")

  min_y, max_y = plot_all_hist(
      dataset_name=args.dataset_name,
      hist_data=avg_hist_data,
      num_rows=num_rows,
      num_cols=num_cols,
      color_base=[0.2, 0.35, 0.9],
      col_names_map=col_names_map,
      categorical_names=categorical_names,
      mean_pred=mean_pred_ensemble,
      unique_features_original=unique_features_original,
      linewidth=3.0,
      alpha=1.0,
  )

  for model_hist, model_mean in zip(per_model_hist_data, per_model_mean_pred):
    plot_all_hist(
        dataset_name=args.dataset_name,
        hist_data=model_hist,
        num_rows=num_rows,
        num_cols=num_cols,
        color_base=[0.3, 0.4, 0.9],
        col_names_map=col_names_map,
        categorical_names=categorical_names,
        mean_pred=model_mean,
        unique_features_original=unique_features_original,
        linewidth=1.0,
        alpha=0.15,
        min_y=min_y,
        max_y=max_y,
    )

  shade_by_density_blocks(
      avg_hist_data,
      num_rows,
      num_cols,
      unique_features_original,
      single_features_original,
      categorical_names,
      n_blocks=20,
  )
  plt.subplots_adjust(hspace=0.23)
  fig.savefig(shape_path, dpi=180)

  metric_name = "RMSE" if is_regression else "AUROC"
  mean_test_metric = float(np.mean(per_model_test_metric)) if per_model_test_metric else None
  payload = {
      "dataset_name": args.dataset_name,
      "n_models": args.n_models,
      "fold_num": args.fold_num,
      "num_folds": args.num_folds,
      "num_splits": args.num_splits,
      "split_idx": args.split_idx,
      "metric_name": metric_name,
      "per_model_test_metric": per_model_test_metric,
      "mean_test_metric": mean_test_metric,
      "run_test_metrics": bool(args.run_test_metrics),
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "importance_png": importance_path,
      "shape_png": shape_path,
  }
  with open(details_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

  summary_lines = [
      f"Dataset: {args.dataset_name}",
      f"Fold: {args.fold_num}/{args.num_folds}",
      f"Models: {args.n_models}",
      f"Importance plot: {importance_path}",
      f"Shape plot: {shape_path}",
  ]
  if per_model_test_metric:
    summary_lines.extend(
        [
            f"Metric: {metric_name}",
            f"Mean test {metric_name}: {mean_test_metric:.6f}",
            "Per-model test: " + ", ".join([f"{x:.6f}" for x in per_model_test_metric]),
        ]
    )
  else:
    summary_lines.append("Test metrics: skipped (--run_test_metrics not set)")
  with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

  if per_model_test_metric:
    print(f"Per-model test {metric_name}: {per_model_test_metric}")
    print(f"Mean test {metric_name}: {mean_test_metric:.6f}")
  else:
    print("Test metrics skipped. Use --run_test_metrics to enable.")
  print(f"Saved: {importance_path}")
  print(f"Saved: {shape_path}")
  print(f"Saved: {details_path}")
  print(f"Saved: {text_summary_path}")


if __name__ == "__main__":
  main()
