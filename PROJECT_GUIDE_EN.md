# NAM Project Guide (English)

This is the new English guide. The old `README.md` is intentionally kept unchanged.

## 1) Core Files (after cleanup)

- `data_utils.py`: local dataset loading, preprocessing (one-hot + scaling), fold/split logic.
- `models.py`: NAM architecture (`NAM`, `FeatureNN`, activation layers).
- `graph_builder.py`: TF1-compatible graph construction (loss/optimizer/metrics).
- `nam_train.py`: low-level training entry with full flag-based configuration.
- `nam_test.py`: standalone test/evaluation entry (testing only, no plotting; supports auto param reuse).
- `nam_train_test.py`: training pipeline smoke test (quick runnability check, not for formal experiments).
- `plot_nam_ensemble.py`: plotting entry (now decoupled from testing by default).
- `activate_nam_gpu_env.ps1` / `verify_nam_gpu.ps1`: GPU environment activation/verification.
- `requirements.txt` / `setup.py`: dependency and package metadata.
- `repro_runs/`: training/testing/plot outputs.
- `data/`: local datasets.

## 2) Dependency Flow

- Training flow:
  - `nam_train.py` -> `data_utils.py` + `graph_builder.py`
  - `graph_builder.py` -> `models.py`
- Smoke-test flow:
  - `nam_train_test.py` -> `nam_train.py` (tiny-epoch classification/regression sanity check)
- Testing flow:
  - `nam_test.py` -> `data_utils.py` + `graph_builder.py` + `models.py`
  - Input checkpoints: `repro_runs/.../training/fold_x/split_y/model_i`
- Plotting flow:
  - `plot_nam_ensemble.py` -> `data_utils.py` + `graph_builder.py` + `models.py`
  - Decoupled from test metrics by default

## 3) Simple Run Commands

> Activate GPU env first in each new terminal:

```powershell
& "e:\Code\Projects\neural_additive_models\activate_nam_gpu_env.ps1"
cd "e:\Code\Projects\neural_additive_models"
```

### 3.1 Training (manual flags)

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" -m neural_additive_models.nam_train --logdir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training" --dataset_name=Housing --regression=True --cross_val=True --activation=relu --shallow=False --batch_size=1024 --training_epochs=1000 --early_stopping_epochs=60 --decay_rate=0.995 --learning_rate=0.00674 --output_regularization=0.001 --l2_regularization=0.000001 --dropout=0.0 --feature_dropout=0.0 --n_models=5 --num_splits=3 --fold_num=1 --num_basis_functions=64
```

After training, `training_params.json` is saved under `repro_runs/<run_name>/training/fold_X/` for test reuse.

### 3.2 Testing (standalone, no plotting)

`--run_dir` accepts two path formats (either works):

```powershell
# Format 1: point to fold dir (recommended, matches actual layout)
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_test.py --run_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training\fold_1"

# Format 2: point to fold name under run root (backward compatible)
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_test.py --run_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\fold_1"
```

Outputs go to `run_dir/test_outputs/`:
- `*_test_details.json`
- `*_test_results.txt`

You can override auto-resolved values with explicit flags (`--model_logdir`, `--dataset_name`, etc.).

### 3.3 Plotting (optional)

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" plot_nam_ensemble.py --model_logdir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training\fold_1\split_1" --dataset_name=Housing --n_models=5 --fold_num=1 --output_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\fold_1\visualization_outputs"
```

Note: `--model_logdir` must point to the directory containing `model_0`, `model_1`, etc. (i.e. `.../training/fold_X/split_Y`).

### 3.4 Training smoke test (optional)

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_train_test.py
```

Notes:
- Runs very short training on `BreastCancer` and `Housing` just to verify the training stack works.
- Intended for environment/regression sanity checks, not for official experiment outputs.

## 4) Plot/Test Decoupling

- `plot_nam_ensemble.py` now focuses on plotting by default.
- To compute test metrics inside plotting, explicitly pass `--run_test_metrics`.

