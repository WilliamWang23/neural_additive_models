# NAM Project Guide (English)

This is the new English guide. The old `README.md` is intentionally kept unchanged.

## 1) Core Files (after cleanup)

- `data_utils.py`: local dataset loading, preprocessing (one-hot + scaling), fold/split logic.
- `models/`: NAM architecture (`NAM`, `FeatureNN`, activation layers).
- `training/`: training loop, losses, metrics, and model construction helpers.
- `nam_train.py`: low-level training entry with full flag-based configuration.
- `nam_test.py`: standalone test/evaluation entry (testing only, no plotting; supports auto param reuse).
- `plot_nam_ensemble.py`: plotting entry (now decoupled from testing by default).
- `requirements.txt` / `setup.py`: dependency and package metadata.
- `repro_runs/`: training/testing/plot outputs.
- `data/`: local datasets.

## 2) Dependency Flow

- Training flow:
  - `nam_train.py` -> `data_utils.py` + `training/`
  - `training/` -> `models/`
- Testing flow:
  - `nam_test.py` -> `data_utils.py` + `training/` + `models/`
  - Input checkpoints: `repro_runs/.../training/fold_x/split_y/model_i`
- Plotting flow:
  - `plot_nam_ensemble.py` -> `data_utils.py` + `training/` + `models/`
  - Decoupled from test metrics by default

## 3) Simple Run Commands

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

## 4) Plot/Test Decoupling

- `plot_nam_ensemble.py` now focuses on plotting by default.
- To compute test metrics inside plotting, explicitly pass `--run_test_metrics`.
