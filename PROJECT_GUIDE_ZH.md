# NAM 项目说明（中文）

本文件是新的中文说明，旧 `README.md` 保留不变。

## 1) 核心文件

- `data_utils.py`：读取本地数据、特征预处理（含 one-hot 与缩放）、fold/split 划分。
- `models.py`：NAM 网络结构定义（`NAM`、`FeatureNN`、激活层）。
- `graph_builder.py`：提供基于 PyTorch 的兼容辅助函数，用于组装模型、loss 和 metric。
- `nam_train.py`：底层训练入口（完整超参数由 flags 控制）。
- `nam_test.py`：独立测试入口（只做测试，不画图；可自动读取训练参数）。
- `nam_train_test.py`：训练链路冒烟测试（快速验证代码能否跑通，不用于正式实验）。
- `plot_nam_ensemble.py`：可视化脚本（默认只画图；可选 `--run_test_metrics`）。
- `activate_nam_gpu_env.ps1` / `verify_nam_gpu.ps1`：GPU 环境激活与校验。
- `requirements.txt` / `setup.py`：依赖与安装配置。
- `repro_runs/`：训练、测试、可视化结果输出目录。
- `data/`：本地数据集目录。

## 2) 文件依赖关系（从训练到测试）

- 训练链路：
  - `nam_train.py` -> `data_utils.py` + `graph_builder.py`
  - `graph_builder.py` -> `models.py`
- 冒烟测试链路：
  - `nam_train_test.py` -> `nam_train.py`（内部用小 epoch 跑分类/回归最小流程）
- 测试链路：
  - `nam_test.py` -> `data_utils.py` + `graph_builder.py` + `models.py`
  - 输入是 `repro_runs/.../training/fold_x/split_y/model_i` 的 checkpoint
- 画图链路：
  - `plot_nam_ensemble.py` -> `data_utils.py` + `graph_builder.py` + `models.py`
  - 默认与测试逻辑解耦（默认不计算 test metric）

## 3) 启动方式（尽量简单）

> 先激活 GPU 环境（每个新终端都要做一次）

```powershell
& "e:\Code\Projects\neural_additive_models\activate_nam_gpu_env.ps1"
cd "e:\Code\Projects\neural_additive_models"
```

### 3.1 训练（手动参数入口）

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" -m neural_additive_models.nam_train --logdir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training" --dataset_name=Housing --regression=True --cross_val=True --activation=relu --shallow=False --batch_size=1024 --training_epochs=1000 --early_stopping_epochs=60 --decay_rate=0.995 --learning_rate=0.00674 --output_regularization=0.001 --l2_regularization=0.000001 --dropout=0.0 --feature_dropout=0.0 --n_models=5 --num_splits=3 --fold_num=1 --num_basis_functions=64
```

训练完成后会在 `repro_runs/<run_name>/training/fold_X/` 下生成 `training_params.json`，供测试脚本复用。

### 3.2 测试（独立脚本，不依赖画图）

`--run_dir` 支持两种路径格式（二选一）：

```powershell
# 格式一：指向 fold 目录（推荐，与实际目录一致）
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_test.py --run_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training\fold_1"

# 格式二：指向 run 根下的 fold 名（兼容旧用法）
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_test.py --run_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\fold_1"
```

测试输出写入 `run_dir/test_outputs/`：
- `*_test_details.json`
- `*_test_results.txt`

如需覆盖自动参数，可额外传 `--model_logdir`、`--dataset_name`、`--n_models` 等参数。

### 3.3 画图（可选）

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" plot_nam_ensemble.py --model_logdir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\training\fold_1\split_1" --dataset_name=Housing --n_models=5 --fold_num=1 --output_dir="e:\Code\Projects\neural_additive_models\repro_runs\housing_nmodels5\fold_1\visualization_outputs"
```

说明：`--model_logdir` 必须指向包含 `model_0`、`model_1` 等子目录的路径（即 `.../training/fold_X/split_Y`）。

### 3.4 训练链路冒烟测试（可选）

```powershell
& "C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe" nam_train_test.py
```

说明：
- 该脚本会用极小训练轮次快速跑通 `BreastCancer` 和 `Housing` 的训练流程；
- 主要用于“环境/代码回归检查”，不是正式实验入口，也不会产出你要汇总的正式结果文件。

## 4) 画图脚本与测试分离说明

- `plot_nam_ensemble.py` 默认只出图，不再默认跑测试指标。
- 如确实需要在画图时顺带算测试分数，显式加 `--run_test_metrics`。
