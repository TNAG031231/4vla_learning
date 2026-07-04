# Phase -1 环境配置记录

## 环境信息

- 新环境：`codex4vla_env`
- 复制来源：`bishe_env`
- Python：3.11.14
- PyTorch：2.9.1
- CUDA available：`False`
- CUDA version：`None`
- 平台：macOS arm64

`codex4vla_env` 通过 `conda create --name codex4vla_env --clone bishe_env` 创建。克隆前后 PyTorch 和 CUDA 状态一致，未在 `bishe_env` 中安装、卸载或降级任何包。

## Phase -1 依赖

环境已安装并通过 `pip check`：

- 数据集：`nuscenes-devkit 1.2.0`
- 数值与数据处理：`numpy 1.26.4`、`pandas 3.0.1`、`scipy 1.17.1`、`scikit-learn 1.8.0`
- 图像与可视化：`opencv-python 4.10.0.84`、`opencv-python-headless 4.11.0.86`、`pillow 12.1.1`、`matplotlib 3.10.8`
- 几何：`pyquaternion 0.9.9`、`shapely 2.0.7`
- 工具：`tqdm 4.67.3`、`pyyaml 6.0.3`、`rich 15.0.0`
- 测试与开发：`pytest 9.1.1`、`ipykernel 7.3.0`

Jupyter kernel 已注册为 `Python (codex4vla_env)`。

## 未安装的训练依赖

当前阶段未安装或配置 `transformers` 最新主线、Qwen3-VL/LLaVA 训练依赖、DeepSpeed、FlashAttention、bitsandbytes、TRL/DPO、Weights & Biases、CARLA 和 nuPlan devkit。进入 Week 3 / Week 4 并锁定实际 checkpoint 后再单独评估这些依赖。

## 使用方法

激活环境：

```bash
conda activate codex4vla_env
```

设置 nuScenes 数据根目录：

```bash
export NUSCENES_ROOT="$PWD/data/nuscenes"
```

验证环境和数据：

```bash
python scripts/check_env.py
```

如果未设置 `NUSCENES_ROOT`，脚本只提示设置方式，不会以错误状态退出。若数据根目录包含 `v1.0-mini`，脚本会初始化 NuScenes 并打印 sample、scene 数量；若包含 `can_bus/`，还会打印其中的文件数量。

## 数据读取验证

使用环境变量将 `NUSCENES_ROOT` 指向仓库内已忽略的 `data/nuscenes` 后，实测结果为：

- NuScenes mini 初始化成功；
- sample：404；
- scene：10；
- `can_bus/` 存在，递归统计文件数：7833；
- 最终输出：`Environment check: PASS`。

## 安装与冲突记录

- Conda clone 成功，未使用降级创建方案。
- `pip check` 输出 `No broken requirements found.`。
- `nuscenes-devkit` 安装保留了克隆环境中的 PyTorch、NumPy、Pandas、SciPy 和 Matplotlib 版本。
- `nuscenes-devkit 1.2.0` 依赖 `opencv-python-headless`，而源环境已有 `opencv-python`；当前两个 distribution 共存，`cv2` 实际导入版本为 4.11.0，功能验证通过，未发现 pip 依赖冲突。
- 未安装指定的重型训练依赖；检查源环境和新环境均未发现对应包。
- `bishe_env` 的本地审计快照为 `env_bishe_pip_freeze.txt` 和 `env_bishe_conda_export.yaml`，两者已加入 `.gitignore`，避免提交本机环境路径与完整包清单。
