# Project Progress

## Current Phase

- 当前阶段：Phase -1，data alignment and visualization。
- 当前 gate 状态：in progress；Phase -1 gate 尚未通过。

## Completed Milestones

- 已定义 open-loop、single-camera、6-class meta-action MVP 范围与阶段门槛。
- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 和 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已建立环境检查与 workspace cleanup dry-run 脚本。

## Active Source Files

- `configs/data.yaml`：定义 nuScenes mini 数据入口及 Phase -1 时间窗口、采样间隔和 agent 距离阈值。
- `data/inspect_nuscenes_sample.py`：读取 sample、future ego trajectory 和 ego-frame nearby agents。
- `data/verify_labels.py`：生成 Phase -1 单样本 one-page alignment visualization。
- `scripts/check_env.py`：检查项目环境与本地 nuScenes 数据可用性。
- `scripts/clean_workspace.py`：以 dry-run 为默认行为检查临时文件、缓存和日志。
- `tests/test_inspect_nuscenes_sample.py`：覆盖数据读取、坐标变换和 trajectory/agent 行为。
- `tests/test_verify_labels.py`：覆盖 one-page visualization 的路径、摘要和渲染行为。

## Stable CLI Commands

以下命令已存在；项目验证必须在 `codex4vla_env` 中运行：

```bash
conda run -n codex4vla_env python scripts/check_env.py
conda run -n codex4vla_env python scripts/clean_workspace.py
```

## Data / Manifest Field Contracts

```text
sample_token
scene_token
timestamp
cam_front_path
future_ego_trajectory
nearby_agents
meta_action
label_rule_version
safety_rule_version
split
```

## Rule Versions

- `label_rule_version`：尚未冻结；当前 visualization 记录为 `unavailable`。
- `safety_rule_version`：尚未冻结；当前 visualization 记录为 `unavailable`。

## Open Questions / Pending Verification

- Phase -1 的坐标系、时间顺序和单位仍需完成规定范围的可视化核验。
- 至少 100 个样本的人工抽检记录尚未完成。
- Meta-action label 与 safety scorer 的规则版本尚未冻结。

## Next Gate

- 完成 `sample_token → CAM_FRONT image → future ego trajectory → nearby 3D agents → one-page visualization` 的可复现核验。
- 完成至少 100 个覆盖 6 类 action、有/无 VRU、safe/unsafe 和规则边界的样本抽检，并记录错误类型。
- 仅在 Phase -1 gate 通过后进入后续 baseline 或训练阶段。
