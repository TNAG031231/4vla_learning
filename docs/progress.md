# Project Progress

## Current Phase

- 当前阶段：Phase -1，data alignment and label verification。
- 当前 gate 状态：接近完成，但尚未通过；本状态不授权进入 Phase 0 或启动训练。

## Confirmed Milestones

- 已定义 single-camera、open-loop、6-class meta-action 的 VLA 起步范围，并记录后续 BEV/OCC-aware spatial evaluation 与 waypoint-level trajectory VLA 路线。
- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 与 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已完成 Phase -1.7 人工 meta-action 审核：108 个样本中 `trajectory_alignment_correct=yes` 为 108，`agent_alignment_correct=yes` 为 108；6 类 action 均有审核覆盖。
- 已建立环境检查与 workspace cleanup dry-run 脚本。

## Active Source Files

- `configs/data.yaml`：定义 nuScenes mini 数据入口及 Phase -1 时间窗口、采样间隔和 agent 距离阈值。
- `configs/action_rules.yaml`：定义当前 meta-action v0 阈值、坐标约定和 `label_rule_version`。
- `src/actions/schema.py`：定义唯一的 6 类 action schema。
- `data/inspect_nuscenes_sample.py`：读取 sample、future ego trajectory 和 ego-frame nearby agents。
- `data/derive_meta_action.py`：派生版本化 meta-action 标签。
- `data/verify_labels.py`：生成 Phase -1 单样本 one-page alignment visualization。
- `data/select_manual_review_samples.py`：选择 Phase -1.7 人工审核样本。
- `scripts/check_env.py`：检查项目环境与本地 nuScenes 数据可用性。
- `scripts/clean_workspace.py`：以 dry-run 为默认行为检查临时文件、缓存和日志。

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

## Rule Versions and Audit Evidence

- 当前审核使用 `label_rule_version=phase-1.6-meta-action-v0.1`。
- 当前审核记录的 `safety_rule_version=not_available`；安全评分尚未作为已完成能力记录。
- 108 个已审核样本的 `label_correct` 为 `yes=103`、`no=5`、`uncertain=0`。
- 已确认的剩余错误集中在 `keep` 与 speed-change、以及 `stop` 与 `keep` 的边界案例；规则修订与版本冻结仍待完成。

## Open Questions / Pending Verification

- 根据已审核错误完成 meta-action 规则修订，并重新核验受影响样本。
- 冻结可训练数据版本、`label_rule_version` 和 scene-level split 前，完成 manifest audit。
- Phase -1 gate 通过前，不启动 Phase 0 baseline、LoRA、DPO、GRPO、完整 occupancy network 或 trajectory-level 训练。

## Next Gate

- 保持 `sample_token → CAM_FRONT image → future ego trajectory → nearby 3D agents → one-page visualization` 的可复现核验。
- 完成剩余标签边界规则的修订、审核和版本冻结；保留 6 类 action、有/无 VRU、safe/unsafe 与规则边界的审核覆盖。
- 仅在数据、审核与规则版本均冻结后，按 [project_mvp_plan.md](../project_mvp_plan.md) 的 Phase 0 顺序决定是否开始 baseline。
