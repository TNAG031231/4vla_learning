# Project Progress

## Current Phase

- 当前阶段：Phase -1 gate 已通过；Phase 0 ready but not started。
- 当前状态不代表已经实现 baseline 或训练模型。

## Confirmed Milestones

- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 与 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已完成 Phase -1.7 人工 meta-action 审核：108 个样本中 `trajectory_alignment_correct=yes` 为 108，`agent_alignment_correct=yes` 为 108；6 类 action 均有审核覆盖。
- 已完成 Phase -1.9 real-data label freeze gate 与 manifest readiness precheck：108 条冻结审核记录均具有完整 3 秒轨迹、存在的相对 `CAM_FRONT` 路径和当前时刻配置半径内的 VRU presence。
- 已建立环境检查与 workspace cleanup dry-run 脚本。

## Active Source Files

- `src/actions/schema.py`：定义唯一的 6 类 action schema。
- `data/inspect_nuscenes_sample.py`：读取 sample、future ego trajectory 和 ego-frame nearby agents。
- `data/derive_meta_action.py`：派生版本化 meta-action 标签。
- `data/verify_labels.py`：生成 Phase -1 单样本 one-page alignment visualization。
- `data/select_manual_review_samples.py`：选择 Phase -1.7 人工审核样本。
- `data/validate_label_freeze.py`：重新派生并验收 Phase -1 meta-action v0.2 frozen labels。
- `scripts/check_env.py`：检查项目环境与本地 nuScenes 数据可用性。
- `scripts/clean_workspace.py`：以 dry-run 为默认行为检查临时文件、缓存和日志。

## Stable CLI Commands

以下命令已存在；项目验证必须在 `codex4vla_env` 中运行：

```bash
conda run -n codex4vla_env python scripts/check_env.py
conda run -n codex4vla_env python scripts/clean_workspace.py
conda run -n codex4vla_env python data/validate_label_freeze.py --dataroot data/nuscenes
```

## Data / Manifest Field Contracts

```text
sample_token
scene_token
timestamp
cam_front_path
current_ego_state
future_ego_trajectory
nearby_agents
meta_action
label_rule_version
safety_rule_version
split
```

## Rule Versions and Audit Evidence

- `label_rule_version=phase-1.6-meta-action-v0.2` 已 frozen；Phase -1.8 regression 与 Phase -1.9 freeze gate 均为 `action_match=108/108`。
- frozen distribution：`accelerate=6`、`decelerate=16`、`keep=55`、`left_lateral=5`、`right_lateral=5`、`stop=21`。
- 历史 source audit 的路径、alignment 与 v0.1 rule version 已重新核验；`label_correct=yes=103/no=5` 保持为历史事实，108 条历史 CAM_FRONT 路径均与当前派生路径一致。
- VRU presence（当前 sample、配置半径内）：`yes=89`、`no=19`；strict boundary-flag cases=17，diagnostic cases=46，含 lateral、speed 与 stop 相关 flags。
- `safety_rule_version=not_available`；安全审核从 Phase 1 开始，不是本次 label freeze gate 的完成条件。

## Open Questions / Pending Verification

- 下一步是 Phase 0 scene-level split 和正式 manifest audit。
- 不启动 baseline、LoRA、DPO、GRPO、完整 occupancy network 或 trajectory-level 训练，直至 Phase 0 的数据 split 与 audit 已完成。

## Next Gate

- 保持 `sample_token → CAM_FRONT image → future ego trajectory → nearby 3D agents → one-page visualization` 的可复现核验。
- 从 frozen v0.2 labels 开始，按 [project_mvp_plan.md](../project_mvp_plan.md) 的 Phase 0 顺序创建 scene-level split 并完成正式 manifest audit。
