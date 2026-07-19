# Project Progress

## Current Phase

- 当前阶段：Phase -1、Phase 0.1 与 Phase 0.1b gate 均已完成；Phase 0.1b dataset protocol v1 已通过全量构建、自动验收、排除原因诊断与 train/validation 视觉审核并冻结。
- 当前状态不代表已经实现 neural baseline 或训练模型。

## Confirmed Milestones

- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 与 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已完成 Phase -1.7 人工 meta-action 审核：108 个样本中 `trajectory_alignment_correct=yes` 为 108，`agent_alignment_correct=yes` 为 108；6 类 action 均有审核覆盖。
- 已完成 Phase -1.9 real-data label freeze gate 与 manifest readiness precheck：108 条冻结审核记录均具有完整 3 秒轨迹、存在的相对 `CAM_FRONT` 路径和当前时刻配置半径内的 VRU presence。
- 已完成 Phase 0.1 audited seed-subset manifest、固定 seed scene split、统一评测协议、完整 contract validator 与 Majority Baseline。
- 已完成并冻结 Phase 0.1b trainval dataset protocol v1：`horizon_sec=3.0`、`sample_interval_sec=0.5`、`time_tolerance_sec=0.075`、`label_rule_version=phase-1.6-meta-action-v0.2`、`split_strategy_version=official_train_scene_label_stratified_v1`、`split_seed=20260710`。
- 完整 850-scene split 为 project train/validation/test `560/140/150`；正式 manifest 扫描 34,149 samples，纳入 21,646 条（train 14,253 / validation 3,594 / test 3,799），排除 12,503 条。
- 已完成 Phase 0.2a current/past ego-motion 输入审计：train/validation/test 的 `full/partial/unavailable` 分别为 `13476/392/385`、`3401/99/94`、`3594/106/99`；输入合同仅包含 speed、longitudinal acceleration、yaw rate、availability 与对应 past interval，test label 未用于统计或调参。Phase 0.2 rule predictor 仍为 planned。
- 已建立环境检查与 workspace cleanup dry-run 脚本。

## Active Source Files

- `src/actions/schema.py`：定义唯一的 6 类 action schema。
- `data/inspect_nuscenes_sample.py`：读取 sample、future ego trajectory 和 ego-frame nearby agents。
- `data/derive_meta_action.py`：派生版本化 meta-action 标签。
- `data/verify_labels.py`：生成 Phase -1 单样本 one-page alignment visualization。
- `data/select_manual_review_samples.py`：选择 Phase -1.7 人工审核样本。
- `data/validate_label_freeze.py`：重新派生并验收 Phase -1 meta-action v0.2 frozen labels。
- `data/build_phase0_manifest.py`：构建 audited seed-subset manifest。
- `data/build_trainval_manifest.py`：按官方 scene split 构建 trainval manifest v1，并支持同一 builder 的 pilot 模式。
- `src/phase0/manifest.py`：提供 audited/trainval 共用的 pose、past-only motion、坐标元数据与 JSONL 序列化逻辑。
- `src/phase0/protocol.py`：提供 scene split、双 manifest schema validator 与统一评测协议。
- `src/baselines/majority.py`：提供 Phase 0.1 Majority Baseline。
- `scripts/check_env.py`：检查项目环境与本地 nuScenes 数据可用性。
- `scripts/clean_workspace.py`：以 dry-run 为默认行为检查临时文件、缓存和日志。

## Stable CLI Commands

以下命令已存在；项目验证必须在 `codex4vla_env` 中运行：

```bash
conda run -n codex4vla_env python scripts/check_env.py
conda run -n codex4vla_env python scripts/clean_workspace.py
conda run -n codex4vla_env python data/validate_label_freeze.py --dataroot data/nuscenes
conda run -n codex4vla_env python data/build_trainval_manifest.py --config configs/trainval_manifest.yaml --pilot
conda run -n codex4vla_env python scripts/audit_ego_motion_inputs.py --config configs/phase0_2_ego_motion.yaml
```

trainval pilot 与 Phase 0.2a 输入审计需要预先设置 `NUSCENES_ROOT` 与 `VLA_DERIVED_ROOT`；原始数据、派生 manifest 与审计 JSON 均不进入 Git。

## Data / Manifest Field Contracts

```text
sample_token
scene_token
timestamp
cam_front_path
current_ego_pose
current_ego_motion
coordinate_metadata
future_ego_trajectory
nearby_agents
meta_action
label_rule_version
safety_rule_version
split
official_split
split_seed
split_strategy_version
split_mapping_sha256
manifest_schema_version
audit_status
source_audit_record
```

## Rule Versions and Audit Evidence

- `label_rule_version=phase-1.6-meta-action-v0.2` 已 frozen；Phase -1.8 regression 与 Phase -1.9 freeze gate 均为 `action_match=108/108`。
- frozen distribution：`accelerate=6`、`decelerate=16`、`keep=55`、`left_lateral=5`、`right_lateral=5`、`stop=21`。
- 历史 source audit 的路径、alignment 与 v0.1 rule version 已重新核验；`label_correct=yes=103/no=5` 保持为历史事实，108 条历史 CAM_FRONT 路径均与当前派生路径一致。
- VRU presence（当前 sample、配置半径内）：`yes=89`、`no=19`；strict boundary-flag cases=17，diagnostic cases=46，含 lateral、speed 与 stop 相关 flags。
- `safety_rule_version=not_available`；安全审核从 Phase 1 开始，不是本次 label freeze gate 的完成条件。
- 正式 trainval schema 为 `phase0_trainval_dataset_manifest_v1`，支持已有 audit token 的 `audited` 完整来源与未匹配记录的 `unaudited/null`；现有 `phase0_audited_seed_subset_v1` 保持兼容。完整 manifest 为 `audited=108`、`unaudited=21538`，108 个历史 audit token 全部匹配、0 过滤、0 缺失。
- official train 有效样本 17847 条，六类分布为 `keep=6322`、`accelerate=1857`、`decelerate=2860`、`stop=3044`、`left_lateral=1691`、`right_lateral=2073`；stratified/fixed-random objective 分别为 `0.0020518908` / `0.0605638706`。
- 正式 manifest 与 mapping sidecar 位于 `$VLA_DERIVED_ROOT/phase_0_1b/trainval_manifest_v1/`，不进入 Git 且不得由后续实验覆盖。manifest 文件 SHA-256 为 `60517f985fec8fe3977a31660a5204942e9fd36baf09ea4d950328b1f225d1b3`，sidecar 文件 SHA-256 为 `fa94cc4c1d7b7b24476d6043cd132fa0b7fa5ace2285a82200c363a3d3501be8`，内部 mapping SHA-256 为 `a96e04aaf068e75b0aa3ecb8412dc5b35fea2412d7090bbee0a6661132923b12`，scene histogram SHA-256 为 `0cee51a6f64e3f2e10382ca7672cc0aa1386065a3fe8a1f927f5469e211a11a2`。
- 全量排除统计为 `insufficient_remaining_horizon=5210`、`timestamp_out_of_tolerance=7293`。前者已完成专项诊断；后者未发现时间单位、timestamp source、nearest-search、scene-chain 或浮点边界实现错误。
- streaming manifest validation、exclusion diagnostic 与 rare-class constraints 均通过；duplicate sample token、scene split overlap、绝对路径泄漏、缺失 CAM_FRONT、official val → project test 违规和 official train → project test 违规均为 0。
- 0.100 秒 nearest candidate 可恢复更多样本，exact-grid interpolation 标签总体一致率为 98.0458%，但 validation `decelerate` 一致率为 91.89%，仍存在边界风险，因此正式协议保持 0.075 秒。exact-grid interpolation 作为可选 v1.1 数据增强 backlog，不阻塞 Phase 0.2。
- visual protocol comparison template 已通过；首批 train/validation 可视化未发现明显轨迹方向、左右坐标或时间顺序错误。test 未用于协议选择且继续封存，0.100/exact-grid 未成为正式协议。

## Open Questions / Pending Verification

- exact-grid interpolation v1.1 是可选数据增强 backlog；如后续评估，应保持现有 v1 manifest、sidecar 与 test split 不变，并单独提升协议版本。
- Phase 0.1b gate 已完成；下一阶段可按既定顺序进入 Phase 0.2，但本次仅冻结数据协议，不实现 rule baseline 或任何模型训练。

## Next Gate

- Phase 0.2 rule baseline 只允许使用 inference-time current/past ego state，不得读取 future ego trajectory、derived meta-action 或 test label。
- 后续实验复用已冻结的 train/validation/test scene mapping、action vocabulary 与 manifest v1；不得根据模型、prompt 或标签分布重新调整 test split。
- test 继续封存；Phase 0.2 的 sample-level 输出、输入字段审计与统一 action metrics 完成前，不进入 Phase 0.3。
