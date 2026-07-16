# Project Progress

## Current Phase

- 当前阶段：Phase -1 gate 与 Phase 0.1 已完成；Phase 0.1b trainval manifest v1 协议、完整 scene label/split 统计和 20-scene pilot 已完成，完整 850-scene manifest 写出 pending。
- 当前状态不代表已经实现 neural baseline 或训练模型。

## Confirmed Milestones

- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 与 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已完成 Phase -1.7 人工 meta-action 审核：108 个样本中 `trajectory_alignment_correct=yes` 为 108，`agent_alignment_correct=yes` 为 108；6 类 action 均有审核覆盖。
- 已完成 Phase -1.9 real-data label freeze gate 与 manifest readiness precheck：108 条冻结审核记录均具有完整 3 秒轨迹、存在的相对 `CAM_FRONT` 路径和当前时刻配置半径内的 VRU presence。
- 已完成 Phase 0.1 audited seed-subset manifest、固定 seed scene split、统一评测协议、完整 contract validator 与 Majority Baseline。
- 已完成 Phase 0.1b trainval manifest v1 协议和 `official_train_scene_label_stratified_v1` split：official train 700 scenes → project train 560 / validation 140，official val 150 scenes → project test 150；完整 manifest 尚未写出。
- 新 20-scene pilot 的 train/validation/test scenes 为 13/3/4，扫描 804 samples，纳入 533，排除 271；完整 scene mapping 的 overlap 为 0，六类长尾硬约束均满足。
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
```

trainval pilot 需要预先设置 `NUSCENES_ROOT` 与 `VLA_DERIVED_ROOT`；原始数据和派生 manifest 均不进入 Git。

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
- 正式 trainval schema 为 `phase0_trainval_dataset_manifest_v1`，支持已有 audit token 的 `audited` 完整来源与未匹配记录的 `unaudited/null`；现有 `phase0_audited_seed_subset_v1` 保持兼容。本次固定 20-scene pilot 未抽中历史 audit token，因此实际为 `audited=0`、`unaudited=533`；完整 850-scene 扫描确认 108 个 audit token 全部匹配、0 过滤、0 缺失。
- official train 有效样本 17847 条，六类分布为 `keep=6322`、`accelerate=1857`、`decelerate=2860`、`stop=3044`、`left_lateral=1691`、`right_lateral=2073`；stratified/fixed-random objective 分别为 `0.0020518908` / `0.0605638706`。
- 完整 scene mapping sidecar 位于 `VLA_DERIVED_ROOT` 且不进入 Git：850 scenes 映射为 project train/validation/test `560/140/150`，mapping SHA-256 为 `a96e04aaf068e75b0aa3ecb8412dc5b35fea2412d7090bbee0a6661132923b12`，scene histogram SHA-256 为 `0cee51a6f64e3f2e10382ca7672cc0aa1386065a3fe8a1f927f5469e211a11a2`；六类硬约束均满足。
- 新 trainval pilot 排除统计：`insufficient_remaining_horizon=124`、`timestamp_out_of_tolerance=147`，其余正式排除原因均为 0；motion availability 为 `full=501`、`partial=17`、`unavailable=15`。

## Open Questions / Pending Verification

- 下一步是完整 850-scene trainval manifest 构建与按 split/action/boundary 的分层人工抽检。
- 不启动 Phase 0.2 rule baseline、Qwen3-VL、LoRA、DPO、safety scorer、完整 occupancy network 或 trajectory-level 训练，直至 Phase 0.1b 全量 gate 完成。

## Next Gate

- 保持 `sample_token → CAM_FRONT image → future ego trajectory → nearby 3D agents → one-page visualization` 的可复现核验。
- 保持 official train 700 → project train 560 / validation 140 的 label-stratified scene-level split，并保持 official val 150 → project test 150 且不参与优化。
- 运行完整 trainval manifest v1 构建，复核排除原因、六类分布、motion availability、绝对路径泄漏与 split overlap，再进行分层人工抽检。
