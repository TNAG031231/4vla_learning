# Data Debug Report

## Phase -1.7 Manual Meta-Action Audit

- 状态：pending human review；当前仅生成可填写审核表，尚未产生人工确认准确率或错误率。
- 输入检查：本地存在 smoke 输入 `data/outputs/phase_1_6_meta_action_v0/derived_meta_action.jsonl` 和 `data/outputs/phase_1_5_manual_review_smoke_v2/review_manifest.jsonl`，各 12 条。
- Phase -1.7 mini 输入：已从 nuScenes mini 生成 `data/outputs/phase_1_7_manual_audit_mini/review_manifest.jsonl` 和 `data/outputs/phase_1_7_meta_action_mini/derived_meta_action.jsonl`，各 100 条。
- 当前审核表：`data/phase_1_7_manual_audit.csv`。
- 当前抽样结果：目标 100 条；nuScenes mini 可用 3 秒 future trajectory 样本 302 条；当前 Phase -1.7 mini derived label 候选池 100 条，已选择 100 条。
- 当前候选 action distribution：`accelerate=5`、`decelerate=13`、`keep=60`、`left_lateral=1`、`right_lateral=1`、`stop=20`。
- 当前 selected action distribution：`accelerate=5`、`decelerate=13`、`keep=60`、`left_lateral=1`、`right_lateral=1`、`stop=20`。
- 当前 rare-class coverage warning：候选池中 `left_lateral=1`、`right_lateral=1`，均少于每类 5 条目标；抽样已选择全部可用 lateral 样本。
- 当前 split：审核 CSV 写入 `phase-1.7-mini-audit`；该标记仅用于本次 nuScenes mini 人工抽检，不替代未来正式 scene-level train/val/test split。
- 当前 VRU / safety 覆盖：输入中未提供显式 VRU 标记或 safety score / penalty 字段，审核表中记录为 `not_available`。
- 当前规则版本：`label_rule_version=phase-1.6-meta-action-v0.1`；`safety_rule_version=not_available`。
- 待人工填写字段：`reviewed_action`、`label_correct`、`trajectory_alignment_correct`、`agent_alignment_correct`、`safety_score_reasonable`、`error_type`、`review_note`。
- 统计命令：`conda run -n codex4vla_env python data/summarize_manual_review.py data/phase_1_7_manual_audit.csv`。
