# Data Debug Report

## Phase -1.7 Manual Meta-Action Audit

- 状态：pending human review；当前仅生成可填写审核表，尚未产生人工确认准确率或错误率。
- 人工结果保护：已将当前 `data/phase_1_7_manual_audit.csv` 复制为 `data/phase_1_7_manual_audit_reviewed_backup.csv`；按 `reviewed_action`、`label_correct` 或 `review_note` 非空/非 `uncertain` 统计，当前已填写人工审核行数为 52。
- 输入检查：本地存在 smoke 输入 `data/outputs/phase_1_6_meta_action_v0/derived_meta_action.jsonl` 和 `data/outputs/phase_1_5_manual_review_smoke_v2/review_manifest.jsonl`，各 12 条。
- Phase -1.7 mini 旧输入：`data/outputs/phase_1_7_manual_audit_mini/review_manifest.jsonl` 和 `data/outputs/phase_1_7_meta_action_mini/derived_meta_action.jsonl` 各 100 条；该 100 条 derived label 不是 nuScenes mini 完整 3 秒 trajectory 的 full pool。
- 当前审核表：`data/phase_1_7_manual_audit.csv`。
- 当前 base CSV：100 行，100 个 `visualization_path` 引用，引用 PNG 均存在；`data/outputs/phase_1_7_manual_audit_mini/visualizations/` 下有 154 张 PNG，因此 base audit 当前存在 54 张 orphan PNG。
- base orphan PNG 对应 full pool action distribution：`accelerate=13`、`keep=7`、`left_lateral=4`、`right_lateral=11`、`stop=16`；另有 3 个 orphan PNG 不在当前完整 3 秒 trajectory full pool 中。
- 完整 3 秒 trajectory 样本：`valid_3s_future_trajectory_samples=302`，由 `data/derive_meta_action.py --all-valid-samples --dataroot data/nuscenes` 重新派生。
- full derived meta-action pool distribution：`accelerate=39`、`decelerate=22`、`keep=111`、`left_lateral=31`、`right_lateral=24`、`stop=75`。
- 当前 base CSV selected action distribution：`accelerate=5`、`decelerate=13`、`keep=60`、`left_lateral=1`、`right_lateral=1`、`stop=20`。
- lateral 覆盖状态：full pool 中 `left_lateral=31`、`right_lateral=24`，base CSV 仅各 1 条；已从 full pool 中按每类至少 `min(5, available_count)` 的目标补足到 base + supplement 各 5 条。
- 补充审核建议：建议补审 lateral supplement；已新增 `data/phase_1_7_lateral_supplement_audit.csv`，共 8 行，其中 `left_lateral=4`、`right_lateral=4`，不包含 base CSV 已有 `sample_token`。
- supplement visualization 检查：`data/outputs/phase_1_7_lateral_supplement_audit_mini/visualizations/` 中有 8 张 PNG；supplement CSV 引用 8 张，orphan PNG 为 0，missing PNG 为 0。
- 当前 split：审核 CSV 写入 `phase-1.7-mini-audit`；该标记仅用于本次 nuScenes mini 人工抽检，不替代未来正式 scene-level train/val/test split。
- 当前 VRU / safety 覆盖：输入中未提供显式 VRU 标记或 safety score / penalty 字段，审核表中记录为 `not_available`。
- 当前规则版本：`label_rule_version=phase-1.6-meta-action-v0.1`；`safety_rule_version=not_available`。
- 待人工填写字段：`reviewed_action`、`label_correct`、`trajectory_alignment_correct`、`agent_alignment_correct`、`safety_score_reasonable`、`error_type`、`review_note`。
- 统计命令：`conda run -n codex4vla_env python data/summarize_manual_review.py data/phase_1_7_manual_audit.csv`。
- 当前 base CSV 仍是 partial human review；未审样本不能视为通过，不能进入 Phase 0 或训练。
