# Data Debug Report

## Phase -1.7 Manual Meta-Action Audit

- 状态：final confirmed local human audit completed for Phase -1.7；本 PR 提交 `data/phase_1_7_manual_audit.csv`、`data/phase_1_7_lateral_supplement_audit.csv` 和最终统计报告；不提交 `data/outputs`、PNG 或 backup CSV。
- base audit CSV：`data/phase_1_7_manual_audit.csv`，100 samples。
- supplement audit CSV：`data/phase_1_7_lateral_supplement_audit.csv`，8 samples。
- combined samples：108。
- combined `label_correct`：`yes=103`、`no=5`、`uncertain=0`。
- combined `trajectory_alignment_correct`：`yes=108`。
- combined `agent_alignment_correct`：`yes=108`。
- combined `safety_score_reasonable`：`not_available=108`。
- combined action distribution：`accelerate=5`、`decelerate=13`、`keep=60`、`left_lateral=5`、`right_lateral=5`、`stop=20`。
- base error types：`keep_vs_accelerate_confusion=1`、`keep_vs_decelerate_confusion=3`、`stop_vs_keep_confusion=1`。
- supplement lateral audit：`left_lateral=4`、`right_lateral=4`，`label_correct yes=8`。
- 当前规则版本：`label_rule_version=phase-1.6-meta-action-v0.1`；`safety_rule_version=not_available`。
- 结论：lateral coverage 已成功补足，且所有 supplement lateral samples 均通过人工审核。
- 结论：剩余规则错误集中在 `keep` vs speed-change 以及 `stop` / `keep` 边界案例。
- 结论：当前人工审核支持后续规则修订聚焦 `keep` / `decelerate` / `accelerate` 和 `stop` / `keep` 边界。
- Gate：该结果不授权进入 Phase 0，不授权训练。

### Error cases for rule revision

| sample_token | derived_action | reviewed_action | error_type | trajectory_delta_x_m | trajectory_delta_y_m | trajectory_path_length_m | approx_delta_speed_mps | review_note |
|---|---|---|---|---:|---:|---:|---:|---|
| `348c8122f47349429a6cd694dcac86e6` | `keep` | `decelerate` | `keep_vs_decelerate_confusion` | 1.8315346537214245 | -0.059116926 | 1.8567856866310095 | -1.577042834 | approx_delta_speed_mps < -1.0 |
| `73eb876167f4419a9a6ec1a601abdcaf` | `keep` | `decelerate` | `keep_vs_decelerate_confusion` | 2.8120932520088755 | -0.099465042 | 2.8141742095829874 | -1.858184065 | The forward distance is greater than 1 meter and approx_delta_speed_mps is less than -1. |
| `7ae00681137b40f5bd7bef3823a82ee2` | `keep` | `decelerate` | `keep_vs_decelerate_confusion` | 1.0294346082681272 | -0.023958488 | 1.057035712556549 | -1.087600437 | The forward distance is greater than 1 meter and approx_delta_speed_mps is less than -1. |
| `8b6d496ed9d84469b75836ca1c56959f` | `keep` | `stop` | `stop_vs_keep_confusion` | 0.48431042404359476 | -0.011709045 | 0.5123994968448523 | -0.634145323 | The forward distance is less than 0.5 meters. |
| `e6b0b282aa174a978272dc2d0a89d560` | `keep` | `accelerate` | `keep_vs_accelerate_confusion` | 2.5465279202146363 | -0.029450612 | 2.546760692079639 | 1.854919082888664 | approx_delta_speed_mps is greater than +1 |
