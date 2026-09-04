# Project Progress

## Current Phase

- 当前阶段：Phase -1、Phase 0.1 与 Phase 0.1b gate 均已完成；Phase 0.2d sealed one-shot evaluation 已调用一次，但因 validation artifact schema adapter 缺失而在正式输出写盘前失败。test split 已永久消费，没有形成可发布的正式 test metrics。Phase 0.3 overall 状态为 `active`。
- Phase 0.3a-1、Phase 0.3a-2、Phase 0.3b、Phase 0.3c、Phase 0.3d 与 Phase 0.3e-1 均为 `completed`；Phase 0.3e-2 real interface verification 已通过，当前等待 Draft PR #37 人工 review / merge。
- 当前状态已确认 Qwen3-VL full-validation zero-shot baseline、real LoRA smoke training chain，以及 pinned processor / planning visual feature interface；不代表 full-validation LoRA performance、test performance、full-validation feature extraction 或最终 VLA model training 已完成。

## Confirmed Milestones

- 已建立 `sample_token` 对应的 `CAM_FRONT`、future ego trajectory 与 nearby 3D agents 读取能力。
- 已实现单样本 one-page alignment visualization，并有对应单元测试。
- 已完成 Phase -1.7 人工 meta-action 审核：108 个样本中 `trajectory_alignment_correct=yes` 为 108，`agent_alignment_correct=yes` 为 108；6 类 action 均有审核覆盖。
- 已完成 Phase -1.9 real-data label freeze gate 与 manifest readiness precheck：108 条冻结审核记录均具有完整 3 秒轨迹、存在的相对 `CAM_FRONT` 路径和当前时刻配置半径内的 VRU presence。
- 已完成 Phase 0.1 audited seed-subset manifest、固定 seed scene split、统一评测协议、完整 contract validator 与 Majority Baseline。
- 已完成并冻结 Phase 0.1b trainval dataset protocol v1：`horizon_sec=3.0`、`sample_interval_sec=0.5`、`time_tolerance_sec=0.075`、`label_rule_version=phase-1.6-meta-action-v0.2`、`split_strategy_version=official_train_scene_label_stratified_v1`、`split_seed=20260710`。
- 完整 850-scene split 为 project train/validation/test `560/140/150`；正式 manifest 扫描 34,149 samples，纳入 21,646 条（train 14,253 / validation 3,594 / test 3,799），排除 12,503 条。
- 已完成 Phase 0.2a current/past ego-motion 输入审计：train/validation/test 的 `full/partial/unavailable` 分别为 `13476/392/385`、`3401/99/94`、`3594/106/99`；输入合同仅包含 speed、longitudinal acceleration、yaw rate、availability 与对应 past interval，test label 未用于统计或调参。
- 已实现 Phase 0.2b deterministic ego-motion rule baseline：固定 625-candidate grid 在 validation 选择 `stop=0.2 m/s`、`lateral=0.05 rad/s`、`accelerate=0.5 m/s²`、`decelerate=0.3 m/s²`，macro-F1/accuracy 为 `0.615681/0.623817`；Majority Baseline 为 `0.087186/0.354201`。该阶段未使用 test。
- 已建立环境检查与 workspace cleanup dry-run 脚本。

## Phase 0.3 Status and AutoDL Evidence

### Phase 0.3a-1 AutoDL Preflight

- 状态为 `completed`：`status=passed`、`exit_code=0`、`git_commit=76f6688a140438b75e563d9aabb0c820469fad22`。
- 硬件为 `1 × NVIDIA vGPU-32GB`；frozen manifest SHA matched。
- 隔离证据为 `manifest records parsed=0`、`test evaluation performed=false`。

### Phase 0.3a-2 Qwen3-VL Smoke

- 状态为 `completed`；执行代码 merge commit 为 `a7bd0f89f94ebc360b2cb92fd859e2642d77294e`。
- Artifact：`$VLA_DERIVED_ROOT/phase_0_3/qwen3vl_smoke_v0_1/smoke_result.json`。
- 模型：`model_id=Qwen/Qwen3-VL-4B-Instruct`；configured/resolved revision 均为 `ebb281ec70b05090aa6165b016eac8ec08e71b17`；`dtype=torch.bfloat16`；`device=cuda:0`；attention implementation 为 `sdpa`；`model load performed=true`。
- 样本：`split=validation`；`sample_token=43c8e708f0734128be1b7a5d56958f6c`；`scene_token=0053e9c440a94c1b84bd9c4223efc4b0`；`CAM_FRONT=samples/CAM_FRONT/n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708431412404.jpg`。
- Processor：image size 为 `[1600, 900]`；`input_ids shape=[1, 1438]`；`attention_mask shape=[1, 1438]`；`pixel_values shape=[5600, 1536]`；`image_grid_thw shape=[1, 3]`；`pixel_values dtype=torch.float32`；`pixel_values device=cuda:0`。
- Generation：`do_sample=false`；`num_beams=1`；`max_new_tokens=16`；`generation completed=true`；`generated token shape=[1, 3]`；`retry_count=0`；`raw_output="driving"`；`parser_success=false`；`invalid_reason=output_not_exact_allowed_action`；artifact `status=completed_with_invalid_output`；smoke `exit_code=0`。
- 该 invalid output 是 strict parser 的预期可审计路径，不构成 Phase 0.3a-2 失败；不得把 `driving` 映射为任何合法动作。该结果不构成 zero-shot 准确率或模型性能结论。
- Phase 0.3c 必须使用预先冻结的有限 prompt variants，不得根据该单样本反复调整 prompt。
- Timing：processor `28.51041942834854` 秒；model load `171.9564354941249` 秒；generation `3.835577502846718` 秒；total `273.8810808286071` 秒。该结果来自首次冷缓存运行，不是稳定延迟 benchmark。
- CUDA memory：allocated after load `8933453824` bytes；peak allocated `9332952576` bytes；peak reserved `9460252672` bytes。该证据只证明单样本推理可运行，不用于推断 LoRA 训练 batch size。
- Isolation：`manifest_records_parsed=0`；`locator_records_parsed=1`；`test_records_read=0`；`test_images_opened=0`；`test_labels_read=0`；`test_evaluation_performed=false`；`validation_label_used_as_model_input=false`；`failures=[]`；`warnings=[]`。

### Phase 0.3b Dataset Adapter

- 状态为 `completed`；Phase 0.3c 的两个 input variants 均消费该 adapter 的 3,594 条 validation records，并保持相同 sample set。

### Phase 0.3c Full-validation Zero-shot Baseline

- 状态为 `completed`；正式 AutoDL artifacts 已生成并人工核验。两组实验均为 `split=validation`、`sample_count=3594`，不构成 test performance。
- 共同 provenance：`model_id=Qwen/Qwen3-VL-4B-Instruct`；model/processor revision 均为 `ebb281ec70b05090aa6165b016eac8ec08e71b17`；`execution_git_commit=43756aa8487c5d7b760f3e19c5e6ce6d602ae6a5`；`prompt_version=phase0.3c-zero-shot-v0.1`；`parser_version=phase0.3a2-strict-legacy-action-v0.1`；`generation_config_version=phase0.3a2-deterministic-generation-v0.1`。
- Generation：`do_sample=false`；`num_beams=1`；`max_new_tokens=16`。

| input variant | accuracy | macro-F1 | parser success rate | invalid output rate | metrics SHA-256 | predictions SHA-256 |
|---|---:|---:|---:|---:|---|---|
| `image_only` | 0.42042292710072343 | 0.23312016560472856 | 1.0 | 0.0 | `cc3fb152d163ce886c5d183a77d31e21726dcaa176d5bb5f8fae7c957d472bd0` | `2b31e463eca05ad1bfd8e18812173d9a2e5cc13fa3b42130a1e48c31ce8b31eb` |
| `image_ego_state` | 0.4602114635503617 | 0.29861793313306656 | 1.0 | 0.0 | `a615e655f43794d4680b5ca55b06b76be4d96b444ad0d2db7f790a90d32a8648` | `00e5f15f524a9008ddebe893cd37b149295a86c551a7761c3629b4f74e269064` |

Prediction distribution：

| input variant | keep | accelerate | decelerate | stop | left_lateral | right_lateral |
|---|---:|---:|---:|---:|---:|---:|
| `image_only` | 2478 | 24 | 540 | 533 | 0 | 19 |
| `image_ego_state` | 2542 | 228 | 330 | 476 | 0 | 18 |

Per-class F1：

| input variant | keep | accelerate | decelerate | stop | left_lateral | right_lateral |
|---|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.5657158091175687 | 0.005025125628140704 | 0.17562724014336917 | 0.5881326352530541 | 0.0 | 0.06422018348623854 |
| `image_ego_state` | 0.5619921363040629 | 0.1794019933554817 | 0.282560706401766 | 0.7125803489439853 | 0.0 | 0.05517241379310344 |

- 在当前固定 zero-shot protocol 下，加入 deterministic current/past ego-state serialization 后，validation accuracy 从 0.420423 提高至 0.460211，macro-F1 从 0.233120 提高至 0.298618；主要增益出现在 `accelerate`、`decelerate` 与 `stop` 等纵向动作类别。该结果是 observation，不构成机制证明。
- Lateral targets 是当前 zero-shot 的主要 failure mode：两个 variants 的 `left_lateral` prediction count 与 F1 均为 0；`right_lateral` F1 分别约为 0.0642 与 0.0552。
- Isolation：两个正式 baseline receipts 均为 `test_records_read=0`、`test_images_opened=0`、`test_labels_read=0`、`test_evaluation_performed=false`、`validation_label_used_as_model_input=false`、`validation_images_opened=3594`。已消费的 project test 仍禁止重新使用。

Validation baseline comparison：

| baseline | macro-F1 | accuracy |
|---|---:|---:|
| Majority baseline | 0.087186 | 0.354201 |
| Phase 0.2 ego-motion rule baseline | 0.615681 | 0.623817 |
| Qwen3-VL image-only zero-shot | 0.233120 | 0.420423 |
| Qwen3-VL image + ego-state zero-shot | 0.298618 | 0.460211 |

### Phase 0.3d Real Qwen3-VL LoRA Smoke

- 状态为 `completed`，artifact `status=smoke_passed`；`execution_git_commit=74aaed2ebcc53fa03c93e208a15e1e47f11b35f8`。
- 模型为 `Qwen/Qwen3-VL-4B-Instruct`，revision 为 `ebb281ec70b05090aa6165b016eac8ec08e71b17`。
- 训练使用 12 条 exact tiny train subset，validation smoke 使用 6 条样本。Train action distribution 为 `accelerate=2`、`decelerate=2`、`keep=4`、`left_lateral=1`、`right_lateral=1`、`stop=2`。
- Optimization：optimizer steps 为 20，gradient accumulation 为 4，total micro-batches 为 80；initial/final/minimum loss 分别为 `5.255437068641186`、`0.373084731050767`、`0.25869851280003786`，`loss_finite=true`、`loss_decreased=true`。
- Trainable parameter evidence：total parameters 为 `4,443,714,048`，trainable parameters 为 `5,898,240`，trainable percentage 为 `0.13273221310571603%`；LoRA target modules 为 `q_proj / k_proj / v_proj / o_proj`。
- Direct parameter-update evidence：训练后的 `adapter_model.safetensors` 经只读检查，144 个 LoRA B tensors 中 `144/144` 含非零权重；被检查的 LoRA B tensors 均具有非零 norm 与非零元素。这是 LoRA adapter parameters 在真实 optimization 中发生更新的直接 checkpoint-level evidence，但 parameter change 本身不证明泛化或 full-validation improvement。
- Checkpoint：已保存 `adapter_model.safetensors`，`full_model_saved=false`；checkpoint SHA-256 为 `cc29acfd4dd8205e8c23bbee0527ab72294c4ca02f2b46823f06b0780fcbb3c5`。Fresh base/adapter reload 已完成，reload prediction count 为 18。

Exact tiny train subset 在训练前后（adapter save 后通过 fresh base/adapter reload 评测）的结果为：

| exact tiny train subset | accuracy | macro-F1 |
|---|---:|---:|
| before training | 0.5 | 0.32478632478632474 |
| after fresh reload | 0.75 | 0.6481481481481481 |
| delta | +0.25 | +0.3233618233618234 |

- Validation smoke：`sample_count=6`、accuracy `0.6666666666666666`、macro-F1 `0.2916666666666667`、parser success rate `1.0`、invalid output rate `0.0`。由于样本数小且类别覆盖不完整，这 6 条结果不能作为正式 validation performance 结论。
- Isolation：`test_files_opened=0`、`test_records_read=0`、`test_images_opened=0`、`test_labels_read=0`、`test_evaluation_performed=false`。
- 该证据证明 real Qwen3-VL LoRA training chain、真实 loss/backward/optimizer path、adapter checkpoint save/reload 均可运行，且训练后 model behavior 在 exact tiny train subset 上发生变化；不得据此声称 full-validation LoRA performance 已完成、LoRA 已超过 rule baseline、LoRA 泛化能力已证明、test performance 已获得或 final VLA model 已训练完成。

### Phase 0.3e-1 Qwen3-VL Baseline Failure Analysis

- 状态为 `completed`。分析只复用冻结的 Phase 0.3c full-validation artifacts，覆盖 `image_only` 与 `image_ego_state` 各 3,594 条 validation predictions；prediction、metrics 与 receipt 的 provenance、SHA、split 和 sample alignment 均已核验，未重跑 inference，未访问 test。
- 两组 full-validation baseline 的 parser success rate 均为 `1.0`、invalid output rate 均为 `0.0`，因此 output format、parser 与 action schema 解析不是当前主要 failure source；后续重点应放在语义分类、视觉信息利用、ego-state 融合与 lateral intent 建模。
- 纵向动作存在明显的单帧视觉歧义：单帧 `CAM_FRONT` 能提供道路结构、障碍物和前车距离等静态线索，但通常不能直接表达当前速度变化趋势，难以稳定区分 `keep / accelerate / decelerate`。这与 Phase 0.3c 中 ego-state 主要改善 `accelerate / decelerate / stop` 的统计 observation 一致，说明 current/past ego-state 为纵向判断提供了有价值的补充信息，但不构成因果证明。
- Ego-state 确实影响了模型输出：代表样本中 Case 01 / 02 / 04 / 05 由 image-only 错误变为 image + ego-state 正确；Case 06 / 11 / 12 则由 image-only 正确变为加入 ego-state 后错误。该双向变化表明 ego-state 同时具有有效补充信息与 shortcut 风险：模型可能过度依赖 speed、acceleration、yaw rate 到 coarse action 的直接映射，而未充分结合视觉场景。
- Lateral collapse 是当前 zero-shot baseline 最主要的建模短板。Case 07 / 08 的 GT 为 `left_lateral`，Case 09 / 10 的 GT 为 `right_lateral`，两个 input variants 均预测为 `keep`；部分人工审核样本的当前图像包含道路转向、车道几何或路口结构线索，reviewer-only GT future trajectory 也支持对应 lateral action，但模型仍未识别。结合 full-validation prediction distribution，当前现象不能仅由 class imbalance 解释，更合理的工程判断是 baseline 对 lateral intent 的视觉感知与推理能力不足。
- 12 个冻结 representative samples 已完成 input-faithful review 与 reviewer-only GT trajectory audit。代表样本中的 GT future trajectory 总体支持对应 meta-action，未发现足以说明系统性标签错误或标签污染的证据；该结论仅限本次小规模代表样本审核，不证明 full dataset 的每条标签均无误。当前主要问题更可能来自模型判别能力与输入信息边界，而不是 GT 标签体系失效。
- Reviewer-only panel 明确标注 `REVIEWER-ONLY GT FUTURE INFORMATION` 与 `NOT AVAILABLE TO MODEL AT INFERENCE`。其中 input-faithful review 用于判断当前 `CAM_FRONT + ego-state` 是否足以支持动作判断；reviewer-only GT audit 只用于核验 GT action 是否受真实 future trajectory 支持，并区分视觉信息不足、模型推理失败与潜在标签问题。Future trajectory / BEV 从未加入 Qwen inference input，也不得被描述为模型推理时可获得的信息。
- 该分析完成时 Phase 0.3 overall 继续保持 `active`，下一子阶段为 Phase 0.3e-2 interface freeze；当前状态已由下节的 real interface verification 结果更新。

### Phase 0.3e-2 Qwen3-VL Real Interface Freeze

- Real verification 状态为 `passed`；使用 `Qwen/Qwen3-VL-4B-Instruct` 与 model / processor revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，只访问少量 validation adapter record，test access counters 全部为 0。
- 真实 processor 类型为 `Qwen3VLProcessor`，稳定 required keys 为 `input_ids`、`attention_mask`、`pixel_values` 与 `image_grid_thw`。本次真实样本观察为：`input_ids [1,1413] int64`、`attention_mask [1,1413] int64`、`pixel_values [5600,1536] float32`、`image_grid_thw [1,3] int64`，grid 为 `[[1,56,100]]`。
- `1413`、`5600` 与下述 `1400` 均只属于该真实 validation CAM_FRONT sample observation，不是全局固定 shape。稳定 processor contract 为 required fields / dtypes、`pixel_values` patch width `1536`、grid 每行 `[T,H,W]`、pixel row count 等于各图像 `T×H×W` 之和，以及 sequence / visual patch axes 保持动态。
- 真实模型结构为 `Qwen3VLForConditionalGeneration.model` 下的 `visual=Qwen3VLVisionModel` 与 `language_model=Qwen3VLTextModel`；规划分支固定使用 public `Qwen3VLForConditionalGeneration.get_image_features(pixel_values, image_grid_thw)`，不得直接绑定 `model.model.visual(...)` 或私有 ViT block API。
- 对 grid `[[1,56,100]]` 且 `spatial_merge_size=2` 的真实样本，primary per-image `image_embeds` 为 `[1400,2560]`、dtype `bfloat16`。稳定 planning feature contract 是 per-image `[N_i,2560]`，其中 `N_i=T_i×H_i×W_i/2²` 动态，feature dimension `2560` 稳定，image / frame 顺序与 processor 输入及 `image_grid_thw` 行顺序一致。
- Native DeepStack 在完整 Qwen3-VL semantic branch 内正常工作；本次观察到 3 个 levels，每个为 `[1400,2560] bfloat16`。这些 DeepStack tensors 不作为 Phase 0.4 temporal planner 的 mandatory public interface，不自行重实现，也不把该单样本 token count 写死。
- Phase 0.3 legacy six-class prompt、strict parser 与 prediction artifact format 保持不变；Phase 0.4 planning feature interface 已与 legacy prompt / parser 解耦。Phase 0.4 的 structured factorized meta-action serialization、assistant masking、strict parser 与 decision adapter hidden-state shape 仍为 `planned`，必须在对应实现子任务中用真实输出冻结。
- 本轮未运行 full-validation feature extraction、LoRA training 或 Phase 0.4 training；未访问、恢复或重跑已消费 project test。

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
conda run -n codex4vla_env python scripts/run_ego_motion_rule_baseline.py --config configs/phase0_2_rule_baseline.yaml
```

trainval pilot、Phase 0.2a 输入审计与 Phase 0.2b rule baseline 需要预先设置 `NUSCENES_ROOT` 与 `VLA_DERIVED_ROOT`；原始数据、派生 manifest、审计 JSON 与 baseline 输出均不进入 Git。

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
- Phase 0.2c failure analysis 已完成，`phase0.2-ego-motion-rule-v0.1` 冻结为 `candidate-0293`：stop speed `0.2 m/s`、lateral yaw rate `0.05 rad/s`、accelerate `0.5 m/s²`、decelerate `0.3 m/s²`；validation prediction 复现为 `3594/3594`。
- validation 的主要错误模式为 `keep → decelerate`（260）与 `decelerate → keep`（181）。该 validation 同时用于 Phase 0.2b 候选选择与报告，不代表无偏 test 性能。
- Phase 0.2d formal execution 在 Git commit `e1cebb4182d1d30ee893c619f6cd45fe1aaaee39`、execution CLI SHA-256 `da160707ee3813b29d81c1e8d06442364843ea523235f464b1d413ce23d7beee` 下调用一次，exit code 为 `1`；execution claim SHA-256 为 `48d6bcccfd43eff529a9a50390bc6851f9f0edb7cdeb1bee6401bf23ae301cea`，状态为 `consumed_failed`。
- 失败点为 `build_formal_outputs → build_validation_to_test_comparison`：正式 `validation_metrics.json` 使用嵌套 `metrics` 与顶层 `predicted_class_distribution`，comparison builder 期望顶层扁平 metrics 与 `prediction_class_distribution`。durable claim 后已访问 test label/motion；正式 test outputs 未生成、正式 test metrics 不可用，rule 与 thresholds 均未根据 test 结果修改，`rerun_permitted=false`。
- 该 test split 已在 Phase 0.2d sealed one-shot execution 中永久消费。尽管正式指标未成功持久化，它也不再是后续规则、模型、阈值或架构选择的 untouched holdout。不得重新执行、恢复、重算或使用该 test split 进行任何调参、候选选择或规则修改。

## Open Questions / Pending Verification

- exact-grid interpolation v1.1 是可选数据增强 backlog；如后续评估，应保持现有 v1 manifest、sidecar 与 test split 不变，并单独提升协议版本。
- Phase 0.2d 状态为 `consumed_failed`；后续必须新增独立的 validation-artifact schema adapter 和真实 artifact-shape regression test。该修复仅适用于未来协议，不得用于重跑当前 test。
- validation artifact adapter 与 producer-shape regression 已完成，但仅适用于未来协议，不授权重跑已消费 test。

## Next Gate

- 当前 test 不得再次使用，也不得重新切分或重命名为新的 holdout。
- Phase 0.3 overall 保持 `active`，直到 Draft PR #37 完成人工 review / merge；Phase 0.3e-2 real interface verification 已通过，不再存在 real processor / visual-feature verification pending item。
- PR #37 merge 后执行 Learning & Capability Closeout，再按修正后的 Phase 0.4a 数据合同开始下一独立阶段；不得提前开始 Phase 0.4 training。
- Phase 0.3 可继续使用 train/validation 进行开发与模型选择，但不得使用本次已消费 test 的任何信息进行调参、候选选择或规则修改。
- 后续无偏最终评估必须使用新的外部 held-out dataset 或新的、未被访问的 evaluation protocol。
