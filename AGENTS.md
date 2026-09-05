# AGENTS.md

本文件约束 Codex 和其他 agentic coding worker 在本仓库中的工作方式。若任务要求与本文件冲突，先停止并向用户说明，不得静默绕过。

## General Rules

### Documentation Discipline

- Do not create new temporary documentation files for routine tasks. This includes files or directories named like plan, design, spec, debug, report, tmp, scratch, notes, experiments, or similar variants.
- Temporary execution plans must stay in the chat response only. They must not be written into repository files.
- If long-term stable project conventions need to be recorded, only update existing relevant sections in:
  - AGENTS.md
  - README.md
  - docs/progress.md
- Do not create new documentation files unless the user explicitly requests it.
- docs/progress.md, if present, should only record confirmed facts, such as completed milestones, dataset paths, input/output field contracts, CLI conventions, known risks, and open questions.
- docs/progress.md must not contain long reasoning traces, temporary plans, debugging logs, one-off task notes, or speculative implementation ideas.
- At the end of each task, report:
  - which files were modified;
  - whether AGENTS.md, README.md, or docs/progress.md was modified;
  - whether any new file was created;
  - whether any new documentation file was created.
- If no documentation was modified, explicitly state: “No documentation files were modified.”
- If a task does not require documentation changes, do not modify any `.md` files.
- If creating a new documentation file seems necessary, stop and ask the user for confirmation before doing so.

### Contract-First Irreversible Workflows

- 对不可逆工作流，Codex 只负责最小实现，不得同时充当需求分析者、接口定义者和唯一测试者。
- 跨模块 artifact 在修改前必须完成 producer artifact → consumer intake，核对真实字段层级、命名、类型、provenance 与 SHA。
- contract fixture 必须由真实 producer helper 或经核验的 golden artifact 生成，禁止手写猜测 consumer schema。
- 修复 contract mismatch 前必须先增加可复现历史失败的 regression test，并确认其在修复前失败。
- 不可逆正式执行前必须完成不访问 sealed data 的 full shadow execution，覆盖真实 producer shape、adapter、consumer、输出持久化与 rerun guard。
- 测试数量不能替代真实 artifact shape 核验与 producer → adapter → consumer 端到端证据。

### Minimal Change and Anti-Overengineering

- 修改前必须明确当前任务的 success criteria、failure criteria 与 verification method；达到验收标准后停止，不得以“更加严谨”或“理论上更安全”为由继续追加无关加固。测试数量、校验字段数量和 abstraction 数量本身不代表更高质量。
- 默认只实现解决当前明确问题所需的最小改动；不得因“以后可能会用到”提前增加功能、helper、class、抽象层、配置项或 fallback，也不得顺手重构与任务无关的代码。
- 只清理由本轮修改直接产生的问题。历史遗留问题若不阻塞当前任务，应记录为 optional cleanup 或 future hardening；一次性或简单流程优先直接实现，不为形式上的架构完整性增加 wrapper 或 abstraction。
- 已由仓库 contract、producer、framework 或现有测试保证的内部状态默认按其合同使用。完成必需的外部 artifact intake 后，不得为理论上不应出现的内部状态层层重复 SHA、schema、metadata 或 provenance 校验，也不得穷举防护人为篡改内部 artifact。
- 只有存在真实历史失败、明确外部输入边界、不可逆操作风险或当前任务可复现的问题时，才增加对应 validation、fallback 或 recovery path。严格校验应集中在用户输入、CLI 参数、外部 API / 网络、文件系统输入、外部 artifact intake、train / validation / test 边界、model inference input contract、GT / future information leakage 与不可逆正式执行。
- 真实错误必须 fail fast 并暴露根因。禁止使用 broad `except Exception`、silent fallback、nil / empty fallback、模糊默认值、改变实验条件的自动重试、fuzzy mapping 或静默纠正 invalid output 让程序“看起来继续工作”；违反正式 contract 的输入必须 hard fail，不得猜测意图或自动修复。
- **Blocking：** 任何可能影响实验结果正确性、train / validation / test isolation、test / target / future / GT leakage、输入输出 contract、label / schema、metrics、sample alignment、数据或模型真实执行链路、不可逆正式执行真实风险或后续阶段实际兼容性的问题，都必须修复后才能继续。典型情况包括读取 test 数据、future trajectory 进入模型输入、baseline sample set 不一致、invalid prediction 使 F1 虚高、label / schema 不匹配、真实 producer / consumer contract mismatch 或模型实际无法运行。
- **Non-blocking：** 仅涉及极端人为篡改、理论上不可能的内部状态、重复 provenance 字段的穷举校验、不改变实验结果的 receipt 防篡改增强、一次性流程的额外 abstraction、为“完整”增加 helper / wrapper / class，或没有真实 failure evidence 的 fallback / recovery，默认不阻塞当前 Phase；可以记录，但不得无限延后模型实验和主线推进。
- 本节不得用于削弱 Contract-First Irreversible Workflows、Data Rules、Evaluation Rules、test isolation、future / GT leakage guard，以及 Git、secret 和 dataset protection。Minimal code 不等于少做必要校验，trust internal contracts 不等于信任外部输入，avoid defensive coding 不等于吞掉异常，fast iteration 不等于降低实验可信度；发生冲突时，数据泄漏、实验正确性和不可逆风险规则优先。
- 多个实现方案都合理时，优先选择能直接提升模型能力、自动驾驶 / VLA 技术含量、系统完整性、实验可解释性和面试展示价值的方案；纯内部防御且不改变模型结果、实验可信度或后续系统能力的工作降低优先级，但不得以 demo 为由牺牲 correctness。

## Environment Rules

- 本项目默认使用 conda 环境 `codex4vla_env`。
- Codex / agentic worker 运行 Python、pytest、数据检查或训练相关命令时，必须优先使用：

  conda run -n codex4vla_env python ...
  conda run -n codex4vla_env pytest ...

- 不得使用 base Python 作为项目验证环境。
- 如果 `conda run -n codex4vla_env ...` 失败，必须停止并报告环境问题，不得改用 base Python 伪造通过结果。
- 如果需要新增依赖，先说明依赖用途和安装位置；不得静默安装到 base 环境。
- README 中的命令若未显式写 conda，仅表示命令形式；实际验证必须在 `codex4vla_env` 下执行。
  
## 1. Project Mission

本仓库服务于 **Safety-Aware VLA for Autonomous Driving with BEV/OCC-aware Spatial Evaluation**。当前路线为 single-camera、open-loop、6-class coarse meta-action MVP；长期目标是 coarse-to-fine、共享多模态 backbone 的多任务 VLA。BEV/OCC-aware layer 仅是后续 GT-derived 离线评估层，不是完整 occupancy prediction 网络。项目计划以 [`project_mvp_plan.md`](project_mvp_plan.md) 为准。

固定 action schema：

```text
keep
accelerate
decelerate
stop
left_lateral
right_lateral
```

当前 6 类仅是 coarse action schema：`left_lateral` / `right_lateral` 只表示稳定左右横向运动，不能解释为 lane change 或 turn。它们继续作为 coarse target、可解释输出、辅助监督和长期 baseline，不是最终固定动作空间。当前已完成的是 coarse 标签派生、冻结、审核与数据基础；coarse neural action head、LoRA、action adapter、fine maneuver、waypoint 与 BEV/OCC auxiliary 均为 planned。

## 2. Non-Negotiable Rules

### MUST NOT

- 不得提交 nuScenes 原始数据、处理后数据、模型权重、checkpoint、日志、缓存、`.env`、API key、token、个人隐私文件或大型二进制文件。
- 不得执行 `git push --force`。
- 不得删除用户已有文件。
- 不得修改与当前任务无关的文件。
- 不得跳过数据闭环直接启动 LoRA、DPO 或 GRPO。
- 不得把未实测的指标、显存、latency、视觉 token 数或 FP8 能力写成事实。
- 不得把 planned work 写成 completed work。
- 不得声称 closed-loop、real-time、CARLA、实车、连续轨迹规划或部署能力，除非仓库已有对应代码、配置和可核查实验结果。
- 不得把论文、官方模型卡或外部宣传结果写成本项目实验结果。
- 不得用 `turn_left` / `turn_right` 替换首版 lateral schema，除非任务已明确引入 map、lane topology 或 route command 并更新项目规格。
- 未接入 map、lane topology、intersection topology、route command 或 short temporal context 的至少一部分前，不得仅根据横向位移派生 `left_turn` / `right_turn`、lane-change 或其他 fine-grained maneuver 标签。
- 不得将 6 类扁平分类直接硬改为更多互斥类别；新增动作空间或输出 head 前，必须更新项目规格、数据 contract、评测协议与验收 gate。
- 推理路径不得使用 future ego trajectory、GT meta-action、GT BEV/OCC raster、未来 GT agents 或 test labels。
- 不得将 GT boxes、future GT agents 或 GT occupancy 作为模型 test-time inference input；GT geometry / occupancy 只可作为 oracle offline scorer backend。不得删除 candidate rollout 与 geometric scorer 而将 occupancy 直接当作 safety score。
- rule-based baseline 不得使用 future ego trajectory、derived meta-action 或 test labels；仅可使用 inference-time current/past ego state。
- 未实现 differentiable soft occupancy 或 distance-field surrogate 时，不得把 safety cost 写成可反向传播的训练 loss。

### MUST

- 修改前阅读相关文件、`project_mvp_plan.md` 和当前 `git status`。
- 对不确定结论标注“待验证”或 `planned`，并给出验证方式。
- 保持修改范围最小，遵循现有文件风格。
- 保留数据、配置、规则和实验的可追溯性。
- 完成后报告 changed / why / how to verify。

## 3. Execution Order

必须按以下顺序推进：

1. **Phase -1:** 数据对齐、6 类 coarse 标签、人工审核、规则冻结与 manifest audit 前置检查；不训练。
2. **Phase 0.1:** audited seed-subset manifest、固定 seed 的 scene-level split、六类统一评测协议、invalid prediction 指标处理、完整 manifest contract validator 与 Majority Baseline；已完成并合并。
3. **Phase 0.1b:** 从 nuScenes mini 扩展至 trainval，生成正式 dataset manifest v1，重统计类别分布并抽检边界样本；正式 LoRA、action adapter 与 DPO 前必须完成。
4. **Phase 0.2:** inference-time current/past ego-motion rule baseline。
5. **Phase 0.3:** Qwen3-VL 数据接口与 legacy coarse action baseline。
6. **Phase 0.4:** factorized meta-action、structured semantic decision、action-conditioned waypoint 与 trajectory-to-action verifier；下一实际执行子阶段为 Phase 0.4a factorized target + temporal dataset contract。
7. **Phase 0.5:** BEV/OCC geometry 与 offline spatial evaluation。
8. **Phase 0.6:** factorized-action-conditioned candidate trajectories 与 safety reranking。
9. **Phase 0.7:** logged sequential evaluation 与 planning interface。
10. **Phase 0.8:** conditional preference learning；只有前序 gate 满足时才执行。
11. **后续扩展:** map / route / lane topology、fine-grained maneuver、predicted BEV / occupancy、外部交互平台与 RL / GRPO。

前一阶段验收条件未满足时，不得推进下一阶段。失败时优先修复数据、标签、scorer 或评测协议，不得通过增加训练规模掩盖问题。

## 4. Coding Standards

- Python 优先，遵循 PEP 8、类型标注和仓库既有格式。
- 遵循 SOLID 原则；每个模块保持单一职责，避免训练、数据解析、评测和可视化混在同一文件。
- 配置参数进入 YAML；action 阈值、safety 阈值、时间窗口、坐标约定和路径不得散落硬编码。
- 所有项目路径使用相对路径或配置文件，不写入个人机器绝对路径。
- 保留 `sample_token`、`scene_token`、`current_ego_pose`、`current_ego_motion`、`future_ego_trajectory`、`nearby_agents` 与 `split` 等稳定基础追溯字段；派生 target 及其 rule version 必须单独可追溯。
- 坐标系必须注明 source frame、target frame、轴方向、单位和 transform 顺序。
- 时间相关逻辑必须注明 timestamp 单位、采样间隔、future horizon 和缺帧策略。
- Action schema 必须由单一模块定义，禁止在多个脚本重复维护字符串列表。
- 每个核心模块必须有最小单元测试；几何模块优先使用人工构造的小型确定性案例。
- 不添加与文件其余部分不一致的多余注释、过度防御性检查或无依据的 `try/except`。
- 不使用 `Any` 或无依据类型转换绕过类型问题。

## 5. Data Rules

- train/validation/test 必须按 scene-level split，禁止相邻帧跨 split。
- Few-shot examples 不得来自 test scene。
- Manifest 必须区分稳定基础字段和可版本化的派生 targets。基础字段至少包含：

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
split
manifest_schema_version
```

- 当前 `phase0_audited_seed_subset_v1` 是 audited seed-subset schema，不是正式 trainval manifest v1。稳定字段必须包括 `sample_token`、`scene_token`、`timestamp`、`cam_front_path`、`current_ego_pose`、`current_ego_motion`、`coordinate_metadata`、`future_ego_trajectory`、`nearby_agents`、`split` 与 `manifest_schema_version`；当前派生/追溯字段为 `meta_action`、`label_rule_version`、`safety_rule_version` 与 `source_audit_record`。`label_rule_version=phase-1.6-meta-action-v0.2`；不得提前重命名为 `meta_action_coarse` / `meta_action_rule_version`。`future_waypoints`、`trajectory_valid_mask`、`longitudinal_action`、`lateral_direction`、`maneuver_type` 与 `fine_action_rule_version` 均为 planned；新增 head 时优先扩展 targets，不重写基础数据管线。
- `current_ego_pose` 与 `current_ego_motion` 的 `timestamp_source` 必须为 `CAM_FRONT_sample_data`；motion 仅可由当前和历史 pose 推导，禁止使用 future pose 或 future trajectory。Phase 0.1b 才生成正式 trainval dataset manifest v1。
- action / safety / fine-action rule 变化必须提升对应 rule version 并重新生成受影响 targets；coarse 与 fine 标签、不同 label version 均不得静默混用。扩展动作空间时必须新增 schema version，不得覆盖旧 schema。
- `safety_rule_version` 必须进入所有安全派生产物。
- mini 只用于 smoke test、快速回归、人工审核和小规模调试；正式 LoRA、action adapter 与 DPO 必须使用 trainval manifest，mini 上仅允许 smoke run。
- `uncertain` 样本不能强行算作正确标签，必须单独记录并排除出高置信度训练/偏好数据。
- Phase -1 / Week 2 必须完成至少 100 个样本的人工抽检记录。
- Phase -1 抽检必须覆盖 6 类 action、有/无 VRU、VRU presence 和 action boundary cases。
- Phase 0.5 geometry evaluator 必须报告 collision / near-miss、VRU distance violation、infeasibility、unnecessary stop、harsh action / jerk，并通过 synthetic tests 与 evaluator audit；未通过不得进入 Phase 0.6 safety reranking。
- Phase 0.6 reranker 必须在固定 candidate set 上比较 rerank 前后 macro-F1、VRU / near-collision、unnecessary stop 与 scorer failure cases；未证明风险改善且不过度增加 stop，不得进入 conditional preference learning。
- 原始数据、处理后数据和生成媒体默认不纳入 Git；只提交 schema、脚本、配置、允许公开的小型测试 fixture 和质检报告。

## 6. Evaluation Rules

### Action prediction

每个 action prediction 实验必须报告：

- macro-F1；
- per-class F1；
- confusion matrix；
- class distribution；
- invalid output rate；
- action parsing success rate；
- accuracy 仅作辅助指标。

### Safety

每个 safety 实验必须同时报告：

- VRU violation rate；
- near-collision rate；
- unnecessary stop rate；
- macro-F1；
- infeasibility；
- harsh action / jerk；
- safety scorer 分项 penalty。

Reranker 的结论必须基于相同 candidate set。若 violation 下降主要来自 `stop` 增加，不得写成安全能力提升。

### Preference learning

Phase 0.8 为 conditional preference learning，仅在 Phase 0.6 candidate / oracle evaluator 与 Phase 0.7 logged sequential protocol 通过对应 gate 后启动。若 learned selector 没有稳定优于非学习基线，保留 Phase 0.6 oracle offline evidence，并记录负结果，不得通过扩大模型或切换到 RL 掩盖失败。

DPO / GRPO 仅在 trajectory tokenization、policy likelihood 与优化目标另行冻结后作为 Optional / Stretch；不得把它们写成当前 MVP 的必经终点。

## 7. Documentation Rules

- 技术文档中文为主，保留 VLA、VLM、MLLM、BEV、LoRA、DPO、GRPO、macro-F1、reranker 等英文术语。
- `README.md` 面向项目展示、边界说明和复现入口。
- `reports/` 面向数据统计、实验结论、消融和限制。
- `data/data_debug_report.md` 面向数据对齐、标签质检和规则变更。
- `reports/failure_cases.md` 必须随实验持续更新。
- 论文结果、官方 benchmark 和外部宣传必须注明来源，不得与本项目结果混写。
- 不确定内容必须标注“待验证”或 `planned`，并说明验证条件。
- 只有真实执行过的命令才能写为可运行命令；尚未实现的入口写为 planned commands。
- 所有实验结论必须能够定位到配置、数据 split、规则版本、checkpoint 和 sample-level 输出。

## 8. Commit Rules

推荐 commit 风格：

```text
docs: define MVP scope and acceptance gates
feat(data): inspect and align nuScenes samples
feat(data): derive versioned meta-action labels
feat(safety): add geometric action scorer
test(safety): cover collision and lazy-stop cases
feat(baseline): add majority and rule-based baselines
feat(model): add structured prompt baseline
feat(train): add L0 adapter training
feat(safety): add offline action reranking
feat(pref): build auditable preference pairs
docs: report experiments and limitations
```

提交要求：

- 每次 commit 只包含一个清晰改动。
- Commit 前运行 `git status` 并检查 staged diff。
- 不得提交数据集、权重、secrets、本地路径或无关文件。
- 修改后总结 changed / why / how to verify。
- 除非用户明确要求，不得自动 commit 或 push。
- 永远不得使用 `git push --force`。

## GitHub PR Collaboration Workflow

1. Codex 接到任务后，必须先从最新 `main` 开始并确认工作区状态：

   ```bash
   git checkout main
   git pull --ff-only origin main
   git status --short
   ```

2. 每个任务使用独立分支，命名格式为 `task_<phase-or-id>_<short-name>`，例如 `task_p1_5_manual_review`；不得直接在 `main` 上修改。
3. 修改前必须读取 `AGENTS.md`、`README.md`、`project_mvp_plan.md`、存在时的 `docs/progress.md`，以及与任务直接相关的源文件和测试文件。
4. 提交前禁止使用 `git add .` 或 `git add -A`。必须显式指定文件路径，例如：

   ```bash
   git add data/verify_labels.py tests/test_verify_labels.py docs/progress.md
   ```

5. Commit 前必须运行并检查：

   ```bash
   git status --short
   git diff --stat
   git diff -- <相关文件>
   ```

   同时运行当前改动所需的测试命令。Commit message 沿用本文件既有风格，例如 `feat(data): add manual review export`、`test(data): cover label verification cases` 或 `docs: update confirmed progress`。
6. 允许使用 `git push -u origin <branch-name>` 推送当前任务分支。永远不得使用 `git push --force`。
7. GitHub CLI 可用且已登录时，可以创建 PR：

   ```bash
   gh pr create --base main --head <branch-name> --title "<title>" --body-file <body-file>
   ```

   若 `gh` 不可用或未登录，不得反复尝试或伪造成功；必须输出 branch name、commit hash、建议 PR title、建议 PR body 和手动创建 PR 的说明。
8. PR description 作为本轮任务 handoff；`docs/progress.md` 只记录长期稳定且已确认的事实；最终聊天回复也必须包含 handoff summary。除非用户明确要求，不创建 `docs/handoff/`。
9. PR 和 handoff 必须说明 changed / why / how to verify、当前 phase / gate、未运行的验证及原因，并确认 diff 仅包含相关文件。
10. 永远不得提交数据集、模型权重、checkpoint、日志、缓存、`.env`、API key、个人文件或大型二进制文件；不得把 planned work 写成 completed work，也不得跳过当前 phase gate。

## PR Learning & Capability Closeout

从本规则生效后，每一个正式 PR merge 后、开始下一个独立任务或 Phase 前，必须在刚刚 merge 的 GitHub PR Conversation 页面发布一条顶层 comment，标题固定为：

```markdown
## Learning & Capability Closeout
```

标准顺序为：

```text
implementation
↓
tests / real execution
↓
PR review
↓
merge
↓
Learning & Capability Closeout comment
↓
next task / next Phase
```

不得跳过 Closeout 直接进入下一项独立工作。核心 Phase、model、data 或 evaluation PR 的完整协作流程为：

```text
Codex implementation
↓
commit + push
↓
Draft PR
↓
ChatGPT review actual PR diff
↓
fix on SAME branch and SAME PR if needed
↓
merge
↓
Learning & Capability Closeout
↓
next task
```

同一 PR 的 review fix 必须继续使用原 branch 和原 PR，不得另建任务分支。

### Closeout Evidence and Structure

Closeout 必须以该 PR 的 actual diff、merged code、tests、实际执行过的 runtime / experiment results、artifact 和已确认事实为依据。禁止把 planned 工作写成 completed、把 synthetic test 写成 real model evidence、把未执行的 GPU / training / evaluation 写成已掌握能力，或为了丰富简历而虚构能力；不得使用“熟悉了 LoRA”“学会了训练模型”等无法由该 PR 证明的空泛表述。

重要 PR 默认使用以下结构：

```markdown
## Learning & Capability Closeout

### 1. What — 这次具体做了什么

### 2. How — 工程上是怎么实现的

### 3. Why — 为什么这样设计

### 4. Capability — 这次能体现什么能力

### 5. Interview Explanation — 面试时怎么讲

### 6. Follow-up Questions — 面试官可能继续追问什么

### 7. Current Boundary — 当前能力边界

### Closeout Status
```

- **What：** 用 input → processing → output 描述实际完成的工程链路，不能只写“完成数据处理”“实现 LoRA”或“实现模型训练”。
- **How：** 解释与本 PR 真正相关的 input、processing、output、model interface、data flow、training target、label construction、metric、checkpoint 和关键实现细节；不为模板完整硬凑无关项目。
- **Why：** 说明主要设计选择、工程判断与 trade-off，以及为何使用当前验证路径。
- **Capability：** 只把该 PR 的真实工作映射为可由代码和结果支撑的求职能力。
- **Interview Explanation：** 形成一段可直接用于技术面试的项目说明，回答任务、实现、设计原因和验证方式；不写成简历 bullet 或论文摘要。
- **Follow-up Questions：** 围绕技术原理、实现、取舍、debug、alternatives 和 failure cases 列出真实可能被追问的问题。重要 PR 通常为 5–15 个，简单 PR 可以更少，不为数量添加无关问题。
- **Current Boundary：** 强制区分已完成与尚不能声称的能力，例如 `代码和 synthetic tests 已验证 ≠ 真实 GPU LoRA 已验证`，以及 `validation result 已确认 ≠ test result 可以重复使用`。
- **Closeout Status：** 明确 comment 是否已经成功发布，以及是否仍需人工操作。

Closeout 长度按 PR 重要程度控制：multimodal dataset、VLM inference、LoRA / SFT、trajectory model、BEV / OCC、evaluator、reranking 和 model deployment 等核心能力 PR 使用完整解释；typo、dependency fix、CLI bugfix 或小型文档修订仍需 Closeout，但允许明显缩短。不得把该制度变成新的过度工程。

### Storage and Publication

- 每次具体 Closeout 的唯一主要存放位置是对应的 GitHub merged PR Conversation comment。
- `project_mvp_plan.md` 只保存项目路线、Phase 设计、执行原则和本制度；`docs/progress.md` 只保存已确认的项目状态、指标、artifact、gate 与实验事实；不得把每个 PR 的完整 Closeout 重复复制到项目文档。
- 环境已安装并登录 `gh` 时，优先通过 `gh pr comment <PR_NUMBER> --body-file <temporary-closeout-file>` 发布；临时文件不得提交仓库，发布成功后必须删除。也可使用不产生仓库文件的 stdin 方式。
- 若 GitHub 写权限、authentication 或网络导致发布失败，不得改写进项目文档替代；必须在任务回复中给出完整 Markdown，并明确标注 `GitHub comment NOT posted. Manual paste required.`。
- 本规则从现在开始执行，不 retroactively 批量补写历史 PR。PR #32 的 Closeout 已由用户手动添加，不得修改；其他历史 PR 仅在未来确有面试复盘需求时按需补充。

## 9. First Task Reminder

项目启动后的第一条工程任务是实现并验证：

```text
sample_token
→ CAM_FRONT image
→ future ego trajectory
→ nearby 3D agents
→ one-page visualization
```

该链路通过前，不进入批量训练、LoRA、DPO 或 GRPO。

## Workspace Hygiene / Temporary File Cleanup

1. 每次完成任务前，必须运行或等效执行一次 workspace cleanup review。
2. 不允许把一次性调试脚本、临时验证脚本、缓存、日志、输出图片、模型权重、数据集文件提交到 Git。
3. 对于任务过程中临时创建的脚本，如果不是项目长期需要的正式模块，任务结束后必须删除，或者移动到 `scratch/` 并确保 `scratch/` 被 `.gitignore` 忽略。
4. 正式测试只能放在 `tests/` 下，且文件名必须表达长期测试目的，例如 `test_meta_action.py`、`test_safety_scorer.py`。不要在项目根目录留下 `test_xxx.py`、`debug_xxx.py`、`try_xxx.py`、`check_xxx_tmp.py` 这类临时文件。
5. 不得删除以下类型文件，除非用户明确要求：
   - `README.md`
   - `AGENTS.md`
   - `project_mvp_plan.md`
   - `configs/*.yaml`
   - `data/*.py` 中的正式数据处理脚本
   - `src/**/*.py`
   - `tests/**/*.py`
   - `reports/*.md`
   - `scripts/check_env.py`
   - `requirements*.txt`
   - `environment*.yaml`
6. 清理前必须先输出 dry-run 清单，让用户或下一步操作能看见将被删除的文件。
7. 只有确认文件属于临时文件、缓存、日志或生成产物时，才允许删除。
8. 删除前后都要运行 `git status --short`，并在总结中说明删除了什么、保留了什么、为什么。
9. 如果不确定某个文件是否有用，默认保留，并在总结中标记为“需要用户确认”。
10. 每次任务结束时，必须给出简短的 repository hygiene summary，包括新增了哪些正式文件、删除了哪些临时文件、还有哪些未跟踪文件，以及是否需要用户确认保留或删除。
11. 不要为每次任务自动创建 `docs/specs/YYYY-MM-DD-*.md` 这类任务记录文件。
12. 除非用户明确要求，不要创建新的 specs 文档。
13. 如果任务过程中需要临时记录计划、草稿或 debug 说明，只能放在 `scratch/` 目录。
14. `scratch/` 必须被 `.gitignore` 忽略，任务结束前必须清理其中已经不需要的内容。
15. 仓库中的正式长期文档只允许放在：
    - `README.md`
    - `project_mvp_plan.md`
    - `AGENTS.md`
    - `reports/*.md`
    - `docs/` 中用户明确要求保留的文档
16. 如果某个新文档不是长期资产，不要放进 `docs/`，也不要提交到 Git。
17. 每次任务结束时必须报告新增正式文件、删除临时文件、仍未跟踪的文件，以及是否存在需要用户确认的临时文档。
18. 若发现 dated spec、Codex 执行计划或一次性设计说明，先核对是否已被正式资产覆盖；确认覆盖后逐文件删除，不确定时默认保留并请求用户确认。
