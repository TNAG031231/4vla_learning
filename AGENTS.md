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

本仓库服务于 **Safety-Aware Meta-Action VLA for Autonomous Driving**。目标不是泛泛编写学习代码，而是按阶段完成一个可复现、可审计、可评测的自动驾驶 VLA MVP：

```text
nuScenes CAM_FRONT
→ 6-class meta-action
→ geometric safety scoring
→ offline reranking
→ auditable preference construction
→ conditional DPO
```

第一版仅覆盖 open-loop、single-camera、discrete meta-action。项目计划以 [`project_mvp_plan.md`](project_mvp_plan.md) 为准。

固定 action schema：

```text
keep
accelerate
decelerate
stop
left_lateral
right_lateral
```

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

### MUST

- 修改前阅读相关文件、`project_mvp_plan.md` 和当前 `git status`。
- 对不确定结论标注“待验证”或 `planned`，并给出验证方式。
- 保持修改范围最小，遵循现有文件风格。
- 保留数据、配置、规则和实验的可追溯性。
- 完成后报告 changed / why / how to verify。

## 3. Execution Order

必须按以下顺序推进：

1. **Phase -1: data alignment and visualization**
   - `sample_token → CAM_FRONT image`；
   - future ego trajectory / CAN bus；
   - nearby 3D agents；
   - one-page visualization；
   - 100 样本人工抽检。
2. **Phase 0: baseline and L0 action prediction**
   - majority；
   - rule-based；
   - zero-shot；
   - few-shot；
   - LoRA / action adapter。
3. **Geometric safety scorer**
   - collision / near miss；
   - VRU violation；
   - infeasibility；
   - unnecessary stop；
   - harsh action / jerk。
4. **Offline safety reranker**
   - 固定 candidate set；
   - 比较 rerank 前后分类与 safety 指标。
5. **Preference pair construction**
   - 仅保留满足 margin 和审计规则的 chosen/rejected pairs。
6. **Conditional DPO**
   - 仅在 reranker gate 和 pair audit 通过后执行。

前一阶段验收条件未满足时，不得推进下一阶段。失败时优先修复数据、标签、scorer 或评测协议，不得通过增加训练规模掩盖问题。

## 4. Coding Standards

- Python 优先，遵循 PEP 8、类型标注和仓库既有格式。
- 遵循 SOLID 原则；每个模块保持单一职责，避免训练、数据解析、评测和可视化混在同一文件。
- 配置参数进入 YAML；action 阈值、safety 阈值、时间窗口、坐标约定和路径不得散落硬编码。
- 所有项目路径使用相对路径或配置文件，不写入个人机器绝对路径。
- 保留 `sample_token`、`scene_token`、`label_rule_version`、`safety_rule_version` 和 `split` 等追溯字段。
- 坐标系必须注明 source frame、target frame、轴方向、单位和 transform 顺序。
- 时间相关逻辑必须注明 timestamp 单位、采样间隔、future horizon 和缺帧策略。
- Action schema 必须由单一模块定义，禁止在多个脚本重复维护字符串列表。
- 每个核心模块必须有最小单元测试；几何模块优先使用人工构造的小型确定性案例。
- 不添加与文件其余部分不一致的多余注释、过度防御性检查或无依据的 `try/except`。
- 不使用 `Any` 或无依据类型转换绕过类型问题。

## 5. Data Rules

- train/validation/test 必须按 scene-level split，禁止相邻帧跨 split。
- Few-shot examples 不得来自 test scene。
- Manifest 必须包含：

```text
sample_token
scene_token
timestamp
cam_front_path
future_ego_trajectory
meta_action
nearby_agents
label_rule_version
safety_rule_version
split
```

- `label_rule_version` 与 `safety_rule_version` 必须进入所有派生输出。
- 修改 action 或 safety 阈值后，必须提升规则版本并重新生成受影响 manifest；不同版本不得静默混用。
- `uncertain` 样本不能强行算作正确标签，必须单独记录并排除出高置信度训练/偏好数据。
- Phase -1 / Week 2 必须完成至少 100 个样本的人工抽检记录。
- 抽检必须覆盖 6 类 action、有/无 VRU、safe/unsafe 和规则边界样本。
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

DPO 必须包含以下对照：

- L0；
- L0 + safety reranker；
- DPO without safety terms；
- DPO with full safety terms。

如果 DPO 不优于 reranker，保留 reranker 作为最终 MVP 方案，不继续堆叠训练。负结果必须保留并分析，不得选择性删除失败样本。

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
