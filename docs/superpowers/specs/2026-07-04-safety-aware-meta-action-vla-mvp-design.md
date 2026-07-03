# Safety-Aware Meta-Action VLA MVP 规划文档设计

## 1. 目标

在 `project_mvp_plan.md` 编写一份可立即执行、单卡可复现、风险可控的 6 周项目规划。规划面向多模态与智能驾驶算法求职作品集，同时保留后续扩展为研究项目的接口。

第一版只覆盖 nuScenes `CAM_FRONT` 单相机、开环、6 类 meta-action：

- `keep`
- `accelerate`
- `decelerate`
- `stop`
- `left_lateral`
- `right_lateral`

第一版不承诺闭环 CARLA、实车部署、连续 waypoint 或未经实测的模型效率结论。

## 2. 核心决策

采用阶段门控，而非一次性推进全量数据清洗、SFT 与偏好优化：

1. Phase -1 先验证数据闭环和标签质量，不训练模型。
2. Phase 0 依次完成 majority、rule-based、zero-shot、few-shot 和小样本 LoRA/action adapter baseline。
3. Phase 1 先证明 safety scorer 与 reranker 有效，再构造 DPO pairs；GRPO 只作后续备选。
4. DriveLM、NuInstruct 与 safety QA SFT 降级为 L0 跑通后的条件增强模块。
5. 连续 waypoint 降级为 stretch goal。

进入下一阶段必须满足上一阶段的验收门槛；未满足时执行预先定义的降级方案。

## 3. 文档结构

`project_mvp_plan.md` 采用以下结构：

1. 项目启动版结论与范围边界
2. 最小系统数据流与自建贡献
3. Phase -1：数据闭环验证
4. Phase 0：VLA-L0 行为克隆
5. Safety scorer 公式与指标定义
6. Phase 1：safety reranking 与 preference learning
7. 条件增强模块与 stretch goals
8. 第一版必须交付的五组实验表
9. 6 周执行时间线与里程碑
10. 脚本、数据产物与推荐目录
11. 风险、停止条件与验证路径
12. README/demo/简历交付清单
13. 稳健版简历项目描述

## 4. 阶段描述规范

Phase -1、Phase 0 和 Phase 1 均固定写明：

- 目标；
- 输入；
- 输出；
- 脚本；
- 执行步骤；
- 验收标准；
- 失败时如何定位；
- 失败时如何降级。

文档中的命令、阈值和模型配置只在有可靠依据时给出。Qwen3-VL 的视觉 token 数、FP8 支持、显存、TTFT 和 latency 均标注为“待官方配置核验”或“待本机实测”，不写成已确认事实。

## 5. Safety scorer 设计

总体分数：

```text
R = w_imit * imitation
  - w_collision * collision_or_near_miss
  - w_vru * vru_distance_violation
  - w_feas * infeasibility
  - w_lazy * unnecessary_stop
  - w_comfort * harsh_action_or_jerk
```

规划文档必须解释每一项的输入、计算对象和失败模式。`unnecessary_stop` 是强制项，用于阻止“永远停车”通过降低碰撞率获得虚假安全收益。权重与距离阈值不预设为通用真值，先由 mini 数据分布、100 样本人工抽检和敏感性实验校准。

## 6. 实验设计

必须交付以下实验表：

1. 数据统计：样本数、类别分布、VRU 比例、安全样本比例及动作条件统计。
2. L0 动作预测：majority、rule-based、zero-shot、few-shot、LoRA/action adapter；主指标为 macro-F1。
3. Safety 消融：L0、L0+reranker、无 safety terms 的 DPO、完整 safety terms 的 DPO。
4. 失败案例：stop 偏置、VRU 漏检、lateral 混淆、scorer 误判和标签派生错误。
5. 效率：分辨率、视觉 token、显存、TTFT 与单样本 latency，全部以实测为准。

Reranker 的进入标准是：VRU violation 或 near-collision 至少一项改善，同时 `unnecessary_stop` 未出现不可接受上升。具体可接受幅度在获得 baseline 后按验证集置信区间设定，不提前拍脑袋填写。

## 7. 交付边界

最终只新建 `project_mvp_plan.md`，不修改原 Obsidian 文件。文档中文为主，保留必要英文术语，结论先行，不使用 SOTA、工业落地或闭环能力等未经实验支撑的表述。

## 8. 自检结果

- 无 `TBD`、`TODO` 或未决占位符。
- 第一版范围与 6 周周期一致。
- 训练阶段均有前置门槛，避免在数据闭环失败时投入算力。
- Safety 指标同时约束碰撞风险与过度保守。
- 大规模 QA SFT 和连续 waypoint 均不阻塞 MVP。
