# 智能体架构改造方案

本文档整理 PID 整定链路中三个核心智能体的职责边界、数据流和改造方向：

1. 系统辨识智能体：输出多个高质量辨识候选。
2. PID 专家智能体：基于辨识候选生成整定入围方案。
3. 评估智能体：对入围方案做独立验收，并最终选出冠军方案。

目标不是推翻现有实现，而是在当前工程基础上逐步演进到更清晰、更工程化的三阶段架构。

## 一、当前问题

- 系统辨识与 PID 整定耦合过深。
  - 过去辨识阶段的最终排序混入了闭环试算和窗口惩罚，导致“拟合最好”和“最终被选中”不一致。
- PID 专家和评估智能体职责重叠。
  - 两边都在使用相近的闭环评分逻辑，评估智能体没有真正成为独立验收层。
- 候选方案缺乏正式 shortlist 机制。
  - 目前 PID 专家虽然会比较多个辨识候选，但尚未以明确门槛形成“入围方案集合”。
- 前端口径易混淆。
  - “辨识最优模型”“整定采用模型”“评估冠军方案”三个概念还没有完全拆开。

## 二、目标架构

### 1. 系统辨识智能体

负责：

- 在“可用于辨识”的候选窗口集合上尝试多模型拟合。
- 计算并输出：
  - `r2_score`
  - `normalized_rmse`
  - `identification_fit_score`
  - `confidence`
- 保留多个高质量辨识候选结果。

不负责：

- 决定最终 PID 参数。
- 用 PID 闭环性能直接决定最终模型。

### 2. PID 专家智能体

负责：

- 接收多个辨识候选。
- 对每个候选模型尝试多种整定策略。
- 进行闭环试算。
- 形成“整定入围方案集合”。
- 给出：
  - `tuning_selection_score`
  - shortlist 入围标记与原因
  - 当前内部首选方案

不负责：

- 给出最终上线结论。

### 3. 评估智能体

负责：

- 接收 PID 专家入围方案。
- 对入围方案逐个做独立验收。
- 在评估阶段选出冠军方案。
- 输出：
  - `acceptance_performance_score`
  - `robustness_score`
  - `constraint_score`
  - `online_readiness_score`
  - `passed`
  - `feedback_target`
  - `feedback_action`
  - `launch_recommendation`

不负责：

- 重新参与 PID 整定排序。

## 三、系统辨识智能体改造点

### 后端

涉及模块：

- `backend/services/identification_service.py`
- `backend/services/tool_adapter_service.py`
- `backend/orchestration/event_mapper.py`
- `backend/orchestration/workflow_runner.py`

改造建议：

1. 保留多个辨识候选。
   - 每个候选至少包含：
     - `window_source`
     - `model_type`
     - `selected_model_params`
     - `r2_score`
     - `normalized_rmse`
     - `identification_fit_score`
     - `confidence`
     - `points`
2. 辨识排序只按拟合质量。
   - 排序优先级：
     1. `identification_fit_score`
     2. `r2_score`
     3. `normalized_rmse`
     4. `confidence`
3. 明确结构化输出。
   - 输出：
     - `identification_best_model_type`
     - `identification_best_window_source`
     - `identification_candidates`
4. `window_quality_score` 仅保留为说明字段。
   - 不再主导辨识排序。

### 前端

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

改造建议：

1. 系统辨识详情展示候选列表。
2. 明确显示：
   - `辨识主候选模型`
   - `辨识主候选窗口`
   - `辨识拟合评分`
3. 候选列表中显示：
   - 模型类型
   - 窗口来源
   - `R²`
   - `NRMSE`
   - `identification_fit_score`
   - 参数摘要

## 四、PID 专家智能体改造点

### 后端

涉及模块：

- `backend/services/pid_tuning_service.py`
- `backend/services/tool_adapter_service.py`
- `backend/skills/rating.py`

改造建议：

1. 输入改为多个辨识候选。
2. 对每个候选模型尝试多种整定策略：
   - `IMC`
   - `LAMBDA`
   - `ZN`
   - `CHR`
3. 生成 `tuning_model_candidates`。
   - 每个元素至少包含：
     - `model_type`
     - `selected_model_params`
     - `window_source`
     - `identification_fit_score`
     - `best_strategy`
     - `best_performance_score`
     - `best_final_rating`
     - `is_stable`
     - `pid_params`
4. 引入 PID 入围标准。

#### PID 入围标准

不采用写死 `top3` 的方式，而采用“硬门槛 + 上限控制”：

- 硬门槛：
  - `identification_fit_score >= 8.0`
  - `best_performance_score >= 7.0`
  - `best_final_rating >= 7.0`
  - `is_stable == true`
- 约束门槛：
  - 不允许明显严重饱和
  - 不允许显著发散或过量振荡
- 上限控制：
  - 所有通过门槛的方案都可入围
  - 仅在入围数过多时，按排序截断到上限，例如 5 组

建议新增字段：

- `shortlist_passed`
- `shortlist_reasons`
- `shortlist_score`

其中：

- `shortlist_passed`：是否进入评估阶段
- `shortlist_reasons`：通过/未通过原因
- `shortlist_score`：仅作为 shortlist 内部排序辅助分，不代替硬门槛

#### PID 试算职责

PID 专家负责“候选比较”，不是“最终验收”。

建议试算场景：

- 正向设定值阶跃
- 反向设定值阶跃
- 扰动抑制

建议输出：

- `tuning_selection_score`
- `tuning_model_candidates`
- `tuning_shortlist_candidates`
- `tuning_selected_model_type`
- `tuning_selected_window_source`
- `selected_pid_params`

### 前端

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

改造建议：

1. 区分：
   - `辨识最优模型`
   - `整定采用模型`
2. 增加“整定候选列表”。
3. 增加“整定入围方案”展示。
4. 显示：
   - 模型
   - 窗口来源
   - 策略
   - PID 参数
   - `tuning_selection_score`
   - shortlist 是否入围

## 五、评估智能体改造点

### 后端

涉及模块：

- `backend/services/pid_evaluation_service.py`
- `backend/services/tool_adapter_service.py`
- `backend/skills/rating.py`
- `backend/orchestration/event_mapper.py`
- `backend/orchestration/workflow_runner.py`

改造建议：

1. 输入改为 PID 入围方案集合。
   - 评估智能体不再只吃 PID 专家当前冠军。
   - 而是接收 `tuning_shortlist_candidates`。
2. 对每个入围方案做独立验收。

#### 评估场景

- 正向阶跃
- 反向阶跃
- 扰动抑制
- 模型参数摄动
- 饱和/约束检查

#### 评估输出

- `acceptance_performance_score`
  - 用于衡量多场景闭环表现
- `robustness_score`
  - 用于衡量参数摄动后的稳定性和性能退化
- `constraint_score`
  - 用于衡量饱和风险和控制输出剧烈程度
- `online_readiness_score`
  - 综合验收分，用于冠军选择和上线建议
- `evaluation_candidates`
  - 所有入围方案的独立验收结果
- `evaluation_selected_candidate`
  - 评估冠军方案

#### 冠军选择原则

冠军选择不再由 PID 专家单独决定。

评估阶段应：

1. 仅在 PID shortlist 中选择。
2. 按以下优先级选择冠军：
   1. `online_readiness_score`
   2. `acceptance_performance_score`
   3. `robustness_score`
   4. `constraint_score`

#### 通过标准

建议第一版采用：

- `online_readiness_score >= 7.0`
- 冠军方案必须稳定
- 不允许严重约束风险

#### 输出语义

评估智能体输出的是“最终验收结果”，不是 PID 试算内部评分。

建议保留兼容字段：

- `performance_score` 可兼容映射到 `acceptance_performance_score`
- `final_rating` 可兼容映射到 `online_readiness_score`

### 前端

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

改造建议：

1. 评估智能体显示“验收冠军方案”。
2. 显示多场景验收明细。
3. 区分：
   - `PID 试算分`
   - `评估验收分`
4. 增加字段：
   - `acceptance_performance_score`
   - `robustness_score`
   - `constraint_score`
   - `online_readiness_score`
   - `launch_recommendation`

## 六、推荐实施顺序

1. 系统辨识输出多个候选。
2. PID 专家形成 shortlist。
3. 评估智能体接管冠军选择。
4. 最后再升级仿真器数值质量。

## 七、验收标准

### 系统辨识智能体

- 用户可看到多个辨识候选。
- 候选按 `identification_fit_score` 排序。

### PID 专家智能体

- 用户可看到 shortlist 规则生效后的入围方案。
- “整定采用模型”不必等于“辨识最优模型”。

### 评估智能体

- 评估智能体基于 shortlist 独立选冠军。
- 可明确看到：
  - 冠军方案
  - 是否通过
  - 是否建议上线
  - 不通过时回流给谁
