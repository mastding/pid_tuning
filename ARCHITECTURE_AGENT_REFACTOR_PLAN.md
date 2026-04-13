# 智能体架构改造方案

本文档整理当前 PID 整定流程中，系统辨识智能体、PID 专家智能体、评估智能体三者的职责重构思路，并分别列出前端、后端建议修改点。

目标不是一次性推翻现有实现，而是在当前工程基础上逐步演进到更清晰、更工程化的三阶段架构：

1. 系统辨识智能体负责“找出高质量模型候选”
2. PID 专家智能体负责“基于候选模型选择最优整定方案”
3. 评估智能体负责“对最终整定方案做独立验收并给出上线/回流建议”


## 一、当前问题概述

当前流程的核心问题有三类：

- 系统辨识阶段与 PID 整定阶段耦合过深
  - 历史实现中，系统辨识选模会混入闭环试算分数和窗口质量惩罚，导致“拟合最好”和“最终选中”不一致。
- PID 专家阶段和评估阶段职责边界不清
  - PID 专家会做闭环试算并给出分数，评估智能体也会再次做仿真和评分，存在职责重叠。
- 前端展示口径容易混淆
  - 用户难以区分：
    - 辨识最优模型
    - 整定采用模型
    - 最终验收结果


## 二、目标架构

建议将三类智能体职责明确拆分为：

### 1. 系统辨识智能体

只负责：

- 在“可用于辨识”的窗口集合中尝试多模型拟合
- 计算拟合指标：
  - `R²`
  - `NRMSE`
  - `identification_fit_score`
- 输出多个高质量候选辨识结果

不负责：

- 直接决定最终 PID 参数
- 用闭环性能分数决定最终整定模型


### 2. PID 专家智能体

负责：

- 接收多个辨识候选模型
- 对每个候选模型尝试多种整定策略
- 针对每个候选模型做闭环试算
- 输出整定采用模型、整定采用策略、最终 PID 参数
- 输出用于候选比较的整定评分：
  - `tuning_selection_score`

不负责：

- 直接给出“可以上线”结论


### 3. 评估智能体

负责：

- 接收 PID 专家最终选中的模型与 PID 参数
- 独立做验收级评估
- 输出：
  - `acceptance_performance_score`
  - `online_readiness_score`
  - `passed`
  - `feedback_target`
  - `feedback_action`
  - 上线建议

不负责：

- 再次参与整定候选排序


## 三、系统辨识智能体改造点

### 后端修改建议

涉及模块：

- `backend/services/identification_service.py`
- `backend/services/tool_adapter_service.py`
- 可能影响：
  - `backend/orchestration/workflow_runner.py`
  - `backend/orchestration/event_mapper.py`

建议改造内容：

1. 保留多窗口、多模型候选结果
- 当前辨识结果不能只保留一个最终模型。
- 应保留前 `N=3~5` 个高质量候选。

每个候选建议至少包含：

- `window_source`
- `model_type`
- `selected_model_params`
- `r2_score`
- `normalized_rmse`
- `identification_fit_score`
- `confidence`
- `window_start/end`

2. 系统辨识排序只看拟合质量
- 当前已经往这个方向走。
- 最终排序建议固定为：
  1. `identification_fit_score`
  2. `R²`
  3. `NRMSE`
  4. `confidence` 仅作弱 tie-break

3. `window_quality_score` 退化为展示信息
- 数据分析阶段可继续计算 `window_quality_score`
- 但系统辨识阶段不再用它主导排序
- 最多作为过滤或说明字段

4. 输出字段区分“辨识最优”
- 后端结果中明确输出：
  - `identification_best_model_type`
  - `identification_best_window_source`
  - `identification_candidates`

不要再默认“辨识最优 = 整定采用模型”。


### 前端修改建议

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

建议改造内容：

1. 系统辨识详情页展示候选列表
- 按 `identification_fit_score` 排序
- 展示每个候选的：
  - 模型类型
  - 窗口来源
  - `R²`
  - `NRMSE`
  - `identification_fit_score`

2. 顶部卡片明确写“辨识最优模型”
- 不再只写“当前选中模型”
- 建议字段：
  - `辨识最优模型`
  - `辨识最优窗口`
  - `辨识拟合评分`

3. 文案统一
- 不再用“候选评分”描述辨识阶段结果
- 统一改成：
  - `辨识拟合评分`


## 四、PID 专家智能体改造点

### 后端修改建议

涉及模块：

- `backend/services/pid_tuning_service.py`
- `backend/services/tool_adapter_service.py`
- 可能影响：
  - `backend/orchestration/event_mapper.py`

建议改造内容：

1. 输入改成“多个辨识候选”
- PID 专家不能只吃单个最优模型
- 应接收 `identification_candidates`

2. 对每个候选模型分别整定
- 对每个候选模型尝试：
  - `IMC`
  - `LAMBDA`
  - `ZN`
  - `CHR`

3. 引入 `tuning_selection_score`
- 用于候选整定方案比较
- 这是 PID 专家内部排序分，不是最终验收分

建议输出：

- `selected_tuning_model_type`
- `selected_tuning_window_source`
- `selected_strategy`
- `selected_pid_params`
- `tuning_selection_score`
- `tuning_candidates`

4. 试算按每个窗口进行
- 同一模型类型，不同窗口必须分开试算
- 比较对象应是：
  - `窗口 + 模型 + 策略`

5. 试算分层
- 快速筛选层
  - 用较少点数
  - 筛掉明显失稳/明显过激方案
- 精细比较层
  - 对前几名做更长时域仿真

6. 试算点数建议
- 不建议固定 500 点
- 建议按模型时间尺度自适应：
  - 快速筛选：`120~300` 点
  - 精细比较：`240~800` 点

建议依据：

- `FO/FOPDT`
  - `T_eq = T`
- `SOPDT`
  - `T_eq = T1 + T2`
- `IPDT`
  - `T_eq = max(L, 60s)`

7. 内部仿真步长建议
- 不直接等于原始数据采样周期
- 建议：
  - `dt_sim = min(dt_data, T_eq / 10)`
- 这样快对象不会因为步长过粗而数值失真

8. `tuning_selection_score` 建议指标

建议权重：

- 跟踪性能：40%
  - 超调
  - 调节时间
  - 稳态误差
- 振荡/阻尼：20%
  - 振荡次数
  - 衰减比
- 控制代价：15%
  - `MV` 总变化量
  - 峰值控制动作
- 饱和风险：10%
  - `MV` 饱和比例
- 鲁棒性粗测：15%
  - 模型参数轻微摄动后的表现

9. PID 专家输出语义要改清楚
- 当前输出容易被理解为“最终验收结论”
- 改造后应明确：
  - 这是“整定候选选择结果”
  - 不是最终上线结论


### 前端修改建议

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

建议改造内容：

1. PID 专家卡片增加“整定采用模型”
- 需要和“辨识最优模型”区分

建议展示：

- `辨识最优模型`
- `整定采用模型`
- `整定采用窗口`
- `整定采用策略`
- `最终 PID`
- `tuning_selection_score`

2. 增加整定候选列表
- 展示前几名整定候选
- 每个候选包含：
  - 模型类型
  - 窗口来源
  - 策略
  - PID 参数
  - `tuning_selection_score`

3. 文案统一
- PID 专家阶段的分数统一叫：
  - `整定候选评分`
  - 或 `tuning_selection_score`

不要和评估阶段的最终分混用。


## 五、评估智能体改造点

### 后端修改建议

涉及模块：

- `backend/services/pid_evaluation_service.py`
- `backend/services/tool_adapter_service.py`
- `backend/skills/rating.py`

建议改造内容：

1. 评估智能体只评最终采用方案
- 输入：
  - `selected_tuning_model_type`
  - `selected_model_params`
  - `selected_pid_params`

2. 评估与 PID 专家的试算职责区分
- PID 专家：
  - 负责“选方案”
- 评估智能体：
  - 负责“验收方案”

3. 引入新的验收分

建议字段：

- `acceptance_performance_score`
- `robustness_score`
- `constraint_score`
- `online_readiness_score`

4. 评估场景比 PID 专家更严格

建议至少包含：

- 正向设定值阶跃
- 反向设定值阶跃
- 扰动抑制
- 参数摄动鲁棒性
- 饱和/约束检查

5. 当前 `performance_score` 可保留，但语义需调整
- 当前 `performance_score` 更接近“单场景闭环表现”
- 后续建议把它作为验收子项，不直接等于最终上线分

6. 输出最终验收结论

建议输出：

- `passed`
- `online_readiness_score`
- `failure_reason`
- `feedback_target`
- `feedback_action`
- `launch_recommendation`

7. 自动回流逻辑建议保留
- 评估不通过时：
  - 回流 `pid_expert`
  - 回流 `system_id_expert`
  - 回流 `data_analyst`

8. 若关闭经验模块，评估后不应再强制做经验沉淀
- 当前已发现“经验检索关闭但评估后收尾仍走经验记录”的问题
- 后续建议：
  - `ENABLE_EXPERIENCE_DISTILLATION=0` 时
  - 直接跳过经验落库


### 前端修改建议

涉及模块：

- `frontend/js/app.js`
- `frontend/public/app-template.html`

建议改造内容：

1. 评估页只展示“最终验收结果”

建议展示：

- `acceptance_performance_score`
- `online_readiness_score`
- `passed`
- `launch_recommendation`
- `failure_reason`
- `feedback_target`

2. 文案明确区分“整定评分”和“验收评分”

建议命名：

- PID 专家：
  - `整定候选评分`
- 评估智能体：
  - `最终验收评分`
  - `上线就绪评分`

3. 若发生回流，前端要明确原因
- 回流给谁
- 为什么回流
- 是模型问题、窗口问题还是 PID 问题


## 六、推荐字段命名

建议最终统一为：

### 系统辨识智能体

- `identification_fit_score`
- `identification_best_model_type`
- `identification_best_window_source`
- `identification_candidates`


### PID 专家智能体

- `tuning_selection_score`
- `selected_tuning_model_type`
- `selected_tuning_window_source`
- `selected_strategy`
- `selected_pid_params`
- `tuning_candidates`


### 评估智能体

- `acceptance_performance_score`
- `robustness_score`
- `constraint_score`
- `online_readiness_score`
- `launch_recommendation`


## 七、推荐实施顺序

建议按以下顺序逐步实施，降低风险：

1. 第一阶段
- 系统辨识阶段保留多个候选模型
- 前端系统辨识页面支持候选列表展示

2. 第二阶段
- PID 专家支持“多候选模型并行整定”
- 输出 `tuning_selection_score`
- 前端区分“辨识最优模型”和“整定采用模型”

3. 第三阶段
- 评估智能体升级为“最终验收”
- 引入 `online_readiness_score`
- 保留自动回流能力

4. 第四阶段
- 升级闭环仿真器
  - 精确离散化
  - 分数延迟
  - 更真实的 PID 控制器结构


## 八、总结

建议将整套流程的判断链改造成：

1. 数据分析智能体：
   - 找窗口
2. 系统辨识智能体：
   - 给模型候选
3. PID 专家智能体：
   - 从候选中选出最优整定方案
4. 评估智能体：
   - 对最终方案做独立验收并决定是否上线

这样可以避免：

- “辨识最优模型直接等于整定最优模型”
- “PID 专家和评估智能体重复评分”
- “前端展示口径混乱”

同时也更符合控制工程实际中的分工：

- 辨识是建模
- 整定是设计
- 评估是验收

