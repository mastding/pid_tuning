# 智能体架构改造开发清单

本文档将 `ARCHITECTURE_AGENT_REFACTOR_PLAN.md` 进一步拆成可执行的开发任务清单，方便逐项落地。


## 一、第一阶段：系统辨识智能体输出多个候选

### 后端

- [ ] 修改 [identification_service.py](D:\code\pid_new\backend\services\identification_service.py)
  - [ ] 保留前 `3~5` 个高质量辨识候选
  - [ ] 输出 `identification_candidates`
  - [ ] 每个候选包含：
    - [ ] `window_source`
    - [ ] `model_type`
    - [ ] `selected_model_params`
    - [ ] `r2_score`
    - [ ] `normalized_rmse`
    - [ ] `identification_fit_score`
    - [ ] `confidence`
- [ ] 修改 [tool_adapter_service.py](D:\code\pid_new\backend\services\tool_adapter_service.py)
  - [ ] 将多个辨识候选写入 `session_store`
  - [ ] 区分：
    - [ ] `identification_best_model_type`
    - [ ] `identification_best_window_source`
- [ ] 修改 [event_mapper.py](D:\code\pid_new\backend\orchestration\event_mapper.py)
  - [ ] 事件映射中增加辨识候选摘要

### 前端

- [ ] 修改 [app.js](D:\code\pid_new\frontend\js\app.js)
  - [ ] 增加 `identification_candidates` 的 payload 解析
  - [ ] 系统辨识详情页支持候选列表排序
- [ ] 修改 [app-template.html](D:\code\pid_new\frontend\public\app-template.html)
  - [ ] 展示“辨识最优模型”
  - [ ] 展示“辨识候选列表”
  - [ ] 候选列表中显示：
    - [ ] `R²`
    - [ ] `NRMSE`
    - [ ] `identification_fit_score`


## 二、第二阶段：PID 专家支持多候选模型整定

### 后端

- [ ] 修改 [pid_tuning_service.py](D:\code\pid_new\backend\services\pid_tuning_service.py)
  - [ ] 支持输入多个辨识候选
  - [ ] 对每个候选模型尝试多种整定策略
  - [ ] 为每个组合计算 `tuning_selection_score`
  - [ ] 输出 `tuning_candidates`
- [ ] 修改 [tool_adapter_service.py](D:\code\pid_new\backend\services\tool_adapter_service.py)
  - [ ] 将 PID 专家输入改成“候选模型集合”
  - [ ] 输出：
    - [ ] `selected_tuning_model_type`
    - [ ] `selected_tuning_window_source`
    - [ ] `selected_strategy`
    - [ ] `selected_pid_params`
    - [ ] `tuning_selection_score`
- [ ] 修改 PID 专家试算逻辑
  - [ ] 引入快速筛选层
  - [ ] 引入精细比较层
  - [ ] 试算点数按模型时间尺度自适应

### 前端

- [ ] 修改 [app.js](D:\code\pid_new\frontend\js\app.js)
  - [ ] 解析 `tuning_candidates`
  - [ ] 解析 `selected_tuning_model_type`
  - [ ] 区分“辨识最优模型”和“整定采用模型”
- [ ] 修改 [app-template.html](D:\code\pid_new\frontend\public\app-template.html)
  - [ ] 在 PID 专家区域显示：
    - [ ] `辨识最优模型`
    - [ ] `整定采用模型`
    - [ ] `整定采用窗口`
    - [ ] `整定采用策略`
    - [ ] `最终 PID`
    - [ ] `tuning_selection_score`
  - [ ] 增加“整定候选列表”展示


## 三、第三阶段：评估智能体升级为最终验收

### 后端

- [ ] 修改 [pid_evaluation_service.py](D:\code\pid_new\backend\services\pid_evaluation_service.py)
  - [ ] 输入只接受 PID 专家最终选定的模型和 PID 参数
  - [ ] 扩展评估场景：
    - [ ] 正向阶跃
    - [ ] 反向阶跃
    - [ ] 扰动抑制
    - [ ] 模型参数摄动
    - [ ] 饱和/约束检查
  - [ ] 输出：
    - [ ] `acceptance_performance_score`
    - [ ] `robustness_score`
    - [ ] `constraint_score`
    - [ ] `online_readiness_score`
- [ ] 修改 [tool_adapter_service.py](D:\code\pid_new\backend\services\tool_adapter_service.py)
  - [ ] 将评估智能体语义改为“最终验收”
  - [ ] 保留自动回流逻辑
  - [ ] 回流依据改为验收分数而不是整定候选分数
- [ ] 修改 [agents_multiagent.py](D:\code\pid_new\backend\agents_multiagent.py)
  - [ ] 关闭经验模块时，跳过经验沉淀/落库

### 前端

- [ ] 修改 [app.js](D:\code\pid_new\frontend\js\app.js)
  - [ ] 解析新的评估字段
  - [ ] 区分：
    - [ ] `tuning_selection_score`
    - [ ] `online_readiness_score`
- [ ] 修改 [app-template.html](D:\code\pid_new\frontend\public\app-template.html)
  - [ ] 评估区域只显示最终验收结果
  - [ ] 显示：
    - [ ] `acceptance_performance_score`
    - [ ] `online_readiness_score`
    - [ ] `passed`
    - [ ] `launch_recommendation`
    - [ ] `failure_reason`
    - [ ] `feedback_target`


## 四、第四阶段：闭环仿真器升级

### 后端

- [ ] 修改 [rating.py](D:\code\pid_new\backend\skills\rating.py)
  - [ ] 一阶模型改为精确离散化
  - [ ] 二阶模型改为稳定串联离散化
  - [ ] 纯滞后支持分数延迟
  - [ ] PID 增加更真实的积分限幅/微分滤波
  - [ ] 内部仿真步长支持自适应细化
- [ ] 修改 PID 专家与评估智能体调用方式
  - [ ] 支持快速仿真与精细仿真两套参数


## 五、推荐实施优先级

建议按下面顺序执行：

1. 第一优先级
- [ ] 系统辨识输出多个候选
- [ ] 前端系统辨识详情支持候选列表

2. 第二优先级
- [ ] PID 专家支持多模型整定比较
- [ ] 区分“辨识最优模型”和“整定采用模型”

3. 第三优先级
- [ ] 评估智能体升级为最终验收
- [ ] 验收分与整定分彻底分开

4. 第四优先级
- [ ] 闭环仿真器数值模型升级


## 六、验收标准

### 系统辨识智能体

- [ ] 用户能在前端看到多个辨识候选
- [ ] 候选按 `identification_fit_score` 排序
- [ ] 能明确区分“辨识最优模型”

### PID 专家智能体

- [ ] 用户能看到整定采用模型不一定等于辨识最优模型
- [ ] 能查看多个整定候选
- [ ] `tuning_selection_score` 仅用于整定候选比较

### 评估智能体

- [ ] 用户能看到独立的最终验收评分
- [ ] 系统能明确判断：
  - [ ] 通过
  - [ ] 回流给 PID 专家
  - [ ] 回流给系统辨识
  - [ ] 回流给数据分析

