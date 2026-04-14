# 智能体架构改造开发清单

本文档将 `ARCHITECTURE_AGENT_REFACTOR_PLAN.md` 拆成可执行任务，便于逐项落地。

## 一、系统辨识智能体

### 后端

- [x] 输出 `identification_candidates`
- [x] 输出：
  - [x] `identification_best_model_type`
  - [x] `identification_best_window_source`
- [x] 候选包含：
  - [x] `window_source`
  - [x] `model_type`
  - [x] `selected_model_params`
  - [x] `r2_score`
  - [x] `normalized_rmse`
  - [x] `identification_fit_score`
  - [x] `confidence`

### 前端

- [x] 系统辨识详情支持展示候选列表
- [x] 候选列表显示：
  - [x] `R²`
  - [x] `NRMSE`
  - [x] `identification_fit_score`
  - [x] 模型参数摘要

## 二、PID 专家智能体

### 后端

- [x] 支持读取多个辨识候选
- [x] 对每个候选模型尝试多种整定策略
- [x] 输出 `tuning_model_candidates`
- [x] 引入正式 shortlist 机制
  - [x] 增加 `shortlist_passed`
  - [x] 增加 `shortlist_reasons`
  - [x] 增加 `shortlist_score`
  - [x] 输出 `tuning_shortlist_candidates`
- [x] shortlist 采用门槛制，不写死 top3
  - [x] `identification_fit_score >= 8.0`
  - [x] `best_performance_score >= 7.0`
  - [x] `best_final_rating >= 7.0`
  - [x] `is_stable == true`
  - [x] 严重饱和/发散方案直接淘汰
  - [x] 仅在入围数过多时按上限截断，例如最多 5 组

### 前端

- [x] 区分“辨识最优模型”和“整定采用模型”
- [x] 显示多模型整定候选
- [ ] 显示 shortlist 入围状态与原因

## 三、评估智能体

### 后端

- [x] 输入改为 `tuning_shortlist_candidates`
- [x] 不再只评 PID 专家当前冠军
- [x] 对每个入围方案做独立验收
- [x] 增加评估场景：
  - [x] 正向阶跃
  - [x] 反向阶跃
  - [x] 扰动抑制
  - [x] 模型参数摄动
  - [x] 饱和/约束检查
- [x] 增加验收字段：
  - [x] `acceptance_performance_score`
  - [x] `robustness_score`
  - [x] `constraint_score`
  - [x] `online_readiness_score`
  - [x] `evaluation_candidates`
  - [x] `evaluation_selected_candidate`
- [x] 评估阶段选冠军
  - [x] 优先按 `online_readiness_score`
  - [x] 其次按 `acceptance_performance_score`
  - [x] 再次按 `robustness_score`
  - [x] 再次按 `constraint_score`
- [x] 通过标准：
  - [x] `online_readiness_score >= 7.0`
  - [x] 冠军方案稳定
  - [x] 无严重约束风险
- [x] 保留兼容字段
  - [x] `performance_score -> acceptance_performance_score`
  - [x] `final_rating -> online_readiness_score`

### 前端

- [ ] 显示评估冠军方案
- [ ] 显示多场景验收结果
- [ ] 区分 PID 试算分和评估验收分
- [ ] 展示：
  - [ ] `acceptance_performance_score`
  - [ ] `robustness_score`
  - [ ] `constraint_score`
  - [ ] `online_readiness_score`
  - [ ] `launch_recommendation`

## 四、闭环仿真器升级

### 后端

- [ ] 一阶对象改为更稳定的离散化方式
- [ ] 二阶对象改为稳定串联离散化
- [ ] 支持更合理的滞后处理
- [ ] 让 PID 仿真支持更真实的抗积分饱和与约束

## 五、当前推荐顺序

1. [x] 系统辨识输出多个候选
2. [x] PID 专家支持多候选比较
3. [x] PID 专家形成正式 shortlist
4. [x] 评估智能体基于 shortlist 选冠军
5. [ ] 前端补齐评估冠军与验收字段展示
