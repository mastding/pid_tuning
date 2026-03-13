# 外部工作流接口说明

本文档说明外部系统如何调用 PID 智能整定平台的异步工作流接口。

## 1. 接口概览

当前提供 3 个外部工作流接口：

1. `POST /api/agent/workflow/run`
   - 提交整定任务
   - 可异步立即返回 `task_id`

2. `GET /api/agent/workflow/status/{task_id}`
   - 查询任务状态与执行进度

3. `GET /api/agent/workflow/result/{task_id}`
   - 查询任务最终结果

## 2. 推荐调用方式

推荐外部系统按如下顺序调用：

1. 调用 `POST /api/agent/workflow/run`
2. 拿到 `task_id`
3. 轮询 `GET /api/agent/workflow/status/{task_id}`
4. 当 `status = success` 或 `failed` 后，再调用 `GET /api/agent/workflow/result/{task_id}`

## 3. 提交任务

### 3.1 请求

`POST /api/agent/workflow/run`

`Content-Type: application/json`

请求体示例：

```json
{
  "start_time": "2025-10-08 17:30:37",
  "end_time": "2025-10-08 18:00:37",
  "loop_type": "流量",
  "loop_uri": "/pid_zd/0b521c82a96d4107a564e4c2678bdeca",
  "response_mode": "async"
}
```

### 3.2 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `start_time` | string | 是 | 历史数据开始时间 |
| `end_time` | string | 是 | 历史数据结束时间 |
| `loop_type` | string | 是 | 回路类型，支持中文或英文，例如 `流量` / `flow` |
| `loop_uri` | string | 是 | 回路 URI |
| `response_mode` | string | 否 | 支持 `async`、`blocking`、`streaming`，默认 `async` |

### 3.3 回路类型映射

系统当前会自动把以下中文类型映射为内部类型：

- `流量 -> flow`
- `温度 -> temperature`
- `压力 -> pressure`
- `液位 -> level`

### 3.4 异步模式返回示例

当 `response_mode = async` 时，接口立即返回：

```json
{
  "code": 0,
  "message": "accepted",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908"
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `code` | `0` 表示受理成功 |
| `message` | 固定为 `accepted` |
| `task_id` | 本次任务唯一标识 |

### 3.5 同步模式返回示例

当 `response_mode = blocking` 时，接口会等待整定完成后直接返回结果：

```json
{
  "code": 0,
  "message": "success",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
  "result": {
    "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
    "loop_type": "flow",
    "loop_uri": "/pid_zd/0b521c82a96d4107a564e4c2678bdeca",
    "model": {
      "model_type": "SOPDT",
      "selected_model_params": {
        "model_type": "SOPDT",
        "K": 0.4444,
        "T1": 1.0,
        "T2": 1.0,
        "L": 0.0
      },
      "confidence": 0.8481,
      "normalized_rmse": 0.0627,
      "r2_score": 0.9961
    },
    "pid": {
      "strategy_used": "LAMBDA",
      "Kp": 0.544,
      "Ki": 0.359,
      "Kd": 0.217
    },
    "evaluation": {
      "performance_score": 8.4,
      "method_confidence": 0.8481,
      "final_rating": 8.1,
      "passed": true,
      "failure_reason": "",
      "feedback_target": "",
      "feedback_action": ""
    },
    "experience": {
      "experience_id": "exp_20260313_120000_000001",
      "match_count": 3,
      "preferred_strategy": "LAMBDA",
      "preferred_model_type": "SOPDT"
    },
    "tuning_advice": {
      "summary": "建议采用当前整定参数，可直接在低风险工况下试投用。",
      "recommendation_level": "recommended",
      "actions": [
        "优先在低风险工况下试投用",
        "投用后重点观察超调与振荡"
      ],
      "risks": [],
      "rollback_advice": "如现场振荡明显增大，建议回退到原 PID 参数。",
      "operator_note": "建议先观察 1 到 2 个完整调节周期。"
    }
  }
}
```

## 4. 查询任务状态

### 4.1 请求

`GET /api/agent/workflow/status/{task_id}`

### 4.2 返回示例

```json
{
  "code": 0,
  "message": "success",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
  "status": "running",
  "created_at": "2026-03-13T17:00:00+08:00",
  "started_at": "2026-03-13T17:00:01+08:00",
  "finished_at": null,
  "progress": {
    "current_stage": "pid_tuning",
    "current_stage_display": "PID参数整定",
    "percent": 75
  },
  "error_message": null
}
```

### 4.3 字段说明

| 字段 | 说明 |
|---|---|
| `status` | 任务状态：`pending` / `running` / `success` / `failed` |
| `created_at` | 任务创建时间 |
| `started_at` | 任务开始执行时间 |
| `finished_at` | 任务结束时间，未结束时为 `null` |
| `progress.current_stage` | 当前阶段编码 |
| `progress.current_stage_display` | 当前阶段中文名称 |
| `progress.percent` | 粗粒度执行百分比 |
| `error_message` | 失败时的错误信息 |

### 4.4 当前阶段枚举

| 枚举值 | 说明 |
|---|---|
| `accepted` | 已受理 |
| `data_analysis` | 数据分析 |
| `model_identification` | 系统辨识 |
| `pid_tuning` | PID 整定 |
| `evaluation` | 结果评估 |
| `completed` | 已完成 |
| `failed` | 已失败 |

## 5. 查询任务结果

### 5.1 请求

`GET /api/agent/workflow/result/{task_id}`

### 5.2 任务未完成时返回示例

```json
{
  "code": 0,
  "message": "task not finished",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
  "status": "running",
  "finished_at": null,
  "result": null,
  "error_message": null
}
```

### 5.3 任务成功时返回示例

```json
{
  "code": 0,
  "message": "success",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
  "status": "success",
  "finished_at": "2026-03-13T17:00:15+08:00",
  "result": {
    "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
    "loop_type": "flow",
    "loop_uri": "/pid_zd/0b521c82a96d4107a564e4c2678bdeca",
    "model": {
      "model_type": "SOPDT",
      "selected_model_params": {
        "model_type": "SOPDT",
        "K": 0.4444,
        "T1": 1.0,
        "T2": 1.0,
        "L": 0.0
      },
      "confidence": 0.8481,
      "normalized_rmse": 0.0627,
      "r2_score": 0.9961
    },
    "pid": {
      "strategy_used": "LAMBDA",
      "Kp": 0.544,
      "Ki": 0.359,
      "Kd": 0.217
    },
    "evaluation": {
      "performance_score": 8.4,
      "method_confidence": 0.8481,
      "final_rating": 8.1,
      "passed": true,
      "failure_reason": "",
      "feedback_target": "",
      "feedback_action": ""
    },
    "experience": {
      "experience_id": "exp_20260313_120000_000001",
      "match_count": 3,
      "preferred_strategy": "LAMBDA",
      "preferred_model_type": "SOPDT"
    },
    "tuning_advice": {
      "summary": "建议采用当前整定参数，可直接在低风险工况下试投用。",
      "recommendation_level": "recommended",
      "actions": [
        "优先在低风险工况下试投用",
        "投用后重点观察超调与振荡"
      ],
      "risks": [],
      "rollback_advice": "如现场振荡明显增大，建议回退到原 PID 参数。",
      "operator_note": "建议先观察 1 到 2 个完整调节周期。"
    }
  },
  "error_message": null
}
```

### 5.4 任务失败时返回示例

```json
{
  "code": 1,
  "message": "workflow failed",
  "task_id": "2c86471f-64dd-4d91-9f1c-fde95e667908",
  "status": "failed",
  "finished_at": "2026-03-13T17:00:08+08:00",
  "result": null,
  "error_message": "上游模型网关连接异常"
}
```

## 6. `tuning_advice` 字段说明

`tuning_advice` 是面向外部系统和现场工程师的整定建议摘要，当前已进入最终结果返回。

字段说明：

| 字段 | 说明 |
|---|---|
| `summary` | 一句话整定建议 |
| `recommendation_level` | 建议等级：`recommended` / `cautious` / `not_recommended` |
| `actions` | 建议执行动作列表 |
| `risks` | 风险提示列表 |
| `rollback_advice` | 回退建议 |
| `operator_note` | 现场操作说明 |

## 7. 错误码约定

当前接口采用简化约定：

| `code` | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 失败 |

HTTP 状态码建议这样理解：

| HTTP 状态码 | 含义 |
|---|---|
| `200` | 请求成功、任务受理或查询成功 |
| `404` | `task_id` 不存在 |
| `500` | 工作流执行失败 |

## 8. 备注

1. 当前外部工作流接口推荐使用 `response_mode = async`。
2. 当前系统内部会自动把外部任务参数映射到现有智能整定工作流。
3. 如果外部系统需要完整长连接流式过程，可使用：
   - `response_mode = streaming`
4. 如果只需要结果，推荐：
   - `run -> status -> result`

