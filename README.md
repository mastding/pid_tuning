# PID多智能体整定系统

基于AutoGen框架和大语言模型的智能PID参数整定系统，通过多智能体协作实现自动化的控制回路参数优化。

## 系统概述

本系统采用4个专业智能体协作完成PID整定任务：
1. **数据分析智能体** - 加载和分析历史数据
2. **系统辨识智能体** - 拟合FOPDT模型
3. **PID专家智能体** - 计算和转换PID参数
4. **评估智能体** - 评估整定质量

## 技术栈

- **后端**: FastAPI + Python 3.11
- **AI框架**: AutoGen (Microsoft)
- **LLM**: 千问3-max (Qwen3-max)
- **前端**: Vue 3 + Tailwind CSS
- **算法**: FOPDT模型、IMC/Lambda/CHR/ZN整定规则

## 项目结构

```
/run/code/dinglei/pid/
├── agents_multiagent.py          # 多智能体核心实现
├── backend_api.py                 # FastAPI后端API
├── frontend/
│   └── index.html                 # Vue3前端界面
├── skills/                        # 核心算法模块
│   ├── system_id_skills.py       # 系统辨识算法
│   ├── pid_tuning_skills.py      # PID整定算法
│   ├── rating.py                 # 评估算法
│   └── data_analysis_skills.py   # 数据分析算法
├── backup/                        # 备份文件
├── .env                          # 环境变量配置
├── requirements.txt              # Python依赖
├── start_services.sh             # 启动脚本
└── stop_services.sh              # 停止脚本
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- 千问API访问权限
- 8GB+ 内存

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 到 `.env` 并配置：

```bash
# LLM配置
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=http://openai.dicp.sixseven.ltd:5924/v1
LLM_MODEL=qwen3-max
```

### 4. 启动服务

```bash
# 使用启动脚本
./start_services.sh

# 或手动启动
nohup python backend_api.py > backend.log 2>&1 &
```

### 5. 访问系统

打开浏览器访问：`http://your-server:3443`

## 使用说明

### 数据格式要求

CSV文件必须包含以下列：
- `MV`: 操纵变量（Manipulated Variable）
- `PV`: 过程变量（Process Variable）

示例：
```csv
MV,PV
0,20
0,20
50,25
50,30
```

### 整定流程

1. **上传数据**: 选择包含MV和PV的CSV文件
2. **配置参数**:
   - 控制回路名称（如：FIC_101A）
   - 控制器品牌（Siemens/ABB/Honeywell/Yokogawa）
   - 整定策略（IMC/Lambda/CHR/ZN）
3. **开始整定**: 点击开始整定按钮
4. **查看过程**: 实时查看4个智能体的协作过程
5. **获取结果**: 查看最终的PID参数和评估结果

### 整定策略说明

- **IMC (Internal Model Control)**: 适用于大多数过程，鲁棒性好
- **Lambda**: 适用于需要平滑控制的场景
- **CHR (Chien-Hrones-Reswick)**: 适用于快速响应场景
- **ZN (Ziegler-Nichols)**: 经典方法，适用于一般场景

## API文档

### POST /api/tune_stream

PID整定流式接口，通过SSE返回实时进度。

**请求参数**:
- `file`: CSV文件（multipart/form-data）
- `loop_name`: 控制回路名称
- `controller_brand`: 控制器品牌
- `strategy`: 整定策略

**响应格式** (SSE):
```json
data: {type: agent_turn, agent: 数据分析智能体, tools: [...], response: ...}
data: {type: result, data: {...}}
data: {type: done, status: succeeded}
```

### GET /api/status

获取系统状态。

**响应**:
```json
{
  status: running,
  version: 2.0.0,
  mode: multi-agent
}
```

## 核心算法

### FOPDT模型

一阶加纯滞后模型（First Order Plus Dead Time）：

```
G(s) = K * e^(-L*s) / (T*s + 1)
```

其中：
- K: 过程增益
- T: 时间常数
- L: 纯滞后时间

### IMC整定规则

```
Kp = (T + 0.5*L) / (K * (λ + 0.5*L))
Ki = Kp / T
Kd = Kp * L / 2
```

其中 λ 为闭环时间常数，通常取 λ = T。

## 多智能体协作流程

```
用户上传CSV
    ↓
数据分析智能体
    ├─ 调用 tool_load_data
    └─ 返回数据统计信息
    ↓
系统辨识智能体
    ├─ 调用 tool_fit_fopdt
    └─ 返回FOPDT模型参数 (K, T, L)
    ↓
PID专家智能体
    ├─ 调用 tool_tune_pid
    ├─ 调用 tool_translate_params
    └─ 返回PID参数和控制器参数
    ↓
评估智能体
    ├─ 调用 tool_evaluate_pid
    └─ 返回评估结果和APPROVE信号
    ↓
返回最终结果
```

## 性能优化

### 全局数据存储

为避免在智能体间传递大数组（如150000个数据点），系统使用全局数据存储：

```python
_shared_data_store = {}

# 数据分析智能体保存数据
_shared_data_store[mv] = mv_array
_shared_data_store[pv] = pv_array

# 系统辨识智能体直接读取
mv = _shared_data_store[mv]
pv = _shared_data_store[pv]
```

### 流式输出

使用SSE（Server-Sent Events）实现实时流式输出，降低首字节时间。

## 故障排查

### 后端无法启动

1. 检查端口占用：`lsof -i :3443`
2. 检查日志：`tail -f backend.log`
3. 检查Python版本：`python --version`

### LLM调用失败

1. 检查API密钥配置
2. 检查网络连接
3. 查看后端日志中的错误信息

### 前端无法访问

1. 检查后端服务状态
2. 检查防火墙设置
3. 检查Nginx配置（如使用）

## 维护和监控

### 查看日志

```bash
# 后端日志
tail -f backend.log

# 前端日志
tail -f frontend/frontend.log
```

### 重启服务

```bash
./stop_services.sh
./start_services.sh
```

### 备份数据

```bash
# 备份整个项目
cd /run/code/dinglei
tar -czf pid_backup_$(date +%Y%m%d_%H%M%S).tar.gz pid/
```

## 开发指南

### 添加新的整定策略

1. 在 `skills/pid_tuning_skills.py` 中添加新策略
2. 更新 `apply_tuning_rules()` 函数
3. 在前端添加新选项

### 添加新的控制器品牌

1. 在 `skills/pid_tuning_skills.py` 中添加转换逻辑
2. 更新 `controller_logic_translator()` 函数
3. 在前端添加新选项

### 自定义智能体

参考 `ARCHITECTURE_REFACTORING.md` 中的智能体模块化方案。

## 版本历史

### v2.0.0 (2026-03-07)
- ✅ 完成代码清理，移除未使用代码
- ✅ 只保留智能体模式
- ✅ 优化多智能体协作流程
- ✅ 修复智能体回复显示问题
- ✅ 创建架构重构建议文档

### v1.0.0 (2026-03-06)
- ✅ 实现基于AutoGen的多智能体协作
- ✅ 集成千问LLM
- ✅ 实现4个专业智能体
- ✅ Vue3前端界面
- ✅ SSE流式输出

## 许可证

内部项目，仅供授权用户使用。

## 联系方式

如有问题，请联系项目维护团队。

## 相关文档

- [架构重构建议](./ARCHITECTURE_REFACTORING.md)
- [代码清理总结](./CLEANUP_SUMMARY.md)
- [备份文档](./backup/docs/)
