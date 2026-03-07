# Multi-Agent PID Tuning System using AutoGen RoundRobinGroupChat
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, AsyncGenerator, Dict, List

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

sys.path.append("/run/code/dinglei/pid")

from skills.system_id_skills import fit_fopdt_model, calculate_model_confidence
from skills.pid_tuning_skills import apply_tuning_rules, controller_logic_translator
from skills.rating import ModelRating


def create_model_client(*, model_api_key: str, model_api_url: str, model: str) -> OpenAIChatCompletionClient:
    """创建OpenAI兼容的模型客户端（千问）"""
    return OpenAIChatCompletionClient(
        api_key=model_api_key,
        base_url=model_api_url,
        model=model,
        temperature=0.7,
        max_tokens=2000,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
            "multiple_system_messages": False,
        },
    )


# ===== 全局数据存储 =====
_shared_data_store = {}

# ===== 工具函数定义 =====

async def tool_load_data(csv_path: str) -> Dict[str, Any]:
    """加载CSV数据"""
    df = pd.read_csv(csv_path)
    required_cols = ['MV', 'PV']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        available_cols = df.columns.tolist()
        col_mapping = {}
        for req_col in required_cols:
            for avail_col in available_cols:
                if req_col.lower() == avail_col.lower():
                    col_mapping[req_col] = avail_col
                    break
        if col_mapping:
            df = df.rename(columns={v: k for k, v in col_mapping.items()})
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV文件缺少必需的列: {', '.join(missing_cols)}")

    data_points = len(df)
    mv = df["MV"].values
    pv = df["PV"].values

    # 保存到全局存储，不返回大数组
    _shared_data_store['mv'] = mv
    _shared_data_store['pv'] = pv

    return {
        "data_points": data_points,
        "mv_range": [float(mv.min()), float(mv.max())],
        "pv_range": [float(pv.min()), float(pv.max())],
        "status": "数据已加载到内存"
    }


async def tool_fit_fopdt(dt: float = 1.0) -> Dict[str, Any]:
    """拟合FOPDT模型（从全局存储读取mv和pv）"""
    import numpy as np

    # 从全局存储读取数据
    if 'mv' not in _shared_data_store or 'pv' not in _shared_data_store:
        raise ValueError("请先调用tool_load_data加载数据")

    mv_array = _shared_data_store['mv']
    pv_array = _shared_data_store['pv']

    model_params = fit_fopdt_model(mv_array, pv_array, dt=dt)
    confidence = calculate_model_confidence(model_params["residue"])

    # 保存模型参数到全局存储
    _shared_data_store['K'] = float(model_params["K"])
    _shared_data_store['T'] = float(model_params["T"])
    _shared_data_store['L'] = float(model_params["L"])

    return {
        "K": float(model_params["K"]),
        "T": float(model_params["T"]),
        "L": float(model_params["L"]),
        "confidence": float(confidence["confidence"])
    }


async def tool_tune_pid(K: float, T: float, L: float, strategy: str) -> Dict[str, Any]:
    """应用PID整定规则"""
    pid_params = apply_tuning_rules(K, T, L, strategy)
    return {
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"])
    }


async def tool_translate_params(Kp: float, Ki: float, Kd: float, brand: str) -> Dict[str, Any]:
    """转换为控制器参数"""
    pid_params = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
    translated = controller_logic_translator(pid_params, brand)
    return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in translated.items()}


async def tool_evaluate_pid(K: float, T: float, L: float, Kp: float, Ki: float, Kd: float, method: str) -> Dict[str, Any]:
    """评估PID整定结果"""
    eval_result = ModelRating.evaluate(
        model_params={"K": K, "T": T, "L": L},
        pid_params={"Kp": Kp, "Ki": Ki, "Kd": Kd},
        method=method.lower()
    )
    return {
        "performance_score": float(eval_result["performance_score"]),
        "method_confidence": float(eval_result["method_confidence"]),
        "final_rating": float(eval_result["final_rating"]),
        "passed": eval_result["final_rating"] >= 6.0
    }


def create_pid_agents(
    *,
    model_client: OpenAIChatCompletionClient,
    csv_path: str,
    controller_brand: str,
    strategy: str
) -> List[AssistantAgent]:
    """创建4个专业智能体"""

    # 1. 数据分析智能体
    data_analyst = AssistantAgent(
        name="data_analyst",
        model_client=model_client,
        system_message=f"""你是数据分析专家。你的任务是加载CSV数据。

当轮到你发言时，立即调用tool_load_data(csv_path="{csv_path}")工具。
工具返回后，用一句话总结数据情况，例如："数据加载成功，共150000个数据点"。
然后说"完成"表示你的任务结束。""",
        tools=[tool_load_data],
        model_client_stream=True,
    )

    # 2. 系统辨识智能体
    system_id_expert = AssistantAgent(
        name="system_id_expert",
        model_client=model_client,
        system_message="""你是系统辨识专家。你的任务是拟合FOPDT模型。

当轮到你发言时：
1. 直接调用tool_fit_fopdt(dt=1.0)工具（数据已经在内存中）
2. 工具返回后，用一句话总结模型参数，例如："模型参数 K=2.5, T=100s, L=10s"
3. 说"完成"表示你的任务结束""",
        tools=[tool_fit_fopdt],
        model_client_stream=True,
    )

    # 3. PID专家智能体
    pid_expert = AssistantAgent(
        name="pid_expert",
        model_client=model_client,
        system_message=f"""你是PID整定专家。你的任务是计算和转换PID参数。

当轮到你发言时：
1. 查看对话历史中tool_fit_fopdt的返回结果，找到K、T、L参数
2. 调用tool_tune_pid(K=..., T=..., L=..., strategy="{strategy}")计算PID参数
3. 获得Kp、Ki、Kd后，立即调用tool_translate_params(Kp=..., Ki=..., Kd=..., brand="{controller_brand}")转换参数
4. 用一句话总结结果，例如："PID参数 Kp=1.5, Ki=0.01, Kd=5.0"
5. 说"完成"表示你的任务结束""",
        tools=[tool_tune_pid, tool_translate_params],
        model_client_stream=True,
    )

    # 4. 评估智能体
    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message=f"""你是评估专家。你的任务是评估PID整定质量。

当轮到你发言时：
1. 查看对话历史，找到K、T、L（来自tool_fit_fopdt）和Kp、Ki、Kd（来自tool_tune_pid）
2. 调用tool_evaluate_pid(K=..., T=..., L=..., Kp=..., Ki=..., Kd=..., method="{strategy.lower()}")
3. 工具返回后，用一句话总结评估结果
4. 最后说"APPROVE"表示整个流程完成

注意：必须说"APPROVE"才能结束整个协作流程。""",
        tools=[tool_evaluate_pid],
        model_client_stream=True,
    )

    return [data_analyst, system_id_expert, pid_expert, evaluation_expert]


async def run_multi_agent_collaboration(
    csv_path: str,
    loop_name: str,
    controller_brand: str,
    strategy: str,
    llm_config: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """运行多智能体协作 - 使用RoundRobinGroupChat"""

    # 清空全局数据存储
    _shared_data_store.clear()

    # 创建模型客户端
    model_client = create_model_client(
        model_api_key=llm_config["api_key"],
        model_api_url=llm_config["base_url"],
        model=llm_config["model"]
    )

    # 创建4个智能体
    agents = create_pid_agents(
        model_client=model_client,
        csv_path=csv_path,
        controller_brand=controller_brand,
        strategy=strategy
    )

    # 创建终止条件：评估智能体说"APPROVE"
    termination = TextMentionTermination("APPROVE")

    # 创建RoundRobinGroupChat团队
    team = RoundRobinGroupChat(
        participants=agents,
        termination_condition=termination,
    )

    # 构建初始任务消息
    task_message = f"""请为控制回路 {loop_name} 执行PID整定任务。

数据文件: {csv_path}
控制器品牌: {controller_brand}
整定策略: {strategy}

请按以下顺序协作完成：
1. 数据分析智能体：加载和分析数据
2. 系统辨识智能体：拟合FOPDT模型
3. PID专家智能体：计算和转换PID参数
4. 评估智能体：评估整定质量

每个智能体完成任务后，请明确告知下一个智能体继续工作。"""

    # 发送用户消息
    yield {
        "type": "user",
        "content": task_message,
        "file_name": csv_path.split("/")[-1]
    }
    await asyncio.sleep(0.3)

    cancel_token = CancellationToken()
    shared_data = {}
    current_agent = ""

    # Agent名称映射（英文->中文）
    agent_name_map = {
        "data_analyst": "数据分析智能体",
        "system_id_expert": "系统辨识智能体",
        "pid_expert": "PID专家智能体",
        "evaluation_expert": "评估智能体"
    }

    try:
        # 流式处理团队对话
        current_turn_data = None  # 当前智能体回合的数据
        last_agent = None

        async for event in team.run_stream(task=task_message, cancellation_token=cancel_token):
            # 从事件中获取source信息
            event_agent = None
            if hasattr(event, 'source'):
                event_agent = agent_name_map.get(str(event.source), str(event.source))

            if isinstance(event, ToolCallRequestEvent):
                # 工具调用请求 - 如果是新智能体，先发送上一个智能体的数据
                if event_agent and event_agent != last_agent:
                    # 发送上一个智能体的完整数据
                    if current_turn_data is not None:
                        yield current_turn_data
                        await asyncio.sleep(0.3)

                    # 初始化新智能体的数据
                    current_turn_data = {
                        "type": "agent_turn",
                        "agent": event_agent,
                        "tools": [],
                        "response": ""
                    }
                    last_agent = event_agent
                    current_agent = event_agent

                # 添加工具调用
                for tc in event.content:
                    tool_call = {
                        "tool_name": tc.name,
                        "args": tc.arguments,
                        "result": None
                    }
                    if current_turn_data:
                        current_turn_data["tools"].append(tool_call)

            elif isinstance(event, ToolCallExecutionEvent):
                # 工具执行结果
                for res in event.content:
                    try:
                        # 尝试解析结果
                        if isinstance(res.content, dict):
                            result_data = res.content
                        elif isinstance(res.content, str):
                            # 先尝试JSON解析
                            try:
                                result_data = json.loads(res.content)
                            except json.JSONDecodeError:
                                # 如果JSON解析失败，尝试用ast.literal_eval解析Python字典字符串
                                import ast
                                result_data = ast.literal_eval(res.content)
                        else:
                            # 如果不是字符串也不是字典，直接转为字符串
                            result_data = {"result": str(res.content)}

                        # 保存到shared_data
                        if isinstance(result_data, dict):
                            shared_data.update(result_data)

                        # 创建前端显示用的精简版本（不包含大数组）
                        display_result = {}
                        if isinstance(result_data, dict):
                            for key, value in result_data.items():
                                if key in ['mv', 'pv'] and isinstance(value, list):
                                    # 大数组只显示长度
                                    display_result[key] = f"[数组长度: {len(value)}]"
                                else:
                                    display_result[key] = value
                        else:
                            display_result = {"result": str(result_data)}

                        # 将结果添加到当前工具调用
                        if current_turn_data and current_turn_data["tools"]:
                            current_turn_data["tools"][-1]["result"] = display_result

                    except Exception as e:
                        # 如果解析失败，显示原始内容
                        error_result = {"raw_content": str(res.content)[:500], "parse_error": str(e)}
                        if current_turn_data and current_turn_data["tools"]:
                            current_turn_data["tools"][-1]["result"] = error_result

            elif isinstance(event, ModelClientStreamingChunkEvent):
                # 流式文本输出 - 跳过，避免与TextMessage重复
                # （前端会在TextMessage中看到完整内容）
                pass

            elif isinstance(event, ToolCallSummaryMessage):
                # 工具调用摘要 - 跳过，避免重复（TextMessage会包含完整内容）
                pass

            elif isinstance(event, TextMessage):
                # 完整消息 - agent的完整回复
                if event.content and hasattr(event, 'source') and event.source != "user":
                    if current_turn_data:
                        current_turn_data["response"] = event.content

            elif isinstance(event, TaskResult):
                # 任务结束 - 发送最后一个智能体的数据
                if current_turn_data is not None:
                    yield current_turn_data
                    await asyncio.sleep(0.3)
                break

            else:
                # 其他事件类型 - 输出以便调试
                event_type = type(event).__name__
                if hasattr(event, 'content') and event.content:
                    content = str(event.content)[:200]
                    yield {
                        "type": "thought",
                        "agent": current_agent or "系统",
                        "content": f"[{event_type}] {content}"
                    }
                    await asyncio.sleep(0.1)

        # 发送最终结果（构建前端期望的结构化格式）
        final_result = {
            "dataAnalysis": {
                "dataPoints": shared_data.get("data_points", 0),
                "stepEvents": 1,
                "currentIAE": 0.0
            },
            "model": {
                "K": shared_data.get("K", 0.0),
                "T": shared_data.get("T", 0.0),
                "L": shared_data.get("L", 0.0),
                "confidence": shared_data.get("confidence", 0.0)
            },
            "pidParams": {
                "Kp": shared_data.get("Kp", 0.0),
                "Ki": shared_data.get("Ki", 0.0),
                "Kd": shared_data.get("Kd", 0.0)
            }
        }

        # 添加translated参数（如果有）
        translated_keys = [k for k in shared_data.keys() if k not in ['data_points', 'mv_range', 'pv_range', 'status', 'K', 'T', 'L', 'confidence', 'Kp', 'Ki', 'Kd', 'performance_score', 'method_confidence', 'final_rating', 'passed']]
        if translated_keys:
            final_result["translated"] = {k: shared_data[k] for k in translated_keys}

        # 添加evaluation结果（如果有）
        if "final_rating" in shared_data:
            final_result["evaluation"] = {
                "performance_score": shared_data.get("performance_score", 0.0),
                "method_confidence": shared_data.get("method_confidence", 0.0),
                "final_rating": shared_data.get("final_rating", 0.0),
                "strategy_used": strategy,
                "passed": shared_data.get("passed", False)
            }

        yield {
            "type": "result",
            "data": final_result
        }

        yield {
            "type": "done",
            "status": "succeeded"
        }

    except asyncio.CancelledError:
        yield {
            "type": "error",
            "message": "任务被取消"
        }
    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "message": f"多智能体协作失败: {str(e)}\n{traceback.format_exc()}"
        }
