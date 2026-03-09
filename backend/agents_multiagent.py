# Multi-Agent PID Tuning System using AutoGen RoundRobinGroupChat
from __future__ import annotations

import asyncio
import contextlib
import os
import json
import sys
from typing import Any, AsyncGenerator, Dict, List

import httpx
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

from skills.data_analysis_skills import (
    DEFAULT_HISTORY_END_TIME,
    DEFAULT_HISTORY_START_TIME,
    DEFAULT_LOOP_URI,
    fetch_history_data_csv,
    prepare_pid_dataset,
)
from skills.system_id_skills import fit_fopdt_model, calculate_model_confidence
from skills.pid_tuning_skills import apply_tuning_rules, controller_logic_translator
from skills.rating import ModelRating


def create_model_client(*, model_api_key: str, model_api_url: str, model: str) -> OpenAIChatCompletionClient:
    """创建OpenAI兼容的模型客户端（千问）"""
    return OpenAIChatCompletionClient(
        api_key=model_api_key,
        base_url=model_api_url,
        http_client=httpx.AsyncClient(
            timeout=60.0,
            trust_env=False,
            http2=False,
            headers={
                "Connection": "close",
                "Accept-Encoding": "identity",
            },
            transport=httpx.AsyncHTTPTransport(retries=0),
        ),
        model=model,
        temperature=0.3,
        max_tokens=2000,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )


# ===== 全局数据存储 =====
_shared_data_store = {}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value

# ===== 工具函数定义 =====

async def tool_fetch_history_data(
    loop_uri: str = DEFAULT_LOOP_URI,
    start_time: str = DEFAULT_HISTORY_START_TIME,
    end_time: str = DEFAULT_HISTORY_END_TIME,
    data_type: str = "interpolated",
) -> Dict[str, Any]:
    """调用外部系统获取历史数据并保存为本地 CSV"""
    result = await asyncio.to_thread(
        fetch_history_data_csv,
        loop_uri=loop_uri,
        start_time=start_time or None,
        end_time=end_time or None,
        data_type=data_type,
    )
    _shared_data_store["history_csv_path"] = result["csv_path"]
    return result

async def tool_load_data(csv_path: str) -> Dict[str, Any]:
    """加载并预处理 PID 历史数据。"""
    prepared = await asyncio.to_thread(prepare_pid_dataset, csv_path)
    cleaned_df = prepared["cleaned_df"]
    window_df = prepared["window_df"]
    dt = float(prepared["dt"])
    step_events = prepared["step_events"]
    selected_event = prepared["selected_event"]
    quality_metrics = prepared["quality_metrics"] or {}

    mv = window_df["MV"].to_numpy(dtype=float)
    pv = window_df["PV"].to_numpy(dtype=float)

    _shared_data_store["csv_path"] = csv_path
    _shared_data_store["cleaned_df"] = cleaned_df
    _shared_data_store["window_df"] = window_df
    _shared_data_store["mv"] = mv
    _shared_data_store["pv"] = pv
    _shared_data_store["dt"] = dt
    _shared_data_store["step_events"] = step_events
    _shared_data_store["selected_event"] = selected_event
    _shared_data_store["quality_metrics"] = quality_metrics

    selected_window = {
        "rows": int(len(window_df)),
        "start_index": int(selected_event["start_idx"]) if selected_event else 0,
        "end_index": int(selected_event["end_idx"]) if selected_event else int(len(window_df)),
        "event_type": str(selected_event.get("type", "full_range")) if selected_event else "full_range",
    }

    return _to_jsonable({
        "data_points": int(len(cleaned_df)),
        "window_points": int(len(window_df)),
        "sampling_time": dt,
        "mv_range": [float(cleaned_df["MV"].min()), float(cleaned_df["MV"].max())],
        "pv_range": [float(cleaned_df["PV"].min()), float(cleaned_df["PV"].max())],
        "available_columns": [str(col) for col in cleaned_df.columns.tolist()],
        "step_events": step_events,
        "selected_window": selected_window,
        "quality_metrics": quality_metrics,
        "status": "数据已完成清洗、降噪和辨识窗口选择",
    })


async def tool_fit_fopdt(dt: float = 1.0) -> Dict[str, Any]:
    """基于预处理后的 MV/PV 窗口拟合 FOPDT 模型。"""
    if "mv" not in _shared_data_store or "pv" not in _shared_data_store:
        raise ValueError("请先调用tool_load_data加载数据")

    mv_array = _shared_data_store["mv"]
    pv_array = _shared_data_store["pv"]
    actual_dt = float(_shared_data_store.get("dt", dt))

    model_params = await asyncio.to_thread(fit_fopdt_model, mv_array, pv_array, actual_dt)
    confidence = calculate_model_confidence(model_params["residue"])

    _shared_data_store["K"] = float(model_params["K"])
    _shared_data_store["T"] = float(model_params["T"])
    _shared_data_store["L"] = float(model_params["L"])
    _shared_data_store["model_confidence"] = confidence

    return _to_jsonable({
        "K": float(model_params["K"]),
        "T": float(model_params["T"]),
        "L": float(model_params["L"]),
        "dt": actual_dt,
        "residue": float(model_params["residue"]),
        "success": bool(model_params["success"]),
        "confidence": float(confidence["confidence"]),
        "confidence_quality": confidence["quality"],
        "confidence_recommendation": confidence["recommendation"],
    })


async def tool_tune_pid(K: float, T: float, L: float, strategy: str) -> Dict[str, Any]:
    """应用 PID 整定规则。"""
    pid_params = apply_tuning_rules(K, T, L, strategy)
    _shared_data_store["pid_params"] = pid_params
    return {
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"]),
        "Ti": float(pid_params["Ti"]),
        "Td": float(pid_params["Td"]),
        "strategy": str(pid_params["strategy"]),
        "description": str(pid_params["description"]),
    }


async def tool_translate_params(Kp: float, Ki: float, Kd: float, brand: str) -> Dict[str, Any]:
    """转换为控制器参数"""
    pid_params = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
    translated = controller_logic_translator(pid_params, brand)
    return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in translated.items()}


async def tool_evaluate_pid(K: float, T: float, L: float, Kp: float, Ki: float, Kd: float, method: str) -> Dict[str, Any]:
    """评估 PID 整定结果。"""
    model_confidence = _shared_data_store.get("model_confidence", {})
    method_confidence = float(model_confidence.get("confidence", 0.6))
    eval_result = ModelRating.evaluate(
        model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
        pid_params={"Kp": Kp, "Ki": Ki, "Kd": Kd},
        method=method.lower(),
        method_confidence=method_confidence,
        method_confidence_details={
            "source": "model_identification_residue",
            "quality": model_confidence.get("quality", "unknown"),
            "recommendation": model_confidence.get("recommendation", ""),
        },
        dt=float(_shared_data_store.get("dt", 1.0)),
    )
    _shared_data_store["evaluation_result"] = eval_result
    return _to_jsonable({
        "performance_score": float(eval_result["performance_score"]),
        "method_confidence": float(eval_result["method_confidence"]),
        "final_rating": float(eval_result["final_rating"]),
        "passed": bool(eval_result["final_rating"] >= 6.0),
        "performance_details": eval_result["performance_details"],
        "final_details": eval_result["final_details"],
        "simulation": eval_result["simulation"],
    })


def create_pid_agents(
    *,
    model_client: OpenAIChatCompletionClient,
    csv_path: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    controller_brand: str,
    strategy: str
) -> List[AssistantAgent]:
    """创建 4 个专业智能体。"""

    if csv_path:
        data_analyst_tools = [tool_load_data]
        data_analyst_prompt = f"""你是数据分析专家。

本次任务已经上传本地 CSV，路径是 "{csv_path}"。
当轮到你发言时，只允许调用 tool_load_data(csv_path="{csv_path}")。
不要调用 tool_fetch_history_data。
读取工具返回的数据摘要后，用一句话总结数据质量、采样时间和选中的辨识窗口，并以“完成”结束。"""
    else:
        data_analyst_tools = [tool_fetch_history_data, tool_load_data]
        data_analyst_prompt = f"""你是数据分析专家。

本次任务未上传本地 CSV。你必须先调用：
tool_fetch_history_data(loop_uri="{loop_uri}", start_time="{start_time}", end_time="{end_time}", data_type="{data_type}")
从工具结果中读取 csv_path 后，再调用 tool_load_data(csv_path=该路径)。
不要跳过这两个步骤，也不要混用其他数据来源。
最后用一句话总结获取的数据量、采样时间、检测到的阶跃事件和选中的辨识窗口，并以“完成”结束。"""

    data_analyst = AssistantAgent(
        name="data_analyst",
        model_client=model_client,
        system_message=data_analyst_prompt,
        tools=data_analyst_tools,
        model_client_stream=False,
        max_tool_iterations=2,
    )

    system_id_expert = AssistantAgent(
        name="system_id_expert",
        model_client=model_client,
        system_message="""你是系统辨识专家。

当轮到你发言时，立即调用 tool_fit_fopdt(dt=1.0)。
读取工具返回的 K、T、L、residue、confidence 后，用一句话总结模型质量，并以“完成”结束。""",
        tools=[tool_fit_fopdt],
        model_client_stream=False,
    )

    pid_expert = AssistantAgent(
        name="pid_expert",
        model_client=model_client,
        system_message=f"""你是 PID 整定专家。

当轮到你发言时：
1. 从对话历史读取 tool_fit_fopdt 返回的 K、T、L。
2. 调用 tool_tune_pid(K=..., T=..., L=..., strategy="{strategy}")。
3. 拿到 Kp、Ki、Kd 后，立即调用 tool_translate_params(Kp=..., Ki=..., Kd=..., brand="{controller_brand}")。
4. 用一句话总结整定策略、标准 PID 参数和控制器参数格式，并以“完成”结束。""",
        tools=[tool_tune_pid, tool_translate_params],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message=f"""你是评估专家。

当轮到你发言时：
1. 从对话历史读取 K、T、L、Kp、Ki、Kd。
2. 调用 tool_evaluate_pid(K=..., T=..., L=..., Kp=..., Ki=..., Kd=..., method="{strategy.lower()}")。
3. 用一句话总结性能评分、方法置信度和最终评分。
4. 最后明确输出 “APPROVE”。""",
        tools=[tool_evaluate_pid],
        model_client_stream=False,
    )

    return [data_analyst, system_id_expert, pid_expert, evaluation_expert]


async def run_multi_agent_collaboration(
    csv_path: str,
    loop_name: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
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
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
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
历史数据回路URI: {loop_uri}
历史数据开始时间: {start_time or "默认最近24小时"}
历史数据结束时间: {end_time or "当前时间"}
历史数据类型: {data_type}
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
        "file_name": csv_path.split("/")[-1] if csv_path else loop_uri.split("/")[-1]
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
        quality_metrics = shared_data.get("quality_metrics") or {}
        step_events = shared_data.get("step_events") or []
        final_result = {
            "dataAnalysis": {
                "dataPoints": shared_data.get("data_points", 0),
                "windowPoints": shared_data.get("window_points", 0),
                "stepEvents": len(step_events),
                "stepEventDetails": step_events,
                "currentIAE": quality_metrics.get("IAE", 0.0),
                "samplingTime": shared_data.get("sampling_time", 1.0),
                "selectedWindow": shared_data.get("selected_window", {}),
                "qualityMetrics": quality_metrics,
            },
            "model": {
                "K": shared_data.get("K", 0.0),
                "T": shared_data.get("T", 0.0),
                "L": shared_data.get("L", 0.0),
                "confidence": shared_data.get("confidence", 0.0),
                "residue": shared_data.get("residue", 0.0),
                "confidenceQuality": shared_data.get("confidence_quality", ""),
            },
            "pidParams": {
                "Kp": shared_data.get("Kp", 0.0),
                "Ki": shared_data.get("Ki", 0.0),
                "Kd": shared_data.get("Kd", 0.0),
                "Ti": shared_data.get("Ti", 0.0),
                "Td": shared_data.get("Td", 0.0),
                "strategy": shared_data.get("strategy", strategy),
                "description": shared_data.get("description", ""),
            },
        }

        # 添加translated参数（如果有）
        translated_keys = [
            k for k in shared_data.keys()
            if k not in [
                "data_points", "window_points", "sampling_time", "mv_range", "pv_range", "available_columns",
                "selected_window", "step_events", "quality_metrics", "status",
                "K", "T", "L", "dt", "residue", "success", "confidence", "confidence_quality",
                "confidence_recommendation", "Kp", "Ki", "Kd", "Ti", "Td", "strategy", "description",
                "performance_score", "method_confidence", "final_rating", "passed",
                "performance_details", "final_details", "simulation",
            ]
        ]
        if translated_keys:
            final_result["translated"] = {k: shared_data[k] for k in translated_keys}

        # 添加evaluation结果（如果有）
        if "final_rating" in shared_data:
            final_result["evaluation"] = {
                "performance_score": shared_data.get("performance_score", 0.0),
                "method_confidence": shared_data.get("method_confidence", 0.0),
                "final_rating": shared_data.get("final_rating", 0.0),
                "strategy_used": strategy,
                "passed": shared_data.get("passed", False),
                "performance_details": shared_data.get("performance_details", {}),
                "final_details": shared_data.get("final_details", {}),
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


# ============ FastAPI Web服务 ============
if __name__ == "__main__":
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import tempfile
    import uvicorn

    app = FastAPI()

    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # LLM配置
    from dotenv import load_dotenv
    load_dotenv()

    LLM_CONFIG = {
        "api_key": os.getenv("MODEL_API_KEY"),
        "base_url": os.getenv("MODEL_API_URL"),
        "model": os.getenv("MODEL", "qwen-plus")
    }

    @app.post("/api/tune_stream")
    async def tune_stream(
        file: UploadFile = File(None),
        loop_name: str = Form(...),
        loop_uri: str = Form(DEFAULT_LOOP_URI),
        start_time: str = Form(DEFAULT_HISTORY_START_TIME),
        end_time: str = Form(DEFAULT_HISTORY_END_TIME),
        data_type: str = Form("interpolated"),
        controller_brand: str = Form(...),
        strategy: str = Form("IMC")
    ):
        """流式PID整定接口 - 使用AutoGen多智能体"""
        
        # 保存上传的文件
        csv_path = ""
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                csv_path = tmp_file.name
        
        async def event_generator():
            try:
                # 调用多智能体协作
                async for event in run_multi_agent_collaboration(
                    csv_path=csv_path,
                    loop_name=loop_name,
                    loop_uri=loop_uri,
                    start_time=start_time,
                    end_time=end_time,
                    data_type=data_type,
                    controller_brand=controller_brand,
                    strategy=strategy,
                    llm_config=LLM_CONFIG
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                import traceback
                error_msg = {"type": "error", "message": f"{str(e)}\n{traceback.format_exc()}"}
                yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            finally:
                # 清理临时文件
                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    print("Starting PID Tuning Multi-Agent System...")
    print(f"API endpoint: http://0.0.0.0:3443/api/tune_stream")
    uvicorn.run(app, host="0.0.0.0", port=3443)
