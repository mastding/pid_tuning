from __future__ import annotations

import ast
import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Dict

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken

from orchestration.constants import DISPLAY_AGENT_NAMES
from services.system_config_service import is_knowledge_expert_enabled


def _build_task_message(
    *,
    csv_path: str,
    loop_name: str,
    loop_type: str,
    plant_type: str,
    scenario: str,
    control_object: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    window: int,
) -> str:
    data_source = "上传CSV" if csv_path else "获取历史数据"
    display_start = start_time or "使用默认开始时间"
    display_end = end_time or "使用默认结束时间"
    knowledge_line = "3. 本体知识智能体：检索专家规则与约束\n" if is_knowledge_expert_enabled() else ""
    step4 = "4. PID专家智能体：计算 PID 参数\n" if is_knowledge_expert_enabled() else "3. PID专家智能体：计算 PID 参数\n"
    step5 = "5. 评估智能体：评估整定质量\n\n" if is_knowledge_expert_enabled() else "4. 评估智能体：评估整定质量\n\n"

    return (
        f"请为控制回路 {loop_name} 整定 PID 参数。\n"
        f"数据来源: {data_source}\n"
        f"loop_uri: {loop_uri}\n"
        f"start_time: {display_start}\n"
        f"end_time: {display_end}\n"
        f"window: {window}s\n"
        f"回路类型: {loop_type}\n"
        f"装置类型: {plant_type or '未指定'}\n"
        f"工况: {scenario or '未指定'}\n"
        f"控制对象: {control_object or '未指定'}\n\n"
        "请按以下顺序协作完成：\n"
        "1. 数据分析智能体：加载和分析数据\n"
        "2. 系统辨识智能体：辨识过程模型\n"
        f"{knowledge_line}"
        f"{step4}"
        f"{step5}"
        "每个智能体完成任务后，请明确交接给下一位智能体。"
    )


def _parse_tool_result(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            try:
                return ast.literal_eval(content)
            except Exception:
                return {"result": content}
    return {"result": str(content)}


def _build_display_result(result_data: Dict[str, Any], *, current_tool_name: str) -> Dict[str, Any]:
    if current_tool_name == "tool_evaluate_pid" and result_data.get("initial_assessment"):
        initial_assessment = result_data.get("initial_assessment") or {}
        initial_eval_result = initial_assessment.get("evaluation_result") or {}
        return {
            "passed": initial_assessment.get("passed", False),
            "pass_threshold": initial_assessment.get("pass_threshold", result_data.get("pass_threshold", 7.0)),
            "performance_score": initial_eval_result.get("performance_score", 0.0),
            "method_confidence": initial_eval_result.get("method_confidence", 0.0),
            "final_rating": initial_eval_result.get("final_rating", 0.0),
            "failure_reason": initial_assessment.get("failure_reason", ""),
            "feedback_target": initial_assessment.get("feedback_target", ""),
            "feedback_action": initial_assessment.get("feedback_action", ""),
            "evaluated_pid": initial_assessment.get("evaluated_pid", {}),
        }

    display_result: Dict[str, Any] = {}
    for key, value in result_data.items():
        if key in {"mv", "pv"} and isinstance(value, list):
            display_result[key] = f"[鏁扮粍闀垮害: {len(value)}]"
        else:
            display_result[key] = value
    return display_result


def _build_tuning_advice(final_result: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = final_result.get("evaluation") or {}
    pid_params = final_result.get("pidParams") or {}
    model = final_result.get("model") or {}

    passed = bool(evaluation.get("passed", False))
    final_rating = float(evaluation.get("final_rating", 0.0) or 0.0)
    performance_score = float(evaluation.get("performance_score", 0.0) or 0.0)
    method_confidence = float(evaluation.get("method_confidence", 0.0) or 0.0)
    model_type = str(model.get("modelType", "FOPDT") or "FOPDT")
    strategy_used = str(pid_params.get("strategyUsed") or pid_params.get("strategy") or "")
    failure_reason = str(evaluation.get("failure_reason", "") or "")
    feedback_action = str(evaluation.get("feedback_action", "") or "")

    if passed and final_rating >= 8.5:
        level = "recommended"
        summary = "建议采用当前整定参数，可直接作为优先投用方案。"
        actions = [
            "优先在低风险工况下试投用。",
            "投用后观察 1 至 2 个完整调节周期，确认超调和振荡可接受。",
            "若现场噪声偏大，可适度减小 Kp。", 
        ]
        risks = [f"当前采用 {model_type} 模型与 {strategy_used or '自动选择策略'} 进行整定，建议继续观察现场工况变化。"]
    elif passed:
        level = "cautious"
        summary = "建议谨慎采用当前整定参数，先在受控工况下试运行。"
        actions = [
            "先在风险较低工况下投用。",
            "重点关注振荡次数、稳态误差与阀位变化。",
            "如现场波动超出预期，可回退到原参数。", 
        ]
        risks = [
            f"综合评分 {final_rating:.2f}，闭环表现可用但仍需现场确认。",
            f"方法置信度 {method_confidence:.2f}，建议结合现场经验复核。",
        ]
    else:
        level = "not_recommended"
        summary = "当前整定参数不建议直接投用，应先根据评估建议继续优化。"
        actions = [feedback_action] if feedback_action else ["建议先继续优化 PID 参数后再重新评估。"]
        risks = [failure_reason] if failure_reason else ["当前综合评分未达标，直接投用风险较高。"]

    return {
        "summary": summary,
        "recommendation_level": level,
        "actions": actions,
        "risks": risks,
        "rollback_advice": "如投用后振荡、超调或稳态误差明显恶化，建议回退到原 PID 参数。",
        "operator_note": f"当前性能评分 {performance_score:.2f}，综合评分 {final_rating:.2f}，建议结合现场工况审慎应用。",
    }


def _classify_workflow_exception(exc: Exception) -> Dict[str, Any]:
    error_text = str(exc or "").strip()
    lowered = error_text.lower()

    if "apitimeouterror" in lowered or "request timed out" in lowered or "connecttimeout" in lowered or "timeout" in lowered:
        return {
            "error_code": 10024,
            "error_type": "upstream_model_timeout",
            "message": "上游模型服务响应超时，请稍后重试或在系统配置中切换可用模型。",
            "detail": error_text,
        }
    if "bad gateway" in lowered or "502" in lowered:
        return {
            "error_code": 10025,
            "error_type": "upstream_model_bad_gateway",
            "message": "上游模型网关返回 502，当前模型服务暂时不可用，请稍后重试或切换模型。",
            "detail": error_text,
        }
    if "service unavailable" in lowered or "503" in lowered:
        return {
            "error_code": 10026,
            "error_type": "upstream_model_unavailable",
            "message": "上游模型服务当前不可用（503），请稍后重试或切换模型。",
            "detail": error_text,
        }
    if "connection error" in lowered or "apiconnectionerror" in lowered:
        return {
            "error_code": 10022,
            "error_type": "upstream_model_connection",
            "message": "上游模型网关连接异常，请检查模型服务地址和网络连通性。",
            "detail": error_text,
        }

    return {
        "error_code": 10020,
        "error_type": "workflow_execution_error",
        "message": f"多智能体协作失败：{error_text or '未知异常'}",
        "detail": error_text,
    }


async def run_multi_agent_collaboration(
    *,
    csv_path: str,
    loop_name: str,
    loop_type: str,
    plant_type: str,
    scenario: str,
    control_object: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    window: int,
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    llm_config: Dict[str, Any],
    shared_data_store: Dict[str, Any],
    create_model_client: Callable[..., Any],
    create_pid_agents: Callable[..., Any],
    finalize_agent_turn: Callable[[Dict[str, Any] | None], Dict[str, Any] | None],
    build_feedback_turns: Callable[[Dict[str, Any]], list[Dict[str, Any]]],
    build_experience_record: Callable[..., Dict[str, Any]],
    persist_experience_record: Callable[[Dict[str, Any]], str],
    register_experience_reuse: Callable[..., Dict[str, Any]] | None,
    to_jsonable: Callable[[Any], Any],
) -> AsyncGenerator[Dict[str, Any], None]:
    shared_data_store.clear()
    shared_data_store["loop_name"] = loop_name
    shared_data_store["loop_type"] = loop_type
    shared_data_store["plant_type"] = plant_type
    shared_data_store["scenario"] = scenario
    shared_data_store["control_object"] = control_object
    if selected_loop_prefix is not None:
        shared_data_store["selected_loop_prefix"] = selected_loop_prefix
    if selected_window_index is not None:
        shared_data_store["selected_window_index"] = selected_window_index

    model_client = create_model_client(
        model_api_key=llm_config["api_key"],
        model_api_url=llm_config["base_url"],
        model=llm_config["model"],
    )

    agents = create_pid_agents(
        model_client=model_client,
        csv_path=csv_path,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        window=window,
        loop_type=loop_type,
    )

    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(12)
    team = RoundRobinGroupChat(participants=agents, termination_condition=termination)
    task_message = _build_task_message(
        csv_path=csv_path,
        loop_name=loop_name,
        loop_type=loop_type,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        window=window,
    )

    yield {
        "type": "user",
        "content": task_message,
        "file_name": csv_path.split("/")[-1] if csv_path else loop_uri.split("/")[-1],
    }
    await asyncio.sleep(0.3)

    cancel_token = CancellationToken()
    shared_data: Dict[str, Any] = {}
    current_agent = ""
    current_turn_data = None
    last_agent = None
    next_event_task: asyncio.Task | None = None

    try:
        stream = team.run_stream(task=task_message, cancellation_token=cancel_token)
        idle_seconds = 0.0
        heartbeat_interval_seconds = 15.0
        max_idle_seconds = 180.0
        print(
            f"[workflow_runner] start loop={loop_name} loop_type={loop_type} "
            f"participants={[getattr(agent, 'name', '') for agent in agents]}"
        )
        next_event_task = asyncio.create_task(stream.__anext__())

        while True:
            try:
                event = await asyncio.wait_for(asyncio.shield(next_event_task), timeout=heartbeat_interval_seconds)
            except asyncio.TimeoutError:
                idle_seconds += heartbeat_interval_seconds
                print(
                    f"[workflow_runner] heartbeat_wait loop={loop_name} idle_seconds={int(idle_seconds)} "
                    f"current_agent={current_agent or last_agent or ''}"
                )
                yield {
                    "type": "thought",
                    "agent": current_agent or (last_agent or "系统"),
                    "content": f"[heartbeat] 仍在等待智能体响应…已等待 {int(idle_seconds)} 秒",
                }
                await asyncio.sleep(0.1)
                if idle_seconds >= max_idle_seconds:
                    cancel_token.cancel()
                    try:
                        if next_event_task is not None:
                            next_event_task.cancel()
                    except Exception:
                        pass
                    yield {
                        "type": "error",
                        "message": "任务长时间无输出，可能上游模型调用超时或阻塞。",
                        "error_type": "TIMEOUT",
                        "error_code": "STREAM_IDLE_TIMEOUT",
                        "error_detail": f"agent={current_agent or last_agent or ''}, idle_seconds={int(idle_seconds)}",
                    }
                    return
                continue
            except StopAsyncIteration:
                print(f"[workflow_runner] stop_async_iteration loop={loop_name}")
                break

            idle_seconds = 0.0
            next_event_task = asyncio.create_task(stream.__anext__())

            event_agent = None
            if hasattr(event, "source"):
                event_agent = DISPLAY_AGENT_NAMES.get(str(event.source), str(event.source))
            print(
                f"[workflow_runner] event loop={loop_name} event_type={type(event).__name__} "
                f"event_agent={event_agent or ''}"
            )

            if isinstance(event, ToolCallRequestEvent):
                if event_agent and event_agent != last_agent:
                    if current_turn_data is not None:
                        finalized = finalize_agent_turn(current_turn_data)
                        if finalized is not None:
                            yield finalized
                            await asyncio.sleep(0.3)

                    current_turn_data = {
                        "type": "agent_turn",
                        "agent": event_agent,
                        "tools": [],
                        "response": "",
                    }
                    last_agent = event_agent
                    current_agent = event_agent

                for tool_call in event.content:
                    if current_turn_data is not None:
                        current_turn_data["tools"].append(
                            {
                                "tool_name": tool_call.name,
                                "args": tool_call.arguments,
                                "result": None,
                            }
                        )
                continue

            if isinstance(event, ToolCallExecutionEvent):
                for tool_result in event.content:
                    try:
                        result_data = _parse_tool_result(tool_result.content)
                        if isinstance(result_data, dict):
                            shared_data.update(result_data)

                        current_tool_name = ""
                        if current_turn_data and current_turn_data["tools"]:
                            current_tool_name = current_turn_data["tools"][-1].get("tool_name", "")

                        display_result = _build_display_result(
                            result_data if isinstance(result_data, dict) else {"result": str(result_data)},
                            current_tool_name=current_tool_name,
                        )
                        if current_turn_data and current_turn_data["tools"]:
                            current_turn_data["tools"][-1]["result"] = display_result

                        if isinstance(result_data, dict) and current_tool_name == "tool_load_data":
                            candidate_windows = result_data.get("candidate_windows") or []
                            usable_windows = [
                                w for w in candidate_windows if isinstance(w, dict) and w.get("window_usable_for_id") is True
                            ]
                            data_points = int(result_data.get("data_points") or 0)
                            sampling_time = result_data.get("sampling_time")
                            step_event_count = len(result_data.get("step_events") or [])
                            candidate_count = len(candidate_windows)
                            usable_count = len(usable_windows)

                            if current_turn_data is not None:
                                current_turn_data["response"] = (
                                    f"数据分析完成：已加载{data_points}个数据点，采样间隔{sampling_time}秒，"
                                    f"检测到{step_event_count}个阶跃事件，构造{candidate_count}个候选窗口，"
                                    f"其中{usable_count}个可用于辨识；已将候选窗口传递给系统辨识智能体。"
                                )
                            if not usable_windows:
                                reason_counter: Dict[str, int] = {}
                                for w in candidate_windows:
                                    if not isinstance(w, dict):
                                        continue
                                    reasons = w.get("window_quality_reasons") or []
                                    if not isinstance(reasons, list):
                                        continue
                                    for r in reasons:
                                        if not r:
                                            continue
                                        key = str(r)
                                        reason_counter[key] = int(reason_counter.get(key, 0)) + 1

                                top_reasons = sorted(reason_counter.items(), key=lambda kv: kv[1], reverse=True)[:3]
                                top_reason_text = "、".join([f"{name}({count})" for name, count in top_reasons]) if top_reasons else ""

                                response_text = (
                                    f"数据分析完成。数据包含{data_points}个采样点，采样间隔{sampling_time}秒，"
                                    f"共识别出{candidate_count}个候选窗口（阶跃事件{step_event_count}个）。"
                                    "未发现可用于系统辨识的窗口，已终止后续流程。"
                                )
                                if top_reason_text:
                                    response_text += f"主要原因：{top_reason_text}。"

                                if current_turn_data is not None:
                                    current_turn_data["response"] = response_text
                                    finalized = finalize_agent_turn(current_turn_data)
                                    if finalized is not None:
                                        yield finalized
                                        await asyncio.sleep(0.2)

                                yield {
                                    "type": "result",
                                    "data": {
                                        "termination": {
                                            "terminated": True,
                                            "stage": "data_analysis",
                                            "reason_code": "NO_USABLE_WINDOWS",
                                            "message": response_text,
                                        },
                                        "dataAnalysis": {
                                            "dataPoints": data_points,
                                            "stepEvents": step_event_count,
                                            "stepEventDetails": result_data.get("step_events") or [],
                                            "samplingTime": sampling_time,
                                            "candidateWindows": candidate_windows,
                                            "qualityMetrics": result_data.get("quality_metrics") or {},
                                        },
                                        "model": {
                                            "modelType": "",
                                            "confidence": 0.0,
                                            "attempts": [],
                                            "windowOverview": result_data.get("window_overview") or {"points": []},
                                            "fitPreview": {"points": []},
                                        },
                                        "pidParams": {},
                                    },
                                }
                                cancel_token.cancel()
                                return

                        if isinstance(result_data, dict) and current_tool_name == "tool_fit_fopdt":
                            confidence_value = 1.0
                            try:
                                confidence_value = float(result_data.get("confidence", 1.0))
                            except Exception:
                                confidence_value = 1.0

                            min_confidence = 0.35
                            if confidence_value < min_confidence:
                                response_text = (
                                    f"系统辨识完成，但模型置信度{confidence_value:.2f}低于阈值{min_confidence:.2f}，"
                                    "为避免不可靠的整定结果，已终止后续整定与评估流程。"
                                )
                                if current_turn_data is not None:
                                    current_turn_data["response"] = response_text
                                    finalized = finalize_agent_turn(current_turn_data)
                                    if finalized is not None:
                                        yield finalized
                                        await asyncio.sleep(0.2)

                                quality_metrics = shared_data.get("quality_metrics") or {}
                                yield {
                                    "type": "result",
                                    "data": {
                                        "termination": {
                                            "terminated": True,
                                            "stage": "system_identification",
                                            "reason_code": "LOW_CONFIDENCE",
                                            "message": response_text,
                                            "min_confidence": min_confidence,
                                            "confidence": confidence_value,
                                        },
                                        "dataAnalysis": {
                                            "dataPoints": shared_data.get("data_points", 0),
                                            "stepEvents": len(shared_data.get("step_events") or []),
                                            "stepEventDetails": shared_data.get("step_events") or [],
                                            "samplingTime": shared_data.get("sampling_time", 1.0),
                                            "historyRange": {
                                                "startTime": shared_data.get("start_time", start_time),
                                                "endTime": shared_data.get("end_time", end_time),
                                            },
                                            "qualityMetrics": quality_metrics,
                                        },
                                        "model": {
                                            "modelType": shared_data.get("model_type", result_data.get("model_type", "")),
                                            "selectedModelParams": shared_data.get(
                                                "selected_model_params", result_data.get("selected_model_params", {})
                                            ),
                                            "modelSelectionReason": shared_data.get(
                                                "model_selection_reason", result_data.get("model_selection_reason", "")
                                            ),
                                            "K": shared_data.get("K", result_data.get("K", 0.0)),
                                            "T": shared_data.get("T", result_data.get("T", 0.0)),
                                            "L": shared_data.get("L", result_data.get("L", 0.0)),
                                            "confidence": confidence_value,
                                            "normalizedRmse": shared_data.get(
                                                "normalized_rmse", result_data.get("normalized_rmse", shared_data.get("residue", 0.0))
                                            ),
                                            "r2Score": shared_data.get("r2_score", result_data.get("r2_score", 0.0)),
                                            "reasonCodes": result_data.get("reason_codes", shared_data.get("reason_codes", [])),
                                            "nextActions": result_data.get("next_actions", shared_data.get("next_actions", [])),
                                            "selectedWindowSource": result_data.get(
                                                "selected_window_source", shared_data.get("selected_window_source", "")
                                            ),
                                            "attempts": result_data.get("attempts", shared_data.get("attempts", [])),
                                            "fitPreview": result_data.get("fit_preview", shared_data.get("fit_preview", {"points": []})),
                                            "windowOverview": result_data.get(
                                                "window_overview", shared_data.get("window_overview", {"points": []})
                                            ),
                                            "windowBenchmark": result_data.get(
                                                "window_benchmark", shared_data.get("window_benchmark", {})
                                            ),
                                        },
                                        "pidParams": {},
                                    },
                                }
                                cancel_token.cancel()
                                return

                        if isinstance(result_data, dict) and current_tool_name == "tool_evaluate_pid":
                            if current_turn_data is not None:
                                finalized = finalize_agent_turn(current_turn_data)
                                if finalized is not None:
                                    yield finalized
                                    await asyncio.sleep(0.2)

                            quality_metrics = shared_data.get("quality_metrics") or {}
                            for feedback_turn in build_feedback_turns(shared_data):
                                yield feedback_turn
                                await asyncio.sleep(0.2)

                            effective_pid_params = shared_data_store.get("selected_pid_params") or {}
                            final_result: Dict[str, Any] = {
                                "dataAnalysis": {
                                    "dataPoints": shared_data.get("data_points", 0),
                                    "windowPoints": shared_data.get("window_points", 0),
                                    "stepEvents": len(shared_data.get("step_events") or []),
                                    "stepEventDetails": shared_data.get("step_events") or [],
                                    "currentIAE": quality_metrics.get("IAE", 0.0),
                                    "samplingTime": shared_data.get("sampling_time", 1.0),
                                    "selectedWindow": shared_data.get("selected_window", {}),
                                    "historyRange": {
                                        "startTime": shared_data.get("start_time", start_time),
                                        "endTime": shared_data.get("end_time", end_time),
                                    },
                                    "qualityMetrics": quality_metrics,
                                },
                                "model": {
                                    "modelType": shared_data.get("model_type", "FOPDT"),
                                    "selectedModelParams": shared_data.get("selected_model_params", {}),
                                    "modelSelectionReason": shared_data.get("model_selection_reason", ""),
                                    "K": shared_data.get("K", 0.0),
                                    "T": shared_data.get("T", 0.0),
                                    "L": shared_data.get("L", 0.0),
                                    "confidence": shared_data.get("confidence", 0.0),
                                    "residue": shared_data.get("residue", 0.0),
                                    "normalizedRmse": shared_data.get("normalized_rmse", shared_data.get("residue", 0.0)),
                                    "rawRmse": shared_data.get("raw_rmse", 0.0),
                                    "r2Score": shared_data.get("r2_score", 0.0),
                                    "confidenceQuality": shared_data.get("confidence_quality", ""),
                                    "confidenceRecommendation": shared_data.get("confidence_recommendation", ""),
                                    "reasonCodes": shared_data.get("reason_codes", []),
                                    "nextActions": shared_data.get("next_actions", []),
                                    "selectedWindowSource": shared_data.get("selected_window_source", ""),
                                    "attempts": shared_data.get("attempts", []),
                                    "fitPreview": shared_data.get("fit_preview", {"points": []}),
                                    "windowOverview": shared_data.get("window_overview", {"points": []}),
                                },
                                "pidParams": {
                                    "Kp": effective_pid_params.get("Kp", shared_data.get("Kp", 0.0)),
                                    "Ki": effective_pid_params.get("Ki", shared_data.get("Ki", 0.0)),
                                    "Kd": effective_pid_params.get("Kd", shared_data.get("Kd", 0.0)),
                                    "Ti": effective_pid_params.get("Ti", shared_data.get("Ti", 0.0)),
                                    "Td": effective_pid_params.get("Td", shared_data.get("Td", 0.0)),
                                    "strategy": effective_pid_params.get("strategy", shared_data.get("strategy", "")),
                                    "strategyRequested": shared_data.get("strategy_requested", "AUTO"),
                                    "strategyUsed": shared_data.get("strategy_used", ""),
                                    "loopType": loop_type,
                                    "selectionReason": shared_data.get("selection_reason", ""),
                                    "selectionInputs": shared_data.get("selection_inputs", {}),
                                    "experienceGuidance": shared_data.get("experience_guidance", {}),
                                    "candidateStrategies": shared_data.get("candidate_strategies", []),
                                    "description": effective_pid_params.get("description", shared_data.get("description", "")),
                                },
                                "knowledge": {
                                    "guidance": shared_data.get("expert_knowledge_guidance", {})
                                    or shared_data_store.get("expert_knowledge_guidance", {}),
                                },
                            }

                            if "final_rating" in shared_data:
                                final_result["evaluation"] = {
                                    "performance_score": shared_data.get("performance_score", 0.0),
                                    "method_confidence": shared_data.get("method_confidence", 0.0),
                                    "final_rating": shared_data.get("final_rating", 0.0),
                                    "strategy_used": shared_data.get("strategy_used", ""),
                                    "passed": shared_data.get("passed", False),
                                    "pass_threshold": shared_data.get("pass_threshold", 7.0),
                                    "failure_reason": shared_data.get("failure_reason", ""),
                                    "feedback_target": shared_data.get("feedback_target", ""),
                                    "feedback_target_display": shared_data.get("feedback_target_display", ""),
                                    "feedback_action": shared_data.get("feedback_action", ""),
                                    "initial_assessment": shared_data.get("initial_assessment", {}),
                                    "auto_refine_result": shared_data.get("auto_refine_result", {}),
                                    "model_retry_result": shared_data.get("model_retry_result", {}),
                                    "performance_details": shared_data.get("performance_details", {}),
                                    "final_details": shared_data.get("final_details", {}),
                                    "replay_evaluation": shared_data.get("replay_evaluation", {}),
                                }

                            final_result["tuningAdvice"] = _build_tuning_advice(final_result)

                            experience_record = build_experience_record(
                                loop_name=loop_name,
                                loop_type=loop_type,
                                loop_uri=loop_uri,
                                data_source="csv" if csv_path else "history",
                                start_time=shared_data.get("start_time", start_time),
                                end_time=shared_data.get("end_time", end_time),
                                shared_data=shared_data,
                                final_result=final_result,
                            )
                            experience_id = persist_experience_record(experience_record)
                            referenced_experience_ids = experience_record.get("referenced_experience_ids") or []
                            reuse_summary = {}
                            if register_experience_reuse and referenced_experience_ids:
                                reuse_summary = register_experience_reuse(
                                    referenced_experience_ids,
                                    follow_up_passed=bool(final_result.get("evaluation", {}).get("passed", False)),
                                    follow_up_final_rating=float(final_result.get("evaluation", {}).get("final_rating", 0.0) or 0.0),
                                )
                            final_result["memory"] = {
                                "experienceId": experience_id,
                                "experienceGuidance": shared_data.get("experience_guidance", {}),
                                "referenceReuse": reuse_summary,
                            }

                            yield {"type": "result", "data": final_result}
                            yield {"type": "done", "status": "succeeded"}
                            cancel_token.cancel()
                            return
                    except Exception as exc:
                        error_result = {
                            "raw_content": str(tool_result.content)[:500],
                            "parse_error": str(exc),
                        }
                        if current_turn_data and current_turn_data["tools"]:
                            current_turn_data["tools"][-1]["result"] = error_result
                continue

            if isinstance(event, (ModelClientStreamingChunkEvent, ToolCallSummaryMessage)):
                continue

            if isinstance(event, TextMessage):
                if event.content and getattr(event, "source", None) != "user":
                    if event_agent and event_agent != last_agent:
                        if current_turn_data is not None:
                            finalized = finalize_agent_turn(current_turn_data)
                            if finalized is not None:
                                yield finalized
                                await asyncio.sleep(0.2)

                        current_turn_data = {
                            "type": "agent_turn",
                            "agent": event_agent,
                            "tools": [],
                            "response": "",
                        }
                        last_agent = event_agent
                        current_agent = event_agent

                    if current_turn_data is None:
                        current_turn_data = {
                            "type": "agent_turn",
                            "agent": event_agent or (current_agent or "系统"),
                            "tools": [],
                            "response": "",
                        }
                        last_agent = event_agent or last_agent
                        current_agent = event_agent or current_agent

                    current_turn_data["response"] = event.content
                continue

            if isinstance(event, TaskResult):
                if current_turn_data is not None:
                    finalized = finalize_agent_turn(current_turn_data)
                    if finalized is not None:
                        yield finalized
                        await asyncio.sleep(0.3)
                try:
                    if next_event_task is not None and not next_event_task.done():
                        next_event_task.cancel()
                except Exception:
                    pass
                break

            event_type = type(event).__name__
            if event_type == "ThoughtEvent":
                continue
            content = getattr(event, "content", None)
            if content and "ThoughtEvent" not in str(content):
                yield {
                    "type": "thought",
                    "agent": current_agent or "绯荤粺",
                    "content": f"[{event_type}] {str(content)[:200]}",
                }
                await asyncio.sleep(0.1)

        quality_metrics = shared_data.get("quality_metrics") or {}
        for feedback_turn in build_feedback_turns(shared_data):
            yield feedback_turn
            await asyncio.sleep(0.2)

        effective_pid_params = shared_data_store.get("selected_pid_params") or {}
        final_result = {
            "dataAnalysis": {
                "dataPoints": shared_data.get("data_points", 0),
                "windowPoints": shared_data.get("window_points", 0),
                "stepEvents": len(shared_data.get("step_events") or []),
                "stepEventDetails": shared_data.get("step_events") or [],
                "currentIAE": quality_metrics.get("IAE", 0.0),
                "samplingTime": shared_data.get("sampling_time", 1.0),
                "selectedWindow": shared_data.get("selected_window", {}),
                "historyRange": {
                    "startTime": shared_data.get("start_time", start_time),
                    "endTime": shared_data.get("end_time", end_time),
                },
                "qualityMetrics": quality_metrics,
            },
            "model": {
                "modelType": shared_data.get("model_type", "FOPDT"),
                "selectedModelParams": shared_data.get("selected_model_params", {}),
                "modelSelectionReason": shared_data.get("model_selection_reason", ""),
                "K": shared_data.get("K", 0.0),
                "T": shared_data.get("T", 0.0),
                "L": shared_data.get("L", 0.0),
                "confidence": shared_data.get("confidence", 0.0),
                "residue": shared_data.get("residue", 0.0),
                "normalizedRmse": shared_data.get("normalized_rmse", shared_data.get("residue", 0.0)),
                "rawRmse": shared_data.get("raw_rmse", 0.0),
                "r2Score": shared_data.get("r2_score", 0.0),
                "confidenceQuality": shared_data.get("confidence_quality", ""),
                "confidenceRecommendation": shared_data.get("confidence_recommendation", ""),
                "reasonCodes": shared_data.get("reason_codes", []),
                "nextActions": shared_data.get("next_actions", []),
                "selectedWindowSource": shared_data.get("selected_window_source", ""),
                "attempts": shared_data.get("attempts", []),
                "fitPreview": shared_data.get("fit_preview", {"points": []}),
                "windowOverview": shared_data.get("window_overview", {"points": []}),
            },
            "pidParams": {
                "Kp": effective_pid_params.get("Kp", shared_data.get("Kp", 0.0)),
                "Ki": effective_pid_params.get("Ki", shared_data.get("Ki", 0.0)),
                "Kd": effective_pid_params.get("Kd", shared_data.get("Kd", 0.0)),
                "Ti": effective_pid_params.get("Ti", shared_data.get("Ti", 0.0)),
                "Td": effective_pid_params.get("Td", shared_data.get("Td", 0.0)),
                "strategy": effective_pid_params.get("strategy", shared_data.get("strategy", "")),
                "strategyRequested": shared_data.get("strategy_requested", "AUTO"),
                "strategyUsed": shared_data.get("strategy_used", ""),
                "loopType": loop_type,
                "selectionReason": shared_data.get("selection_reason", ""),
                "selectionInputs": shared_data.get("selection_inputs", {}),
                "experienceGuidance": shared_data.get("experience_guidance", {}),
                "candidateStrategies": shared_data.get("candidate_strategies", []),
                "description": effective_pid_params.get("description", shared_data.get("description", "")),
            },
            "knowledge": {
                "guidance": shared_data.get("expert_knowledge_guidance", {}) or shared_data_store.get("expert_knowledge_guidance", {}),
            },
        }

        if "final_rating" in shared_data:
            final_result["evaluation"] = {
                "performance_score": shared_data.get("performance_score", 0.0),
                "method_confidence": shared_data.get("method_confidence", 0.0),
                "final_rating": shared_data.get("final_rating", 0.0),
                "strategy_used": shared_data.get("strategy_used", ""),
                "passed": shared_data.get("passed", False),
                "pass_threshold": shared_data.get("pass_threshold", 7.0),
                "failure_reason": shared_data.get("failure_reason", ""),
                "feedback_target": shared_data.get("feedback_target", ""),
                "feedback_target_display": shared_data.get("feedback_target_display", ""),
                "feedback_action": shared_data.get("feedback_action", ""),
                "initial_assessment": shared_data.get("initial_assessment", {}),
                "auto_refine_result": shared_data.get("auto_refine_result", {}),
                "model_retry_result": shared_data.get("model_retry_result", {}),
                "performance_details": shared_data.get("performance_details", {}),
                "final_details": shared_data.get("final_details", {}),
                "replay_evaluation": shared_data.get("replay_evaluation", {}),
            }

        final_result["tuningAdvice"] = _build_tuning_advice(final_result)

        experience_record = build_experience_record(
            loop_name=loop_name,
            loop_type=loop_type,
            loop_uri=loop_uri,
            data_source="csv" if csv_path else "history",
            start_time=shared_data.get("start_time", start_time),
            end_time=shared_data.get("end_time", end_time),
            shared_data=shared_data,
            final_result=final_result,
        )
        experience_id = persist_experience_record(experience_record)
        referenced_experience_ids = experience_record.get("referenced_experience_ids") or []
        reuse_summary = {}
        if register_experience_reuse and referenced_experience_ids:
            reuse_summary = register_experience_reuse(
                referenced_experience_ids,
                follow_up_passed=bool(final_result.get("evaluation", {}).get("passed", False)),
                follow_up_final_rating=float(final_result.get("evaluation", {}).get("final_rating", 0.0) or 0.0),
            )
        final_result["memory"] = {
            "experienceId": experience_id,
            "experienceGuidance": shared_data.get("experience_guidance", {}),
            "referenceReuse": reuse_summary,
        }

        yield {"type": "result", "data": final_result}
        yield {"type": "done", "status": "succeeded"}
    except asyncio.CancelledError:
        print(f"[workflow_runner] cancelled loop={loop_name} current_agent={current_agent or last_agent or ''}")
        yield {"type": "error", "message": "任务已取消"}
    except Exception as exc:
        print(f"[workflow_runner] exception loop={loop_name} type={type(exc).__name__} detail={exc}")
        mapped_error = _classify_workflow_exception(exc)
        yield {
            "type": "error",
            "message": mapped_error["message"],
            "error_code": mapped_error["error_code"],
            "error_type": mapped_error["error_type"],
            "error_detail": mapped_error["detail"],
        }
    finally:
        print(f"[workflow_runner] finalize loop={loop_name}")
        if next_event_task is not None:
            try:
                if not next_event_task.done():
                    next_event_task.cancel()
                try:
                    _ = next_event_task.exception()
                except asyncio.CancelledError:
                    pass
                except StopAsyncIteration:
                    pass
            except Exception:
                pass
