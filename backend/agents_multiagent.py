# Multi-agent PID tuning entry module.
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List

import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from skills.data_analysis_skills import (
    DEFAULT_HISTORY_END_TIME,
    DEFAULT_HISTORY_START_TIME,
    DEFAULT_LOOP_URI,
    fetch_history_data_csv,
)
from services.pid_tuning_service import (
    benchmark_pid_strategies as service_benchmark_pid_strategies,
    refine_pid_for_performance as service_refine_pid_for_performance,
    select_best_pid_strategy,
)
from services.pid_evaluation_service import (
    build_initial_assessment,
    choose_alternative_model_attempt,
    diagnose_evaluation_failure as service_diagnose_evaluation_failure,
    evaluate_pid_model,
)
from services.data_service import build_window_overview as service_build_window_overview, load_pid_dataset
from services.identification_service import (
    extract_candidate_windows as service_extract_candidate_windows,
    fit_best_fopdt_window,
)
from services.tool_adapter_service import (
    evaluate_pid_tool as service_evaluate_pid_tool,
    fetch_history_data_tool as service_fetch_history_data_tool,
    fit_fopdt_tool as service_fit_fopdt_tool,
    load_data_tool as service_load_data_tool,
    query_expert_knowledge_tool as service_query_expert_knowledge_tool,
    tune_pid_tool as service_tune_pid_tool,
)
from services.system_config_service import (
    DEFAULT_KNOWLEDGE_GRAPH_ID,
    get_backend_port_config,
    get_knowledge_graph_runtime_config,
    get_model_runtime_config,
    get_model_timeout_config,
    get_server_port_config,
    is_knowledge_expert_enabled,
    is_llm_orchestration_enabled,
)
from orchestration.event_mapper import (
    build_agent_response as orchestration_build_agent_response,
    build_feedback_turns as orchestration_build_feedback_turns,
    finalize_agent_turn as orchestration_finalize_agent_turn,
)
from orchestration.agent_factory import create_pid_agents as orchestration_create_pid_agents
from orchestration.constants import DISPLAY_AGENT_NAMES
from orchestration.workflow_runner import (
    _build_tuning_advice,
    run_multi_agent_collaboration as orchestration_run_multi_agent_collaboration,
)
from api.tune_app import create_app
from memory.experience_service import build_experience_record, persist_experience_record, register_experience_reuse
from state.session_store import SessionStore

def create_model_client(*, model_api_key: str, model_api_url: str, model: str) -> OpenAIChatCompletionClient:
    """创建OpenAI兼容的模型客户端（千问）"""
    timeout_config = get_model_timeout_config()
    return OpenAIChatCompletionClient(
        api_key=model_api_key,
        base_url=model_api_url,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=float(timeout_config.get("connect", 20.0)),
                read=float(timeout_config.get("read", 120.0)),
                write=float(timeout_config.get("write", 60.0)),
                pool=float(timeout_config.get("pool", 30.0)),
            ),
            trust_env=False,
            http2=False,
            headers={
                "Connection": "close",
                "Accept-Encoding": "identity",
            },
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=20),
            transport=httpx.AsyncHTTPTransport(retries=3),
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


_shared_data_store = SessionStore()


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



def _benchmark_pid_strategies(
    K: float,
    T: float,
    L: float,
    dt: float,
    confidence_score: float,
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return service_benchmark_pid_strategies(
        K,
        T,
        L,
        dt,
        confidence_score,
        model_type=model_type,
        selected_model_params=selected_model_params,
    )


def _refine_pid_for_performance(
    model_params: Dict[str, float],
    base_pid_params: Dict[str, float],
    method_confidence: float,
    dt: float,
    base_strategy: str,
) -> Dict[str, Any]:
    return service_refine_pid_for_performance(
        model_params=model_params,
        base_pid_params=base_pid_params,
        method_confidence=method_confidence,
        dt=dt,
        base_strategy=base_strategy,
    )


def _extract_candidate_windows() -> List[Dict[str, Any]]:
    return service_extract_candidate_windows(
        _shared_data_store.get("cleaned_df"),
        _shared_data_store.get("candidate_windows") or [],
    )



def _build_window_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    max_points: int = 240,
) -> Dict[str, Any]:
    return service_build_window_overview(cleaned_df, selected_window, max_points=max_points)



def _finalize_agent_turn(current_turn_data: Dict[str, Any] | None) -> Dict[str, Any] | None:
    return orchestration_finalize_agent_turn(
        current_turn_data,
        build_agent_response=orchestration_build_agent_response,
        display_agent_names=DISPLAY_AGENT_NAMES,
    )


def _build_feedback_turns(shared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return orchestration_build_feedback_turns(
        shared_data,
        display_agent_names=DISPLAY_AGENT_NAMES,
        to_jsonable=_to_jsonable,
    )


async def tool_fetch_history_data(
    loop_uri: str = DEFAULT_LOOP_URI,
    start_time: str = DEFAULT_HISTORY_START_TIME,
    end_time: str = DEFAULT_HISTORY_END_TIME,
    data_type: str = "interpolated",
    window: int = 1,
) -> Dict[str, Any]:
    """Fetch historical CSV data and persist the local file path in session state."""
    result = await asyncio.to_thread(
        service_fetch_history_data_tool,
        session_store=_shared_data_store,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        window=window,
        fetch_history_data_csv_fn=fetch_history_data_csv,
    )
    return _to_jsonable(result)


async def tool_load_data(csv_path: str, selected_window_index: int | None = None) -> Dict[str, Any]:
    """Load and preprocess PID historical data via the data service."""
    if selected_window_index is None:
        selected_window_index = _shared_data_store.get("selected_window_index")
    selected_loop_prefix = _shared_data_store.get("selected_loop_prefix")
    start_time = _shared_data_store.get("start_time")
    end_time = _shared_data_store.get("end_time")
    result = await asyncio.to_thread(
        service_load_data_tool,
        session_store=_shared_data_store,
        csv_path=csv_path,
        selected_loop_prefix=selected_loop_prefix,
        selected_window_index=selected_window_index,
        start_time=start_time,
        end_time=end_time,
        load_pid_dataset_fn=load_pid_dataset,
    )
    return _to_jsonable(result)


async def tool_fit_fopdt(dt: float = 1.0) -> Dict[str, Any]:
    """Identify the best process model from candidate windows via the identification service."""
    result = await asyncio.to_thread(
        service_fit_fopdt_tool,
        session_store=_shared_data_store,
        dt=dt,
        fit_best_fopdt_window_fn=fit_best_fopdt_window,
        build_window_overview_fn=_build_window_overview,
        benchmark_fn=_benchmark_pid_strategies,
        loop_type=str(_shared_data_store.get("loop_type", "flow")),
    )
    return _to_jsonable(result)


async def tool_tune_pid(
    loop_type: str,
    model_type: str = "AUTO",
    selected_model_params: Any = None,
    K: float | None = None,
    T: float | None = None,
    L: float | None = None,
) -> Dict[str, Any]:
    """Apply and benchmark PID tuning rules for the identified process model."""
    result = await asyncio.to_thread(
        service_tune_pid_tool,
        session_store=_shared_data_store,
        K=K,
        T=T,
        L=L,
        loop_type=loop_type,
        model_type=model_type,
        selected_model_params=selected_model_params,
        select_best_pid_strategy_fn=select_best_pid_strategy,
    )
    return _to_jsonable(result)


async def tool_query_expert_knowledge(
    loop_type: str,
    loop_name: str = "",
    plant_type: str = "",
    scenario: str = "",
    control_object: str = "",
    tower_section: str = "",
    control_target: str = "",
) -> Dict[str, Any]:
    """Query distillation-column expert knowledge and store the guidance for PID tuning."""
    if not is_knowledge_expert_enabled():
        _shared_data_store["expert_knowledge_guidance_full"] = {"answers": [], "graph_hints": [], "graph_summary": "已关闭知识检索步骤"}
        _shared_data_store["expert_knowledge_guidance"] = {
            "preferred_strategy": "",
            "risk_hints": [],
            "answers": [],
            "summary": "已关闭知识检索步骤",
            "questions": [],
        }
        _shared_data_store["knowledge_questions"] = []
        return _to_jsonable(_shared_data_store["expert_knowledge_guidance"])

    knowledge_runtime_config = get_knowledge_graph_runtime_config()
    result = await asyncio.to_thread(
        service_query_expert_knowledge_tool,
        session_store=_shared_data_store,
        loop_type=loop_type,
        loop_name=loop_name,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        tower_section=tower_section,
        control_target=control_target,
        graph_id=knowledge_runtime_config.get("graph_id", DEFAULT_KNOWLEDGE_GRAPH_ID),
        graph_api_url=knowledge_runtime_config.get("graph_api_url", ""),
        query_mode=os.getenv("KNOWLEDGE_GRAPH_QUERY_MODE", "local"),
        response_type=os.getenv("KNOWLEDGE_GRAPH_RESPONSE_TYPE", "要点式，尽量精炼"),
        include_context=True,
    )
    return _to_jsonable(result)


async def tool_evaluate_pid(
    model_type: str = "AUTO",
    selected_model_params: Any = None,
    K: float = 0.0,
    T: float = 0.0,
    L: float = 0.0,
    Kp: float = 0.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    method: str = "auto",
) -> Dict[str, Any]:
    """Evaluate tuned PID parameters and trigger bounded feedback refinement."""
    result = await asyncio.to_thread(
        service_evaluate_pid_tool,
        session_store=_shared_data_store,
        model_type=model_type,
        selected_model_params=selected_model_params,
        K=K,
        T=T,
        L=L,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        method=method,
        display_agent_names=DISPLAY_AGENT_NAMES,
        evaluate_pid_model_fn=evaluate_pid_model,
        diagnose_failure_fn=service_diagnose_evaluation_failure,
        build_initial_assessment_fn=build_initial_assessment,
        refine_pid_for_performance_fn=_refine_pid_for_performance,
        choose_alternative_model_attempt_fn=choose_alternative_model_attempt,
        benchmark_fn=_benchmark_pid_strategies,
        extract_candidate_windows_fn=_extract_candidate_windows,
    )
    return _to_jsonable(result)


def create_pid_agents(
    *,
    model_client: OpenAIChatCompletionClient,
    csv_path: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    window: int,
    loop_type: str,
) -> List[AssistantAgent]:
    return orchestration_create_pid_agents(
        model_client=model_client,
        csv_path=csv_path,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        window=window,
        loop_type=loop_type,
        tool_load_data=tool_load_data,
        tool_fetch_history_data=tool_fetch_history_data,
        tool_fit_fopdt=tool_fit_fopdt,
        tool_query_expert_knowledge=tool_query_expert_knowledge,
        tool_tune_pid=tool_tune_pid,
        tool_evaluate_pid=tool_evaluate_pid,
    )


async def run_multi_agent_collaboration(
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
    llm_config: Dict[str, Any],
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    task_session_id: str = "",
    uploaded_file_name: str = "",
    uploaded_file_hash: str = "",
    uploaded_original_file_path: str = "",
    task_artifact_dir: str = "",
) -> AsyncGenerator[Dict[str, Any], None]:
    async def _fallback_without_llm() -> AsyncGenerator[Dict[str, Any], None]:
        shared_data: Dict[str, Any] = {}
        _shared_data_store.clear()
        _shared_data_store["loop_name"] = loop_name
        _shared_data_store["loop_type"] = loop_type
        _shared_data_store["plant_type"] = plant_type
        _shared_data_store["scenario"] = scenario
        _shared_data_store["control_object"] = control_object
        _shared_data_store["start_time"] = start_time
        _shared_data_store["end_time"] = end_time
        if task_session_id:
            _shared_data_store["task_session_id"] = task_session_id
        if uploaded_file_name:
            _shared_data_store["uploaded_file_name"] = uploaded_file_name
        if uploaded_file_hash:
            _shared_data_store["uploaded_file_hash"] = uploaded_file_hash
        if uploaded_original_file_path:
            _shared_data_store["uploaded_original_file_path"] = uploaded_original_file_path
        if task_artifact_dir:
            _shared_data_store["task_artifact_dir"] = task_artifact_dir
        if selected_loop_prefix is not None:
            _shared_data_store["selected_loop_prefix"] = selected_loop_prefix
        if selected_window_index is not None:
            _shared_data_store["selected_window_index"] = selected_window_index

        effective_csv_path = csv_path
        if not effective_csv_path:
            try:
                fetch_result = await tool_fetch_history_data(
                    loop_uri=loop_uri,
                    start_time=start_time,
                    end_time=end_time,
                    data_type=data_type,
                    window=int(window or 1),
                )
            except Exception as exc:
                yield {
                    "type": "error",
                    "message": "本地整定流程失败：历史数据获取失败",
                    "error_type": "LOCAL_FLOW_ERROR",
                    "error_code": "LOCAL_FETCH_HISTORY_FAILED",
                    "error_detail": str(exc),
                }
                yield {"type": "done", "status": "failed"}
                return
            shared_data.update(fetch_result)
            effective_csv_path = str(fetch_result.get("csv_path") or "")
            yield {
                "type": "agent_turn",
                "agent": "数据分析智能体",
                "tools": [
                    {
                        "tool_name": "tool_fetch_history_data",
                        "args": {
                            "loop_uri": loop_uri,
                            "start_time": start_time,
                            "end_time": end_time,
                            "data_type": data_type,
                            "window": int(window or 1),
                        },
                        "result": fetch_result,
                    }
                ],
                "response": "已从历史接口获取数据并生成CSV，准备加载与预处理。",
            }

        try:
            load_result = await tool_load_data(effective_csv_path)
        except Exception as exc:
            yield {
                "type": "error",
                "message": "本地整定流程失败：数据加载失败",
                "error_type": "LOCAL_FLOW_ERROR",
                "error_code": "LOCAL_LOAD_DATA_FAILED",
                "error_detail": str(exc),
            }
            yield {"type": "done", "status": "failed"}
            return
        shared_data.update(load_result)
        dt = float(_shared_data_store.get("dt", 1.0) or 1.0)
        try:
            data_points = int(load_result.get("data_points") or 0)
        except Exception:
            data_points = 0
        try:
            sampling_time = float(load_result.get("sampling_time") or dt)
        except Exception:
            sampling_time = dt
        step_event_count = len(load_result.get("step_events") or [])
        candidate_windows = load_result.get("candidate_windows") or []
        usable_count = len(
            [w for w in candidate_windows if isinstance(w, dict) and w.get("window_usable_for_id") is True]
        )
        yield {
            "type": "agent_turn",
            "agent": "数据分析智能体",
            "tools": [
                {
                    "tool_name": "tool_load_data",
                    "args": {"csv_path": effective_csv_path},
                    "result": load_result,
                }
            ],
            "response": (
                f"数据分析完成：已加载{data_points}个数据点，采样间隔{sampling_time}秒，"
                f"检测到{step_event_count}个阶跃事件，构造{len(candidate_windows)}个候选窗口，"
                f"其中{usable_count}个可用于辨识；已将候选窗口传递给系统辨识阶段。"
            ),
        }

        try:
            id_result = await tool_fit_fopdt(dt=dt)
        except Exception as exc:
            yield {
                "type": "error",
                "message": "本地整定流程失败：系统辨识失败",
                "error_type": "LOCAL_FLOW_ERROR",
                "error_code": "LOCAL_IDENTIFICATION_FAILED",
                "error_detail": str(exc),
            }
            yield {"type": "done", "status": "failed"}
            return
        shared_data.update(id_result)
        yield {
            "type": "agent_turn",
            "agent": "系统辨识智能体",
            "tools": [
                {
                    "tool_name": "tool_fit_fopdt",
                    "args": {"dt": dt},
                    "result": id_result,
                }
            ],
            "response": "系统辨识完成，已选取最优窗口与过程模型并输出拟合指标与置信度。",
        }

        if is_knowledge_expert_enabled():
            try:
                knowledge_result = await tool_query_expert_knowledge(
                    loop_type=loop_type,
                    loop_name=loop_name,
                    plant_type=plant_type,
                    scenario=scenario,
                    control_object=control_object,
                )
            except Exception as exc:
                yield {
                    "type": "error",
                    "message": "本地整定流程失败：知识检索失败",
                    "error_type": "LOCAL_FLOW_ERROR",
                    "error_code": "LOCAL_KNOWLEDGE_FAILED",
                    "error_detail": str(exc),
                }
                yield {"type": "done", "status": "failed"}
                return
            shared_data.update({"knowledge_guidance": knowledge_result})
            yield {
                "type": "agent_turn",
                "agent": "知识增强智能体",
                "tools": [
                    {
                        "tool_name": "tool_query_expert_knowledge",
                        "args": {
                            "loop_type": loop_type,
                            "loop_name": loop_name,
                            "plant_type": plant_type,
                            "scenario": scenario,
                            "control_object": control_object,
                        },
                        "result": knowledge_result,
                    }
                ],
                "response": "知识检索完成，已生成策略偏好与风险提示，供PID整定参考。",
            }
        else:
            shared_data.update({"knowledge_guidance": {"summary": "已关闭知识检索步骤"}})

        model_type_value = str(_shared_data_store.get("model_type", "FOPDT"))
        selected_model_params = _shared_data_store.get("selected_model_params") or {}
        try:
            tune_result = await tool_tune_pid(
                loop_type=loop_type,
                model_type=model_type_value,
                selected_model_params=selected_model_params,
            )
        except Exception as exc:
            yield {
                "type": "error",
                "message": "本地整定流程失败：PID整定失败",
                "error_type": "LOCAL_FLOW_ERROR",
                "error_code": "LOCAL_TUNE_PID_FAILED",
                "error_detail": str(exc),
            }
            yield {"type": "done", "status": "failed"}
            return
        shared_data.update(tune_result)
        yield {
            "type": "agent_turn",
            "agent": "PID专家智能体",
            "tools": [
                {
                    "tool_name": "tool_tune_pid",
                    "args": {
                        "loop_type": loop_type,
                        "model_type": model_type_value,
                        "selected_model_params": selected_model_params,
                    },
                    "result": tune_result,
                }
            ],
            "response": "PID整定完成，已生成候选策略并选定最终PID参数。",
        }

        try:
            evaluation_result = await tool_evaluate_pid(
                model_type=model_type_value,
                selected_model_params=selected_model_params,
                method="auto",
            )
        except Exception as exc:
            yield {
                "type": "error",
                "message": "本地整定流程失败：评估失败",
                "error_type": "LOCAL_FLOW_ERROR",
                "error_code": "LOCAL_EVALUATE_PID_FAILED",
                "error_detail": str(exc),
            }
            yield {"type": "done", "status": "failed"}
            return
        shared_data.update(evaluation_result)
        yield {
            "type": "agent_turn",
            "agent": "评估智能体",
            "tools": [
                {
                    "tool_name": "tool_evaluate_pid",
                    "args": {
                        "model_type": model_type_value,
                        "selected_model_params": selected_model_params,
                        "method": "auto",
                    },
                    "result": evaluation_result,
                }
            ],
            "response": "评估完成，已输出综合评分、通过与否及下一步建议。",
        }

        try:
            quality_metrics = shared_data.get("quality_metrics") or {}
            effective_pid_params = _shared_data_store.get("selected_pid_params") or {}
            initial_assessment = shared_data.get("initial_assessment") or {}
            evaluated_pid = initial_assessment.get("evaluated_pid") or {}
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
                    "startTime": (shared_data.get("history_range") or {}).get("start_time")
                    or shared_data.get("start_time", start_time),
                    "endTime": (shared_data.get("history_range") or {}).get("end_time")
                    or shared_data.get("end_time", end_time),
                },
                "qualityMetrics": quality_metrics,
                "artifacts": {
                    "taskId": shared_data.get("task_session_id", ""),
                    "artifactDirectory": shared_data.get("task_artifact_dir", ""),
                    "uploadedOriginalCsvPath": shared_data.get("uploaded_original_file_path", ""),
                    "processedCsvPath": shared_data.get("processed_csv_path", ""),
                    "uploadedFileName": shared_data.get("uploaded_file_name", ""),
                    "uploadedFileHash": shared_data.get("uploaded_file_hash", ""),
                },
            },
            "model": {
                "modelType": _shared_data_store.get("identification_best_model_type", _shared_data_store.get("model_type", "FOPDT")),
                "selectedModelParams": _shared_data_store.get("identification_best_model_params", _shared_data_store.get("selected_model_params", {})),
                "modelSelectionReason": _shared_data_store.get("model_selection_reason", ""),
                "K": (_shared_data_store.get("identification_best_model_params", {}) or {}).get("K", _shared_data_store.get("K", 0.0)),
                "T": (_shared_data_store.get("identification_best_model_params", {}) or {}).get(
                    "T",
                    ((_shared_data_store.get("identification_best_model_params", {}) or {}).get("T1", 0.0) + (_shared_data_store.get("identification_best_model_params", {}) or {}).get("T2", 0.0))
                    or _shared_data_store.get("T", 0.0),
                ),
                "L": (_shared_data_store.get("identification_best_model_params", {}) or {}).get("L", _shared_data_store.get("L", 0.0)),
                "confidence": (_shared_data_store.get("model_confidence") or {}).get("confidence", 0.0),
                "residue": _shared_data_store.get("residue", 0.0),
                "normalizedRmse": _shared_data_store.get("normalized_rmse", _shared_data_store.get("residue", 0.0)),
                "rawRmse": _shared_data_store.get("raw_rmse", 0.0),
                "r2Score": _shared_data_store.get("r2_score", 0.0),
                "confidenceQuality": (_shared_data_store.get("model_confidence") or {}).get("quality", ""),
                "confidenceRecommendation": (_shared_data_store.get("model_confidence") or {}).get("recommendation", ""),
                "reasonCodes": _shared_data_store.get("model_reason_codes", []),
                "nextActions": _shared_data_store.get("model_next_actions", []),
                "selectedWindowSource": _shared_data_store.get("identification_best_window_source", _shared_data_store.get("model_selected_source", "")),
                "attempts": _shared_data_store.get("model_attempts", []),
                "identificationCandidates": _shared_data_store.get("model_identification_candidates", []),
                "tuningSelectedModelType": _shared_data_store.get("tuning_selected_model_type", _shared_data_store.get("model_type", "FOPDT")),
                "tuningSelectedModelParams": _shared_data_store.get("tuning_selected_model_params", _shared_data_store.get("selected_model_params", {})),
                "tuningSelectedWindowSource": _shared_data_store.get("tuning_selected_window_source", _shared_data_store.get("model_selected_source", "")),
                "fitPreview": _shared_data_store.get("fit_preview", {"points": []}),
                "windowOverview": _shared_data_store.get("window_overview", {"points": []}),
            },
            "pidParams": {
                "Kp": effective_pid_params.get("Kp", shared_data.get("Kp", evaluated_pid.get("Kp", 0.0))),
                "Ki": effective_pid_params.get("Ki", shared_data.get("Ki", evaluated_pid.get("Ki", 0.0))),
                "Kd": effective_pid_params.get("Kd", shared_data.get("Kd", evaluated_pid.get("Kd", 0.0))),
                "Ti": effective_pid_params.get("Ti", shared_data.get("Ti", 0.0)),
                "Td": effective_pid_params.get("Td", shared_data.get("Td", 0.0)),
                "strategy": effective_pid_params.get("strategy", _shared_data_store.get("strategy_used", "")),
                "strategyRequested": _shared_data_store.get("strategy_requested", "AUTO"),
                "strategyUsed": _shared_data_store.get("strategy_used", ""),
                "loopType": loop_type,
                "selectionReason": _shared_data_store.get("selection_reason", ""),
                "selectionInputs": _shared_data_store.get("selection_inputs", {}),
                "experienceGuidance": _shared_data_store.get("experience_guidance", {}),
                "candidateStrategies": _shared_data_store.get("pid_candidate_results", []),
                "tuningModelCandidates": _shared_data_store.get("pid_tuning_model_candidates", []),
                "tuningSelectedModelType": _shared_data_store.get("tuning_selected_model_type", _shared_data_store.get("model_type", "FOPDT")),
                "tuningSelectedModelParams": _shared_data_store.get("tuning_selected_model_params", _shared_data_store.get("selected_model_params", {})),
                "tuningSelectedWindowSource": _shared_data_store.get("tuning_selected_window_source", _shared_data_store.get("model_selected_source", "")),
                "identificationBestModelType": _shared_data_store.get("identification_best_model_type", _shared_data_store.get("model_type", "FOPDT")),
                "identificationBestWindowSource": _shared_data_store.get("identification_best_window_source", _shared_data_store.get("model_selected_source", "")),
                "description": effective_pid_params.get("description", ""),
            },
            "knowledge": {
                "guidance": _shared_data_store.get("expert_knowledge_guidance", {}),
            },
            }

            if "final_rating" in shared_data:
                final_result["evaluation"] = {
                "performance_score": shared_data.get("performance_score", 0.0),
                "method_confidence": shared_data.get("method_confidence", 0.0),
                "final_rating": shared_data.get("final_rating", 0.0),
                "strategy_used": _shared_data_store.get("strategy_used", ""),
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
            if referenced_experience_ids:
                reuse_summary = register_experience_reuse(
                    referenced_experience_ids,
                    follow_up_passed=bool(final_result.get("evaluation", {}).get("passed", False)),
                    follow_up_final_rating=float(final_result.get("evaluation", {}).get("final_rating", 0.0) or 0.0),
                )
            final_result["memory"] = {
                "experienceId": experience_id,
                "experienceGuidance": _shared_data_store.get("experience_guidance", {}),
                "referenceReuse": reuse_summary,
            }
        except Exception as exc:
            yield {
                "type": "error",
                "message": "本地整定流程失败：结果汇总失败",
                "error_type": "LOCAL_FLOW_ERROR",
                "error_code": "LOCAL_FINALIZE_FAILED",
                "error_detail": str(exc),
            }
            yield {"type": "done", "status": "failed"}
            return
        yield {"type": "result", "data": final_result}
        yield {"type": "done", "status": "succeeded"}


    if not is_llm_orchestration_enabled():
        yield {
            "type": "thought",
            "agent": "系统",
            "content": "已关闭LLM多智能体编排，改用本地确定性整定流程。",
        }
        async for fallback_event in _fallback_without_llm():
            yield fallback_event
        return

    if selected_window_index is not None:
        _shared_data_store["selected_window_index"] = selected_window_index
    if selected_loop_prefix is not None:
        _shared_data_store["selected_loop_prefix"] = selected_loop_prefix

    orchestration_completed = False
    try:
        async for event in orchestration_run_multi_agent_collaboration(
            csv_path=csv_path,
            task_session_id=task_session_id,
            uploaded_file_name=uploaded_file_name,
            uploaded_file_hash=uploaded_file_hash,
            uploaded_original_file_path=uploaded_original_file_path,
            task_artifact_dir=task_artifact_dir,
            loop_name=loop_name,
            loop_type=loop_type,
            plant_type=plant_type,
            scenario=scenario,
            control_object=control_object,
            loop_uri=loop_uri,
            start_time=start_time,
            end_time=end_time,
            data_type=data_type,
            window=window,
            selected_loop_prefix=selected_loop_prefix,
            selected_window_index=selected_window_index,
            llm_config=llm_config,
            shared_data_store=_shared_data_store,
            create_model_client=create_model_client,
            create_pid_agents=create_pid_agents,
            finalize_agent_turn=_finalize_agent_turn,
            build_feedback_turns=_build_feedback_turns,
            build_experience_record=build_experience_record,
            persist_experience_record=persist_experience_record,
            register_experience_reuse=register_experience_reuse,
            to_jsonable=_to_jsonable,
        ):
            if event.get("type") == "result":
                orchestration_completed = True
            if event.get("type") != "error":
                yield event
                continue

            detail = str(event.get("error_detail") or event.get("message") or "")
            lowered = detail.lower()
            fallback_markers = (
                "insufficient balance",
                "error code: 402",
                "cancellederror",
                "apiconnectionerror",
                "connection error",
                "read error",
                "readerror",
                "timeout",
                "timed out",
                "bad gateway",
                "service unavailable",
            )
            if any(marker in lowered for marker in fallback_markers):
                yield {
                    "type": "error",
                    "message": "LLM 编排中断，任务已终止。",
                    "error_type": "llm_orchestration_interrupted",
                    "error_code": "LLM_ORCHESTRATION_INTERRUPTED",
                    "error_detail": detail or "上游模型调用在响应阶段被中断或超时。",
                }
                return

            yield event
            return
    except asyncio.CancelledError:
        yield {
            "type": "error",
            "message": "LLM 编排被取消，任务已终止。",
            "error_type": "llm_orchestration_cancelled",
            "error_code": "LLM_ORCHESTRATION_CANCELLED",
            "error_detail": "上游模型调用或事件流在等待响应时被取消。",
        }
        return

    if not orchestration_completed:
        yield {
            "type": "error",
            "message": "LLM 编排异常结束且未产出结果，任务已终止。",
            "error_type": "llm_orchestration_incomplete",
            "error_code": "LLM_ORCHESTRATION_INCOMPLETE",
            "error_detail": "事件流提前结束，未生成最终结果。",
        }
        return


# ============ FastAPI Web服务 ============
if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn

    load_dotenv()
    
    # 获取后端服务器配置
    backend_config = get_backend_port_config()
    backend_host = backend_config["host"]
    backend_port = backend_config["port"]
    
    llm_config = get_model_runtime_config()
    app = create_app(
        run_multi_agent_collaboration=run_multi_agent_collaboration,
        llm_config=llm_config,
        default_loop_uri=DEFAULT_LOOP_URI,
        default_history_start_time=DEFAULT_HISTORY_START_TIME,
        default_history_end_time=DEFAULT_HISTORY_END_TIME,
    )
    print("Starting PID Tuning Multi-Agent System...")
    print(f"API endpoint: http://{backend_host}:{backend_port}/api/tune_stream")
    uvicorn.run(app, host=backend_host, port=backend_port)
