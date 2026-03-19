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
from orchestration.event_mapper import (
    build_agent_response as orchestration_build_agent_response,
    build_feedback_turns as orchestration_build_feedback_turns,
    finalize_agent_turn as orchestration_finalize_agent_turn,
)
from orchestration.agent_factory import create_pid_agents as orchestration_create_pid_agents
from orchestration.constants import DISPLAY_AGENT_NAMES
from orchestration.workflow_runner import run_multi_agent_collaboration as orchestration_run_multi_agent_collaboration
from api.tune_app import create_app
from memory.experience_service import build_experience_record, persist_experience_record, register_experience_reuse
from state.session_store import SessionStore

DEFAULT_KNOWLEDGE_GRAPH_API_URL = "http://graphrag.dicp.sixseven.ltd:5924/api/query"
DEFAULT_KNOWLEDGE_GRAPH_ID = "build_20260317_003858"


def create_model_client(*, model_api_key: str, model_api_url: str, model: str) -> OpenAIChatCompletionClient:
    """创建OpenAI兼容的模型客户端（千问）"""
    return OpenAIChatCompletionClient(
        api_key=model_api_key,
        base_url=model_api_url,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=30.0),
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
) -> Dict[str, Any]:
    """Fetch historical CSV data and persist the local file path in session state."""
    result = await asyncio.to_thread(
        service_fetch_history_data_tool,
        session_store=_shared_data_store,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        fetch_history_data_csv_fn=fetch_history_data_csv,
    )
    return _to_jsonable(result)


async def tool_load_data(csv_path: str) -> Dict[str, Any]:
    """Load and preprocess PID historical data via the data service."""
    result = await asyncio.to_thread(
        service_load_data_tool,
        session_store=_shared_data_store,
        csv_path=csv_path,
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
        graph_id=os.getenv("KNOWLEDGE_GRAPH_ID", DEFAULT_KNOWLEDGE_GRAPH_ID),
        graph_api_url=os.getenv("KNOWLEDGE_GRAPH_API_URL", DEFAULT_KNOWLEDGE_GRAPH_API_URL),
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
    loop_type: str,
) -> List[AssistantAgent]:
    return orchestration_create_pid_agents(
        model_client=model_client,
        csv_path=csv_path,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
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
    llm_config: Dict[str, Any],
) -> AsyncGenerator[Dict[str, Any], None]:
    async for event in orchestration_run_multi_agent_collaboration(
        csv_path=csv_path,
        loop_name=loop_name,
        loop_type=loop_type,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
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
        yield event


# ============ FastAPI Web服务 ============
if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn

    load_dotenv()
    llm_config = {
        "api_key": os.getenv("MODEL_API_KEY"),
        "base_url": os.getenv("MODEL_API_URL"),
        "model": os.getenv("MODEL", "qwen-plus"),
    }
    app = create_app(
        run_multi_agent_collaboration=run_multi_agent_collaboration,
        llm_config=llm_config,
        default_loop_uri=DEFAULT_LOOP_URI,
        default_history_start_time=DEFAULT_HISTORY_START_TIME,
        default_history_end_time=DEFAULT_HISTORY_END_TIME,
    )
    print("Starting PID Tuning Multi-Agent System...")
    print("API endpoint: http://0.0.0.0:3443/api/tune_stream")
    uvicorn.run(app, host="0.0.0.0", port=3443)
