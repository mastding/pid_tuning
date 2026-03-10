# Multi-Agent PID Tuning System using AutoGen RoundRobinGroupChat
from __future__ import annotations

import asyncio
import os
import json
import sys
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
)
from services.pid_evaluation_service import (
    build_initial_assessment,
    choose_alternative_model_attempt,
    diagnose_evaluation_failure as service_diagnose_evaluation_failure,
    evaluate_pid_model,
)
from services.data_service import build_window_overview as service_build_window_overview, load_pid_dataset
from services.identification_service import (
    build_fit_preview as service_build_fit_preview,
    derive_model_reason_codes as service_derive_model_reason_codes,
    derive_next_actions as service_derive_next_actions,
    extract_candidate_windows as service_extract_candidate_windows,
    fit_best_fopdt_window,
)
from orchestration.event_mapper import (
    build_agent_response as orchestration_build_agent_response,
    build_feedback_turns as orchestration_build_feedback_turns,
    finalize_agent_turn as orchestration_finalize_agent_turn,
)
from orchestration.agent_factory import create_pid_agents as orchestration_create_pid_agents
from orchestration.workflow_runner import run_multi_agent_collaboration as orchestration_run_multi_agent_collaboration
from state.session_store import SessionStore


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


# ===== 全局会话状态 =====
_shared_data_store = SessionStore()
DISPLAY_AGENT_NAMES = {
    "data_analyst": "数据分析智能体",
    "system_id_expert": "系统辨识智能体",
    "pid_expert": "PID专家智能体",
    "evaluation_expert": "评估智能体",
}


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


def _format_float(value: Any, digits: int = 3) -> str:
    agent_name_map = DISPLAY_AGENT_NAMES

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _benchmark_pid_strategies(K: float, T: float, L: float, dt: float, confidence_score: float) -> Dict[str, Any]:
    return service_benchmark_pid_strategies(K, T, L, dt, confidence_score)


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


def _derive_model_reason_codes(model_params: Dict[str, Any], confidence: Dict[str, Any], quality_metrics: Dict[str, Any] | None) -> List[str]:
    return service_derive_model_reason_codes(model_params, confidence, quality_metrics)


def _derive_next_actions(confidence_score: float, reason_codes: List[str]) -> List[str]:
    return service_derive_next_actions(confidence_score, reason_codes)


def _build_fit_preview(window_df: Any, model_params: Dict[str, Any], max_points: int = 200) -> Dict[str, Any]:
    return service_build_fit_preview(window_df, model_params, float(_shared_data_store.get("dt", 1.0)), max_points=max_points)


def _build_window_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    max_points: int = 240,
) -> Dict[str, Any]:
    return service_build_window_overview(cleaned_df, selected_window, max_points=max_points)


def _build_agent_response(agent_name: str, tools: List[Dict[str, Any]]) -> str:
    return orchestration_build_agent_response(
        agent_name,
        tools,
        display_agent_names=DISPLAY_AGENT_NAMES,
    )


def _finalize_agent_turn(current_turn_data: Dict[str, Any] | None) -> Dict[str, Any] | None:
    return orchestration_finalize_agent_turn(
        current_turn_data,
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
        fetch_history_data_csv,
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
    )
    _shared_data_store["csv_path"] = result["csv_path"]
    _shared_data_store["loop_uri"] = result["loop_uri"]
    _shared_data_store["start_time"] = result["start_time"]
    _shared_data_store["end_time"] = result["end_time"]
    _shared_data_store["data_type"] = result["data_type"]
    return _to_jsonable(result)


async def tool_load_data(csv_path: str) -> Dict[str, Any]:
    """Load and preprocess PID historical data via the data service."""
    prepared = await asyncio.to_thread(load_pid_dataset, csv_path)
    _shared_data_store["csv_path"] = prepared["csv_path"]
    _shared_data_store["cleaned_df"] = prepared["cleaned_df"]
    _shared_data_store["window_df"] = prepared["window_df"]
    _shared_data_store["mv"] = prepared["mv"]
    _shared_data_store["pv"] = prepared["pv"]
    _shared_data_store["dt"] = prepared["dt"]
    _shared_data_store["step_events"] = prepared["step_events"]
    _shared_data_store["candidate_windows"] = prepared["candidate_windows"]
    _shared_data_store["selected_event"] = prepared["selected_event"]
    _shared_data_store["quality_metrics"] = prepared["quality_metrics"]
    _shared_data_store["selected_window"] = prepared["selected_window"]
    _shared_data_store["window_overview"] = prepared["window_overview"]

    return _to_jsonable({
        "data_points": prepared["data_points"],
        "window_points": prepared["window_points"],
        "sampling_time": prepared["sampling_time"],
        "mv_range": prepared["mv_range"],
        "pv_range": prepared["pv_range"],
        "available_columns": prepared["available_columns"],
        "step_events": prepared["step_events"],
        "candidate_windows": prepared["candidate_windows"],
        "selected_window": prepared["selected_window"],
        "window_overview": prepared["window_overview"],
        "quality_metrics": prepared["quality_metrics"],
        "status": prepared["status"],
    })


async def tool_fit_fopdt(dt: float = 1.0) -> Dict[str, Any]:
    """Fit the best FOPDT model from candidate windows via the identification service."""
    if "mv" not in _shared_data_store or "pv" not in _shared_data_store:
        raise ValueError("请先调用tool_load_data加载数据")

    actual_dt = float(_shared_data_store.get("dt", dt))
    identification = await asyncio.to_thread(
        fit_best_fopdt_window,
        cleaned_df=_shared_data_store.get("cleaned_df"),
        candidate_windows=_shared_data_store.get("candidate_windows") or [],
        quality_metrics=_shared_data_store.get("quality_metrics") or {},
        actual_dt=actual_dt,
        benchmark_fn=_benchmark_pid_strategies,
    )

    best_model_params = identification["model_params"]
    best_confidence = identification["confidence"]
    best_benchmark = identification["benchmark"]
    best_candidate_df = identification["candidate_df"]
    best_event = identification["event"]
    best_source = identification["source"]
    attempts = identification["attempts"]
    reason_codes = identification["reason_codes"]
    next_actions = identification["next_actions"]
    fit_preview = identification["fit_preview"]
    selected_window_payload = identification["selected_window"]

    if best_candidate_df is not None:
        _shared_data_store["window_df"] = best_candidate_df
        _shared_data_store["mv"] = best_candidate_df["MV"].to_numpy(dtype=float)
        _shared_data_store["pv"] = best_candidate_df["PV"].to_numpy(dtype=float)

    if best_event:
        _shared_data_store["selected_event"] = best_event
    if selected_window_payload:
        _shared_data_store["selected_window"] = selected_window_payload
        _shared_data_store["window_overview"] = _build_window_overview(_shared_data_store.get("cleaned_df"), selected_window_payload)

    _shared_data_store["K"] = float(best_model_params["K"])
    _shared_data_store["T"] = float(best_model_params["T"])
    _shared_data_store["L"] = float(best_model_params["L"])
    _shared_data_store["residue"] = float(best_model_params["residue"])
    _shared_data_store["normalized_rmse"] = float(best_model_params["normalized_rmse"])
    _shared_data_store["raw_rmse"] = float(best_model_params["raw_rmse"])
    _shared_data_store["r2_score"] = float(best_model_params["r2_score"])
    _shared_data_store["model_confidence"] = best_confidence
    _shared_data_store["model_attempts"] = attempts
    _shared_data_store["model_reason_codes"] = reason_codes
    _shared_data_store["model_next_actions"] = next_actions
    _shared_data_store["model_selected_source"] = best_source
    _shared_data_store["fit_preview"] = fit_preview
    _shared_data_store["window_benchmark"] = (best_benchmark or {}).get("best", {})

    return _to_jsonable({
        "K": float(best_model_params["K"]),
        "T": float(best_model_params["T"]),
        "L": float(best_model_params["L"]),
        "dt": actual_dt,
        "residue": float(best_model_params["residue"]),
        "normalized_rmse": float(best_model_params["normalized_rmse"]),
        "raw_rmse": float(best_model_params["raw_rmse"]),
        "r2_score": float(best_model_params["r2_score"]),
        "success": bool(best_model_params["success"]),
        "confidence": float(best_confidence["confidence"]),
        "confidence_quality": best_confidence["quality"],
        "confidence_recommendation": best_confidence["recommendation"],
        "rmse_score": float(best_confidence["rmse_score"]),
        "reason_codes": reason_codes,
        "next_actions": next_actions,
        "selected_window_source": best_source,
        "selected_window": selected_window_payload or _shared_data_store.get("selected_window", {}),
        "window_overview": _shared_data_store.get("window_overview", {"points": []}),
        "attempts": attempts,
        "fit_preview": fit_preview,
        "window_benchmark": (best_benchmark or {}).get("best", {}),
    })


async def tool_tune_pid(K: float, T: float, L: float, loop_type: str) -> Dict[str, Any]:
    """Apply and benchmark PID tuning rules, then keep the best candidate."""
    confidence_score = _safe_float((_shared_data_store.get("model_confidence") or {}).get("confidence"), 1.0)
    selected_model = {
        "normalized_rmse": _safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue"))),
        "r2_score": _safe_float(_shared_data_store.get("r2_score")),
    }
    selection = select_best_pid_strategy(
        K=float(K),
        T=float(T),
        L=float(L),
        loop_type=loop_type,
        confidence_score=confidence_score,
        normalized_rmse=selected_model["normalized_rmse"],
        r2_score=selected_model["r2_score"],
        dt=float(_shared_data_store.get("dt", 1.0)),
    )
    best_candidate = selection["best_candidate"]
    pid_params = selection["pid_params"]
    public_candidate_results = selection["public_candidate_results"]

    _shared_data_store["pid_params"] = pid_params
    _shared_data_store["pid_candidate_results"] = public_candidate_results
    _shared_data_store["strategy_used"] = best_candidate["strategy"]
    _shared_data_store["selection_reason"] = selection["selection_reason"]
    _shared_data_store["selection_inputs"] = selection["selection_inputs"]
    _shared_data_store["selected_pid_params"] = {
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"]),
        "Ti": float(pid_params["Ti"]),
        "Td": float(pid_params["Td"]),
        "strategy": best_candidate["strategy"],
        "description": str(pid_params["description"]),
    }
    _shared_data_store["selected_pid_evaluation"] = best_candidate["evaluation_result"]

    return _to_jsonable({
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"]),
        "Ti": float(pid_params["Ti"]),
        "Td": float(pid_params["Td"]),
        "strategy": str(pid_params["strategy"]),
        "strategy_requested": "AUTO_BENCHMARK",
        "strategy_used": best_candidate["strategy"],
        "model_confidence": confidence_score,
        "loop_type": loop_type,
        "selection_reason": _shared_data_store["selection_reason"],
        "selection_inputs": _shared_data_store["selection_inputs"],
        "candidate_strategies": public_candidate_results,
        "description": str(pid_params["description"]),
    })


async def tool_evaluate_pid(
    K: float,
    T: float,
    L: float,
    Kp: float = 0.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    method: str = "auto",
) -> Dict[str, Any]:
    """Evaluate tuned PID parameters and trigger bounded feedback refinement."""
    model_confidence = _shared_data_store.get("model_confidence", {})
    method_confidence = float(model_confidence.get("confidence", 0.6))
    selected_pid_params = _shared_data_store.get("selected_pid_params") or {}
    selected_pid_evaluation = _shared_data_store.get("selected_pid_evaluation")
    auto_refine_result = None
    model_retry_result = None

    if selected_pid_params:
        Kp = float(selected_pid_params.get("Kp", Kp))
        Ki = float(selected_pid_params.get("Ki", Ki))
        Kd = float(selected_pid_params.get("Kd", Kd))

    if selected_pid_evaluation:
        eval_result = selected_pid_evaluation
    else:
        eval_result = evaluate_pid_model(
            K=float(K),
            T=float(T),
            L=float(L),
            Kp=float(Kp),
            Ki=float(Ki),
            Kd=float(Kd),
            method=method,
            method_confidence=method_confidence,
            model_confidence=model_confidence,
            dt=float(_shared_data_store.get("dt", 1.0)),
        )

    base_eval_result = eval_result
    _shared_data_store["evaluation_result"] = eval_result
    pass_threshold = 7.0
    passed = bool(_safe_float(eval_result.get("final_rating")) >= pass_threshold)
    diagnosis = service_diagnose_evaluation_failure(
        eval_result=eval_result,
        model_r2=_safe_float(_shared_data_store.get("r2_score")),
        model_rmse=_safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue"))),
        candidate_window_count=len(_shared_data_store.get("candidate_windows") or []),
    ) if not passed else {
        "failure_reason": "",
        "feedback_target": "",
        "feedback_action": "",
    }
    initial_assessment = build_initial_assessment(
        eval_result=base_eval_result,
        pass_threshold=pass_threshold,
        diagnosis=diagnosis,
        evaluated_pid={"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    )
    _shared_data_store["initial_assessment"] = initial_assessment

    if not passed and diagnosis.get("feedback_target") == "pid_expert":
        refined = _refine_pid_for_performance(
            model_params={"K": float(K), "T1": float(T), "T2": 0.0, "L": float(L)},
            base_pid_params={"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
            method_confidence=method_confidence,
            dt=float(_shared_data_store.get("dt", 1.0)),
            base_strategy=str(_shared_data_store.get("strategy_used", method or "auto")),
        )
        best_refined = refined.get("best") or {}
        if best_refined:
            improved = float(best_refined.get("final_rating", 0.0)) > float(eval_result.get("final_rating", 0.0)) + 1e-9
            if improved:
                Kp = float(best_refined["Kp"])
                Ki = float(best_refined["Ki"])
                Kd = float(best_refined["Kd"])
                eval_result = best_refined["evaluation_result"]
                _shared_data_store["selected_pid_params"] = {
                    **(_shared_data_store.get("selected_pid_params") or {}),
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "strategy": str(_shared_data_store.get("strategy_used", method or "auto")),
                    "description": "Auto refined after evaluation feedback",
                }
                _shared_data_store["selected_pid_evaluation"] = eval_result
                _shared_data_store["evaluation_result"] = eval_result
                auto_refine_result = {
                    "applied": True,
                    "base_final_rating": float(base_eval_result.get("final_rating", 0.0)),
                    "refined_final_rating": float(eval_result.get("final_rating", 0.0)),
                    "refined_performance_score": float(eval_result.get("performance_score", 0.0)),
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                }
                passed = bool(_safe_float(eval_result.get("final_rating")) >= pass_threshold)
                diagnosis = service_diagnose_evaluation_failure(
                    eval_result=eval_result,
                    model_r2=_safe_float(_shared_data_store.get("r2_score")),
                    model_rmse=_safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue"))),
                    candidate_window_count=len(_shared_data_store.get("candidate_windows") or []),
                ) if not passed else {
                    "failure_reason": "",
                    "feedback_target": "",
                    "feedback_action": "",
                }
            else:
                auto_refine_result = {
                    "applied": False,
                    "base_final_rating": float(base_eval_result.get("final_rating", 0.0)),
                    "refined_final_rating": float(best_refined.get("final_rating", 0.0)),
                    "refined_performance_score": float(best_refined.get("performance_score", 0.0)),
                }

    if not passed:
        alternative_model = choose_alternative_model_attempt(
            attempts=_shared_data_store.get("model_attempts") or [],
            current_source=str(_shared_data_store.get("model_selected_source", "")),
            candidate_map={candidate["name"]: candidate for candidate in _extract_candidate_windows()},
            loop_type=str(_shared_data_store.get("loop_type", "flow")),
            dt=float(_shared_data_store.get("dt", 1.0)),
            pass_threshold=pass_threshold,
            benchmark_fn=_benchmark_pid_strategies,
            refine_fn=_refine_pid_for_performance,
        )
        if alternative_model:
            alternative_eval = alternative_model["evaluation_result"]
            if float(alternative_eval.get("final_rating", 0.0)) > float(eval_result.get("final_rating", 0.0)) + 1e-9:
                K = float(alternative_model["K"])
                T = float(alternative_model["T"])
                L = float(alternative_model["L"])
                Kp = float(alternative_model["Kp"])
                Ki = float(alternative_model["Ki"])
                Kd = float(alternative_model["Kd"])
                eval_result = alternative_eval
                passed = bool(_safe_float(eval_result.get("final_rating")) >= pass_threshold)
                diagnosis = service_diagnose_evaluation_failure(
                    eval_result=eval_result,
                    model_r2=_safe_float(_shared_data_store.get("r2_score")),
                    model_rmse=_safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue"))),
                    candidate_window_count=len(_shared_data_store.get("candidate_windows") or []),
                ) if not passed else {
                    "failure_reason": "",
                    "feedback_target": "",
                    "feedback_action": "",
                }
                _shared_data_store["K"] = K
                _shared_data_store["T"] = T
                _shared_data_store["L"] = L
                _shared_data_store["strategy_used"] = str(alternative_model.get("strategy", _shared_data_store.get("strategy_used", "")))
                _shared_data_store["model_selected_source"] = str(alternative_model.get("window_source", _shared_data_store.get("model_selected_source", "")))
                _shared_data_store["selected_pid_params"] = {
                    **(_shared_data_store.get("selected_pid_params") or {}),
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "strategy": str(alternative_model.get("strategy", "")),
                    "description": "Switched to alternative identification window",
                }
                _shared_data_store["selected_pid_evaluation"] = eval_result
                model_retry_result = {
                    "applied": True,
                    "window_source": str(alternative_model.get("window_source", "")),
                    "strategy": str(alternative_model.get("strategy", "")),
                    "final_rating": float(eval_result.get("final_rating", 0.0)),
                    "performance_score": float(eval_result.get("performance_score", 0.0)),
                    "K": K,
                    "T": T,
                    "L": L,
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                }

    _shared_data_store["evaluation_pass_threshold"] = pass_threshold
    _shared_data_store["evaluation_feedback"] = diagnosis
    _shared_data_store["initial_assessment"] = initial_assessment
    _shared_data_store["auto_refine_result"] = auto_refine_result or {}
    _shared_data_store["model_retry_result"] = model_retry_result or {}
    _shared_data_store["performance_score"] = float(eval_result["performance_score"])
    _shared_data_store["method_confidence"] = float(eval_result["method_confidence"])
    _shared_data_store["final_rating"] = float(eval_result["final_rating"])
    _shared_data_store["passed"] = passed
    _shared_data_store["pass_threshold"] = pass_threshold
    _shared_data_store["failure_reason"] = diagnosis["failure_reason"]
    _shared_data_store["feedback_target"] = diagnosis["feedback_target"]
    _shared_data_store["feedback_target_display"] = DISPLAY_AGENT_NAMES.get(diagnosis["feedback_target"], diagnosis["feedback_target"])
    _shared_data_store["feedback_action"] = diagnosis["feedback_action"]
    _shared_data_store["performance_details"] = eval_result["performance_details"]
    _shared_data_store["final_details"] = eval_result["final_details"]
    _shared_data_store["simulation"] = eval_result["simulation"]
    return _to_jsonable({
        "performance_score": float(eval_result["performance_score"]),
        "method_confidence": float(eval_result["method_confidence"]),
        "final_rating": float(eval_result["final_rating"]),
        "passed": passed,
        "pass_threshold": pass_threshold,
        "performance_details": eval_result["performance_details"],
        "final_details": eval_result["final_details"],
        "failure_reason": diagnosis["failure_reason"],
        "feedback_target": diagnosis["feedback_target"],
        "feedback_target_display": DISPLAY_AGENT_NAMES.get(diagnosis["feedback_target"], diagnosis["feedback_target"]),
        "feedback_action": diagnosis["feedback_action"],
        "initial_assessment": initial_assessment,
        "auto_refine_result": auto_refine_result or {},
        "model_retry_result": model_retry_result or {},
        "simulation": eval_result["simulation"],
        "evaluated_pid": {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    })


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
        tool_tune_pid=tool_tune_pid,
        tool_evaluate_pid=tool_evaluate_pid,
    )


async def run_multi_agent_collaboration(
    csv_path: str,
    loop_name: str,
    loop_type: str,
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
        to_jsonable=_to_jsonable,
    ):
        yield event


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
        loop_type: str = Form("flow"),
        loop_uri: str = Form(DEFAULT_LOOP_URI),
        start_time: str = Form(DEFAULT_HISTORY_START_TIME),
        end_time: str = Form(DEFAULT_HISTORY_END_TIME),
        data_type: str = Form("interpolated"),
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
                    loop_type=loop_type,
                    loop_uri=loop_uri,
                    start_time=start_time,
                    end_time=end_time,
                    data_type=data_type,
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
