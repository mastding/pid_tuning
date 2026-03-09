# Multi-Agent PID Tuning System using AutoGen RoundRobinGroupChat
from __future__ import annotations

import asyncio
import contextlib
import os
import json
import sys
from typing import Any, AsyncGenerator, Dict, List

import httpx
import numpy as np
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
from skills.pid_tuning_skills import apply_tuning_rules, select_tuning_strategy
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


def _build_agent_response(agent_name: str, tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""

    latest_result = None
    latest_tool_name = ""
    for tool in reversed(tools):
        if tool.get("result") is not None:
            latest_result = tool["result"]
            latest_tool_name = tool.get("tool_name", "")
            break

    if not isinstance(latest_result, dict):
        return ""

    if agent_name == DISPLAY_AGENT_NAMES["data_analyst"]:
        points = latest_result.get("data_points", 0)
        window_points = latest_result.get("window_points", 0)
        sampling_time = latest_result.get("sampling_time", 1.0)
        step_events = latest_result.get("step_events", [])
        return (
            f"??????????????? {points} ?????"
            f"?????????? {window_points} ??????? {_format_float(sampling_time, 2)} s?"
            f"??? {len(step_events) if isinstance(step_events, list) else 0} ????????"
        )

    if agent_name == DISPLAY_AGENT_NAMES["system_id_expert"]:
        extra = ""
        reason_codes = latest_result.get("reason_codes", [])
        if isinstance(reason_codes, list) and reason_codes:
            extra = f" ????: {', '.join(reason_codes)}?"
        next_actions = latest_result.get("next_actions", [])
        if isinstance(next_actions, list) and next_actions:
            extra += f" ????: {', '.join(next_actions)}?"
        return (
            f"FOPDT ????????? K={_format_float(latest_result.get('K'))}?"
            f"T={_format_float(latest_result.get('T'))}?L={_format_float(latest_result.get('L'))}?"
            f"???RMSE {_format_float(latest_result.get('normalized_rmse', latest_result.get('residue')))}?"
            f"R? {_format_float(latest_result.get('r2_score'), 3)}?????? {_format_float(latest_result.get('confidence'), 2)}?{extra}"
        )

    if agent_name == DISPLAY_AGENT_NAMES["pid_expert"]:
        tune_result = None
        for tool in tools:
            if tool.get("tool_name") == "tool_tune_pid" and isinstance(tool.get("result"), dict):
                tune_result = tool["result"]

        if tune_result is None:
            tune_result = latest_result if latest_tool_name == "tool_tune_pid" else {}

        response = (
            f"?? {tune_result.get('strategy_used', tune_result.get('strategy', '??'))} ???????"
            f"Kp={_format_float(tune_result.get('Kp'))}?Ki={_format_float(tune_result.get('Ki'))}?"
            f"Kd={_format_float(tune_result.get('Kd'))}?"
        )
        if tune_result.get("selection_reason"):
            response += f" ???????{tune_result.get('selection_reason')}"
        return response

    if agent_name == DISPLAY_AGENT_NAMES["evaluation_expert"]:
        extra = ""
        if not latest_result.get("passed") and latest_result.get("feedback_target"):
            extra = (
                f" ??????{latest_result.get('failure_reason', '')}"
                f" ???????? {latest_result.get('feedback_target')}?"
                f"{latest_result.get('feedback_action', '')}"
            )
        return (
            f"??????????? {_format_float(latest_result.get('performance_score'), 2)}?"
            f"????? {_format_float(latest_result.get('method_confidence'), 2)}?"
            f"???? {_format_float(latest_result.get('final_rating'), 2)}?"
            f"{'??' if latest_result.get('passed') else '???'}???????{extra}"
        )

    return ""

def _finalize_agent_turn(current_turn_data: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if current_turn_data is None:
        return None

    existing_response = (current_turn_data.get("response") or "").strip()
    generated = _build_agent_response(current_turn_data.get("agent", ""), current_turn_data.get("tools", []))
    latest_result = None
    for tool in reversed(current_turn_data.get("tools", [])):
        if isinstance(tool.get("result"), dict):
            latest_result = tool.get("result")
            break

    force_generated = (
        current_turn_data.get("agent") == DISPLAY_AGENT_NAMES["evaluation_expert"]
        and isinstance(latest_result, dict)
        and (latest_result.get("feedback_target") or latest_result.get("passed") is False)
    )
    if force_generated:
        current_turn_data["response"] = generated or existing_response
        return current_turn_data
    if not existing_response or existing_response in {"完成", "APPROVE"}:
        current_turn_data["response"] = generated or existing_response
    elif generated and len(existing_response) < 12:
        current_turn_data["response"] = f"{existing_response}\n{generated}"

    return current_turn_data


def _build_feedback_turns(shared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    initial_assessment = shared_data.get("initial_assessment") or {}
    if not initial_assessment or initial_assessment.get("passed", True):
        return []

    turns: List[Dict[str, Any]] = []
    auto_refine_result = shared_data.get("auto_refine_result") or {}
    if auto_refine_result.get("applied"):
        selection_inputs = shared_data.get("selection_inputs") or {}
        strategy_used = str(shared_data.get("strategy_used", ""))
        turns.append({
            "type": "agent_turn",
            "agent": DISPLAY_AGENT_NAMES["pid_expert"],
            "tools": [{
                "tool_name": "tool_tune_pid",
                "args": {
                    "K": selection_inputs.get("K", shared_data.get("K", 0.0)),
                    "T": selection_inputs.get("T", shared_data.get("T", 0.0)),
                    "L": selection_inputs.get("L", shared_data.get("L", 0.0)),
                    "loop_type": selection_inputs.get("loop_type", shared_data.get("loop_type", "flow")),
                    "phase": "feedback_refine",
                    "base_strategy": strategy_used,
                },
                "result": _to_jsonable({
                    "Kp": auto_refine_result.get("Kp", 0.0),
                    "Ki": auto_refine_result.get("Ki", 0.0),
                    "Kd": auto_refine_result.get("Kd", 0.0),
                    "strategy_used": strategy_used,
                    "performance_score": auto_refine_result.get("refined_performance_score", 0.0),
                    "final_rating": auto_refine_result.get("refined_final_rating", 0.0),
                    "base_final_rating": auto_refine_result.get("base_final_rating", 0.0),
                }),
            }],
            "response": (
                "根据首次评估反馈，已自动回流 PID 专家继续细调参数，"
                f"将 final_rating 从 {float(auto_refine_result.get('base_final_rating', 0.0)):.2f} "
                f"提升到 {float(auto_refine_result.get('refined_final_rating', 0.0)):.2f}。"
            ),
        })

    model_retry_result = shared_data.get("model_retry_result") or {}
    if model_retry_result.get("applied"):
        turns.append({
            "type": "agent_turn",
            "agent": DISPLAY_AGENT_NAMES["system_id_expert"],
            "tools": [{
                "tool_name": "tool_fit_fopdt",
                "args": {"phase": "feedback_retry_window"},
                "result": _to_jsonable(model_retry_result),
            }],
            "response": (
                f"PID 细调后仍未达标，已自动回流系统辨识智能体并切换到候选窗口 "
                f"{model_retry_result.get('window_source', '-')}"
                "，重新辨识模型并生成新的整定结果。"
            ),
        })

    final_rating = float(shared_data.get("final_rating", 0.0) or 0.0)
    performance_score = float(shared_data.get("performance_score", 0.0) or 0.0)
    passed = bool(shared_data.get("passed", False))
    if auto_refine_result.get("applied") or model_retry_result.get("applied"):
        turns.append({
            "type": "agent_turn",
            "agent": DISPLAY_AGENT_NAMES["evaluation_expert"],
            "tools": [{
                "tool_name": "tool_evaluate_pid",
                "args": {"phase": "post_feedback_evaluation"},
                "result": _to_jsonable({
                    "passed": passed,
                    "performance_score": performance_score,
                    "final_rating": final_rating,
                    "feedback_target": shared_data.get("feedback_target", ""),
                    "failure_reason": shared_data.get("failure_reason", ""),
                }),
            }],
            "response": (
                f"自动回流后已完成重新评估，当前 performance_score={performance_score:.2f}，"
                f"final_rating={final_rating:.2f}，结果为{'通过' if passed else '未通过'}。"
            ),
        })

    return turns


def _extract_candidate_windows() -> List[Dict[str, Any]]:
    cleaned_df = _shared_data_store.get("cleaned_df")
    candidate_windows = _shared_data_store.get("candidate_windows") or []
    candidates: List[Dict[str, Any]] = []

    if cleaned_df is not None:
        for idx, event in enumerate(candidate_windows):
            start_idx = int(event.get("window_start_idx", 0))
            end_idx = int(event.get("window_end_idx", 0))
            candidate_df = cleaned_df.iloc[start_idx:end_idx].reset_index(drop=True)
            if len(candidate_df) >= 10:
                candidates.append({
                    "name": f"step_event_{idx + 1}",
                    "df": candidate_df,
                    "event": event,
                })

    if cleaned_df is not None and len(cleaned_df) >= 10:
        candidates.append({"name": "full_cleaned", "df": cleaned_df})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        event = candidate.get("event") or {}
        key = (
            candidate["name"],
            len(candidate["df"]),
            int(event.get("window_start_idx", 0)),
            int(event.get("window_end_idx", len(candidate["df"]))),
        )
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _benchmark_pid_strategies(K: float, T: float, L: float, dt: float, confidence_score: float) -> Dict[str, Any]:
    best_candidate: Dict[str, Any] | None = None
    best_evaluation: Dict[str, Any] | None = None
    summaries: List[Dict[str, Any]] = []
    for strategy_name in ["IMC", "LAMBDA", "ZN", "CHR"]:
        pid_params = apply_tuning_rules(K, T, L, strategy_name)
        eval_result = ModelRating.evaluate(
            model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
            pid_params={"Kp": pid_params["Kp"], "Ki": pid_params["Ki"], "Kd": pid_params["Kd"]},
            method=strategy_name.lower(),
            method_confidence=confidence_score,
            method_confidence_details={"source": "window_identification_confidence"},
            dt=dt,
        )
        summary = {
            "strategy": strategy_name,
            "Kp": float(pid_params["Kp"]),
            "Ki": float(pid_params["Ki"]),
            "Kd": float(pid_params["Kd"]),
            "performance_score": float(eval_result["performance_score"]),
            "final_rating": float(eval_result.get("final_rating", 0.0)),
            "is_stable": bool(eval_result["simulation"].get("is_stable", False)),
        }
        summaries.append(summary)
        if best_candidate is None or summary["performance_score"] > best_candidate["performance_score"] + 1e-9 or (
            abs(summary["performance_score"] - best_candidate["performance_score"]) <= 1e-9
            and summary["final_rating"] > best_candidate["final_rating"] + 1e-9
        ):
            best_candidate = summary
            best_evaluation = eval_result

    return {
        "best": best_candidate or {},
        "all": summaries,
        "best_evaluation": best_evaluation or {},
    }


def _refine_pid_for_performance(
    model_params: Dict[str, float],
    base_pid_params: Dict[str, float],
    method_confidence: float,
    dt: float,
    base_strategy: str,
) -> Dict[str, Any]:
    kp_scales = [1.0, 0.85, 0.7, 0.55, 0.4]
    ki_scales = [1.0, 0.7, 0.5, 0.35, 0.2]
    kd_scales = [1.0, 0.7, 0.4] if abs(float(base_pid_params.get("Kd", 0.0))) > 1e-9 else [1.0]

    best: Dict[str, Any] | None = None
    candidates: List[Dict[str, Any]] = []

    for kp_scale in kp_scales:
        for ki_scale in ki_scales:
            for kd_scale in kd_scales:
                pid_candidate = {
                    "Kp": float(base_pid_params["Kp"]) * kp_scale,
                    "Ki": float(base_pid_params["Ki"]) * ki_scale,
                    "Kd": float(base_pid_params["Kd"]) * kd_scale,
                }
                eval_result = ModelRating.evaluate(
                    model_params=model_params,
                    pid_params=pid_candidate,
                    method=f"{base_strategy.lower()}_refined",
                    method_confidence=method_confidence,
                    method_confidence_details={"source": "auto_refine"},
                    dt=dt,
                )
                summary = {
                    "Kp": pid_candidate["Kp"],
                    "Ki": pid_candidate["Ki"],
                    "Kd": pid_candidate["Kd"],
                    "kp_scale": kp_scale,
                    "ki_scale": ki_scale,
                    "kd_scale": kd_scale,
                    "performance_score": float(eval_result["performance_score"]),
                    "final_rating": float(eval_result.get("final_rating", 0.0)),
                    "is_stable": bool(eval_result["simulation"].get("is_stable", False)),
                    "evaluation_result": eval_result,
                }
                candidates.append(summary)
                if best is None:
                    best = summary
                    continue
                better_score = summary["final_rating"] > best["final_rating"] + 1e-9
                tie_break = (
                    abs(summary["final_rating"] - best["final_rating"]) <= 1e-9
                    and summary["performance_score"] > best["performance_score"] + 1e-9
                )
                stable_break = (
                    abs(summary["final_rating"] - best["final_rating"]) <= 1e-9
                    and abs(summary["performance_score"] - best["performance_score"]) <= 1e-9
                    and summary["is_stable"]
                    and not best["is_stable"]
                )
                if better_score or tie_break or stable_break:
                    best = summary

    return {
        "best": best or {},
        "candidates": [
            {
                "Kp": item["Kp"],
                "Ki": item["Ki"],
                "Kd": item["Kd"],
                "kp_scale": item["kp_scale"],
                "ki_scale": item["ki_scale"],
                "kd_scale": item["kd_scale"],
                "performance_score": item["performance_score"],
                "final_rating": item["final_rating"],
                "is_stable": item["is_stable"],
            }
            for item in candidates
        ],
    }


def _try_alternative_model_attempts(pass_threshold: float) -> Dict[str, Any]:
    attempts = _shared_data_store.get("model_attempts") or []
    if len(attempts) <= 1:
        return {}

    current_source = str(_shared_data_store.get("model_selected_source", ""))
    candidate_map = {candidate["name"]: candidate for candidate in _extract_candidate_windows()}
    loop_type = str(_shared_data_store.get("loop_type", "flow"))
    dt = float(_shared_data_store.get("dt", 1.0))

    best_result: Dict[str, Any] | None = None
    for attempt in attempts:
        source = str(attempt.get("window_source", ""))
        if source == current_source or source not in candidate_map:
            continue

        K = float(attempt["K"])
        T = float(attempt["T"])
        L = float(attempt["L"])
        confidence_score = float(attempt.get("confidence", 0.0))
        benchmark = _benchmark_pid_strategies(K, T, L, dt, confidence_score)
        best_strategy = benchmark.get("best") or {}
        if not best_strategy:
            continue

        refined = _refine_pid_for_performance(
            model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
            base_pid_params={
                "Kp": float(best_strategy["Kp"]),
                "Ki": float(best_strategy["Ki"]),
                "Kd": float(best_strategy["Kd"]),
            },
            method_confidence=confidence_score,
            dt=dt,
            base_strategy=str(best_strategy.get("strategy", "auto")),
        )
        refined_best = refined.get("best") or {}
        final_eval = refined_best.get("evaluation_result") if refined_best else benchmark.get("best_evaluation") or {}
        if not final_eval:
            continue

        result = {
            "window_source": source,
            "loop_type": loop_type,
            "K": K,
            "T": T,
            "L": L,
            "confidence": confidence_score,
            "strategy": str(best_strategy.get("strategy", "")),
            "evaluation_result": final_eval,
            "Kp": float(refined_best.get("Kp", best_strategy["Kp"])),
            "Ki": float(refined_best.get("Ki", best_strategy["Ki"])),
            "Kd": float(refined_best.get("Kd", best_strategy["Kd"])),
            "passed": float(final_eval.get("final_rating", 0.0)) >= pass_threshold,
        }

        if best_result is None:
            best_result = result
            continue

        better_score = float(result["evaluation_result"]["final_rating"]) > float(best_result["evaluation_result"]["final_rating"]) + 1e-9
        tie_break = (
            abs(float(result["evaluation_result"]["final_rating"]) - float(best_result["evaluation_result"]["final_rating"])) <= 1e-9
            and float(result["evaluation_result"]["performance_score"]) > float(best_result["evaluation_result"]["performance_score"]) + 1e-9
        )
        if better_score or tie_break:
            best_result = result

    return best_result or {}


def _derive_model_reason_codes(model_params: Dict[str, Any], confidence: Dict[str, Any], quality_metrics: Dict[str, Any] | None) -> List[str]:
    reason_codes: List[str] = []
    residue = _safe_float(model_params.get("normalized_rmse", model_params.get("residue")))
    r2_score = _safe_float(model_params.get("r2_score"))
    confidence_score = _safe_float(confidence.get("confidence"))
    T = _safe_float(model_params.get("T"))
    L = _safe_float(model_params.get("L"))
    overshoot = _safe_float((quality_metrics or {}).get("overshoot_percent"))
    settling_time = _safe_float((quality_metrics or {}).get("settling_time"), default=-1.0)

    # Good-fit band: do not surface risk flags for models that are already reliable.
    if confidence_score >= 0.8 and r2_score >= 0.9 and residue <= 0.08:
        return []

    if residue > 0.1:
        reason_codes.append("残差偏高")
    if r2_score < 0.6:
        reason_codes.append("拟合解释度偏低")
    if confidence_score < 0.55:
        reason_codes.append("模型置信度偏低")
    if T <= 2.0 and confidence_score < 0.7 and residue > 0.08:
        reason_codes.append("动态较快或采样粒度偏粗")
    if L <= 0.0 and confidence_score < 0.7 and r2_score < 0.9:
        reason_codes.append("未观察到明显死区")
    if overshoot > 20:
        reason_codes.append("窗口内响应偏激进")
    if settling_time > 0 and settling_time < 3:
        reason_codes.append("辨识窗口可能偏短")
    return reason_codes


def _derive_next_actions(confidence_score: float, reason_codes: List[str]) -> List[str]:
    actions: List[str] = []
    if not reason_codes:
        return actions

    if "残差偏高" in reason_codes or "辨识窗口可能偏短" in reason_codes:
        actions.append("尝试其他辨识窗口")
    if "拟合解释度偏低" in reason_codes:
        actions.append("确认对象是否偏离FOPDT假设")
    if "动态较快或采样粒度偏粗" in reason_codes or "未观察到明显死区" in reason_codes:
        actions.append("检查采样周期或补采更高频数据")
    if confidence_score < 0.5:
        actions.append("采用更保守的整定策略")
    if confidence_score < 0.35:
        actions.append("建议重新采集阶跃试验数据")
    return actions


def _diagnose_evaluation_failure(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    performance_details = eval_result.get("performance_details") or {}
    performance_score = _safe_float(eval_result.get("performance_score"))
    final_rating = _safe_float(eval_result.get("final_rating"))
    method_confidence = _safe_float(eval_result.get("method_confidence"))
    overshoot = _safe_float(performance_details.get("overshoot"))
    settling_time = _safe_float(performance_details.get("settling_time"), default=-1.0)
    steady_state_error = _safe_float(performance_details.get("steady_state_error"))
    oscillation_count = int(performance_details.get("oscillation_count", 0) or 0)
    decay_ratio = _safe_float(performance_details.get("decay_ratio"))
    is_stable = bool(performance_details.get("is_stable", True))
    model_r2 = _safe_float(_shared_data_store.get("r2_score"))
    model_rmse = _safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue")))
    candidate_window_count = len(_shared_data_store.get("candidate_windows") or [])

    if not is_stable or overshoot > 40 or oscillation_count > 20 or decay_ratio >= 0.8:
        return {
            "failure_reason": "当前整定参数偏激进，闭环振荡和超调仍然过大。",
            "feedback_target": "pid_expert",
            "feedback_action": "请在当前模型基础上继续收紧Kp和Ki，优先压低超调和振荡。",
        }

    if method_confidence < 0.7 or model_r2 < 0.8 or model_rmse > 0.1:
        return {
            "failure_reason": "当前模型辨识可信度不足，参数整定建立在不够稳的模型上。",
            "feedback_target": "system_id_expert",
            "feedback_action": "请复核当前辨识结果，优先比较候选窗口并重新确认K/T/L。",
        }

    if candidate_window_count > 1 and performance_score < 5.0:
        return {
            "failure_reason": "当前辨识窗口对整定不够友好，候选窗口中仍可能存在更合适的区间。",
            "feedback_target": "data_analyst",
            "feedback_action": "请重新审视候选阶跃窗口，优先选择响应更完整、扰动更少的区间。",
        }

    if settling_time < 0 or steady_state_error > 5.0:
        return {
            "failure_reason": "闭环响应收敛偏慢或稳态误差偏大，当前参数需要进一步校正。",
            "feedback_target": "pid_expert",
            "feedback_action": "请在当前模型基础上继续优化积分和比例参数，提高收敛质量。",
        }

    return {
        "failure_reason": "综合评分未达到阈值，建议先从PID参数细调开始，再视结果决定是否回退到模型或数据层。",
        "feedback_target": "pid_expert",
        "feedback_action": "请基于当前模型做一轮更保守的参数再优化。",
    }


def _build_fit_preview(window_df: Any, model_params: Dict[str, Any], max_points: int = 200) -> Dict[str, Any]:
    if window_df is None or len(window_df) == 0:
        return {"points": []}

    dt = float(_shared_data_store.get("dt", 1.0))
    pv = window_df["PV"].to_numpy(dtype=float)
    mv = window_df["MV"].to_numpy(dtype=float)

    n = len(window_df)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    K = float(model_params["K"])
    T = float(model_params["T"])
    L = float(model_params["L"])
    delay_steps = int(round(max(L, 0.0) / max(dt, 1e-6)))
    alpha = dt / (max(T, 1e-6) + dt)
    mv_delta = mv - mv[0]
    simulated_delta = []
    y = 0.0
    for i in range(n):
        delayed_u = mv_delta[i - delay_steps] if i >= delay_steps else 0.0
        y = (1.0 - alpha) * y + K * alpha * delayed_u
        simulated_delta.append(y)
    pv_fit = pv[0] + np.asarray(simulated_delta)

    timestamp_strings = None
    if "timestamp" in window_df.columns:
        timestamp_strings = window_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    points = []
    for i in indices:
        point = {
            "index": int(i),
            "pv": float(pv[i]),
            "pv_fit": float(pv_fit[i]),
            "mv": float(mv[i]),
        }
        if timestamp_strings is not None:
            point["time"] = timestamp_strings[i]
        points.append(point)

    return {
        "points": points,
        "x_axis": "timestamp" if timestamp_strings is not None else "index",
        "start_time": timestamp_strings[0] if timestamp_strings is not None else None,
        "end_time": timestamp_strings[-1] if timestamp_strings is not None else None,
    }


def _build_window_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    max_points: int = 240,
) -> Dict[str, Any]:
    if cleaned_df is None or len(cleaned_df) == 0:
        return {"points": [], "window_start": 0, "window_end": 0}

    pv = cleaned_df["PV"].to_numpy(dtype=float)
    mv = cleaned_df["MV"].to_numpy(dtype=float)
    n = len(cleaned_df)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    window_start = int((selected_window or {}).get("start_index", 0))
    window_end = int((selected_window or {}).get("end_index", n - 1))
    window_start = max(0, min(window_start, n - 1))
    window_end = max(window_start, min(window_end, n - 1))

    timestamp_strings = None
    if "timestamp" in cleaned_df.columns:
        timestamp_strings = cleaned_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    points = []
    for i in indices:
        point = {
            "index": int(i),
            "pv": float(pv[i]),
            "mv": float(mv[i]),
            "in_window": bool(window_start <= i <= window_end),
        }
        if timestamp_strings is not None:
            point["time"] = timestamp_strings[i]
        points.append(point)

    return {
        "points": points,
        "window_start": window_start,
        "window_end": window_end,
        "total_points": int(n),
        "x_axis": "timestamp" if timestamp_strings is not None else "index",
        "start_time": timestamp_strings[0] if timestamp_strings is not None else None,
        "end_time": timestamp_strings[-1] if timestamp_strings is not None else None,
        "window_start_time": timestamp_strings[window_start] if timestamp_strings is not None else None,
        "window_end_time": timestamp_strings[window_end] if timestamp_strings is not None else None,
    }

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
    candidate_windows = prepared.get("candidate_windows") or []
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
    _shared_data_store["candidate_windows"] = candidate_windows
    _shared_data_store["selected_event"] = selected_event
    _shared_data_store["quality_metrics"] = quality_metrics

    selected_window = {
        "rows": int(len(window_df)),
        "start_index": int(selected_event.get("window_start_idx", selected_event.get("start_idx", 0))) if selected_event else 0,
        "end_index": int(selected_event.get("window_end_idx", selected_event.get("end_idx", int(len(window_df))))) if selected_event else int(len(window_df)),
        "event_start_index": int(selected_event["start_idx"]) if selected_event else 0,
        "event_end_index": int(selected_event["end_idx"]) if selected_event else int(len(window_df)),
        "event_type": str(selected_event.get("type", "full_range")) if selected_event else "full_range",
    }
    window_overview = _build_window_overview(cleaned_df, selected_window)
    _shared_data_store["window_overview"] = window_overview

    return _to_jsonable({
        "data_points": int(len(cleaned_df)),
        "window_points": int(len(window_df)),
        "sampling_time": dt,
        "mv_range": [float(cleaned_df["MV"].min()), float(cleaned_df["MV"].max())],
        "pv_range": [float(cleaned_df["PV"].min()), float(cleaned_df["PV"].max())],
        "available_columns": [str(col) for col in cleaned_df.columns.tolist()],
        "step_events": step_events,
        "candidate_windows": candidate_windows,
        "selected_window": selected_window,
        "window_overview": window_overview,
        "quality_metrics": quality_metrics,
        "status": "数据已完成清洗、降噪和辨识窗口选择",
    })


async def tool_fit_fopdt(dt: float = 1.0) -> Dict[str, Any]:
    """基于预处理后的 MV/PV 窗口拟合 FOPDT 模型。"""
    if "mv" not in _shared_data_store or "pv" not in _shared_data_store:
        raise ValueError("请先调用tool_load_data加载数据")

    actual_dt = float(_shared_data_store.get("dt", dt))
    quality_metrics = _shared_data_store.get("quality_metrics") or {}
    attempts: List[Dict[str, Any]] = []
    best_model_params: Dict[str, Any] | None = None
    best_confidence: Dict[str, Any] | None = None
    best_benchmark: Dict[str, Any] | None = None
    best_candidate_df: Any = None
    best_event: Dict[str, Any] | None = None
    best_source = ""
    loop_type = str(_shared_data_store.get("loop_type", "flow"))

    for candidate in _extract_candidate_windows():
        candidate_df = candidate["df"]
        mv_array = candidate_df["MV"].to_numpy(dtype=float)
        pv_array = candidate_df["PV"].to_numpy(dtype=float)
        model_params = await asyncio.to_thread(fit_fopdt_model, mv_array, pv_array, actual_dt)
        confidence = calculate_model_confidence(model_params["normalized_rmse"], model_params.get("r2_score"))
        benchmark = _benchmark_pid_strategies(
            float(model_params["K"]),
            float(model_params["T"]),
            float(model_params["L"]),
            actual_dt,
            float(confidence["confidence"]),
        )
        best_strategy = benchmark["best"] or {}
        attempt_result = {
            "window_source": candidate["name"],
            "points": int(len(candidate_df)),
            "K": float(model_params["K"]),
            "T": float(model_params["T"]),
            "L": float(model_params["L"]),
            "residue": float(model_params["residue"]),
            "normalized_rmse": float(model_params["normalized_rmse"]),
            "raw_rmse": float(model_params["raw_rmse"]),
            "r2_score": float(model_params["r2_score"]),
            "confidence": float(confidence["confidence"]),
            "confidence_quality": confidence["quality"],
            "benchmark_strategy": best_strategy.get("strategy", ""),
            "benchmark_performance_score": float(best_strategy.get("performance_score", 0.0)),
            "benchmark_final_rating": float(best_strategy.get("final_rating", 0.0)),
            "benchmark_stable": bool(best_strategy.get("is_stable", False)),
            "success": bool(model_params["success"]),
        }
        if candidate.get("event"):
            attempt_result["window_start_index"] = int(candidate["event"].get("window_start_idx", 0))
            attempt_result["window_end_index"] = int(candidate["event"].get("window_end_idx", len(candidate_df)))
            attempt_result["event_type"] = str(candidate["event"].get("type", ""))
        attempts.append(attempt_result)
        if best_model_params is None:
            best_model_params = model_params
            best_confidence = confidence
            best_benchmark = benchmark
            best_candidate_df = candidate_df
            best_event = candidate.get("event")
            best_source = candidate["name"]
            continue

        current_score = float(best_strategy.get("performance_score", 0.0))
        best_score = float((best_benchmark or {}).get("best", {}).get("performance_score", 0.0))
        better_score = current_score > best_score + 1e-9
        tie_break = (
            abs(current_score - best_score) <= 1e-9
            and float(confidence["confidence"]) > _safe_float((best_confidence or {}).get("confidence")) + 1e-9
        )
        if better_score or tie_break:
            best_model_params = model_params
            best_confidence = confidence
            best_benchmark = benchmark
            best_candidate_df = candidate_df
            best_event = candidate.get("event")
            best_source = candidate["name"]

    if best_model_params is None or best_confidence is None:
        raise ValueError("未能完成 FOPDT 模型辨识")

    reason_codes = _derive_model_reason_codes(best_model_params, best_confidence, quality_metrics)
    next_actions = _derive_next_actions(_safe_float(best_confidence.get("confidence")), reason_codes)
    fit_preview = _build_fit_preview(best_candidate_df, best_model_params)

    if best_candidate_df is not None:
        _shared_data_store["window_df"] = best_candidate_df
        _shared_data_store["mv"] = best_candidate_df["MV"].to_numpy(dtype=float)
        _shared_data_store["pv"] = best_candidate_df["PV"].to_numpy(dtype=float)

    selected_window_payload = None
    if best_event:
        _shared_data_store["selected_event"] = best_event
        selected_window = {
            "rows": int(len(best_candidate_df)),
            "start_index": int(best_event.get("window_start_idx", 0)),
            "end_index": int(best_event.get("window_end_idx", len(best_candidate_df))),
            "event_start_index": int(best_event.get("start_idx", 0)),
            "event_end_index": int(best_event.get("end_idx", len(best_candidate_df))),
            "event_type": str(best_event.get("type", "full_range")),
        }
        selected_window_payload = selected_window
        _shared_data_store["selected_window"] = selected_window
        _shared_data_store["window_overview"] = _build_window_overview(_shared_data_store.get("cleaned_df"), selected_window)

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
    """应用多种 PID 整定规则并选择闭环性能最优的一组。"""
    confidence_score = _safe_float((_shared_data_store.get("model_confidence") or {}).get("confidence"), 1.0)
    tau_ratio = max(_safe_float(L), 0.0) / max(_safe_float(T), 1e-6)
    selected_model = {
        "normalized_rmse": _safe_float(_shared_data_store.get("normalized_rmse"), _safe_float(_shared_data_store.get("residue"))),
        "r2_score": _safe_float(_shared_data_store.get("r2_score")),
    }
    heuristic_selection = select_tuning_strategy(
        loop_type=loop_type,
        K=K,
        T=T,
        L=L,
        model_confidence=confidence_score,
        r2_score=selected_model["r2_score"],
        normalized_rmse=selected_model["normalized_rmse"],
    )
    candidate_strategies = ["IMC", "LAMBDA", "ZN", "CHR"]
    candidate_results: List[Dict[str, Any]] = []
    best_candidate: Dict[str, Any] | None = None

    for strategy_name in candidate_strategies:
        pid_params = apply_tuning_rules(K, T, L, strategy_name)
        eval_result = ModelRating.evaluate(
            model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
            pid_params={"Kp": pid_params["Kp"], "Ki": pid_params["Ki"], "Kd": pid_params["Kd"]},
            method=strategy_name.lower(),
            method_confidence=confidence_score,
            method_confidence_details={"source": "model_identification_confidence"},
            dt=float(_shared_data_store.get("dt", 1.0)),
        )
        candidate = {
            "strategy": strategy_name,
            "Kp": float(pid_params["Kp"]),
            "Ki": float(pid_params["Ki"]),
            "Kd": float(pid_params["Kd"]),
            "Ti": float(pid_params["Ti"]),
            "Td": float(pid_params["Td"]),
            "description": str(pid_params["description"]),
            "performance_score": float(eval_result["performance_score"]),
            "final_rating": float(eval_result.get("final_rating", 0.0)),
            "is_stable": bool(eval_result["simulation"].get("is_stable", False)),
            "evaluation_result": eval_result,
        }
        candidate_results.append(candidate)

        if best_candidate is None:
            best_candidate = candidate
            continue

        better_score = candidate["performance_score"] > best_candidate["performance_score"] + 1e-9
        tie_break = (
            abs(candidate["performance_score"] - best_candidate["performance_score"]) <= 1e-9
            and candidate["final_rating"] > best_candidate["final_rating"] + 1e-9
        )
        stable_break = (
            abs(candidate["performance_score"] - best_candidate["performance_score"]) <= 1e-9
            and abs(candidate["final_rating"] - best_candidate["final_rating"]) <= 1e-9
            and candidate["is_stable"]
            and not best_candidate["is_stable"]
        )
        if better_score or tie_break or stable_break:
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError("未能生成可用的 PID 候选参数")

    pid_params = apply_tuning_rules(K, T, L, best_candidate["strategy"])
    _shared_data_store["pid_params"] = pid_params
    public_candidate_results = [
        {
            "strategy": item["strategy"],
            "Kp": item["Kp"],
            "Ki": item["Ki"],
            "Kd": item["Kd"],
            "Ti": item["Ti"],
            "Td": item["Td"],
            "description": item["description"],
            "performance_score": item["performance_score"],
            "final_rating": item["final_rating"],
            "is_stable": item["is_stable"],
        }
        for item in candidate_results
    ]

    _shared_data_store["pid_candidate_results"] = public_candidate_results
    _shared_data_store["strategy_used"] = best_candidate["strategy"]
    _shared_data_store["selection_reason"] = (
        f"已对 {', '.join(candidate_strategies)} 进行闭环试算，"
        f"最终选择 performance_score 最高的 {best_candidate['strategy']}。"
    )
    _shared_data_store["selection_inputs"] = {
        "loop_type": loop_type,
        "model_confidence": confidence_score,
        "normalized_rmse": selected_model["normalized_rmse"],
        "r2_score": selected_model["r2_score"],
        "tau_ratio": tau_ratio,
        "K": float(K),
        "T": float(T),
        "L": float(L),
        "heuristic_strategy": heuristic_selection["strategy"],
    }
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
    """评估 PID 整定结果。"""
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
    base_eval_result = eval_result
    _shared_data_store["evaluation_result"] = eval_result
    pass_threshold = 7.0
    passed = bool(_safe_float(eval_result.get("final_rating")) >= pass_threshold)
    diagnosis = _diagnose_evaluation_failure(eval_result) if not passed else {
        "failure_reason": "",
        "feedback_target": "",
        "feedback_action": "",
    }
    initial_assessment = {
        "passed": passed,
        "pass_threshold": pass_threshold,
        "failure_reason": diagnosis["failure_reason"],
        "feedback_target": diagnosis["feedback_target"],
        "feedback_action": diagnosis["feedback_action"],
        "evaluation_result": {
            "performance_score": float(base_eval_result.get("performance_score", 0.0)),
            "method_confidence": float(base_eval_result.get("method_confidence", 0.0)),
            "final_rating": float(base_eval_result.get("final_rating", 0.0)),
        },
        "evaluated_pid": {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    }
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
                diagnosis = _diagnose_evaluation_failure(eval_result) if not passed else {
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
        alternative_model = _try_alternative_model_attempts(pass_threshold)
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
                diagnosis = _diagnose_evaluation_failure(eval_result) if not passed else {
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
    loop_type: str
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
        system_message=f"""?? PID ?????

????????
1. ??????? tool_fit_fopdt ??? K?T?L?
2. ?? tool_tune_pid(K=..., T=..., L=..., loop_type="{loop_type}")?
3. ???????????????? PID ????????????""",
        tools=[tool_tune_pid],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message="""???????

????????
1. ??????? K?T?L?Kp?Ki?Kd?
2. ?? tool_evaluate_pid(K=..., T=..., L=..., Kp=..., Ki=..., Kd=..., method="auto")?
3. ?? passed=true????????????????????????????? APPROVE?
4. ?? passed=false????????????????feedback_target ? feedback_action????? APPROVE?""",
        tools=[tool_evaluate_pid],
        model_client_stream=False,
    )

    return [data_analyst, system_id_expert, pid_expert, evaluation_expert]


async def run_multi_agent_collaboration(
    csv_path: str,
    loop_name: str,
    loop_type: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    llm_config: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """运行多智能体协作 - 使用RoundRobinGroupChat"""

    # 清空全局数据存储
    _shared_data_store.clear()
    _shared_data_store["loop_type"] = loop_type

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
        loop_type=loop_type
    )

    # 创建终止条件：评估智能体说"APPROVE"
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(12)

    # 创建RoundRobinGroupChat团队
    team = RoundRobinGroupChat(
        participants=agents,
        termination_condition=termination,
    )

    # 构建初始任务消息
    task_message = f"""?????? {loop_name} ??PID?????

????: {csv_path}
??????URI: {loop_uri}
????????: {start_time or "????24??"}
????????: {end_time or "????"}
??????: {data_type}
????: {loop_type}

???????????
1. ???????????????
2. ??????????FOPDT??
3. PID????????PID??
4. ????????????

???????????????????????????"""

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
                        yield _finalize_agent_turn(current_turn_data)
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
                            current_tool_name = ""
                            if current_turn_data and current_turn_data["tools"]:
                                current_tool_name = current_turn_data["tools"][-1].get("tool_name", "")

                            if current_tool_name == "tool_evaluate_pid" and result_data.get("initial_assessment"):
                                initial_assessment = result_data.get("initial_assessment") or {}
                                initial_eval_result = initial_assessment.get("evaluation_result") or {}
                                display_result = {
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
                            else:
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
                    yield _finalize_agent_turn(current_turn_data)
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
        for feedback_turn in _build_feedback_turns(shared_data):
            yield feedback_turn
            await asyncio.sleep(0.2)

        step_events = shared_data.get("step_events") or []
        effective_pid_params = _shared_data_store.get("selected_pid_params") or {}
        final_result = {
            "dataAnalysis": {
                "dataPoints": shared_data.get("data_points", 0),
                "windowPoints": shared_data.get("window_points", 0),
                "stepEvents": len(step_events),
                "stepEventDetails": step_events,
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
                "candidateStrategies": shared_data.get("candidate_strategies", []),
                "description": effective_pid_params.get("description", shared_data.get("description", "")),
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
