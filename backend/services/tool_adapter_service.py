from __future__ import annotations

import ast
import json
import numpy as np
from typing import Any, Callable, Dict, Mapping

from memory.experience_service import retrieve_experience_guidance
from services.identification_service import sanitize_selected_model_params
from services.system_config_service import is_experience_distillation_enabled
from state.task_artifacts import persist_processed_csv
from services.knowledge_graph_service import (
    build_knowledge_context,
    compact_knowledge_guidance,
    merge_knowledge_guidance,
    query_knowledge_graph_api,
    search_distillation_rules,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_model_params(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _sanitize_selected_model_params(model_type: str, value: Any) -> Dict[str, Any]:
    params = _coerce_model_params(value)
    if not params:
        return {}
    return sanitize_selected_model_params(model_type or str(params.get("model_type", "")), params)


def _derive_tuning_metrics(model_type: str, selected_model_params: Dict[str, Any], fallback: Mapping[str, Any] | None = None) -> Dict[str, float]:
    fallback = fallback or {}
    normalized_model_type = str(model_type or selected_model_params.get("model_type", fallback.get("model_type", "FOPDT"))).upper()
    params = _sanitize_selected_model_params(normalized_model_type, selected_model_params) or dict(selected_model_params or {})

    if normalized_model_type == "SOPDT":
        active_k = _safe_float(params.get("K"), _safe_float(fallback.get("K")))
        t1 = _safe_float(params.get("T1"), _safe_float(params.get("T"), _safe_float(fallback.get("T"), 1.0)))
        t2 = _safe_float(params.get("T2"), _safe_float(params.get("T"), _safe_float(fallback.get("T"), 1.0)))
        active_t = t1 + t2
        if active_t <= 0:
            active_t = _safe_float(fallback.get("T"), 1.0)
        active_l = _safe_float(params.get("L"), _safe_float(fallback.get("L")))
    elif normalized_model_type == "IPDT":
        active_k = _safe_float(params.get("K"), _safe_float(fallback.get("K")))
        active_l = _safe_float(params.get("L"), _safe_float(fallback.get("L"), 1.0))
        active_t = max(_safe_float(fallback.get("T"), 1.0), active_l, 1e-3)
    elif normalized_model_type == "FO":
        active_k = _safe_float(params.get("K"), _safe_float(fallback.get("K")))
        active_t = _safe_float(params.get("T"), _safe_float(fallback.get("T"), 1.0))
        active_l = 0.0
    else:
        active_k = _safe_float(params.get("K"), _safe_float(fallback.get("K")))
        active_t = _safe_float(params.get("T"), _safe_float(fallback.get("T"), 1.0))
        active_l = _safe_float(params.get("L"), _safe_float(fallback.get("L")))

    return {
        "model_type": normalized_model_type,
        "K": float(active_k),
        "T": float(active_t),
        "L": float(active_l),
        "selected_model_params": _sanitize_selected_model_params(normalized_model_type, params) or params,
    }


def _build_experience_guidance(
    *,
    loop_type: str,
    model_type: str,
    selected_model_params: Dict[str, Any],
    active_k: float,
    active_t: float,
    active_l: float,
    knowledge_preferred_strategy: str,
    knowledge_summary: str,
) -> Dict[str, Any]:
    if is_experience_distillation_enabled():
        experience_guidance = retrieve_experience_guidance(
            loop_type=loop_type,
            model_type=model_type,
            K=active_k,
            T=active_t,
            L=active_l,
            selected_model_params=selected_model_params,
            limit=3,
            candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
        )
    else:
        experience_guidance = {
            "matches": [],
            "summary": {"disabled": True},
            "guidance": "",
            "preferred_strategy": "",
            "preferred_model_type": "",
            "preferred_refine_pattern": "",
            "recommended_kp_scale": 1.0,
            "recommended_ki_scale": 1.0,
            "recommended_kd_scale": 1.0,
        }

    if knowledge_preferred_strategy:
        experience_guidance = {
            **experience_guidance,
            "preferred_strategy": knowledge_preferred_strategy,
            "guidance": "；".join(
                part
                for part in [
                    str(experience_guidance.get("guidance") or "").strip(),
                    knowledge_summary,
                ]
                if part
            ),
            "summary": {
                **dict(experience_guidance.get("summary") or {}),
                "preferred_strategy": knowledge_preferred_strategy,
                "preferred_model_type": model_type,
            },
        }
    return experience_guidance


def _is_better_tuning_selection(candidate: Dict[str, Any], best: Dict[str, Any] | None) -> bool:
    if best is None:
        return True

    current_best = best.get("best_candidate") or {}
    current = candidate.get("best_candidate") or {}
    current_score = _safe_float(current.get("performance_score")) + _safe_float(current.get("experience_bonus"))
    best_score = _safe_float(current_best.get("performance_score")) + _safe_float(current_best.get("experience_bonus"))
    if current_score > best_score + 1e-9:
        return True
    if abs(current_score - best_score) <= 1e-9:
        current_final = _safe_float(current.get("final_rating"))
        best_final = _safe_float(current_best.get("final_rating"))
        if current_final > best_final + 1e-9:
            return True
        if abs(current_final - best_final) <= 1e-9:
            current_fit = _safe_float(candidate.get("identification_fit_score"))
            best_fit = _safe_float(best.get("identification_fit_score"))
            if current_fit > best_fit + 1e-9:
                return True
    return False


def _simulation_saturation_pct(evaluation_result: Mapping[str, Any] | None) -> float:
    simulation = dict((evaluation_result or {}).get("simulation") or {})
    mv_history = simulation.get("mv_history") or []
    if not mv_history:
        return 0.0
    saturated = 0
    for value in mv_history:
        mv = _safe_float(value)
        if mv <= 0.5 or mv >= 99.5:
            saturated += 1
    return (saturated / float(len(mv_history))) * 100.0


def _build_shortlist_candidate(candidate_summary: Dict[str, Any]) -> Dict[str, Any]:
    best_candidate = dict(candidate_summary.get("best_candidate") or {})
    evaluation_result = dict(best_candidate.get("evaluation_result") or {})
    identification_fit_score = _safe_float(candidate_summary.get("identification_fit_score"))
    performance_score = _safe_float(candidate_summary.get("best_performance_score"))
    final_rating = _safe_float(candidate_summary.get("best_final_rating"))
    is_stable = bool(candidate_summary.get("is_stable", False))
    saturation_pct = _simulation_saturation_pct(evaluation_result)
    oscillation_count = int(((evaluation_result.get("performance_details") or {}).get("oscillation_count", 0)) or 0)

    reasons: list[str] = []
    shortlist_passed = True

    if identification_fit_score < 8.0:
        shortlist_passed = False
        reasons.append(f"辨识拟合评分 {identification_fit_score:.2f} 低于 8.0")
    if performance_score < 7.0:
        shortlist_passed = False
        reasons.append(f"试算性能 {performance_score:.2f} 低于 7.0")
    if final_rating < 7.0:
        shortlist_passed = False
        reasons.append(f"综合评分 {final_rating:.2f} 低于 7.0")
    if not is_stable:
        shortlist_passed = False
        reasons.append("闭环试算未稳定")
    if saturation_pct >= 35.0:
        shortlist_passed = False
        reasons.append(f"MV 饱和占比 {saturation_pct:.1f}% 过高")
    if oscillation_count > 20:
        shortlist_passed = False
        reasons.append(f"振荡次数 {oscillation_count} 过多")

    shortlist_score = round(
        0.3 * identification_fit_score + 0.4 * performance_score + 0.3 * final_rating,
        2,
    )
    if shortlist_passed and not reasons:
        reasons.append("满足拟合、试算性能、综合评分与稳定性入围条件")

    return {
        "model_type": candidate_summary.get("model_type"),
        "selected_model_params": dict(candidate_summary.get("selected_model_params") or {}),
        "K": _safe_float(candidate_summary.get("K")),
        "T": _safe_float(candidate_summary.get("T"), 1.0),
        "L": _safe_float(candidate_summary.get("L")),
        "window_source": str(candidate_summary.get("window_source", "")),
        "identification_fit_score": identification_fit_score,
        "normalized_rmse": _safe_float(candidate_summary.get("normalized_rmse")),
        "r2_score": _safe_float(candidate_summary.get("r2_score")),
        "model_confidence": _safe_float(candidate_summary.get("model_confidence")),
        "best_strategy": str(candidate_summary.get("best_strategy", "")),
        "best_performance_score": performance_score,
        "best_final_rating": final_rating,
        "is_stable": is_stable,
        "pid_params": dict(candidate_summary.get("pid_params") or {}),
        "shortlist_passed": shortlist_passed,
        "shortlist_reasons": reasons,
        "shortlist_score": shortlist_score,
        "saturation_pct": round(saturation_pct, 3),
        "evaluation_result": evaluation_result,
    }


def _summarize_step_events_for_llm(step_events: Any, *, limit: int = 6) -> list[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for event in list(step_events or [])[:limit]:
        if not isinstance(event, Mapping):
            continue
        summary.append(
            {
                "type": str(event.get("type", "")),
                "amplitude": round(_safe_float(event.get("amplitude")), 4),
                "start_idx": int(event.get("start_idx") or 0),
                "end_idx": int(event.get("end_idx") or 0),
            }
        )
    return summary


def _summarize_candidate_windows_for_llm(cleaned_df: Any, candidate_windows: Any, *, limit: int = 8) -> list[Dict[str, Any]]:
    events = list(candidate_windows or [])

    summary: list[Dict[str, Any]] = []
    for idx, window in enumerate(events[:limit]):
        if not isinstance(window, Mapping):
            continue
        window_type = str(window.get("type") or "")
        window_start_idx = int(window.get("window_start_idx", window.get("start_idx", 0)) or 0)
        window_end_idx = int(window.get("window_end_idx", window.get("end_idx", 0)) or 0)
        base_name = "step_event"
        if window_type == "mv_peak":
            base_name = "mv_peak"
        window_source = str(window.get("window_source") or window.get("source") or f"{base_name}_{idx + 1}")

        window_start_time = None
        window_end_time = None
        if cleaned_df is not None and "timestamp" in getattr(cleaned_df, "columns", []):
            try:
                last = len(cleaned_df) - 1
                start_i = max(0, min(window_start_idx, last))
                end_i = max(0, min(max(window_end_idx - 1, window_start_idx), last))
                ts = cleaned_df["timestamp"]
                window_start_time = ts.iloc[start_i].strftime("%Y-%m-%d %H:%M:%S")
                window_end_time = ts.iloc[end_i].strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                window_start_time = None
                window_end_time = None
        summary.append(
            {
                "window_source": window_source,
                "type": window_type,
                "start_idx": int(window.get("start_idx") or 0),
                "end_idx": int(window.get("end_idx") or 0),
                "window_start_idx": window_start_idx,
                "window_end_idx": window_end_idx,
                "amplitude": round(_safe_float(window.get("amplitude")), 4),
                "window_usable_for_id": bool(window.get("window_usable_for_id", False)),
                "window_quality_score": round(_safe_float(window.get("window_quality_score")), 4),
                "window_saturation_ratio": round(_safe_float(window.get("window_saturation_ratio", window.get("saturation_ratio"))), 4),
                "saturation_ratio": round(_safe_float(window.get("saturation_ratio")), 4),
                "drift_ratio": round(_safe_float(window.get("drift_ratio")), 4),
                "window_start_time": window_start_time,
                "window_end_time": window_end_time,
                "window_quality_reasons": list(window.get("window_quality_reasons") or []),
            }
        )
    return summary


def _summarize_identification_candidates_for_llm(candidates: Any, *, limit: int = 8) -> list[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        model_type = str(candidate.get("model_type") or "").upper()
        params = _sanitize_selected_model_params(model_type, candidate.get("selected_model_params") or {})
        summary.append(
            {
                "window_source": str(candidate.get("window_source") or ""),
                "model_type": model_type,
                "selected_model_params": params,
                "normalized_rmse": round(_safe_float(candidate.get("normalized_rmse")), 6),
                "r2_score": round(_safe_float(candidate.get("r2_score")), 6),
                "identification_fit_score": round(_safe_float(candidate.get("identification_fit_score")), 4),
                "confidence": round(_safe_float(candidate.get("confidence")), 4),
                "points": int(candidate.get("points") or 0),
                "is_selected": bool(candidate.get("is_selected", False)),
            }
        )
    return summary


def _summarize_tuning_model_candidates_for_llm(candidates: Any, *, limit: int = 5) -> list[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        summary.append(
            {
                "model_type": str(candidate.get("model_type") or ""),
                "window_source": str(candidate.get("window_source") or ""),
                "best_strategy": str(candidate.get("best_strategy") or ""),
                "identification_fit_score": round(_safe_float(candidate.get("identification_fit_score")), 4),
                "best_performance_score": round(_safe_float(candidate.get("best_performance_score")), 4),
                "best_final_rating": round(_safe_float(candidate.get("best_final_rating")), 4),
                "is_stable": bool(candidate.get("is_stable", False)),
                "pid_params": dict(candidate.get("pid_params") or {}),
            }
        )
    return summary


def _summarize_shortlist_candidates_for_llm(candidates: Any, *, limit: int = 5) -> list[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        summary.append(
            {
                "model_type": str(candidate.get("model_type") or ""),
                "window_source": str(candidate.get("window_source") or ""),
                "best_strategy": str(candidate.get("best_strategy") or ""),
                "shortlist_passed": bool(candidate.get("shortlist_passed", False)),
                "shortlist_score": round(_safe_float(candidate.get("shortlist_score")), 4),
                "identification_fit_score": round(_safe_float(candidate.get("identification_fit_score")), 4),
                "best_performance_score": round(_safe_float(candidate.get("best_performance_score")), 4),
                "best_final_rating": round(_safe_float(candidate.get("best_final_rating")), 4),
                "shortlist_reasons": list(candidate.get("shortlist_reasons") or []),
                "pid_params": dict(candidate.get("pid_params") or {}),
            }
        )
    return summary


def _summarize_scenario_evaluations_for_llm(scenarios: Any) -> Dict[str, Any]:
    summarized: Dict[str, Any] = {}
    for name, payload in dict(scenarios or {}).items():
        if not isinstance(payload, Mapping):
            continue
        summarized[str(name)] = {
            "score": round(_safe_float(payload.get("score")), 4),
            "passed": bool(payload.get("passed", False)),
            "overshoot": round(_safe_float(payload.get("overshoot")), 4),
            "settling_time": round(_safe_float(payload.get("settling_time")), 4),
            "steady_state_error": round(_safe_float(payload.get("steady_state_error")), 4),
            "constraint_penalty": round(_safe_float(payload.get("constraint_penalty")), 4),
        }
    return summarized


def _summarize_evaluation_candidates_for_llm(candidates: Any, *, limit: int = 5) -> list[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        summary.append(
            {
                "rank": int(candidate.get("rank") or 0),
                "model_type": str(candidate.get("model_type") or ""),
                "window_source": str(candidate.get("window_source") or ""),
                "strategy": str(candidate.get("strategy") or ""),
                "acceptance_performance_score": round(_safe_float(candidate.get("acceptance_performance_score")), 4),
                "robustness_score": round(_safe_float(candidate.get("robustness_score")), 4),
                "constraint_score": round(_safe_float(candidate.get("constraint_score")), 4),
                "online_readiness_score": round(_safe_float(candidate.get("online_readiness_score")), 4),
                "passed": bool(candidate.get("passed", False)),
                "is_selected": bool(candidate.get("is_selected", False)),
            }
        )
    return summary


def fetch_history_data_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    window: int | str,
    fetch_history_data_csv_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    result = fetch_history_data_csv_fn(
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        window=window,
    )
    session_store["csv_path"] = result["csv_path"]
    session_store["loop_uri"] = result["loop_uri"]
    session_store["start_time"] = result["start_time"]
    session_store["end_time"] = result["end_time"]
    session_store["data_type"] = result["data_type"]
    session_store["history_window"] = result.get("window")
    return result


def load_data_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    csv_path: str,
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    load_pid_dataset_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = load_pid_dataset_fn(
        csv_path,
        selected_loop_prefix=selected_loop_prefix,
        selected_window_index=selected_window_index,
        start_time=start_time,
        end_time=end_time,
    )
    artifact_payload: Dict[str, Any] = {}
    task_session_id = str(session_store.get("task_session_id") or "").strip()
    if task_session_id:
        processed_meta = persist_processed_csv(
            task_id=task_session_id,
            cleaned_df=prepared["cleaned_df"],
            selected_loop_prefix=selected_loop_prefix or str(session_store.get("selected_loop_prefix") or ""),
        )
        artifact_payload = {
            "task_id": processed_meta.get("task_id", task_session_id),
            "artifact_dir": processed_meta.get("artifact_dir", ""),
            "original_file_path": str(session_store.get("uploaded_original_file_path") or session_store.get("csv_path") or ""),
            "processed_file_path": processed_meta.get("processed_file_path", ""),
            "uploaded_file_name": str(session_store.get("uploaded_file_name") or ""),
            "uploaded_file_hash": str(session_store.get("uploaded_file_hash") or ""),
        }
        session_store["task_artifact_dir"] = artifact_payload.get("artifact_dir", "")
        session_store["processed_csv_path"] = artifact_payload.get("processed_file_path", "")

    session_store["csv_path"] = prepared["csv_path"]
    session_store["cleaned_df"] = prepared["cleaned_df"]
    session_store["window_df"] = prepared["window_df"]
    session_store["mv"] = prepared["mv"]
    session_store["pv"] = prepared["pv"]
    session_store["dt"] = prepared["dt"]
    session_store["step_events"] = prepared["step_events"]
    session_store["candidate_windows"] = prepared["candidate_windows"]
    session_store["selected_event"] = prepared["selected_event"]
    session_store["quality_metrics"] = prepared["quality_metrics"]
    session_store["selected_window"] = prepared["selected_window"]
    session_store["window_overview"] = prepared["window_overview"]
    session_store["history_range"] = prepared.get("history_range") or {}
    if start_time is not None:
        session_store["start_time"] = start_time
    if end_time is not None:
        session_store["end_time"] = end_time

    return {
        "data_points": prepared["data_points"],
        "sampling_time": prepared["sampling_time"],
        "mv_range": prepared["mv_range"],
        "pv_range": prepared["pv_range"],
        "available_columns": prepared["available_columns"],
        "history_range": prepared.get("history_range") or {},
        "step_events": _summarize_step_events_for_llm(prepared["step_events"]),
        "step_event_count": len(prepared["step_events"] or []),
        "candidate_windows": _summarize_candidate_windows_for_llm(prepared["cleaned_df"], prepared["candidate_windows"]),
        "candidate_window_count": len(prepared["candidate_windows"] or []),
        "artifacts": artifact_payload,
        "status": prepared["status"],
        "instruction": "数据加载成功。已提取多个候选窗口并存入上下文，后续由辨识智能体(tool_fit_fopdt)做多窗口评估，无需你做单窗口选择。"
    }


def fit_fopdt_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    dt: float,
    fit_best_fopdt_window_fn: Callable[..., Dict[str, Any]],
    build_window_overview_fn: Callable[..., Dict[str, Any]],
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
    loop_type: str = "flow",
) -> Dict[str, Any]:
    if "mv" not in session_store or "pv" not in session_store:
        raise ValueError("请先调用 tool_load_data 加载数据")

    actual_dt = float(session_store.get("dt", dt))
    identification = fit_best_fopdt_window_fn(
        cleaned_df=session_store.get("cleaned_df"),
        candidate_windows=session_store.get("candidate_windows") or [],
        quality_metrics=session_store.get("quality_metrics") or {},
        actual_dt=actual_dt,
        benchmark_fn=benchmark_fn,
        loop_type=loop_type,
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
    selected_model_type = identification.get("selected_model_type", "FOPDT")
    identification_candidates = identification.get("identification_candidates") or []
    identification_best_model_type = identification.get("identification_best_model_type", selected_model_type)
    identification_best_window_source = identification.get("identification_best_window_source", best_source)
    selected_model_params = sanitize_selected_model_params(
        identification.get("selected_model_type", "FOPDT"),
        identification.get("selected_model_params", best_model_params),
    )
    tuning_model = identification.get("tuning_model", best_model_params)
    selection_reason = identification.get("selection_reason", "")

    if best_candidate_df is not None:
        session_store["window_df"] = best_candidate_df
        session_store["mv"] = best_candidate_df["MV"].to_numpy(dtype=float)
        session_store["pv"] = best_candidate_df["PV"].to_numpy(dtype=float)

    if best_event:
        session_store["selected_event"] = best_event
    if selected_window_payload:
        session_store["selected_window"] = selected_window_payload
        session_store["window_overview"] = build_window_overview_fn(
            session_store.get("cleaned_df"),
            selected_window_payload,
        )

    session_store["model_type"] = selected_model_type
    session_store["selected_model_params"] = selected_model_params
    session_store["tuning_model"] = tuning_model
    session_store["K"] = float(tuning_model["K"])
    session_store["T"] = float(tuning_model["T"])
    session_store["L"] = float(tuning_model["L"])
    session_store["residue"] = float(best_model_params["residue"])
    session_store["normalized_rmse"] = float(best_model_params["normalized_rmse"])
    session_store["raw_rmse"] = float(best_model_params["raw_rmse"])
    session_store["r2_score"] = float(best_model_params["r2_score"])
    session_store["model_confidence"] = best_confidence
    session_store["model_attempts"] = attempts
    session_store["model_identification_candidates"] = identification_candidates
    session_store["identification_best_model_type"] = identification_best_model_type
    session_store["identification_best_window_source"] = identification_best_window_source
    session_store["identification_best_model_params"] = selected_model_params
    session_store["identification_fit_score"] = float(identification.get("identification_fit_score", 0.0) or 0.0)
    session_store["model_reason_codes"] = reason_codes
    session_store["model_next_actions"] = next_actions
    session_store["model_selected_source"] = best_source
    session_store["identification_best_model_type"] = identification_best_model_type
    session_store["identification_best_window_source"] = identification_best_window_source
    session_store["fit_preview"] = fit_preview
    session_store["window_benchmark"] = (best_benchmark or {}).get("best", {})
    session_store["model_selection_reason"] = selection_reason

    return {
        "model_type": selected_model_type,
        "selected_model_params": selected_model_params,
        "model_selection_reason": selection_reason,
        "K": float(tuning_model["K"]),
        "T": float(tuning_model["T"]),
        "L": float(tuning_model["L"]),
        "T1": selected_model_params.get("T1"),
        "T2": selected_model_params.get("T2"),
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
        "identification_best_model_type": identification_best_model_type,
        "identification_best_window_source": identification_best_window_source,
        "identification_candidates": _summarize_identification_candidates_for_llm(identification_candidates),
        "identification_candidate_count": len(identification_candidates or []),
        "selected_window_source": best_source,
        "selected_window": selected_window_payload or session_store.get("selected_window", {}),
        "window_overview": {
            "start_time": (session_store.get("window_overview") or {}).get("start_time", ""),
            "end_time": (session_store.get("window_overview") or {}).get("end_time", ""),
            "window_source": (session_store.get("window_overview") or {}).get("window_source", best_source),
        },
        "attempts": [],
        "fit_preview": {
            "model_type": selected_model_type,
            "point_count": len((fit_preview or {}).get("points") or []),
        },
        "window_benchmark": {
            "best_strategy": ((best_benchmark or {}).get("best") or {}).get("strategy", ""),
            "performance_score": _safe_float(((best_benchmark or {}).get("best") or {}).get("performance_score")),
            "final_rating": _safe_float(((best_benchmark or {}).get("best") or {}).get("final_rating")),
        },
    }


def tune_pid_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    loop_type: str,
    model_type: str = "AUTO",
    selected_model_params: Any = None,
    K: float | None = None,
    T: float | None = None,
    L: float | None = None,
    select_best_pid_strategy_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    confidence_score = _safe_float((session_store.get("model_confidence") or {}).get("confidence"), 1.0)
    selected_model = {
        "normalized_rmse": _safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
        "r2_score": _safe_float(session_store.get("r2_score")),
    }
    session_model_type = str(session_store.get("model_type", "FOPDT"))
    incoming_model_type = str(model_type or "").upper()
    model_type = session_model_type if incoming_model_type in {"", "AUTO"} else incoming_model_type
    selected_model_params = _sanitize_selected_model_params(model_type, selected_model_params) or _sanitize_selected_model_params(
        session_store.get("model_type", model_type),
        session_store.get("selected_model_params") or {},
    )
    tuning_model = dict(session_store.get("tuning_model") or {})
    knowledge_guidance_full = dict(session_store.get("expert_knowledge_guidance_full") or session_store.get("expert_knowledge_guidance") or {})
    knowledge_guidance = compact_knowledge_guidance(knowledge_guidance_full)
    knowledge_preferred_strategy = str(knowledge_guidance.get("preferred_strategy") or "").upper()
    knowledge_summary = str(knowledge_guidance.get("summary") or "").strip()
    knowledge_rule_count = int(knowledge_guidance.get("matched_count") or 0)

    identification_candidates = list(session_store.get("model_identification_candidates") or [])
    candidate_inputs: list[Dict[str, Any]] = []
    for candidate in identification_candidates:
        candidate_model_type = str(candidate.get("model_type", "") or "").upper()
        candidate_model_params = _sanitize_selected_model_params(candidate_model_type, candidate.get("selected_model_params") or {})
        if not candidate_model_type or not candidate_model_params:
            continue
        derived = _derive_tuning_metrics(candidate_model_type, candidate_model_params, fallback=tuning_model)
        candidate_inputs.append(
            {
                "model_type": derived["model_type"],
                "selected_model_params": derived["selected_model_params"],
                "K": derived["K"],
                "T": derived["T"],
                "L": derived["L"],
                "window_source": str(candidate.get("window_source", "")),
                "identification_fit_score": _safe_float(candidate.get("identification_fit_score")),
                "normalized_rmse": _safe_float(candidate.get("normalized_rmse")),
                "r2_score": _safe_float(candidate.get("r2_score")),
                "confidence": _safe_float(candidate.get("confidence"), confidence_score),
            }
        )

    if not candidate_inputs:
        derived = _derive_tuning_metrics(
            model_type,
            selected_model_params,
            fallback={
                "K": _safe_float(tuning_model.get("K"), _safe_float(K)),
                "T": _safe_float(tuning_model.get("T"), _safe_float(T, 1.0)),
                "L": _safe_float(tuning_model.get("L"), _safe_float(L)),
            },
        )
        candidate_inputs.append(
            {
                "model_type": derived["model_type"],
                "selected_model_params": derived["selected_model_params"],
                "K": derived["K"],
                "T": derived["T"],
                "L": derived["L"],
                "window_source": str(session_store.get("model_selected_source", "")),
                "identification_fit_score": _safe_float(session_store.get("identification_fit_score")),
                "normalized_rmse": selected_model["normalized_rmse"],
                "r2_score": selected_model["r2_score"],
                "confidence": confidence_score,
            }
        )

    tuning_model_candidates: list[Dict[str, Any]] = []
    selected_tuning: Dict[str, Any] | None = None
    for candidate_input in candidate_inputs[: min(len(candidate_inputs), 5)]:
        experience_guidance = _build_experience_guidance(
            loop_type=loop_type,
            model_type=str(candidate_input["model_type"]),
            selected_model_params=dict(candidate_input["selected_model_params"] or {}),
            active_k=float(candidate_input["K"]),
            active_t=float(candidate_input["T"]),
            active_l=float(candidate_input["L"]),
            knowledge_preferred_strategy=knowledge_preferred_strategy,
            knowledge_summary=knowledge_summary,
        )
        selection = select_best_pid_strategy_fn(
            K=float(candidate_input["K"]),
            T=float(candidate_input["T"]),
            L=float(candidate_input["L"]),
            loop_type=loop_type,
            model_type=str(candidate_input["model_type"]),
            selected_model_params=dict(candidate_input["selected_model_params"] or {}),
            confidence_score=float(candidate_input.get("confidence", confidence_score)),
            normalized_rmse=float(candidate_input["normalized_rmse"]),
            r2_score=float(candidate_input["r2_score"]),
            dt=float(session_store.get("dt", 1.0)),
            experience_guidance=experience_guidance,
            knowledge_guidance=knowledge_guidance_full,
        )
        best_candidate = selection["best_candidate"]
        candidate_summary = {
            "model_type": str(candidate_input["model_type"]),
            "selected_model_params": dict(candidate_input["selected_model_params"] or {}),
            "K": float(candidate_input["K"]),
            "T": float(candidate_input["T"]),
            "L": float(candidate_input["L"]),
            "window_source": str(candidate_input.get("window_source", "")),
            "identification_fit_score": float(candidate_input.get("identification_fit_score", 0.0)),
            "normalized_rmse": float(candidate_input["normalized_rmse"]),
            "r2_score": float(candidate_input["r2_score"]),
            "model_confidence": float(candidate_input.get("confidence", confidence_score)),
            "best_strategy": str(best_candidate.get("strategy", "")),
            "best_performance_score": _safe_float(best_candidate.get("performance_score")),
            "best_final_rating": _safe_float(best_candidate.get("final_rating")),
            "is_stable": bool(best_candidate.get("is_stable", False)),
            "selection_reason": selection["selection_reason"],
            "selection_inputs": dict(selection["selection_inputs"] or {}),
            "experience_guidance": selection.get("experience_guidance", experience_guidance),
            "candidate_strategies": list(selection["public_candidate_results"] or []),
            "pid_params": {
                "Kp": float(selection["pid_params"]["Kp"]),
                "Ki": float(selection["pid_params"]["Ki"]),
                "Kd": float(selection["pid_params"]["Kd"]),
                "Ti": float(selection["pid_params"]["Ti"]),
                "Td": float(selection["pid_params"]["Td"]),
                "strategy": str(selection["pid_params"]["strategy"]),
                "description": str(selection["pid_params"]["description"]),
            },
            "best_candidate": best_candidate,
        }
        tuning_model_candidates.append(candidate_summary)
        if _is_better_tuning_selection(candidate_summary, selected_tuning):
            selected_tuning = candidate_summary

    if selected_tuning is None:
        raise ValueError("Failed to compare identification candidates for PID tuning")

    best_candidate = selected_tuning["best_candidate"]
    pid_params = selected_tuning["pid_params"]
    public_candidate_results = selected_tuning["candidate_strategies"]
    experience_guidance = selected_tuning["experience_guidance"]
    selected_model_params = dict(selected_tuning["selected_model_params"] or {})
    model_type = str(selected_tuning["model_type"])
    active_K = float(selected_tuning["K"])
    active_T = float(selected_tuning["T"])
    active_L = float(selected_tuning["L"])

    session_store["pid_params"] = pid_params
    session_store["pid_candidate_results"] = public_candidate_results
    session_store["pid_tuning_model_candidates"] = [
        {
            "model_type": item["model_type"],
            "selected_model_params": item["selected_model_params"],
            "window_source": item["window_source"],
            "identification_fit_score": item["identification_fit_score"],
            "normalized_rmse": item["normalized_rmse"],
            "r2_score": item["r2_score"],
            "model_confidence": item["model_confidence"],
            "best_strategy": item["best_strategy"],
            "best_performance_score": item["best_performance_score"],
            "best_final_rating": item["best_final_rating"],
            "is_stable": item["is_stable"],
            "pid_params": item["pid_params"],
            "evaluation_result": item["best_candidate"].get("evaluation_result", {}),
        }
        for item in tuning_model_candidates
    ]
    shortlist_candidates = [
        _build_shortlist_candidate(item)
        for item in tuning_model_candidates
    ]
    shortlist_candidates = [item for item in shortlist_candidates if item.get("shortlist_passed")]
    shortlist_candidates.sort(
        key=lambda item: (
            _safe_float(item.get("shortlist_score")),
            _safe_float(item.get("best_final_rating")),
            _safe_float(item.get("best_performance_score")),
        ),
        reverse=True,
    )
    shortlist_candidates = shortlist_candidates[:5]
    session_store["pid_tuning_shortlist_candidates"] = shortlist_candidates
    session_store["tuning_selected_model_type"] = model_type
    session_store["tuning_selected_model_params"] = selected_model_params
    session_store["tuning_selected_window_source"] = str(selected_tuning.get("window_source", ""))
    session_store["model_type"] = model_type
    session_store["selected_model_params"] = selected_model_params
    session_store["model_selected_source"] = str(selected_tuning.get("window_source", ""))
    session_store["K"] = active_K
    session_store["T"] = active_T
    session_store["L"] = active_L
    session_store["strategy_used"] = best_candidate["strategy"]
    session_store["selection_reason"] = selected_tuning["selection_reason"]
    session_store["selection_inputs"] = selected_tuning["selection_inputs"]
    session_store["experience_guidance"] = experience_guidance
    session_store["expert_knowledge_guidance_full"] = knowledge_guidance_full
    session_store["expert_knowledge_guidance"] = knowledge_guidance
    selection_inputs = session_store["selection_inputs"]
    if isinstance(selection_inputs, dict):
        selection_inputs["knowledge_preferred_strategy"] = knowledge_preferred_strategy
        selection_inputs["knowledge_rule_count"] = knowledge_rule_count
        selection_inputs["knowledge_summary"] = knowledge_summary
        selection_inputs["knowledge_risk_hints"] = list(knowledge_guidance.get("risk_hints") or [])
        selection_inputs["knowledge_constraints"] = list(knowledge_guidance.get("constraints") or [])
        selection_inputs["tuning_selected_model_type"] = model_type
        selection_inputs["tuning_selected_window_source"] = str(selected_tuning.get("window_source", ""))
        selection_inputs["identification_best_model_type"] = str(session_store.get("identification_best_model_type", session_model_type))
        selection_inputs["identification_best_window_source"] = str(session_store.get("identification_best_window_source", ""))
    session_store["selected_pid_params"] = {
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"]),
        "Ti": float(pid_params["Ti"]),
        "Td": float(pid_params["Td"]),
        "strategy": best_candidate["strategy"],
        "description": str(pid_params["description"]),
    }
    session_store["selected_pid_evaluation"] = best_candidate["evaluation_result"]

    return {
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
        "selection_reason": session_store["selection_reason"],
        "selection_inputs": session_store["selection_inputs"],
        "experience_guidance": session_store.get("experience_guidance", {}),
        "expert_knowledge_guidance": knowledge_guidance,
        "selected_model_params": selected_model_params,
        "candidate_strategies": public_candidate_results[:4],
        "tuning_model_candidates": _summarize_tuning_model_candidates_for_llm(session_store["pid_tuning_model_candidates"]),
        "tuning_shortlist_candidates": _summarize_shortlist_candidates_for_llm(session_store["pid_tuning_shortlist_candidates"]),
        "tuning_selected_model_type": session_store["tuning_selected_model_type"],
        "tuning_selected_model_params": session_store["tuning_selected_model_params"],
        "tuning_selected_window_source": session_store["tuning_selected_window_source"],
        "identification_best_model_type": str(session_store.get("identification_best_model_type", session_model_type)),
        "identification_best_window_source": str(session_store.get("identification_best_window_source", "")),
        "description": str(pid_params["description"]),
    }


def query_expert_knowledge_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    loop_type: str,
    loop_name: str = "",
    plant_type: str = "",
    scenario: str = "",
    control_object: str = "",
    tower_section: str = "",
    control_target: str = "",
    graph_id: str = "",
    graph_api_url: str = "",
    query_mode: str = "local",
    response_type: str = "要点式，尽量精炼",
    include_context: bool = True,
) -> Dict[str, Any]:
    selected_model_params = _sanitize_selected_model_params(
        session_store.get("model_type", "FOPDT"),
        session_store.get("selected_model_params") or {},
    )
    context = build_knowledge_context(
        {
            "plant_type": plant_type or session_store.get("plant_type", "distillation_column"),
            "scenario": scenario or session_store.get("scenario", ""),
            "loop_type": loop_type or session_store.get("loop_type", "unknown"),
            "loop_name": loop_name or session_store.get("loop_name", ""),
            "control_object": control_object or session_store.get("control_object", ""),
            "tower_section": tower_section,
            "control_target": control_target,
            "model_type": session_store.get("model_type", ""),
            "selected_model_params": selected_model_params,
            "window_readiness": session_store.get("window_readiness", ""),
            "identification_reliability": session_store.get("identification_reliability", ""),
            "cross_window_consistency": session_store.get("cross_window_consistency", ""),
            "risk_tags": session_store.get("risk_tags") or [],
        }
    )
    local_guidance = search_distillation_rules(context)

    graph_guidance: Dict[str, Any] = {"answers": [], "graph_hints": [], "graph_summary": ""}
    if graph_api_url and graph_id:
        try:
            graph_guidance = query_knowledge_graph_api(
                base_url=graph_api_url,
                graph_id=graph_id,
                context=context,
                query_mode=query_mode,
                response_type=response_type,
                include_context=include_context,
            )
        except Exception as exc:
            graph_guidance = {
                "answers": [],
                "graph_hints": [],
                "graph_summary": f"知识图谱调用失败：{exc}",
            }

    merged = merge_knowledge_guidance(local_guidance=local_guidance, graph_guidance=graph_guidance)
    compact = compact_knowledge_guidance(merged)
    session_store["expert_knowledge_guidance_full"] = merged
    session_store["expert_knowledge_guidance"] = compact
    session_store["knowledge_questions"] = compact.get("questions", [])
    return compact


def evaluate_pid_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    model_type: str = "AUTO",
    selected_model_params: Any = None,
    K: float = 0.0,
    T: float = 0.0,
    L: float = 0.0,
    Kp: float = 0.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    method: str,
    display_agent_names: Dict[str, str],
    evaluate_pid_model_fn: Callable[..., Dict[str, Any]],
    evaluate_pid_acceptance_fn: Callable[..., Dict[str, Any]],
    diagnose_failure_fn: Callable[..., Dict[str, str]],
    choose_best_evaluation_candidate_fn: Callable[[list[Dict[str, Any]]], Dict[str, Any]],
    build_initial_assessment_fn: Callable[..., Dict[str, Any]],
    refine_pid_for_performance_fn: Callable[..., Dict[str, Any]],
    choose_alternative_model_attempt_fn: Callable[..., Dict[str, Any]],
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
    extract_candidate_windows_fn: Callable[[], list[Dict[str, Any]]],
) -> Dict[str, Any]:
    model_confidence = session_store.get("model_confidence", {})
    method_confidence = float(model_confidence.get("confidence", 0.6))
    session_model_type = str(session_store.get("model_type", "FOPDT"))
    incoming_model_type = str(model_type or "").upper()
    active_model_type = session_model_type if incoming_model_type in {"", "AUTO"} else incoming_model_type
    selected_model_params = _sanitize_selected_model_params(model_type, selected_model_params) or _sanitize_selected_model_params(
        session_store.get("model_type", model_type),
        session_store.get("selected_model_params") or {},
    )

    shortlist = [item for item in list(session_store.get("pid_tuning_shortlist_candidates") or []) if item.get("shortlist_passed", True)]
    evaluation_inputs: list[Dict[str, Any]] = []
    for item in shortlist:
        pid_params = dict(item.get("pid_params") or {})
        if not pid_params:
            continue
        evaluation_inputs.append(
            {
                "model_type": str(item.get("model_type", active_model_type)),
                "selected_model_params": dict(item.get("selected_model_params") or {}),
                "window_source": str(item.get("window_source", "")),
                "identification_fit_score": _safe_float(item.get("identification_fit_score")),
                "shortlist_score": _safe_float(item.get("shortlist_score")),
                "r2_score": _safe_float(item.get("r2_score")),
                "normalized_rmse": _safe_float(item.get("normalized_rmse")),
                "model_confidence": _safe_float(item.get("model_confidence"), method_confidence),
                "strategy": str(item.get("best_strategy") or pid_params.get("strategy") or method),
                "K": _safe_float(item.get("K")),
                "T": _safe_float(item.get("T"), 1.0),
                "L": _safe_float(item.get("L")),
                "pid_params": pid_params,
            }
        )

    if not evaluation_inputs:
        selected_pid_params = dict(session_store.get("selected_pid_params") or {})
        if selected_pid_params:
            Kp = float(selected_pid_params.get("Kp", Kp))
            Ki = float(selected_pid_params.get("Ki", Ki))
            Kd = float(selected_pid_params.get("Kd", Kd))
        derived = _derive_tuning_metrics(active_model_type, selected_model_params, {"K": K, "T": T, "L": L})
        evaluation_inputs.append(
            {
                "model_type": derived["model_type"],
                "selected_model_params": dict(derived["selected_model_params"] or {}),
                "window_source": str(session_store.get("tuning_selected_window_source", session_store.get("selected_window_source", ""))),
                "identification_fit_score": _safe_float(session_store.get("identification_fit_score")),
                "shortlist_score": 0.0,
                "r2_score": _safe_float(session_store.get("r2_score")),
                "normalized_rmse": _safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
                "model_confidence": method_confidence,
                "strategy": str(selected_pid_params.get("strategy") or session_store.get("strategy_used") or method),
                "K": float(derived["K"]),
                "T": float(derived["T"]),
                "L": float(derived["L"]),
                "pid_params": {
                    "Kp": float(Kp),
                    "Ki": float(Ki),
                    "Kd": float(Kd),
                    "Ti": _safe_float(selected_pid_params.get("Ti")),
                    "Td": _safe_float(selected_pid_params.get("Td")),
                    "strategy": str(selected_pid_params.get("strategy") or session_store.get("strategy_used") or method),
                    "description": str(selected_pid_params.get("description") or ""),
                },
            }
        )

    dt_value = float(session_store.get("dt", 1.0))
    loop_type_value = str(session_store.get("loop_type", "flow"))
    evaluated_candidates: list[Dict[str, Any]] = []
    for item in evaluation_inputs:
        pid_params = item["pid_params"]
        evaluation_result = evaluate_pid_acceptance_fn(
            K=float(item["K"]),
            T=float(item["T"]),
            L=float(item["L"]),
            Kp=_safe_float(pid_params.get("Kp")),
            Ki=_safe_float(pid_params.get("Ki")),
            Kd=_safe_float(pid_params.get("Kd")),
            method=str(item.get("strategy") or method),
            method_confidence=_safe_float(item.get("model_confidence"), method_confidence),
            model_confidence=model_confidence,
            dt=dt_value,
            loop_type=loop_type_value,
            model_type=str(item.get("model_type", active_model_type)),
            selected_model_params=dict(item.get("selected_model_params") or {}),
        )
        evaluated_candidates.append({**item, "evaluation_result": evaluation_result, "passed": bool(evaluation_result.get("passed", False))})

    selected_candidate = choose_best_evaluation_candidate_fn(evaluated_candidates)
    if not selected_candidate:
        raise ValueError("Failed to evaluate PID shortlist candidates")

    eval_result = dict(selected_candidate.get("evaluation_result") or {})
    active_model_type = str(selected_candidate.get("model_type", active_model_type))
    selected_model_params = dict(selected_candidate.get("selected_model_params") or {})
    K = float(selected_candidate.get("K", K))
    T = float(selected_candidate.get("T", T))
    L = float(selected_candidate.get("L", L))
    selected_pid_params = dict(selected_candidate.get("pid_params") or {})
    Kp = float(selected_pid_params.get("Kp", Kp))
    Ki = float(selected_pid_params.get("Ki", Ki))
    Kd = float(selected_pid_params.get("Kd", Kd))
    pass_threshold = 7.0
    passed = bool(eval_result.get("passed", False))

    diagnosis = diagnose_failure_fn(
        eval_result=eval_result,
        model_r2=_safe_float(selected_candidate.get("r2_score"), _safe_float(session_store.get("r2_score"))),
        model_rmse=_safe_float(selected_candidate.get("normalized_rmse"), _safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue")))),
        candidate_window_count=len(session_store.get("candidate_windows") or []),
    ) if not passed else {"failure_reason": "", "feedback_target": "", "feedback_action": ""}
    initial_assessment = build_initial_assessment_fn(
        eval_result=eval_result,
        pass_threshold=pass_threshold,
        diagnosis=diagnosis,
        evaluated_pid={"Kp": Kp, "Ki": Ki, "Kd": Kd},
    )

    ranked_candidates = sorted(
        evaluated_candidates,
        key=lambda item: (
            _safe_float((item.get("evaluation_result") or {}).get("online_readiness_score", (item.get("evaluation_result") or {}).get("final_rating"))),
            _safe_float((item.get("evaluation_result") or {}).get("acceptance_performance_score", (item.get("evaluation_result") or {}).get("performance_score"))),
            _safe_float((item.get("evaluation_result") or {}).get("robustness_score")),
            _safe_float((item.get("evaluation_result") or {}).get("constraint_score")),
        ),
        reverse=True,
    )
    evaluation_candidates = []
    for idx, item in enumerate(ranked_candidates, start=1):
        candidate_eval = item.get("evaluation_result") or {}
        evaluation_candidates.append(
            {
                "rank": idx,
                "model_type": item.get("model_type"),
                "selected_model_params": item.get("selected_model_params"),
                "window_source": item.get("window_source"),
                "strategy": item.get("strategy"),
                "pid_params": item.get("pid_params"),
                "identification_fit_score": item.get("identification_fit_score"),
                "shortlist_score": item.get("shortlist_score"),
                "acceptance_performance_score": _safe_float(candidate_eval.get("acceptance_performance_score", candidate_eval.get("performance_score"))),
                "robustness_score": _safe_float(candidate_eval.get("robustness_score")),
                "constraint_score": _safe_float(candidate_eval.get("constraint_score")),
                "online_readiness_score": _safe_float(candidate_eval.get("online_readiness_score", candidate_eval.get("final_rating"))),
                "passed": bool(candidate_eval.get("passed", False)),
                "is_selected": item is selected_candidate,
            }
        )

    selected_summary = {
        "model_type": active_model_type,
        "selected_model_params": selected_model_params,
        "window_source": str(selected_candidate.get("window_source", "")),
        "strategy": str(selected_candidate.get("strategy", method)),
        "pid_params": selected_pid_params,
        "identification_fit_score": _safe_float(selected_candidate.get("identification_fit_score")),
        "shortlist_score": _safe_float(selected_candidate.get("shortlist_score")),
        "acceptance_performance_score": _safe_float(eval_result.get("acceptance_performance_score", eval_result.get("performance_score"))),
        "robustness_score": _safe_float(eval_result.get("robustness_score")),
        "constraint_score": _safe_float(eval_result.get("constraint_score")),
        "online_readiness_score": _safe_float(eval_result.get("online_readiness_score", eval_result.get("final_rating"))),
        "passed": passed,
    }

    session_store["evaluation_candidates"] = evaluation_candidates
    session_store["evaluation_selected_candidate"] = selected_summary
    session_store["evaluation_result"] = eval_result
    session_store["evaluation_pass_threshold"] = pass_threshold
    session_store["evaluation_feedback"] = diagnosis
    session_store["initial_assessment"] = initial_assessment
    session_store["auto_refine_result"] = {}
    session_store["model_retry_result"] = {}
    session_store["performance_score"] = float(eval_result.get("acceptance_performance_score", eval_result.get("performance_score", 0.0)))
    session_store["acceptance_performance_score"] = float(eval_result.get("acceptance_performance_score", eval_result.get("performance_score", 0.0)))
    session_store["method_confidence"] = float(eval_result.get("method_confidence", 0.0))
    session_store["robustness_score"] = float(eval_result.get("robustness_score", 0.0))
    session_store["constraint_score"] = float(eval_result.get("constraint_score", 0.0))
    session_store["final_rating"] = float(eval_result.get("online_readiness_score", eval_result.get("final_rating", 0.0)))
    session_store["online_readiness_score"] = float(eval_result.get("online_readiness_score", eval_result.get("final_rating", 0.0)))
    session_store["passed"] = passed
    session_store["pass_threshold"] = pass_threshold
    session_store["failure_reason"] = diagnosis["failure_reason"]
    session_store["feedback_target"] = diagnosis["feedback_target"]
    session_store["feedback_target_display"] = display_agent_names.get(diagnosis["feedback_target"], diagnosis["feedback_target"])
    session_store["feedback_action"] = diagnosis["feedback_action"]
    session_store["performance_details"] = eval_result.get("performance_details", {})
    session_store["final_details"] = eval_result.get("final_details", {})
    session_store["simulation"] = eval_result.get("simulation", {})
    session_store["replay_evaluation"] = (eval_result.get("scenario_evaluations") or {}).get("disturbance_rejection", {})
    session_store["scenario_evaluations"] = eval_result.get("scenario_evaluations", {})
    session_store["launch_recommendation"] = eval_result.get("launch_recommendation", "")
    session_store["strategy_used"] = str(selected_candidate.get("strategy", method))
    session_store["tuning_selected_model_type"] = active_model_type
    session_store["tuning_selected_model_params"] = selected_model_params
    session_store["tuning_selected_window_source"] = str(selected_candidate.get("window_source", ""))
    session_store["selected_pid_params"] = {**selected_pid_params, "Kp": Kp, "Ki": Ki, "Kd": Kd, "strategy": str(selected_candidate.get("strategy", method))}
    session_store["selected_pid_evaluation"] = eval_result
    session_store["model_type"] = active_model_type
    session_store["selected_model_params"] = selected_model_params
    session_store["K"] = K
    session_store["T"] = T
    session_store["L"] = L

    return {
        "model_type": active_model_type,
        "selected_model_params": selected_model_params,
        "performance_score": float(eval_result.get("acceptance_performance_score", eval_result.get("performance_score", 0.0))),
        "acceptance_performance_score": float(eval_result.get("acceptance_performance_score", eval_result.get("performance_score", 0.0))),
        "method_confidence": float(eval_result.get("method_confidence", 0.0)),
        "robustness_score": float(eval_result.get("robustness_score", 0.0)),
        "constraint_score": float(eval_result.get("constraint_score", 0.0)),
        "final_rating": float(eval_result.get("online_readiness_score", eval_result.get("final_rating", 0.0))),
        "online_readiness_score": float(eval_result.get("online_readiness_score", eval_result.get("final_rating", 0.0))),
        "passed": passed,
        "pass_threshold": pass_threshold,
        "performance_details": eval_result.get("performance_details", {}),
        "final_details": eval_result.get("final_details", {}),
        "failure_reason": diagnosis["failure_reason"],
        "feedback_target": diagnosis["feedback_target"],
        "feedback_target_display": display_agent_names.get(diagnosis["feedback_target"], diagnosis["feedback_target"]),
        "feedback_action": diagnosis["feedback_action"],
        "initial_assessment": initial_assessment,
        "auto_refine_result": {},
        "model_retry_result": {},
        "simulation": {},
        "replay_evaluation": {},
        "scenario_evaluations": _summarize_scenario_evaluations_for_llm(eval_result.get("scenario_evaluations", {})),
        "evaluation_candidates": _summarize_evaluation_candidates_for_llm(evaluation_candidates),
        "evaluation_selected_candidate": selected_summary,
        "launch_recommendation": eval_result.get("launch_recommendation", ""),
        "evaluated_pid": {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    }
