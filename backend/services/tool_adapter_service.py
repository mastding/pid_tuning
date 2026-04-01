from __future__ import annotations

import ast
import json
from typing import Any, Callable, Dict, Mapping

from memory.experience_service import retrieve_experience_guidance
from services.identification_service import sanitize_selected_model_params
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
    load_pid_dataset_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = load_pid_dataset_fn(
        csv_path,
        selected_loop_prefix=selected_loop_prefix,
        selected_window_index=selected_window_index,
    )
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

    return {
        "data_points": prepared["data_points"],
        "sampling_time": prepared["sampling_time"],
        "mv_range": prepared["mv_range"],
        "pv_range": prepared["pv_range"],
        "available_columns": prepared["available_columns"],
        "step_events": prepared["step_events"],
        "candidate_windows": prepared["candidate_windows"],
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
    session_store["model_reason_codes"] = reason_codes
    session_store["model_next_actions"] = next_actions
    session_store["model_selected_source"] = best_source
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
        "selected_window_source": best_source,
        "selected_window": selected_window_payload or session_store.get("selected_window", {}),
        "window_overview": session_store.get("window_overview", {"points": []}),
        "attempts": attempts,
        "fit_preview": fit_preview,
        "window_benchmark": (best_benchmark or {}).get("best", {}),
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
    normalized_model_type = str(model_type).upper()
    if normalized_model_type == "SOPDT":
        active_K = _safe_float(selected_model_params.get("K"), _safe_float(K))
        active_T = _safe_float(selected_model_params.get("T1"), 0.0) + _safe_float(selected_model_params.get("T2"), 0.0)
        if active_T <= 0:
            active_T = _safe_float(tuning_model.get("T"), _safe_float(T, 1.0))
        active_L = _safe_float(selected_model_params.get("L"), _safe_float(L))
    elif normalized_model_type == "IPDT":
        active_K = _safe_float(selected_model_params.get("K"), _safe_float(K))
        active_L = _safe_float(selected_model_params.get("L"), _safe_float(L, 1.0))
        active_T = max(_safe_float(T, 1.0), active_L)
    elif normalized_model_type == "FO":
        active_K = _safe_float(selected_model_params.get("K"), _safe_float(K))
        active_T = _safe_float(selected_model_params.get("T"), _safe_float(T, 1.0))
        active_L = 0.0
    else:
        active_K = _safe_float(selected_model_params.get("K"), _safe_float(tuning_model.get("K"), _safe_float(K)))
        active_T = _safe_float(selected_model_params.get("T"), _safe_float(tuning_model.get("T"), _safe_float(T, 1.0)))
        active_L = _safe_float(selected_model_params.get("L"), _safe_float(tuning_model.get("L"), _safe_float(L)))

    knowledge_guidance_full = dict(session_store.get("expert_knowledge_guidance_full") or session_store.get("expert_knowledge_guidance") or {})
    knowledge_guidance = compact_knowledge_guidance(knowledge_guidance_full)
    knowledge_preferred_strategy = str(knowledge_guidance.get("preferred_strategy") or "").upper()
    knowledge_summary = str(knowledge_guidance.get("summary") or "").strip()
    knowledge_rule_count = int(knowledge_guidance.get("matched_count") or 0)

    experience_guidance = retrieve_experience_guidance(
        loop_type=loop_type,
        model_type=model_type,
        K=active_K,
        T=active_T,
        L=active_L,
        selected_model_params=selected_model_params,
        limit=3,
        candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
    )
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
    selection = select_best_pid_strategy_fn(
        K=active_K,
        T=active_T,
        L=active_L,
        loop_type=loop_type,
        model_type=model_type,
        selected_model_params=selected_model_params,
        confidence_score=confidence_score,
        normalized_rmse=selected_model["normalized_rmse"],
        r2_score=selected_model["r2_score"],
        dt=float(session_store.get("dt", 1.0)),
        experience_guidance=experience_guidance,
        knowledge_guidance=knowledge_guidance_full,
    )
    best_candidate = selection["best_candidate"]
    pid_params = selection["pid_params"]
    public_candidate_results = selection["public_candidate_results"]

    session_store["pid_params"] = pid_params
    session_store["pid_candidate_results"] = public_candidate_results
    session_store["strategy_used"] = best_candidate["strategy"]
    session_store["selection_reason"] = selection["selection_reason"]
    session_store["selection_inputs"] = selection["selection_inputs"]
    session_store["experience_guidance"] = selection.get("experience_guidance", experience_guidance)
    session_store["expert_knowledge_guidance_full"] = knowledge_guidance_full
    session_store["expert_knowledge_guidance"] = knowledge_guidance
    selection_inputs = session_store["selection_inputs"]
    if isinstance(selection_inputs, dict):
        selection_inputs["knowledge_preferred_strategy"] = knowledge_preferred_strategy
        selection_inputs["knowledge_rule_count"] = knowledge_rule_count
        selection_inputs["knowledge_summary"] = knowledge_summary
        selection_inputs["knowledge_risk_hints"] = list(knowledge_guidance.get("risk_hints") or [])
        selection_inputs["knowledge_constraints"] = list(knowledge_guidance.get("constraints") or [])
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
        "candidate_strategies": public_candidate_results,
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
    diagnose_failure_fn: Callable[..., Dict[str, str]],
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
    selected_pid_params = session_store.get("selected_pid_params") or {}
    selected_pid_evaluation = session_store.get("selected_pid_evaluation")
    auto_refine_result = None
    model_retry_result = None

    if selected_pid_params:
        Kp = float(selected_pid_params.get("Kp", Kp))
        Ki = float(selected_pid_params.get("Ki", Ki))
        Kd = float(selected_pid_params.get("Kd", Kd))

    if selected_pid_evaluation:
        eval_result = selected_pid_evaluation
    else:
        if active_model_type == "SOPDT":
            active_K = float(selected_model_params.get("K", K))
            active_T = float(selected_model_params.get("T1", T)) + float(selected_model_params.get("T2", T))
            active_L = float(selected_model_params.get("L", L))
        elif active_model_type == "IPDT":
            active_K = float(selected_model_params.get("K", K))
            active_L = float(selected_model_params.get("L", L))
            active_T = max(float(T or 0.0), active_L, 1e-3)
        elif active_model_type == "FO":
            active_K = float(selected_model_params.get("K", K))
            active_T = float(selected_model_params.get("T", T))
            active_L = 0.0
        else:
            active_K = float(selected_model_params.get("K", K))
            active_T = float(selected_model_params.get("T", T))
            active_L = float(selected_model_params.get("L", L))

        eval_result = evaluate_pid_model_fn(
            K=active_K,
            T=active_T,
            L=active_L,
            Kp=float(Kp),
            Ki=float(Ki),
            Kd=float(Kd),
            method=method,
            method_confidence=method_confidence,
            model_confidence=model_confidence,
            dt=float(session_store.get("dt", 1.0)),
            model_type=active_model_type,
            selected_model_params=selected_model_params,
        )

    base_eval_result = eval_result
    session_store["evaluation_result"] = eval_result
    pass_threshold = 7.0
    passed = bool(_safe_float(eval_result.get("final_rating")) >= pass_threshold)
    diagnosis = diagnose_failure_fn(
        eval_result=eval_result,
        model_r2=_safe_float(session_store.get("r2_score")),
        model_rmse=_safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
        candidate_window_count=len(session_store.get("candidate_windows") or []),
    ) if not passed else {
        "failure_reason": "",
        "feedback_target": "",
        "feedback_action": "",
    }
    initial_assessment = build_initial_assessment_fn(
        eval_result=base_eval_result,
        pass_threshold=pass_threshold,
        diagnosis=diagnosis,
        evaluated_pid={"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    )
    session_store["initial_assessment"] = initial_assessment

    if not passed and diagnosis.get("feedback_target") == "pid_expert":
        if active_model_type == "SOPDT":
            refine_model_params = {
                "model_type": "SOPDT",
                "K": float(selected_model_params.get("K", K)),
                "T1": float(selected_model_params.get("T1", T)),
                "T2": float(selected_model_params.get("T2", T)),
                "L": float(selected_model_params.get("L", L)),
            }
        elif active_model_type == "IPDT":
            refine_model_params = {
                "model_type": "IPDT",
                "K": float(selected_model_params.get("K", K)),
                "L": max(float(selected_model_params.get("L", L)), 1e-3),
            }
        elif active_model_type == "FO":
            refine_model_params = {
                "model_type": "FO",
                "K": float(selected_model_params.get("K", K)),
                "T1": float(selected_model_params.get("T", T)),
                "T2": 0.0,
                "L": 0.0,
            }
        else:
            refine_model_params = {"model_type": "FOPDT", "K": float(K), "T1": float(T), "T2": 0.0, "L": float(L)}

        refined = refine_pid_for_performance_fn(
            model_params=refine_model_params,
            base_pid_params={"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
            method_confidence=method_confidence,
            dt=float(session_store.get("dt", 1.0)),
            base_strategy=str(session_store.get("strategy_used", method or "auto")),
        )
        best_refined = refined.get("best") or {}
        if best_refined:
            improved = float(best_refined.get("final_rating", 0.0)) > float(eval_result.get("final_rating", 0.0)) + 1e-9
            if improved:
                Kp = float(best_refined["Kp"])
                Ki = float(best_refined["Ki"])
                Kd = float(best_refined["Kd"])
                eval_result = best_refined["evaluation_result"]
                session_store["selected_pid_params"] = {
                    **(session_store.get("selected_pid_params") or {}),
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "strategy": str(session_store.get("strategy_used", method or "auto")),
                    "description": "Auto refined after evaluation feedback",
                }
                session_store["selected_pid_evaluation"] = eval_result
                session_store["evaluation_result"] = eval_result
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
                diagnosis = diagnose_failure_fn(
                    eval_result=eval_result,
                    model_r2=_safe_float(session_store.get("r2_score")),
                    model_rmse=_safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
                    candidate_window_count=len(session_store.get("candidate_windows") or []),
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
        alternative_model = choose_alternative_model_attempt_fn(
            attempts=session_store.get("model_attempts") or [],
            current_source=str(session_store.get("model_selected_source", "")),
            candidate_map={candidate["name"]: candidate for candidate in extract_candidate_windows_fn()},
            loop_type=str(session_store.get("loop_type", "flow")),
            dt=float(session_store.get("dt", 1.0)),
            pass_threshold=pass_threshold,
            benchmark_fn=benchmark_fn,
            refine_fn=refine_pid_for_performance_fn,
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
                diagnosis = diagnose_failure_fn(
                    eval_result=eval_result,
                    model_r2=_safe_float(session_store.get("r2_score")),
                    model_rmse=_safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
                    candidate_window_count=len(session_store.get("candidate_windows") or []),
                ) if not passed else {
                    "failure_reason": "",
                    "feedback_target": "",
                    "feedback_action": "",
                }
                session_store["K"] = K
                session_store["T"] = T
                session_store["L"] = L
                session_store["strategy_used"] = str(alternative_model.get("strategy", session_store.get("strategy_used", "")))
                session_store["model_selected_source"] = str(alternative_model.get("window_source", session_store.get("model_selected_source", "")))
                session_store["selected_pid_params"] = {
                    **(session_store.get("selected_pid_params") or {}),
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "strategy": str(alternative_model.get("strategy", "")),
                    "description": "Switched to alternative identification window",
                }
                session_store["selected_pid_evaluation"] = eval_result
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

    replay_evaluation: Dict[str, Any] | None = None
    try:
        sp_series: list[float] = []
        pv_initial: float | None = None
        mv_initial: float | None = None
        replay_source = ""
        window_df = session_store.get("window_df")
        if window_df is not None and hasattr(window_df, "columns") and "SV" in getattr(window_df, "columns", []):
            sp_series = [float(x) for x in window_df["SV"].tolist()]
            replay_source = "window_df"
            if "PV" in window_df.columns:
                try:
                    pv_initial = float(window_df["PV"].iloc[0])
                except Exception:
                    pv_initial = None
            if "MV" in window_df.columns:
                try:
                    mv_initial = float(window_df["MV"].iloc[0])
                except Exception:
                    mv_initial = None

        if not sp_series:
            overview = session_store.get("window_overview") or {}
            points = overview.get("points") or []
            sp_series = [float(p.get("sv")) for p in points if p.get("sv") is not None]
            replay_source = "window_overview" if sp_series else ""
            if points:
                first = points[0] or {}
                pv_value = first.get("pv")
                mv_value = first.get("mv")
                pv_initial = float(pv_value) if pv_value is not None else pv_initial
                mv_initial = float(mv_value) if mv_value is not None else mv_initial

        if sp_series:
            sp_align_offset = 0.0
            try:
                sp0 = float(sp_series[0])
                sp_min = float(min(sp_series))
                sp_max = float(max(sp_series))
                sp_range = sp_max - sp_min
                if pv_initial is not None:
                    candidate_offset = float(pv_initial) - sp0
                    if abs(candidate_offset) > max(3.0 * max(sp_range, 1e-6), 1.0):
                        sp_series = [float(value) + float(candidate_offset) for value in sp_series]
                        sp_align_offset = float(candidate_offset)
            except Exception:
                sp_align_offset = 0.0

            if active_model_type == "SOPDT":
                replay_model_params = {
                    "model_type": "SOPDT",
                    "K": float(selected_model_params.get("K", K)),
                    "T1": float(selected_model_params.get("T1", T)),
                    "T2": float(selected_model_params.get("T2", 0.0)),
                    "L": float(selected_model_params.get("L", L)),
                }
            elif active_model_type == "IPDT":
                replay_model_params = {
                    "model_type": "IPDT",
                    "K": float(selected_model_params.get("K", K)),
                    "T1": 1.0,
                    "T2": 0.0,
                    "L": max(float(selected_model_params.get("L", L)), 1e-3),
                }
            elif active_model_type == "FO":
                replay_model_params = {
                    "model_type": "FO",
                    "K": float(selected_model_params.get("K", K)),
                    "T1": float(selected_model_params.get("T", T)),
                    "T2": 0.0,
                    "L": 0.0,
                }
            else:
                replay_model_params = {
                    "model_type": "FOPDT",
                    "K": float(selected_model_params.get("K", K)),
                    "T1": float(selected_model_params.get("T", T)),
                    "T2": 0.0,
                    "L": float(selected_model_params.get("L", L)),
                }

            from skills.rating import ModelRating

            replay_evaluation = ModelRating.evaluate_replay(
                model_params=replay_model_params,
                pid_params={"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
                sp_series=sp_series,
                pv_initial=pv_initial,
                mv_initial=mv_initial,
                dt=float(session_store.get("dt", 1.0)),
                loop_type=str(session_store.get("loop_type", "flow")),
            )
            replay_evaluation["source"] = replay_source
            replay_evaluation["sp_align_offset"] = sp_align_offset
    except Exception:
        replay_evaluation = None

    session_store["evaluation_pass_threshold"] = pass_threshold
    session_store["evaluation_feedback"] = diagnosis
    session_store["initial_assessment"] = initial_assessment
    session_store["auto_refine_result"] = auto_refine_result or {}
    session_store["model_retry_result"] = model_retry_result or {}
    session_store["performance_score"] = float(eval_result["performance_score"])
    session_store["method_confidence"] = float(eval_result["method_confidence"])
    session_store["final_rating"] = float(eval_result["final_rating"])
    session_store["passed"] = passed
    session_store["pass_threshold"] = pass_threshold
    session_store["failure_reason"] = diagnosis["failure_reason"]
    session_store["feedback_target"] = diagnosis["feedback_target"]
    session_store["feedback_target_display"] = display_agent_names.get(diagnosis["feedback_target"], diagnosis["feedback_target"])
    session_store["feedback_action"] = diagnosis["feedback_action"]
    session_store["performance_details"] = eval_result["performance_details"]
    session_store["final_details"] = eval_result["final_details"]
    session_store["simulation"] = eval_result["simulation"]
    if replay_evaluation:
        session_store["replay_evaluation"] = replay_evaluation

    return {
        "model_type": active_model_type,
        "selected_model_params": selected_model_params,
        "performance_score": float(eval_result["performance_score"]),
        "method_confidence": float(eval_result["method_confidence"]),
        "final_rating": float(eval_result["final_rating"]),
        "passed": passed,
        "pass_threshold": pass_threshold,
        "performance_details": eval_result["performance_details"],
        "final_details": eval_result["final_details"],
        "failure_reason": diagnosis["failure_reason"],
        "feedback_target": diagnosis["feedback_target"],
        "feedback_target_display": display_agent_names.get(diagnosis["feedback_target"], diagnosis["feedback_target"]),
        "feedback_action": diagnosis["feedback_action"],
        "initial_assessment": initial_assessment,
        "auto_refine_result": auto_refine_result or {},
        "model_retry_result": model_retry_result or {},
        "simulation": eval_result["simulation"],
        "replay_evaluation": replay_evaluation or {},
        "evaluated_pid": {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    }
