from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from memory.experience_service import retrieve_experience_guidance


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fetch_history_data_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    fetch_history_data_csv_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    result = fetch_history_data_csv_fn(
        loop_uri=loop_uri,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
    )
    session_store["csv_path"] = result["csv_path"]
    session_store["loop_uri"] = result["loop_uri"]
    session_store["start_time"] = result["start_time"]
    session_store["end_time"] = result["end_time"]
    session_store["data_type"] = result["data_type"]
    return result


def load_data_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    csv_path: str,
    load_pid_dataset_fn: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = load_pid_dataset_fn(csv_path)
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
    }


def fit_fopdt_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    dt: float,
    fit_best_fopdt_window_fn: Callable[..., Dict[str, Any]],
    build_window_overview_fn: Callable[..., Dict[str, Any]],
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
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

    session_store["K"] = float(best_model_params["K"])
    session_store["T"] = float(best_model_params["T"])
    session_store["L"] = float(best_model_params["L"])
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

    return {
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
        "selected_window": selected_window_payload or session_store.get("selected_window", {}),
        "window_overview": session_store.get("window_overview", {"points": []}),
        "attempts": attempts,
        "fit_preview": fit_preview,
        "window_benchmark": (best_benchmark or {}).get("best", {}),
    }


def tune_pid_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    K: float,
    T: float,
    L: float,
    loop_type: str,
    select_best_pid_strategy_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    confidence_score = _safe_float((session_store.get("model_confidence") or {}).get("confidence"), 1.0)
    selected_model = {
        "normalized_rmse": _safe_float(session_store.get("normalized_rmse"), _safe_float(session_store.get("residue"))),
        "r2_score": _safe_float(session_store.get("r2_score")),
    }
    experience_guidance = retrieve_experience_guidance(
        loop_type=loop_type,
        K=float(K),
        T=float(T),
        L=float(L),
        limit=3,
        candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
    )
    selection = select_best_pid_strategy_fn(
        K=float(K),
        T=float(T),
        L=float(L),
        loop_type=loop_type,
        confidence_score=confidence_score,
        normalized_rmse=selected_model["normalized_rmse"],
        r2_score=selected_model["r2_score"],
        dt=float(session_store.get("dt", 1.0)),
        experience_guidance=experience_guidance,
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
        "candidate_strategies": public_candidate_results,
        "description": str(pid_params["description"]),
    }


def evaluate_pid_tool(
    *,
    session_store: Mapping[str, Any] | dict[str, Any],
    K: float,
    T: float,
    L: float,
    Kp: float,
    Ki: float,
    Kd: float,
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
        eval_result = evaluate_pid_model_fn(
            K=float(K),
            T=float(T),
            L=float(L),
            Kp=float(Kp),
            Ki=float(Ki),
            Kd=float(Kd),
            method=method,
            method_confidence=method_confidence,
            model_confidence=model_confidence,
            dt=float(session_store.get("dt", 1.0)),
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
        refined = refine_pid_for_performance_fn(
            model_params={"K": float(K), "T1": float(T), "T2": 0.0, "L": float(L)},
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

    return {
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
        "evaluated_pid": {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)},
    }
