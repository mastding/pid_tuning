from __future__ import annotations

from typing import Any, Callable, Dict, List

from skills.rating import ModelRating


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _build_model_params(
    *,
    K: float,
    T: float,
    L: float,
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_model_params = selected_model_params or {}
    normalized_type = str(model_type or selected_model_params.get("model_type", "FOPDT")).upper()
    if normalized_type == "SOPDT":
        return {
            "model_type": "SOPDT",
            "K": float(selected_model_params.get("K", K)),
            "T1": float(selected_model_params.get("T1", T)),
            "T2": float(selected_model_params.get("T2", T)),
            "L": float(selected_model_params.get("L", L)),
        }
    if normalized_type == "IPDT":
        return {
            "model_type": "IPDT",
            "K": float(selected_model_params.get("K", K)),
            "L": max(float(selected_model_params.get("L", L)), 1e-3),
        }
    if normalized_type == "FO":
        return {
            "model_type": "FO",
            "K": float(selected_model_params.get("K", K)),
            "T1": float(selected_model_params.get("T", T)),
            "T2": 0.0,
            "L": 0.0,
        }
    return {"model_type": "FOPDT", "K": float(K), "T1": float(T), "T2": 0.0, "L": float(L)}


def _mv_saturation_pct(simulation: Dict[str, Any]) -> float:
    mv_history = simulation.get("mv_history") or []
    if not mv_history:
        return 0.0
    saturated = 0
    for value in mv_history:
        mv = _safe_float(value)
        if mv <= 0.5 or mv >= 99.5:
            saturated += 1
    return (saturated / float(len(mv_history))) * 100.0


def _mv_total_variation(simulation: Dict[str, Any]) -> float:
    mv_history = simulation.get("mv_history") or []
    if len(mv_history) <= 1:
        return 0.0
    last = _safe_float(mv_history[0])
    total = 0.0
    for value in mv_history[1:]:
        current = _safe_float(value)
        total += abs(current - last)
        last = current
    return total


def _constraint_score_from_simulation(simulation: Dict[str, Any]) -> Dict[str, Any]:
    saturation_pct = _mv_saturation_pct(simulation)
    mv_total_variation = _mv_total_variation(simulation)
    is_stable = bool(simulation.get("is_stable", True))

    score = 10.0
    score -= min(5.0, saturation_pct / 8.0)
    score -= min(3.0, mv_total_variation / 250.0)
    if not is_stable:
        score -= 2.5

    return {
        "constraint_score": round(_clamp(score, 0.0, 10.0), 2),
        "saturation_pct": round(saturation_pct, 3),
        "mv_total_variation": round(mv_total_variation, 3),
        "is_stable": is_stable,
    }


def _evaluate_disturbance_rejection(
    *,
    model_params: Dict[str, Any],
    pid_params: Dict[str, Any],
    dt: float,
    loop_type: str,
) -> Dict[str, Any]:
    replay = ModelRating.evaluate_replay(
        model_params=model_params,
        pid_params=pid_params,
        sp_series=[50.0] * 240,
        pv_initial=55.0,
        mv_initial=50.0,
        dt=dt,
        loop_type=loop_type,
    )
    return {
        "tracking_score": float(replay.get("tracking_score", 0.0) or 0.0),
        "metrics": replay.get("metrics", {}) or {},
        "simulation_preview": replay.get("simulation_preview", {}) or {},
    }


def _perturbed_model_variants(model_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_type = str(model_params.get("model_type", "FOPDT")).upper()
    combos = [
        {"k_scale": 0.9, "t_scale": 0.9, "l_scale": 1.1},
        {"k_scale": 1.1, "t_scale": 1.1, "l_scale": 0.9},
        {"k_scale": 1.0, "t_scale": 1.2, "l_scale": 1.2},
    ]
    variants: List[Dict[str, Any]] = []
    for combo in combos:
        if model_type == "SOPDT":
            variants.append(
                {
                    "model_type": "SOPDT",
                    "K": _safe_float(model_params.get("K")) * combo["k_scale"],
                    "T1": max(_safe_float(model_params.get("T1")) * combo["t_scale"], 1e-3),
                    "T2": max(_safe_float(model_params.get("T2")) * combo["t_scale"], 1e-3),
                    "L": max(_safe_float(model_params.get("L")) * combo["l_scale"], 0.0),
                }
            )
        elif model_type == "IPDT":
            variants.append(
                {
                    "model_type": "IPDT",
                    "K": _safe_float(model_params.get("K")) * combo["k_scale"],
                    "L": max(_safe_float(model_params.get("L")) * combo["l_scale"], 1e-3),
                }
            )
        elif model_type == "FO":
            variants.append(
                {
                    "model_type": "FO",
                    "K": _safe_float(model_params.get("K")) * combo["k_scale"],
                    "T1": max(_safe_float(model_params.get("T1")) * combo["t_scale"], 1e-3),
                    "T2": 0.0,
                    "L": 0.0,
                }
            )
        else:
            variants.append(
                {
                    "model_type": "FOPDT",
                    "K": _safe_float(model_params.get("K")) * combo["k_scale"],
                    "T1": max(_safe_float(model_params.get("T1")) * combo["t_scale"], 1e-3),
                    "T2": 0.0,
                    "L": max(_safe_float(model_params.get("L")) * combo["l_scale"], 0.0),
                }
            )
    return variants


def _evaluate_robustness(
    *,
    model_params: Dict[str, Any],
    pid_params: Dict[str, Any],
    method_confidence: float,
    dt: float,
    loop_type: str,
    method: str,
) -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    scores: List[float] = []
    stable_count = 0
    for idx, perturbed in enumerate(_perturbed_model_variants(model_params), start=1):
        result = ModelRating.evaluate(
            model_params=perturbed,
            pid_params=pid_params,
            method=f"{method.lower()}_perturbed_{idx}",
            method_confidence=method_confidence,
            method_confidence_details={"source": "acceptance_perturbation"},
            dt=dt,
            loop_type=loop_type,
            sp_initial=50.0,
            sp_final=60.0,
            n_steps=400,
        )
        performance_score = _safe_float(result.get("performance_score"))
        is_stable = bool((result.get("simulation") or {}).get("is_stable", False))
        if is_stable:
            stable_count += 1
        scores.append(performance_score)
        scenarios.append(
            {
                "index": idx,
                "performance_score": round(performance_score, 2),
                "is_stable": is_stable,
                "overshoot": _safe_float((result.get("performance_details") or {}).get("overshoot")),
                "settling_time": _safe_float((result.get("performance_details") or {}).get("settling_time"), -1.0),
            }
        )

    if not scores:
        return {"robustness_score": 0.0, "stable_ratio": 0.0, "scenarios": []}

    average_score = sum(scores) / len(scores)
    minimum_score = min(scores)
    stable_ratio = stable_count / float(len(scores))
    robustness_score = _clamp(minimum_score * 0.6 + average_score * 0.4 + stable_ratio * 1.5, 0.0, 10.0)
    return {
        "robustness_score": round(robustness_score, 2),
        "stable_ratio": round(stable_ratio, 3),
        "scenarios": scenarios,
    }


def evaluate_pid_acceptance(
    *,
    K: float,
    T: float,
    L: float,
    Kp: float,
    Ki: float,
    Kd: float,
    method: str,
    method_confidence: float,
    model_confidence: Dict[str, Any],
    dt: float,
    loop_type: str = "flow",
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model_params = _build_model_params(
        K=K,
        T=T,
        L=L,
        model_type=model_type,
        selected_model_params=selected_model_params,
    )
    pid_params = {"Kp": float(Kp), "Ki": float(Ki), "Kd": float(Kd)}

    forward = ModelRating.evaluate(
        model_params=model_params,
        pid_params=pid_params,
        method=method.lower(),
        method_confidence=method_confidence,
        method_confidence_details={
            "source": "model_identification_residue",
            "quality": model_confidence.get("quality", "unknown"),
            "recommendation": model_confidence.get("recommendation", ""),
        },
        dt=dt,
        loop_type=loop_type,
        sp_initial=50.0,
        sp_final=60.0,
        n_steps=500,
    )
    reverse = ModelRating.evaluate(
        model_params=model_params,
        pid_params=pid_params,
        method=f"{method.lower()}_reverse",
        method_confidence=method_confidence,
        method_confidence_details={"source": "acceptance_reverse"},
        dt=dt,
        loop_type=loop_type,
        sp_initial=60.0,
        sp_final=50.0,
        n_steps=500,
    )
    disturbance = _evaluate_disturbance_rejection(
        model_params=model_params,
        pid_params=pid_params,
        dt=dt,
        loop_type=loop_type,
    )
    robustness = _evaluate_robustness(
        model_params=model_params,
        pid_params=pid_params,
        method_confidence=method_confidence,
        dt=dt,
        loop_type=loop_type,
        method=method,
    )
    constraints = _constraint_score_from_simulation(forward.get("simulation", {}) or {})

    acceptance_performance_score = _clamp(
        0.45 * _safe_float(forward.get("performance_score"))
        + 0.20 * _safe_float(reverse.get("performance_score"))
        + 0.35 * _safe_float(disturbance.get("tracking_score")),
        0.0,
        10.0,
    )
    confidence_as_score = _clamp(method_confidence * 10.0, 0.0, 10.0)
    online_readiness_score = _clamp(
        0.50 * acceptance_performance_score
        + 0.20 * _safe_float(robustness.get("robustness_score"))
        + 0.15 * _safe_float(constraints.get("constraint_score"))
        + 0.15 * confidence_as_score,
        0.0,
        10.0,
    )

    major_stable = bool((forward.get("simulation") or {}).get("is_stable", False)) and bool(
        (reverse.get("simulation") or {}).get("is_stable", False)
    )
    severe_constraint_risk = _safe_float(constraints.get("saturation_pct")) >= 35.0
    passed = bool(online_readiness_score >= 7.0 and major_stable and not severe_constraint_risk)

    final_details = {
        "acceptance_performance_score": round(acceptance_performance_score, 2),
        "robustness_score": round(_safe_float(robustness.get("robustness_score")), 2),
        "constraint_score": round(_safe_float(constraints.get("constraint_score")), 2),
        "confidence_as_score": round(confidence_as_score, 2),
        "online_readiness_score": round(online_readiness_score, 2),
        "stable_ratio": robustness.get("stable_ratio", 0.0),
    }

    return {
        "performance_score": round(acceptance_performance_score, 2),
        "acceptance_performance_score": round(acceptance_performance_score, 2),
        "method_confidence": method_confidence,
        "robustness_score": round(_safe_float(robustness.get("robustness_score")), 2),
        "constraint_score": round(_safe_float(constraints.get("constraint_score")), 2),
        "final_rating": round(online_readiness_score, 2),
        "online_readiness_score": round(online_readiness_score, 2),
        "passed": passed,
        "performance_details": forward.get("performance_details", {}) or {},
        "final_details": final_details,
        "simulation": forward.get("simulation", {}) or {},
        "scenario_evaluations": {
            "forward_step": {
                "performance_score": round(_safe_float(forward.get("performance_score")), 2),
                "performance_details": forward.get("performance_details", {}) or {},
            },
            "reverse_step": {
                "performance_score": round(_safe_float(reverse.get("performance_score")), 2),
                "performance_details": reverse.get("performance_details", {}) or {},
            },
            "disturbance_rejection": disturbance,
            "parameter_perturbation": robustness,
            "constraint_check": constraints,
        },
        "launch_recommendation": (
            "建议进入受控条件下的小扰动试投。"
            if passed and online_readiness_score >= 8.5
            else "建议保守试投，并保留人工确认。"
            if passed
            else "暂不建议直接上线，建议继续回流优化。"
        ),
    }


def choose_best_evaluation_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for candidate in candidates:
        if best is None:
            best = candidate
            continue
        current_eval = candidate.get("evaluation_result") or {}
        best_eval = best.get("evaluation_result") or {}
        current_key = (
            _safe_float(current_eval.get("online_readiness_score"), _safe_float(current_eval.get("final_rating"))),
            _safe_float(current_eval.get("acceptance_performance_score"), _safe_float(current_eval.get("performance_score"))),
            _safe_float(current_eval.get("robustness_score")),
            _safe_float(current_eval.get("constraint_score")),
        )
        best_key = (
            _safe_float(best_eval.get("online_readiness_score"), _safe_float(best_eval.get("final_rating"))),
            _safe_float(best_eval.get("acceptance_performance_score"), _safe_float(best_eval.get("performance_score"))),
            _safe_float(best_eval.get("robustness_score")),
            _safe_float(best_eval.get("constraint_score")),
        )
        if current_key > best_key:
            best = candidate
    return best or {}


def evaluate_pid_model(
    *,
    K: float,
    T: float,
    L: float,
    Kp: float,
    Ki: float,
    Kd: float,
    method: str,
    method_confidence: float,
    model_confidence: Dict[str, Any],
    dt: float,
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model_params = _build_model_params(
        K=K,
        T=T,
        L=L,
        model_type=model_type,
        selected_model_params=selected_model_params,
    )
    return ModelRating.evaluate(
        model_params=model_params,
        pid_params={"Kp": Kp, "Ki": Ki, "Kd": Kd},
        method=method.lower(),
        method_confidence=method_confidence,
        method_confidence_details={
            "source": "model_identification_residue",
            "quality": model_confidence.get("quality", "unknown"),
            "recommendation": model_confidence.get("recommendation", ""),
        },
        dt=dt,
    )


def diagnose_evaluation_failure(
    *,
    eval_result: Dict[str, Any],
    model_r2: float,
    model_rmse: float,
    candidate_window_count: int,
) -> Dict[str, str]:
    performance_details = eval_result.get("performance_details") or {}
    performance_score = _safe_float(
        eval_result.get("acceptance_performance_score", eval_result.get("performance_score"))
    )
    method_confidence = _safe_float(eval_result.get("method_confidence"))
    online_readiness_score = _safe_float(
        eval_result.get("online_readiness_score", eval_result.get("final_rating"))
    )
    robustness_score = _safe_float(eval_result.get("robustness_score"))
    constraint_score = _safe_float(eval_result.get("constraint_score"))
    overshoot = _safe_float(performance_details.get("overshoot"))
    settling_time = _safe_float(performance_details.get("settling_time"), -1.0)
    steady_state_error = _safe_float(performance_details.get("steady_state_error"))
    oscillation_count = int(performance_details.get("oscillation_count", 0) or 0)
    decay_ratio = _safe_float(performance_details.get("decay_ratio"))
    is_stable = bool(performance_details.get("is_stable", True))

    if constraint_score < 6.0:
        return {
            "failure_reason": "当前方案存在较高的控制约束风险，输出容易贴边或动作过于剧烈。",
            "feedback_target": "pid_expert",
            "feedback_action": "请在当前模型基础上生成更保守的 PID 候选，优先降低饱和风险和 MV 剧烈波动。",
        }

    if not is_stable or overshoot > 40 or oscillation_count > 20 or decay_ratio >= 0.8:
        return {
            "failure_reason": "当前方案闭环过于激进，仍存在明显超调、振荡或收敛不足。",
            "feedback_target": "pid_expert",
            "feedback_action": "请在当前模型基础上输出更保守的候选参数，优先压低超调、振荡和调节时间。",
        }

    if robustness_score < 6.0 or method_confidence < 0.7 or model_r2 < 0.8 or model_rmse > 0.1:
        return {
            "failure_reason": "当前模型在参数摄动下鲁棒性不足，或辨识可信度仍偏低。",
            "feedback_target": "system_id_expert",
            "feedback_action": "请复核候选辨识模型与窗口，优先确认更稳健的 K/T/L 后再重新整定。",
        }

    if candidate_window_count > 1 and performance_score < 6.0 and online_readiness_score < 6.5:
        return {
            "failure_reason": "当前 shortlist 方案整体表现偏弱，可能仍需回到更合适的数据窗口。",
            "feedback_target": "data_analyst",
            "feedback_action": "请复查候选窗口，优先保留响应更完整、扰动更少的区间再重新进入辨识。",
        }

    if settling_time < 0 or steady_state_error > 5.0:
        return {
            "failure_reason": "当前方案调节偏慢或稳态误差偏大，仍需继续优化控制参数。",
            "feedback_target": "pid_expert",
            "feedback_action": "请继续优化比例与积分参数，提高收敛质量并降低稳态误差。",
        }

    return {
        "failure_reason": "综合验收分未达到上线阈值，建议先从 PID 参数与方案选择继续收敛。",
        "feedback_target": "pid_expert",
        "feedback_action": "请基于当前 shortlist 继续收敛更保守、更稳健的整定候选。",
    }


def build_initial_assessment(
    *,
    eval_result: Dict[str, Any],
    pass_threshold: float,
    diagnosis: Dict[str, str],
    evaluated_pid: Dict[str, float],
) -> Dict[str, Any]:
    online_readiness = _safe_float(eval_result.get("online_readiness_score", eval_result.get("final_rating")))
    passed = bool(eval_result.get("passed", False)) if "passed" in eval_result else bool(online_readiness >= pass_threshold)
    return {
        "passed": passed,
        "pass_threshold": pass_threshold,
        "failure_reason": diagnosis.get("failure_reason", ""),
        "feedback_target": diagnosis.get("feedback_target", ""),
        "feedback_action": diagnosis.get("feedback_action", ""),
        "evaluation_result": {
            "performance_score": _safe_float(
                eval_result.get("acceptance_performance_score", eval_result.get("performance_score"))
            ),
            "method_confidence": _safe_float(eval_result.get("method_confidence")),
            "final_rating": online_readiness,
            "acceptance_performance_score": _safe_float(
                eval_result.get("acceptance_performance_score", eval_result.get("performance_score"))
            ),
            "robustness_score": _safe_float(eval_result.get("robustness_score")),
            "constraint_score": _safe_float(eval_result.get("constraint_score")),
            "online_readiness_score": online_readiness,
        },
        "evaluated_pid": {
            "Kp": float(evaluated_pid.get("Kp", 0.0)),
            "Ki": float(evaluated_pid.get("Ki", 0.0)),
            "Kd": float(evaluated_pid.get("Kd", 0.0)),
        },
    }


def choose_alternative_model_attempt(
    *,
    attempts: List[Dict[str, Any]],
    current_source: str,
    candidate_map: Dict[str, Dict[str, Any]],
    loop_type: str,
    dt: float,
    pass_threshold: float,
    benchmark_fn: Callable[..., Dict[str, Any]],
    refine_fn: Callable[[Dict[str, float], Dict[str, float], float, float, str], Dict[str, Any]],
) -> Dict[str, Any]:
    if len(attempts) <= 1:
        return {}

    best_result: Dict[str, Any] | None = None
    for attempt in attempts:
        source = str(attempt.get("window_source", ""))
        if source == current_source or source not in candidate_map:
            continue

        K = float(attempt["K"])
        T = float(attempt["T"])
        L = float(attempt["L"])
        model_type = str(attempt.get("model_type", "FOPDT")).upper()
        selected_model_params = dict(attempt.get("selected_model_params") or {})
        confidence_score = float(attempt.get("confidence", 0.0))
        benchmark = benchmark_fn(K, T, L, dt, confidence_score, model_type, selected_model_params)
        best_strategy = benchmark.get("best") or {}
        if not best_strategy:
            continue

        model_params = _build_model_params(
            K=K,
            T=T,
            L=L,
            model_type=model_type,
            selected_model_params=selected_model_params,
        )
        refined = refine_fn(
            model_params,
            {
                "Kp": float(best_strategy["Kp"]),
                "Ki": float(best_strategy["Ki"]),
                "Kd": float(best_strategy["Kd"]),
            },
            confidence_score,
            dt,
            str(best_strategy.get("strategy", "auto")),
        )
        refined_best = refined.get("best") or {}
        final_eval = refined_best.get("evaluation_result") if refined_best else benchmark.get("best_evaluation") or {}
        if not final_eval:
            continue

        result = {
            "window_source": source,
            "loop_type": loop_type,
            "model_type": model_type,
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

        better_score = float(result["evaluation_result"].get("final_rating", 0.0)) > float(
            best_result["evaluation_result"].get("final_rating", 0.0)
        ) + 1e-9
        tie_break = (
            abs(
                float(result["evaluation_result"].get("final_rating", 0.0))
                - float(best_result["evaluation_result"].get("final_rating", 0.0))
            )
            <= 1e-9
            and float(result["evaluation_result"].get("performance_score", 0.0))
            > float(best_result["evaluation_result"].get("performance_score", 0.0)) + 1e-9
        )
        if better_score or tie_break:
            best_result = result

    return best_result or {}
