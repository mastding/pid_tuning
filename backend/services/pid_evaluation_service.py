from __future__ import annotations

from typing import Any, Callable, Dict

from skills.rating import ModelRating


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
    selected_model_params = selected_model_params or {}
    normalized_type = str(model_type or selected_model_params.get("model_type", "FOPDT")).upper()
    if normalized_type == "SOPDT":
        model_params = {
            "K": float(selected_model_params.get("K", K)),
            "T1": float(selected_model_params.get("T1", T)),
            "T2": float(selected_model_params.get("T2", T)),
            "L": float(selected_model_params.get("L", L)),
        }
    else:
        model_params = {"K": K, "T1": T, "T2": 0.0, "L": L}

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
    performance_score = float(eval_result.get("performance_score", 0.0) or 0.0)
    method_confidence = float(eval_result.get("method_confidence", 0.0) or 0.0)
    overshoot = float(performance_details.get("overshoot", 0.0) or 0.0)
    settling_time = float(performance_details.get("settling_time", -1.0) or -1.0)
    steady_state_error = float(performance_details.get("steady_state_error", 0.0) or 0.0)
    oscillation_count = int(performance_details.get("oscillation_count", 0) or 0)
    decay_ratio = float(performance_details.get("decay_ratio", 0.0) or 0.0)
    is_stable = bool(performance_details.get("is_stable", True))

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


def build_initial_assessment(
    *,
    eval_result: Dict[str, Any],
    pass_threshold: float,
    diagnosis: Dict[str, str],
    evaluated_pid: Dict[str, float],
) -> Dict[str, Any]:
    passed = bool(float(eval_result.get("final_rating", 0.0) or 0.0) >= pass_threshold)
    return {
        "passed": passed,
        "pass_threshold": pass_threshold,
        "failure_reason": diagnosis.get("failure_reason", ""),
        "feedback_target": diagnosis.get("feedback_target", ""),
        "feedback_action": diagnosis.get("feedback_action", ""),
        "evaluation_result": {
            "performance_score": float(eval_result.get("performance_score", 0.0) or 0.0),
            "method_confidence": float(eval_result.get("method_confidence", 0.0) or 0.0),
            "final_rating": float(eval_result.get("final_rating", 0.0) or 0.0),
        },
        "evaluated_pid": {
            "Kp": float(evaluated_pid.get("Kp", 0.0)),
            "Ki": float(evaluated_pid.get("Ki", 0.0)),
            "Kd": float(evaluated_pid.get("Kd", 0.0)),
        },
    }


def choose_alternative_model_attempt(
    *,
    attempts: list[Dict[str, Any]],
    current_source: str,
    candidate_map: Dict[str, Dict[str, Any]],
    loop_type: str,
    dt: float,
    pass_threshold: float,
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
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
        confidence_score = float(attempt.get("confidence", 0.0))
        benchmark = benchmark_fn(K, T, L, dt, confidence_score)
        best_strategy = benchmark.get("best") or {}
        if not best_strategy:
            continue

        refined = refine_fn(
            {"K": K, "T1": T, "T2": 0.0, "L": L},
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
