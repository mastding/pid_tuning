from __future__ import annotations

from typing import Any, Dict, List

from skills.pid_tuning_skills import apply_tuning_rules, select_tuning_strategy
from skills.rating import ModelRating


def benchmark_pid_strategies(K: float, T: float, L: float, dt: float, confidence_score: float) -> Dict[str, Any]:
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


def refine_pid_for_performance(
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


def select_best_pid_strategy(
    *,
    K: float,
    T: float,
    L: float,
    loop_type: str,
    confidence_score: float,
    normalized_rmse: float,
    r2_score: float,
    dt: float,
) -> Dict[str, Any]:
    tau_ratio = max(float(L), 0.0) / max(float(T), 1e-6)
    heuristic_selection = select_tuning_strategy(
        loop_type=loop_type,
        K=K,
        T=T,
        L=L,
        model_confidence=confidence_score,
        r2_score=r2_score,
        normalized_rmse=normalized_rmse,
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
            dt=dt,
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
        raise ValueError("Failed to generate a usable PID candidate")

    pid_params = apply_tuning_rules(K, T, L, best_candidate["strategy"])
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
    selection_reason = (
        f"已对 {', '.join(candidate_strategies)} 进行闭环试算，"
        f"最终选择 performance_score 最高的 {best_candidate['strategy']}。"
    )
    selection_inputs = {
        "loop_type": loop_type,
        "model_confidence": confidence_score,
        "normalized_rmse": normalized_rmse,
        "r2_score": r2_score,
        "tau_ratio": tau_ratio,
        "K": float(K),
        "T": float(T),
        "L": float(L),
        "heuristic_strategy": heuristic_selection["strategy"],
    }
    return {
        "heuristic_selection": heuristic_selection,
        "candidate_strategies": candidate_strategies,
        "candidate_results": candidate_results,
        "public_candidate_results": public_candidate_results,
        "best_candidate": best_candidate,
        "pid_params": pid_params,
        "selection_reason": selection_reason,
        "selection_inputs": selection_inputs,
    }

