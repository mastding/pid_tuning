from __future__ import annotations

from typing import Any, Dict, List

from memory.experience_service import describe_experience_guidance
from skills.pid_tuning_skills import apply_tuning_rules, select_tuning_strategy
from skills.rating import ModelRating


ALL_STRATEGIES = ["IMC", "LAMBDA", "ZN", "CHR"]


def _evaluate_strategy(
    *,
    K: float,
    T: float,
    L: float,
    dt: float,
    confidence_score: float,
    strategy_name: str,
    experience_bonus: float = 0.0,
) -> Dict[str, Any]:
    pid_params = apply_tuning_rules(K, T, L, strategy_name)
    eval_result = ModelRating.evaluate(
        model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
        pid_params={"Kp": pid_params["Kp"], "Ki": pid_params["Ki"], "Kd": pid_params["Kd"]},
        method=strategy_name.lower(),
        method_confidence=confidence_score,
        method_confidence_details={"source": "model_identification_confidence"},
        dt=dt,
    )
    return {
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
        "experience_bonus": experience_bonus,
        "evaluation_result": eval_result,
    }


def _evaluate_pid_params(
    *,
    K: float,
    T: float,
    L: float,
    dt: float,
    confidence_score: float,
    strategy_name: str,
    pid_params: Dict[str, Any],
    description_suffix: str = "",
    experience_bonus: float = 0.0,
) -> Dict[str, Any]:
    eval_result = ModelRating.evaluate(
        model_params={"K": K, "T1": T, "T2": 0.0, "L": L},
        pid_params={"Kp": pid_params["Kp"], "Ki": pid_params["Ki"], "Kd": pid_params["Kd"]},
        method=strategy_name.lower(),
        method_confidence=confidence_score,
        method_confidence_details={"source": "model_identification_confidence"},
        dt=dt,
    )
    return {
        "strategy": strategy_name,
        "Kp": float(pid_params["Kp"]),
        "Ki": float(pid_params["Ki"]),
        "Kd": float(pid_params["Kd"]),
        "Ti": float(pid_params.get("Ti", 0.0)),
        "Td": float(pid_params.get("Td", 0.0)),
        "description": f"{pid_params.get('description', strategy_name)}{description_suffix}",
        "performance_score": float(eval_result["performance_score"]),
        "final_rating": float(eval_result.get("final_rating", 0.0)),
        "is_stable": bool(eval_result["simulation"].get("is_stable", False)),
        "experience_bonus": experience_bonus,
        "evaluation_result": eval_result,
    }


def _is_better_candidate(candidate: Dict[str, Any], best: Dict[str, Any] | None) -> bool:
    if best is None:
        return True

    candidate_composite = candidate["performance_score"] + candidate.get("experience_bonus", 0.0)
    best_composite = best["performance_score"] + best.get("experience_bonus", 0.0)
    if candidate_composite > best_composite + 1e-9:
        return True
    if abs(candidate_composite - best_composite) <= 1e-9:
        if candidate["final_rating"] > best["final_rating"] + 1e-9:
            return True
        if (
            abs(candidate["final_rating"] - best["final_rating"]) <= 1e-9
            and candidate["is_stable"]
            and not best["is_stable"]
        ):
            return True
    return False


def benchmark_pid_strategies(K: float, T: float, L: float, dt: float, confidence_score: float) -> Dict[str, Any]:
    best_candidate: Dict[str, Any] | None = None
    best_evaluation: Dict[str, Any] | None = None
    summaries: List[Dict[str, Any]] = []

    for strategy_name in ALL_STRATEGIES:
        candidate = _evaluate_strategy(
            K=K,
            T=T,
            L=L,
            dt=dt,
            confidence_score=confidence_score,
            strategy_name=strategy_name,
        )
        summaries.append(
            {
                "strategy": candidate["strategy"],
                "Kp": candidate["Kp"],
                "Ki": candidate["Ki"],
                "Kd": candidate["Kd"],
                "performance_score": candidate["performance_score"],
                "final_rating": candidate["final_rating"],
                "is_stable": candidate["is_stable"],
            }
        )
        if _is_better_candidate(candidate, best_candidate):
            best_candidate = candidate
            best_evaluation = candidate["evaluation_result"]

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
    experience_guidance: Dict[str, Any] | None = None,
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

    preferred_strategy = str((experience_guidance or {}).get("preferred_strategy", "")).upper()
    preferred_matches = (experience_guidance or {}).get("matches") or []
    summary = (experience_guidance or {}).get("summary") or {}
    guidance_text = describe_experience_guidance(experience_guidance or {})
    recommended_kp_scale = float(summary.get("recommended_kp_scale", 1.0) or 1.0)
    recommended_ki_scale = float(summary.get("recommended_ki_scale", 1.0) or 1.0)
    recommended_kd_scale = float(summary.get("recommended_kd_scale", 1.0) or 1.0)
    preferred_refine_pattern = str(summary.get("preferred_refine_pattern", ""))

    prioritized: List[str] = []
    for strategy_name in [preferred_strategy, str(heuristic_selection.get("strategy", "")).upper()]:
        if strategy_name in ALL_STRATEGIES and strategy_name not in prioritized:
            prioritized.append(strategy_name)

    if not prioritized:
        prioritized = [str(heuristic_selection.get("strategy", "IMC")).upper()]

    candidate_results: List[Dict[str, Any]] = []
    best_candidate: Dict[str, Any] | None = None

    def evaluate_group(strategies: List[str]) -> None:
        nonlocal best_candidate
        for strategy_name in strategies:
            experience_bonus = 0.15 if preferred_strategy and strategy_name == preferred_strategy else 0.0
            candidate = _evaluate_strategy(
                K=K,
                T=T,
                L=L,
                dt=dt,
                confidence_score=confidence_score,
                strategy_name=strategy_name,
                experience_bonus=experience_bonus,
            )
            candidate_results.append(candidate)
            if _is_better_candidate(candidate, best_candidate):
                best_candidate = candidate

            should_try_refined = (
                preferred_refine_pattern
                and strategy_name in {preferred_strategy, str(heuristic_selection.get("strategy", "")).upper()}
                and (
                    abs(recommended_kp_scale - 1.0) > 1e-3
                    or abs(recommended_ki_scale - 1.0) > 1e-3
                    or abs(recommended_kd_scale - 1.0) > 1e-3
                )
            )
            if should_try_refined:
                refined_pid = {
                    "Kp": candidate["Kp"] * recommended_kp_scale,
                    "Ki": candidate["Ki"] * recommended_ki_scale,
                    "Kd": candidate["Kd"] * recommended_kd_scale,
                    "Ti": candidate["Ti"],
                    "Td": candidate["Td"],
                    "description": f"{candidate['description']} + history_refine",
                }
                refined_candidate = _evaluate_pid_params(
                    K=K,
                    T=T,
                    L=L,
                    dt=dt,
                    confidence_score=confidence_score,
                    strategy_name=strategy_name,
                    pid_params=refined_pid,
                    description_suffix=" (history_refined)",
                    experience_bonus=experience_bonus + 0.1,
                )
                refined_candidate["history_refined"] = True
                refined_candidate["refine_scales"] = {
                    "kp_scale": recommended_kp_scale,
                    "ki_scale": recommended_ki_scale,
                    "kd_scale": recommended_kd_scale,
                }
                candidate_results.append(refined_candidate)
                if _is_better_candidate(refined_candidate, best_candidate):
                    best_candidate = refined_candidate

    evaluate_group(prioritized)

    full_benchmark_triggered = (
        best_candidate is None
        or float(best_candidate.get("performance_score", 0.0)) < 7.0
        or float(best_candidate.get("final_rating", 0.0)) < 7.0
        or not bool(best_candidate.get("is_stable", False))
    )

    if full_benchmark_triggered:
        remaining = [strategy for strategy in ALL_STRATEGIES if strategy not in prioritized]
        evaluate_group(remaining)

    if best_candidate is None:
        raise ValueError("Failed to generate a usable PID candidate")

    pid_params = apply_tuning_rules(K, T, L, best_candidate["strategy"])
    if best_candidate.get("history_refined"):
        pid_params = {
            **pid_params,
            "Kp": float(best_candidate["Kp"]),
            "Ki": float(best_candidate["Ki"]),
            "Kd": float(best_candidate["Kd"]),
            "Ti": float(best_candidate["Ti"]),
            "Td": float(best_candidate["Td"]),
            "description": str(best_candidate.get("description", pid_params.get("description", ""))),
        }
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
            "experience_bonus": item["experience_bonus"],
            "history_refined": bool(item.get("history_refined", False)),
        }
        for item in candidate_results
    ]

    if full_benchmark_triggered:
        selection_reason = (
            f"先优先试算历史偏好策略与启发式策略（{', '.join(prioritized)}），"
            f"因评分未达标，已扩展为 {', '.join(ALL_STRATEGIES)} 的全量闭环试算，"
            f"最终选择 performance_score 最高的 {best_candidate['strategy']}。"
        )
    else:
        selection_reason = (
            f"先优先试算历史偏好策略与启发式策略（{', '.join(prioritized)}），"
            f"无需展开全量 benchmark，直接选择 {best_candidate['strategy']}。"
        )
    if guidance_text:
        selection_reason += f" {guidance_text}"

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
        "experience_preferred_strategy": preferred_strategy,
        "experience_match_count": len(preferred_matches) if isinstance(preferred_matches, list) else 0,
        "preferred_refine_pattern": preferred_refine_pattern,
        "recommended_kp_scale": recommended_kp_scale,
        "recommended_ki_scale": recommended_ki_scale,
        "recommended_kd_scale": recommended_kd_scale,
        "tested_strategies": [item["strategy"] for item in candidate_results],
        "full_benchmark_triggered": full_benchmark_triggered,
    }
    return {
        "heuristic_selection": heuristic_selection,
        "candidate_strategies": [item["strategy"] for item in candidate_results],
        "candidate_results": candidate_results,
        "public_candidate_results": public_candidate_results,
        "best_candidate": best_candidate,
        "pid_params": pid_params,
        "selection_reason": selection_reason,
        "selection_inputs": selection_inputs,
        "experience_guidance": experience_guidance or {},
    }
