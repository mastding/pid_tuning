from __future__ import annotations

from typing import Any, Dict, List

from memory.experience_service import describe_experience_guidance
from skills.pid_tuning_skills import apply_tuning_rules, select_tuning_strategy
from skills.rating import ModelRating


ALL_STRATEGIES = ["IMC", "LAMBDA", "ZN", "CHR"]


def _normalize_strategy_hint(strategy: str) -> str:
    strategy_name = str(strategy or "").strip().upper()
    if strategy_name in ALL_STRATEGIES:
        return strategy_name
    if strategy_name in {"PI", "CONSERVATIVE_PI", "VERY_CONSERVATIVE_PI", "LAMBDA_TUNING"}:
        return "LAMBDA"
    if strategy_name in {"PID", "CONSERVATIVE_PID"}:
        return "IMC"
    return ""


def _apply_knowledge_bias_to_pid_params(
    *,
    pid_params: Dict[str, Any],
    tuning_bias: Dict[str, Any] | None,
    strategy_name: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not tuning_bias:
        return dict(pid_params), {}

    adjusted = dict(pid_params)
    applied: Dict[str, Any] = {}
    kp_scale_max = float(tuning_bias.get("kp_scale_max", 1.0) or 1.0)
    ki_scale_max = float(tuning_bias.get("ki_scale_max", 1.0) or 1.0)
    kd_scale_max = float(tuning_bias.get("kd_scale_max", 1.0) or 1.0)
    discourage_derivative = bool(tuning_bias.get("discourage_derivative"))
    conservative_mode = bool(tuning_bias.get("conservative_mode"))

    if kp_scale_max < 0.999:
        adjusted["Kp"] = float(adjusted["Kp"]) * kp_scale_max
        applied["kp_scale_max"] = kp_scale_max
    if ki_scale_max < 0.999:
        adjusted["Ki"] = float(adjusted["Ki"]) * ki_scale_max
        applied["ki_scale_max"] = ki_scale_max
    if discourage_derivative:
        adjusted["Kd"] = min(float(adjusted["Kd"]), float(adjusted["Kd"]) * kd_scale_max)
        applied["discourage_derivative"] = True
        if kd_scale_max < 0.999:
            applied["kd_scale_max"] = kd_scale_max
    elif kd_scale_max < 0.999 and abs(float(adjusted.get("Kd", 0.0))) > 1e-9:
        adjusted["Kd"] = float(adjusted["Kd"]) * kd_scale_max
        applied["kd_scale_max"] = kd_scale_max

    if conservative_mode and str(strategy_name or "").upper() in {"ZN", "CHR"}:
        adjusted["Kp"] = float(adjusted["Kp"]) * 0.92
        adjusted["Ki"] = float(adjusted["Ki"]) * 0.9
        applied["conservative_strategy_trim"] = str(strategy_name or "").upper()

    return adjusted, applied


def _normalize_model_params(
    *,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
    K: float,
    T: float,
    L: float,
) -> Dict[str, Any]:
    params = dict(selected_model_params or {})
    normalized_model_type = str(model_type or params.get("model_type", "FOPDT")).upper()
    params["model_type"] = normalized_model_type

    if normalized_model_type == "SOPDT":
        params.setdefault("K", float(K))
        params.setdefault("T1", float(params.get("T", T)))
        params.setdefault("T2", float(params.get("T", T)))
        params.setdefault("L", float(L))
        return params
    if normalized_model_type == "IPDT":
        params.setdefault("K", float(K))
        params.setdefault("L", float(L))
        return params
    if normalized_model_type == "FO":
        params.setdefault("K", float(K))
        params.setdefault("T", float(T))
        return params

    params.setdefault("K", float(K))
    params.setdefault("T", float(T))
    params.setdefault("L", float(L))
    return params


def _derive_primary_tuning_inputs(
    *,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
    K: float,
    T: float,
    L: float,
) -> Dict[str, Any]:
    params = _normalize_model_params(
        model_type=model_type,
        selected_model_params=selected_model_params,
        K=K,
        T=T,
        L=L,
    )
    normalized_model_type = str(params.get("model_type", model_type or "FOPDT")).upper()

    if normalized_model_type == "SOPDT":
        t1 = max(float(params.get("T1", params.get("T", T))), 1e-6)
        t2 = max(float(params.get("T2", params.get("T", T))), 1e-6)
        dominant_tau = max(t1, t2)
        secondary_tau = min(t1, t2)
        tau_ratio = secondary_tau / max(dominant_tau, 1e-6)
        shape_index = min(max(tau_ratio, 0.0), 1.0)
        apparent_order = 1.0 + shape_index
        raw_l = max(float(params.get("L", L)), 0.0)
        aggregate_tau = dominant_tau + secondary_tau
        t_work = dominant_tau + secondary_tau * (0.55 + 0.25 * shape_index)
        l_work = raw_l + secondary_tau * (0.35 + 0.20 * shape_index)
        return {
            "model_type": normalized_model_type,
            "K": float(params.get("K", K)),
            "T1": float(t1),
            "T2": float(t2),
            "L": float(raw_l),
            "selected_model_params": params,
            "shape_index": float(shape_index),
            "apparent_order": float(apparent_order),
            "tau_ratio": float(raw_l / max(aggregate_tau, 1e-6)),
            "aggregate_tau": float(max(aggregate_tau, 1e-6)),
            "T_work": float(max(t_work, 1e-6)),
            "L_work": float(max(l_work, 0.0)),
        }

    if normalized_model_type == "IPDT":
        lag_value = max(float(params.get("L", L)), 1e-6)
        effective_t = max(float(T), lag_value)
        return {
            "model_type": normalized_model_type,
            "K": float(params.get("K", K)),
            "L": float(lag_value),
            "selected_model_params": params,
            "tau_ratio": float(lag_value / max(effective_t, 1e-6)),
            "T_work": float(effective_t),
        }

    if normalized_model_type == "FO":
        t_value = max(float(params.get("T", T)), 1e-6)
        return {
            "model_type": normalized_model_type,
            "K": float(params.get("K", K)),
            "T": float(t_value),
            "L": 0.0,
            "selected_model_params": params,
            "tau_ratio": 0.0,
        }

    t_value = max(float(params.get("T", T)), 1e-6)
    l_value = max(float(params.get("L", L)), 0.0)
    return {
        "model_type": normalized_model_type,
        "K": float(params.get("K", K)),
        "T": float(t_value),
        "L": float(l_value),
        "selected_model_params": params,
        "tau_ratio": float(l_value / max(t_value, 1e-6)),
    }


def _build_history_seed_pid(
    *,
    top_match: Dict[str, Any] | None,
    base_candidate: Dict[str, Any],
    K: float,
    T: float,
    L: float,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not top_match:
        return None

    top_pid = dict((top_match.get("pid") or {}).get("final") or {})
    if not top_pid:
        return None

    top_model = dict(top_match.get("model") or {})
    top_selected = dict(top_model.get("selected_model_params") or {})
    normalized_model_type = str(model_type or "FOPDT").upper()
    if str(top_match.get("model_type", top_model.get("model_type", "FOPDT"))).upper() != normalized_model_type:
        return None

    k_scale = max(abs(float(K)), 1e-6) / max(abs(float(top_model.get("K", K))), 1e-6)
    t_scale = 1.0
    l_scale = 1.0
    if normalized_model_type == "SOPDT":
        current_t_sum = max(float((selected_model_params or {}).get("T1", 0.0)) + float((selected_model_params or {}).get("T2", 0.0)), 1e-6)
        ref_t_sum = max(float(top_selected.get("T1", 0.0)) + float(top_selected.get("T2", 0.0)), 1e-6)
        t_scale = current_t_sum / ref_t_sum
        l_scale = max(float((selected_model_params or {}).get("L", L)), 1e-6) / max(float(top_selected.get("L", top_model.get("L", L))), 1e-6)
    elif normalized_model_type == "IPDT":
        t_scale = 1.0
        l_scale = max(float((selected_model_params or {}).get("L", L)), 1e-6) / max(float(top_selected.get("L", top_model.get("L", L))), 1e-6)
    else:
        t_scale = max(float(T), 1e-6) / max(float(top_model.get("T", T)), 1e-6)
        l_scale = max(float(L), 1e-6) / max(float(top_model.get("L", L)), 1e-6)

    seed_kp = float(top_pid.get("Kp", base_candidate["Kp"])) * k_scale
    seed_ki = float(top_pid.get("Ki", base_candidate["Ki"])) * min(max(t_scale, 0.35), 2.0)
    seed_kd = float(top_pid.get("Kd", base_candidate["Kd"])) * min(max(l_scale, 0.35), 2.0)
    if seed_kd < 1e-9:
        seed_kd = 0.0

    return {
        "Kp": seed_kp,
        "Ki": seed_ki,
        "Kd": seed_kd,
        "Ti": float(base_candidate["Ti"]),
        "Td": float(base_candidate["Td"]),
        "description": f"{base_candidate['description']} + history_seed",
    }


def _build_model_params_for_evaluation(
    *,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
    K: float,
    T: float,
    L: float,
) -> Dict[str, float]:
    selected_model_params = selected_model_params or {}
    normalized_model_type = str(model_type or selected_model_params.get("model_type", "FOPDT")).upper()

    if normalized_model_type == "SOPDT":
        return {
            "model_type": "SOPDT",
            "K": float(selected_model_params.get("K", K)),
            "T1": float(selected_model_params.get("T1", selected_model_params.get("T", T))),
            "T2": float(selected_model_params.get("T2", selected_model_params.get("T", T))),
            "L": float(selected_model_params.get("L", L)),
        }
    if normalized_model_type == "IPDT":
        return {
            "model_type": "IPDT",
            "K": float(selected_model_params.get("K", K)),
            "L": max(float(selected_model_params.get("L", L)), 1e-3),
        }
    if normalized_model_type == "FO":
        return {
            "model_type": "FO",
            "K": float(selected_model_params.get("K", K)),
            "T1": float(selected_model_params.get("T", T)),
            "T2": 0.0,
            "L": 0.0,
        }
    return {"model_type": "FOPDT", "K": float(K), "T1": float(T), "T2": 0.0, "L": float(L)}


def _evaluate_strategy(
    *,
    K: float,
    T: float,
    L: float,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
    dt: float,
    confidence_score: float,
    strategy_name: str,
    experience_bonus: float = 0.0,
    tuning_bias: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pid_params = apply_tuning_rules(K, T, L, strategy_name, model_type=model_type, model_params=selected_model_params)
    pid_params, guidance_applied = _apply_knowledge_bias_to_pid_params(
        pid_params=pid_params,
        tuning_bias=tuning_bias,
        strategy_name=strategy_name,
    )
    model_params = _build_model_params_for_evaluation(
        model_type=model_type,
        selected_model_params=selected_model_params,
        K=K,
        T=T,
        L=L,
    )
    eval_result = ModelRating.evaluate(
        model_params=model_params,
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
        "knowledge_constraints_applied": guidance_applied,
    }


def _evaluate_pid_params(
    *,
    K: float,
    T: float,
    L: float,
    model_type: str,
    selected_model_params: Dict[str, Any] | None,
    dt: float,
    confidence_score: float,
    strategy_name: str,
    pid_params: Dict[str, Any],
    description_suffix: str = "",
    experience_bonus: float = 0.0,
    tuning_bias: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pid_params, guidance_applied = _apply_knowledge_bias_to_pid_params(
        pid_params=pid_params,
        tuning_bias=tuning_bias,
        strategy_name=strategy_name,
    )
    model_params = _build_model_params_for_evaluation(
        model_type=model_type,
        selected_model_params=selected_model_params,
        K=K,
        T=T,
        L=L,
    )
    eval_result = ModelRating.evaluate(
        model_params=model_params,
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
        "knowledge_constraints_applied": guidance_applied,
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


def benchmark_pid_strategies(
    K: float,
    T: float,
    L: float,
    dt: float,
    confidence_score: float,
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    best_candidate: Dict[str, Any] | None = None
    best_evaluation: Dict[str, Any] | None = None
    summaries: List[Dict[str, Any]] = []

    for strategy_name in ALL_STRATEGIES:
        candidate = _evaluate_strategy(
            K=K,
            T=T,
            L=L,
            model_type=model_type,
            selected_model_params=selected_model_params,
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
    model_type: str = "FOPDT",
    selected_model_params: Dict[str, Any] | None = None,
    confidence_score: float,
    normalized_rmse: float,
    r2_score: float,
    dt: float,
    experience_guidance: Dict[str, Any] | None = None,
    knowledge_guidance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    primary_inputs = _derive_primary_tuning_inputs(
        model_type=model_type,
        selected_model_params=selected_model_params,
        K=K,
        T=T,
        L=L,
    )
    normalized_model_type = str(primary_inputs["model_type"]).upper()
    selected_model_params = dict(primary_inputs["selected_model_params"] or {})
    if normalized_model_type == "SOPDT":
        K = float(selected_model_params.get("K", primary_inputs["K"]))
        T = float(primary_inputs.get("aggregate_tau", float(selected_model_params.get("T1", 0.0)) + float(selected_model_params.get("T2", 0.0))))
        L = float(selected_model_params.get("L", primary_inputs.get("L", L)))
    elif normalized_model_type == "IPDT":
        K = float(selected_model_params.get("K", primary_inputs["K"]))
        T = float(primary_inputs.get("T_work", T))
        L = float(selected_model_params.get("L", primary_inputs.get("L", L)))
    else:
        K = float(primary_inputs["K"])
        T = float(primary_inputs.get("T_work", primary_inputs.get("T", T)))
        L = float(primary_inputs.get("L_work", primary_inputs.get("L", L)))
    tau_ratio = float(primary_inputs["tau_ratio"])
    heuristic_selection = select_tuning_strategy(
        loop_type=loop_type,
        K=K,
        T=T,
        L=L,
        model_type=normalized_model_type,
        model_params=selected_model_params,
        model_confidence=confidence_score,
        r2_score=r2_score,
        normalized_rmse=normalized_rmse,
    )
    if normalized_model_type == "SOPDT" and heuristic_selection.get("strategy") == "ZN":
        heuristic_selection = {
            **heuristic_selection,
            "strategy": "LAMBDA",
            "reason": "SOPDT 模型更适合优先采用保守的 Lambda/IMC 类整定。",
        }
    if normalized_model_type == "IPDT" and heuristic_selection.get("strategy") in {"ZN", "CHR"}:
        heuristic_selection = {
            **heuristic_selection,
            "strategy": "LAMBDA",
            "reason": "积分过程优先采用更保守的 Lambda/IMC 类整定。",
        }

    preferred_strategy = str((experience_guidance or {}).get("preferred_strategy", "")).upper()
    preferred_matches = (experience_guidance or {}).get("matches") or []
    top_match = preferred_matches[0] if preferred_matches else None
    summary = (experience_guidance or {}).get("summary") or {}
    guidance_text = describe_experience_guidance(experience_guidance or {})
    recommended_kp_scale = float(summary.get("recommended_kp_scale", 1.0) or 1.0)
    recommended_ki_scale = float(summary.get("recommended_ki_scale", 1.0) or 1.0)
    recommended_kd_scale = float(summary.get("recommended_kd_scale", 1.0) or 1.0)
    preferred_refine_pattern = str(summary.get("preferred_refine_pattern", ""))
    knowledge_guidance = dict(knowledge_guidance or {})
    tuning_bias = dict(knowledge_guidance.get("tuning_bias") or {})
    knowledge_preferred_strategy = _normalize_strategy_hint(tuning_bias.get("preferred_strategy") or preferred_strategy)
    knowledge_constraints = list(knowledge_guidance.get("constraints") or [])
    knowledge_risk_hints = list(knowledge_guidance.get("risk_hints") or [])
    avoid_aggressive = bool(tuning_bias.get("avoid_aggressive_strategies")) or bool(tuning_bias.get("conservative_mode"))

    prioritized: List[str] = []
    for strategy_name in [knowledge_preferred_strategy, preferred_strategy, str(heuristic_selection.get("strategy", "")).upper()]:
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
            if knowledge_preferred_strategy and strategy_name == knowledge_preferred_strategy:
                experience_bonus += 0.18
            if avoid_aggressive and strategy_name in {"ZN", "CHR"}:
                experience_bonus -= 0.18
            candidate = _evaluate_strategy(
                K=K,
                T=T,
                L=L,
                model_type=normalized_model_type,
                selected_model_params=selected_model_params,
                dt=dt,
                confidence_score=confidence_score,
                strategy_name=strategy_name,
                experience_bonus=experience_bonus,
                tuning_bias=tuning_bias,
            )
            candidate_results.append(candidate)
            if _is_better_candidate(candidate, best_candidate):
                best_candidate = candidate

            history_seed_pid = _build_history_seed_pid(
                top_match=top_match,
                base_candidate=candidate,
                K=K,
                T=T,
                L=L,
                model_type=normalized_model_type,
                selected_model_params=selected_model_params,
            )
            if history_seed_pid:
                seeded_candidate = _evaluate_pid_params(
                    K=K,
                    T=T,
                    L=L,
                    model_type=normalized_model_type,
                    selected_model_params=selected_model_params,
                    dt=dt,
                    confidence_score=confidence_score,
                    strategy_name=strategy_name,
                    pid_params=history_seed_pid,
                    description_suffix=" (history_seeded)",
                    experience_bonus=experience_bonus + 0.12,
                    tuning_bias=tuning_bias,
                )
                seeded_candidate["history_seeded"] = True
                if top_match:
                    seeded_candidate["seed_experience_id"] = top_match.get("experience_id", "")
                candidate_results.append(seeded_candidate)
                if _is_better_candidate(seeded_candidate, best_candidate):
                    best_candidate = seeded_candidate

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
                    model_type=normalized_model_type,
                    selected_model_params=selected_model_params,
                    dt=dt,
                    confidence_score=confidence_score,
                    strategy_name=strategy_name,
                    pid_params=refined_pid,
                    description_suffix=" (history_refined)",
                    experience_bonus=experience_bonus + 0.1,
                    tuning_bias=tuning_bias,
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
        if confidence_score < 0.55:
            remaining = [strategy for strategy in remaining if strategy not in {"ZN", "CHR"}]
        if confidence_score < 0.35:
            remaining = [strategy for strategy in remaining if strategy in {"IMC", "LAMBDA"}]
        evaluate_group(remaining)

    if best_candidate is None:
        raise ValueError("Failed to generate a usable PID candidate")

    pid_params = apply_tuning_rules(
        K,
        T,
        L,
        best_candidate["strategy"],
        model_type=normalized_model_type,
        model_params=selected_model_params,
    )
    pid_params, final_constraints_applied = _apply_knowledge_bias_to_pid_params(
        pid_params=pid_params,
        tuning_bias=tuning_bias,
        strategy_name=best_candidate["strategy"],
    )
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
    else:
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
            "history_seeded": bool(item.get("history_seeded", False)),
            "seed_experience_id": item.get("seed_experience_id", ""),
        }
        for item in candidate_results
    ]

    if full_benchmark_triggered:
        selection_reason = (
            f"先优先试算历史偏好策略与启发式策略（{', '.join(prioritized)}），"
            f"因评分未达标，已扩展为 {', '.join(remaining or prioritized)} 的闭环试算，"
            f"最终选择 performance_score 最高的 {best_candidate['strategy']}。"
        )
    else:
        selection_reason = (
            f"先优先试算历史偏好策略与启发式策略（{', '.join(prioritized)}），"
            f"无需展开全量 benchmark，直接选择 {best_candidate['strategy']}。"
        )
    if guidance_text:
        selection_reason += f" {guidance_text}"
    if knowledge_guidance.get("summary"):
        selection_reason += f" 专家规则：{knowledge_guidance.get('summary')}"

    selection_inputs = {
        "loop_type": loop_type,
        "model_type": normalized_model_type,
        "model_confidence": confidence_score,
        "normalized_rmse": normalized_rmse,
        "r2_score": r2_score,
        "tau_ratio": tau_ratio,
        "selected_model_params": selected_model_params or {},
        "heuristic_strategy": heuristic_selection["strategy"],
        "experience_preferred_strategy": preferred_strategy,
        "experience_preferred_model_type": str(summary.get("preferred_model_type", "")),
        "experience_match_count": len(preferred_matches) if isinstance(preferred_matches, list) else 0,
        "experience_top_match_id": (top_match or {}).get("experience_id", ""),
        "preferred_refine_pattern": preferred_refine_pattern,
        "recommended_kp_scale": recommended_kp_scale,
        "recommended_ki_scale": recommended_ki_scale,
        "recommended_kd_scale": recommended_kd_scale,
        "knowledge_preferred_strategy": knowledge_preferred_strategy,
        "knowledge_risk_hints": knowledge_risk_hints[:3],
        "knowledge_constraints": knowledge_constraints[:3],
        "knowledge_tuning_bias": tuning_bias,
        "tested_candidates": [
            {
                "strategy": item["strategy"],
                "source": (
                    "history_refined"
                    if item.get("history_refined")
                    else "history_seeded"
                    if item.get("history_seeded")
                    else "heuristic_or_benchmark"
                ),
                "knowledge_constraints_applied": item.get("knowledge_constraints_applied", {}),
            }
            for item in candidate_results
        ],
        "full_benchmark_triggered": full_benchmark_triggered,
    }
    if normalized_model_type == "SOPDT":
        selection_inputs["derived_tuning_features"] = {
            "shape_index": primary_inputs.get("shape_index"),
            "apparent_order": primary_inputs.get("apparent_order"),
            "tau_ratio": tau_ratio,
            "aggregate_tau": float(primary_inputs.get("aggregate_tau", T)),
            "T_work": float(primary_inputs.get("T_work", T)),
            "L_work": float(primary_inputs.get("L_work", L)),
        }
    elif normalized_model_type == "IPDT":
        selection_inputs["derived_tuning_features"] = {
            "tau_ratio": tau_ratio,
            "T_work": float(T),
            "L_work": float(L),
        }
    elif normalized_model_type in {"FOPDT", "FO"}:
        selection_inputs.update(
            {
                "K": float(K),
                "T": float(T),
                "L": float(L),
            }
        )
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
        "knowledge_guidance": knowledge_guidance,
        "knowledge_constraints_applied": final_constraints_applied,
    }
