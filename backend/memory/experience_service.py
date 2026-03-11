from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List
from uuid import uuid4

from .experience_store import (
    append_experience_record,
    get_experience_detail,
    get_experience_stats,
    list_experiences,
    load_experience_records,
    load_indexed_experience_candidates,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _log_distance(lhs: float, rhs: float) -> float:
    left = max(abs(lhs), 1e-6)
    right = max(abs(rhs), 1e-6)
    try:
        import math

        return abs(math.log10(left) - math.log10(right))
    except Exception:
        return abs(left - right)


def _tau_ratio(T: float, L: float) -> float:
    return max(_safe_float(L), 0.0) / max(_safe_float(T), 1e-6)


def _experience_similarity_score(
    *,
    target_loop_type: str,
    target_K: float,
    target_T: float,
    target_L: float,
    record: Dict[str, Any],
) -> float:
    model = record.get("model") or {}
    evaluation = record.get("evaluation") or {}
    strategy = record.get("strategy") or {}

    score = 0.0
    if str(record.get("loop_type", "")).lower() == target_loop_type.lower():
        score += 5.0

    score += max(0.0, 3.0 - _log_distance(target_K, _safe_float(model.get("K"), 0.0)))
    score += max(0.0, 3.0 - _log_distance(target_T, _safe_float(model.get("T"), 0.0)))
    score += max(
        0.0,
        2.0 - abs(_tau_ratio(target_T, target_L) - _tau_ratio(_safe_float(model.get("T")), _safe_float(model.get("L")))) * 5.0,
    )
    score += min(max(_safe_float(evaluation.get("final_rating")) / 2.0, 0.0), 5.0)
    if bool(evaluation.get("passed", False)):
        score += 2.0
    if bool(strategy.get("auto_refined", False)):
        score += 0.5
    return score


def retrieve_experience_guidance(
    *,
    loop_type: str,
    K: float,
    T: float,
    L: float,
    limit: int = 3,
    candidate_strategies: List[str] | None = None,
) -> Dict[str, Any]:
    records = load_indexed_experience_candidates(loop_type, limit=120)
    if not records:
        records = load_experience_records(limit=200)
    if not records:
        return {"matches": [], "preferred_strategy": "", "guidance": "", "summary": {}}

    scored_matches: List[Dict[str, Any]] = []
    for record in records:
        score = _experience_similarity_score(
            target_loop_type=loop_type,
            target_K=K,
            target_T=T,
            target_L=L,
            record=record,
        )
        if score <= 0:
            continue
        scored_matches.append(
            {
                "experience_id": record.get("experience_id", ""),
                "loop_name": record.get("loop_name", ""),
                "loop_type": record.get("loop_type", ""),
                "similarity_score": round(score, 3),
                "strategy": (record.get("strategy") or {}).get("final", ""),
                "final_rating": _safe_float((record.get("evaluation") or {}).get("final_rating")),
                "performance_score": _safe_float((record.get("evaluation") or {}).get("performance_score")),
                "passed": bool((record.get("evaluation") or {}).get("passed", False)),
                "lessons": record.get("lessons", []),
                "model": record.get("model", {}),
            }
        )

    scored_matches.sort(key=lambda item: (item["similarity_score"], item["final_rating"], item["performance_score"]), reverse=True)
    top_matches = scored_matches[: max(limit, 0)]
    if not top_matches:
        return {"matches": [], "preferred_strategy": "", "guidance": "", "summary": {}}

    strategy_counter = Counter(
        match["strategy"]
        for match in top_matches
        if match["strategy"] and (not candidate_strategies or match["strategy"] in candidate_strategies)
    )
    preferred_strategy = strategy_counter.most_common(1)[0][0] if strategy_counter else ""
    ratings = [match["final_rating"] for match in top_matches if match["final_rating"] > 0]
    avg_rating = mean(ratings) if ratings else 0.0
    summary = {
        "match_count": len(top_matches),
        "avg_final_rating": round(avg_rating, 3),
        "passed_count": sum(1 for match in top_matches if match["passed"]),
        "preferred_strategy": preferred_strategy,
        "top_loop_names": [str(match["loop_name"]) for match in top_matches[:2] if match.get("loop_name")],
    }
    guidance = describe_experience_guidance(
        {
            "matches": top_matches,
            "preferred_strategy": preferred_strategy,
            "summary": summary,
        }
    )
    return {
        "matches": top_matches,
        "preferred_strategy": preferred_strategy,
        "guidance": guidance,
        "summary": summary,
    }


def describe_experience_guidance(experience_guidance: Dict[str, Any]) -> str:
    matches = experience_guidance.get("matches") or []
    summary = experience_guidance.get("summary") or {}
    preferred_strategy = str(experience_guidance.get("preferred_strategy") or summary.get("preferred_strategy") or "")
    if not matches:
        return ""

    top_names = summary.get("top_loop_names") or []
    names_text = f"，典型回路如 {', '.join(top_names)}" if top_names else ""
    avg_rating = _safe_float(summary.get("avg_final_rating"))
    passed_count = int(summary.get("passed_count", 0) or 0)
    strategy_text = f"优先参考历史上表现较好的 {preferred_strategy} 策略" if preferred_strategy else "可参考历史成功案例"
    return (
        f"已检索到 {len(matches)} 条相似回路经验{names_text}，其中 {passed_count} 条最终通过，"
        f"平均综合评分约 {avg_rating:.2f}；{strategy_text}。"
    )


def build_experience_record(
    *,
    loop_name: str,
    loop_type: str,
    loop_uri: str,
    data_source: str,
    start_time: str,
    end_time: str,
    shared_data: Dict[str, Any],
    final_result: Dict[str, Any],
) -> Dict[str, Any]:
    model = final_result.get("model") or {}
    pid = final_result.get("pidParams") or {}
    evaluation = final_result.get("evaluation") or {}
    data_analysis = final_result.get("dataAnalysis") or {}
    initial_assessment = evaluation.get("initial_assessment") or {}
    evaluated_pid = initial_assessment.get("evaluated_pid") or {}
    auto_refine_result = evaluation.get("auto_refine_result") or {}

    lessons: List[str] = []
    if evaluation.get("passed"):
        lessons.append(
            f"{pid.get('strategyUsed') or pid.get('strategy')} 策略最终通过，综合评分 {float(evaluation.get('final_rating', 0.0)):.2f}"
        )
    if auto_refine_result.get("applied"):
        lessons.append("首次评估未通过后，收紧Kp和Ki可显著提升闭环表现")
    if evaluation.get("failure_reason"):
        lessons.append(str(evaluation.get("failure_reason")))

    tags: List[str] = [str(loop_type)]
    if _safe_float(model.get("L")) <= 1e-9:
        tags.append("dead_time_small")
    if _safe_float(model.get("T")) <= 5.0:
        tags.append("fast_process")
    if auto_refine_result.get("applied"):
        tags.append("auto_refined")
    if evaluation.get("passed"):
        tags.append("passed")

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "experience_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        "created_at": created_at,
        "loop_name": loop_name,
        "loop_type": loop_type,
        "loop_uri": loop_uri,
        "data_source": data_source,
        "history_range": {"start_time": start_time, "end_time": end_time},
        "model": {
            "K": _safe_float(model.get("K")),
            "T": _safe_float(model.get("T")),
            "L": _safe_float(model.get("L")),
            "normalized_rmse": _safe_float(model.get("normalizedRmse", model.get("residue"))),
            "raw_rmse": _safe_float(model.get("rawRmse")),
            "r2_score": _safe_float(model.get("r2Score")),
            "confidence": _safe_float(model.get("confidence")),
        },
        "window": {
            "selected_source": model.get("selectedWindowSource", ""),
            "window_points": int(data_analysis.get("windowPoints", 0) or 0),
            "step_events": int(data_analysis.get("stepEvents", 0) or 0),
        },
        "strategy": {
            "initial": pid.get("strategyRequested", ""),
            "heuristic": (pid.get("selectionInputs") or {}).get("heuristic_strategy", ""),
            "final": pid.get("strategyUsed") or pid.get("strategy", ""),
            "auto_refined": bool(auto_refine_result.get("applied")),
        },
        "pid": {
            "initial": {
                "Kp": _safe_float(evaluated_pid.get("Kp")),
                "Ki": _safe_float(evaluated_pid.get("Ki")),
                "Kd": _safe_float(evaluated_pid.get("Kd")),
            },
            "final": {
                "Kp": _safe_float(pid.get("Kp")),
                "Ki": _safe_float(pid.get("Ki")),
                "Kd": _safe_float(pid.get("Kd")),
            },
        },
        "evaluation": {
            "performance_score": _safe_float(evaluation.get("performance_score")),
            "method_confidence": _safe_float(evaluation.get("method_confidence")),
            "final_rating": _safe_float(evaluation.get("final_rating")),
            "passed": bool(evaluation.get("passed", False)),
            "failure_reason": str(evaluation.get("failure_reason", "")),
            "feedback_target": str(evaluation.get("feedback_target", "")),
        },
        "lessons": lessons,
        "tags": tags,
        "experience_guidance_used": shared_data.get("experience_guidance") or {},
    }


def persist_experience_record(record: Dict[str, Any]) -> str:
    return append_experience_record(record)


def list_experience_summaries(
    *,
    loop_type: str = "",
    passed: str = "",
    strategy: str = "",
    keyword: str = "",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    return list_experiences(
        loop_type=loop_type,
        passed=passed,
        strategy=strategy,
        keyword=keyword,
        limit=limit,
    )


def get_experience_center_stats() -> Dict[str, Any]:
    return get_experience_stats()


def get_experience_record(experience_id: str) -> Dict[str, Any] | None:
    return get_experience_detail(experience_id)
