from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from skills.system_id_skills import calculate_model_confidence, fit_fopdt_model


def extract_candidate_windows(cleaned_df: Any, candidate_windows: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    candidate_windows = candidate_windows or []

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


def derive_model_reason_codes(model_params: Dict[str, Any], confidence: Dict[str, Any], quality_metrics: Dict[str, Any] | None) -> List[str]:
    reason_codes: List[str] = []
    residue = float(model_params.get("normalized_rmse", model_params.get("residue", 0.0)) or 0.0)
    r2_score = float(model_params.get("r2_score", 0.0) or 0.0)
    confidence_score = float(confidence.get("confidence", 0.0) or 0.0)
    T = float(model_params.get("T", 0.0) or 0.0)
    L = float(model_params.get("L", 0.0) or 0.0)
    overshoot = float((quality_metrics or {}).get("overshoot_percent", 0.0) or 0.0)
    settling_time = float((quality_metrics or {}).get("settling_time", -1.0) or -1.0)

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


def derive_next_actions(confidence_score: float, reason_codes: List[str]) -> List[str]:
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


def build_fit_preview(window_df: Any, model_params: Dict[str, Any], dt: float, max_points: int = 200) -> Dict[str, Any]:
    if window_df is None or len(window_df) == 0:
        return {"points": []}

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


def fit_best_fopdt_window(
    *,
    cleaned_df: Any,
    candidate_windows: List[Dict[str, Any]] | None,
    quality_metrics: Dict[str, Any] | None,
    actual_dt: float,
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    best_model_params: Dict[str, Any] | None = None
    best_confidence: Dict[str, Any] | None = None
    best_benchmark: Dict[str, Any] | None = None
    best_candidate_df: Any = None
    best_event: Dict[str, Any] | None = None
    best_source = ""

    for candidate in extract_candidate_windows(cleaned_df, candidate_windows):
        candidate_df = candidate["df"]
        mv_array = candidate_df["MV"].to_numpy(dtype=float)
        pv_array = candidate_df["PV"].to_numpy(dtype=float)
        attempt_result = {
            "window_source": candidate["name"],
            "points": int(len(candidate_df)),
        }
        if candidate.get("event"):
            attempt_result["window_start_index"] = int(candidate["event"].get("window_start_idx", 0))
            attempt_result["window_end_index"] = int(candidate["event"].get("window_end_idx", len(candidate_df)))
            attempt_result["event_type"] = str(candidate["event"].get("type", ""))

        try:
            model_params = fit_fopdt_model(mv_array, pv_array, actual_dt)
        except ValueError as exc:
            attempt_result.update({
                "success": False,
                "error": str(exc),
                "mv_std": float(np.std(mv_array)) if len(mv_array) else 0.0,
                "pv_std": float(np.std(pv_array)) if len(pv_array) else 0.0,
            })
            attempts.append(attempt_result)
            continue

        confidence = calculate_model_confidence(model_params["normalized_rmse"], model_params.get("r2_score"))
        benchmark = benchmark_fn(
            float(model_params["K"]),
            float(model_params["T"]),
            float(model_params["L"]),
            actual_dt,
            float(confidence["confidence"]),
        )
        best_strategy = benchmark["best"] or {}
        attempt_result.update({
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
        })
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
        tie_confidence = float(confidence["confidence"])
        best_confidence_score = float((best_confidence or {}).get("confidence", 0.0))
        if current_score > best_score + 1e-9 or (
            abs(current_score - best_score) <= 1e-9 and tie_confidence > best_confidence_score + 1e-9
        ):
            best_model_params = model_params
            best_confidence = confidence
            best_benchmark = benchmark
            best_candidate_df = candidate_df
            best_event = candidate.get("event")
            best_source = candidate["name"]

    if best_model_params is None or best_confidence is None:
        raise ValueError("所有候选辨识窗口中的 MV/PV 变化都过小，无法完成 FOPDT 辨识")

    reason_codes = derive_model_reason_codes(best_model_params, best_confidence, quality_metrics)
    next_actions = derive_next_actions(float(best_confidence.get("confidence", 0.0)), reason_codes)
    fit_preview = build_fit_preview(best_candidate_df, best_model_params, actual_dt)

    selected_window = None
    if best_event and best_candidate_df is not None:
        selected_window = {
            "rows": int(len(best_candidate_df)),
            "start_index": int(best_event.get("window_start_idx", 0)),
            "end_index": int(best_event.get("window_end_idx", len(best_candidate_df))),
            "event_start_index": int(best_event.get("start_idx", 0)),
            "event_end_index": int(best_event.get("end_idx", len(best_candidate_df))),
            "event_type": str(best_event.get("type", "full_range")),
        }

    return {
        "model_params": best_model_params,
        "confidence": best_confidence,
        "benchmark": best_benchmark or {},
        "candidate_df": best_candidate_df,
        "event": best_event,
        "source": best_source,
        "attempts": attempts,
        "reason_codes": reason_codes,
        "next_actions": next_actions,
        "fit_preview": fit_preview,
        "selected_window": selected_window,
    }

