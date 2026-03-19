from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from skills.system_id_skills import (
    calculate_model_confidence,
    fit_fo_model,
    fit_fopdt_model,
    fit_ipdt_model,
    fit_sopdt_model,
)


MODEL_ORDER_BY_LOOP_TYPE = {
    "flow": ["FO", "FOPDT", "SOPDT", "IPDT"],
    "pressure": ["FO", "FOPDT", "SOPDT", "IPDT"],
    "temperature": ["SOPDT", "FOPDT", "FO", "IPDT"],
    "level": ["IPDT", "FOPDT", "FO", "SOPDT"],
}


def sanitize_selected_model_params(model_type: str, model_params: Dict[str, Any] | None) -> Dict[str, Any]:
    params = dict(model_params or {})
    normalized_type = str(model_type or params.get("model_type", "FOPDT")).upper()

    if normalized_type == "SOPDT":
        return {
            "model_type": "SOPDT",
            "K": float(params.get("K", 0.0) or 0.0),
            "T1": float(params.get("T1", params.get("T", 0.0)) or 0.0),
            "T2": float(params.get("T2", params.get("T", 0.0)) or 0.0),
            "L": float(params.get("L", 0.0) or 0.0),
        }
    if normalized_type == "IPDT":
        return {
            "model_type": "IPDT",
            "K": float(params.get("K", 0.0) or 0.0),
            "L": float(params.get("L", 0.0) or 0.0),
        }
    if normalized_type == "FO":
        return {
            "model_type": "FO",
            "K": float(params.get("K", 0.0) or 0.0),
            "T": float(params.get("T", 0.0) or 0.0),
        }
    return {
        "model_type": "FOPDT",
        "K": float(params.get("K", 0.0) or 0.0),
        "T": float(params.get("T", 0.0) or 0.0),
        "L": float(params.get("L", 0.0) or 0.0),
    }


def extract_candidate_windows(cleaned_df: Any, candidate_windows: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    candidate_windows = candidate_windows or []

    if cleaned_df is not None:
        for idx, event in enumerate(candidate_windows):
            start_idx = int(event.get("window_start_idx", 0))
            end_idx = int(event.get("window_end_idx", 0))
            candidate_df = cleaned_df.iloc[start_idx:end_idx].reset_index(drop=True)
            if len(candidate_df) >= 10:
                candidates.append(
                    {
                        "name": f"step_event_{idx + 1}",
                        "df": candidate_df,
                        "event": event,
                    }
                )

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


def derive_model_reason_codes(
    model_params: Dict[str, Any],
    confidence: Dict[str, Any],
    quality_metrics: Dict[str, Any] | None,
) -> List[str]:
    reason_codes: List[str] = []
    residue = float(model_params.get("normalized_rmse", model_params.get("residue", 0.0)) or 0.0)
    r2_score = float(model_params.get("r2_score", 0.0) or 0.0)
    confidence_score = float(confidence.get("confidence", 0.0) or 0.0)
    T = float(model_params.get("T", 0.0) or 0.0)
    L = float(model_params.get("L", 0.0) or 0.0)
    model_type = str(model_params.get("model_type", "FOPDT")).upper()
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
    if model_type == "FOPDT" and T <= 2.0 and confidence_score < 0.7 and residue > 0.08:
        reason_codes.append("动态较快或采样粒度偏粗")
    if model_type == "FOPDT" and L <= 0.0 and confidence_score < 0.7 and r2_score < 0.9:
        reason_codes.append("未观察到明显死区")
    if model_type == "IPDT":
        reason_codes.append("对象可能呈积分特性")
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
        actions.append("确认对象是否偏离当前模型假设")
    if "动态较快或采样粒度偏粗" in reason_codes or "未观察到明显死区" in reason_codes:
        actions.append("检查采样周期或补采更高频数据")
    if "对象可能呈积分特性" in reason_codes:
        actions.append("优先采用积分过程模型或更保守整定策略")
    if confidence_score < 0.5:
        actions.append("采用更保守的整定策略")
    if confidence_score < 0.35:
        actions.append("建议重新采集阶跃试验数据")
    return actions


def _simulate_model_preview(
    model_type: str,
    mv: np.ndarray,
    model_params: Dict[str, Any],
    dt: float,
) -> np.ndarray:
    mv = np.asarray(mv, dtype=float)
    mv_delta = mv - mv[0]
    K = float(model_params.get("K", 0.0))
    T = float(model_params.get("T", 0.0) or 0.0)
    L = float(model_params.get("L", 0.0) or 0.0)

    if model_type == "FO":
        alpha = dt / (max(T, 1e-6) + dt)
        y = np.zeros_like(mv_delta, dtype=float)
        for i in range(len(mv_delta)):
            prev_y = y[i - 1] if i > 0 else 0.0
            y[i] = (1.0 - alpha) * prev_y + K * alpha * mv_delta[i]
        return y

    if model_type == "IPDT":
        y = np.zeros_like(mv_delta, dtype=float)
        delay_steps = int(round(max(L, 0.0) / max(dt, 1e-6)))
        for i in range(len(mv_delta)):
            delayed_u = mv_delta[i - delay_steps] if i >= delay_steps else 0.0
            prev_y = y[i - 1] if i > 0 else 0.0
            y[i] = prev_y + K * dt * delayed_u
        return y
    if model_type == "SOPDT":
        T1 = float(model_params.get("T1", max(T, dt)))
        T2 = float(model_params.get("T2", max(T, dt)))
        y1 = np.zeros_like(mv_delta, dtype=float)
        y2 = np.zeros_like(mv_delta, dtype=float)
        delay_steps = int(round(max(L, 0.0) / max(dt, 1e-6)))
        alpha1 = dt / (max(T1, 1e-6) + dt)
        alpha2 = dt / (max(T2, 1e-6) + dt)
        for i in range(len(mv_delta)):
            delayed_u = mv_delta[i - delay_steps] if i >= delay_steps else 0.0
            prev_y1 = y1[i - 1] if i > 0 else 0.0
            prev_y2 = y2[i - 1] if i > 0 else 0.0
            y1[i] = (1.0 - alpha1) * prev_y1 + alpha1 * delayed_u
            y2[i] = (1.0 - alpha2) * prev_y2 + alpha2 * y1[i]
        return K * y2

    delay_steps = int(round(max(L, 0.0) / max(dt, 1e-6)))
    alpha = dt / (max(T, 1e-6) + dt)
    y = np.zeros_like(mv_delta, dtype=float)
    for i in range(len(mv_delta)):
        delayed_u = mv_delta[i - delay_steps] if i >= delay_steps else 0.0
        prev_y = y[i - 1] if i > 0 else 0.0
        y[i] = (1.0 - alpha) * prev_y + K * alpha * delayed_u
    return y


def build_fit_preview(
    window_df: Any,
    model_params: Dict[str, Any],
    dt: float,
    max_points: int = 200,
) -> Dict[str, Any]:
    if window_df is None or len(window_df) == 0:
        return {"points": []}

    pv = window_df["PV"].to_numpy(dtype=float)
    mv = window_df["MV"].to_numpy(dtype=float)
    n = len(window_df)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    model_type = str(model_params.get("model_type", "FOPDT")).upper()
    simulated_delta = _simulate_model_preview(model_type, mv, model_params, dt)
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
        "model_type": model_type,
        "x_axis": "timestamp" if timestamp_strings is not None else "index",
        "start_time": timestamp_strings[0] if timestamp_strings is not None else None,
        "end_time": timestamp_strings[-1] if timestamp_strings is not None else None,
    }


def _build_tuning_model(model_type: str, model_params: Dict[str, Any], dt: float, n_points: int) -> Dict[str, float]:
    K = float(model_params.get("K", 0.0))
    T = float(model_params.get("T", 0.0) or 0.0)
    L = float(model_params.get("L", 0.0) or 0.0)

    if model_type == "FO":
        return {"K": K, "T": max(T, dt), "L": 0.0}

    if model_type == "IPDT":
        # Conservative self-regulating surrogate for current tuning pipeline.
        surrogate_t = max(float(n_points) * dt / 4.0, dt * 20.0)
        return {"K": max(abs(K), 1e-3), "T": surrogate_t, "L": max(L, dt)}
    if model_type == "SOPDT":
        t1 = float(model_params.get("T1", T or dt))
        t2 = float(model_params.get("T2", T or dt))
        return {"K": K, "T": max(t1 + t2, dt), "L": max(L, 0.0)}

    return {"K": K, "T": max(T, dt), "L": max(L, 0.0)}


def _fit_model_by_type(model_type: str, mv_array: np.ndarray, pv_array: np.ndarray, actual_dt: float) -> Dict[str, Any]:
    if model_type == "FO":
        return fit_fo_model(mv_array, pv_array, actual_dt)
    if model_type == "IPDT":
        return fit_ipdt_model(mv_array, pv_array, actual_dt)
    if model_type == "SOPDT":
        return fit_sopdt_model(mv_array, pv_array, actual_dt)
    return fit_fopdt_model(mv_array, pv_array, actual_dt)


def fit_best_fopdt_window(
    *,
    cleaned_df: Any,
    candidate_windows: List[Dict[str, Any]] | None,
    quality_metrics: Dict[str, Any] | None,
    actual_dt: float,
    benchmark_fn: Callable[[float, float, float, float, float], Dict[str, Any]],
    loop_type: str = "flow",
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    best_attempt: Dict[str, Any] | None = None
    loop_name = (loop_type or "flow").strip().lower()
    model_order = MODEL_ORDER_BY_LOOP_TYPE.get(loop_name, ["FOPDT", "FO", "SOPDT", "IPDT"])

    for candidate in extract_candidate_windows(cleaned_df, candidate_windows):
        candidate_df = candidate["df"]
        mv_array = candidate_df["MV"].to_numpy(dtype=float)
        pv_array = candidate_df["PV"].to_numpy(dtype=float)

        for model_type in model_order:
            attempt_result: Dict[str, Any] = {
                "window_source": candidate["name"],
                "model_type": model_type,
                "points": int(len(candidate_df)),
            }
            if candidate.get("event"):
                attempt_result["window_start_index"] = int(candidate["event"].get("window_start_idx", 0))
                attempt_result["window_end_index"] = int(candidate["event"].get("window_end_idx", len(candidate_df)))
                attempt_result["event_type"] = str(candidate["event"].get("type", ""))

            try:
                fitted_model = _fit_model_by_type(model_type, mv_array, pv_array, actual_dt)
            except ValueError as exc:
                attempt_result.update(
                    {
                        "success": False,
                        "error": str(exc),
                        "mv_std": float(np.std(mv_array)) if len(mv_array) else 0.0,
                        "pv_std": float(np.std(pv_array)) if len(pv_array) else 0.0,
                    }
                )
                attempts.append(attempt_result)
                continue

            confidence = calculate_model_confidence(
                fitted_model["normalized_rmse"],
                fitted_model.get("r2_score"),
            )
            tuning_model = _build_tuning_model(model_type, fitted_model, actual_dt, len(candidate_df))
            benchmark = benchmark_fn(
                float(tuning_model["K"]),
                float(tuning_model["T"]),
                float(tuning_model["L"]),
                actual_dt,
                float(confidence["confidence"]),
            )
            best_strategy = benchmark.get("best") or {}

            attempt_result.update(
                {
                    "K": float(fitted_model["K"]),
                    "T": float(fitted_model.get("T", 0.0)),
                    "L": float(fitted_model.get("L", 0.0)),
                    "selected_model_params": fitted_model,
                    "residue": float(fitted_model["residue"]),
                    "normalized_rmse": float(fitted_model["normalized_rmse"]),
                    "raw_rmse": float(fitted_model["raw_rmse"]),
                    "r2_score": float(fitted_model["r2_score"]),
                    "confidence": float(confidence["confidence"]),
                    "confidence_quality": confidence["quality"],
                    "benchmark_strategy": best_strategy.get("strategy", ""),
                    "benchmark_performance_score": float(best_strategy.get("performance_score", 0.0)),
                    "benchmark_final_rating": float(best_strategy.get("final_rating", 0.0)),
                    "benchmark_stable": bool(best_strategy.get("is_stable", False)),
                    "success": bool(fitted_model["success"]),
                    "tuning_model": tuning_model,
                }
            )
            attempts.append(attempt_result)

            if best_attempt is None:
                best_attempt = {
                    "candidate_df": candidate_df,
                    "event": candidate.get("event"),
                    "source": candidate["name"],
                    "model_params": fitted_model,
                    "confidence": confidence,
                    "benchmark": benchmark,
                    "tuning_model": tuning_model,
                    "attempt_result": attempt_result,
                }
                continue

            current_score = float(best_strategy.get("performance_score", 0.0))
            best_score = float((best_attempt["benchmark"].get("best") or {}).get("performance_score", 0.0))
            current_confidence = float(confidence["confidence"])
            best_confidence = float(best_attempt["confidence"].get("confidence", 0.0))
            if current_score > best_score + 1e-9 or (
                abs(current_score - best_score) <= 1e-9 and current_confidence > best_confidence + 1e-9
            ):
                best_attempt = {
                    "candidate_df": candidate_df,
                    "event": candidate.get("event"),
                    "source": candidate["name"],
                    "model_params": fitted_model,
                    "confidence": confidence,
                    "benchmark": benchmark,
                    "tuning_model": tuning_model,
                    "attempt_result": attempt_result,
                }

    if best_attempt is None:
        raise ValueError("所有候选辨识窗口中的 MV/PV 变化都过小，无法完成过程模型辨识")

    best_model_params = best_attempt["model_params"]
    best_confidence = best_attempt["confidence"]
    best_benchmark = best_attempt["benchmark"]
    best_candidate_df = best_attempt["candidate_df"]
    best_event = best_attempt["event"]
    best_source = best_attempt["source"]
    tuning_model = best_attempt["tuning_model"]
    selected_model_type = str(best_model_params.get("model_type", "FOPDT")).upper()

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

    selection_reason = (
        f"已对候选窗口尝试 {', '.join(model_order)} 多种过程模型，"
        f"最终选择 {selected_model_type} 作为当前最优辨识模型，并进入对应模型类型的 PID 试算。"
    )

    clean_selected_model_params = sanitize_selected_model_params(selected_model_type, best_model_params)

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
        "selected_model_type": selected_model_type,
        "selected_model_params": clean_selected_model_params,
        "tuning_model": tuning_model,
        "selection_reason": selection_reason,
    }
