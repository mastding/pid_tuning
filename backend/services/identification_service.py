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

    def build_mv_peak_windows(df: Any) -> List[Dict[str, Any]]:
        if df is None or len(df) < 20 or "MV" not in df.columns:
            return []
        mv = df["MV"].to_numpy(dtype=float)
        mv_diff = np.abs(np.diff(mv))
        if mv_diff.size < 3:
            return []
        top_k = 6
        min_gap = max(40, int(len(df) // 5000))
        padding = max(80, min(500, int(len(df) // 200)))
        order = np.argsort(mv_diff)[::-1]
        selected: List[int] = []
        for idx in order:
            if len(selected) >= top_k:
                break
            if mv_diff[idx] <= 1e-9:
                break
            if any(abs(idx - prev) < min_gap for prev in selected):
                continue
            selected.append(int(idx))

        selected.sort()
        windows: List[Dict[str, Any]] = []
        for i, center in enumerate(selected):
            start = max(0, center - padding)
            end = min(len(df), center + padding)
            if end - start < 10:
                continue
            windows.append(
                {
                    "start_idx": center,
                    "end_idx": min(len(df), center + 1),
                    "window_start_idx": start,
                    "window_end_idx": end,
                    "amplitude": float(mv_diff[center]),
                    "type": "mv_peak",
                    "_mv_peak_rank": i + 1,
                }
            )
        return windows

    if cleaned_df is not None:
        combined_events = list(candidate_windows)
        has_mv_peak = any(str(evt.get("type", "")).strip().lower() == "mv_peak" for evt in combined_events if isinstance(evt, dict))
        if not has_mv_peak:
            combined_events.extend(build_mv_peak_windows(cleaned_df))
        for idx, event in enumerate(combined_events):
            start_idx = int(event.get("window_start_idx", 0))
            end_idx = int(event.get("window_end_idx", 0))
            candidate_df = cleaned_df.iloc[start_idx:end_idx].reset_index(drop=True)
            if len(candidate_df) >= 10:
                base_name = "step_event"
                if str(event.get("type", "")) == "mv_peak":
                    base_name = "mv_peak"
                candidates.append(
                    {
                        "name": f"{base_name}_{idx + 1}",
                        "df": candidate_df,
                        "event": event,
                    }
                )

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
    sv = window_df["SV"].to_numpy(dtype=float) if "SV" in window_df.columns else None
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
        if sv is not None:
            point["sv"] = float(sv[i])
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
    benchmark_fn: Callable[..., Dict[str, Any]],
    loop_type: str = "flow",
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    best_attempt: Dict[str, Any] | None = None
    loop_name = (loop_type or "flow").strip().lower()
    model_order = MODEL_ORDER_BY_LOOP_TYPE.get(loop_name, ["FOPDT", "FO", "SOPDT", "IPDT"])

    best_stable_attempt: Dict[str, Any] | None = None
    filtered_candidates = 0

    def _robust_diff_noise(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size < 3:
            return 0.0
        diffs = np.diff(values)
        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med)))
        return mad

    def _candidate_penalty(source: str) -> float:
        if source == "full_cleaned":
            return 0.25
        return 0.0

    def _is_better_attempt(
        *,
        current_best: Dict[str, Any] | None,
        current_strategy: Dict[str, Any],
        current_confidence: Dict[str, Any],
        current_source: str,
        current_quality_score: float,
    ) -> bool:
        if current_best is None:
            return True

        best_strategy = current_best["benchmark"].get("best") or {}
        best_confidence = current_best["confidence"] or {}
        best_source = str(current_best.get("source") or "")
        best_quality_score = float(current_best.get("window_quality_score", 0.0) or 0.0)

        current_perf = float(current_strategy.get("performance_score", 0.0))
        best_perf = float(best_strategy.get("performance_score", 0.0))
        current_final = float(current_strategy.get("final_rating", 0.0))
        best_final = float(best_strategy.get("final_rating", 0.0))
        current_conf = float(current_confidence.get("confidence", 0.0))
        best_conf = float(best_confidence.get("confidence", 0.0))

        current_penalty = _candidate_penalty(str(current_source)) + max(0.0, 1.0 - float(current_quality_score)) * 0.6
        best_penalty = _candidate_penalty(best_source) + max(0.0, 1.0 - best_quality_score) * 0.6
        current_perf -= current_penalty
        best_perf -= best_penalty

        if current_perf > best_perf + 1e-9:
            return True
        if abs(current_perf - best_perf) <= 1e-9:
            if current_final > best_final + 1e-9:
                return True
            if abs(current_final - best_final) <= 1e-9:
                if current_quality_score > best_quality_score + 1e-9:
                    return True
                if abs(current_quality_score - best_quality_score) <= 1e-9 and current_conf > best_conf + 1e-9:
                    return True
        return False

    def _window_quality(df: Any) -> Dict[str, Any]:
        mv = df["MV"].to_numpy(dtype=float)
        pv = df["PV"].to_numpy(dtype=float)
        mv_std = float(np.std(mv)) if mv.size else 0.0
        pv_std = float(np.std(pv)) if pv.size else 0.0
        mv_span = float(np.max(mv) - np.min(mv)) if mv.size else 0.0
        pv_span = float(np.max(pv) - np.min(pv)) if pv.size else 0.0
        mv_noise = _robust_diff_noise(mv)
        pv_noise = _robust_diff_noise(pv)
        mv_effective = mv_span >= max(mv_noise * 12.0, 1e-6) and mv_std >= max(mv_noise * 4.0, 1e-6)
        pv_effective = pv_span >= max(pv_noise * 10.0, 1e-6) and pv_std >= max(pv_noise * 3.0, 1e-6)

        corr = 0.0
        if mv.size >= 5:
            mv_centered = mv - float(np.mean(mv))
            pv_centered = pv - float(np.mean(pv))
            denom = float(np.std(mv_centered) * np.std(pv_centered))
            if denom > 1e-12:
                corr = float(np.mean(mv_centered * pv_centered) / denom)

        saturation_ratio = 0.0
        if mv_span > 1e-9:
            low = float(np.min(mv)) + 0.01 * mv_span
            high = float(np.max(mv)) - 0.01 * mv_span
            near_low = float(np.mean(mv <= low))
            near_high = float(np.mean(mv >= high))
            saturation_ratio = max(near_low, near_high)

        drift_ratio = 0.0
        if pv.size >= 5 and pv_span > 1e-9:
            if pv.size > 5000:
                slope = float((pv[-1] - pv[0]) / max(float(pv.size - 1), 1.0))
            else:
                x = np.arange(pv.size, dtype=float)
                slope = float(np.polyfit(x, pv, 1)[0])
            drift_ratio = abs(slope) * float(pv.size - 1) / max(pv_span, 1e-9)

        reasons: List[str] = []
        if not mv_effective:
            reasons.append("MV激励不足")
        if not pv_effective:
            reasons.append("PV响应不足")
        if abs(corr) < 0.05:
            reasons.append("MV与PV相关性弱")
        if saturation_ratio > 0.4:
            reasons.append("MV疑似饱和")

        passed = mv_effective and pv_effective and abs(corr) >= 0.05 and saturation_ratio <= 0.6
        score = 0.0
        score += 0.4 if mv_effective else 0.0
        score += 0.4 if pv_effective else 0.0
        score += 0.2 * min(abs(corr) / 0.4, 1.0)
        if saturation_ratio > 0.0:
            score *= 1.0 - min(saturation_ratio / 0.6, 1.0) * 0.7
        if drift_ratio > 0.0:
            score *= 1.0 - min(drift_ratio / 1.0, 1.0) * 0.25
        score = float(max(0.0, min(score, 1.0)))
        return {
            "passed": bool(passed),
            "score": score,
            "reasons": reasons,
            "mv_span": mv_span,
            "pv_span": pv_span,
            "mv_std": mv_std,
            "pv_std": pv_std,
            "corr": corr,
            "saturation_ratio": saturation_ratio,
            "drift_ratio": drift_ratio,
        }

    def _detrend_pv_if_needed(pv: np.ndarray) -> tuple[np.ndarray, bool]:
        pv = np.asarray(pv, dtype=float)
        if pv.size < 10:
            return pv, False
        span = float(np.max(pv) - np.min(pv))
        if span <= 1e-9:
            return pv, False
        if pv.size > 5000:
            slope = float((pv[-1] - pv[0]) / max(float(pv.size - 1), 1.0))
            intercept = float(pv[0])
            drift = abs(slope) * float(pv.size - 1)
        else:
            x = np.arange(pv.size, dtype=float)
            slope, intercept = [float(v) for v in np.polyfit(x, pv, 1)]
            drift = abs(slope) * float(pv.size - 1)
        if drift / max(span, 1e-9) < 0.35:
            return pv, False
        x = np.arange(pv.size, dtype=float)
        trend = slope * x + intercept
        return pv - trend, True

    def _align_series(mv: np.ndarray, pv: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        mv = np.asarray(mv, dtype=float)
        pv = np.asarray(pv, dtype=float)
        if mv.size < 20 or pv.size < 20:
            return mv, pv, 0, 0.0
        max_lag = int(max(3, min(12, round(60.0 / max(dt, 1e-6)))))
        mv_centered = mv - float(np.mean(mv))
        pv_centered = pv - float(np.mean(pv))

        best_lag = 0
        best_score = 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                a = mv_centered
                b = pv_centered
            elif lag > 0:
                a = mv_centered[:-lag]
                b = pv_centered[lag:]
            else:
                a = mv_centered[-lag:]
                b = pv_centered[:lag]
            if a.size < 15 or b.size < 15:
                continue
            denom = float(np.std(a) * np.std(b))
            if denom <= 1e-12:
                continue
            score = float(abs(np.mean(a * b) / denom))
            if score > best_score + 1e-9:
                best_score = score
                best_lag = lag

        if best_lag == 0:
            return mv, pv, 0, best_score
        if best_lag > 0:
            mv_adj = mv[:-best_lag]
            pv_adj = pv[best_lag:]
        else:
            mv_adj = mv[-best_lag:]
            pv_adj = pv[:best_lag]
        if mv_adj.size < 10 or pv_adj.size < 10:
            return mv, pv, 0, best_score
        return mv_adj, pv_adj, best_lag, best_score

    extracted_candidates = extract_candidate_windows(cleaned_df, candidate_windows)

    def _empty_quality() -> Dict[str, Any]:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [],
            "mv_span": 0.0,
            "pv_span": 0.0,
            "mv_std": 0.0,
            "pv_std": 0.0,
            "corr": 0.0,
            "saturation_ratio": 0.0,
            "drift_ratio": 0.0,
        }

    def _passed_for_candidate(candidate: Dict[str, Any], quality: Dict[str, Any]) -> bool:
        event = candidate.get("event")
        if isinstance(event, dict) and "window_usable_for_id" in event:
            value = event.get("window_usable_for_id")
            if value is True:
                return True
            if value is False:
                return False
        return bool(quality.get("passed"))

    candidate_quality: Dict[str, Dict[str, Any]] = {}
    has_usable_windows = False
    non_full_candidates: List[Dict[str, Any]] = []
    full_candidate: Dict[str, Any] | None = None
    for candidate in extracted_candidates:
        if str(candidate.get("name")) == "full_cleaned":
            full_candidate = candidate
        else:
            non_full_candidates.append(candidate)

    for candidate in non_full_candidates:
        quality = _window_quality(candidate["df"])
        candidate_quality[str(candidate["name"])] = quality
        if _passed_for_candidate(candidate, quality):
            has_usable_windows = True

    if not has_usable_windows and full_candidate is not None:
        candidate_quality[str(full_candidate["name"])] = _window_quality(full_candidate["df"])

    for candidate in extracted_candidates:
        candidate_df = candidate["df"]
        quality = candidate_quality.get(str(candidate["name"])) or _empty_quality()
        if str(candidate.get("name")) != "full_cleaned" and quality == _empty_quality():
            quality = _window_quality(candidate_df)

        passed = _passed_for_candidate(candidate, quality)
        should_skip = (not passed) and (has_usable_windows or candidate["name"] != "full_cleaned")
        if should_skip:
            filtered_candidates += 1
            attempts.append(
                {
                    "window_source": candidate["name"],
                    "model_type": "WINDOW_FILTER",
                    "points": int(len(candidate_df)),
                    "success": False,
                    "error": (
                        ("存在可用辨识窗口，跳过该窗口： " if has_usable_windows else "")
                        + (" / ".join(quality["reasons"]) or "窗口质量不足")
                    ),
                    "mv_span": float(quality["mv_span"]),
                    "pv_span": float(quality["pv_span"]),
                    "mv_std": float(quality["mv_std"]),
                    "pv_std": float(quality["pv_std"]),
                    "corr": float(quality["corr"]),
                    "saturation_ratio": float(quality["saturation_ratio"]),
                    "drift_ratio": float(quality["drift_ratio"]),
                    "has_usable_windows": bool(has_usable_windows),
                    "passed_for_id": bool(passed),
                }
            )
            continue

        mv_array = candidate_df["MV"].to_numpy(dtype=float)
        pv_array = candidate_df["PV"].to_numpy(dtype=float)
        window_quality_score = float(quality.get("score", 0.0) or 0.0)

        for model_type in model_order:
            attempt_result: Dict[str, Any] = {
                "window_source": candidate["name"],
                "model_type": model_type,
                "points": int(len(candidate_df)),
                "window_quality_score": window_quality_score,
                "window_saturation_ratio": float(quality.get("saturation_ratio", 0.0) or 0.0),
                "window_corr": float(quality.get("corr", 0.0) or 0.0),
                "window_drift_ratio": float(quality.get("drift_ratio", 0.0) or 0.0),
            }
            if candidate.get("event"):
                attempt_result["window_start_index"] = int(candidate["event"].get("window_start_idx", 0))
                attempt_result["window_end_index"] = int(candidate["event"].get("window_end_idx", len(candidate_df)))
                attempt_result["event_type"] = str(candidate["event"].get("type", ""))

            try:
                fitted_model_base = _fit_model_by_type(model_type, mv_array, pv_array, actual_dt)
                confidence_base = calculate_model_confidence(
                    fitted_model_base["normalized_rmse"],
                    fitted_model_base.get("r2_score"),
                )

                fitted_model = fitted_model_base
                confidence = confidence_base
                attempt_result["pv_detrended"] = False
                attempt_result["alignment_lag_steps"] = 0
                attempt_result["alignment_score"] = 0.0
                attempt_result["baseline_r2_score"] = float(fitted_model_base.get("r2_score", 0.0))
                attempt_result["baseline_normalized_rmse"] = float(fitted_model_base.get("normalized_rmse", 0.0))
                attempt_result["baseline_confidence"] = float(confidence_base.get("confidence", 0.0))

                should_try_preprocess = bool(
                    float(fitted_model_base.get("r2_score", 0.0)) < 0.3
                    or float(fitted_model_base.get("normalized_rmse", 0.0)) > 0.5
                )

                if should_try_preprocess:
                    detrended_pv, pv_detrended = _detrend_pv_if_needed(pv_array)
                    aligned_mv, aligned_pv, best_lag, align_score = _align_series(mv_array, detrended_pv, actual_dt)
                    fitted_model_alt = _fit_model_by_type(model_type, aligned_mv, aligned_pv, actual_dt)
                    confidence_alt = calculate_model_confidence(
                        fitted_model_alt["normalized_rmse"],
                        fitted_model_alt.get("r2_score"),
                    )

                    attempt_result["alt_pv_detrended"] = bool(pv_detrended)
                    attempt_result["alt_alignment_lag_steps"] = int(best_lag)
                    attempt_result["alt_alignment_score"] = float(align_score)
                    attempt_result["alt_r2_score"] = float(fitted_model_alt.get("r2_score", 0.0))
                    attempt_result["alt_normalized_rmse"] = float(fitted_model_alt.get("normalized_rmse", 0.0))
                    attempt_result["alt_confidence"] = float(confidence_alt.get("confidence", 0.0))

                    if float(confidence_alt.get("confidence", 0.0)) > float(confidence_base.get("confidence", 0.0)) + 1e-9:
                        fitted_model = fitted_model_alt
                        confidence = confidence_alt
                        attempt_result["pv_detrended"] = bool(pv_detrended)
                        attempt_result["alignment_lag_steps"] = int(best_lag)
                        attempt_result["alignment_score"] = float(align_score)
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

            tuning_model = _build_tuning_model(model_type, fitted_model, actual_dt, len(candidate_df))
            benchmark = benchmark_fn(
                float(tuning_model["K"]),
                float(tuning_model["T"]),
                float(tuning_model["L"]),
                actual_dt,
                float(confidence["confidence"]),
                model_type=model_type,
                selected_model_params=fitted_model,
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

            packaged = {
                "candidate_df": candidate_df,
                "event": candidate.get("event"),
                "source": candidate["name"],
                "model_params": fitted_model,
                "confidence": confidence,
                "benchmark": benchmark,
                "tuning_model": tuning_model,
                "attempt_result": attempt_result,
                "window_quality": quality,
                "window_quality_score": window_quality_score,
            }

            if _is_better_attempt(
                current_best=best_attempt,
                current_strategy=best_strategy,
                current_confidence=confidence,
                current_source=candidate["name"],
                current_quality_score=window_quality_score,
            ):
                best_attempt = packaged

            if bool(best_strategy.get("is_stable", False)) and _is_better_attempt(
                current_best=best_stable_attempt,
                current_strategy=best_strategy,
                current_confidence=confidence,
                current_source=candidate["name"],
                current_quality_score=window_quality_score,
            ):
                best_stable_attempt = packaged

    if best_stable_attempt is not None:
        best_attempt = best_stable_attempt

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
        f"综合闭环试算评分（优先稳定）最终选择 {selected_model_type} 作为当前最优辨识模型，并进入对应模型类型的 PID 试算。"
    )
    if has_usable_windows:
        selection_reason = "已检测到可用于辨识的候选窗口，辨识阶段仅在可用窗口集合上进行多窗口多模型评估。" + selection_reason

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
