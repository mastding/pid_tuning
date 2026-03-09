"""
系统辨识智能体的 Skills
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import optimize
from scipy.signal import correlate


def estimate_dead_time(mv: np.ndarray, pv: np.ndarray, dt: float = 1.0) -> float:
    """使用互相关估计死区时间。"""
    mv = np.asarray(mv, dtype=float)
    pv = np.asarray(pv, dtype=float)
    mv_centered = mv - np.mean(mv)
    pv_centered = pv - np.mean(pv)

    correlation = correlate(pv_centered, mv_centered, mode="full")
    lags = np.arange(-len(mv) + 1, len(mv))
    lag = int(lags[int(np.argmax(correlation))])
    return abs(lag) * float(dt)


def _simulate_fopdt_response(mv: np.ndarray, K: float, T: float, L: float, dt: float) -> np.ndarray:
    mv = np.asarray(mv, dtype=float)
    T = max(float(T), 1e-6)
    L = max(float(L), 0.0)
    dt = max(float(dt), 1e-6)

    y = np.zeros_like(mv, dtype=float)
    delay_steps = int(round(L / dt))
    alpha = dt / (T + dt)

    for i in range(len(mv)):
        delayed_u = mv[i - delay_steps] if i >= delay_steps else 0.0
        prev_y = y[i - 1] if i > 0 else 0.0
        y[i] = (1.0 - alpha) * prev_y + K * alpha * delayed_u
    return y


def fit_fopdt_model(mv: np.ndarray, pv: np.ndarray, dt: float = 1.0) -> Dict:
    """拟合一阶纯滞后模型 FOPDT。"""
    mv = np.asarray(mv, dtype=float).reshape(-1)
    pv = np.asarray(pv, dtype=float).reshape(-1)
    dt = max(float(dt), 1e-6)

    if mv.size != pv.size:
        raise ValueError("MV and PV must have the same length")
    if mv.size < 10:
        raise ValueError("At least 10 samples are required for FOPDT fitting")

    mv_std = float(np.std(mv))
    pv_std = float(np.std(pv))
    if mv_std < 1e-9 or pv_std < 1e-9:
        raise ValueError("MV/PV variance is too small for identification")

    mv_centered = mv - np.mean(mv)
    pv_centered = pv - np.mean(pv)
    mv_norm = mv_centered / mv_std
    pv_norm = pv_centered / pv_std

    mv_span = float(np.max(mv) - np.min(mv))
    pv_span = float(np.max(pv) - np.min(pv))
    gain_guess = pv_span / mv_span if mv_span > 1e-9 else pv_std / mv_std
    gain_sign = np.sign(np.corrcoef(mv_centered, pv_centered)[0, 1]) if mv.size > 2 else 1.0
    if not np.isfinite(gain_sign) or gain_sign == 0:
        gain_sign = 1.0

    K_init = float(np.clip(gain_sign * max(gain_guess, 0.1), -10.0, 10.0))
    T_init = max(dt * 10.0, min(len(mv) * dt / 5.0, 300.0))
    L_init = min(estimate_dead_time(mv, pv, dt), len(mv) * dt / 3.0)

    def objective(params: np.ndarray) -> float:
        K, T, L = params
        if T <= 0 or L < 0:
            return 1e9
        y_pred = _simulate_fopdt_response(mv_norm, K, T, L, dt)
        return float(np.mean((pv_norm - y_pred) ** 2))

    result = optimize.minimize(
        objective,
        x0=np.array([K_init, T_init, L_init], dtype=float),
        bounds=[
            (-20.0, 20.0),
            (max(dt, 1e-3), max(len(mv) * dt, 10.0)),
            (0.0, max(len(mv) * dt / 2.0, dt)),
        ],
        method="L-BFGS-B",
    )

    K_opt, T_opt, L_opt = [float(x) for x in result.x]
    y_pred = _simulate_fopdt_response(mv_norm, K_opt, T_opt, L_opt, dt)
    normalized_rmse = float(np.sqrt(np.mean((pv_norm - y_pred) ** 2)))
    y_pred_raw = y_pred * pv_std + np.mean(pv)
    raw_rmse = float(np.sqrt(np.mean((pv - y_pred_raw) ** 2)))
    ss_res = float(np.sum((pv_norm - y_pred) ** 2))
    ss_tot = float(np.sum((pv_norm - np.mean(pv_norm)) ** 2))
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    K_real = K_opt * pv_std / mv_std

    return {
        "K": float(K_real),
        "T": T_opt,
        "L": L_opt,
        "residue": normalized_rmse,
        "normalized_rmse": normalized_rmse,
        "raw_rmse": raw_rmse,
        "r2_score": float(r2_score),
        "success": bool(result.success),
        "message": result.message,
    }


def calculate_model_confidence(
    residue: float,
    r2_score: float | None = None,
    threshold: float = 0.25,
) -> Dict:
    """根据拟合残差和 R² 给出置信度。"""
    residue = float(residue)
    rmse_score = max(0.0, 1.0 - residue / max(float(threshold), 1e-6))
    r2_score = float(r2_score) if r2_score is not None else 0.0
    r2_component = min(1.0, max(0.0, r2_score))
    confidence = 0.6 * rmse_score + 0.4 * r2_component

    if residue > 0.15:
        confidence = min(confidence, 0.45)
    elif residue > 0.1:
        confidence = min(confidence, 0.65)
    if r2_component < 0.4:
        confidence = min(confidence, 0.45)

    if confidence > 0.85:
        recommendation = "模型可信，可直接用于 PID 整定"
        quality = "excellent"
    elif confidence > 0.7:
        recommendation = "模型基本可信，建议结合现场经验校核"
        quality = "good"
    elif confidence > 0.5:
        recommendation = "模型置信度偏低，建议复查数据窗口或重新试验"
        quality = "fair"
    else:
        recommendation = "模型不可信，可能存在强扰动、非线性或数据质量问题"
        quality = "poor"

    return {
        "confidence": float(confidence),
        "quality": quality,
        "recommendation": recommendation,
        "residue": residue,
        "rmse_score": float(rmse_score),
        "r2_score": float(r2_component),
        "threshold": float(threshold),
    }


def validate_model(model_params: Dict, mv_test: np.ndarray, pv_test: np.ndarray, dt: float = 1.0) -> Dict:
    """使用测试数据验证模型。"""
    K = float(model_params["K"])
    T = float(model_params["T"])
    L = float(model_params["L"])

    mv_test = np.asarray(mv_test, dtype=float)
    pv_test = np.asarray(pv_test, dtype=float)
    y_pred = _simulate_fopdt_response(mv_test - mv_test[0], K, T, L, dt) + pv_test[0]

    mse = float(np.mean((pv_test - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pv_test - y_pred)))

    ss_res = float(np.sum((pv_test - y_pred) ** 2))
    ss_tot = float(np.sum((pv_test - np.mean(pv_test)) ** 2))
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2_score": float(r2_score),
        "validation_passed": bool(r2_score > 0.7),
    }
