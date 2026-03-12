from __future__ import annotations

from typing import Dict


def _safe_div(numerator: float, denominator: float, fallback: float) -> float:
    if abs(denominator) < 1e-9:
        return fallback
    return numerator / denominator


def _clamp_pid_params(Kp: float, Ki: float, Kd: float) -> Dict[str, float]:
    return {
        "Kp": float(min(max(Kp, 0.0), 1e4)),
        "Ki": float(min(max(Ki, 0.0), 1e4)),
        "Kd": float(min(max(Kd, 0.0), 1e4)),
    }


def select_tuning_strategy(
    *,
    loop_type: str,
    K: float,
    T: float,
    L: float,
    model_type: str = "FOPDT",
    model_params: Dict | None = None,
    model_confidence: float,
    r2_score: float,
    normalized_rmse: float,
) -> Dict[str, str]:
    loop_name = (loop_type or "flow").strip().lower()
    normalized_model_type = (model_type or "FOPDT").strip().upper()
    model_params = model_params or {}

    tau_ratio = max(float(L), 0.0) / max(float(T), 1e-6)
    fast_process = float(T) <= 5.0
    high_quality_model = model_confidence >= 0.88 and normalized_rmse <= 0.05 and r2_score >= 0.97

    shape_index = 0.0
    apparent_order = 1.0
    if normalized_model_type == "SOPDT":
        t1 = max(float(model_params.get("T1", model_params.get("T", T))), 1e-6)
        t2 = max(float(model_params.get("T2", model_params.get("T", T))), 1e-6)
        dominant_tau = max(t1, t2)
        secondary_tau = min(t1, t2)
        shape_index = min(max(secondary_tau / max(dominant_tau, 1e-6), 0.0), 1.0)
        apparent_order = 1.0 + shape_index
        aggregate_tau = dominant_tau + secondary_tau
        raw_l = max(float(model_params.get("L", L)), 0.0)
        tau_ratio = raw_l / max(aggregate_tau, 1e-6)
        fast_process = aggregate_tau <= 5.0
    elif normalized_model_type == "IPDT":
        lag_value = max(float(model_params.get("L", L)), 1e-6)
        tau_ratio = lag_value / max(float(T), 1e-6)
        fast_process = lag_value <= 5.0

    if model_confidence < 0.35:
        return {
            "strategy": "IMC",
            "reason": "Model confidence is very low, so the controller falls back to the most conservative IMC tuning.",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if model_confidence < 0.55 or normalized_rmse > 0.1 or r2_score < 0.75:
        return {
            "strategy": "LAMBDA",
            "reason": "Model quality is only moderate, so the controller prefers robust Lambda tuning.",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if normalized_model_type == "FO":
        if loop_name in {"flow", "pressure"} and high_quality_model and not fast_process:
            return {
                "strategy": "IMC",
                "reason": "The process looks first-order and the model quality is high, so FO-IMC tuning is preferred.",
                "loop_type": loop_name,
                "model_type": normalized_model_type,
            }
        return {
            "strategy": "LAMBDA",
            "reason": "For first-order processes the controller prefers conservative Lambda/IMC style tuning.",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if normalized_model_type == "IPDT":
        return {
            "strategy": "LAMBDA",
            "reason": "Integrating processes default to conservative integrating-process Lambda tuning.",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if normalized_model_type == "SOPDT":
        if shape_index >= 0.72 or apparent_order >= 1.72:
            strategy = "LAMBDA"
            reason = "The second-order shape is widely distributed, so Lambda is preferred to suppress overshoot and oscillation."
        elif loop_name in {"temperature", "level"}:
            strategy = "LAMBDA"
            reason = "Temperature and level loops with SOPDT dynamics prefer robust native Lambda tuning."
        elif high_quality_model and 0.05 <= tau_ratio <= 0.30 and not fast_process and shape_index <= 0.45:
            strategy = "IMC"
            reason = "The dominant and secondary time constants are clearly separated under a strong SOPDT fit, so native IMC is preferred."
        elif fast_process or tau_ratio < 0.05:
            strategy = "IMC"
            reason = "The raw SOPDT parameters indicate a fast process or very small physical delay, so IMC is used to keep the loop well damped."
        else:
            strategy = "LAMBDA"
            reason = "The raw SOPDT parameters indicate a moderately distributed second-order process, so Lambda is chosen as the safer default."
        return {
            "strategy": strategy,
            "reason": reason,
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if loop_name == "temperature":
        strategy = "IMC"
        reason = "Temperature loops are usually inertial, so IMC is preferred to suppress overshoot."
    elif loop_name == "level":
        strategy = "LAMBDA"
        reason = "Level loops prioritize robustness and smoothness, so Lambda is preferred."
    elif loop_name == "pressure":
        strategy = "IMC" if tau_ratio >= 0.3 else "LAMBDA"
        reason = "Pressure loops are often fast, so the controller prefers conservative strategies to control oscillation."
    elif loop_name == "flow":
        if high_quality_model and tau_ratio >= 0.08 and not fast_process:
            strategy = "ZN"
            reason = "The flow-loop model is very strong and has enough apparent delay, so moderated ZN can be attempted."
        elif fast_process or tau_ratio < 0.08:
            strategy = "IMC"
            reason = "The flow process is fast or has very small delay, so IMC is preferred to suppress oscillation."
        else:
            strategy = "LAMBDA"
            reason = "The flow-loop model is usable but not ideal for aggressive tuning, so Lambda is chosen."
    else:
        strategy = "IMC"
        reason = "The loop type is unknown, so the default robust IMC strategy is used."

    return {"strategy": strategy, "reason": reason, "loop_type": loop_name, "model_type": normalized_model_type}


def tune_fo(K: float, T: float, strategy: str) -> Dict:
    strategy_name = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T = max(float(T), 1e-3)

    if strategy_name in {"IMC", "LAMBDA", "LAMBDA_TUNING"}:
        lambda_c = max(0.8 * T, 1e-3)
        Kp = T / (abs_k * lambda_c)
        Ti = T
        Td = 0.0
        description = "FO Lambda/IMC"
    elif strategy_name == "ZN":
        Kp = 0.8 / abs_k
        Ti = max(T, 1e-3)
        Td = 0.0
        description = "FO moderated ZN"
    else:
        Kp = 0.7 / abs_k
        Ti = max(1.1 * T, 1e-3)
        Td = 0.0
        description = "FO conservative fallback"

    Ki = _safe_div(Kp, Ti, 0.0)
    Kd = Kp * Td
    params = _clamp_pid_params(Kp, Ki, Kd)
    params.update({"strategy": strategy_name, "model_type": "FO", "description": description, "Ti": float(Ti), "Td": float(Td)})
    return params


def tune_fopdt(K: float, T: float, L: float, strategy: str) -> Dict:
    strategy_name = (strategy or "IMC").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T = max(float(T), 1e-3)
    L = max(float(L), 0.0)

    if strategy_name == "IMC":
        lambda_c = max(L, 0.8 * T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T, 1e-3)
        Td = 0.5 * L if L > 0 else 0.0
        description = "Conservative IMC"
    elif strategy_name in {"LAMBDA", "LAMBDA_TUNING"}:
        lambda_c = max(1.5 * L, T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T + 0.5 * L, 1e-3)
        Td = 0.0
        description = "Lambda Tuning"
    elif strategy_name == "ZN":
        effective_l = max(L, 0.1 * T, 1e-3)
        Kp = 1.2 * T / (abs_k * effective_l)
        Ti = 2.0 * effective_l
        Td = 0.5 * effective_l
        description = "Aggressive Ziegler-Nichols"
    elif strategy_name in {"CHR", "CHR_0OS"}:
        effective_l = max(L, 0.1 * T, 1e-3)
        Kp = 0.6 * T / (abs_k * effective_l)
        Ti = max(T, 1e-3)
        Td = 0.5 * effective_l
        description = "CHR 0% Overshoot"
    else:
        lambda_c = max(L, T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T, 1e-3)
        Td = 0.0
        description = "Fallback IMC-like"

    Ki = _safe_div(Kp, Ti, 0.0)
    Kd = Kp * Td
    params = _clamp_pid_params(Kp, Ki, Kd)
    params.update({"strategy": strategy_name, "model_type": "FOPDT", "description": description, "Ti": float(Ti), "Td": float(Td)})
    return params


def tune_sopdt(K: float, T1: float, T2: float, L: float, strategy: str) -> Dict:
    strategy_name = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    T1 = max(float(T1), 1e-3)
    T2 = max(float(T2), 1e-3)
    L = max(float(L), 0.0)

    dominant_tau = max(T1, T2)
    secondary_tau = min(T1, T2)
    tau_ratio = secondary_tau / max(dominant_tau, 1e-6)
    shape_index = min(max(tau_ratio, 0.0), 1.0)
    apparent_order = 1.0 + shape_index
    distributed_lag = L + secondary_tau * (0.35 + 0.20 * shape_index)
    aggregate_tau = dominant_tau + secondary_tau * (0.55 + 0.25 * shape_index)
    t_work = max(aggregate_tau, 1e-3)
    l_work = max(distributed_lag, 0.0)
    derivative_ceiling = max(0.18 * dominant_tau + 0.30 * secondary_tau, 0.0)

    if strategy_name in {"LAMBDA", "LAMBDA_TUNING"}:
        lambda_c = max((1.05 + 0.30 * shape_index) * t_work, 2.0 * l_work, 1e-3)
        Kp = t_work / (abs_k * (lambda_c + l_work))
        Ti = max(dominant_tau + (1.05 + 0.35 * shape_index) * secondary_tau + 0.45 * l_work, 1e-3)
        Td = min(secondary_tau * (0.22 + 0.18 * shape_index), derivative_ceiling)
        description = "SOPDT native Lambda"
    elif strategy_name == "IMC":
        lambda_c = max((0.90 + 0.20 * shape_index) * t_work, 1.6 * l_work, 1e-3)
        Kp = t_work / (abs_k * (lambda_c + l_work))
        Ti = max(dominant_tau + (0.75 + 0.20 * shape_index) * secondary_tau + 0.25 * l_work, 1e-3)
        Td = min(secondary_tau * (0.16 + 0.12 * shape_index), derivative_ceiling)
        description = "SOPDT native IMC"
    elif strategy_name == "ZN":
        effective_l = max(l_work, (0.22 + 0.08 * shape_index) * t_work, 1e-3)
        Kp = 0.55 * t_work / (abs_k * effective_l)
        Ti = max(2.8 * effective_l + 0.40 * secondary_tau, 1e-3)
        Td = min(0.40 * effective_l, 0.35 * secondary_tau)
        description = "SOPDT native moderated ZN"
    else:
        effective_l = max(l_work, (0.20 + 0.05 * shape_index) * t_work, 1e-3)
        Kp = 0.38 * t_work / (abs_k * effective_l)
        Ti = max(t_work + 0.35 * secondary_tau, 1e-3)
        Td = min(0.30 * effective_l, 0.25 * secondary_tau)
        description = "SOPDT native CHR-like"

    Ki = _safe_div(Kp, Ti, 0.0)
    Kd = Kp * Td
    params = _clamp_pid_params(Kp, Ki, Kd)
    params.update(
        {
            "strategy": strategy_name,
            "model_type": "SOPDT",
            "description": description,
            "Ti": float(Ti),
            "Td": float(Td),
            "T1": float(T1),
            "T2": float(T2),
            "T_dominant": float(dominant_tau),
            "T_secondary": float(secondary_tau),
            "tau_ratio": float(tau_ratio),
            "shape_index": float(shape_index),
            "apparent_order": float(apparent_order),
            "L_work": float(l_work),
            "T_work": float(t_work),
        }
    )
    return params


def tune_ipdt(K: float, L: float, strategy: str) -> Dict:
    strategy_name = (strategy or "LAMBDA").strip().upper()
    abs_k = max(abs(float(K)), 1e-6)
    effective_l = max(float(L), 1e-3)

    if strategy_name in {"LAMBDA", "LAMBDA_TUNING", "IMC"}:
        lambda_c = max(2.5 * effective_l, 1e-3)
        Kp = 1.0 / (abs_k * max(lambda_c + effective_l, 1e-3))
        Ti = max(4.0 * effective_l, 1e-3)
        Td = 0.0
        description = "Conservative Integrating Lambda"
    elif strategy_name == "ZN":
        Kp = 0.35 / max(abs_k * effective_l, 1e-3)
        Ti = max(3.5 * effective_l, 1e-3)
        Td = 0.0
        description = "Integrating ZN-like"
    else:
        Kp = 0.30 / max(abs_k * effective_l, 1e-3)
        Ti = max(4.0 * effective_l, 1e-3)
        Td = 0.0
        description = "Integrating conservative fallback"

    Ki = _safe_div(Kp, Ti, 0.0)
    Kd = Kp * Td
    params = _clamp_pid_params(Kp, Ki, Kd)
    params.update({"strategy": strategy_name, "model_type": "IPDT", "description": description, "Ti": float(Ti), "Td": float(Td)})
    return params


def apply_tuning_rules(
    K: float,
    T: float,
    L: float,
    strategy: str = "IMC",
    model_type: str = "FOPDT",
    model_params: Dict | None = None,
) -> Dict:
    normalized_model_type = (model_type or "FOPDT").strip().upper()
    model_params = model_params or {}

    if normalized_model_type == "FO":
        return tune_fo(
            K=float(model_params.get("K", K)),
            T=float(model_params.get("T", T)),
            strategy=strategy,
        )
    if normalized_model_type == "SOPDT":
        return tune_sopdt(
            K=float(model_params.get("K", K)),
            T1=float(model_params.get("T1", model_params.get("T", T))),
            T2=float(model_params.get("T2", model_params.get("T", T))),
            L=float(model_params.get("L", L)),
            strategy=strategy,
        )
    if normalized_model_type == "IPDT":
        return tune_ipdt(
            K=float(model_params.get("K", K)),
            L=float(model_params.get("L", L)),
            strategy=strategy,
        )
    return tune_fopdt(
        K=float(model_params.get("K", K)),
        T=float(model_params.get("T", T)),
        L=float(model_params.get("L", L)),
        strategy=strategy,
    )


def controller_logic_translator(raw_params: Dict, brand: str = "Siemens") -> Dict:
    Kp = float(raw_params["Kp"])
    Ki = float(raw_params["Ki"])
    Kd = float(raw_params["Kd"])
    brand_name = (brand or "Generic").strip()
    brand_key = brand_name.lower()

    if brand_key == "siemens":
        PB = 100 / Kp if Kp > 1e-9 else 100.0
        Ti = 1 / Ki if Ki > 1e-9 else 0.0
        Td = Kd / Kp if Kp > 1e-9 else 0.0
        return {"PB": PB, "Ti": Ti, "Td": Td, "format": "PB-Ti-Td", "brand": "Siemens"}

    if brand_key in {"abb", "emerson", "yokogawa", "hollysys"}:
        return {
            "Kp": Kp,
            "Ti": (Kp / Ki) if Ki > 1e-9 else 0.0,
            "Td": (Kd / Kp) if Kp > 1e-9 else 0.0,
            "format": "Kp-Ti-Td",
            "brand": brand_name,
        }

    return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "format": "Standard", "brand": brand_name or "Generic"}
