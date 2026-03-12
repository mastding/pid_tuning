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
    model_confidence: float,
    r2_score: float,
    normalized_rmse: float,
) -> Dict[str, str]:
    loop_name = (loop_type or "flow").strip().lower()
    normalized_model_type = (model_type or "FOPDT").strip().upper()
    tau_ratio = max(float(L), 0.0) / max(float(T), 1e-6)
    fast_process = float(T) <= 5.0
    high_quality_model = model_confidence >= 0.88 and normalized_rmse <= 0.05 and r2_score >= 0.97

    if model_confidence < 0.35:
        return {
            "strategy": "IMC",
            "reason": "模型可信度很低，优先采用最保守的 IMC 整定。",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if model_confidence < 0.55 or normalized_rmse > 0.1 or r2_score < 0.75:
        return {
            "strategy": "LAMBDA",
            "reason": "模型质量一般，优先采用更稳健的 Lambda 整定。",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if normalized_model_type == "FO":
        if loop_name in {"flow", "pressure"} and high_quality_model and not fast_process:
            strategy = "IMC"
            reason = "一阶对象且模型质量较高，采用 FO-IMC 整定。"
        else:
            strategy = "LAMBDA"
            reason = "一阶对象优先采用保守的 Lambda/IMC 类整定。"
        return {"strategy": strategy, "reason": reason, "loop_type": loop_name, "model_type": normalized_model_type}

    if normalized_model_type == "IPDT":
        return {
            "strategy": "LAMBDA",
            "reason": "积分过程优先采用积分过程专用的保守 Lambda 整定。",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if normalized_model_type == "SOPDT":
        strategy = "LAMBDA" if loop_name in {"temperature", "level"} else "IMC"
        return {
            "strategy": strategy,
            "reason": "SOPDT 模型优先采用保守的 Lambda/IMC，并保留主导与次级时间常数信息。",
            "loop_type": loop_name,
            "model_type": normalized_model_type,
        }

    if loop_name == "temperature":
        strategy = "IMC"
        reason = "温度回路惯性较大，优先抑制超调。"
    elif loop_name == "level":
        strategy = "LAMBDA"
        reason = "液位回路更关注稳定性与鲁棒性。"
    elif loop_name == "pressure":
        strategy = "IMC" if tau_ratio >= 0.3 else "LAMBDA"
        reason = "压力回路偏快，优先选择稳健策略控制波动。"
    elif loop_name == "flow":
        if high_quality_model and tau_ratio >= 0.08 and not fast_process:
            strategy = "ZN"
            reason = "流量回路模型质量很高且存在一定时滞，可尝试更积极的 ZN 整定。"
        elif fast_process or tau_ratio < 0.08:
            strategy = "IMC"
            reason = "流量回路对象较快或时滞很小，优先使用更稳健的 IMC 抑制振荡。"
        else:
            strategy = "LAMBDA"
            reason = "流量回路模型可用但未达到激进整定条件，优先采用平衡的 Lambda 整定。"
    else:
        strategy = "IMC"
        reason = "未识别回路类型，使用默认稳健策略 IMC。"

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

    # Use the native second-order shape rather than collapsing directly to T1+T2.
    # shape_index reflects how distributed the two time constants are.
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
