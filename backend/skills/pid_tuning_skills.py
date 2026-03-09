# PID Tuning Skills
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
    model_confidence: float,
    r2_score: float,
    normalized_rmse: float,
) -> Dict[str, str]:
    loop_name = (loop_type or "flow").strip().lower()
    tau_ratio = max(float(L), 0.0) / max(float(T), 1e-6)
    fast_process = float(T) <= 5.0
    high_quality_model = (
        model_confidence >= 0.88
        and normalized_rmse <= 0.05
        and r2_score >= 0.97
    )

    if model_confidence < 0.35:
        strategy = "IMC"
        reason = "模型可信度很低，采用最保守的 IMC。"
    elif model_confidence < 0.55 or normalized_rmse > 0.1 or r2_score < 0.75:
        strategy = "LAMBDA"
        reason = "模型质量一般，优先采用更稳健的 Lambda 整定。"
    elif loop_name == "temperature":
        strategy = "IMC"
        reason = "温度回路通常惯性较大，优先抑制超调。"
    elif loop_name == "level":
        strategy = "LAMBDA"
        reason = "液位回路更关注平稳性与鲁棒性。"
    elif loop_name == "pressure":
        strategy = "IMC" if tau_ratio >= 0.3 else "LAMBDA"
        reason = "压力回路偏快，优先选稳健策略控制波动。"
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

    return {"strategy": strategy, "reason": reason, "loop_type": loop_name}


def apply_tuning_rules(K: float, T: float, L: float, strategy: str = "IMC") -> Dict:
    strategy_name = (strategy or "IMC").strip().upper()
    abs_k = max(abs(K), 1e-6)
    T = max(float(T), 1e-3)
    L = max(float(L), 0.0)

    if strategy_name == "IMC":
        lambda_c = max(L, 0.8 * T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T, 1e-3)
        Td = 0.5 * L if L > 0 else 0.0
        tuning_type = "Conservative IMC"
    elif strategy_name in {"LAMBDA", "LAMBDA_TUNING"}:
        lambda_c = max(1.5 * L, T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T + 0.5 * L, 1e-3)
        Td = 0.0
        tuning_type = "Lambda Tuning"
    elif strategy_name == "ZN":
        effective_l = max(L, 0.1 * T, 1e-3)
        Kp = 1.2 * T / (abs_k * effective_l)
        Ti = 2.0 * effective_l
        Td = 0.5 * effective_l
        tuning_type = "Aggressive Ziegler-Nichols"
    elif strategy_name in {"CHR", "CHR_0OS"}:
        effective_l = max(L, 0.1 * T, 1e-3)
        Kp = 0.6 * T / (abs_k * effective_l)
        Ti = max(T, 1e-3)
        Td = 0.5 * effective_l
        tuning_type = "CHR 0% Overshoot"
    else:
        lambda_c = max(L, T, 1e-3)
        Kp = T / (abs_k * (lambda_c + L))
        Ti = max(T, 1e-3)
        Td = 0.0
        tuning_type = "Fallback IMC-like"

    Ki = _safe_div(Kp, Ti, 0.0)
    Kd = Kp * Td
    params = _clamp_pid_params(Kp, Ki, Kd)
    params.update(
        {
            "strategy": strategy_name,
            "description": tuning_type,
            "Ti": float(Ti),
            "Td": float(Td),
        }
    )
    return params


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
