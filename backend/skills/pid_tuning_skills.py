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
