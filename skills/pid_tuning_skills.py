# PID Tuning Skills
import numpy as np
from typing import Dict

def apply_tuning_rules(K: float, T: float, L: float, strategy: str = "IMC") -> Dict:
    if strategy == "IMC":
        lambda_c = max(L, 0.8 * T)
        Kp = T / (K * (lambda_c + L))
        Ki = Kp / T
        Kd = 0
        tuning_type = "Conservative"
    elif strategy == "ZN":
        Kp = 1.2 * T / (K * L)
        Ki = Kp / (2 * L)
        Kd = 0.5 * Kp * L
        tuning_type = "Aggressive"
    else:
        Kp, Ki, Kd = 1.0, 0.1, 0.0
        tuning_type = "Default"
    
    return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "strategy": strategy, "description": tuning_type}

def controller_logic_translator(raw_params: Dict, brand: str = "Siemens") -> Dict:
    Kp = raw_params["Kp"]
    Ki = raw_params["Ki"]
    Kd = raw_params["Kd"]
    
    if brand == "Siemens":
        PB = 100 / Kp if Kp > 0 else 100
        Ti = 1 / Ki if Ki > 0 else 0
        Td = Kd / Kp if Kp > 0 else 0
        return {"PB": PB, "Ti": Ti, "Td": Td, "format": "PB-Ti-Td", "brand": brand}
    
    return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "format": "Standard", "brand": "Generic"}
