"""
系统辨识智能体的Skills
"""
import numpy as np
from scipy import signal, optimize
from scipy.signal import correlate
from typing import Dict, Tuple
import pandas as pd


def estimate_dead_time(mv: np.ndarray, pv: np.ndarray, dt: float = 1.0) -> float:
    """
    Skill 1: 使用互相关函数估计死区时间
    
    Args:
        mv: 操纵变量序列
        pv: 过程变量序列
        dt: 采样周期（秒）
    
    Returns:
        估计的死区时间L（秒）
    """
    # 去除均值
    mv_centered = mv - np.mean(mv)
    pv_centered = pv - np.mean(pv)
    
    # 计算互相关
    correlation = correlate(pv_centered, mv_centered, mode='full')
    
    # 找到最大相关的位置
    lags = np.arange(-len(mv) + 1, len(mv))
    max_corr_idx = np.argmax(correlation)
    lag = lags[max_corr_idx]
    
    # 转换为时间
    dead_time = abs(lag) * dt
    
    return dead_time


def fit_fopdt_model(mv: np.ndarray, pv: np.ndarray, dt: float = 1.0) -> Dict:
    """
    Skill 2: 拟合一阶加纯滞后模型 (FOPDT)
    
    模型形式: G(s) = K * exp(-L*s) / (T*s + 1)
    
    Args:
        mv: 操纵变量序列
        pv: 过程变量序列
        dt: 采样周期（秒）
    
    Returns:
        Dict with K (gain), T (time constant), L (dead time), residue
    """
    # 首先估计死区时间
    L_init = estimate_dead_time(mv, pv, dt)
    
    # 归一化数据
    mv_mean = np.mean(mv)
    pv_mean = np.mean(pv)
    mv_std = np.std(mv)
    pv_std = np.std(pv)
    
    mv_norm = (mv - mv_mean) / mv_std
    pv_norm = (pv - pv_mean) / pv_std
    
    # 定义FOPDT模型
    def fopdt_response(params, t, u):
        K, T, L = params
        
        # 离散化FOPDT模型
        y = np.zeros_like(t)
        L_steps = int(L / dt)
        
        for i in range(len(t)):
            if i < L_steps:
                y[i] = 0
            else:
                # 一阶系统的离散化
                alpha = dt / (T + dt)
                if i == L_steps:
                    y[i] = K * alpha * u[i - L_steps]
                else:
                    y[i] = (1 - alpha) * y[i-1] + K * alpha * u[i - L_steps]
        
        return y
    
    # 定义目标函数
    def objective(params):
        K, T, L = params
        if T <= 0 or L < 0:
            return 1e10
        
        t = np.arange(len(mv_norm)) * dt
        y_pred = fopdt_response(params, t, mv_norm)
        residue = np.sum((pv_norm - y_pred)**2)
        return residue
    
    # 初始猜测
    K_init = pv_std / mv_std if mv_std > 0 else 1.0
    T_init = 10.0  # 初始时间常数
    
    # 优化
    result = optimize.minimize(
        objective,
        x0=[K_init, T_init, L_init],
        bounds=[(0.1, 10), (1, 100), (0, 50)],
        method='L-BFGS-B'
    )
    
    K_opt, T_opt, L_opt = result.x
    
    # 计算残差
    t = np.arange(len(mv_norm)) * dt
    y_pred = fopdt_response(result.x, t, mv_norm)
    residue = np.sqrt(np.mean((pv_norm - y_pred)**2))
    
    # 反归一化增益
    K_real = K_opt * pv_std / mv_std
    
    return {
        'K': K_real,
        'T': T_opt,
        'L': L_opt,
        'residue': residue,
        'success': result.success
    }


def calculate_model_confidence(residue: float, threshold: float = 0.15) -> Dict:
    """
    Skill 3: 计算模型置信度
    
    Args:
        residue: 拟合残差
        threshold: 置信度阈值
    
    Returns:
        Dict with confidence score and recommendation
    """
    # 基于残差计算置信度
    confidence = max(0, 1 - residue / threshold)
    
    if confidence > 0.85:
        recommendation = "模型可信，可以直接用于PID整定"
        quality = "excellent"
    elif confidence > 0.7:
        recommendation = "模型基本可信，建议谨慎使用"
        quality = "good"
    elif confidence > 0.5:
        recommendation = "模型置信度较低，建议进行开环阶跃测试"
        quality = "fair"
    else:
        recommendation = "模型不可信，可能存在严重扰动或非线性，需要重新采集数据"
        quality = "poor"
    
    return {
        'confidence': confidence,
        'quality': quality,
        'recommendation': recommendation,
        'residue': residue
    }


def validate_model(model_params: Dict, mv_test: np.ndarray, pv_test: np.ndarray, dt: float = 1.0) -> Dict:
    """
    验证模型性能
    
    Args:
        model_params: 模型参数 (K, T, L)
        mv_test: 测试集MV数据
        pv_test: 测试集PV数据
        dt: 采样周期
    
    Returns:
        验证结果
    """
    K = model_params['K']
    T = model_params['T']
    L = model_params['L']
    
    # 模拟模型响应
    L_steps = int(L / dt)
    y_pred = np.zeros_like(pv_test)
    
    for i in range(len(pv_test)):
        if i < L_steps:
            y_pred[i] = pv_test[0]
        else:
            alpha = dt / (T + dt)
            if i == L_steps:
                y_pred[i] = pv_test[0] + K * alpha * mv_test[i - L_steps]
            else:
                y_pred[i] = (1 - alpha) * y_pred[i-1] + K * alpha * mv_test[i - L_steps]
    
    # 计算验证指标
    mse = np.mean((pv_test - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pv_test - y_pred))
    
    # R²分数
    ss_res = np.sum((pv_test - y_pred)**2)
    ss_tot = np.sum((pv_test - np.mean(pv_test))**2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score,
        'validation_passed': r2_score > 0.7
    }
