"""
统一评分模块 (Unified Rating Module)
====================================

提供了三层评分架构，所有整定算法必须按照此规范统一打分。

📌【核心接口说明】📌

1. 一键全自动打分 (推荐所有整定算法使用!)
----------------------------------------
接口: `ModelRating.evaluate(model_params, pid_params, **kwargs)`
输入:
    - model_params (Dict): 用来充当考场的虚拟沙盘，必须包含 {'K', 'T1', 'T2', 'L'}。
                           如果方法算不出全局参数(如振荡/继电)，必须提交倒推的局域参数。
    - pid_params (Dict):   考生的控制参数，必须包含 {'Kp', 'Ki', 'Kd'} 或 {'pb', 'ti', 'td'}。
    - **kwargs: 
        - method (str): 算法名称 ('model_identification' / 'oscillation' / 'llm' / 'relay')
        - method_confidence (float): (可选) 该算法第二层的背景置信分。如果传了，会自动算 Layer 3 总分。
        - method_confidence_details (Dict): (可选) 置信度计算细节明细。
        - 仿真环境参数 (loop_type, sp_initial, sp_final, n_steps, dt)
输出 (Dict): 
    {
        'performance_score': 8.5,           # Layer 1 仿真控制品质得分 (0-10)
        'performance_details': dict,        # 超调、时间等具体明细
        'method_confidence': 0.8,           # Layer 2 算法置信度分数 (原样透传)
        'method_confidence_details': dict,  # 原样透传
        'final_rating': 8.35,               # Layer 3 考虑硬约束后的最终加权总分 (0-10)
        'final_details': dict,              # 各层在计算总分时的生效权重信息
        'simulation': dict,                 # 原始那 500 步的每一步历史仿真曲线数组
    }

2. 手动三层打分接口 (用于特殊解耦场景)
----------------------------------------
如果你不想用上方的仿真器自动算，你需要自己走完三步：

- [Layer 1 物理考核]
  - 接口: `ModelRating.performance_score(metrics)`
  - 输入: 一个具有特定属性的对象 (is_stable:bool, overshoot:float, settling_time:float, 
          steady_state_error:float, oscillation_count:int, decay_ratio:float)
  - 输出: `(score: float(0->10), details: dict)`

- [Layer 2 方法背调]
  - 接口: 这里有 4 个不同的函数，因材施教 (例如 `llm_confidence`, `oscillation_confidence`)。
  - 输入: 个性化指标。模型法看 `FusionResult`；LLM法看 `self_score`, `reasoning_quality`...
  - 输出: `(confidence_score: float(0->1), details: dict)` 

- [Layer 3 综合判决]
  - 接口: `ModelRating.final_rating(performance_score, method_confidence)`
  - 输入: 拿上述 L1 输出的 `score` 和 L2 输出的 `confidence` 传进来。
  - 输出: `(final_score: float(0->10), details: dict)`。带有不合格自动熔断骨折机制。

==============================================================================
代码使用示例 (以大模型整定为例)
==============================================================================

    # 1. 大模型输出的参数及自带信心
    pid_params = {'Kp': 1.5, 'Ki': 0.1, 'Kd': 0.0}
    model_params = {'K': 1.2, 'T1': 15.0, 'T2': 0.0, 'L': 2.0}
    llm_self_conf = 0.8
    reasoning_quality = 0.9

    # 2. 先算大模型专享的 Layer 2 置信度
    method_conf, conf_details = ModelRating.llm_confidence(
        llm_self_score=llm_self_conf,
        model_params=model_params,
        pid_params=pid_params,
        reasoning_quality=reasoning_quality,
        consistency_score=0.85,
        process_match_score=1.0
    )

    # 3. 把参数和刚算好的 置信度 扔进 evaluate 进行一条龙打分
    result = ModelRating.evaluate(
        model_params=model_params,
        pid_params=pid_params,
        method='llm',
        method_confidence=method_conf,           # <--- 注入 Layer 2 分数
        method_confidence_details=conf_details
    )

    print(f"闭环性能 (Layer 1): {result['performance_score']}")
    print(f"方法置信度 (Layer 2): {result['method_confidence']}")
    print(f"最终总分 (Layer 3): {result['final_rating']}")

"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class ClosedLoopMetrics:
    """闭环性能指标 (仅用于类型提示，实际使用 tuning.core.data_classes.ClosedLoopMetrics)"""
    is_stable: bool
    settling_time: float
    overshoot: float
    rise_time: float
    steady_state_error: float
    oscillation_count: int
    decay_ratio: float


class ModelRating:
    """
    统一三层评分器
    
    所有方法均为 @staticmethod，无状态，可随时调用。
    """
    
    # ================================================================
    # Layer 1: 闭环性能评分 (0-10)
    # ================================================================
    
    @staticmethod
    def performance_score(metrics) -> Tuple[float, Dict[str, float]]:
        """
        Layer 1: 纯闭环阶跃响应的控制品质评分
        
        始终基于所有五个维度评分，不存在一票否决：
        1. 超调量 (overshoot)        — 权重 25%
        2. 调节时间 (settling_time)   — 权重 20%
        3. 稳态误差 (steady_state_error) — 权重 25%
        4. 振荡次数 (oscillation_count)  — 权重 15%
        5. 衰减比 (decay_ratio)       — 权重 15%
        
        稳定性作为乘性因子，不稳定时整体打折而非直接归零。
        
        Args:
            metrics: ClosedLoopMetrics 对象
        
        Returns:
            (score, details) — score 范围 0.0~10.0
        """
        details = {}
        
        if metrics is None:
            return 5.0, {'reason': 'no_metrics'}
        
        is_stable = getattr(metrics, 'is_stable', True)
        details['is_stable'] = is_stable
        
        # 【1】超调量评分 (0-10)，权重 25%
        overshoot = getattr(metrics, 'overshoot', 0)
        if overshoot <= 2:
            os_score = 10.0
        elif overshoot <= 5:
            os_score = 9.0
        elif overshoot <= 10:
            os_score = 8.0
        elif overshoot <= 15:
            os_score = 7.0
        elif overshoot <= 25:
            os_score = 6.0
        elif overshoot <= 40:
            os_score = 4.0
        elif overshoot <= 60:
            os_score = 2.5
        elif overshoot <= 100:
            os_score = 1.5
        else:
            os_score = max(0.0, 1.0 - (overshoot - 100) / 200)
        details['overshoot'] = round(overshoot, 2)
        details['overshoot_score'] = round(os_score, 2)
        
        # 【2】调节时间评分 (0-10)，权重 20%
        settling_time = getattr(metrics, 'settling_time', float('inf'))
        if settling_time < float('inf'):
            if settling_time <= 15:
                st_score = 10.0
            elif settling_time <= 30:
                st_score = 9.0
            elif settling_time <= 60:
                st_score = 7.5
            elif settling_time <= 120:
                st_score = 6.0
            elif settling_time <= 300:
                st_score = 4.0
            elif settling_time <= 600:
                st_score = 2.0
            else:
                st_score = 1.0
        else:
            st_score = 0.0  # 未收敛
        details['settling_time'] = round(settling_time, 2) if settling_time < float('inf') else -1
        details['settling_time_score'] = round(st_score, 2)
        
        # 【3】稳态误差评分 (0-10)，权重 25%
        sse = getattr(metrics, 'steady_state_error', 0)
        if sse <= 0.5:
            sse_score = 10.0
        elif sse <= 1.0:
            sse_score = 9.0
        elif sse <= 2.0:
            sse_score = 7.5
        elif sse <= 5.0:
            sse_score = 5.5
        elif sse <= 10.0:
            sse_score = 3.5
        elif sse <= 20.0:
            sse_score = 2.0
        else:
            sse_score = max(0.0, 1.0 - (sse - 20) / 50)
        details['steady_state_error'] = round(sse, 2)
        details['steady_state_error_score'] = round(sse_score, 2)
        
        # 【4】振荡次数评分 (0-10)，权重 15%
        osc_count = getattr(metrics, 'oscillation_count', 0)
        if osc_count == 0:
            oc_score = 7.0    # 过阻尼，没有振荡
        elif osc_count <= 2:
            oc_score = 10.0   # 经典欠阻尼，最优
        elif osc_count <= 4:
            oc_score = 7.0
        elif osc_count <= 6:
            oc_score = 5.0
        elif osc_count <= 10:
            oc_score = 3.0
        else:
            oc_score = max(0.0, 2.0 - (osc_count - 10) / 5)
        details['oscillation_count'] = osc_count
        details['oscillation_count_score'] = round(oc_score, 2)
        
        # 【5】衰减比评分 (0-10)，权重 15%
        decay_ratio = getattr(metrics, 'decay_ratio', 0)
        if decay_ratio <= 0.1:
            dr_score = 10.0
        elif decay_ratio <= 0.25:
            dr_score = 9.0   # 经典 4:1 衰减
        elif decay_ratio <= 0.5:
            dr_score = 6.0
        elif decay_ratio <= 0.8:
            dr_score = 3.0
        elif decay_ratio <= 1.0:
            dr_score = 1.0
        else:
            dr_score = 0.0   # 发散
        details['decay_ratio'] = round(decay_ratio, 4)
        details['decay_ratio_score'] = round(dr_score, 2)
        
        # 加权综合
        weights = {
            'overshoot': 0.25,
            'settling_time': 0.20,
            'steady_state_error': 0.25,
            'oscillation_count': 0.15,
            'decay_ratio': 0.15,
        }
        raw_score = (
            weights['overshoot'] * os_score +
            weights['settling_time'] * st_score +
            weights['steady_state_error'] * sse_score +
            weights['oscillation_count'] * oc_score +
            weights['decay_ratio'] * dr_score
        )
        
        # 稳定性因子：不稳定时整体打折（而非一票否决）
        if is_stable:
            stability_factor = 1.0
        else:
            # 不稳定 → 最高只能拿到 raw_score 的 40%
            stability_factor = 0.4
        
        score = raw_score * stability_factor
        score = round(min(10.0, max(0.0, score)), 2)
        
        details['raw_score'] = round(raw_score, 2)
        details['stability_factor'] = stability_factor
        details['weights'] = weights
        
        return score, details
    
    # ================================================================
    # Layer 2: 方法置信度 (0-1)
    # ================================================================
    
    @staticmethod
    def model_id_confidence(fusion, epsilon: float = 1e-10) -> Tuple[float, Dict]:
        """
        Layer 2: 模型辨识路径的方法置信度
        
        维度:
        1. R² 拟合质量   (40%)
        2. 参数一致性     (30%)
        3. 参数物理合理性 (30%)
        
        Args:
            fusion: FusionResult 对象
            epsilon: 数值稳定小量
            
        Returns:
            (confidence, details) — confidence 范围 0.0~1.0
        """
        details = {'method': 'model_identification'}
        
        # 1. R² 质量 (0-1)
        r2 = getattr(fusion, 'global_r2', 0.5)
        if r2 >= 0.95:
            r2_q = 1.0
        elif r2 >= 0.9:
            r2_q = 0.9 + (r2 - 0.9) * 2
        elif r2 >= 0.8:
            r2_q = 0.75 + (r2 - 0.8) * 1.5
        elif r2 >= 0.6:
            r2_q = 0.5 + (r2 - 0.6) * 1.25
        elif r2 >= 0.4:
            r2_q = 0.3 + (r2 - 0.4) * 1.0
        elif r2 >= 0.2:
            r2_q = 0.1 + (r2 - 0.2) * 1.0
        else:
            r2_q = max(0.0, r2 * 0.5)
        details['r2_quality'] = round(r2_q, 4)
        
        # 2. 参数一致性 (0-1)
        n_seg = getattr(fusion, 'n_segments_used', 1)
        consistency = 1.0
        if n_seg > 1:
            k_mean = abs(getattr(fusion, 'K', 1.0)) + epsilon
            k_cv = getattr(fusion, 'K_std', 0.0) / k_mean
            t1_mean = abs(getattr(fusion, 'T1', 1.0)) + epsilon
            t1_cv = getattr(fusion, 'T1_std', 0.0) / t1_mean
            k_c = max(0, 1.0 - k_cv * 1.5)
            t1_c = max(0, 1.0 - t1_cv * 1.5)
            consistency = 0.6 * k_c + 0.4 * t1_c
            cs = getattr(fusion, 'consistency_score', 0)
            if cs > 0:
                consistency = 0.7 * consistency + 0.3 * cs
        else:
            cs = getattr(fusion, 'consistency_score', 0)
            consistency = cs if cs > 0 else 0.6
        consistency = min(1.0, max(0.0, consistency))
        details['param_consistency'] = round(consistency, 4)
        
        # 3. 参数合理性 (0-1)
        validity = 1.0
        K = getattr(fusion, 'K', 1.0)
        T1 = getattr(fusion, 'T1', 10.0)
        L = getattr(fusion, 'L', 1.0)
        T2 = getattr(fusion, 'T2', 0.0)
        
        if abs(K) < 0.001:     validity -= 0.4
        elif abs(K) > 50:      validity -= 0.2
        elif abs(K) < 0.01:    validity -= 0.1
        if T1 <= 0:            validity -= 0.5
        elif T1 < 0.1:         validity -= 0.2
        elif T1 > 500:         validity -= 0.15
        if L < 0:              validity -= 0.3
        elif L > T1 * 2 and T1 > 0: validity -= 0.1
        if T2 < 0:             validity -= 0.2
        validity = max(0.0, validity)
        details['param_validity'] = round(validity, 4)
        
        # 加权
        confidence = 0.4 * r2_q + 0.3 * consistency + 0.3 * validity
        
        # 硬约束
        if r2 < 0.3:
            confidence = min(confidence, 0.3)
        elif r2 < 0.5:
            confidence = min(confidence, 0.5)
        
        confidence = round(min(1.0, max(0.0, confidence)), 4)
        details['confidence_weights'] = {'r2': 0.4, 'consistency': 0.3, 'validity': 0.3}
        return confidence, details
    
    @staticmethod
    def oscillation_confidence(pid_params: Dict, osc_info: Dict,
                                osc_result: Dict,
                                config: Dict = None) -> Tuple[float, Dict, List[str]]:
        """
        Layer 2: 振荡整定路径的方法置信度
        
        维度:
        1. 数据质量   (40%)
        2. 参数边界   (30%)
        3. 方法可靠性 (30%)
        
        Args:
            pid_params: PID 参数字典 (Kp, Ki, Kd, pb, Ti, method...)
            osc_info: 振荡分析信息 (oscillation_ratio, ...)
            osc_result: 振荡整定结果 (data_quality, nonlinearity, valve_issues...)
            config: 振荡整定配置 (pb_min, pb_max...), 默认使用标准值
            
        Returns:
            (confidence, details, warnings)
        """
        if config is None:
            config = {'pb_min': 120.0, 'pb_max': 600.0}
        
        details = {'method': 'oscillation_tuning'}
        warnings = []
        
        # 1. 数据质量 (0-1)
        osc_ratio = osc_info.get('oscillation_ratio', 0.5)
        raw_quality = osc_result.get('data_quality', 0.5)
        nonlinearity = osc_result.get('nonlinearity', 0.0)
        
        if osc_ratio < 0.4:      dq = 0.9
        elif osc_ratio < 0.6:    dq = 0.7 - (osc_ratio - 0.4) * 1.0
        elif osc_ratio < 0.8:    dq = 0.5 - (osc_ratio - 0.6) * 1.0
        else:                    dq = 0.3 - (osc_ratio - 0.8) * 1.5
        dq -= max(0, (0.4 - raw_quality) * 0.5)
        dq -= max(0, (nonlinearity - 0.5) * 0.3)
        dq = max(0.1, min(0.95, dq))
        details['data_quality'] = round(dq, 4)
        details['oscillation_ratio'] = round(osc_ratio, 4)
        
        # 2. 参数边界 (0-1)
        pb = pid_params.get('pb', 200)
        pb_min = config.get('pb_min', 120.0)
        pb_max = config.get('pb_max', 600.0)
        pb_range = pb_max - pb_min
        pb_margin = min(pb - pb_min, pb_max - pb) / (pb_range / 2) if pb_range > 0 else 0.5
        boundary = 0.5 + pb_margin * 0.5
        if pb <= pb_min * 1.05 or pb >= pb_max * 0.95:
            boundary = 0.3
        Ti = pid_params.get('Ti', 2.5)
        if Ti <= 1.6 or Ti >= 9.5:
            boundary -= 0.1
        boundary = min(1.0, max(0.0, boundary))
        details['param_boundary'] = round(boundary, 4)
        
        # 3. 方法可靠性 (0-1)
        method = pid_params.get('method', 'unknown')
        if 'low_gain' in method or 'high_gain' in method:
            rel = 0.5
        elif 'oscillation' in method:
            rel = 0.6
        elif 'integrating' in method:
            rel = 0.55
        else:
            rel = 0.55
        if pid_params.get('Kd', 0) > 0:
            rel += 0.05
        rel = min(1.0, rel)
        details['method_reliability'] = round(rel, 4)
        
        # 加权
        confidence = 0.4 * dq + 0.3 * boundary + 0.3 * rel
        
        # 特殊限制
        valve_issues = osc_result.get('valve_issues', {})
        if osc_ratio > 0.9:
            confidence = min(confidence, 0.5)
            warnings.append(f'极高振荡({osc_ratio:.0%})')
        elif osc_ratio > 0.85:
            confidence = min(confidence, 0.6)
            warnings.append(f'高振荡({osc_ratio:.0%})')
        if pb <= pb_min * 1.02 or pb >= pb_max * 0.98:
            confidence = min(confidence, 0.5)
            warnings.append('PID参数触达边界')
        if raw_quality < 0.3:
            confidence = min(confidence, 0.5)
            warnings.append(f'数据质量极差({raw_quality:.2f})')
        if nonlinearity > 0.6:
            confidence = min(confidence, 0.55)
            warnings.append(f'高非线性({nonlinearity:.2f})')
        if valve_issues.get('has_stiction', False):
            confidence = min(confidence, 0.5)
            warnings.append('阀门粘滞')
        if valve_issues.get('has_deadband', False):
            confidence = min(confidence, 0.55)
            warnings.append('阀门死区')
        
        # 多重风险
        risk = sum([
            osc_ratio > 0.85,
            raw_quality < 0.35,
            valve_issues.get('has_stiction', False) or valve_issues.get('has_deadband', False),
        ])
        if risk >= 2:
            confidence = min(confidence, 0.4)
            warnings.append('❗多重风险因素')
        
        confidence = round(min(1.0, max(0.0, confidence)), 4)
        details['confidence_weights'] = {'data_quality': 0.4, 'param_boundary': 0.3, 'method_reliability': 0.3}
        return confidence, details, warnings
    
    @staticmethod
    def relay_confidence(data_confidence: float,
                          gain_margin: float = 1.0,
                          phase_margin: float = 0.0) -> Tuple[float, Dict]:
        """
        Layer 2: 继电反馈路径的方法置信度
        
        维度:
        1. 数据置信度   (50%)
        2. 稳定性裕度   (50%)
        
        Args:
            data_confidence: 继电识别的数据质量置信度 (0-1)
            gain_margin: 增益裕度 (理想 > 2)
            phase_margin: 相位裕度 (理想 > 45°)
        
        Returns:
            (confidence, details)
        """
        gm_score = min(1.0, gain_margin / 5.0)
        pm_score = min(1.0, phase_margin / 90.0)
        stability_conf = 0.4 * gm_score + 0.6 * pm_score
        
        confidence = 0.5 * data_confidence + 0.5 * stability_conf
        confidence = round(min(1.0, max(0.0, confidence)), 4)
        
        details = {
            'method': 'relay_feedback',
            'data_confidence': round(data_confidence, 4),
            'stability_confidence': round(stability_conf, 4),
            'gain_margin': round(gain_margin, 4),
            'phase_margin': round(phase_margin, 4),
            'confidence_weights': {'data': 0.5, 'stability': 0.5},
        }
        return confidence, details
    
    @staticmethod
    def llm_confidence(llm_self_score: float = 0.5,
                        param_range_ok: bool = True,
                        pid_params: Dict = None,
                        model_params: Dict = None,
                        reasoning_quality: float = None,
                        consistency_score: float = None,
                        process_type_match: bool = None) -> Tuple[float, Dict]:
        """
        Layer 2: 大模型路径的方法置信度
        
        维度 (5项):
        1. LLM 自评信心    (20%) — LLM 回复中的 confidence 字段
        2. 参数合理性       (25%) — PID/模型参数是否在物理合理范围
        3. 推理质量         (20%) — LLM 推理链是否有依据 (可选)
        4. 多次调用一致性    (20%) — 同输入多次调用的变异系数 (可选)
        5. 过程类型匹配     (15%) — LLM 判断的过程类型是否与数据一致 (可选)
        
        Args:
            llm_self_score: LLM 自评信心 (0-1)
            param_range_ok: 参数是否在合理范围（简单 bool，向后兼容）
            pid_params: PID 参数 {'Kp','Ki','Kd'} 或 {'pb','ti','td'}，用于细化参数合理性
            model_params: 模型参数 {'K','T1','T2','L'}，用于交叉检验 PID 合理性
            reasoning_quality: LLM 推理链质量 (0-1)，None 则用默认值 0.5
            consistency_score: 多次调用一致性 (0-1)，None 则用默认值 0.5
            process_type_match: 过程类型匹配，None 则用默认值 True
        
        Returns:
            (confidence, details)
        """
        details = {'method': 'llm'}
        
        # --- 1. LLM 自评 (0-1) ---
        self_score = min(1.0, max(0.0, llm_self_score))
        details['llm_self_score'] = round(self_score, 4)
        
        # --- 2. 参数合理性 (0-1) ---
        if pid_params is not None and model_params is not None:
            # 细化检查
            checks = {}
            
            # 解析 PID
            Kp = pid_params.get('Kp', pid_params.get('kp', 0))
            Ki = pid_params.get('Ki', pid_params.get('ki', 0))
            Kd = pid_params.get('Kd', pid_params.get('kd', 0))
            if 'pb' in pid_params:
                pb = pid_params['pb']
                Kp = 100.0 / pb if pb > 0 else 0
                ti = pid_params.get('ti', 0)
                Ki = Kp / ti if ti > 0 else 0
                Kd = Kp * pid_params.get('td', 0)
            
            K = model_params.get('K', 1.0)
            T1 = model_params.get('T1', 10.0)
            L = model_params.get('L', 1.0)
            
            # 增益检查: Kp 不为零且方向合理
            checks['kp_nonzero'] = abs(Kp) > 0.001
            checks['kp_sign_ok'] = (Kp * K > 0) if abs(K) > 0.001 else True
            
            # 积分检查: Ki >= 0 (不反向积分)
            checks['ki_nonneg'] = Ki >= 0
            
            # 微分检查: Kd 合理
            checks['kd_reasonable'] = Kd >= 0 and (Kd < abs(Kp) * T1 * 2 if T1 > 0 else True)
            
            # 增益裕度粗估: Kp*K 不宜远超临界
            kp_k = abs(Kp * K)
            checks['gain_not_extreme'] = 0.01 < kp_k < 50
            
            # 积分时间 vs 过程时间常数
            Ti_pid = Kp / Ki if Ki > 0.001 else float('inf')
            checks['ti_vs_t1'] = 0.1 < Ti_pid / T1 < 20 if T1 > 0 and Ti_pid < float('inf') else True
            
            param_score = sum(checks.values()) / len(checks)
            details['param_checks'] = checks
        elif param_range_ok:
            param_score = 0.8
        else:
            param_score = 0.3
        param_score = min(1.0, max(0.0, param_score))
        details['param_range_score'] = round(param_score, 4)
        
        # --- 3. 推理质量 (0-1) ---
        rq = reasoning_quality if reasoning_quality is not None else 0.5
        rq = min(1.0, max(0.0, rq))
        details['reasoning_quality'] = round(rq, 4)
        
        # --- 4. 一致性 (0-1) ---
        cs = consistency_score if consistency_score is not None else 0.5
        cs = min(1.0, max(0.0, cs))
        details['consistency_score'] = round(cs, 4)
        
        # --- 5. 过程匹配 (0-1) ---
        pm = 0.9 if (process_type_match is True or process_type_match is None) else 0.3
        details['process_match_score'] = round(pm, 4)
        
        # 加权
        weights = {
            'self_score': 0.20,
            'param_range': 0.25,
            'reasoning': 0.20,
            'consistency': 0.20,
            'process_match': 0.15,
        }
        confidence = (
            weights['self_score'] * self_score +
            weights['param_range'] * param_score +
            weights['reasoning'] * rq +
            weights['consistency'] * cs +
            weights['process_match'] * pm
        )
        
        # 硬约束
        if param_score < 0.3:
            confidence = min(confidence, 0.4)  # 参数明显不合理
        if self_score < 0.2:
            confidence = min(confidence, 0.5)  # LLM 自己都没信心
        
        confidence = round(min(1.0, max(0.0, confidence)), 4)
        details['confidence_weights'] = weights
        return confidence, details
    
    # ================================================================
    # Layer 3: 最终综合评分 (0-10)
    # ================================================================
    
    @staticmethod
    def final_rating(performance_score: float,
                      method_confidence: float,
                      performance_weight: float = 0.7,
                      confidence_weight: float = 0.3) -> Tuple[float, Dict]:
        """
        Layer 3: 结合 Layer 1 和 Layer 2 给出最终评分
        
        公式: final = performance_score * perf_weight + (confidence * 10) * conf_weight
        
        可通过 performance_weight 和 confidence_weight 调整两层的相对重要性。
        默认 70% 看控制品质，30% 看方法可信度。
        
        Args:
            performance_score: Layer 1 闭环性能评分 (0-10)
            method_confidence: Layer 2 方法置信度 (0-1)
            performance_weight: 性能评分权重，默认 0.7
            confidence_weight:  置信度权重，默认 0.3
        
        Returns:
            (final_score, details) — final_score 范围 0.0~10.0
        """
        # 归一化权重
        total_w = performance_weight + confidence_weight
        pw = performance_weight / total_w
        cw = confidence_weight / total_w
        
        # 置信度映射到 0-10 尺度
        confidence_score = method_confidence * 10.0
        
        final = pw * performance_score + cw * confidence_score
        
        # 硬约束：性能极差时，置信度再高也封顶
        if performance_score <= 1.0:
            final = min(final, 3.0)   # 不稳定
        elif performance_score <= 3.0:
            final = min(final, 5.0)   # 控制品质很差
        
        # 硬约束：置信度极低时，性能分打折
        if method_confidence < 0.2:
            final = min(final, 6.0)   # 方法极不可靠
        
        final = round(min(10.0, max(0.0, final)), 2)
        
        details = {
            'performance_score': round(performance_score, 2),
            'method_confidence': round(method_confidence, 4),
            'performance_weight': round(pw, 2),
            'confidence_weight': round(cw, 2),
            'confidence_as_score': round(confidence_score, 2),
        }
        return final, details
    
    # ================================================================
    # 一站式接口: 仿真 + 三层评分
    # ================================================================
    
    @staticmethod
    def simulate_step_response(model_params: Dict, pid_params: Dict,
                                sp_initial: float = 50.0, sp_final: float = 60.0,
                                pv_initial: float = None,
                                n_steps: int = 500, dt: float = 1.0,
                                loop_type: str = 'flow') -> Dict:
        """
        独立的闭环阶跃仿真（不依赖任何 Mixin）
        
        输入:
            model_params: {'K': float, 'T1': float, 'T2': float, 'L': float}
                - K:  过程增益
                - T1: 主时间常数 (s)
                - T2: 二阶时间常数 (s, 0 表示一阶)
                - L:  纯滞后 (s)
            pid_params: {'Kp': float, 'Ki': float, 'Kd': float} 或
                        {'pb': float, 'ti': float, 'td': float}
                - Kp/pb: 比例增益/比例带
                - Ki/ti: 积分增益/积分时间
                - Kd/td: 微分增益/微分时间
            sp_initial: 设定值初值
            sp_final: 设定值终值（阶跃后）
            pv_initial: PV 初值，默认 = sp_initial
            n_steps: 仿真步数
            dt: 采样周期 (s)
            loop_type: 回路类型 ('flow'/'temperature'/'level'/'pressure')
            
        输出:
            {
                'is_stable': bool,
                'overshoot': float (%),
                'settling_time': float (s),
                'steady_state_error': float (%),
                'oscillation_count': int,
                'decay_ratio': float,
                'rise_time': float (s),
                'pv_history': list,
                'mv_history': list,
                'sp_history': list,
            }
        """
        eps = 1e-10
        
        # 解析模型参数
        model_type = str(model_params.get('model_type', 'FOPDT')).upper()
        K = model_params.get('K', 1.0)
        T1 = max(model_params.get('T1', 10.0), eps)
        T2 = model_params.get('T2', 0.0)
        L = max(model_params.get('L', 0.0), 0.0)
        
        # 解析 PID 参数（兼容两种格式）
        if 'Kp' in pid_params:
            Kp = pid_params['Kp']
            Ki = pid_params.get('Ki', 0.0)
            Kd = pid_params.get('Kd', 0.0)
        elif 'kp' in pid_params:
            Kp = pid_params['kp']
            Ki = pid_params.get('ki', 0.0)
            Kd = pid_params.get('kd', 0.0)
        else:
            pb = pid_params.get('pb', 100.0)
            ti = pid_params.get('ti', 0.0)
            td = pid_params.get('td', 0.0)
            Kp = 100.0 / pb if pb > 0 else 1.0
            Ki = Kp / ti if ti > 0 else 0.0
            Kd = Kp * td
        
        if pv_initial is None:
            pv_initial = sp_initial
        
        # 仿真
        pv_hist = np.zeros(n_steps)
        mv_hist = np.zeros(n_steps)
        sp_hist = np.zeros(n_steps)
        
        sp_change = sp_final - sp_initial
        mv_mid = 50.0
        if abs(K) > eps:
            mv_offset = np.clip(-sp_change / K * 0.3, -25, 25)
            mv0 = np.clip(mv_mid + mv_offset, 5, 95)
        else:
            mv0 = mv_mid
        
        delta_pv = 0.0
        delta_x2 = 0.0
        integral = 0.0
        prev_error = 0.0
        from collections import deque
        delay_steps = int(L / dt)
        delta_mv_buf = deque([0.0] * (delay_steps + 1))
        step_time = 10
        
        for t in range(n_steps):
            sp = sp_initial if t < step_time else sp_final
            sp_hist[t] = sp
            pv = pv_initial + delta_pv
            pv_hist[t] = pv
            
            error = sp - pv
            integral += error * dt
            integral_limit = 100.0 / (abs(Ki) + eps)
            integral = np.clip(integral, -integral_limit, integral_limit)
            derivative = (error - prev_error) / dt if t > 0 else 0.0
            
            mv = mv0 + Kp * error + Ki * integral + Kd * derivative
            mv = np.clip(mv, 0.0, 100.0)
            delta_mv = mv - mv0
            
            mv_hist[t] = mv
            prev_error = error
            
            delta_mv_buf.append(delta_mv)
            delta_mv_delayed = delta_mv_buf.popleft()
            
            # 过程模型更新
            if model_type == 'IPDT':
                # Integrating-plus-dead-time: dPV/dt = K * delayed_dMV
                delta_pv = delta_pv + K * delta_mv_delayed * dt
            else:
                alpha1 = dt / T1
                if T2 > eps:
                    # 二阶
                    alpha2 = dt / max(T2, T1 * 0.1)
                    delta_pv_new = delta_pv + alpha1 * (K * delta_mv_delayed - delta_pv)
                    delta_x2 = delta_x2 + alpha2 * (delta_pv_new - delta_x2)
                    delta_pv = delta_x2
                else:
                    # 一阶
                    delta_pv = delta_pv + alpha1 * (K * delta_mv_delayed - delta_pv)
        
        # 计算指标
        pv_resp = pv_hist[step_time:]
        
        if len(pv_resp) < 10 or abs(sp_change) < eps:
            return {
                'is_stable': False, 'overshoot': 0.0, 'settling_time': float('inf'),
                'steady_state_error': 100.0, 'oscillation_count': 0, 'decay_ratio': 1.0,
                'rise_time': float('inf'),
                'pv_history': pv_hist.tolist(), 'mv_history': mv_hist.tolist(),
                'sp_history': sp_hist.tolist(),
            }
        
        # 稳态误差
        final_portion = pv_resp[-max(10, len(pv_resp)//10):]
        sse = abs(np.mean(final_portion) - sp_final) / (abs(sp_change) + eps) * 100
        
        # 超调量
        if sp_change > 0:
            overshoot = max(0, (np.max(pv_resp) - sp_final) / sp_change * 100)
        else:
            overshoot = max(0, (sp_final - np.min(pv_resp)) / abs(sp_change) * 100)
        
        # 上升时间
        t10 = sp_initial + 0.1 * sp_change
        t90 = sp_initial + 0.9 * sp_change
        r_start = r_end = None
        for i, p in enumerate(pv_resp):
            if sp_change > 0:
                if r_start is None and p >= t10: r_start = i
                if r_end is None and p >= t90: r_end = i; break
            else:
                if r_start is None and p <= t10: r_start = i
                if r_end is None and p <= t90: r_end = i; break
        rise_time = (r_end - r_start) * dt if r_start is not None and r_end is not None else float('inf')
        
        # 调节时间
        tolerance = 0.02 * abs(sp_change)
        settling_time = float('inf')
        for i in range(len(pv_resp) - 1, -1, -1):
            if abs(pv_resp[i] - sp_final) > tolerance:
                if i < len(pv_resp) - 1:
                    settling_time = (i + 1) * dt
                break
        else:
            settling_time = 0.0
        
        # 振荡 + 衰减比
        err_sig = pv_resp - sp_final
        zero_cross = np.where(np.diff(np.signbit(err_sig)))[0]
        osc_count = len(zero_cross) // 2
        
        peaks = []
        for i in range(1, len(err_sig) - 1):
            if err_sig[i] > err_sig[i-1] and err_sig[i] > err_sig[i+1]:
                peaks.append(abs(err_sig[i]))
            if len(peaks) >= 2:
                break
        decay_ratio = peaks[1] / peaks[0] if len(peaks) >= 2 and peaks[0] > eps else (0.0 if len(peaks) <= 1 else 1.0)
        
        # 稳定性判定
        max_settling = 600.0
        max_overshoot = 65.0 if decay_ratio <= 0.6 else 30.0
        max_sse = 8.0
        
        is_settled = settling_time < max_settling
        is_accurate = sse < max_sse
        is_smooth = overshoot < max_overshoot
        is_decaying = decay_ratio < 0.8
        is_stable = is_settled and is_accurate and is_smooth and is_decaying
        
        # 边界容忍
        if not is_stable and is_settled:
            fail_count = sum([not is_accurate, not is_smooth, not is_decaying])
            if fail_count == 1:
                marginal = False
                if not is_accurate and sse < max_sse * 1.5: marginal = True
                if not is_smooth and overshoot < max_overshoot * 1.3: marginal = True
                if not is_decaying and decay_ratio < 1.0: marginal = True
                if marginal:
                    is_stable = True
        
        return {
            'is_stable': is_stable,
            'overshoot': round(overshoot, 2),
            'settling_time': round(settling_time, 2) if settling_time < float('inf') else -1,
            'steady_state_error': round(sse, 2),
            'oscillation_count': osc_count,
            'decay_ratio': round(decay_ratio, 4),
            'rise_time': round(rise_time, 2) if rise_time < float('inf') else -1,
            'pv_history': pv_hist.tolist(),
            'mv_history': mv_hist.tolist(),
            'sp_history': sp_hist.tolist(),
        }
    
    @staticmethod
    def evaluate(model_params: Dict, pid_params: Dict,
                  method: str = 'unknown',
                  method_confidence: float = None,
                  method_confidence_details: Dict = None,
                  sp_initial: float = 50.0, sp_final: float = 60.0,
                  n_steps: int = 500, dt: float = 1.0,
                  loop_type: str = 'flow') -> Dict:
        """
        一站式评估接口：模型参数 + PID 参数 → 三层评分
        
        任何整定路径都可以调用，包括大模型。
        
        输入:
            model_params: {'K': float, 'T1': float, 'T2': float, 'L': float}
            pid_params:   {'Kp': float, 'Ki': float, 'Kd': float} 或 pb/ti/td 格式
            method:       整定方法名 ('model_identification'/'oscillation'/'relay'/'llm')
            method_confidence: Layer 2 置信度 (0-1)，不传则跳过 L2/L3
            method_confidence_details: Layer 2 详情字典
            sp_initial/sp_final: 仿真设定值
            n_steps/dt: 仿真参数
            loop_type: 回路类型
        
        输出:
            {
                'performance_score': float (0-10),     # Layer 1
                'performance_details': dict,
                'method_confidence': float (0-1),      # Layer 2
                'method_confidence_details': dict,
                'final_rating': float (0-10),           # Layer 3
                'final_details': dict,
                'simulation': dict,                     # 仿真原始结果
            }
        """
        # 仿真
        sim = ModelRating.simulate_step_response(
            model_params, pid_params,
            sp_initial=sp_initial, sp_final=sp_final,
            n_steps=n_steps, dt=dt, loop_type=loop_type
        )
        
        from types import SimpleNamespace
        
        # 构造 metrics 对象给 performance_score
        m = SimpleNamespace(
            is_stable=sim['is_stable'],
            overshoot=sim['overshoot'],
            settling_time=sim['settling_time'] if sim['settling_time'] >= 0 else float('inf'),
            steady_state_error=sim['steady_state_error'],
            oscillation_count=sim['oscillation_count'],
            decay_ratio=sim['decay_ratio']
        )
        
        # Layer 1
        perf_score, perf_details = ModelRating.performance_score(m)
        
        result = {
            'performance_score': perf_score,
            'performance_details': perf_details,
            'simulation': sim,
        }
        
        # Layer 2 + Layer 3（如果提供了置信度）
        if method_confidence is not None:
            result['method_confidence'] = method_confidence
            result['method_confidence_details'] = method_confidence_details or {'method': method}
            
            final, final_details = ModelRating.final_rating(perf_score, method_confidence)
            result['final_rating'] = final
            result['final_details'] = final_details
        
        return result

