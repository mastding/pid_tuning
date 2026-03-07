"""
数据分析智能体的Skills
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import ruptures as rpt
from typing import Dict, List, Tuple


def load_and_slice_data(csv_path: str, max_pv_change: bool = True) -> pd.DataFrame:
    """
    Skill 1: 读取CSV数据并提取最大PV变化的数据切片
    
    Args:
        csv_path: CSV文件路径
        max_pv_change: 是否提取最大PV变化段
    
    Returns:
        DataFrame with columns: timestamp, SV, PV, MV
    """
    df = pd.read_csv(csv_path)
    
    # 确保必要的列存在
    required_cols = ['timestamp', 'SV', 'PV', 'MV']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 转换时间戳
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if max_pv_change:
        # 计算PV的滑动窗口变化量
        window_size = 100
        df['pv_change'] = df['PV'].rolling(window=window_size).apply(
            lambda x: x.max() - x.min()
        )
        
        # 找到最大变化的位置
        max_change_idx = df['pv_change'].idxmax()
        
        # 提取该区域前后的数据
        start_idx = max(0, max_change_idx - window_size * 2)
        end_idx = min(len(df), max_change_idx + window_size * 2)
        
        df_slice = df.iloc[start_idx:end_idx].copy()
        df_slice = df_slice.drop(columns=['pv_change'])
        
        return df_slice
    
    return df


def detect_step_events(df: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
    """
    Skill 2: 检测SV的阶跃事件
    
    Args:
        df: 数据DataFrame
        threshold: 阶跃检测阈值
    
    Returns:
        List of step events with start_idx, end_idx, amplitude
    """
    # 计算SV的一阶差分
    sv_diff = np.diff(df['SV'].values)
    
    # 使用ruptures库检测变点
    algo = rpt.Pelt(model="rbf").fit(df['SV'].values)
    change_points = algo.predict(pen=10)
    
    step_events = []
    
    for i in range(len(change_points) - 1):
        start_idx = change_points[i] if i == 0 else change_points[i-1]
        end_idx = change_points[i]
        
        # 计算该段的SV变化幅值
        sv_start = df['SV'].iloc[start_idx]
        sv_end = df['SV'].iloc[end_idx-1] if end_idx < len(df) else df['SV'].iloc[-1]
        amplitude = abs(sv_end - sv_start)
        
        if amplitude > threshold:
            step_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'amplitude': amplitude,
                'sv_start': sv_start,
                'sv_end': sv_end,
                'type': 'step_up' if sv_end > sv_start else 'step_down'
            })
    
    return step_events


def assess_control_quality(df: pd.DataFrame, step_event: Dict) -> Dict:
    """
    Skill 3: 评估控制质量
    
    Args:
        df: 数据DataFrame
        step_event: 阶跃事件信息
    
    Returns:
        Dict with IAE, ISE, overshoot, rise_time, settling_time
    """
    start_idx = step_event['start_idx']
    end_idx = step_event['end_idx']
    
    # 提取该段数据
    segment = df.iloc[start_idx:end_idx].copy()
    
    # 计算误差
    error = segment['SV'] - segment['PV']
    
    # 计算时间间隔（假设采样周期为1秒）
    if 'timestamp' in segment.columns:
        time_diff = segment['timestamp'].diff().dt.total_seconds()
        dt = time_diff.median()
    else:
        dt = 1.0
    
    # IAE: 积分绝对误差
    iae = np.trapz(np.abs(error), dx=dt)
    
    # ISE: 积分平方误差
    ise = np.trapz(error**2, dx=dt)
    
    # 超调量
    sv_final = step_event['sv_end']
    pv_max = segment['PV'].max()
    pv_min = segment['PV'].min()
    
    if step_event['type'] == 'step_up':
        overshoot = max(0, (pv_max - sv_final) / step_event['amplitude'] * 100)
    else:
        overshoot = max(0, (sv_final - pv_min) / step_event['amplitude'] * 100)
    
    # 上升时间（10%-90%）
    pv_10 = step_event['sv_start'] + 0.1 * step_event['amplitude']
    pv_90 = step_event['sv_start'] + 0.9 * step_event['amplitude']
    
    try:
        idx_10 = segment[segment['PV'] >= pv_10].index[0]
        idx_90 = segment[segment['PV'] >= pv_90].index[0]
        rise_time = (idx_90 - idx_10) * dt
    except:
        rise_time = None
    
    # 调节时间（2%误差带）
    tolerance = 0.02 * step_event['amplitude']
    settled_mask = np.abs(error) <= tolerance
    
    try:
        # 找到最后一次超出误差带的位置
        unsettled_indices = np.where(~settled_mask)[0]
        if len(unsettled_indices) > 0:
            settling_idx = unsettled_indices[-1] + 1
            settling_time = (settling_idx - start_idx) * dt
        else:
            settling_time = 0
    except:
        settling_time = None
    
    return {
        'IAE': iae,
        'ISE': ise,
        'overshoot_percent': overshoot,
        'rise_time': rise_time,
        'settling_time': settling_time,
        'description': f"阶跃幅值{step_event['amplitude']:.2f}, 超调{overshoot:.1f}%, 调节时间{settling_time:.1f}s"
    }


def adaptive_denoise(signal_data: np.ndarray, noise_level: str = 'auto') -> np.ndarray:
    """
    Skill 4: 自适应去噪
    
    Args:
        signal_data: 信号数据
        noise_level: 噪声水平 ('low', 'medium', 'high', 'auto')
    
    Returns:
        去噪后的信号
    """
    if noise_level == 'auto':
        # 自动检测噪声水平
        diff = np.diff(signal_data)
        noise_std = np.std(diff)
        signal_std = np.std(signal_data)
        noise_ratio = noise_std / signal_std
        
        if noise_ratio < 0.05:
            noise_level = 'low'
        elif noise_ratio < 0.15:
            noise_level = 'medium'
        else:
            noise_level = 'high'
    
    if noise_level == 'low':
        # 轻度去噪：移动平均
        window_size = 3
        denoised = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
    
    elif noise_level == 'medium':
        # 中度去噪：中值滤波
        from scipy.ndimage import median_filter
        denoised = median_filter(signal_data, size=5)
    
    else:  # high
        # 重度去噪：Butterworth低通滤波
        fs = 1.0  # 采样频率
        cutoff = 0.1  # 截止频率
        order = 4
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        denoised = signal.filtfilt(b, a, signal_data)
    
    return denoised
