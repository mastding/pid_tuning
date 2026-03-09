"""
数据分析智能体的Skills
"""
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import signal
from scipy.interpolate import interp1d


HISTORY_DATA_EXPORT_URL = "http://holli-pid-agent.hollysys-project.sit-cloud.ieccloud.hollicube.com/api/data_query/export-history-data-csv"
DEFAULT_LOOP_URI = "/pid_zd/5989fb05a2ce4828a7ae36c682906f2b"
DEFAULT_HISTORY_START_TIME = "1772467200000"
# The user-provided 1771718400 is earlier than the requested start time once normalized.
# Use a valid default 1-day window ending exactly 24 hours after the default start.
DEFAULT_HISTORY_END_TIME = "1772553600000"
MAX_HISTORY_RANGE_MS = 24 * 60 * 60 * 1000
LOCAL_TIMEZONE = "Asia/Shanghai"
PID_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "ts"],
    "SV": ["sv", "sp", "setpoint"],
    "PV": ["pv", "cv", "process_value"],
    "MV": ["mv", "op", "output", "manipulated_value"],
}
MAX_DENOISE_POINTS = 200000


def _median_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - np.median(values))))


def _normalize_time_value(value: str | None, *, fallback: str) -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    return text


def _parse_time_to_ms(value: str) -> int:
    text = str(value).strip()
    if text.isdigit():
        if len(text) <= 10:
            return int(text) * 1000
        return int(text)

    parsed = pd.to_datetime(text)
    if pd.isna(parsed):
        raise ValueError(f"Invalid time value: {value}")
    return int(parsed.timestamp() * 1000)


def fetch_history_data_csv(
    loop_uri: str = DEFAULT_LOOP_URI,
    start_time: str | None = None,
    end_time: str | None = None,
    data_type: str = "interpolated",
    timeout: int = 60,
) -> Dict:
    """
    Skill 0: 调用外部历史数据接口并保存为本地 CSV
    """
    normalized_start_time = _normalize_time_value(start_time, fallback=DEFAULT_HISTORY_START_TIME)
    normalized_end_time = _normalize_time_value(end_time, fallback=DEFAULT_HISTORY_END_TIME)
    normalized_data_type = (data_type or "interpolated").strip().lower()

    if normalized_data_type not in {"raw", "interpolated"}:
        raise ValueError("data_type must be 'raw' or 'interpolated'")

    start_ms = _parse_time_to_ms(normalized_start_time)
    end_ms = _parse_time_to_ms(normalized_end_time)

    if end_ms <= start_ms:
        raise ValueError(
            f"Invalid history range: end_time must be later than start_time. "
            f"Received start_time={normalized_start_time}, end_time={normalized_end_time}"
        )

    if end_ms - start_ms > MAX_HISTORY_RANGE_MS:
        raise ValueError(
            "Invalid history range: each request can fetch at most 1 day of data."
        )

    params = {
        "loop_uri": loop_uri or DEFAULT_LOOP_URI,
        "start_time": str(start_ms),
        "end_time": str(end_ms),
        "data_type": normalized_data_type,
    }

    response = requests.get(HISTORY_DATA_EXPORT_URL, params=params, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/csv" not in content_type and "application/octet-stream" not in content_type:
        preview = response.text[:300]
        if "," not in preview:
            raise ValueError(f"Unexpected response content type: {content_type}, body preview: {preview}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(response.content)
        csv_path = tmp_file.name

    return {
        "csv_path": csv_path,
        "loop_uri": params["loop_uri"],
        "start_time": normalized_start_time,
        "end_time": normalized_end_time,
        "data_type": normalized_data_type,
        "file_size": os.path.getsize(csv_path),
        "status": "历史数据已下载为本地CSV",
    }


def normalize_pid_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    lowered = {col.lower(): col for col in df.columns}
    for standard_name, aliases in PID_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                rename_map[lowered[alias]] = standard_name
                break

    normalized = df.rename(columns=rename_map).copy()
    if "PV" not in normalized.columns or "MV" not in normalized.columns:
        raise ValueError("CSV must contain MV/PV columns or equivalent aliases")
    return normalized


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if ts.dropna().abs().median() > 1e11 else "s"
        df["timestamp"] = (
            pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )
    else:
        df["timestamp"] = pd.to_datetime(ts, errors="coerce")

    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def estimate_sampling_time(df: pd.DataFrame, fallback: float = 1.0) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return fallback

    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return fallback
    return float(deltas.median())


def clean_pid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_pid_columns(df)
    cleaned = parse_timestamp_column(cleaned)

    numeric_cols = [col for col in ["SV", "PV", "MV"] if col in cleaned.columns]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["PV", "MV"]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid MV/PV rows after cleaning")

    if numeric_cols:
        cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(limit_direction="both")
        cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()

    return cleaned


def _adaptive_event_window(
    df: pd.DataFrame,
    event: Dict,
    event_index: int,
    total_events: int,
) -> Dict:
    n = len(df)
    amplitude = abs(float(event.get("amplitude", 0.0)))
    pre_padding = max(30, min(180, int(max(amplitude * 18, 60))))
    max_post_padding = max(180, min(900, int(max(amplitude * 45, 240))))
    settle_span = max(8, min(45, pre_padding // 3))

    window_start = max(0, int(event["start_idx"]) - pre_padding)
    search_limit = min(n, int(event["end_idx"]) + max_post_padding)

    if event_index + 1 < total_events:
        next_event = int(df.attrs.get("step_events", [])[event_index + 1]["start_idx"])
        search_limit = min(search_limit, max(int(event["end_idx"]) + settle_span, next_event - pre_padding // 2))

    sv_end = float(event.get("sv_end", df["SV"].iloc[min(n - 1, int(event["end_idx"]))])) if "SV" in df.columns else None
    tolerance = max(0.05, amplitude * 0.05)
    pv = df["PV"].to_numpy(dtype=float)

    stable_end = None
    if sv_end is not None:
        for idx in range(int(event["end_idx"]), max(int(event["end_idx"]) + 1, search_limit - settle_span)):
            tail = pv[idx: idx + settle_span]
            if len(tail) < settle_span:
                break
            if np.max(np.abs(tail - sv_end)) <= tolerance and (np.max(tail) - np.min(tail)) <= tolerance:
                stable_end = idx + settle_span
                break

    window_end = stable_end if stable_end is not None else search_limit
    window_end = max(window_start + 20, min(n, int(window_end)))
    return {
        **event,
        "window_start_idx": int(window_start),
        "window_end_idx": int(window_end),
    }


def build_candidate_windows(df: pd.DataFrame) -> Tuple[List[Dict], Dict | None]:
    if len(df) < 10:
        return [], None

    if "SV" in df.columns and df["SV"].nunique(dropna=True) > 1:
        threshold = max(0.5, float(df["SV"].std(ddof=0) * 0.2))
        step_events = detect_step_events(df, threshold=threshold)
        if step_events:
            df.attrs["step_events"] = step_events
            candidate_windows = [
                _adaptive_event_window(df, event, idx, len(step_events))
                for idx, event in enumerate(step_events)
            ]
            best_event = max(candidate_windows, key=lambda item: abs(item["amplitude"]))
            return candidate_windows, best_event
        return [], None

    mv_diff = np.abs(np.diff(df["MV"].to_numpy()))
    if len(mv_diff) == 0:
        return [], None

    center = int(np.argmax(mv_diff))
    padding = max(60, min(300, len(df) // 50))
    start_idx = max(0, center - padding)
    end_idx = min(len(df), center + padding)
    event = {
        "start_idx": center,
        "end_idx": min(len(df), center + 1),
        "window_start_idx": start_idx,
        "window_end_idx": end_idx,
        "amplitude": float(mv_diff[center]),
        "type": "mv_change",
    }
    return [event], event


def select_identification_window(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], Dict | None, List[Dict]]:
    candidate_windows, selected_event = build_candidate_windows(df)
    if candidate_windows:
        selected_event = selected_event or candidate_windows[0]
        start_idx = int(selected_event["window_start_idx"])
        end_idx = int(selected_event["window_end_idx"])
        return df.iloc[start_idx:end_idx].reset_index(drop=True), candidate_windows, selected_event, candidate_windows

    if len(df) < 20:
        return df.copy(), [], None, []
    return df.copy(), [], None, []


def prepare_pid_dataset(csv_path: str) -> Dict:
    raw_df = pd.read_csv(csv_path)
    cleaned_df = clean_pid_dataframe(raw_df)
    dt = estimate_sampling_time(cleaned_df)

    denoised_df = cleaned_df.copy()
    if len(denoised_df) >= 25:
        denoised_df["PV"] = adaptive_denoise(denoised_df["PV"].to_numpy(), noise_level="auto")
        denoised_df["MV"] = adaptive_denoise(denoised_df["MV"].to_numpy(), noise_level="low")

    window_df, step_events, selected_event, candidate_windows = select_identification_window(denoised_df)
    if len(window_df) < 20:
        window_df = denoised_df.copy().reset_index(drop=True)
        selected_event = None

    quality_metrics = None
    if selected_event and "SV" in window_df.columns and selected_event.get("type") in {"step_up", "step_down"}:
        quality_metrics = assess_control_quality(
            window_df,
            {
                "start_idx": 0,
                "end_idx": len(window_df),
                "amplitude": selected_event["amplitude"],
                "sv_start": selected_event.get("sv_start", float(window_df["SV"].iloc[0])),
                "sv_end": selected_event.get("sv_end", float(window_df["SV"].iloc[-1])),
                "type": selected_event["type"],
            },
        )

    return {
        "raw_df": raw_df,
        "cleaned_df": denoised_df,
        "window_df": window_df,
        "dt": dt,
        "step_events": step_events,
        "selected_event": selected_event,
        "candidate_windows": candidate_windows,
        "quality_metrics": quality_metrics,
    }


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
    sv = df["SV"].to_numpy(dtype=float)
    if sv.size < 4:
        return []

    sv_diff = np.diff(sv)
    robust_noise = _median_abs(sv_diff)
    diff_threshold = max(float(threshold), robust_noise * 6.0, 1e-6)
    candidate_indices = np.where(np.abs(sv_diff) >= diff_threshold)[0] + 1
    if candidate_indices.size == 0:
        return []

    merge_gap = max(3, min(20, len(df) // 200))
    window = max(3, min(30, len(df) // 100))

    grouped_events: List[List[int]] = []
    current_group = [int(candidate_indices[0])]
    for idx in candidate_indices[1:]:
        if int(idx) - current_group[-1] <= merge_gap:
            current_group.append(int(idx))
        else:
            grouped_events.append(current_group)
            current_group = [int(idx)]
    grouped_events.append(current_group)

    step_events = []
    for group in grouped_events:
        center = int(round(sum(group) / len(group)))
        pre_start = max(0, center - window)
        pre_end = max(pre_start + 1, center)
        post_start = min(len(df) - 1, center)
        post_end = min(len(df), center + window)

        sv_start = float(np.median(sv[pre_start:pre_end]))
        sv_end = float(np.median(sv[post_start:post_end]))
        amplitude = abs(sv_end - sv_start)

        if amplitude < threshold:
            continue

        event_start = max(0, center - merge_gap)
        event_end = min(len(df), center + merge_gap + 1)
        step_events.append({
            "start_idx": int(event_start),
            "end_idx": int(event_end),
            "amplitude": float(amplitude),
            "sv_start": sv_start,
            "sv_end": sv_end,
            "type": "step_up" if sv_end > sv_start else "step_down",
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
    iae = np.trapezoid(np.abs(error), dx=dt)
    
    # ISE: 积分平方误差
    ise = np.trapezoid(error**2, dx=dt)
    
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
    signal_data = np.asarray(signal_data, dtype=float)
    if signal_data.size > MAX_DENOISE_POINTS:
        return signal_data

    if noise_level == 'auto':
        # 自动检测噪声水平
        diff = np.diff(signal_data)
        noise_std = np.std(diff)
        signal_std = np.std(signal_data)
        noise_ratio = noise_std / signal_std if signal_std > 1e-9 else 0.0
        
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
