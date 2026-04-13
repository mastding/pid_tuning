from __future__ import annotations

from typing import Any, Dict

import numpy as np

import pandas as pd

from skills.data_analysis_skills import prepare_pid_dataset


def enrich_step_events_with_time(cleaned_df: Any, step_events: list[Dict[str, Any]] | None) -> list[Dict[str, Any]]:
    if cleaned_df is None or len(cleaned_df) == 0 or not step_events:
        return step_events or []
    if "timestamp" not in cleaned_df.columns:
        return step_events or []

    timestamps = cleaned_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    last_index = len(timestamps) - 1
    enriched: list[Dict[str, Any]] = []
    for event in step_events:
        start_idx = max(0, min(int(event.get("start_idx", 0)), last_index))
        end_idx = max(start_idx, min(int(event.get("end_idx", start_idx)), last_index))
        enriched.append({
            **event,
            "start_time": timestamps[start_idx],
            "end_time": timestamps[end_idx],
        })
    return enriched


def build_window_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    max_points: int = 240,
    allowed_positions: list[int] | None = None,
) -> Dict[str, Any]:
    if cleaned_df is None or len(cleaned_df) == 0:
        return {"points": [], "window_start": 0, "window_end": 0}

    pv = cleaned_df["PV"].to_numpy(dtype=float)
    mv = cleaned_df["MV"].to_numpy(dtype=float)
    sv = cleaned_df["SV"].to_numpy(dtype=float) if "SV" in cleaned_df.columns else None
    n = len(cleaned_df)
    source_positions = [int(pos) for pos in (allowed_positions or list(range(n))) if 0 <= int(pos) < n]
    if not source_positions:
        return {"points": [], "window_start": 0, "window_end": 0}
    sample_count = len(source_positions)
    step = max(1, sample_count // max_points)
    indices = source_positions[::step]
    if indices[-1] != source_positions[-1]:
        indices.append(source_positions[-1])

    window_start = int((selected_window or {}).get("start_index", 0))
    window_end = int((selected_window or {}).get("end_index", n - 1))
    window_start = max(0, min(window_start, n - 1))
    window_end = max(window_start, min(window_end, n - 1))

    timestamp_strings = None
    if "timestamp" in cleaned_df.columns:
        timestamp_strings = cleaned_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    points = []
    for i in indices:
        point = {
            "index": int(i),
            "pv": float(pv[i]),
            "mv": float(mv[i]),
            "in_window": bool(window_start <= i <= window_end),
        }
        if sv is not None:
            point["sv"] = float(sv[i])
        if timestamp_strings is not None:
            point["time"] = timestamp_strings[i]
        points.append(point)

    return {
        "points": points,
        "window_start": window_start,
        "window_end": window_end,
        "total_points": int(sample_count),
        "x_axis": "timestamp" if timestamp_strings is not None else "index",
        "start_time": timestamp_strings[source_positions[0]] if timestamp_strings is not None else None,
        "end_time": timestamp_strings[source_positions[-1]] if timestamp_strings is not None else None,
        "window_start_time": timestamp_strings[window_start] if timestamp_strings is not None else None,
        "window_end_time": timestamp_strings[window_end] if timestamp_strings is not None else None,
    }


def _format_timestamp_value(value: pd.Timestamp | Any) -> str:
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _interpolate_series_value(array: Any, left_idx: int, right_idx: int, ratio: float) -> float:
    left_value = float(array[left_idx])
    right_value = float(array[right_idx])
    return left_value + (right_value - left_value) * float(ratio)


def _build_interpolated_point(
    *,
    cleaned_df: Any,
    target_time: pd.Timestamp,
    left_idx: int,
    right_idx: int,
    window_start: int,
    window_end: int,
) -> Dict[str, Any]:
    timestamps = cleaned_df["timestamp"]
    pv = cleaned_df["PV"].to_numpy(dtype=float)
    mv = cleaned_df["MV"].to_numpy(dtype=float)
    sv = cleaned_df["SV"].to_numpy(dtype=float) if "SV" in cleaned_df.columns else None
    left_time = timestamps.iloc[left_idx]
    right_time = timestamps.iloc[right_idx]
    left_ns = left_time.value
    right_ns = right_time.value
    ratio = 0.0 if right_ns == left_ns else (target_time.value - left_ns) / (right_ns - left_ns)
    synthetic_index = float(left_idx) + float(ratio)
    point = {
        "index": synthetic_index,
        "pv": _interpolate_series_value(pv, left_idx, right_idx, ratio),
        "mv": _interpolate_series_value(mv, left_idx, right_idx, ratio),
        "in_window": bool(window_start <= synthetic_index <= window_end),
        "time": _format_timestamp_value(target_time),
    }
    if sv is not None:
        point["sv"] = _interpolate_series_value(sv, left_idx, right_idx, ratio)
    return point


def build_time_range_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    *,
    start_time: Any,
    end_time: Any,
    max_points: int = 240,
) -> Dict[str, Any]:
    if cleaned_df is None or len(cleaned_df) == 0 or "timestamp" not in cleaned_df.columns:
        return {"points": [], "window_start": 0, "window_end": 0}

    timestamps = cleaned_df["timestamp"]
    timestamp_index = pd.DatetimeIndex(timestamps)
    start_dt = pd.to_datetime(start_time, errors="coerce")
    end_dt = pd.to_datetime(end_time, errors="coerce")
    if pd.isna(start_dt) and pd.isna(end_dt):
        return build_window_overview(cleaned_df, selected_window, max_points=max_points)

    if pd.isna(start_dt):
        start_dt = timestamps.iloc[0]
    if pd.isna(end_dt):
        end_dt = timestamps.iloc[-1]
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    if end_dt < timestamp_index[0] or start_dt > timestamp_index[-1]:
        return {"points": [], "window_start": 0, "window_end": 0}

    left_bound = max(start_dt, timestamp_index[0])
    right_bound = min(end_dt, timestamp_index[-1])
    positions = np.where((timestamp_index >= left_bound) & (timestamp_index <= right_bound))[0].tolist()
    window_start = int((selected_window or {}).get("start_index", 0))
    window_end = int((selected_window or {}).get("end_index", len(cleaned_df) - 1))

    sampled_points: list[Dict[str, Any]] = []
    left_insert_idx = int(timestamp_index.searchsorted(left_bound, side="left"))
    right_insert_idx = int(timestamp_index.searchsorted(right_bound, side="left"))

    def build_boundary_point(boundary_time: pd.Timestamp, insert_idx: int) -> Dict[str, Any]:
        if insert_idx < len(timestamp_index) and timestamp_index[insert_idx] == boundary_time:
            idx = int(insert_idx)
            point = {
                "index": float(idx),
                "pv": float(cleaned_df["PV"].iloc[idx]),
                "mv": float(cleaned_df["MV"].iloc[idx]),
                "in_window": bool(window_start <= idx <= window_end),
                "time": _format_timestamp_value(boundary_time),
            }
            if "SV" in cleaned_df.columns:
                point["sv"] = float(cleaned_df["SV"].iloc[idx])
            return point
        if insert_idx > 0 and insert_idx < len(timestamp_index):
            return _build_interpolated_point(
                cleaned_df=cleaned_df,
                target_time=pd.Timestamp(boundary_time),
                left_idx=insert_idx - 1,
                right_idx=insert_idx,
                window_start=window_start,
                window_end=window_end,
            )
        idx = max(0, min(len(cleaned_df) - 1, insert_idx if insert_idx < len(cleaned_df) else len(cleaned_df) - 1))
        point = {
            "index": float(idx),
            "pv": float(cleaned_df["PV"].iloc[idx]),
            "mv": float(cleaned_df["MV"].iloc[idx]),
            "in_window": bool(window_start <= idx <= window_end),
            "time": _format_timestamp_value(boundary_time),
        }
        if "SV" in cleaned_df.columns:
            point["sv"] = float(cleaned_df["SV"].iloc[idx])
        return point

    start_point = build_boundary_point(pd.Timestamp(left_bound), left_insert_idx)
    end_point = build_boundary_point(pd.Timestamp(right_bound), right_insert_idx)
    sampled_points.extend([start_point, end_point])

    max_points = max(2, int(max_points or 2))
    middle_budget = max(0, max_points - 2)
    middle_positions = [
        int(pos) for pos in positions
        if float(start_point.get("index", -1)) < int(pos) < float(end_point.get("index", -1))
    ]
    if middle_positions and middle_budget > 0:
        step = max(1, len(middle_positions) // middle_budget)
        sampled_positions = middle_positions[::step][:middle_budget]
        core_overview = build_window_overview(
            cleaned_df,
            selected_window,
            max_points=max(1, len(sampled_positions)),
            allowed_positions=sampled_positions,
        )
        sampled_points.extend(core_overview.get("points") or [])

    dedup_map: Dict[str, Dict[str, Any]] = {}
    for point in sampled_points:
        dedup_map[str(point.get("time") or point.get("index"))] = point
    points = sorted(
        dedup_map.values(),
        key=lambda item: (
            pd.Timestamp(item.get("time")).value if item.get("time") else 0,
            float(item.get("index") or 0),
        ),
    )
    points = [
        {
            key: value
            for key, value in point.items()
            if value is not None
        }
        for point in points
    ]

    return {
        "points": points,
        "window_start": window_start,
        "window_end": window_end,
        "total_points": len(points),
        "x_axis": "timestamp",
        "start_time": _format_timestamp_value(pd.Timestamp(left_bound)),
        "end_time": _format_timestamp_value(pd.Timestamp(right_bound)),
        "window_start_time": _format_timestamp_value(timestamps.iloc[max(0, min(window_start, len(cleaned_df) - 1))]),
        "window_end_time": _format_timestamp_value(timestamps.iloc[max(0, min(window_end, len(cleaned_df) - 1))]),
    }


def load_pid_dataset(
    csv_path: str,
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    max_points: int = 240,
    start_time: str | None = None,
    end_time: str | None = None,
) -> Dict[str, Any]:
    prepared = prepare_pid_dataset(
        csv_path,
        selected_loop_prefix=selected_loop_prefix,
        selected_window_index=selected_window_index,
        start_time=start_time,
        end_time=end_time,
    )
    cleaned_df = prepared["cleaned_df"]
    window_df = prepared["window_df"]
    dt = float(prepared["dt"])
    step_events = enrich_step_events_with_time(cleaned_df, prepared["step_events"])
    candidate_windows = prepared.get("candidate_windows") or []
    selected_event = prepared["selected_event"]
    quality_metrics = prepared["quality_metrics"] or {}

    selected_window = {
        "rows": int(len(window_df)),
        "start_index": int(selected_event.get("window_start_idx", selected_event.get("start_idx", 0))) if selected_event else 0,
        "end_index": int(selected_event.get("window_end_idx", selected_event.get("end_idx", int(len(window_df))))) if selected_event else int(len(window_df)),
        "event_start_index": int(selected_event["start_idx"]) if selected_event else 0,
        "event_end_index": int(selected_event["end_idx"]) if selected_event else int(len(window_df)),
        "event_type": str(selected_event.get("type", "full_range")) if selected_event else "full_range",
    }
    window_overview = build_window_overview(cleaned_df, selected_window, max_points=max_points)
    history_range = {
        "start_time": window_overview.get("start_time"),
        "end_time": window_overview.get("end_time"),
    }

    return {
        "csv_path": csv_path,
        "cleaned_df": cleaned_df,
        "window_df": window_df,
        "mv": window_df["MV"].to_numpy(dtype=float),
        "pv": window_df["PV"].to_numpy(dtype=float),
        "dt": dt,
        "step_events": step_events,
        "candidate_windows": candidate_windows,
        "selected_event": selected_event,
        "selected_window": selected_window,
        "window_overview": window_overview,
        "history_range": history_range,
        "quality_metrics": quality_metrics,
        "available_columns": [str(col) for col in cleaned_df.columns.tolist()],
        "mv_range": [float(cleaned_df["MV"].min()), float(cleaned_df["MV"].max())],
        "pv_range": [float(cleaned_df["PV"].min()), float(cleaned_df["PV"].max())],
        "data_points": int(len(cleaned_df)),
        "window_points": int(len(window_df)),
        "sampling_time": dt,
        "status": "数据已完成清洗、降噪和辨识窗口选择",
    }

