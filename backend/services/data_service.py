from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from skills.data_analysis_skills import prepare_pid_dataset


def build_window_overview(
    cleaned_df: Any,
    selected_window: Dict[str, Any] | None,
    max_points: int = 240,
) -> Dict[str, Any]:
    if cleaned_df is None or len(cleaned_df) == 0:
        return {"points": [], "window_start": 0, "window_end": 0}

    pv = cleaned_df["PV"].to_numpy(dtype=float)
    mv = cleaned_df["MV"].to_numpy(dtype=float)
    n = len(cleaned_df)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

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
        if timestamp_strings is not None:
            point["time"] = timestamp_strings[i]
        points.append(point)

    return {
        "points": points,
        "window_start": window_start,
        "window_end": window_end,
        "total_points": int(n),
        "x_axis": "timestamp" if timestamp_strings is not None else "index",
        "start_time": timestamp_strings[0] if timestamp_strings is not None else None,
        "end_time": timestamp_strings[-1] if timestamp_strings is not None else None,
        "window_start_time": timestamp_strings[window_start] if timestamp_strings is not None else None,
        "window_end_time": timestamp_strings[window_end] if timestamp_strings is not None else None,
    }


def load_pid_dataset(csv_path: str) -> Dict[str, Any]:
    prepared = prepare_pid_dataset(csv_path)
    cleaned_df = prepared["cleaned_df"]
    window_df = prepared["window_df"]
    dt = float(prepared["dt"])
    step_events = prepared["step_events"]
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
    window_overview = build_window_overview(cleaned_df, selected_window)

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
        "quality_metrics": quality_metrics,
        "available_columns": [str(col) for col in cleaned_df.columns.tolist()],
        "mv_range": [float(cleaned_df["MV"].min()), float(cleaned_df["MV"].max())],
        "pv_range": [float(cleaned_df["PV"].min()), float(cleaned_df["PV"].max())],
        "data_points": int(len(cleaned_df)),
        "window_points": int(len(window_df)),
        "sampling_time": dt,
        "status": "数据已完成清洗、降噪和辨识窗口选择",
    }

