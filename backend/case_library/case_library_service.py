from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EXTERNAL_CASE_DIR = WORKSPACE_ROOT / "pid-tuning" / "backend" / "data" / "external" / "cases"


def _safe_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_cases() -> List[Dict[str, Any]]:
    if not EXTERNAL_CASE_DIR.exists():
        return []

    items: List[Dict[str, Any]] = []
    for case_path in sorted(EXTERNAL_CASE_DIR.glob("*.case.json")):
        try:
            payload = _safe_read_json(case_path)
        except Exception:
            continue
        items.append(
            {
                "case_id": payload.get("case_id") or case_path.stem,
                "path": str(case_path),
                "payload": payload,
            }
        )
    return items


def refresh_case_library_cache() -> None:
    _load_cases.cache_clear()


def _get_provider(payload: Dict[str, Any]) -> str:
    return str(payload.get("source", {}).get("provider") or "unknown")


def _get_track(payload: Dict[str, Any]) -> str:
    return str(payload.get("benchmark", {}).get("track") or "auxiliary")


LOOP_TYPE_ALIASES = {
    "flow": "flow",
    "flowrate": "flow",
    "mass_flow": "flow",
    "temperature": "temperature",
    "temp": "temperature",
    "thermal": "temperature",
    "pressure": "pressure",
    "press": "pressure",
    "level": "level",
    "liquid_level": "level",
    "tank_level": "level",
    "composition": "composition",
    "concentration": "composition",
    "ph": "composition",
    "unknown": "unknown",
    "": "unknown",
    "none": "unknown",
    "null": "unknown",
}


def _normalize_loop_type(raw_loop_type: Any) -> Dict[str, str]:
    raw = str(raw_loop_type or "").strip()
    lowered = raw.lower()
    normalized = LOOP_TYPE_ALIASES.get(lowered)
    if normalized:
        return {
            "loop_type": normalized,
            "loop_type_raw": raw or "unknown",
            "loop_type_source": "explicit",
            "loop_type_confidence": "high" if normalized != "unknown" else "low",
        }
    return {
        "loop_type": "unknown",
        "loop_type_raw": raw or "unknown",
        "loop_type_source": "explicit",
        "loop_type_confidence": "low",
    }


def _get_loop_type_info(payload: Dict[str, Any]) -> Dict[str, str]:
    return _normalize_loop_type(payload.get("loop", {}).get("loop_type"))


def _get_loop_type(payload: Dict[str, Any]) -> str:
    return _get_loop_type_info(payload)["loop_type"]


def _get_model_type(payload: Dict[str, Any]) -> str:
    return str(payload.get("identified_model", {}).get("family") or "unknown")


def _get_failure_modes(payload: Dict[str, Any]) -> List[str]:
    values = payload.get("labels", {}).get("failure_modes", [])
    return [str(v) for v in values if str(v).strip()]


def _get_quality_flags(payload: Dict[str, Any]) -> List[str]:
    values = payload.get("labels", {}).get("quality_flags", [])
    return [str(v) for v in values if str(v).strip()]


def _get_suitability(payload: Dict[str, Any]) -> Dict[str, Any]:
    return dict(payload.get("benchmark", {}).get("suitability") or {})


def _build_case_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = item["payload"]
    identified = payload.get("identified_model", {}) or {}
    benchmark = payload.get("benchmark", {}) or {}
    scenario = payload.get("scenario", {}) or {}
    loop_type_info = _get_loop_type_info(payload)
    return {
        "case_id": item["case_id"],
        "provider": _get_provider(payload),
        "track": _get_track(payload),
        "loop_type": loop_type_info["loop_type"],
        "loop_type_raw": loop_type_info["loop_type_raw"],
        "loop_type_source": loop_type_info["loop_type_source"],
        "loop_type_confidence": loop_type_info["loop_type_confidence"],
        "model_type": _get_model_type(payload),
        "objective": scenario.get("objective") or "",
        "failure_modes": _get_failure_modes(payload),
        "quality_flags": _get_quality_flags(payload),
        "sample_time_s": payload.get("loop", {}).get("sample_time_s"),
        "fit_metrics": identified.get("fit_metrics") or {},
        "suitability": _get_suitability(payload),
        "source_dataset_id": payload.get("source", {}).get("dataset_id") or item["case_id"],
        "raw_uri": payload.get("source", {}).get("raw_uri") or "",
        "tags": list(scenario.get("tags") or []),
        "path": item["path"],
    }


def get_case_library_stats() -> Dict[str, Any]:
    provider_counts: Dict[str, int] = {}
    track_counts: Dict[str, int] = {}
    loop_type_counts: Dict[str, int] = {}
    failure_mode_counts: Dict[str, int] = {}
    model_type_counts: Dict[str, int] = {}

    cases = _load_cases()
    for item in cases:
        payload = item["payload"]
        provider = _get_provider(payload)
        track = _get_track(payload)
        loop_type = _get_loop_type(payload)
        model_type = _get_model_type(payload)

        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        track_counts[track] = track_counts.get(track, 0) + 1
        loop_type_counts[loop_type] = loop_type_counts.get(loop_type, 0) + 1
        model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
        for failure_mode in _get_failure_modes(payload):
            failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1

    def _dict_to_sorted_items(values: Dict[str, int], key_name: str) -> List[Dict[str, Any]]:
        return [
            {key_name: key, "count": count}
            for key, count in sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))
        ]

    return {
        "total_count": len(cases),
        "provider_distribution": _dict_to_sorted_items(provider_counts, "provider"),
        "track_distribution": _dict_to_sorted_items(track_counts, "track"),
        "loop_type_distribution": _dict_to_sorted_items(loop_type_counts, "loop_type"),
        "model_type_distribution": _dict_to_sorted_items(model_type_counts, "model_type"),
        "failure_mode_distribution": _dict_to_sorted_items(failure_mode_counts, "failure_mode"),
        "source_root": str(EXTERNAL_CASE_DIR),
    }


def list_case_library_items(
    *,
    provider: str = "",
    loop_type: str = "",
    model_type: str = "",
    track: str = "",
    failure_mode: str = "",
    keyword: str = "",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    keyword_norm = keyword.strip().lower()
    failure_mode_norm = failure_mode.strip().lower()

    items: List[Dict[str, Any]] = []
    for item in _load_cases():
        payload = item["payload"]
        summary = _build_case_summary(item)

        if provider and summary["provider"] != provider:
            continue
        if loop_type and summary["loop_type"] != loop_type:
            continue
        if model_type and summary["model_type"] != model_type:
            continue
        if track and summary["track"] != track:
            continue
        if failure_mode_norm and failure_mode_norm not in {v.lower() for v in summary["failure_modes"]}:
            continue
        if keyword_norm:
            haystack = " ".join(
                [
                    summary["case_id"],
                    summary["provider"],
                    summary["loop_type"],
                    summary["model_type"],
                    summary["raw_uri"],
                    str(summary["source_dataset_id"]),
                ]
            ).lower()
            if keyword_norm not in haystack:
                continue
        items.append(summary)
        if len(items) >= limit:
            break
    return items


def get_case_library_detail(case_id: str) -> Dict[str, Any] | None:
    for item in _load_cases():
        payload = item["payload"]
        source_dataset_id = payload.get("source", {}).get("dataset_id")
        if item["case_id"] == case_id or source_dataset_id == case_id:
            return {
                **_build_case_summary(item),
                "source": payload.get("source") or {},
                "loop": payload.get("loop") or {},
                "scenario": payload.get("scenario") or {},
                "identified_model": payload.get("identified_model") or {},
                "current_pid": payload.get("current_pid") or {},
                "labels": payload.get("labels") or {},
                "benchmark": payload.get("benchmark") or {},
                "raw_timeseries_summary": _summarize_timeseries(payload.get("raw_timeseries") or {}),
            }
    return None


def _summarize_timeseries(raw_timeseries: Dict[str, Any]) -> Dict[str, Any]:
    def _series_len(name: str) -> int:
        values = raw_timeseries.get(name) or []
        return len(values) if isinstance(values, list) else 0

    return {
        "sample_count": _series_len("time"),
        "sp_count": _series_len("sp"),
        "pv_count": _series_len("pv"),
        "mv_count": _series_len("mv"),
        "time_start": (raw_timeseries.get("time") or [None])[0] if _series_len("time") else None,
        "time_end": (raw_timeseries.get("time") or [None])[-1] if _series_len("time") else None,
    }


def list_similar_case_library_items(case_id: str, *, limit: int = 5) -> List[Dict[str, Any]]:
    cases = _load_cases()
    target_item: Dict[str, Any] | None = None
    for item in cases:
        payload = item["payload"]
        source_dataset_id = payload.get("source", {}).get("dataset_id")
        if item["case_id"] == case_id or source_dataset_id == case_id:
            target_item = item
            break
    if target_item is None:
        return []

    target_payload = target_item["payload"]
    target_track = _get_track(target_payload)
    target_provider = _get_provider(target_payload)
    target_loop = _get_loop_type(target_payload)
    target_objective = str(target_payload.get("scenario", {}).get("objective") or "")
    target_flags = set(_get_quality_flags(target_payload))
    target_failures = set(_get_failure_modes(target_payload))

    scored: List[Dict[str, Any]] = []
    for item in cases:
        if item["case_id"] == target_item["case_id"]:
            continue
        payload = item["payload"]
        score = 0.0
        reasons: List[str] = []
        if _get_track(payload) == target_track:
            score += 0.30
            reasons.append("track_match")
        if _get_provider(payload) == target_provider:
            score += 0.15
            reasons.append("provider_match")
        if _get_loop_type(payload) == target_loop:
            score += 0.20
            reasons.append("loop_type_match")
        if str(payload.get("scenario", {}).get("objective") or "") == target_objective:
            score += 0.10
            reasons.append("objective_match")
        flags = set(_get_quality_flags(payload))
        failures = set(_get_failure_modes(payload))
        if target_flags and flags:
            overlap = len(target_flags & flags) / max(len(target_flags | flags), 1)
            if overlap > 0:
                score += 0.10 * overlap
                reasons.append("quality_flag_overlap")
        if target_failures and failures:
            overlap = len(target_failures & failures) / max(len(target_failures | failures), 1)
            if overlap > 0:
                score += 0.15 * overlap
                reasons.append("failure_mode_overlap")
        if score <= 0:
            continue
        scored.append(
            {
                **_build_case_summary(item),
                "match_score": round(score, 4),
                "match_reasons": reasons,
            }
        )

    scored.sort(key=lambda value: (-value["match_score"], value["case_id"]))
    return scored[:limit]
