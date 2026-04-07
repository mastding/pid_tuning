import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

SESSIONS_FILE = Path(__file__).resolve().parent / "frontend_sessions.json"


def _parse_session_time(value: Any) -> float:
    if not value:
        return 0.0
    raw = str(value).strip()
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})$", raw)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        try:
            return datetime(y, mo, d, h, mi, s).timestamp()
        except Exception:
            return 0.0
    try:
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return 0.0


def _session_sort_key(session: Dict[str, Any]) -> float:
    return max(_parse_session_time(session.get("updatedAt")), _parse_session_time(session.get("createdAt")))


def get_frontend_sessions() -> Dict[str, Any]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_frontend_sessions(data: Dict[str, Any]) -> None:
    existing = get_frontend_sessions()

    incoming_items = data.get("items")
    existing_items = existing.get("items")
    if isinstance(incoming_items, list) and isinstance(existing_items, list):
        merged_map: Dict[str, Dict[str, Any]] = {}
        for item in existing_items:
            if isinstance(item, dict) and item.get("id"):
                merged_map[str(item["id"])] = item
        for item in incoming_items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            sid = str(item["id"])
            prev = merged_map.get(sid)
            if prev is None or _session_sort_key(item) >= _session_sort_key(prev):
                merged_map[sid] = item

        merged_items = list(merged_map.values())
        merged_items.sort(key=_session_sort_key, reverse=True)

        merged_payload: Dict[str, Any] = {**existing, **data}
        merged_payload["items"] = merged_items
        if not merged_payload.get("selectedTaskSessionId"):
            merged_payload["selectedTaskSessionId"] = existing.get("selectedTaskSessionId")
        merged_payload["taskSessionCounter"] = max(
            int(existing.get("taskSessionCounter") or 0),
            int(data.get("taskSessionCounter") or 0),
        )
        SESSIONS_FILE.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
