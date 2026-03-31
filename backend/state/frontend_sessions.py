import json
from pathlib import Path
from typing import Dict, Any

SESSIONS_FILE = Path(__file__).resolve().parent / "frontend_sessions.json"

def get_frontend_sessions() -> Dict[str, Any]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_frontend_sessions(data: Dict[str, Any]) -> None:
    SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
