from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_MODEL_NAME = "qwen-plus"
DEFAULT_MODEL_API_URL = ""
DEFAULT_MODEL_API_KEY = ""
DEFAULT_HISTORY_DATA_API_URL = (
    "http://holli-pid-agent.hollysys-project.sit-cloud.ieccloud.hollicube.com/api/agent/history-data-raw"
)
DEFAULT_KNOWLEDGE_GRAPH_API_URL = "http://graphrag.dicp.sixseven.ltd:5924/api/query"
DEFAULT_KNOWLEDGE_GRAPH_ID = "build_20260317_003858"

ENV_FILE_PATH = Path(__file__).resolve().parents[2] / ".env"


def _read_env_map() -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    if not ENV_FILE_PATH.exists():
        return env_map

    for raw_line in ENV_FILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_map[key.strip()] = value.strip()
    return env_map


def _write_env_map(env_map: Dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in env_map.items()]
    ENV_FILE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _env_value(key: str, fallback: str = "") -> str:
    value = os.getenv(key)
    if value is not None and str(value).strip() != "":
        return str(value).strip()
    file_value = _read_env_map().get(key)
    if file_value is not None and str(file_value).strip() != "":
        return str(file_value).strip()
    return fallback


def get_runtime_system_config() -> Dict[str, Any]:
    return {
        "model": {
            "name": _env_value("MODEL", DEFAULT_MODEL_NAME),
            "api_url": _env_value("MODEL_API_URL", DEFAULT_MODEL_API_URL),
            "api_key": _env_value("MODEL_API_KEY", DEFAULT_MODEL_API_KEY),
        },
        "integration": {
            "history_data_api_url": _env_value("HISTORY_DATA_API_URL", DEFAULT_HISTORY_DATA_API_URL),
            "knowledge_graph_api_url": _env_value("KNOWLEDGE_GRAPH_API_URL", DEFAULT_KNOWLEDGE_GRAPH_API_URL),
            "knowledge_graph_id": _env_value("KNOWLEDGE_GRAPH_ID", DEFAULT_KNOWLEDGE_GRAPH_ID),
        },
    }


def update_runtime_system_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    current_env = _read_env_map()

    model_payload = dict(payload.get("model") or {})
    integration_payload = dict(payload.get("integration") or {})

    updates = {
        "MODEL": str(model_payload.get("name") or _env_value("MODEL", DEFAULT_MODEL_NAME)).strip(),
        "MODEL_API_URL": str(model_payload.get("api_url") or _env_value("MODEL_API_URL", DEFAULT_MODEL_API_URL)).strip(),
        "MODEL_API_KEY": str(model_payload.get("api_key") or _env_value("MODEL_API_KEY", DEFAULT_MODEL_API_KEY)).strip(),
        "HISTORY_DATA_API_URL": str(
            integration_payload.get("history_data_api_url")
            or _env_value("HISTORY_DATA_API_URL", DEFAULT_HISTORY_DATA_API_URL)
        ).strip(),
        "KNOWLEDGE_GRAPH_API_URL": str(
            integration_payload.get("knowledge_graph_api_url")
            or _env_value("KNOWLEDGE_GRAPH_API_URL", DEFAULT_KNOWLEDGE_GRAPH_API_URL)
        ).strip(),
    }

    for key, value in updates.items():
        current_env[key] = value
        os.environ[key] = value

    if "KNOWLEDGE_GRAPH_ID" not in current_env:
        current_env["KNOWLEDGE_GRAPH_ID"] = _env_value("KNOWLEDGE_GRAPH_ID", DEFAULT_KNOWLEDGE_GRAPH_ID)
        os.environ["KNOWLEDGE_GRAPH_ID"] = current_env["KNOWLEDGE_GRAPH_ID"]

    _write_env_map(current_env)
    return get_runtime_system_config()


def get_model_runtime_config() -> Dict[str, str]:
    config = get_runtime_system_config()["model"]
    return {
        "api_key": str(config["api_key"]),
        "base_url": str(config["api_url"]),
        "model": str(config["name"]),
    }


def get_history_data_api_url() -> str:
    return str(get_runtime_system_config()["integration"]["history_data_api_url"])


def get_knowledge_graph_runtime_config() -> Dict[str, str]:
    config = get_runtime_system_config()["integration"]
    return {
        "graph_api_url": str(config["knowledge_graph_api_url"]),
        "graph_id": str(config["knowledge_graph_id"]),
    }
