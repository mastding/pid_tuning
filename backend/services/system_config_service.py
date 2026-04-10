from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


# ============ 服务端口配置 ============
DEFAULT_BACKEND_PORT = "4443"
DEFAULT_BACKEND_HOST = "0.0.0.0"
DEFAULT_FRONTEND_PORT = "5873"
DEFAULT_FRONTEND_HOST = "127.0.0.1"

# ============ LLM 模型配置 ============
DEFAULT_MODEL_NAME = "qwen-plus"
DEFAULT_MODEL_API_URL = ""
DEFAULT_MODEL_API_KEY = ""
DEFAULT_MODEL_TIMEOUT_CONNECT = "20"
DEFAULT_MODEL_TIMEOUT_READ = "120"
DEFAULT_MODEL_TIMEOUT_WRITE = "60"
DEFAULT_MODEL_TIMEOUT_POOL = "30"
DEFAULT_ENABLE_LLM_ORCHESTRATION = "1"

# ============ 外部集成配置 ============
DEFAULT_HISTORY_DATA_API_URL = (
    "http://holli-pid-agent.hollysys-project.sit-cloud.ieccloud.hollicube.com/api/agent/history-data-raw"
)
DEFAULT_KNOWLEDGE_GRAPH_API_URL = "http://graphrag.dicp.sixseven.ltd:5924/api/query"
DEFAULT_KNOWLEDGE_GRAPH_ID = "build_20260317_003858"
DEFAULT_ENABLE_KNOWLEDGE_EXPERT = "1"

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


def _env_bool(key: str, fallback: str = "0") -> bool:
    raw = str(_env_value(key, fallback)).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(raw)


def get_runtime_system_config() -> Dict[str, Any]:
    return {
        "server": {
            "backend": {
                "host": _env_value("BACKEND_HOST", DEFAULT_BACKEND_HOST),
                "port": int(_env_value("BACKEND_PORT", DEFAULT_BACKEND_PORT)),
            },
            "frontend": {
                "host": _env_value("FRONTEND_HOST", DEFAULT_FRONTEND_HOST),
                "port": int(_env_value("FRONTEND_PORT", DEFAULT_FRONTEND_PORT)),
            },
        },
        "model": {
            "name": _env_value("MODEL", DEFAULT_MODEL_NAME),
            "api_url": _env_value("MODEL_API_URL", DEFAULT_MODEL_API_URL),
            "api_key": _env_value("MODEL_API_KEY", DEFAULT_MODEL_API_KEY),
            "timeout_connect_seconds": float(_env_value("MODEL_TIMEOUT_CONNECT", DEFAULT_MODEL_TIMEOUT_CONNECT) or 0),
            "timeout_read_seconds": float(_env_value("MODEL_TIMEOUT_READ", DEFAULT_MODEL_TIMEOUT_READ) or 0),
            "timeout_write_seconds": float(_env_value("MODEL_TIMEOUT_WRITE", DEFAULT_MODEL_TIMEOUT_WRITE) or 0),
            "timeout_pool_seconds": float(_env_value("MODEL_TIMEOUT_POOL", DEFAULT_MODEL_TIMEOUT_POOL) or 0),
            "enable_llm_orchestration": _env_bool("ENABLE_LLM_ORCHESTRATION", DEFAULT_ENABLE_LLM_ORCHESTRATION),
        },
        "integration": {
            "history_data_api_url": _env_value("HISTORY_DATA_API_URL", DEFAULT_HISTORY_DATA_API_URL),
            "knowledge_graph_api_url": _env_value("KNOWLEDGE_GRAPH_API_URL", DEFAULT_KNOWLEDGE_GRAPH_API_URL),
            "knowledge_graph_id": _env_value("KNOWLEDGE_GRAPH_ID", DEFAULT_KNOWLEDGE_GRAPH_ID),
            "enable_knowledge_expert": _env_bool("ENABLE_KNOWLEDGE_EXPERT", DEFAULT_ENABLE_KNOWLEDGE_EXPERT),
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
        "MODEL_TIMEOUT_CONNECT": str(
            model_payload.get("timeout_connect_seconds") or _env_value("MODEL_TIMEOUT_CONNECT", DEFAULT_MODEL_TIMEOUT_CONNECT)
        ).strip(),
        "MODEL_TIMEOUT_READ": str(
            model_payload.get("timeout_read_seconds") or _env_value("MODEL_TIMEOUT_READ", DEFAULT_MODEL_TIMEOUT_READ)
        ).strip(),
        "MODEL_TIMEOUT_WRITE": str(
            model_payload.get("timeout_write_seconds") or _env_value("MODEL_TIMEOUT_WRITE", DEFAULT_MODEL_TIMEOUT_WRITE)
        ).strip(),
        "MODEL_TIMEOUT_POOL": str(
            model_payload.get("timeout_pool_seconds") or _env_value("MODEL_TIMEOUT_POOL", DEFAULT_MODEL_TIMEOUT_POOL)
        ).strip(),
        "ENABLE_LLM_ORCHESTRATION": "1"
        if bool(model_payload.get("enable_llm_orchestration", _env_bool("ENABLE_LLM_ORCHESTRATION", DEFAULT_ENABLE_LLM_ORCHESTRATION)))
        else "0",
        "HISTORY_DATA_API_URL": str(
            integration_payload.get("history_data_api_url")
            or _env_value("HISTORY_DATA_API_URL", DEFAULT_HISTORY_DATA_API_URL)
        ).strip(),
        "KNOWLEDGE_GRAPH_API_URL": str(
            integration_payload.get("knowledge_graph_api_url")
            or _env_value("KNOWLEDGE_GRAPH_API_URL", DEFAULT_KNOWLEDGE_GRAPH_API_URL)
        ).strip(),
        "ENABLE_KNOWLEDGE_EXPERT": "1"
        if bool(integration_payload.get("enable_knowledge_expert", _env_bool("ENABLE_KNOWLEDGE_EXPERT", DEFAULT_ENABLE_KNOWLEDGE_EXPERT)))
        else "0",
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


def get_model_timeout_config() -> Dict[str, float]:
    config = get_runtime_system_config()["model"]
    return {
        "connect": float(config.get("timeout_connect_seconds", float(DEFAULT_MODEL_TIMEOUT_CONNECT))),
        "read": float(config.get("timeout_read_seconds", float(DEFAULT_MODEL_TIMEOUT_READ))),
        "write": float(config.get("timeout_write_seconds", float(DEFAULT_MODEL_TIMEOUT_WRITE))),
        "pool": float(config.get("timeout_pool_seconds", float(DEFAULT_MODEL_TIMEOUT_POOL))),
    }


def get_server_port_config() -> Dict[str, Any]:
    """获取后端和前端服务器端口配置"""
    return {
        "backend": {
            "host": _env_value("BACKEND_HOST", DEFAULT_BACKEND_HOST),
            "port": int(_env_value("BACKEND_PORT", DEFAULT_BACKEND_PORT)),
        },
        "frontend": {
            "host": _env_value("FRONTEND_HOST", DEFAULT_FRONTEND_HOST),
            "port": int(_env_value("FRONTEND_PORT", DEFAULT_FRONTEND_PORT)),
        },
    }


def get_backend_port_config() -> Dict[str, Any]:
    """获取后端服务器配置（用于启动uvicorn）"""
    config = get_server_port_config()["backend"]
    return config


def get_history_data_api_url() -> str:
    return str(get_runtime_system_config()["integration"]["history_data_api_url"])


def get_knowledge_graph_runtime_config() -> Dict[str, str]:
    config = get_runtime_system_config()["integration"]
    return {
        "graph_api_url": str(config["knowledge_graph_api_url"]),
        "graph_id": str(config["knowledge_graph_id"]),
    }


def is_knowledge_expert_enabled() -> bool:
    return bool(get_runtime_system_config()["integration"].get("enable_knowledge_expert", True))


def is_llm_orchestration_enabled() -> bool:
    return bool(get_runtime_system_config()["model"].get("enable_llm_orchestration", True))
