from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Dict


ARTIFACTS_ROOT = Path(__file__).resolve().parent / "task_artifacts"


def _safe_segment(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return text or fallback


def ensure_task_artifact_dir(task_id: str | None) -> Dict[str, str]:
    resolved_task_id = _safe_segment(task_id or "", f"task_{uuid.uuid4().hex[:12]}")
    task_dir = ARTIFACTS_ROOT / resolved_task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return {
        "task_id": resolved_task_id,
        "artifact_dir": str(task_dir),
    }


def persist_uploaded_csv(*, task_id: str | None, original_filename: str, content: bytes) -> Dict[str, Any]:
    target = ensure_task_artifact_dir(task_id)
    suffix = Path(original_filename or "uploaded.csv").suffix or ".csv"
    stem = _safe_segment(Path(original_filename or "uploaded.csv").stem, "uploaded")
    saved_name = f"uploaded__{stem}{suffix}"
    saved_path = Path(target["artifact_dir"]) / saved_name
    saved_path.write_bytes(content)
    file_hash = hashlib.sha256(content).hexdigest()
    return {
        **target,
        "uploaded_file_name": original_filename or saved_name,
        "uploaded_file_hash": file_hash,
        "original_file_path": str(saved_path),
        "original_file_size": len(content),
    }


def persist_processed_csv(
    *,
    task_id: str | None,
    cleaned_df: Any,
    selected_loop_prefix: str | None = None,
) -> Dict[str, Any]:
    target = ensure_task_artifact_dir(task_id)
    loop_part = _safe_segment(selected_loop_prefix or "all_loops", "all_loops")
    saved_name = f"processed__{loop_part}.csv"
    saved_path = Path(target["artifact_dir"]) / saved_name
    cleaned_df.to_csv(saved_path, index=False, encoding="utf-8-sig")
    return {
        **target,
        "processed_file_name": saved_name,
        "processed_file_path": str(saved_path),
        "processed_rows": int(len(cleaned_df)) if cleaned_df is not None else 0,
        "processed_columns": [str(col) for col in getattr(cleaned_df, "columns", [])],
    }
