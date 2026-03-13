from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class WorkflowTaskStore:
    """In-memory task registry for external workflow requests."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create_task(self, task_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        task = {
            "task_id": task_id,
            "status": "pending",
            "request_payload": dict(request_payload),
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "progress": {
                "current_stage": "accepted",
                "current_stage_display": "已受理",
                "percent": 5,
            },
            "result": None,
            "error_message": None,
        }
        with self._lock:
            self._tasks[task_id] = task
        return dict(task)

    def start_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "running"
            task["started_at"] = _now_iso()
            task["progress"] = {
                "current_stage": "accepted",
                "current_stage_display": "任务启动",
                "percent": 10,
            }

    def update_progress(
        self,
        task_id: str,
        *,
        stage: str,
        stage_display: str,
        percent: int,
    ) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["progress"] = {
                "current_stage": stage,
                "current_stage_display": stage_display,
                "percent": percent,
            }

    def complete_task(self, task_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "success"
            task["finished_at"] = _now_iso()
            task["progress"] = {
                "current_stage": "completed",
                "current_stage_display": "任务完成",
                "percent": 100,
            }
            task["result"] = result
            task["error_message"] = None

    def fail_task(self, task_id: str, error_message: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "failed"
            task["finished_at"] = _now_iso()
            task["progress"] = {
                "current_stage": "failed",
                "current_stage_display": "任务失败",
                "percent": 100,
            }
            task["error_message"] = error_message
            task["result"] = None

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task is not None else None
