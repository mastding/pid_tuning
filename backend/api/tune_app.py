from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import tempfile
import uuid
from typing import Any, AsyncGenerator, Callable, Dict

import httpx
from fastapi import FastAPI, File, Form, Header, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from case_library import (
    get_case_library_detail,
    get_case_library_stats,
    list_case_library_items,
    list_similar_case_library_items,
)
from benchmarking import (
    clone_candidate,
    evaluate_candidate,
    generate_candidate,
    get_candidate_detail,
    list_candidates,
    list_cases,
)
from memory.experience_service import (
    clear_experience_center,
    get_experience_center_stats,
    get_experience_record,
    list_experience_summaries,
    rebuild_experience_center_index,
    retrieve_experience_guidance,
)
from services.data_service import load_pid_dataset
from services.pid_tuning_service import _build_model_params_for_evaluation
from services.system_config_service import (
    get_runtime_system_config,
    update_runtime_system_config,
)
from skills.data_analysis_skills import _read_csv_with_fallback, detect_pid_loops, fetch_history_data_csv
from skills.rating import ModelRating
from state.workflow_task_store import WorkflowTaskStore

RunCollaborationFn = Callable[..., AsyncGenerator[Dict[str, Any], None]]

LOOP_TYPE_ALIASES = {
    "flow": "flow",
    "流量": "flow",
    "temperature": "temperature",
    "温度": "temperature",
    "pressure": "pressure",
    "压力": "pressure",
    "level": "level",
    "液位": "level",
}


def _map_workflow_error(exc: Exception | str) -> Dict[str, Any]:
    error_text = str(exc or "").strip()
    lowered = error_text.lower()

    if "arrearage" in lowered or "overdue-payment" in lowered or "access denied" in lowered:
        return {
            "error_code": 10021,
            "error_type": "upstream_model_arrearage",
            "message": "上游模型服务不可用，请检查模型账户余额或服务状态。",
            "detail": error_text,
        }
    if "apiconnectionerror" in lowered or "connection error" in lowered:
        return {
            "error_code": 10022,
            "error_type": "upstream_model_connection",
            "message": "上游模型网关连接异常，请稍后重试或检查模型服务网络状态。",
            "detail": error_text,
        }
    if "readerror" in lowered or "read error" in lowered:
        return {
            "error_code": 10023,
            "error_type": "upstream_model_read_error",
            "message": "上游模型服务读取异常，请稍后重试。",
            "detail": error_text,
        }
    if "timeout" in lowered or "timed out" in lowered:
        return {
            "error_code": 10024,
            "error_type": "upstream_model_timeout",
            "message": "上游模型服务响应超时，请稍后重试。",
            "detail": error_text,
        }
    return {
        "error_code": 10020,
        "error_type": "workflow_execution_error",
        "message": "智能整定任务执行失败，请检查后端日志或稍后重试。",
        "detail": error_text,
    }


class WorkflowRunRequest(BaseModel):
    start_time: str = Field(..., description="开始时间")
    end_time: str = Field(..., description="结束时间")
    loop_type: str = Field(..., description="回路类型")
    loop_uri: str = Field(..., description="回路URI")
    window: int = Field(1, description="历史数据时间戳间隔（秒）")
    plant_type: str = Field("distillation_column", description="装置类型")
    scenario: str = Field("", description="工况")
    control_object: str = Field("", description="控制对象")
    response_mode: str = Field("async", description="响应模式：async、blocking 或 streaming")


class StrategyLabGenerateRequest(BaseModel):
    candidate_id: str = Field("", description="Candidate ID")
    profile_id: str = Field("default", description="Profile ID")
    plugin_ids: list[str] = Field(default_factory=list, description="Plugin IDs")
    objective: str = Field("", description="Objective")
    case_id: str = Field("distillation_bidirectional", description="Case ID")
    notes: str = Field("", description="Notes")


class ModelConfigPayload(BaseModel):
    name: str = Field(..., description="模型名称")
    api_url: str = Field(..., description="模型服务地址")
    api_key: str = Field(..., description="模型 API Key")


class IntegrationConfigPayload(BaseModel):
    history_data_api_url: str = Field(..., description="历史数据服务地址")
    knowledge_graph_api_url: str = Field(..., description="本体知识图谱服务地址")


class SystemConfigPayload(BaseModel):
    model: ModelConfigPayload
    integration: IntegrationConfigPayload


class ModelConnectivityTestPayload(BaseModel):
    name: str = Field("", description="模型名称")
    api_url: str = Field("", description="模型服务地址")
    api_key: str = Field("", description="模型 API Key")


class PidChartDataRequest(BaseModel):
    loop_uri: str = Field(..., description="回路 URI")
    start_time: str = Field(..., description="开始时间")
    end_time: str = Field(..., description="结束时间")
    window: int = Field(1, description="历史数据时间戳间隔（秒）")


class PidPredictionPointPayload(BaseModel):
    sv: float = Field(..., description="SV")
    pv: float = Field(..., description="PV")
    mv: float = Field(..., description="MV")
    index: int | None = Field(None, description="采样索引")


class PidPredictionPidPayload(BaseModel):
    Kp: float = Field(..., description="Kp")
    Ki: float = Field(..., description="Ki")
    Kd: float = Field(..., description="Kd")


class PidPredictionCurveRequest(BaseModel):
    points: list[PidPredictionPointPayload] = Field(..., description="回路分析图点序列")
    dt: float = Field(1.0, description="点间隔（秒）")
    loop_type: str = Field("flow", description="回路类型")
    pid_params: PidPredictionPidPayload
    model_type: str = Field("FOPDT", description="模型类型")
    selected_model_params: Dict[str, Any] = Field(default_factory=dict, description="模型参数")
    K: float = Field(0.0, description="过程增益K")
    T: float = Field(0.0, description="时间常数T")
    L: float = Field(0.0, description="纯滞后L")


def _normalize_loop_type(loop_type: str) -> str:
    normalized = (loop_type or "").strip()
    return LOOP_TYPE_ALIASES.get(normalized, normalized or "flow")


def _coerce_model_params(value: str) -> Dict[str, Any]:
    text = (value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _event_progress(event: Dict[str, Any]) -> tuple[str, str, int] | None:
    event_type = str(event.get("type") or "")
    if event_type == "accepted":
        return ("accepted", "已受理", 5)
    if event_type == "user":
        return ("accepted", "任务启动", 10)
    if event_type == "agent_turn":
        agent = str(event.get("agent") or "")
        mapping = {
            "数据分析智能体": ("data_analysis", "数据分析", 20),
            "系统辨识智能体": ("model_identification", "系统辨识", 40),
            "本体知识智能体": ("knowledge_graph", "知识检索", 60),
            "PID专家智能体": ("pid_tuning", "PID整定", 80),
            "评估智能体": ("evaluation", "整定评估", 95),
        }
        return mapping.get(agent)
    if event_type == "result":
        return ("completed", "结果生成", 95)
    if event_type == "done":
        return ("completed", "任务完成", 100)
    if event_type == "error":
        return ("failed", "任务失败", 100)
    return None


def _build_external_result(
    *,
    task_id: str,
    loop_type: str,
    loop_uri: str,
    final_result: Dict[str, Any],
) -> Dict[str, Any]:
    model = final_result.get("model") or {}
    pid = final_result.get("pidParams") or {}
    evaluation = final_result.get("evaluation") or {}
    memory = final_result.get("memory") or {}
    guidance = memory.get("experienceGuidance") or {}
    tuning_advice = final_result.get("tuningAdvice") or {}

    return {
        "task_id": task_id,
        "loop_type": loop_type,
        "loop_uri": loop_uri,
        "model": {
            "model_type": model.get("modelType", "FOPDT"),
            "selected_model_params": model.get("selectedModelParams", {}),
            "confidence": model.get("confidence", 0.0),
            "normalized_rmse": model.get("normalizedRmse", 0.0),
            "r2_score": model.get("r2Score", 0.0),
        },
        "pid": {
            "strategy_used": pid.get("strategyUsed") or pid.get("strategy") or "",
            "Kp": pid.get("Kp", 0.0),
            "Ki": pid.get("Ki", 0.0),
            "Kd": pid.get("Kd", 0.0),
        },
        "evaluation": {
            "performance_score": evaluation.get("performance_score", 0.0),
            "method_confidence": evaluation.get("method_confidence", 0.0),
            "final_rating": evaluation.get("final_rating", 0.0),
            "passed": evaluation.get("passed", False),
            "failure_reason": evaluation.get("failure_reason", ""),
            "feedback_target": evaluation.get("feedback_target", ""),
            "feedback_action": evaluation.get("feedback_action", ""),
        },
        "experience": {
            "experience_id": memory.get("experienceId"),
            "match_count": guidance.get("match_count", 0),
            "preferred_strategy": guidance.get("preferred_strategy", ""),
            "preferred_model_type": guidance.get("preferred_model_type", ""),
        },
        "tuning_advice": {
            "summary": tuning_advice.get("summary", ""),
            "recommendation_level": tuning_advice.get("recommendation_level", ""),
            "actions": tuning_advice.get("actions", []),
            "risks": tuning_advice.get("risks", []),
            "rollback_advice": tuning_advice.get("rollback_advice", ""),
            "operator_note": tuning_advice.get("operator_note", ""),
        },
    }


def create_app(
    *,
    run_multi_agent_collaboration: RunCollaborationFn,
    llm_config: Dict[str, Any],
    default_loop_uri: str,
    default_history_start_time: str,
    default_history_end_time: str,
) -> FastAPI:
    app = FastAPI()
    task_store = WorkflowTaskStore()
    background_jobs: set[asyncio.Task[Any]] = set()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _workflow_event_generator(
        *,
        csv_path: str,
        loop_name: str,
        loop_type: str,
        plant_type: str,
        scenario: str,
        control_object: str,
        loop_uri: str,
        start_time: str,
        end_time: str,
        data_type: str,
        window: int,
        selected_loop_prefix: str | None = None,
        selected_window_index: int | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in run_multi_agent_collaboration(
            csv_path=csv_path,
            loop_name=loop_name,
            loop_type=loop_type,
            plant_type=plant_type,
            scenario=scenario,
            control_object=control_object,
            loop_uri=loop_uri,
            start_time=start_time,
            end_time=end_time,
            data_type=data_type,
            window=window,
            selected_loop_prefix=selected_loop_prefix,
            selected_window_index=selected_window_index,
            llm_config=llm_config,
        ):
            yield event

    async def _run_background_workflow(
        *,
        task_id: str,
        loop_name: str,
        loop_type: str,
        plant_type: str,
        scenario: str,
        control_object: str,
        loop_uri: str,
        start_time: str,
        end_time: str,
        window: int,
    ) -> None:
        task_store.start_task(task_id)
        final_result: Dict[str, Any] | None = None
        try:
            async for event in _workflow_event_generator(
                csv_path="",
                loop_name=loop_name,
                loop_type=loop_type,
                plant_type=plant_type,
                scenario=scenario,
                control_object=control_object,
                loop_uri=loop_uri,
                start_time=start_time,
                end_time=end_time,
                data_type="interpolated",
                window=window,
            ):
                progress = _event_progress(event)
                if progress is not None:
                    stage, stage_display, percent = progress
                    task_store.update_progress(
                        task_id,
                        stage=stage,
                        stage_display=stage_display,
                        percent=percent,
                    )
                if event.get("type") == "result":
                    final_result = event.get("data") or {}

            if final_result is None:
                task_store.fail_task(task_id, "workflow finished without result")
                return

            task_store.complete_task(
                task_id,
                _build_external_result(
                    task_id=task_id,
                    loop_type=loop_type,
                    loop_uri=loop_uri,
                    final_result=final_result,
                ),
            )
        except Exception as exc:
            mapped_error = _map_workflow_error(exc)
            task_store.fail_task(
                task_id,
                mapped_error["message"],
                error_code=mapped_error["error_code"],
                error_type=mapped_error["error_type"],
            )

    @app.post("/api/tune_stream")
    async def tune_stream(
        file: UploadFile = File(None),
        loop_name: str = Form(...),
        loop_type: str = Form("flow"),
        plant_type: str = Form("distillation_column"),
        scenario: str = Form(""),
        control_object: str = Form(""),
        loop_uri: str = Form(default_loop_uri),
        start_time: str = Form(default_history_start_time),
        end_time: str = Form(default_history_end_time),
        data_type: str = Form("interpolated"),
        window: int = Form(1),
        selected_loop_prefix: str | None = Form(default=None),
        selected_window_index: int | None = Form(default=None),
    ) -> StreamingResponse:
        csv_path = ""
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                csv_path = tmp_file.name

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in _workflow_event_generator(
                    csv_path=csv_path,
                    loop_name=loop_name,
                    loop_type=loop_type,
                    plant_type=plant_type,
                    scenario=scenario,
                    control_object=control_object,
                    loop_uri=loop_uri,
                    start_time=start_time,
                    end_time=end_time,
                    data_type=data_type,
                    window=window,
                    selected_loop_prefix=selected_loop_prefix,
                    selected_window_index=selected_window_index,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                import traceback

                mapped_error = _map_workflow_error(exc)
                error_msg = {
                    "type": "error",
                    "message": mapped_error["message"],
                    "error_code": mapped_error["error_code"],
                    "error_type": mapped_error["error_type"],
                    "error_detail": mapped_error["detail"],
                    "traceback": traceback.format_exc(),
                }
                yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            finally:
                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/tuning/csv/inspect-loops")
    async def inspect_csv_loops(file: UploadFile = File(...)) -> Any:
        csv_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                csv_path = tmp_file.name

            raw_df = _read_csv_with_fallback(csv_path)
            loops = detect_pid_loops(raw_df)
            options = [
                {
                    "prefix": loop.get("prefix", ""),
                    "has_sv": bool(loop.get("sv_col")),
                }
                for loop in loops
                if loop.get("prefix")
            ]
            recommended_prefix = options[0]["prefix"] if options else None
            return JSONResponse(
                {
                    "code": 0,
                    "message": "ok",
                    "data": {
                        "loops": options,
                        "recommended_prefix": recommended_prefix,
                        "available_columns": [str(col) for col in raw_df.columns.tolist()],
                    },
                }
            )
        except Exception as exc:
            mapped_error = _map_workflow_error(exc)
            return JSONResponse(
                {
                    "code": 1,
                    "message": mapped_error["message"],
                    "error_code": mapped_error["error_code"],
                    "error_type": mapped_error["error_type"],
                    "detail": mapped_error["detail"],
                },
                status_code=400,
            )
        finally:
            if csv_path and os.path.exists(csv_path):
                os.remove(csv_path)

    @app.post("/api/tuning/csv/inspect-windows")
    async def inspect_csv_windows(
        file: UploadFile = File(...),
        selected_loop_prefix: str | None = Form(default=None),
    ) -> Any:
        csv_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                csv_path = tmp_file.name

            dataset = load_pid_dataset(csv_path, selected_loop_prefix=selected_loop_prefix)
            cleaned_df = dataset.get("cleaned_df")
            candidate_windows = dataset.get("candidate_windows") or []
            timestamps = cleaned_df["timestamp"] if cleaned_df is not None and "timestamp" in cleaned_df.columns else None

            ranked: list[dict[str, Any]] = []
            for idx, event in enumerate(candidate_windows):
                amplitude = float(event.get("amplitude", 0.0) or 0.0)
                score = abs(amplitude)
                ranked.append({"index": int(idx), "score": score, "event": event})

            ranked.sort(key=lambda item: item["score"], reverse=True)
            top = ranked[:2]
            recommended_index = int(ranked[0]["index"]) if ranked else None

            display = []
            for item in top:
                event = item["event"] or {}
                start_idx = int(event.get("window_start_idx", event.get("start_idx", 0)) or 0)
                end_idx = int(event.get("window_end_idx", event.get("end_idx", start_idx)) or start_idx)
                event_start_idx = int(event.get("start_idx", start_idx) or start_idx)
                event_end_idx = int(event.get("end_idx", event_start_idx) or event_start_idx)
                payload: dict[str, Any] = {
                    "index": int(item["index"]),
                    "event_type": str(event.get("type", "")),
                    "amplitude": float(event.get("amplitude", 0.0) or 0.0),
                    "window_start_idx": start_idx,
                    "window_end_idx": end_idx,
                    "event_start_idx": event_start_idx,
                    "event_end_idx": event_end_idx,
                }
                if timestamps is not None and len(timestamps) > 0:
                    last = len(timestamps) - 1
                    start_idx = max(0, min(start_idx, last))
                    end_idx = max(start_idx, min(end_idx, last))
                    event_start_idx = max(0, min(event_start_idx, last))
                    event_end_idx = max(event_start_idx, min(event_end_idx, last))
                    payload.update(
                        {
                            "window_start_time": timestamps.iloc[start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "window_end_time": timestamps.iloc[end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "event_start_time": timestamps.iloc[event_start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "event_end_time": timestamps.iloc[event_end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                display.append(payload)

            return JSONResponse(
                {
                    "code": 0,
                    "message": "ok",
                    "data": {
                        "candidate_windows": display,
                        "recommended_index": recommended_index,
                        "available_columns": dataset.get("available_columns") or [],
                        "sampling_time": dataset.get("sampling_time"),
                        "data_points": dataset.get("data_points"),
                    },
                }
            )
        except Exception as exc:
            mapped_error = _map_workflow_error(exc)
            return JSONResponse(
                {
                    "code": 1,
                    "message": mapped_error["message"],
                    "error_code": mapped_error["error_code"],
                    "error_type": mapped_error["error_type"],
                    "detail": mapped_error["detail"],
                },
                status_code=400,
            )
        finally:
            if csv_path and os.path.exists(csv_path):
                os.remove(csv_path)

    @app.post("/api/agent/workflow/run", response_model=None)
    async def workflow_run(
        payload: WorkflowRunRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> Any:
        del authorization
        task_id = str(uuid.uuid4())
        normalized_loop_type = _normalize_loop_type(payload.loop_type)
        loop_name = payload.loop_uri.rstrip("/").split("/")[-1] or "external_workflow"
        response_mode = (payload.response_mode or "async").strip().lower()

        task_store.create_task(
            task_id,
            {
                "start_time": payload.start_time,
                "end_time": payload.end_time,
                "window": payload.window,
                "loop_type": normalized_loop_type,
                "loop_uri": payload.loop_uri,
                "response_mode": response_mode,
            },
        )

        if response_mode == "streaming":
            async def stream_events() -> AsyncGenerator[str, None]:
                accepted = {"type": "accepted", "task_id": task_id}
                yield f"data: {json.dumps(accepted, ensure_ascii=False)}\n\n"
                try:
                    async for event in _workflow_event_generator(
                        csv_path="",
                        loop_name=loop_name,
                        loop_type=normalized_loop_type,
                        plant_type=payload.plant_type,
                        scenario=payload.scenario,
                        control_object=payload.control_object,
                        loop_uri=payload.loop_uri,
                        start_time=payload.start_time,
                        end_time=payload.end_time,
                        data_type="interpolated",
                        window=payload.window,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    import traceback

                    mapped_error = _map_workflow_error(exc)
                    error_msg = {
                        "type": "error",
                        "task_id": task_id,
                        "message": mapped_error["message"],
                        "error_code": mapped_error["error_code"],
                        "error_type": mapped_error["error_type"],
                        "error_detail": mapped_error["detail"],
                        "traceback": traceback.format_exc(),
                    }
                    yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        if response_mode == "async":
            job = asyncio.create_task(
                _run_background_workflow(
                    task_id=task_id,
                    loop_name=loop_name,
                    loop_type=normalized_loop_type,
                    plant_type=payload.plant_type,
                    scenario=payload.scenario,
                    control_object=payload.control_object,
                    loop_uri=payload.loop_uri,
                    start_time=payload.start_time,
                    end_time=payload.end_time,
                    window=payload.window,
                )
            )
            background_jobs.add(job)
            job.add_done_callback(background_jobs.discard)
            return JSONResponse(
                {
                    "code": 0,
                    "message": "accepted",
                    "task_id": task_id,
                }
            )

        final_result: Dict[str, Any] | None = None
        try:
            task_store.start_task(task_id)
            async for event in _workflow_event_generator(
                csv_path="",
                loop_name=loop_name,
                loop_type=normalized_loop_type,
                plant_type=payload.plant_type,
                scenario=payload.scenario,
                control_object=payload.control_object,
                loop_uri=payload.loop_uri,
                start_time=payload.start_time,
                end_time=payload.end_time,
                data_type="interpolated",
                window=payload.window,
            ):
                progress = _event_progress(event)
                if progress is not None:
                    stage, stage_display, percent = progress
                    task_store.update_progress(
                        task_id,
                        stage=stage,
                        stage_display=stage_display,
                        percent=percent,
                    )
                if event.get("type") == "result":
                    final_result = event.get("data")
        except Exception as exc:
            mapped_error = _map_workflow_error(exc)
            task_store.fail_task(
                task_id,
                mapped_error["message"],
                error_code=mapped_error["error_code"],
                error_type=mapped_error["error_type"],
            )
            return JSONResponse(
                {
                    "code": mapped_error["error_code"],
                    "message": mapped_error["message"],
                    "task_id": task_id,
                    "error_type": mapped_error["error_type"],
                    "error_message": mapped_error["message"],
                    "error_detail": mapped_error["detail"],
                },
                status_code=500,
            )

        external_result = _build_external_result(
            task_id=task_id,
            loop_type=normalized_loop_type,
            loop_uri=payload.loop_uri,
            final_result=final_result or {},
        )
        task_store.complete_task(task_id, external_result)
        return JSONResponse(
            {
                "code": 0,
                "message": "success",
                "task_id": task_id,
                "result": external_result,
            }
        )

    @app.get("/api/agent/workflow/status/{task_id}")
    async def workflow_status(task_id: str) -> Any:
        task = task_store.get_task(task_id)
        if task is None:
            return JSONResponse(
                {
                    "code": 1,
                    "message": "task not found",
                    "task_id": task_id,
                },
                status_code=404,
            )
        return JSONResponse(
            {
                "code": 0,
                "message": "success",
                "task_id": task_id,
                "status": task.get("status"),
                "created_at": task.get("created_at"),
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
                "progress": task.get("progress"),
                "error_message": task.get("error_message"),
                "error_code": task.get("error_code"),
                "error_type": task.get("error_type"),
            }
        )

    @app.get("/api/agent/workflow/result/{task_id}")
    async def workflow_result(task_id: str) -> Any:
        task = task_store.get_task(task_id)
        if task is None:
            return JSONResponse(
                {
                    "code": 1,
                    "message": "task not found",
                    "task_id": task_id,
                },
                status_code=404,
            )

        status = task.get("status")
        if status in {"pending", "running"}:
            return JSONResponse(
                {
                    "code": 0,
                    "message": "task not finished",
                    "task_id": task_id,
                    "status": status,
                    "finished_at": task.get("finished_at"),
                    "result": None,
                    "error_message": task.get("error_message"),
                    "error_code": task.get("error_code"),
                    "error_type": task.get("error_type"),
                }
            )

        if status == "failed":
            return JSONResponse(
                {
                    "code": task.get("error_code") or 10020,
                    "message": task.get("error_message") or "智能整定任务执行失败，请稍后重试。",
                    "task_id": task_id,
                    "status": status,
                    "finished_at": task.get("finished_at"),
                    "result": None,
                    "error_message": task.get("error_message"),
                    "error_code": task.get("error_code"),
                    "error_type": task.get("error_type"),
                },
                status_code=500,
            )

        return JSONResponse(
            {
                "code": 0,
                "message": "success",
                "task_id": task_id,
                "status": status,
                "finished_at": task.get("finished_at"),
                "result": task.get("result"),
                "error_message": task.get("error_message"),
                "error_code": task.get("error_code"),
                "error_type": task.get("error_type"),
            }
        )

    @app.get("/api/experiences/stats")
    async def experience_stats() -> JSONResponse:
        return JSONResponse(get_experience_center_stats())

    @app.get("/api/experiences")
    async def experience_list(
        loop_type: str = "",
        model_type: str = "",
        passed: str = "",
        strategy: str = "",
        keyword: str = "",
        limit: int = 50,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "items": list_experience_summaries(
                    loop_type=loop_type,
                    model_type=model_type,
                    passed=passed,
                    strategy=strategy,
                    keyword=keyword,
                    limit=limit,
                )
            }
        )

    @app.get("/api/experiences/{experience_id}")
    async def experience_detail(experience_id: str) -> JSONResponse:
        record = get_experience_record(experience_id)
        return JSONResponse({"item": record})

    @app.post("/api/experiences/search")
    async def experience_search(
        loop_type: str = Form("flow"),
        model_type: str = Form("FOPDT"),
        K: float = Form(0.0),
        T: float = Form(0.0),
        L: float = Form(0.0),
        selected_model_params: str = Form(""),
        limit: int = Form(3),
    ) -> JSONResponse:
        model_params = _coerce_model_params(selected_model_params)
        normalized_model_type = (model_type or "FOPDT").strip().upper()
        if model_params:
            K = float(model_params.get("K", K))
            if normalized_model_type == "SOPDT":
                T = float(model_params.get("T1", 0.0)) + float(model_params.get("T2", 0.0))
                L = float(model_params.get("L", L))
            elif normalized_model_type == "IPDT":
                L = float(model_params.get("L", L))
                T = max(float(T), L)
            elif normalized_model_type == "FO":
                T = float(model_params.get("T", T))
            else:
                T = float(model_params.get("T", T))
                L = float(model_params.get("L", L))
        return JSONResponse(
            retrieve_experience_guidance(
                loop_type=loop_type,
                model_type=model_type,
                K=K,
                T=T,
                L=L,
                selected_model_params=model_params,
                limit=limit,
                candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
            )
        )

    @app.post("/api/experiences/actions/clear")
    async def experience_clear() -> JSONResponse:
        return JSONResponse(clear_experience_center())

    @app.post("/api/experiences/actions/reindex")
    async def experience_reindex() -> JSONResponse:
        return JSONResponse(rebuild_experience_center_index())

    @app.get("/api/case-library/stats")
    async def case_library_stats() -> JSONResponse:
        return JSONResponse(get_case_library_stats())

    @app.get("/api/case-library")
    async def case_library_list(
        provider: str = "",
        loop_type: str = "",
        model_type: str = "",
        track: str = "",
        failure_mode: str = "",
        keyword: str = "",
        limit: int = 100,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "items": list_case_library_items(
                    provider=provider,
                    loop_type=loop_type,
                    model_type=model_type,
                    track=track,
                    failure_mode=failure_mode,
                    keyword=keyword,
                    limit=limit,
                )
            }
        )

    @app.get("/api/case-library/{case_id}")
    async def case_library_detail(case_id: str) -> JSONResponse:
        return JSONResponse({"item": get_case_library_detail(case_id)})

    @app.get("/api/case-library/{case_id}/similar")
    async def case_library_similar(case_id: str, limit: int = 5) -> JSONResponse:
        return JSONResponse({"items": list_similar_case_library_items(case_id, limit=limit)})

    @app.get("/api/task-sessions")
    async def get_task_sessions() -> JSONResponse:
        from state.frontend_sessions import get_frontend_sessions
        return JSONResponse(get_frontend_sessions())

    @app.post("/api/task-sessions")
    async def save_task_sessions(request: Request) -> JSONResponse:
        from state.frontend_sessions import save_frontend_sessions
        payload = await request.json()
        save_frontend_sessions(payload)
        return JSONResponse({"status": "ok"})

    @app.get("/api/system-config")
    async def system_config_get() -> JSONResponse:
        return JSONResponse(get_runtime_system_config())

    @app.put("/api/system-config")
    async def system_config_update(payload: SystemConfigPayload) -> JSONResponse:
        config = update_runtime_system_config(payload.model_dump())
        llm_config.clear()
        llm_config.update(
            {
                "api_key": config["model"]["api_key"],
                "base_url": config["model"]["api_url"],
                "model": config["model"]["name"],
            }
        )
        return JSONResponse({"message": "system_config_updated", "config": config})

    @app.post("/api/system-config/test-model")
    async def system_config_test_model(payload: ModelConnectivityTestPayload) -> JSONResponse:
        runtime = get_runtime_system_config()
        model_name = str(payload.name or runtime["model"]["name"] or "").strip()
        api_url = str(payload.api_url or runtime["model"]["api_url"] or "").strip().rstrip("/")
        api_key = str(payload.api_key or runtime["model"]["api_key"] or "").strip()

        if not model_name or not api_url:
            return JSONResponse(
                {
                    "ok": False,
                    "error_type": "invalid_config",
                    "message": "模型名称或模型服务地址为空，请先完善系统配置。",
                },
                status_code=400,
            )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": "请仅回复：ok"}],
            "max_tokens": 8,
            "temperature": 0,
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(f"{api_url}/chat/completions", headers=headers, json=body)
            if response.status_code >= 400:
                detail = response.text[:800]
                error_type = "upstream_model_bad_gateway" if response.status_code == 502 else (
                    "upstream_model_unavailable" if response.status_code == 503 else "upstream_model_http_error"
                )
                return JSONResponse(
                    {
                        "ok": False,
                        "error_type": error_type,
                        "status_code": response.status_code,
                        "message": f"模型服务测试失败（HTTP {response.status_code}）。",
                        "detail": detail,
                        "model": model_name,
                        "api_url": api_url,
                    }
                )
            payload_json = response.json()
            content = ""
            try:
                content = str(payload_json["choices"][0]["message"]["content"])
            except Exception:
                content = ""
            return JSONResponse(
                {
                    "ok": True,
                    "message": "模型服务连通性正常，可以发起整定任务。",
                    "model": model_name,
                    "api_url": api_url,
                    "reply": content.strip(),
                }
            )
        except httpx.TimeoutException as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "error_type": "upstream_model_timeout",
                    "message": "模型服务响应超时，请稍后重试或切换可用模型。",
                    "detail": str(exc),
                    "model": model_name,
                    "api_url": api_url,
                }
            )
        except Exception as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "error_type": "upstream_model_connection",
                    "message": "模型服务连接异常，请检查服务地址、网关状态或网络连通性。",
                    "detail": str(exc),
                    "model": model_name,
                    "api_url": api_url,
                }
            )

    @app.post("/api/tuning/pid-chart-data")
    async def tuning_pid_chart_data(payload: PidChartDataRequest) -> JSONResponse:
        csv_path = ""
        try:
            csv_meta = fetch_history_data_csv(
                loop_uri=payload.loop_uri,
                start_time=payload.start_time,
                end_time=payload.end_time,
                window=payload.window,
            )
            csv_path = str(csv_meta.get("csv_path") or "")
            dataset = load_pid_dataset(csv_path)
            overview = dataset.get("window_overview") or {}
            return JSONResponse(
                {
                    "points": overview.get("points") or [],
                    "x_axis": overview.get("x_axis") or "timestamp",
                    "total_points": int(overview.get("total_points") or dataset.get("data_points") or 0),
                    "sampling_time": float(dataset.get("sampling_time") or payload.window or 1),
                    "window_start": overview.get("window_start"),
                    "window_end": overview.get("window_end"),
                    "start_time": overview.get("start_time"),
                    "end_time": overview.get("end_time"),
                }
            )
        except Exception as exc:
            return JSONResponse(
                {
                    "error": "pid_chart_data_error",
                    "detail": str(exc),
                },
                status_code=400,
            )
        finally:
            if csv_path and os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except Exception:
                    pass

    @app.post("/api/tuning/pid-prediction-curve")
    async def tuning_pid_prediction_curve(payload: PidPredictionCurveRequest) -> JSONResponse:
        try:
            points = payload.points or []
            if not points:
                return JSONResponse({"pv_pred": [], "mv_pred": [], "sp_pred": [], "dt": float(payload.dt or 1.0)})

            dt = float(payload.dt or 1.0)
            dt = max(dt, 1e-6)
            loop_type = _normalize_loop_type(payload.loop_type)

            sp_series = [float(item.sv) for item in points]
            pv_initial = float(points[0].pv)
            mv_initial = float(points[0].mv)
            sp_align_offset = 0.0
            try:
                if sp_series:
                    sp0 = float(sp_series[0])
                    sp_min = float(min(sp_series))
                    sp_max = float(max(sp_series))
                    sp_range = sp_max - sp_min
                    candidate_offset = pv_initial - sp0
                    if abs(candidate_offset) > max(3.0 * max(sp_range, 1e-6), 1.0):
                        sp_series = [float(value) + float(candidate_offset) for value in sp_series]
                        sp_align_offset = float(candidate_offset)
            except Exception:
                sp_align_offset = 0.0

            model_params = _build_model_params_for_evaluation(
                model_type=payload.model_type,
                selected_model_params=payload.selected_model_params,
                K=float(payload.K or 0.0),
                T=float(payload.T or 0.0),
                L=float(payload.L or 0.0),
            )

            sim = ModelRating.simulate_setpoint_trajectory(
                model_params=model_params,
                pid_params={"Kp": float(payload.pid_params.Kp), "Ki": float(payload.pid_params.Ki), "Kd": float(payload.pid_params.Kd)},
                sp_series=sp_series,
                pv_initial=pv_initial,
                mv_initial=mv_initial,
                dt=dt,
                loop_type=loop_type,
            )

            def _json_safe_series(values: Any) -> list[float | None]:
                if not isinstance(values, list):
                    return []
                safe: list[float | None] = []
                for item in values:
                    try:
                        value = float(item)
                    except Exception:
                        safe.append(None)
                        continue
                    safe.append(value if math.isfinite(value) else None)
                return safe

            return JSONResponse(
                {
                    "pv_pred": _json_safe_series(sim.get("pv_history")),
                    "mv_pred": _json_safe_series(sim.get("mv_history")),
                    "sp_pred": _json_safe_series(sim.get("sp_history")),
                    "dt": float(sim.get("dt", dt)),
                    "model_type": sim.get("model_type") or str(model_params.get("model_type") or payload.model_type or ""),
                    "sp_align_offset": sp_align_offset,
                }
            )
        except Exception as exc:
            return JSONResponse(
                {
                    "error": "pid_prediction_curve_error",
                    "detail": str(exc),
                },
                status_code=400,
            )

    @app.get("/api/strategy-lab/cases")
    async def strategy_lab_cases() -> JSONResponse:
        return JSONResponse({"items": list_cases()})

    @app.get("/api/strategy-lab/candidates")
    async def strategy_lab_candidates() -> JSONResponse:
        return JSONResponse({"items": list_candidates()})

    @app.get("/api/strategy-lab/candidates/{candidate_id}")
    async def strategy_lab_candidate_detail(candidate_id: str) -> JSONResponse:
        try:
            return JSONResponse(get_candidate_detail(candidate_id))
        except FileNotFoundError:
            return JSONResponse({"error": "candidate_not_found", "candidate_id": candidate_id}, status_code=404)

    @app.post("/api/strategy-lab/candidates/generate")
    async def strategy_lab_generate(request: StrategyLabGenerateRequest) -> JSONResponse:
        summary = generate_candidate(request.model_dump())
        return JSONResponse({"item": summary})

    @app.post("/api/strategy-lab/candidates/{candidate_id}/evaluate")
    async def strategy_lab_evaluate(candidate_id: str) -> JSONResponse:
        try:
            return JSONResponse(evaluate_candidate(candidate_id))
        except FileNotFoundError:
            return JSONResponse({"error": "candidate_not_found", "candidate_id": candidate_id}, status_code=404)

    @app.post("/api/strategy-lab/candidates/{candidate_id}/clone")
    async def strategy_lab_clone(candidate_id: str) -> JSONResponse:
        try:
            return JSONResponse({"item": clone_candidate(candidate_id)})
        except FileNotFoundError:
            return JSONResponse({"error": "candidate_not_found", "candidate_id": candidate_id}, status_code=404)

    return app
