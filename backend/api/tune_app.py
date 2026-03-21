from __future__ import annotations

import ast
import asyncio
import json
import os
import tempfile
import uuid
from typing import Any, AsyncGenerator, Callable, Dict

from fastapi import FastAPI, File, Form, Header, UploadFile
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
from services.system_config_service import (
    get_runtime_system_config,
    update_runtime_system_config,
)
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
