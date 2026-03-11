from __future__ import annotations

import json
import os
import tempfile
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from memory.experience_service import (
    clear_experience_center,
    get_experience_center_stats,
    get_experience_record,
    list_experience_summaries,
    rebuild_experience_center_index,
    retrieve_experience_guidance,
)

RunCollaborationFn = Callable[..., AsyncGenerator[Dict[str, Any], None]]


def create_app(
    *,
    run_multi_agent_collaboration: RunCollaborationFn,
    llm_config: Dict[str, Any],
    default_loop_uri: str,
    default_history_start_time: str,
    default_history_end_time: str,
) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/tune_stream")
    async def tune_stream(
        file: UploadFile = File(None),
        loop_name: str = Form(...),
        loop_type: str = Form("flow"),
        loop_uri: str = Form(default_loop_uri),
        start_time: str = Form(default_history_start_time),
        end_time: str = Form(default_history_end_time),
        data_type: str = Form("interpolated"),
    ) -> StreamingResponse:
        csv_path = ""
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                csv_path = tmp_file.name

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in run_multi_agent_collaboration(
                    csv_path=csv_path,
                    loop_name=loop_name,
                    loop_type=loop_type,
                    loop_uri=loop_uri,
                    start_time=start_time,
                    end_time=end_time,
                    data_type=data_type,
                    llm_config=llm_config,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                import traceback

                error_msg = {"type": "error", "message": f"{exc}\n{traceback.format_exc()}"}
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
        K: float = Form(...),
        T: float = Form(...),
        L: float = Form(...),
        limit: int = Form(3),
    ) -> JSONResponse:
        return JSONResponse(
            retrieve_experience_guidance(
                loop_type=loop_type,
                model_type=model_type,
                K=K,
                T=T,
                L=L,
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

    return app
