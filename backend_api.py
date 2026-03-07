from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import sys
import traceback
import json
import asyncio
import os
from dotenv import load_dotenv

sys.path.append("/run/code/dinglei/pid")
load_dotenv()

from skills.system_id_skills import fit_fopdt_model, calculate_model_confidence
from skills.pid_tuning_skills import apply_tuning_rules, controller_logic_translator
from skills.rating import ModelRating

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM配置
LLM_CONFIG = {
    "model": os.getenv("MODEL", "qwen3-max"),
    "api_key": os.getenv("MODEL_API_KEY"),
    "base_url": os.getenv("MODEL_API_URL"),
    "temperature": 0.7,
}

# 检查是否启用LLM
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true" and bool(LLM_CONFIG["api_key"])

if USE_LLM:
    try:
        from agents_multiagent import run_multi_agent_collaboration
        print("✅ LLM多智能体协作模式已启用（AutoGen框架）")
    except Exception as e:
        print(f"⚠️ LLM模块加载失败，使用传统模式: {e}")
        USE_LLM = False
else:
    print("ℹ️ 使用传统算法模式（不调用LLM）")

async def generate_sse_events_traditional(csv_path, loop_name, controller_brand, strategy):
    """传统模式：不使用LLM，仅展示效果"""
    try:
        # 发送思维链：数据加载
        msg = {"type": "thought", "agent": "数据分析智能体", "content": f"正在加载CSV文件: {csv_path}\n检查数据格式和必需列..."}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # Load data
        df = pd.read_csv(csv_path)

        # Check required columns
        required_cols = ['MV', 'PV']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            available_cols = df.columns.tolist()
            col_mapping = {}
            for req_col in required_cols:
                for avail_col in available_cols:
                    if req_col.lower() == avail_col.lower():
                        col_mapping[req_col] = avail_col
                        break
            if col_mapping:
                df = df.rename(columns={v: k for k, v in col_mapping.items()})
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                msg = {"type": "error", "message": f"CSV文件缺少必需的列: {', '.join(missing_cols)}"}
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                return

        data_points = len(df)
        msg = {"type": "thought", "agent": "数据分析智能体", "content": f"数据加载成功！\n数据点数: {data_points}\n列: {list(df.columns)}"}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送工具调用：提取数据
        mv = df["MV"].values
        pv = df["PV"].values
        msg = {"type": "tool_call", "tool_name": "extract_data", "args": {"columns": ["MV", "PV"], "rows": data_points}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        msg = {"type": "tool_result", "result": {"mv_range": [float(mv.min()), float(mv.max())], "pv_range": [float(pv.min()), float(pv.max())]}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送思维链：系统辨识
        msg = {"type": "thought", "agent": "系统辨识智能体", "content": "开始系统辨识...\n使用FOPDT模型拟合\n采样周期: 1.0秒"}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送工具调用：模型拟合
        msg = {"type": "tool_call", "tool_name": "fit_fopdt_model", "args": {"dt": 1.0, "data_points": data_points}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        # System identification
        model_params = fit_fopdt_model(mv, pv, dt=1.0)
        confidence = calculate_model_confidence(model_params["residue"])

        msg = {"type": "tool_result", "result": {"K": float(model_params["K"]), "T": float(model_params["T"]), "L": float(model_params["L"]), "confidence": float(confidence["confidence"])}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送思维链：PID整定
        msg = {"type": "thought", "agent": "PID专家智能体", "content": f"系统模型辨识完成！\nK={model_params['K']:.4f}, T={model_params['T']:.2f}s, L={model_params['L']:.2f}s\n\n开始应用{strategy}整定策略..."}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送工具调用：PID整定
        msg = {"type": "tool_call", "tool_name": "apply_tuning_rules", "args": {"K": float(model_params["K"]), "T": float(model_params["T"]), "L": float(model_params["L"]), "strategy": strategy}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        # PID tuning
        pid_params = apply_tuning_rules(
            model_params["K"],
            model_params["T"],
            model_params["L"],
            strategy
        )

        msg = {"type": "tool_result", "result": {"Kp": float(pid_params["Kp"]), "Ki": float(pid_params["Ki"]), "Kd": float(pid_params["Kd"])}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送思维链：参数转换
        msg = {"type": "thought", "agent": "PID专家智能体", "content": f"PID参数计算完成！\nKp={pid_params['Kp']:.4f}, Ki={pid_params['Ki']:.4f}, Kd={pid_params['Kd']:.4f}\n\n正在转换为{controller_brand}控制器格式..."}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送工具调用：参数转换
        msg = {"type": "tool_call", "tool_name": "controller_logic_translator", "args": {"brand": controller_brand}}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        # Translate parameters
        translated = controller_logic_translator(pid_params, controller_brand)

        msg = {"type": "tool_result", "result": translated}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送思维链：评估开始
        msg = {"type": "thought", "agent": "评估智能体", "content": "开始评估PID整定结果...\n将进行三层评估：\n1. 性能评分 (0-10)\n2. 方法置信度 (0-1)\n3. 综合评级"}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)

        # 发送工具调用：评估
        msg = {"type": "tool_call", "tool_name": "ModelRating.evaluate", "args": {
            "model_params": {"K": float(model_params["K"]), "T": float(model_params["T"]), "L": float(model_params["L"])},
            "pid_params": {"Kp": float(pid_params["Kp"]), "Ki": float(pid_params["Ki"]), "Kd": float(pid_params["Kd"])},
            "method": strategy
        }}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        # 执行评估
        try:
            eval_result = ModelRating.evaluate(
                model_params={"K": model_params["K"], "T": model_params["T"], "L": model_params["L"]},
                pid_params={"Kp": pid_params["Kp"], "Ki": pid_params["Ki"], "Kd": pid_params["Kd"]},
                method=strategy.lower()
            )

            msg = {"type": "tool_result", "result": {
                "performance_score": float(eval_result["performance_score"]),
                "method_confidence": float(eval_result["method_confidence"]),
                "final_rating": float(eval_result["final_rating"])
            }}
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

            final_rating = eval_result["final_rating"]
            threshold = 6.0

            if final_rating >= threshold:
                msg = {"type": "thought", "agent": "评估智能体", "content": f"评估通过！\n综合评级: {final_rating:.2f} >= {threshold}\n性能评分: {eval_result['performance_score']:.2f}\n方法置信度: {eval_result['method_confidence']:.2f}"}
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.5)
            else:
                msg = {"type": "thought", "agent": "评估智能体", "content": f"评估结果不达标！\n综合评级: {final_rating:.2f} < {threshold}\n性能评分: {eval_result['performance_score']:.2f}\n方法置信度: {eval_result['method_confidence']:.2f}"}
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.5)

        except Exception as eval_error:
            msg = {"type": "tool_result", "result": f"评估失败: {str(eval_error)}", "is_error": True}
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)
            eval_result = None

        # 发送最终结果
        result = {
            "dataAnalysis": {
                "dataPoints": data_points,
                "stepEvents": 1,
                "currentIAE": 0.0
            },
            "model": {
                "K": float(model_params["K"]),
                "T": float(model_params["T"]),
                "L": float(model_params["L"]),
                "confidence": float(confidence["confidence"])
            },
            "pidParams": {
                "Kp": float(pid_params["Kp"]),
                "Ki": float(pid_params["Ki"]),
                "Kd": float(pid_params["Kd"])
            },
            "translated": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in translated.items()}
        }

        if eval_result:
            result["evaluation"] = {
                "performance_score": float(eval_result["performance_score"]),
                "method_confidence": float(eval_result["method_confidence"]),
                "final_rating": float(eval_result["final_rating"]),
                "strategy_used": strategy,
                "passed": eval_result["final_rating"] >= 6.0
            }

        msg = {"type": "assistant", "content": f"✅ 整定完成！\n\n为控制回路 {loop_name} 生成的PID参数已准备就绪，可以部署到{controller_brand}控制器。"}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        msg = {"type": "result", "data": result}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        msg = {"type": "error", "message": error_msg}
        yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

@app.post("/api/tune_stream")
async def tune_pid_stream(
    file: UploadFile = File(...),
    loop_name: str = Form(...),
    controller_brand: str = Form(...),
    strategy: str = Form("IMC")
):
    """SSE流式返回整定过程"""
    try:
        # Save uploaded file
        csv_path = f"/tmp/{file.filename}"
        with open(csv_path, "wb") as f:
            f.write(await file.read())

        if USE_LLM:
            # 使用真正的LLM多智能体协作（AutoGen框架）
            async def llm_stream():
                async for msg in run_multi_agent_collaboration(csv_path, loop_name, controller_brand, strategy, LLM_CONFIG):
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                llm_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 使用传统模式
            return StreamingResponse(
                generate_sse_events_traditional(csv_path, loop_name, controller_brand, strategy),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return {
        "llm_enabled": USE_LLM,
        "llm_model": LLM_CONFIG["model"] if USE_LLM else None,
        "mode": "LLM多智能体" if USE_LLM else "传统算法",
        "api_configured": bool(LLM_CONFIG.get("api_key"))
    }

# 静态文件服务
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3443)
