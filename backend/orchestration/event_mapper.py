from __future__ import annotations

from typing import Any, Callable, Dict, List


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_agent_response(
    agent_name: str,
    tools: List[Dict[str, Any]],
    *,
    display_agent_names: Dict[str, str],
) -> str:
    if not tools:
        return ""

    latest_result = None
    latest_tool_name = ""
    for tool in reversed(tools):
        if tool.get("result") is not None:
            latest_result = tool["result"]
            latest_tool_name = tool.get("tool_name", "")
            break

    if not isinstance(latest_result, dict):
        return ""

    if agent_name == display_agent_names["data_analyst"]:
        points = latest_result.get("data_points", 0)
        window_points = latest_result.get("window_points", 0)
        sampling_time = latest_result.get("sampling_time", 1.0)
        step_events = latest_result.get("step_events", [])
        return (
            f"已完成数据加载与预处理，共获得 {points} 个数据点，"
            f"当前用于辨识的窗口为 {window_points} 点，采样周期约 {_format_float(sampling_time, 2)} s，"
            f"检测到 {len(step_events) if isinstance(step_events, list) else 0} 个候选阶跃事件。"
        )

    if agent_name == display_agent_names["system_id_expert"]:
        extra_parts: List[str] = []
        reason_codes = latest_result.get("reason_codes", [])
        if isinstance(reason_codes, list) and reason_codes:
            extra_parts.append(f"风险提示：{', '.join(reason_codes)}。")
        next_actions = latest_result.get("next_actions", [])
        if isinstance(next_actions, list) and next_actions:
            extra_parts.append(f"建议动作：{', '.join(next_actions)}。")
        extra = f" {' '.join(extra_parts)}" if extra_parts else ""
        return (
            f"FOPDT 模型辨识完成，得到 K={_format_float(latest_result.get('K'))}，"
            f"T={_format_float(latest_result.get('T'))}，L={_format_float(latest_result.get('L'))}，"
            f"标准化 RMSE {_format_float(latest_result.get('normalized_rmse', latest_result.get('residue')))}，"
            f"R² {_format_float(latest_result.get('r2_score'), 3)}，模型置信度 {_format_float(latest_result.get('confidence'), 2)}。"
            f"{extra}"
        )

    if agent_name == display_agent_names["pid_expert"]:
        tune_result = None
        for tool in tools:
            if tool.get("tool_name") == "tool_tune_pid" and isinstance(tool.get("result"), dict):
                tune_result = tool["result"]
        if tune_result is None:
            tune_result = latest_result if latest_tool_name == "tool_tune_pid" else {}

        response = (
            f"已按 {tune_result.get('strategy_used', tune_result.get('strategy', '当前策略'))} 策略完成整定，"
            f"Kp={_format_float(tune_result.get('Kp'))}，Ki={_format_float(tune_result.get('Ki'))}，"
            f"Kd={_format_float(tune_result.get('Kd'))}。"
        )
        if tune_result.get("selection_reason"):
            response += f" 策略选择依据：{tune_result.get('selection_reason')}"
        return response

    if agent_name == display_agent_names["evaluation_expert"]:
        extra = ""
        if not latest_result.get("passed") and latest_result.get("feedback_target"):
            extra = (
                f" 未通过主因：{latest_result.get('failure_reason', '')}"
                f" 建议下一步回流给 {latest_result.get('feedback_target')}，"
                f"{latest_result.get('feedback_action', '')}"
            )
        return (
            f"闭环评估完成，性能评分 {_format_float(latest_result.get('performance_score'), 2)}，"
            f"方法置信度 {_format_float(latest_result.get('method_confidence'), 2)}，"
            f"最终评分 {_format_float(latest_result.get('final_rating'), 2)}，"
            f"{'通过' if latest_result.get('passed') else '未通过'}当前整定校核。{extra}"
        )

    return ""


def finalize_agent_turn(
    current_turn_data: Dict[str, Any] | None,
    *,
    display_agent_names: Dict[str, str],
) -> Dict[str, Any] | None:
    if current_turn_data is None:
        return None

    existing_response = (current_turn_data.get("response") or "").strip()
    generated = build_agent_response(
        current_turn_data.get("agent", ""),
        current_turn_data.get("tools", []),
        display_agent_names=display_agent_names,
    )

    latest_result = None
    for tool in reversed(current_turn_data.get("tools", [])):
        if isinstance(tool.get("result"), dict):
            latest_result = tool.get("result")
            break

    force_generated = (
        current_turn_data.get("agent") == display_agent_names["evaluation_expert"]
        and isinstance(latest_result, dict)
        and (latest_result.get("feedback_target") or latest_result.get("passed") is False)
    )
    if force_generated:
        current_turn_data["response"] = generated or existing_response
        return current_turn_data

    if not existing_response or existing_response in {"完成", "APPROVE"}:
        current_turn_data["response"] = generated or existing_response
    elif generated and len(existing_response) < 12:
        current_turn_data["response"] = f"{existing_response}\n{generated}"

    return current_turn_data


def build_feedback_turns(
    shared_data: Dict[str, Any],
    *,
    display_agent_names: Dict[str, str],
    to_jsonable: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    initial_assessment = shared_data.get("initial_assessment") or {}
    if not initial_assessment or initial_assessment.get("passed", True):
        return []

    turns: List[Dict[str, Any]] = []

    auto_refine_result = shared_data.get("auto_refine_result") or {}
    if auto_refine_result.get("applied"):
        selection_inputs = shared_data.get("selection_inputs") or {}
        strategy_used = str(shared_data.get("strategy_used", ""))
        turns.append(
            {
                "type": "agent_turn",
                "agent": display_agent_names["pid_expert"],
                "tools": [
                    {
                        "tool_name": "tool_tune_pid",
                        "args": {
                            "K": selection_inputs.get("K", shared_data.get("K", 0.0)),
                            "T": selection_inputs.get("T", shared_data.get("T", 0.0)),
                            "L": selection_inputs.get("L", shared_data.get("L", 0.0)),
                            "loop_type": selection_inputs.get("loop_type", shared_data.get("loop_type", "flow")),
                            "phase": "feedback_refine",
                            "base_strategy": strategy_used,
                        },
                        "result": to_jsonable(
                            {
                                "Kp": auto_refine_result.get("Kp", 0.0),
                                "Ki": auto_refine_result.get("Ki", 0.0),
                                "Kd": auto_refine_result.get("Kd", 0.0),
                                "strategy_used": strategy_used,
                                "performance_score": auto_refine_result.get("refined_performance_score", 0.0),
                                "final_rating": auto_refine_result.get("refined_final_rating", 0.0),
                                "base_final_rating": auto_refine_result.get("base_final_rating", 0.0),
                            }
                        ),
                    }
                ],
                "response": (
                    "根据首次评估反馈，已自动回流 PID 专家继续细调参数，"
                    f"将 final_rating 从 {float(auto_refine_result.get('base_final_rating', 0.0)):.2f} "
                    f"提升到 {float(auto_refine_result.get('refined_final_rating', 0.0)):.2f}。"
                ),
            }
        )

    model_retry_result = shared_data.get("model_retry_result") or {}
    if model_retry_result.get("applied"):
        turns.append(
            {
                "type": "agent_turn",
                "agent": display_agent_names["system_id_expert"],
                "tools": [
                    {
                        "tool_name": "tool_fit_fopdt",
                        "args": {"phase": "feedback_retry_window"},
                        "result": to_jsonable(model_retry_result),
                    }
                ],
                "response": (
                    "PID 细调后仍未达标，已自动回流系统辨识智能体并切换到候选窗口 "
                    f"{model_retry_result.get('window_source', '-')}"
                    "，重新辨识模型并生成新的整定结果。"
                ),
            }
        )

    final_rating = float(shared_data.get("final_rating", 0.0) or 0.0)
    performance_score = float(shared_data.get("performance_score", 0.0) or 0.0)
    passed = bool(shared_data.get("passed", False))
    if auto_refine_result.get("applied") or model_retry_result.get("applied"):
        turns.append(
            {
                "type": "agent_turn",
                "agent": display_agent_names["evaluation_expert"],
                "tools": [
                    {
                        "tool_name": "tool_evaluate_pid",
                        "args": {"phase": "post_feedback_evaluation"},
                        "result": to_jsonable(
                            {
                                "passed": passed,
                                "performance_score": performance_score,
                                "final_rating": final_rating,
                                "feedback_target": shared_data.get("feedback_target", ""),
                                "failure_reason": shared_data.get("failure_reason", ""),
                            }
                        ),
                    }
                ],
                "response": (
                    f"自动回流后已完成重新评估，当前 performance_score={performance_score:.2f}，"
                    f"final_rating={final_rating:.2f}，结果为{'通过' if passed else '未通过'}。"
                ),
            }
        )

    return turns
