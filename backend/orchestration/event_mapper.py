from __future__ import annotations

from typing import Any, Callable, Dict, List


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _inject_experience_tool(
    agent_name: str,
    tools: List[Dict[str, Any]],
    *,
    display_agent_names: Dict[str, str],
) -> List[Dict[str, Any]]:
    if agent_name != display_agent_names["pid_expert"]:
        return tools
    if any(tool.get("tool_name") == "tool_search_experience" for tool in tools):
        return tools

    tune_tool = next(
        (
            tool
            for tool in tools
            if tool.get("tool_name") == "tool_tune_pid" and isinstance(tool.get("result"), dict)
        ),
        None,
    )
    if tune_tool is None:
        return tools

    tune_result = tune_tool["result"]
    experience_guidance = tune_result.get("experience_guidance")
    if not isinstance(experience_guidance, dict):
        return tools

    selection_inputs = tune_result.get("selection_inputs") or {}
    selected_model_params = tune_result.get("selected_model_params") or selection_inputs.get("selected_model_params") or {}
    normalized_model_type = str(selection_inputs.get("model_type") or tune_result.get("model_type") or "").upper()
    experience_args = {
        "loop_type": selection_inputs.get("loop_type"),
        "model_type": selection_inputs.get("model_type"),
        "selected_model_params": selected_model_params,
        "limit": 3,
    }
    lookup_tool = {
        "tool_name": "tool_search_experience",
        "args": experience_args,
        "result": {
            "preferred_strategy": experience_guidance.get("preferred_strategy", ""),
            "preferred_model_type": experience_guidance.get("preferred_model_type", ""),
            "summary": experience_guidance.get("summary", {}),
            "guidance": experience_guidance.get("guidance", ""),
            "matches": experience_guidance.get("matches", []),
            "preferred_refine_pattern": experience_guidance.get("preferred_refine_pattern", ""),
            "recommended_kp_scale": experience_guidance.get("recommended_kp_scale"),
            "recommended_ki_scale": experience_guidance.get("recommended_ki_scale"),
            "recommended_kd_scale": experience_guidance.get("recommended_kd_scale"),
        },
        "is_error": False,
    }
    return [lookup_tool, *tools]


def _summarize_raw_model(model_type: str, model_params: Dict[str, Any]) -> str:
    normalized = (model_type or "FOPDT").upper()
    if normalized == "SOPDT":
        return (
            f"\u539f\u59cb\u6a21\u578b\u53c2\u6570 K={_format_float(model_params.get('K'))}, "
            f"T1={_format_float(model_params.get('T1'))}, "
            f"T2={_format_float(model_params.get('T2'))}, "
            f"L={_format_float(model_params.get('L'))}\u3002"
        )
    if normalized == "FO":
        return (
            f"\u539f\u59cb\u6a21\u578b\u53c2\u6570 K={_format_float(model_params.get('K'))}, "
            f"T={_format_float(model_params.get('T'))}\u3002"
        )
    if normalized == "IPDT":
        return (
            f"\u539f\u59cb\u6a21\u578b\u53c2\u6570 K={_format_float(model_params.get('K'))}, "
            f"L={_format_float(model_params.get('L'))}\u3002"
        )
    return (
        f"\u539f\u59cb\u6a21\u578b\u53c2\u6570 K={_format_float(model_params.get('K'))}, "
        f"T={_format_float(model_params.get('T'))}, "
        f"L={_format_float(model_params.get('L'))}\u3002"
    )


def _summarize_tuning_model(model_type: str, latest_result: Dict[str, Any]) -> str:
    if model_type == "SOPDT":
        return (
            f"\u5f53\u524d\u6574\u5b9a\u9636\u6bb5\u91c7\u7528\u7684\u5de5\u4f5c\u6a21\u578b\u53c2\u6570\u4e3a K={_format_float(latest_result.get('K'))}, "
            f"T={_format_float(latest_result.get('T'))}, L={_format_float(latest_result.get('L'))}\u3002"
        )
    if model_type == "IPDT":
        return (
            f"\u5f53\u524d\u6574\u5b9a\u9636\u6bb5\u91c7\u7528\u79ef\u5206\u8fc7\u7a0b\u5de5\u4f5c\u53c2\u6570 K={_format_float(latest_result.get('K'))}, "
            f"L={_format_float(latest_result.get('L'))}\u3002"
        )
    if model_type == "FO":
        return (
            f"\u5f53\u524d\u6574\u5b9a\u9636\u6bb5\u91c7\u7528\u4e00\u9636\u5de5\u4f5c\u6a21\u578b\u53c2\u6570 K={_format_float(latest_result.get('K'))}, "
            f"T={_format_float(latest_result.get('T'))}\u3002"
        )
    return (
        f"\u5f53\u524d\u6574\u5b9a\u9636\u6bb5\u91c7\u7528\u5de5\u4f5c\u6a21\u578b\u53c2\u6570 K={_format_float(latest_result.get('K'))}, "
        f"T={_format_float(latest_result.get('T'))}, L={_format_float(latest_result.get('L'))}\u3002"
    )


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
            f"\u5df2\u5b8c\u6210\u6570\u636e\u52a0\u8f7d\u4e0e\u9884\u5904\u7406\uff0c\u5171\u83b7\u5f97 {points} \u4e2a\u6570\u636e\u70b9\uff0c"
            f"\u5f53\u524d\u7528\u4e8e\u8fa8\u8bc6\u7684\u7a97\u53e3\u4e3a {window_points} \u70b9\uff0c\u91c7\u6837\u5468\u671f\u7ea6 {float(sampling_time):.2f} s\uff0c"
            f"\u68c0\u6d4b\u5230 {len(step_events)} \u4e2a\u5019\u9009\u9636\u8dc3\u4e8b\u4ef6\u3002"
        )

    if agent_name == display_agent_names["system_id_expert"]:
        model_type = str(latest_result.get("model_type", "FOPDT")).upper()
        selected_model_params = latest_result.get("selected_model_params", {}) or {}
        raw_model_summary = _summarize_raw_model(model_type, selected_model_params)
        quality_summary = (
            f"\u6807\u51c6\u5316RMSE {_format_float(latest_result.get('normalized_rmse'), 3)}\uff0c"
            f"R\u00b2 {_format_float(latest_result.get('r2_score'), 3)}\uff0c"
            f"\u6a21\u578b\u7f6e\u4fe1\u5ea6 {_format_float(latest_result.get('confidence'), 2)}\u3002"
        )
        selection_reason = latest_result.get("model_selection_reason") or latest_result.get("selection_reason") or ""
        suffix = f" \u9009\u6a21\u4f9d\u636e\uff1a{selection_reason}\u3002" if selection_reason else ""
        return f"{model_type} \u8fc7\u7a0b\u6a21\u578b\u8fa8\u8bc6\u5b8c\u6210\uff0c{raw_model_summary} {quality_summary}{suffix}"

    if agent_name == display_agent_names["pid_expert"]:
        if latest_tool_name == "tool_search_experience":
            match_count = len(latest_result.get("matches") or [])
            preferred_strategy = latest_result.get("preferred_strategy") or "\u672a\u5f62\u6210\u660e\u786e\u504f\u597d"
            guidance_text = latest_result.get("guidance") or ""
            message = f"\u5df2\u68c0\u7d22\u5230 {match_count} \u6761\u76f8\u4f3c\u56de\u8def\u7ecf\u9a8c\uff0c\u5386\u53f2\u504f\u597d\u7b56\u7565\u4e3a {preferred_strategy}\u3002"
            if guidance_text:
                message += f" \u5386\u53f2\u7ecf\u9a8c\u53c2\u8003\uff1a{guidance_text}"
            return message
        if latest_tool_name == "tool_tune_pid":
            strategy_used = latest_result.get("strategy_used", latest_result.get("strategy", ""))
            guidance = latest_result.get("experience_guidance") or {}
            guidance_text = guidance.get("guidance", "") if isinstance(guidance, dict) else ""
            response = (
                f"\u5df2\u6309 {strategy_used} \u7b56\u7565\u5b8c\u6210\u6574\u5b9a\uff0c"
                f"Kp={_format_float(latest_result.get('Kp'), 4)}\uff0c"
                f"Ki={_format_float(latest_result.get('Ki'), 4)}\uff0c"
                f"Kd={_format_float(latest_result.get('Kd'), 4)}\u3002"
                f"\u7b56\u7565\u9009\u62e9\u4f9d\u636e\uff1a{latest_result.get('selection_reason', '')}\u3002"
            )
            if guidance_text:
                response += f" \u5386\u53f2\u7ecf\u9a8c\u53c2\u8003\uff1a{guidance_text}"
            return response
        return "PID \u53c2\u6570\u6574\u5b9a\u5b8c\u6210\u3002"

    if agent_name == display_agent_names["evaluation_expert"]:
        if latest_result.get("passed") is True:
            model_type = str(latest_result.get("model_type") or latest_result.get("evaluated_model_type") or "").upper()
            model_hint = f"\uff0c\u8bc4\u4f30\u6a21\u578b\u7c7b\u578b {model_type}" if model_type else ""
            return (
                f"\u6027\u80fd\u8bc4\u5206 {_format_float(latest_result.get('performance_score'), 2)}\uff0c"
                f"\u65b9\u6cd5\u7f6e\u4fe1\u5ea6 {_format_float(latest_result.get('method_confidence'), 3)}\uff0c"
                f"\u6700\u7ec8\u8bc4\u5206 {_format_float(latest_result.get('final_rating'), 2)}{model_hint}\uff0cAPPROVE\u3002"
            )
        target_display = latest_result.get("feedback_target_display") or latest_result.get("feedback_target") or "\u540e\u7eed\u667a\u80fd\u4f53"
        model_type = str(latest_result.get("model_type") or latest_result.get("evaluated_model_type") or "").upper()
        model_hint = f" \u5f53\u524d\u6309 {model_type} \u6a21\u578b\u5b8c\u6210\u95ed\u73af\u8bc4\u4f30\u3002" if model_type else ""
        failure_reason = latest_result.get("failure_reason") or "\u672a\u77e5\u539f\u56e0"
        feedback_action = latest_result.get("feedback_action", "")
        return (
            f"\u9996\u6b21\u8bc4\u4f30\u672a\u901a\u8fc7\uff0c\u4e3b\u56e0\uff1a{failure_reason}\u3002"
            f"\u5efa\u8bae\u4e0b\u4e00\u6b65\u56de\u6d41\u7ed9 {target_display}\uff0c{feedback_action}\u3002{model_hint}"
        )

    return ""


def finalize_agent_turn(
    current_turn_data: Dict[str, Any] | None,
    *,
    build_agent_response: Callable[..., str],
    display_agent_names: Dict[str, str],
) -> Dict[str, Any] | None:
    if current_turn_data is None:
        return None

    current_turn_data["tools"] = _inject_experience_tool(
        current_turn_data.get("agent", ""),
        current_turn_data.get("tools", []),
        display_agent_names=display_agent_names,
    )

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

    if not existing_response or existing_response in {"\u5b8c\u6210", "APPROVE"}:
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
                            "model_type": selection_inputs.get("model_type", shared_data.get("model_type", "FOPDT")),
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
                                "selection_inputs": selection_inputs,
                                "selected_model_params": shared_data.get("selected_model_params", {}),
                            }
                        ),
                    }
                ],
                "response": (
                    "\u6839\u636e\u9996\u6b21\u8bc4\u4f30\u53cd\u9988\uff0c\u5df2\u81ea\u52a8\u56de\u6d41 PID \u4e13\u5bb6\u7ee7\u7eed\u7ec6\u8c03\u53c2\u6570\u3002\n"
                    f"\u5c06 final_rating \u4ece {float(auto_refine_result.get('base_final_rating', 0.0)):.2f} "
                    f"\u63d0\u5347\u5230 {float(auto_refine_result.get('refined_final_rating', 0.0)):.2f}\u3002"
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
                    "\u57fa\u4e8e\u8bc4\u4f30\u53cd\u9988\uff0c\u7cfb\u7edf\u5df2\u81ea\u52a8\u5207\u6362\u5230\u5176\u4ed6\u5019\u9009\u8fa8\u8bc6\u7a97\u53e3\u5e76\u91cd\u65b0\u8fa8\u8bc6\u6a21\u578b\u3002\n"
                    f"\u5f53\u524d\u91c7\u7528\u7a97\u53e3\u6765\u6e90\uff1a{model_retry_result.get('selected_window_source', '')}\u3002"
                ),
            }
        )

    return turns
