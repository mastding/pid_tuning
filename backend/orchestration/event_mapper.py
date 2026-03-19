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
    selected_model_params = (
        tune_result.get("selected_model_params")
        or selection_inputs.get("selected_model_params")
        or {}
    )
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
            f"原始模型参数 K={_format_float(model_params.get('K'))}, "
            f"T1={_format_float(model_params.get('T1'))}, "
            f"T2={_format_float(model_params.get('T2'))}, "
            f"L={_format_float(model_params.get('L'))}。"
        )
    if normalized == "FO":
        return (
            f"原始模型参数 K={_format_float(model_params.get('K'))}, "
            f"T={_format_float(model_params.get('T'))}。"
        )
    if normalized == "IPDT":
        return (
            f"原始模型参数 K={_format_float(model_params.get('K'))}, "
            f"L={_format_float(model_params.get('L'))}。"
        )
    return (
        f"原始模型参数 K={_format_float(model_params.get('K'))}, "
        f"T={_format_float(model_params.get('T'))}, "
        f"L={_format_float(model_params.get('L'))}。"
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
            f"已完成数据加载与预处理，共获得 {points} 个数据点，"
            f"当前用于辨识的窗口为 {window_points} 点，采样周期约 {float(sampling_time):.2f} s，"
            f"检测到 {len(step_events)} 个候选阶跃事件。"
        )

    if agent_name == display_agent_names["system_id_expert"]:
        model_type = str(latest_result.get("model_type", "FOPDT")).upper()
        selected_model_params = latest_result.get("selected_model_params", {}) or {}
        raw_model_summary = _summarize_raw_model(model_type, selected_model_params)
        quality_summary = (
            f"标准化RMSE {_format_float(latest_result.get('normalized_rmse'), 3)}，"
            f"R² {_format_float(latest_result.get('r2_score'), 3)}，"
            f"模型置信度 {_format_float(latest_result.get('confidence'), 2)}。"
        )
        selection_reason = latest_result.get("model_selection_reason") or latest_result.get("selection_reason") or ""
        suffix = f" 选模依据：{selection_reason}。" if selection_reason else ""
        return f"{model_type} 过程模型辨识完成，{raw_model_summary} {quality_summary}{suffix}"

    if agent_name == display_agent_names["knowledge_expert"]:
        if latest_tool_name == "tool_query_expert_knowledge":
            match_count = int(latest_result.get("matched_count") or 0)
            preferred_strategy = latest_result.get("preferred_strategy") or "未形成明确偏好"
            summary = latest_result.get("summary") or ""
            risk_hints = latest_result.get("risk_hints") or []
            matched_rules = latest_result.get("matched_rules") or []
            constraints = latest_result.get("constraints") or []
            message = f"已检索到 {match_count} 条专家规则，推荐优先策略为 {preferred_strategy}。"
            if matched_rules:
                message += f" 命中规则：{matched_rules[0].get('title', '')}"
                if matched_rules[0].get("summary"):
                    message += f"，{matched_rules[0].get('summary', '')}"
            elif summary:
                message += f" 规则摘要：{summary}"
            if constraints:
                message += f" 主要约束：{constraints[0].get('title', '')}"
            if risk_hints:
                message += f" 关注要点：{risk_hints[0]}"
            return message
        return "已完成专家规则知识检索。"

    if agent_name == display_agent_names["pid_expert"]:
        if latest_tool_name == "tool_search_experience":
            match_count = len(latest_result.get("matches") or [])
            preferred_strategy = latest_result.get("preferred_strategy") or "未形成明确偏好"
            guidance_text = latest_result.get("guidance") or ""
            message = f"已检索到 {match_count} 条相似回路经验，历史偏好策略为 {preferred_strategy}。"
            if guidance_text:
                message += f" 历史经验参考：{guidance_text}"
            return message
        if latest_tool_name == "tool_tune_pid":
            strategy_used = latest_result.get("strategy_used", latest_result.get("strategy", ""))
            guidance = latest_result.get("experience_guidance") or {}
            knowledge = latest_result.get("expert_knowledge_guidance") or {}
            guidance_text = guidance.get("guidance", "") if isinstance(guidance, dict) else ""
            knowledge_text = knowledge.get("summary", "") if isinstance(knowledge, dict) else ""
            knowledge_strategy = knowledge.get("preferred_strategy", "") if isinstance(knowledge, dict) else ""
            knowledge_rules = knowledge.get("matched_rules", []) if isinstance(knowledge, dict) else []
            knowledge_constraints = knowledge.get("constraints", []) if isinstance(knowledge, dict) else []
            response = (
                f"已按 {strategy_used} 策略完成整定，"
                f"Kp={_format_float(latest_result.get('Kp'), 4)}，"
                f"Ki={_format_float(latest_result.get('Ki'), 4)}，"
                f"Kd={_format_float(latest_result.get('Kd'), 4)}。"
                f"策略选择依据：{latest_result.get('selection_reason', '')}。"
            )
            if knowledge_rules:
                top_rule = knowledge_rules[0]
                response += f" 本体知识参考：{top_rule.get('title', '')}"
                if top_rule.get("summary"):
                    response += f"，{top_rule.get('summary', '')}"
            elif knowledge_text:
                response += f" 专家规则参考：{knowledge_text}"
            if knowledge_strategy:
                response += f" 专家偏好策略：{knowledge_strategy}。"
            if knowledge_constraints:
                response += f" 主要约束：{knowledge_constraints[0].get('title', '')}。"
            if guidance_text:
                response += f" 历史经验参考：{guidance_text}"
            return response
        return "PID 参数整定完成。"

    if agent_name == display_agent_names["evaluation_expert"]:
        if latest_result.get("passed") is True:
            model_type = str(latest_result.get("model_type") or latest_result.get("evaluated_model_type") or "").upper()
            model_hint = f"，评估模型类型 {model_type}" if model_type else ""
            return (
                f"性能评分 {_format_float(latest_result.get('performance_score'), 2)}，"
                f"方法置信度 {_format_float(latest_result.get('method_confidence'), 3)}，"
                f"最终评分 {_format_float(latest_result.get('final_rating'), 2)}{model_hint}，APPROVE。"
            )
        target_display = latest_result.get("feedback_target_display") or latest_result.get("feedback_target") or "后续智能体"
        model_type = str(latest_result.get("model_type") or latest_result.get("evaluated_model_type") or "").upper()
        model_hint = f" 当前按 {model_type} 模型完成闭环评估。" if model_type else ""
        failure_reason = latest_result.get("failure_reason") or "未知原因"
        feedback_action = latest_result.get("feedback_action", "")
        return (
            f"首次评估未通过，主因：{failure_reason}。"
            f"建议下一步回流给 {target_display}，{feedback_action}。{model_hint}"
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
                    "根据首次评估反馈，已自动回流 PID 专家继续细调参数。\n"
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
                    "基于评估反馈，系统已自动切换到其他候选辨识窗口并重新辨识模型。\n"
                    f"当前采用窗口来源：{model_retry_result.get('selected_window_source', '')}。"
                ),
            }
        )

    return turns
