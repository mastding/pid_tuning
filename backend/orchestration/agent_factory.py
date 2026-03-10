from __future__ import annotations

from typing import Any, Callable, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def create_pid_agents(
    *,
    model_client: OpenAIChatCompletionClient,
    csv_path: str,
    loop_uri: str,
    start_time: str,
    end_time: str,
    data_type: str,
    loop_type: str,
    tool_load_data: Callable[..., Any],
    tool_fetch_history_data: Callable[..., Any],
    tool_fit_fopdt: Callable[..., Any],
    tool_tune_pid: Callable[..., Any],
    tool_evaluate_pid: Callable[..., Any],
) -> List[AssistantAgent]:
    if csv_path:
        data_analyst_tools = [tool_load_data]
        data_analyst_prompt = f"""你是数据分析专家。
本次任务已经上传本地 CSV，路径是 "{csv_path}"。
当轮到你发言时，只允许调用 tool_load_data(csv_path="{csv_path}")，不要调用 tool_fetch_history_data。
读取工具返回的数据摘要后，用一句话总结数据质量、采样时间和选中的辨识窗口，并以“完成”结束。"""
    else:
        data_analyst_tools = [tool_fetch_history_data, tool_load_data]
        data_analyst_prompt = f"""你是数据分析专家。
本次任务未上传本地 CSV。你必须先调用：
tool_fetch_history_data(loop_uri="{loop_uri}", start_time="{start_time}", end_time="{end_time}", data_type="{data_type}")
从工具结果中读取 csv_path 后，再调用 tool_load_data(csv_path=该路径)。
不要跳过这两个步骤，也不要混用其他数据来源。
最后用一句话总结获取的数据量、采样时间、检测到的阶跃事件和选中的辨识窗口，并以“完成”结束。"""

    data_analyst = AssistantAgent(
        name="data_analyst",
        model_client=model_client,
        system_message=data_analyst_prompt,
        tools=data_analyst_tools,
        model_client_stream=False,
        max_tool_iterations=2,
    )

    system_id_expert = AssistantAgent(
        name="system_id_expert",
        model_client=model_client,
        system_message="""你是系统辨识专家。
当轮到你发言时，立即调用 tool_fit_fopdt(dt=1.0)。
读取工具返回的 K、T、L、residue、confidence 后，用一句话总结模型质量，并以“完成”结束。""",
        tools=[tool_fit_fopdt],
        model_client_stream=False,
    )

    pid_expert = AssistantAgent(
        name="pid_expert",
        model_client=model_client,
        system_message=f"""你是 PID 整定专家。
请按以下步骤工作：
1. 从系统辨识结果读取 K、T、L。
2. 调用 tool_tune_pid(K=..., T=..., L=..., loop_type="{loop_type}")。
3. 用一句话总结选中的整定策略和 PID 参数，并以“完成”结束。""",
        tools=[tool_tune_pid],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message="""你是整定评估专家。
请按以下步骤工作：
1. 读取当前 K、T、L、Kp、Ki、Kd。
2. 调用 tool_evaluate_pid(K=..., T=..., L=..., Kp=..., Ki=..., Kd=..., method="auto")。
3. 若 passed=true，明确说明通过并输出 APPROVE。
4. 若 passed=false，明确说明未通过主因、建议回流的智能体和后续动作，但不要输出 APPROVE。""",
        tools=[tool_evaluate_pid],
        model_client_stream=False,
    )

    return [data_analyst, system_id_expert, pid_expert, evaluation_expert]
