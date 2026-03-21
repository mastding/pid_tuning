from __future__ import annotations

from typing import Any, Callable, List

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
    window: int,
    loop_type: str,
    tool_load_data: Callable[..., Any],
    tool_fetch_history_data: Callable[..., Any],
    tool_fit_fopdt: Callable[..., Any],
    tool_query_expert_knowledge: Callable[..., Any],
    tool_tune_pid: Callable[..., Any],
    tool_evaluate_pid: Callable[..., Any],
) -> List[AssistantAgent]:
    if csv_path:
        data_analyst_tools = [tool_load_data]
        data_analyst_prompt = f"""You are the data analysis agent.
The user already provided a local CSV file: "{csv_path}".
Call tool_load_data(csv_path="{csv_path}") directly. Do not call tool_fetch_history_data.
After the tool succeeds, summarize the data size, sampling time, candidate step count, and selected identification window in one concise Chinese sentence."""
    else:
        data_analyst_tools = [tool_fetch_history_data, tool_load_data]
        data_analyst_prompt = f"""You are the data analysis agent.
No CSV file was uploaded, so you must fetch historical data first.
Call:
tool_fetch_history_data(loop_uri="{loop_uri}", start_time="{start_time}", end_time="{end_time}", data_type="{data_type}", window={window})
Then call tool_load_data(csv_path=...) with the returned csv_path.
After both steps succeed, summarize the data size, sampling time, candidate step count, and selected identification window in one concise Chinese sentence."""

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
        system_message="""You are the system identification agent.
Call tool_fit_fopdt(dt=1.0) to identify the best process model.
Focus on:
- model_type
- selected_model_params
- working K/T/L parameters
- normalized_rmse
- r2_score
- confidence
- model_selection_reason
Reply with one concise Chinese sentence that distinguishes the raw model parameters from the current working model parameters.""",
        tools=[tool_fit_fopdt],
        model_client_stream=False,
    )

    knowledge_expert = AssistantAgent(
        name="knowledge_expert",
        model_client=model_client,
        system_message=f"""You are the expert knowledge graph agent.
Use the identified model, plant context, and control scenario to retrieve expert rules before PID tuning.
Call:
tool_query_expert_knowledge(loop_type="{loop_type}", loop_name="...", plant_type="...", scenario="...", control_object="...", tower_section="...", control_target="...")
Prefer using the user-provided plant_type, scenario, and control_object when available. If unknown, pass empty strings.
After the tool succeeds, summarize the matched rule count, preferred strategy, major risk hints, and one concise expert-rule takeaway in one concise Chinese sentence.""",
        tools=[tool_query_expert_knowledge],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    pid_expert = AssistantAgent(
        name="pid_expert",
        model_client=model_client,
        system_message=f"""You are the PID tuning expert.
Read model_type, selected_model_params, and the retrieved expert knowledge guidance from shared state first.
Treat the compatibility K/T/L values as display fields only.
Call:
tool_tune_pid(loop_type="{loop_type}", model_type="...", selected_model_params={{...}})
Pass model_type and selected_model_params as the primary tuning inputs.
Do not pass compatibility K/T/L when selected_model_params is available.
Only use compatibility K/T/L for FO/FOPDT legacy fallback if the raw model parameters are unavailable.
Pass the raw model parameters in selected_model_params:
- SOPDT: K/T1/T2/L
- IPDT: integrating-process parameters and L
- FO/FOPDT: the corresponding raw parameters
Do not paste a quoted JSON string into selected_model_params.
Do not include fitting metadata such as success, message, residue, normalized_rmse, raw_rmse, or r2_score.
Examples:
- SOPDT -> selected_model_params={{"model_type":"SOPDT","K":0.44,"T1":1.0,"T2":1.0,"L":0.0}}
- FOPDT -> selected_model_params={{"model_type":"FOPDT","K":0.44,"T":2.0,"L":0.0}}
After the tools succeed, summarize the model type, selected strategy, PID parameters, expert knowledge guidance, and experience guidance in one concise Chinese sentence.""",
        tools=[tool_tune_pid],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message="""You are the evaluation agent.
Evaluate the tuned PID parameters against the selected process model.
Call:
tool_evaluate_pid(model_type="...", selected_model_params={...}, Kp=..., Ki=..., Kd=..., method="auto")
Pass model_type and selected_model_params as the primary evaluation inputs.
Do not pass compatibility K/T/L when selected_model_params is available.
Only use compatibility K/T/L as a fallback if the raw model parameters are unavailable.
If passed=true, end your final reply with APPROVE.
If passed=false, explain the primary reason, the recommended feedback target, and the next action in concise Chinese. Do not output APPROVE in that case.""",
        tools=[tool_evaluate_pid],
        model_client_stream=False,
    )

    return [data_analyst, system_id_expert, knowledge_expert, pid_expert, evaluation_expert]
