from __future__ import annotations

from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from services.system_config_service import is_knowledge_expert_enabled

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
        data_analyst_prompt = f"""你是数据分析智能体。
用户已经提供了本地CSV文件："{csv_path}"。
直接调用 tool_load_data(csv_path="{csv_path}")，不要调用 tool_fetch_history_data。
工具成功后，用一句简洁的中文总结数据规模、采样时间、候选阶跃数量，并清楚地说明"所有识别的候选窗口已传递给系统辨识智能体进行统一的多窗口多模型择优"。"""
    else:
        data_analyst_tools = [tool_fetch_history_data, tool_load_data]
        data_analyst_prompt = f"""你是数据分析智能体。
用户未上传CSV文件，必须先获取历史数据。
调用：
tool_fetch_history_data(loop_uri="{loop_uri}", start_time="{start_time}", end_time="{end_time}", data_type="{data_type}", window={window})
然后用返回的csv_path调用 tool_load_data(csv_path=...)。
两个步骤成功后，用一句简洁的中文总结数据规模、采样时间、候选阶跃数量，并清楚地说明"所有识别的候选窗口已传递给系统辨识智能体进行统一的多窗口多模型择优"。"""

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
        system_message="""你是系统辨识智能体。
调用 tool_fit_fopdt(dt=1.0) 来辨识最优过程模型。
重点关注：
- model_type（模型类型）
- selected_model_params（选中的模型参数）
- working K/T/L parameters（工作参数）
- normalized_rmse（归一化均方根误差）
- r2_score（拟合优度）
- confidence（置信度）
- model_selection_reason（模型选择理由）
用一句简洁的中文回答，区分原始模型参数和当前工作模型参数的差异。""",
        tools=[tool_fit_fopdt],
        model_client_stream=False,
    )

    knowledge_expert = AssistantAgent(
        name="knowledge_expert",
        model_client=model_client,
        system_message=f"""你是专家知识图谱智能体。
使用已辨识的模型、装置背景和控制工况来检索PID整定前的专家规则。
调用：
tool_query_expert_knowledge(loop_type="{loop_type}", loop_name="...", plant_type="...", scenario="...", control_object="...", tower_section="...", control_target="...")
优先使用用户提供的plant_type、scenario和control_object。如果未知，传递空字符串。
工具成功后，用一句简洁的中文总结匹配的规则数量、首选策略、主要风险提示以及专家规则的核心启示。""",
        tools=[tool_query_expert_knowledge],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    pid_expert = AssistantAgent(
        name="pid_expert",
        model_client=model_client,
        system_message=f"""你是PID整定专家。
先从共享状态读取model_type、selected_model_params和检索到的专家知识指导。
将兼容性K/T/L值仅视为显示字段。
调用：
tool_tune_pid(loop_type="{loop_type}", model_type="...", selected_model_params={{...}})
将model_type和selected_model_params作为主要整定输入。
当selected_model_params可用时，不传递兼容性K/T/L。
仅在原始模型参数不可用时，使用兼容性K/T/L作为FO/FOPDT遗留备选。
在selected_model_params中传递原始模型参数：
- SOPDT: K/T1/T2/L
- IPDT: 积分过程参数和L
- FO/FOPDT: 对应的原始参数
不要将引用的JSON字符串粘贴到selected_model_params中。
不包含拟合元数据，如success、message、residue、normalized_rmse、raw_rmse或r2_score。
示例：
- SOPDT -> selected_model_params={{\"model_type\":\"SOPDT\",\"K\":0.44,\"T1\":1.0,\"T2\":1.0,\"L\":0.0}}
- FOPDT -> selected_model_params={{\"model_type\":\"FOPDT\",\"K\":0.44,\"T\":2.0,\"L\":0.0}}
工具成功后，用一句简洁的中文总结模型类型、选定策略、PID参数、专家知识指导和经验指导。""",
        tools=[tool_tune_pid],
        model_client_stream=False,
        max_tool_iterations=2,
    )

    evaluation_expert = AssistantAgent(
        name="evaluation_expert",
        model_client=model_client,
        system_message="""你是评估智能体。
根据选定的过程模型评估已整定的PID参数。
调用：
tool_evaluate_pid(model_type="...", selected_model_params={...}, Kp=..., Ki=..., Kd=..., method="auto")
将model_type和selected_model_params作为主要评估输入。
当selected_model_params可用时，不传递兼容性K/T/L。
仅在原始模型参数不可用时，使用兼容性K/T/L作为备选。
如果passed=true，在最终回答末尾输出APPROVE。
如果passed=false，用简洁的中文解释主要原因、推荐的反馈目标和后续行动。在这种情况下不输出APPROVE。""",
        tools=[tool_evaluate_pid],
        model_client_stream=False,
    )

    if is_knowledge_expert_enabled():
        return [data_analyst, system_id_expert, knowledge_expert, pid_expert, evaluation_expert]
    return [data_analyst, system_id_expert, pid_expert, evaluation_expert]
