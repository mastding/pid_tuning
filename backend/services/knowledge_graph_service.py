from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import requests


RULES_PATHS = [
    Path(__file__).resolve().parents[1] / "knowledge_graph" / "distillation_pid_expert_rules.jsonl",
    Path(__file__).resolve().parents[1] / "knowledge_graph" / "generic_pid_expert_rules.jsonl",
]

LOOP_TYPE_LABELS = {
    "flow": "流量",
    "temperature": "温度",
    "pressure": "压力",
    "level": "液位",
    "composition": "成分",
    "ph": "pH",
    "speed": "转速",
    "position": "位置",
    "tension": "张力",
    "humidity": "湿度",
    "dissolved_oxygen": "溶解氧",
    "unknown": "未知",
    "any": "通用",
}

PLANT_TYPE_LABELS = {
    "any": "通用设备",
    "distillation_column": "精馏塔",
    "atmospheric_column": "常压塔",
    "vacuum_column": "减压塔",
    "side_draw_column": "侧线抽出塔",
    "reboiler": "再沸器",
    "condenser": "冷凝器",
    "reflux_system": "回流系统",
    "heat_exchanger": "换热器",
    "reactor": "反应器",
    "cstr": "连续搅拌釜",
    "ph_neutralization": "pH中和槽",
    "boiler": "锅炉",
    "steam_drum": "汽包",
    "evaporator": "蒸发器",
    "dryer": "干燥器",
    "furnace": "工业炉",
    "compressor": "压缩机",
    "pump_station": "泵站",
    "tank": "储罐",
    "pipeline": "管道",
    "chiller": "冷冻机组",
    "ahu": "空调机组",
    "cooling_tower": "冷却塔",
    "motor_drive": "电机调速系统",
    "servo_axis": "伺服轴",
    "web_tension_system": "张力系统",
    "aeration_basin": "曝气池",
}

PLANT_TYPE_ALIASES = {
    "any": "any",
    "通用设备": "any",
    "distillation_column": "distillation_column",
    "精馏塔": "distillation_column",
    "atmospheric_column": "atmospheric_column",
    "常压塔": "atmospheric_column",
    "vacuum_column": "vacuum_column",
    "减压塔": "vacuum_column",
    "side_draw_column": "side_draw_column",
    "侧线抽出塔": "side_draw_column",
    "reboiler": "reboiler",
    "再沸器": "reboiler",
    "condenser": "condenser",
    "冷凝器": "condenser",
    "reflux_system": "reflux_system",
    "回流系统": "reflux_system",
    "heat_exchanger": "heat_exchanger",
    "换热器": "heat_exchanger",
    "reactor": "reactor",
    "反应器": "reactor",
    "cstr": "cstr",
    "连续搅拌釜": "cstr",
    "ph_neutralization": "ph_neutralization",
    "ph中和槽": "ph_neutralization",
    "p h中和槽": "ph_neutralization",
    "pH中和槽": "ph_neutralization",
    "boiler": "boiler",
    "锅炉": "boiler",
    "steam_drum": "steam_drum",
    "汽包": "steam_drum",
    "evaporator": "evaporator",
    "蒸发器": "evaporator",
    "dryer": "dryer",
    "干燥器": "dryer",
    "furnace": "furnace",
    "工业炉": "furnace",
    "compressor": "compressor",
    "压缩机": "compressor",
    "pump_station": "pump_station",
    "泵站": "pump_station",
    "tank": "tank",
    "储罐": "tank",
    "pipeline": "pipeline",
    "管道": "pipeline",
    "chiller": "chiller",
    "冷冻机组": "chiller",
    "ahu": "ahu",
    "空调机组": "ahu",
    "air_handling_unit": "ahu",
    "cooling_tower": "cooling_tower",
    "冷却塔": "cooling_tower",
    "motor_drive": "motor_drive",
    "电机调速系统": "motor_drive",
    "servo_axis": "servo_axis",
    "伺服轴": "servo_axis",
    "web_tension_system": "web_tension_system",
    "张力系统": "web_tension_system",
    "aeration_basin": "aeration_basin",
    "曝气池": "aeration_basin",
}

INDUSTRY_LABELS = {
    "chemical": "化工",
    "oil_gas": "石油天然气",
    "power": "电力",
    "water_treatment": "水处理",
    "hvac": "暖通空调",
    "pharma": "制药",
    "food_beverage": "食品饮料",
    "metallurgy": "冶金",
    "paper": "造纸",
    "semiconductor": "半导体",
    "manufacturing": "通用制造",
}

INDUSTRY_ALIASES = {
    "chemical": "chemical",
    "化工": "chemical",
    "oil_gas": "oil_gas",
    "oilgas": "oil_gas",
    "石油天然气": "oil_gas",
    "油气": "oil_gas",
    "power": "power",
    "电力": "power",
    "water_treatment": "water_treatment",
    "水处理": "water_treatment",
    "hvac": "hvac",
    "暖通空调": "hvac",
    "暖通": "hvac",
    "pharma": "pharma",
    "制药": "pharma",
    "food_beverage": "food_beverage",
    "食品饮料": "food_beverage",
    "metallurgy": "metallurgy",
    "冶金": "metallurgy",
    "paper": "paper",
    "造纸": "paper",
    "semiconductor": "semiconductor",
    "半导体": "semiconductor",
    "manufacturing": "manufacturing",
    "通用制造": "manufacturing",
    "制造": "manufacturing",
}

SCENARIO_LABELS = {
    "startup": "开车",
    "shutdown": "停车",
    "steady_operation": "稳态生产",
    "load_change": "变负荷",
    "product_switch": "切产品",
    "feed_switch": "原料切换",
    "tower_pressure_fluctuation": "塔压波动",
    "reflux_fluctuation": "回流波动",
    "steam_fluctuation": "蒸汽波动",
    "analyzer_maintenance": "分析仪维护",
    "low_load": "低负荷",
    "full_load": "满负荷",
    "mode_switch": "模式切换",
    "batch_switch": "批次切换",
    "season_change": "季节切换",
}

SCENARIO_ALIASES = {
    "startup": "startup",
    "开车": "startup",
    "shutdown": "shutdown",
    "停车": "shutdown",
    "steady_operation": "steady_operation",
    "稳态生产": "steady_operation",
    "load_change": "load_change",
    "变负荷": "load_change",
    "product_switch": "product_switch",
    "切产品": "product_switch",
    "feed_switch": "feed_switch",
    "原料切换": "feed_switch",
    "tower_pressure_fluctuation": "tower_pressure_fluctuation",
    "塔压波动": "tower_pressure_fluctuation",
    "reflux_fluctuation": "reflux_fluctuation",
    "回流波动": "reflux_fluctuation",
    "steam_fluctuation": "steam_fluctuation",
    "蒸汽波动": "steam_fluctuation",
    "analyzer_maintenance": "analyzer_maintenance",
    "分析仪维护": "analyzer_maintenance",
    "low_load": "low_load",
    "低负荷": "low_load",
    "full_load": "full_load",
    "满负荷": "full_load",
    "mode_switch": "mode_switch",
    "模式切换": "mode_switch",
    "batch_switch": "batch_switch",
    "批次切换": "batch_switch",
    "season_change": "season_change",
    "季节切换": "season_change",
}

CONTROL_OBJECT_LABELS = {
    "any": "通用控制对象",
    "top_temperature": "塔顶温度",
    "middle_temperature": "塔中温度",
    "bottom_temperature": "塔釜温度",
    "tower_pressure": "塔压",
    "reflux_flow": "回流流量",
    "steam_flow": "蒸汽流量",
    "feed_flow": "进料流量",
    "cooling_flow": "冷却流量",
    "side_draw_flow": "侧线抽出流量",
    "level": "液位",
    "composition": "产品组成",
    "outlet_temperature": "出口温度",
    "reactor_temperature": "反应釜温度",
    "main_steam_pressure": "主汽压力",
    "drum_level": "汽包液位",
    "network_pressure": "管网压力",
    "discharge_pressure": "出口压力",
    "supply_air_temperature": "送风温度",
    "static_pressure": "静压",
    "ph_value": "pH值",
    "motor_speed": "电机转速",
    "servo_position": "伺服位置",
    "web_tension": "纸幅张力",
    "dissolved_oxygen": "溶解氧",
    "humidity": "湿度",
}

CONTROL_OBJECT_ALIASES = {
    "any": "any",
    "通用控制对象": "any",
    "top_temperature": "top_temperature",
    "塔顶温度": "top_temperature",
    "middle_temperature": "middle_temperature",
    "塔中温度": "middle_temperature",
    "bottom_temperature": "bottom_temperature",
    "塔釜温度": "bottom_temperature",
    "tower_pressure": "tower_pressure",
    "塔压": "tower_pressure",
    "reflux_flow": "reflux_flow",
    "回流流量": "reflux_flow",
    "steam_flow": "steam_flow",
    "蒸汽流量": "steam_flow",
    "feed_flow": "feed_flow",
    "进料流量": "feed_flow",
    "cooling_flow": "cooling_flow",
    "冷却流量": "cooling_flow",
    "side_draw_flow": "side_draw_flow",
    "侧线抽出流量": "side_draw_flow",
    "level": "level",
    "液位": "level",
    "composition": "composition",
    "产品组成": "composition",
    "outlet_temperature": "outlet_temperature",
    "出口温度": "outlet_temperature",
    "reactor_temperature": "reactor_temperature",
    "反应釜温度": "reactor_temperature",
    "main_steam_pressure": "main_steam_pressure",
    "主汽压力": "main_steam_pressure",
    "drum_level": "drum_level",
    "汽包液位": "drum_level",
    "network_pressure": "network_pressure",
    "管网压力": "network_pressure",
    "discharge_pressure": "discharge_pressure",
    "出口压力": "discharge_pressure",
    "supply_air_temperature": "supply_air_temperature",
    "送风温度": "supply_air_temperature",
    "static_pressure": "static_pressure",
    "静压": "static_pressure",
    "ph_value": "ph_value",
    "ph值": "ph_value",
    "pH值": "ph_value",
    "motor_speed": "motor_speed",
    "电机转速": "motor_speed",
    "servo_position": "servo_position",
    "伺服位置": "servo_position",
    "web_tension": "web_tension",
    "纸幅张力": "web_tension",
    "张力": "web_tension",
    "dissolved_oxygen": "dissolved_oxygen",
    "溶解氧": "dissolved_oxygen",
    "humidity": "humidity",
    "湿度": "humidity",
}

LOOP_TYPE_ALIASES = {
    "flow": "flow",
    "flowrate": "flow",
    "流量": "flow",
    "temperature": "temperature",
    "temp": "temperature",
    "温度": "temperature",
    "pressure": "pressure",
    "press": "pressure",
    "压力": "pressure",
    "level": "level",
    "液位": "level",
    "composition": "composition",
    "成分": "composition",
    "ph": "ph",
    "speed": "speed",
    "转速": "speed",
    "position": "position",
    "位置": "position",
    "tension": "tension",
    "张力": "tension",
    "humidity": "humidity",
    "湿度": "humidity",
    "dissolved_oxygen": "dissolved_oxygen",
    "do": "dissolved_oxygen",
    "溶解氧": "dissolved_oxygen",
    "unknown": "unknown",
    "any": "any",
}

RISK_LABELS = {
    "high_noise_risk": "噪声偏高",
    "weak_excitation": "激励不足",
    "window_not_ready": "窗口可整定性不足",
    "candidate_window_interference": "候选窗口干扰风险",
    "saturation": "执行器饱和",
    "stiction": "阀门卡涩",
    "analyzer_delay": "分析仪滞后",
    "measurement_delay": "测量滞后",
    "slow_sampling": "采样周期长",
    "coupling": "回路耦合",
    "large_dead_time": "滞后偏大",
    "low_frequency_oscillation": "低频振荡风险",
    "thermal_inertia": "热惯性强",
    "process_variation": "工况变化",
    "deadband": "死区明显",
    "quantization": "分辨率不足",
    "nonlinear_gain": "非线性增益",
    "mechanical_resonance": "机械共振",
    "backlash": "齿隙",
    "inventory": "库存缓冲特性",
    "shrink_swell": "虚假液位效应",
    "inverse_response": "逆响应",
    "actuator_wear": "执行器磨损风险",
}

UNHELPFUL_GRAPH_PATTERNS = [
    "没有直接可用的信息",
    "信息非常有限",
    "具体规则有限",
    "未包含针对",
    "未包含",
    "未命中",
    "规则信息有限",
    "当前的这个问题",
]

SECTION_LABELS = {
    "top": "塔顶",
    "middle": "塔中",
    "bottom": "塔釜",
    "reflux": "回流段",
    "feed": "进料段",
}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = _as_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _is_unhelpful_graph_text(value: Any) -> bool:
    text = _as_text(value)
    if not text:
        return True
    return any(pattern in text for pattern in UNHELPFUL_GRAPH_PATTERNS)


def _to_list(values: Any) -> List[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    text = _as_text(values)
    return [text] if text else []


def _infer_flow_object(context: Mapping[str, Any]) -> str:
    text = " ".join(
        part.lower()
        for part in [
            _as_text(context.get("control_target")),
            _as_text(context.get("loop_name")),
            _as_text(context.get("tower_section")),
        ]
        if part
    )
    if not text:
        return ""
    if any(token in text for token in ["reflux", "回流"]):
        return "回流流量"
    if any(token in text for token in ["steam", "蒸汽", "reboiler", "再沸器"]):
        return "蒸汽流量"
    if any(token in text for token in ["feed", "进料"]):
        return "进料流量"
    if any(token in text for token in ["cool", "cooling", "冷却", "condenser", "冷凝器"]):
        return "冷却流量"
    if any(token in text for token in ["side", "侧线", "draw"]):
        return "侧线抽出流量"
    return ""


def normalize_loop_type(value: Any) -> str:
    text = _as_text(value).lower()
    if not text:
        return "unknown"
    return LOOP_TYPE_ALIASES.get(text, text if text in LOOP_TYPE_LABELS else "unknown")


def loop_type_display(loop_type: Any) -> str:
    return LOOP_TYPE_LABELS.get(normalize_loop_type(loop_type), LOOP_TYPE_LABELS["unknown"])


def _normalize_alias(value: Any, aliases: Mapping[str, str], default: str = "") -> str:
    text = _as_text(value)
    if not text:
        return default
    return aliases.get(text, aliases.get(text.lower(), default or text))


def plant_type_display(plant_type: Any) -> str:
    normalized = _normalize_alias(plant_type, PLANT_TYPE_ALIASES, "distillation_column")
    return PLANT_TYPE_LABELS.get(normalized, _as_text(plant_type) or PLANT_TYPE_LABELS["distillation_column"])


def industry_type_display(industry_type: Any) -> str:
    normalized = _normalize_alias(industry_type, INDUSTRY_ALIASES, "")
    return INDUSTRY_LABELS.get(normalized, _as_text(industry_type))


def scenario_display(scenario: Any) -> str:
    normalized = _normalize_alias(scenario, SCENARIO_ALIASES, "")
    return SCENARIO_LABELS.get(normalized, _as_text(scenario))


def control_object_display(control_object: Any) -> str:
    normalized = _normalize_alias(control_object, CONTROL_OBJECT_ALIASES, "")
    return CONTROL_OBJECT_LABELS.get(normalized, _as_text(control_object))


def load_pid_rules(paths: Iterable[str | Path] | None = None) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    rule_paths = [Path(path) for path in paths] if paths is not None else RULES_PATHS
    for rules_path in rule_paths:
        if not rules_path.exists():
            continue
        for line in rules_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rules.append(item)
    return rules


def load_distillation_rules(path: str | Path | None = None) -> List[Dict[str, Any]]:
    if path is not None:
        return load_pid_rules([path])
    return load_pid_rules()


def build_knowledge_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    loop_type = normalize_loop_type(context.get("loop_type"))
    tower_section = _as_text(context.get("tower_section")).lower()
    selected_model_params = context.get("selected_model_params")
    plant_type = _normalize_alias(context.get("plant_type"), PLANT_TYPE_ALIASES, "distillation_column")
    industry_type = _normalize_alias(context.get("industry_type"), INDUSTRY_ALIASES, "")
    scenario = _normalize_alias(context.get("scenario"), SCENARIO_ALIASES, "")
    control_object = _normalize_alias(context.get("control_object"), CONTROL_OBJECT_ALIASES, "")
    return {
        "industry_type": industry_type,
        "industry_type_display": industry_type_display(industry_type),
        "plant_type": plant_type,
        "plant_type_display": plant_type_display(plant_type),
        "loop_type": loop_type,
        "loop_type_display": loop_type_display(loop_type),
        "loop_name": _as_text(context.get("loop_name")),
        "tower_section": tower_section,
        "tower_section_display": SECTION_LABELS.get(tower_section, ""),
        "scenario": scenario,
        "scenario_display": scenario_display(scenario),
        "control_object": control_object,
        "control_object_display": control_object_display(control_object),
        "control_target": _as_text(context.get("control_target")),
        "model_type": _as_text(context.get("model_type")).upper(),
        "selected_model_params": dict(selected_model_params) if isinstance(selected_model_params, dict) else {},
        "window_readiness": _as_text(context.get("window_readiness")).lower(),
        "identification_reliability": _as_text(context.get("identification_reliability")).lower(),
        "cross_window_consistency": _as_text(context.get("cross_window_consistency")).lower(),
        "risk_tags": [str(item).strip() for item in _to_list(context.get("risk_tags") or context.get("risks"))],
    }


def build_knowledge_questions(context: Mapping[str, Any]) -> List[str]:
    normalized = build_knowledge_context(context)
    plant_label = normalized["plant_type_display"]
    industry_label = normalized["industry_type_display"]
    scenario_label = normalized["scenario_display"] or "当前工况"
    control_object_label = normalized["control_object_display"]
    loop_label = normalized["loop_type_display"]
    model_type = normalized["model_type"]
    risk_texts = [RISK_LABELS.get(tag, tag) for tag in normalized["risk_tags"]]
    flow_object = _infer_flow_object(context) if normalized["loop_type"] == "flow" else ""
    control_subject = control_object_label or flow_object or f"{loop_label}回路"
    subject_prefix = f"{industry_label}行业的{plant_label}" if industry_label else plant_label

    questions = []
    if risk_texts:
        questions.append(
            f"{subject_prefix}的{control_subject}在{scenario_label}和{model_type or '当前'}模型下，存在{'、'.join(risk_texts[:2])}时的整定策略有哪些？"
        )
    if model_type:
        questions.append(f"{subject_prefix}的{control_subject}在{scenario_label}和{model_type}模型下的整定策略有哪些？")
    questions.append(f"{subject_prefix}的{control_subject}在{scenario_label}下的PID整定有哪些规则？")

    deduped: List[str] = []
    for question in questions:
        if question and question not in deduped:
            deduped.append(question)
    return deduped[:3]


def build_graph_query_payloads(
    context: Mapping[str, Any],
    *,
    graph_id: str,
    query_mode: str = "local",
    response_type: str = "要点式，尽量精炼",
    include_context: bool = True,
) -> List[Dict[str, Any]]:
    return [
        {
            "graph_id": graph_id,
            "question": question,
            "query_mode": query_mode,
            "response_type": response_type,
            "include_context": include_context,
        }
        for question in build_knowledge_questions(context)
    ]


def search_distillation_rules(
    context: Mapping[str, Any],
    *,
    rules: List[Dict[str, Any]] | None = None,
    limit: int = 8,
) -> Dict[str, Any]:
    normalized = build_knowledge_context(context)
    rules = rules if rules is not None else load_distillation_rules()

    scored: List[tuple[float, Dict[str, Any], List[str]]] = []
    for rule in rules:
        score = 0.0
        reasons: List[str] = []

        rule_plant = _normalize_alias(rule.get("plant_type"), PLANT_TYPE_ALIASES, "any")
        if rule_plant == normalized["plant_type"]:
            score += 2.6
            reasons.append("设备类型匹配")
        elif rule_plant == "any":
            score += 0.6

        rule_industries = {
            _normalize_alias(item, INDUSTRY_ALIASES, "")
            for item in _to_list(rule.get("industry_tags"))
            if _normalize_alias(item, INDUSTRY_ALIASES, "")
        }
        if normalized["industry_type"] and normalized["industry_type"] in rule_industries:
            score += 1.6
            reasons.append("行业匹配")

        rule_loop = normalize_loop_type(rule.get("loop_type"))
        if rule_loop == normalized["loop_type"]:
            score += 3.0
            reasons.append("回路类型匹配")
        elif rule_loop in {"unknown", "any"}:
            score += 0.8

        rule_models = {str(item).upper() for item in _to_list(rule.get("model_types"))}
        if normalized["model_type"] and normalized["model_type"] in rule_models:
            score += 2.4
            reasons.append("模型类型匹配")

        rule_target = _normalize_alias(rule.get("control_target"), CONTROL_OBJECT_ALIASES, _as_text(rule.get("control_target")).lower())
        if normalized["control_object"] and normalized["control_object"] == rule_target:
            score += 2.0
            reasons.append("控制对象匹配")
        elif normalized["control_target"] and normalized["control_target"].lower() in _as_text(rule.get("control_target")).lower():
            score += 1.6
            reasons.append("控制对象匹配")

        rule_scenarios = {str(item).lower() for item in _to_list(rule.get("scenario_tags"))}
        if normalized["scenario"] and normalized["scenario"] in rule_scenarios:
            score += 1.8
            reasons.append("工况匹配")

        rule_risks = {str(item).lower() for item in _to_list(rule.get("risk_tags"))}
        overlap = sorted(set(tag.lower() for tag in normalized["risk_tags"]) & rule_risks)
        if overlap:
            score += min(2.0, 0.8 * len(overlap))
            reasons.append("风险标签匹配")

        if normalized["tower_section_display"]:
            title = _as_text(rule.get("title"))
            summary = _as_text(rule.get("summary"))
            if normalized["tower_section_display"] in title or normalized["tower_section_display"] in summary:
                score += 1.0
                reasons.append("塔段语义匹配")

        if score > 0:
            scored.append((score, rule, reasons))

    scored.sort(key=lambda item: item[0], reverse=True)
    matched_rules: List[Dict[str, Any]] = []
    strategy_counter: Counter[str] = Counter()
    risk_hints: List[str] = []
    for score, rule, reasons in scored[:limit]:
        preferred = _to_list(rule.get("preferred_strategy"))
        discouraged = _to_list(rule.get("discouraged_strategy"))
        matched_rules.append(
            {
                "rule_id": rule.get("rule_id"),
                "knowledge_type": _as_text(rule.get("knowledge_type")),
                "title": _as_text(rule.get("title")),
                "summary": _as_text(rule.get("summary")),
                "recommendation_level": _as_text(rule.get("recommendation_level")),
                "preferred_strategy": preferred,
                "discouraged_strategy": discouraged,
                "suggested_actions": _to_list(rule.get("suggested_actions")),
                "rollback_advice": _as_text(rule.get("rollback_advice")),
                "operator_note": _as_text(rule.get("operator_note")),
                "match_score": round(score, 3),
                "match_reasons": reasons,
                "source": _as_text(rule.get("source")) or "distillation_expert_graph",
            }
        )
        for strategy in preferred:
            strategy_counter[strategy.upper()] += 1
        summary = _as_text(rule.get("summary"))
        if summary and summary not in risk_hints:
            risk_hints.append(summary)

    preferred_strategy = strategy_counter.most_common(1)[0][0] if strategy_counter else ""
    summary_text = ""
    if matched_rules:
        top_summaries = []
        for item in matched_rules[:2]:
            summary = _as_text(item.get("summary"))
            if summary and summary not in top_summaries:
                top_summaries.append(summary)
        summary_text = "；".join(top_summaries)
    return {
        "matched": bool(matched_rules),
        "matched_count": len(matched_rules),
        "matched_rules": matched_rules,
        "preferred_strategy": preferred_strategy,
        "risk_hints": risk_hints[:3],
        "questions": build_knowledge_questions(normalized),
        "summary": summary_text,
        "knowledge_context": normalized,
    }


def normalize_graph_answers(raw_answers: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    answers: List[Dict[str, Any]] = []
    key_points: List[str] = []
    for item in raw_answers:
        question = _truncate_text(item.get("question"), 120)
        response = item.get("response")
        answers.append({"question": question, "response": response})

        if isinstance(response, dict):
            for key in ("answer", "summary", "content", "result"):
                text = _truncate_text(response.get(key), 300)
                if text and not _is_unhelpful_graph_text(text):
                    key_points.append(text)
                    break
        else:
            text = _truncate_text(response, 300)
            if text and not _is_unhelpful_graph_text(text):
                key_points.append(text)

    return {
        "answers": answers[:3],
        "graph_hints": key_points[:3],
        "graph_summary": "；".join(key_points[:2]) if key_points else "",
    }


def merge_knowledge_guidance(
    *,
    local_guidance: Mapping[str, Any] | None = None,
    graph_guidance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    local_guidance = dict(local_guidance or {})
    graph_guidance = dict(graph_guidance or {})

    matched_rules = list(local_guidance.get("matched_rules") or [])
    preferred_strategy = _as_text(local_guidance.get("preferred_strategy")).upper()
    risk_hints = list(local_guidance.get("risk_hints") or [])
    summary_parts = []
    local_summary = _as_text(local_guidance.get("summary"))
    if local_summary and not _is_unhelpful_graph_text(local_summary):
        summary_parts.append(local_summary)

    graph_hints = list(graph_guidance.get("graph_hints") or [])
    for hint in graph_hints:
        if hint and not _is_unhelpful_graph_text(hint) and hint not in risk_hints:
            risk_hints.append(hint)

    graph_summary = _as_text(graph_guidance.get("graph_summary"))
    if graph_summary and not _is_unhelpful_graph_text(graph_summary):
        summary_parts.append(graph_summary)

    recommended_level = _as_text(matched_rules[0].get("recommendation_level")) if matched_rules else ""

    strategy_counter: Counter[str] = Counter()
    for item in matched_rules:
        for strategy in _to_list(item.get("preferred_strategy")):
            strategy_counter[strategy.upper()] += 1
    if not preferred_strategy and strategy_counter:
        preferred_strategy = strategy_counter.most_common(1)[0][0]

    constraint_items = [
        {
            "rule_id": item.get("rule_id"),
            "title": _as_text(item.get("title")),
            "summary": _as_text(item.get("summary")),
        }
        for item in matched_rules
        if _as_text(item.get("knowledge_type")).lower() == "constraint"
    ]

    discourage_derivative = any(
        "strong_derivative" in _to_list(item.get("discouraged_strategy")) for item in matched_rules
    )
    avoid_aggressive = any(
        marker in _to_list(item.get("discouraged_strategy"))
        for item in matched_rules
        for marker in ("aggressive_pid", "over_aggressive")
    )
    conservative_mode = bool(constraint_items) or any(
        marker in _to_list(item.get("suggested_actions"))
        for item in matched_rules
        for marker in (
            "prefer_robust_parameters",
            "temporary_use_only",
            "use_model_as_reference",
            "validate_in_shadow_mode",
        )
    )

    tuning_bias = {
        "preferred_strategy": preferred_strategy,
        "discourage_derivative": discourage_derivative,
        "avoid_aggressive_strategies": avoid_aggressive,
        "conservative_mode": conservative_mode,
        "kp_scale_max": 0.8 if any("reduce_kp" in _to_list(item.get("suggested_actions")) for item in matched_rules) else 1.0,
        "ki_scale_max": 0.85 if any("reduce_ki" in _to_list(item.get("suggested_actions")) for item in matched_rules) else 1.0,
        "kd_scale_max": 0.4 if any("reduce_kd" in _to_list(item.get("suggested_actions")) for item in matched_rules) else (0.25 if discourage_derivative else 1.0),
    }

    return {
        "matched": bool(matched_rules or graph_guidance.get("answers")),
        "questions": list(local_guidance.get("questions") or []),
        "matched_count": len(matched_rules),
        "matched_rules": matched_rules,
        "constraints": constraint_items,
        "preferred_strategy": preferred_strategy,
        "risk_hints": risk_hints[:5],
        "recommendation_level": recommended_level,
        "tuning_bias": tuning_bias,
        "graph_answers": list(graph_guidance.get("answers") or []),
        "summary": "；".join(part for part in summary_parts if part).strip("； "),
        "knowledge_context": dict(local_guidance.get("knowledge_context") or {}),
    }


def compact_knowledge_guidance(guidance: Mapping[str, Any] | None) -> Dict[str, Any]:
    guidance = dict(guidance or {})
    compact_rules = []
    for item in list(guidance.get("matched_rules") or [])[:3]:
        compact_rules.append(
            {
                "rule_id": item.get("rule_id"),
                "knowledge_type": item.get("knowledge_type"),
                "title": _truncate_text(item.get("title"), 80),
                "summary": _truncate_text(item.get("summary"), 120),
                "match_score": item.get("match_score"),
            }
        )

    compact_answers = []
    for item in list(guidance.get("graph_answers") or [])[:2]:
        response = item.get("response")
        if isinstance(response, dict):
            response = response.get("answer") or response.get("summary") or response.get("content") or response.get("result")
        compact_answers.append(
            {
                "question": _truncate_text(item.get("question"), 120),
                "response": _truncate_text(response, 180),
            }
        )

    return {
        "matched": bool(guidance.get("matched")),
        "matched_count": int(guidance.get("matched_count") or 0),
        "preferred_strategy": _as_text(guidance.get("preferred_strategy")).upper(),
        "recommendation_level": _as_text(guidance.get("recommendation_level")),
        "summary": _truncate_text(guidance.get("summary"), 180),
        "risk_hints": [_truncate_text(item, 100) for item in list(guidance.get("risk_hints") or [])[:3]],
        "constraints": [
            {
                "rule_id": item.get("rule_id"),
                "title": _truncate_text(item.get("title"), 80),
                "summary": _truncate_text(item.get("summary"), 120),
            }
            for item in list(guidance.get("constraints") or [])[:3]
        ],
        "questions": [_truncate_text(item, 100) for item in list(guidance.get("questions") or [])[:3]],
        "matched_rules": compact_rules,
        "graph_answers": compact_answers,
        "tuning_bias": dict(guidance.get("tuning_bias") or {}),
    }


def query_knowledge_graph_api(
    *,
    base_url: str,
    graph_id: str,
    context: Mapping[str, Any],
    query_mode: str = "local",
    response_type: str = "要点式，尽量精炼",
    include_context: bool = True,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    payloads = build_graph_query_payloads(
        context,
        graph_id=graph_id,
        query_mode=query_mode,
        response_type=response_type,
        include_context=include_context,
    )
    answers: List[Dict[str, Any]] = []
    for payload in payloads:
        response = requests.post(base_url, json=payload, timeout=timeout)
        response.raise_for_status()
        answers.append({"question": payload["question"], "response": response.json()})
    return normalize_graph_answers(answers)
