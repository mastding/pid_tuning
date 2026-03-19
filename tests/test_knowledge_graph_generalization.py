import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
for path in (str(ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.services.knowledge_graph_service import build_knowledge_context, build_knowledge_questions, search_distillation_rules


class KnowledgeGraphGeneralizationTests(unittest.TestCase):
    def test_build_knowledge_context_normalizes_industry_and_equipment_entities(self) -> None:
        context = build_knowledge_context(
            {
                "industry_type": "power",
                "plant_type": "heat_exchanger",
                "loop_type": "temperature",
                "control_object": "outlet_temperature",
            }
        )
        self.assertEqual(context["industry_type"], "power")
        self.assertEqual(context["plant_type"], "heat_exchanger")
        self.assertEqual(context["control_object"], "outlet_temperature")

    def test_build_knowledge_questions_include_industry_and_equipment_labels(self) -> None:
        questions = build_knowledge_questions(
            {
                "industry_type": "power",
                "plant_type": "heat_exchanger",
                "scenario": "load_change",
                "loop_type": "temperature",
                "control_object": "outlet_temperature",
                "model_type": "FOPDT",
            }
        )
        self.assertTrue(any("电力行业的换热器的出口温度" in item for item in questions))
        self.assertTrue(any("FOPDT" in item for item in questions))

    def test_search_distillation_rules_matches_generic_pid_rules(self) -> None:
        result = search_distillation_rules(
            {
                "industry_type": "water_treatment",
                "plant_type": "ph_neutralization",
                "loop_type": "ph",
                "control_object": "ph_value",
                "model_type": "SOPDT",
                "risk_tags": ["measurement_delay"],
            }
        )
        self.assertTrue(result["matched"])
        self.assertGreaterEqual(result["matched_count"], 1)
        self.assertTrue(any(item["source"] == "generic_pid_expert_graph" for item in result["matched_rules"]))
        self.assertTrue(any("pH" in item["title"] for item in result["matched_rules"]))


if __name__ == "__main__":
    unittest.main()
