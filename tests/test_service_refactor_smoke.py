import unittest
import sys
from pathlib import Path
import asyncio
import tempfile
import shutil
import gc

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
for path in (str(ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.services.data_service import build_window_overview
from backend.services.identification_service import (
    derive_model_reason_codes,
    derive_next_actions,
    fit_best_fopdt_window,
)
from backend.services.pid_evaluation_service import build_initial_assessment, diagnose_evaluation_failure
from backend.services.pid_tuning_service import benchmark_pid_strategies, refine_pid_for_performance
from backend.memory import experience_store
from backend.memory.experience_service import build_experience_record, retrieve_experience_guidance
from backend.orchestration.workflow_runner import run_multi_agent_collaboration


class ServiceRefactorSmokeTests(unittest.TestCase):
    def test_build_window_overview_marks_selected_region(self) -> None:
        df = pd.DataFrame(
            {
                "PV": [1.0, 1.5, 2.0, 2.5, 3.0],
                "MV": [10.0, 11.0, 12.0, 13.0, 14.0],
                "timestamp": pd.date_range("2026-03-01 00:00:00", periods=5, freq="s"),
            }
        )
        overview = build_window_overview(df, {"start_index": 1, "end_index": 3}, max_points=10)
        self.assertEqual(overview["x_axis"], "timestamp")
        self.assertEqual(overview["window_start"], 1)
        self.assertEqual(overview["window_end"], 3)
        in_window = [point["index"] for point in overview["points"] if point["in_window"]]
        self.assertEqual(in_window, [1, 2, 3])

    def test_identification_reason_codes_and_actions(self) -> None:
        reason_codes = derive_model_reason_codes(
            model_params={"normalized_rmse": 0.12, "r2_score": 0.52, "T": 1.0, "L": 0.0},
            confidence={"confidence": 0.48},
            quality_metrics={"overshoot_percent": 25.0, "settling_time": 2.0},
        )
        self.assertIn("残差偏高", reason_codes)
        self.assertIn("拟合解释度偏低", reason_codes)
        self.assertIn("模型置信度偏低", reason_codes)
        actions = derive_next_actions(0.48, reason_codes)
        self.assertIn("尝试其他辨识窗口", actions)
        self.assertIn("确认对象是否偏离当前模型假设", actions)
        self.assertIn("检查采样周期或补采更高频数据", actions)

    def test_identification_service_returns_selected_model_type(self) -> None:
        df = pd.DataFrame(
            {
                "MV": [0.0] * 20 + [1.0] * 80,
                "PV": [0.0] * 20
                + [0.10, 0.18, 0.25, 0.31, 0.36, 0.40, 0.43, 0.46, 0.48, 0.50]
                + [0.50] * 70,
            }
        )
        result = fit_best_fopdt_window(
            cleaned_df=df,
            candidate_windows=[],
            quality_metrics={},
            actual_dt=1.0,
            benchmark_fn=lambda K, T, L, dt, confidence: {
                "best": {"strategy": "IMC", "performance_score": 6.0, "final_rating": 6.5, "is_stable": True}
            },
            loop_type="flow",
        )
        self.assertIn(result["selected_model_type"], {"FO", "FOPDT", "IPDT"})
        self.assertIn("K", result["tuning_model"])
        self.assertIn("T", result["tuning_model"])
        self.assertIn("L", result["tuning_model"])

    def test_temperature_loop_can_try_sopdt(self) -> None:
        df = pd.DataFrame(
            {
                "MV": [0.0] * 10 + [1.0] * 110,
                "PV": [0.0] * 10
                + [0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.21, 0.27, 0.33, 0.40]
                + [0.48, 0.55, 0.61, 0.66, 0.70, 0.74, 0.77, 0.80, 0.83, 0.85]
                + [0.87] * 90,
            }
        )
        result = fit_best_fopdt_window(
            cleaned_df=df,
            candidate_windows=[],
            quality_metrics={},
            actual_dt=1.0,
            benchmark_fn=lambda K, T, L, dt, confidence: {
                "best": {"strategy": "LAMBDA", "performance_score": 6.0, "final_rating": 6.5, "is_stable": True}
            },
            loop_type="temperature",
        )
        self.assertIn(result["selected_model_type"], {"SOPDT", "FOPDT", "FO", "IPDT"})

    def test_pid_benchmark_and_refine_return_structured_results(self) -> None:
        benchmark = benchmark_pid_strategies(K=0.45, T=1.95, L=0.0, dt=1.0, confidence_score=0.8)
        self.assertIn("best", benchmark)
        self.assertTrue(benchmark["all"])
        self.assertIn("performance_score", benchmark["best"])

        refined = refine_pid_for_performance(
            model_params={"K": 0.45, "T1": 1.95, "T2": 0.0, "L": 0.0},
            base_pid_params={"Kp": 2.18, "Ki": 1.12, "Kd": 0.0},
            method_confidence=0.8,
            dt=1.0,
            base_strategy="LAMBDA",
        )
        self.assertIn("best", refined)
        self.assertTrue(refined["candidates"])
        self.assertIn("final_rating", refined["best"])

    def test_evaluation_diagnosis_and_assessment(self) -> None:
        eval_result = {
            "performance_score": 5.82,
            "method_confidence": 0.80,
            "final_rating": 6.48,
            "performance_details": {
                "overshoot": 120.0,
                "settling_time": -1,
                "steady_state_error": 12.0,
                "oscillation_count": 80,
                "decay_ratio": 1.1,
                "is_stable": False,
            },
        }
        diagnosis = diagnose_evaluation_failure(
            eval_result=eval_result,
            model_r2=0.99,
            model_rmse=0.08,
            candidate_window_count=3,
        )
        self.assertEqual(diagnosis["feedback_target"], "pid_expert")

        assessment = build_initial_assessment(
            eval_result=eval_result,
            pass_threshold=7.0,
            diagnosis=diagnosis,
            evaluated_pid={"Kp": 2.18, "Ki": 1.12, "Kd": 0.0},
        )
        self.assertFalse(assessment["passed"])
        self.assertEqual(assessment["feedback_target"], "pid_expert")

    def test_workflow_runner_emits_user_result_done(self) -> None:
        class DummyTeam:
            def __init__(self, *args, **kwargs):
                pass

            async def run_stream(self, *args, **kwargs):
                if False:
                    yield None
                return

        async def collect() -> list[dict]:
            from backend.orchestration import workflow_runner as wr

            original_team = wr.RoundRobinGroupChat
            wr.RoundRobinGroupChat = DummyTeam
            try:
                shared_store = {
                    "selected_pid_params": {
                        "Kp": 1.0,
                        "Ki": 0.5,
                        "Kd": 0.0,
                        "Ti": 2.0,
                        "Td": 0.0,
                        "strategy": "IMC",
                        "description": "test",
                    }
                }

                async def fake_iter():
                    events = []
                    async for event in run_multi_agent_collaboration(
                        csv_path="demo.csv",
                        loop_name="FIC_101A",
                        loop_type="flow",
                        loop_uri="/pid/demo",
                        start_time="1",
                        end_time="2",
                        data_type="interpolated",
                        llm_config={"api_key": "k", "base_url": "u", "model": "m"},
                        shared_data_store=shared_store,
                        create_model_client=lambda **kwargs: object(),
                        create_pid_agents=lambda **kwargs: [],
                        finalize_agent_turn=lambda turn: turn,
                        build_feedback_turns=lambda shared: [],
                        build_experience_record=lambda **kwargs: {
                            "experience_id": "exp_test",
                            "loop_type": kwargs["loop_type"],
                            "evaluation": {"final_rating": 8.0, "passed": True},
                        },
                        persist_experience_record=lambda record: record["experience_id"],
                        to_jsonable=lambda value: value,
                    ):
                        events.append(event)
                    return events

                return await fake_iter()
            finally:
                wr.RoundRobinGroupChat = original_team

        events = asyncio.run(collect())
        self.assertEqual(events[0]["type"], "user")
        self.assertEqual(events[-2]["type"], "result")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-2]["data"]["memory"]["experienceId"], "exp_test")

    def test_experience_guidance_prefers_successful_similar_strategy(self) -> None:
        old_root = experience_store.MEMORY_ROOT
        old_file = experience_store.EXPERIENCE_FILE
        old_index = experience_store.INDEX_FILE
        tmpdir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmpdir)
            experience_store.MEMORY_ROOT = tmp_path
            experience_store.EXPERIENCE_FILE = tmp_path / "pid_experiences.jsonl"
            experience_store.INDEX_FILE = tmp_path / "pid_experiences.db"

            record = build_experience_record(
                loop_name="FIC_101A",
                loop_type="flow",
                loop_uri="/pid/demo",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {"K": 0.45, "T": 2.0, "L": 0.0, "normalizedRmse": 0.08, "r2Score": 0.99, "confidence": 0.8},
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 1.5, "Ki": 0.5, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 9.2,
                        "method_confidence": 0.8,
                        "final_rating": 8.9,
                        "passed": True,
                        "failure_reason": "",
                        "feedback_target": "",
                        "initial_assessment": {"evaluated_pid": {"Kp": 2.0, "Ki": 1.0, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 180, "stepEvents": 3},
                },
            )
            experience_store.append_experience_record(record)
            guidance = retrieve_experience_guidance(
                loop_type="flow",
                model_type="FOPDT",
                K=0.46,
                T=1.95,
                L=0.0,
                limit=3,
                candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
            )
            self.assertEqual(guidance["preferred_strategy"], "LAMBDA")
            self.assertTrue(guidance["matches"])
            self.assertEqual(guidance["summary"]["preferred_refine_pattern"], "tighten_kp+tighten_ki")
            self.assertAlmostEqual(guidance["summary"]["recommended_kp_scale"], 0.75, places=2)
            self.assertAlmostEqual(guidance["summary"]["recommended_ki_scale"], 0.5, places=2)
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_experience_guidance_prefers_same_model_type(self) -> None:
        old_root = experience_store.MEMORY_ROOT
        old_file = experience_store.EXPERIENCE_FILE
        old_index = experience_store.INDEX_FILE
        tmpdir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmpdir)
            experience_store.MEMORY_ROOT = tmp_path
            experience_store.EXPERIENCE_FILE = tmp_path / "pid_experiences.jsonl"
            experience_store.INDEX_FILE = tmp_path / "pid_experiences.db"

            base_final_result = {
                "model": {
                    "modelType": "IPDT",
                    "selectedModelParams": {"K": 0.2, "L": 3.0},
                    "K": 0.2,
                    "T": 20.0,
                    "L": 3.0,
                    "normalizedRmse": 0.07,
                    "r2Score": 0.95,
                    "confidence": 0.82,
                },
                "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 0.9, "Ki": 0.1, "Kd": 0.0},
                "evaluation": {
                    "performance_score": 8.5,
                    "method_confidence": 0.82,
                    "final_rating": 8.1,
                    "passed": True,
                    "failure_reason": "",
                    "feedback_target": "",
                    "initial_assessment": {"evaluated_pid": {"Kp": 1.0, "Ki": 0.2, "Kd": 0.0}},
                    "auto_refine_result": {"applied": True},
                },
                "dataAnalysis": {"windowPoints": 200, "stepEvents": 2},
            }
            record = build_experience_record(
                loop_name="LIC_101A",
                loop_type="level",
                loop_uri="/pid/demo2",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result=base_final_result,
            )
            experience_store.append_experience_record(record)
            guidance = retrieve_experience_guidance(
                loop_type="level",
                model_type="IPDT",
                K=0.21,
                T=22.0,
                L=3.0,
                limit=3,
                candidate_strategies=["IMC", "LAMBDA"],
            )
            self.assertTrue(guidance["matches"])
            self.assertEqual(guidance["matches"][0]["model_type"], "IPDT")
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clear_experience_store_removes_records(self) -> None:
        old_root = experience_store.MEMORY_ROOT
        old_file = experience_store.EXPERIENCE_FILE
        old_index = experience_store.INDEX_FILE
        tmpdir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmpdir)
            experience_store.MEMORY_ROOT = tmp_path
            experience_store.EXPERIENCE_FILE = tmp_path / "pid_experiences.jsonl"
            experience_store.INDEX_FILE = tmp_path / "pid_experiences.db"

            record = build_experience_record(
                loop_name="FIC_101A",
                loop_type="flow",
                loop_uri="/pid/demo",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {"K": 0.45, "T": 2.0, "L": 0.0, "normalizedRmse": 0.08, "r2Score": 0.99, "confidence": 0.8},
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 1.5, "Ki": 0.5, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 9.2,
                        "method_confidence": 0.8,
                        "final_rating": 8.9,
                        "passed": True,
                        "failure_reason": "",
                        "feedback_target": "",
                        "initial_assessment": {"evaluated_pid": {"Kp": 2.0, "Ki": 1.0, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 180, "stepEvents": 3},
                },
            )
            experience_store.append_experience_record(record)
            cleared = experience_store.clear_experience_store()
            self.assertTrue(cleared["cleared"])
            self.assertEqual(cleared["before"]["total_count"], 1)
            self.assertEqual(cleared["after"]["total_count"], 0)
            self.assertEqual(experience_store.load_experience_records(), [])
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
