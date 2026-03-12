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
from backend.services.pid_evaluation_service import build_initial_assessment, diagnose_evaluation_failure, evaluate_pid_model
from backend.services.pid_tuning_service import benchmark_pid_strategies, refine_pid_for_performance, select_best_pid_strategy
from backend.services.tool_adapter_service import tune_pid_tool
from backend.skills.pid_tuning_skills import apply_tuning_rules, select_tuning_strategy
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
        self.assertIn("K", result["model_params"])
        self.assertIn("normalized_rmse", result["model_params"])

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

    def test_ipdt_evaluation_uses_integrating_process_model(self) -> None:
        result = evaluate_pid_model(
            K=0.12,
            T=0.0,
            L=2.0,
            Kp=1.2,
            Ki=0.08,
            Kd=0.0,
            method="lambda",
            method_confidence=0.75,
            model_confidence={"quality": "good", "recommendation": ""},
            dt=1.0,
            model_type="IPDT",
            selected_model_params={"model_type": "IPDT", "K": 0.12, "L": 2.0},
        )
        self.assertIn("simulation", result)
        self.assertIn("performance_score", result)
        self.assertIsInstance(result["simulation"]["pv_history"], list)

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
                        register_experience_reuse=lambda *args, **kwargs: {"updated": 0},
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
            self.assertIn("selected_model_params", record["model"])
            experience_store.append_experience_record(record)
            guidance = retrieve_experience_guidance(
                loop_type="flow",
                model_type="FOPDT",
                K=0.46,
                T=1.95,
                L=0.0,
                selected_model_params={"model_type": "FOPDT", "K": 0.46, "T": 1.95, "L": 0.0},
                limit=3,
                candidate_strategies=["IMC", "LAMBDA", "ZN", "CHR"],
            )
            self.assertEqual(guidance["preferred_strategy"], "LAMBDA")
            self.assertTrue(guidance["matches"])
            self.assertEqual(guidance["preferred_model_type"], "FOPDT")
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
                selected_model_params={"model_type": "IPDT", "K": 0.21, "L": 3.0},
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

    def test_experience_guidance_prefers_matching_raw_sopdt_model(self) -> None:
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
                loop_name="TIC_201",
                loop_type="temperature",
                loop_uri="/pid/temp",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {
                        "modelType": "SOPDT",
                        "selectedModelParams": {"model_type": "SOPDT", "K": 1.2, "T1": 15.0, "T2": 30.0, "L": 5.0},
                        "K": 1.2,
                        "T": 45.0,
                        "L": 5.0,
                        "normalizedRmse": 0.06,
                        "r2Score": 0.99,
                        "confidence": 0.82,
                    },
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 1.0, "Ki": 0.1, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 8.8,
                        "method_confidence": 0.82,
                        "final_rating": 8.2,
                        "passed": True,
                        "failure_reason": "",
                        "feedback_target": "",
                        "initial_assessment": {"evaluated_pid": {"Kp": 1.2, "Ki": 0.15, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 200, "stepEvents": 2},
                },
            )
            experience_store.append_experience_record(record)
            guidance = retrieve_experience_guidance(
                loop_type="temperature",
                model_type="SOPDT",
                K=1.18,
                T=45.0,
                L=5.2,
                selected_model_params={"model_type": "SOPDT", "K": 1.18, "T1": 14.5, "T2": 31.0, "L": 5.2},
                limit=3,
                candidate_strategies=["IMC", "LAMBDA"],
            )
            self.assertTrue(guidance["matches"])
            self.assertEqual(guidance["matches"][0]["model_type"], "SOPDT")
            self.assertEqual(guidance["preferred_strategy"], "LAMBDA")
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

    def test_register_experience_references_updates_reuse_counters(self) -> None:
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
                loop_name="TIC_301",
                loop_type="temperature",
                loop_uri="/pid/tic301",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {
                        "modelType": "SOPDT",
                        "selectedModelParams": {"model_type": "SOPDT", "K": 1.1, "T1": 12.0, "T2": 24.0, "L": 4.0},
                        "K": 1.1,
                        "T": 36.0,
                        "L": 4.0,
                        "normalizedRmse": 0.05,
                        "r2Score": 0.99,
                        "confidence": 0.86,
                    },
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 0.9, "Ki": 0.08, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 8.9,
                        "method_confidence": 0.86,
                        "final_rating": 8.4,
                        "passed": True,
                        "failure_reason": "",
                        "feedback_target": "",
                        "initial_assessment": {"evaluated_pid": {"Kp": 1.1, "Ki": 0.12, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 220, "stepEvents": 2},
                },
            )
            exp_id = experience_store.append_experience_record(record)
            summary = experience_store.register_experience_references(
                [exp_id],
                hit_time="2026-03-12T10:00:00+08:00",
                follow_up_passed=True,
                follow_up_final_rating=8.8,
            )
            self.assertEqual(summary["updated"], 1)

            listed = experience_store.list_experiences(limit=5)
            self.assertEqual(listed[0]["reuse"]["hit_count"], 1)
            self.assertEqual(listed[0]["reuse"]["follow_up_success_count"], 1)
            self.assertEqual(listed[0]["reuse"]["last_hit_at"], "2026-03-12T10:00:00+08:00")

            detail = experience_store.get_experience_detail(exp_id)
            self.assertEqual(detail["reuse"]["hit_count"], 1)
            self.assertTrue(detail["reuse"]["last_follow_up_passed"])
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_select_best_pid_strategy_can_emit_history_seeded_candidate(self) -> None:
        guidance = {
            "preferred_strategy": "LAMBDA",
            "matches": [
                {
                    "experience_id": "exp_seed",
                    "model_type": "SOPDT",
                    "model": {"selected_model_params": {"model_type": "SOPDT", "K": 1.0, "T1": 10.0, "T2": 20.0, "L": 4.0}},
                    "pid": {"final": {"Kp": 0.8, "Ki": 0.06, "Kd": 0.0}},
                    "evaluation": {"final_rating": 8.5, "performance_score": 8.7, "passed": True},
                    "similarity_score": 12.0,
                }
            ],
            "summary": {
                "preferred_strategy": "LAMBDA",
                "preferred_model_type": "SOPDT",
                "recommended_kp_scale": 0.85,
                "recommended_ki_scale": 0.8,
                "recommended_kd_scale": 1.0,
                "preferred_refine_pattern": "tighten_kp+tighten_ki",
            },
        }
        result = select_best_pid_strategy(
            K=1.02,
            T=30.0,
            L=4.2,
            loop_type="temperature",
            model_type="SOPDT",
            selected_model_params={"model_type": "SOPDT", "K": 1.02, "T1": 11.0, "T2": 19.0, "L": 4.2},
            confidence_score=0.82,
            normalized_rmse=0.05,
            r2_score=0.98,
            dt=1.0,
            experience_guidance=guidance,
        )
        self.assertTrue(any(item.get("history_seeded") for item in result["public_candidate_results"]))
        self.assertEqual(result["selection_inputs"]["experience_top_match_id"], "exp_seed")

    def test_sopdt_selection_inputs_prioritize_raw_model_params(self) -> None:
        result = select_best_pid_strategy(
            K=1.02,
            T=30.0,
            L=4.2,
            loop_type="temperature",
            model_type="SOPDT",
            selected_model_params={"model_type": "SOPDT", "K": 1.02, "T1": 11.0, "T2": 19.0, "L": 4.2},
            confidence_score=0.82,
            normalized_rmse=0.05,
            r2_score=0.98,
            dt=1.0,
            experience_guidance={},
        )
        inputs = result["selection_inputs"]
        self.assertEqual(inputs["model_type"], "SOPDT")
        self.assertEqual(inputs["selected_model_params"]["T1"], 11.0)
        self.assertEqual(inputs["selected_model_params"]["T2"], 19.0)
        self.assertIn("derived_tuning_features", inputs)
        self.assertIn("shape_index", inputs["derived_tuning_features"])
        self.assertIn("apparent_order", inputs["derived_tuning_features"])
        self.assertIn("aggregate_tau", inputs["derived_tuning_features"])
        self.assertIn("tested_candidates", inputs)
        self.assertNotIn("K", inputs)
        self.assertNotIn("T", inputs)
        self.assertNotIn("L", inputs)
        self.assertNotIn("compatibility_ktl", inputs)

    def test_sopdt_strategy_selection_uses_raw_time_constants(self) -> None:
        decision = select_tuning_strategy(
            loop_type="flow",
            K=1.0,
            T=99.0,
            L=99.0,
            model_type="SOPDT",
            model_params={"model_type": "SOPDT", "K": 1.0, "T1": 10.0, "T2": 1.0, "L": 0.0},
            model_confidence=0.9,
            r2_score=0.99,
            normalized_rmse=0.03,
        )
        self.assertEqual(decision["model_type"], "SOPDT")
        self.assertEqual(decision["strategy"], "IMC")

    def test_tune_pid_tool_prefers_session_model_context_over_tool_args(self) -> None:
        session_store = {
            "model_type": "SOPDT",
            "selected_model_params": {"model_type": "SOPDT", "K": 1.2, "T1": 10.0, "T2": 20.0, "L": 4.0},
            "tuning_model": {"K": 1.2, "T": 30.0, "L": 4.0},
            "model_confidence": {"confidence": 0.84},
            "normalized_rmse": 0.05,
            "r2_score": 0.98,
            "dt": 1.0,
        }
        captured = {}
        def fake_select(**kwargs):
            captured.update(kwargs)
            return {
                "best_candidate": {"strategy": "LAMBDA", "evaluation_result": {"performance_score": 8.0}},
                "pid_params": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.0, "Ti": 10.0, "Td": 0.0, "strategy": "LAMBDA", "description": "demo"},
                "public_candidate_results": [],
                "selection_reason": "ok",
                "selection_inputs": {"model_type": "SOPDT"},
                "experience_guidance": {},
            }
        result = tune_pid_tool(
            session_store=session_store,
            K=0.1,
            T=1.0,
            L=0.0,
            loop_type="temperature",
            select_best_pid_strategy_fn=fake_select,
        )
        self.assertEqual(captured["model_type"], "SOPDT")
        self.assertEqual(captured["selected_model_params"]["T1"], 10.0)
        self.assertAlmostEqual(captured["K"], 1.2)
        self.assertAlmostEqual(captured["T"], 30.0)
        self.assertAlmostEqual(captured["L"], 4.0)
        self.assertEqual(result["strategy_used"], "LAMBDA")

    def test_tune_pid_tool_prefers_explicit_selected_model_params_over_compatibility_triplet(self) -> None:
        session_store = {
            "model_type": "SOPDT",
            "selected_model_params": {"model_type": "SOPDT", "K": 0.2, "T1": 1.0, "T2": 1.0, "L": 0.0},
            "tuning_model": {"K": 0.2, "T": 2.0, "L": 0.0},
            "model_confidence": {"confidence": 0.84},
            "normalized_rmse": 0.05,
            "r2_score": 0.98,
            "dt": 1.0,
        }
        captured = {}

        def fake_select(**kwargs):
            captured.update(kwargs)
            return {
                "best_candidate": {"strategy": "LAMBDA", "evaluation_result": {"performance_score": 8.0}},
                "pid_params": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.0, "Ti": 10.0, "Td": 0.0, "strategy": "LAMBDA", "description": "demo"},
                "public_candidate_results": [],
                "selection_reason": "ok",
                "selection_inputs": {"model_type": "SOPDT", "selected_model_params": kwargs["selected_model_params"]},
                "experience_guidance": {},
            }

        result = tune_pid_tool(
            session_store=session_store,
            K=99.0,
            T=88.0,
            L=77.0,
            loop_type="temperature",
            model_type="SOPDT",
            selected_model_params={"model_type": "SOPDT", "K": 0.9, "T1": 10.0, "T2": 4.0, "L": 1.5},
            select_best_pid_strategy_fn=fake_select,
        )
        self.assertEqual(captured["model_type"], "SOPDT")
        self.assertEqual(captured["selected_model_params"]["T1"], 10.0)
        self.assertAlmostEqual(captured["K"], 0.9)
        self.assertAlmostEqual(captured["T"], 14.0)
        self.assertAlmostEqual(captured["L"], 1.5, places=6)
        self.assertEqual(result["selected_model_params"]["T2"], 4.0)

    def test_tune_pid_tool_accepts_json_string_selected_model_params(self) -> None:
        session_store = {
            "model_type": "SOPDT",
            "selected_model_params": {"model_type": "SOPDT", "K": 0.2, "T1": 1.0, "T2": 1.0, "L": 0.0},
            "tuning_model": {"K": 0.2, "T": 2.0, "L": 0.0},
            "model_confidence": {"confidence": 0.84},
            "normalized_rmse": 0.05,
            "r2_score": 0.98,
            "dt": 1.0,
        }
        captured = {}

        def fake_select(**kwargs):
            captured.update(kwargs)
            return {
                "best_candidate": {"strategy": "LAMBDA", "evaluation_result": {"performance_score": 8.0}},
                "pid_params": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.0, "Ti": 10.0, "Td": 0.0, "strategy": "LAMBDA", "description": "demo"},
                "public_candidate_results": [],
                "selection_reason": "ok",
                "selection_inputs": {"model_type": "SOPDT", "selected_model_params": kwargs["selected_model_params"]},
                "experience_guidance": {},
            }

        result = tune_pid_tool(
            session_store=session_store,
            K=99.0,
            T=88.0,
            L=77.0,
            loop_type="temperature",
            model_type="SOPDT",
            selected_model_params='{"model_type":"SOPDT","K":0.9,"T1":10.0,"T2":4.0,"L":1.5}',
            select_best_pid_strategy_fn=fake_select,
        )
        self.assertEqual(captured["selected_model_params"]["T1"], 10.0)
        self.assertEqual(result["selected_model_params"]["T2"], 4.0)

    def test_sopdt_native_tuning_rule_preserves_second_order_shape_fields(self) -> None:
        params = apply_tuning_rules(
            K=1.1,
            T=30.0,
            L=4.0,
            strategy="LAMBDA",
            model_type="SOPDT",
            model_params={"model_type": "SOPDT", "K": 1.1, "T1": 12.0, "T2": 24.0, "L": 4.0},
        )
        self.assertEqual(params["model_type"], "SOPDT")
        self.assertAlmostEqual(params["T1"], 12.0)
        self.assertAlmostEqual(params["T2"], 24.0)
        self.assertGreater(params["shape_index"], 0.0)
        self.assertGreater(params["apparent_order"], 1.0)
        self.assertGreater(params["T_work"], params["T_dominant"])
        self.assertGreaterEqual(params["L_work"], 4.0)
        self.assertGreaterEqual(params["Td"], 0.0)

    def test_sopdt_strategy_selection_uses_shape_features(self) -> None:
        lambda_choice = select_tuning_strategy(
            loop_type="temperature",
            K=0.8,
            T=18.0,
            L=2.5,
            model_type="SOPDT",
            model_params={"K": 0.8, "T1": 20.0, "T2": 18.0, "L": 2.5},
            model_confidence=0.9,
            r2_score=0.99,
            normalized_rmse=0.03,
        )
        self.assertEqual(lambda_choice["strategy"], "LAMBDA")

        imc_choice = select_tuning_strategy(
            loop_type="flow",
            K=0.6,
            T=12.0,
            L=1.8,
            model_type="SOPDT",
            model_params={"K": 0.6, "T1": 14.0, "T2": 3.0, "L": 1.8},
            model_confidence=0.92,
            r2_score=0.985,
            normalized_rmse=0.04,
        )
        self.assertEqual(imc_choice["strategy"], "IMC")

    def test_experience_stats_include_reuse_quality_metrics(self) -> None:
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
                loop_name="FIC_REUSE",
                loop_type="flow",
                loop_uri="/pid/reuse",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {"modelType": "FOPDT", "selectedModelParams": {"model_type": "FOPDT", "K": 0.4, "T": 2.0, "L": 0.2}, "K": 0.4, "T": 2.0, "L": 0.2},
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 1.5, "Ki": 0.5, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 8.0,
                        "method_confidence": 0.82,
                        "final_rating": 8.1,
                        "passed": True,
                        "initial_assessment": {"evaluated_pid": {"Kp": 2.0, "Ki": 1.0, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 160, "stepEvents": 2},
                },
            )
            experience_store.append_experience_record(record)
            experience_store.register_experience_references(
                [record["experience_id"]],
                hit_time="2026-03-12T12:00:00+08:00",
                follow_up_passed=True,
                follow_up_final_rating=8.6,
            )

            stats = experience_store.get_experience_stats()
            self.assertEqual(stats["total_hits"], 1)
            self.assertEqual(stats["total_follow_up_success"], 1)
            self.assertAlmostEqual(stats["follow_up_success_rate"], 1.0, places=4)
            self.assertEqual(stats["top_reused_experiences"][0]["experience_id"], record["experience_id"])
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_experience_index_preserves_sopdt_selected_model_params(self) -> None:
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
                loop_name="TIC_SOPDT",
                loop_type="temperature",
                loop_uri="/pid/sopdt",
                data_source="history",
                start_time="1",
                end_time="2",
                shared_data={"experience_guidance": {}},
                final_result={
                    "model": {
                        "modelType": "SOPDT",
                        "selectedModelParams": {"model_type": "SOPDT", "K": 0.7, "T1": 12.0, "T2": 5.0, "L": 1.5},
                        "K": 0.7,
                        "T": 17.0,
                        "L": 1.5,
                    },
                    "pidParams": {"strategyRequested": "AUTO", "strategyUsed": "LAMBDA", "Kp": 1.2, "Ki": 0.2, "Kd": 0.0},
                    "evaluation": {
                        "performance_score": 8.6,
                        "method_confidence": 0.91,
                        "final_rating": 8.8,
                        "passed": True,
                        "initial_assessment": {"evaluated_pid": {"Kp": 1.5, "Ki": 0.3, "Kd": 0.0}},
                        "auto_refine_result": {"applied": True},
                    },
                    "dataAnalysis": {"windowPoints": 220, "stepEvents": 1},
                },
            )
            experience_store.append_experience_record(record)
            listed = experience_store.list_experiences(model_type="SOPDT", limit=10)
            self.assertEqual(listed[0]["model"]["selected_model_params"]["T1"], 12.0)
            self.assertEqual(listed[0]["model"]["selected_model_params"]["T2"], 5.0)
        finally:
            experience_store.MEMORY_ROOT = old_root
            experience_store.EXPERIENCE_FILE = old_file
            experience_store.INDEX_FILE = old_index
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
