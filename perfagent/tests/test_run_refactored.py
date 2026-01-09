from unittest.mock import patch

import pytest

from perfagent.agent import EffiBenchXInstance, PerfAgent
from perfagent.config import PerfAgentConfig


class TestRunRefactored:
    @pytest.fixture
    def mock_llm_client(self):
        with patch("perfagent.agent.LLMClient") as mock:
            client_instance = mock.return_value
            # Configure call_llm to return a diff that "optimizes" code
            client_instance.call_llm.return_value = """
<<<<<<< SEARCH
def solve():
    pass
=======
def solve():
    return 42
>>>>>>> REPLACE
"""
            yield client_instance

    @pytest.fixture
    def mock_benchmark(self):
        with patch("perfagent.agent.run_performance_benchmark") as mock:
            # We will set side_effect in individual tests
            yield mock

    @pytest.fixture
    def config(self, tmp_path):
        cfg = PerfAgentConfig(max_iterations=1)
        cfg.logging.log_dir = tmp_path / "logs"
        cfg.logging.trajectory_dir = tmp_path / "traj"
        cfg.model.api_key = "dummy"  # Trigger LLM client init
        cfg.model.api_base = "http://dummy"
        cfg.language_cfg.language = "python3"
        # Set runtime config for consistent testing
        cfg.runtime.num_runs = 5
        return cfg

    @pytest.fixture
    def agent(self, config, mock_llm_client):
        return PerfAgent(config)

    @pytest.fixture
    def instance(self):
        return EffiBenchXInstance(
            id="test-inst-1",
            title="Test Instance",
            title_slug="test-instance",
            description="A test problem",
            description_md="# Test Problem",
            source="leetcode",
            url="http://example.com",
            type="functional",
            starter_code={"python3": "def solve():\n    pass"},
            generated_tests=[{"input": "1", "output": "1"}],
            evaluator="def check(): pass",  # Added evaluator
        )

    def test_run_success_flow(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test the full run flow with mocked dependencies."""

        # Setup benchmark side effects (4 calls due to cascading evaluation logic)
        # 1. Initial Single Run (Pass)
        # 2. Initial Full Run (Pass, Runtime 1.0)
        # 3. Optimized Single Run (Pass)
        # 4. Optimized Full Run (Pass, Runtime 0.5 - Improved)
        mock_benchmark.side_effect = [
            # Initial Evaluation
            {
                "performance_analysis": {"runtime": 1.0, "memory": 100, "passed": True},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 1.0, "memory": 100, "passed": True},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
            # Optimization Iteration 1
            {
                "performance_analysis": {"runtime": 0.5, "memory": 100, "passed": True},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 0.5, "memory": 100, "passed": True},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
        ]

        # Override _extract_initial_code to return simple code instead of reading from file/config
        with patch.object(agent, "_extract_initial_code", return_value="def solve():\n    pass"):
            result = agent.run(instance)

        # Verify result structure
        assert result["success"] is True
        assert result["total_iterations"] >= 1
        assert result["final_performance"] == 0.5
        assert result["initial_performance"] == 1.0

        # Verify optimization history
        assert len(result["optimization_history"]) == 1
        hist = result["optimization_history"][0]
        assert hist["success"] is True
        assert hist["performance_before"] == 1.0
        assert hist["performance_after"] == 0.5

        # Verify LLM was called
        mock_llm_client.call_llm.assert_called()

        # Verify benchmark was called 4 times
        assert mock_benchmark.call_count == 4

    def test_run_no_improvement(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test flow where optimization does not improve performance."""
        # Set max_iterations > 1 so we don't force save
        agent.config.max_iterations = 2
        agent.config.early_stop_no_improve = 1  # Stop after 1 failed iter

        # Mock benchmark to show NO improvement (4 calls)
        # 1. Initial Single Run (Pass)
        # 2. Initial Full Run (Pass, Runtime 1.0)
        # 3. Optimized Single Run (Pass)
        # 4. Optimized Full Run (Pass, Runtime 1.5 - Worse)
        mock_benchmark.side_effect = [
            # Initial Evaluation
            {
                "performance_analysis": {"runtime": 1.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 1.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            # Optimization Iteration 1 (Worse performance)
            {
                "performance_analysis": {"runtime": 1.5, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 1.5, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
        ]

        with patch.object(agent, "_extract_initial_code", return_value="def solve():\n    pass"):
            result = agent.run(instance)

        assert result["success"] is False
        assert result["final_performance"] == 1.0  # Kept best (initial)

        hist = result["optimization_history"][0]
        assert hist["success"] is False

    def test_run_single_iteration_forced_save(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test that single iteration run saves generated code even if performance is worse."""
        agent.config.runtime.max_iterations = 1

        # Mock benchmark: Initial=1.0, Optimized=2.0 (Worse)
        # Note: side_effect length depends on implementation.
        # With single iteration:
        # 1. Initial Eval (Single)
        # 2. Initial Eval (Full)
        # 3. Iter 1 Eval (Single) -> Passed
        # 4. Iter 1 Eval (Full) -> Runtime 2.0 (Worse)
        mock_benchmark.side_effect = [
            # Initial
            {
                "performance_analysis": {"runtime": 1.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 1.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            # Optimized
            {
                "performance_analysis": {"runtime": 2.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
            {
                "performance_analysis": {"runtime": 2.0, "passed": True},
                "pass_rates": [1.0],
                "first_run_details": [{"passed": True}],
            },
        ]

        with patch.object(agent, "_extract_initial_code", return_value="def solve():\n    pass"):
            result = agent.run(instance)

        assert result["total_iterations"] == 1  # Initial + 1? No, logic is base 0 for extracted.
        assert result["final_performance"] == 2.0  # Saved the worse one
        assert result["success"] is False  # Because it got worse
        assert len(result["optimization_history"]) == 1
        assert result["optimization_history"][0]["success"] is True  # Forced success flag
