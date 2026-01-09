"""
轨迹记录器的 pytest 测试用例
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from perfagent.trajectory import TrajectoryLogger


class TestTrajectoryLogger:
    """测试轨迹记录器"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def logger(self, temp_dir):
        """创建 TrajectoryLogger 实例"""
        return TrajectoryLogger("test_instance", temp_dir)

    def test_step_logging(self, logger):
        """测试步骤记录"""
        step_id = logger.start_step("test_action", query="input: test")
        assert step_id == "step_1"
        assert len(logger.steps) == 1

        logger.end_step(step_id, response="output: test")
        assert logger.steps[0].response == "output: test"
        assert logger.steps[0].action == "test_action"

    def test_trajectory_save_load(self, logger):
        """测试轨迹保存和加载"""
        step_id = logger.start_step("test", query="data: test")
        logger.end_step(step_id, response="result: success")

        # 保存轨迹
        trajectory_file = logger.save_trajectory()
        assert os.path.exists(trajectory_file)

        # 验证文件内容
        with open(trajectory_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["info"]["instance_id"] == "test_instance" or data["metadata"]["instance_id"] == "test_instance"
        assert len(data["steps"]) == 1

    def test_multiple_steps(self, logger):
        """测试多个步骤的记录"""
        # 记录多个步骤
        step1_id = logger.start_step("action1", query="input1: data1")
        step2_id = logger.start_step("action2", query="input2: data2")

        logger.end_step(step1_id, response="output1: result1")
        logger.end_step(step2_id, response="output2: result2")

        assert len(logger.steps) == 2
        assert logger.steps[0].action == "action1"
        assert logger.steps[1].action == "action2"

    def test_step_with_error(self, logger):
        """测试带错误的步骤记录"""
        step_id = logger.start_step("error_action", query="input: test")
        logger.end_step(step_id, response="output: failed", error="Something went wrong")

        assert logger.steps[0].error == "Something went wrong"
        assert logger.steps[0].response == "output: failed"

    def test_finalize_success(self, logger):
        """测试成功完成的轨迹记录"""
        step_id = logger.start_step("test", query="data: test")
        logger.end_step(step_id, response="result: success")

        logger.finalize(success=True, final_performance={"score": 95.0})

        assert logger.metadata.success is True
        assert logger.metadata.final_performance["score"] == 95.0
        assert logger.metadata.error_message is None

    def test_finalize_failure(self, logger):
        """测试失败的轨迹记录"""
        step_id = logger.start_step("test", query="data: test")
        logger.end_step(step_id, response="result: failed", error="Test error")

        logger.finalize(success=False, error_message="Optimization failed")

        assert logger.metadata.success is False
        assert logger.metadata.error_message == "Optimization failed"
        assert logger.metadata.final_performance is None

    def test_trajectory_summary(self, logger):
        """测试轨迹摘要"""
        # 添加一些步骤
        step1_id = logger.start_step("optimize", query="code: test")
        logger.end_step(step1_id, response="improved_code: optimized")

        step2_id = logger.start_step("evaluate", query="code: optimized")
        logger.end_step(step2_id, response="performance: 0.8")

        # 获取摘要
        summary = logger.get_trajectory_summary()

        # 验证摘要内容
        assert summary["instance_id"] == logger.metadata.instance_id
        assert summary["total_steps"] == 2
        assert summary["performance_trend"] == "improving"  # 修正字段名
        assert "total_duration" in summary

    def test_invalid_step_id(self, logger):
        """测试无效的步骤 ID"""
        # TrajectoryLogger 实际上不会抛出异常，而是记录错误日志
        logger.end_step("invalid_step_id", response={"output": "test"})
        # 验证步骤数量没有变化
        assert len(logger.steps) == 0

    def test_duplicate_step_end(self, logger):
        """测试重复结束步骤"""
        step_id = logger.start_step("test", query="input: test")
        logger.end_step(step_id, response="output: test")

        # TrajectoryLogger 允许重复结束步骤，会覆盖之前的结果
        logger.end_step(step_id, response="output: test2")
        assert logger.steps[0].response == "output: test2"

    def test_trajectory_file_path(self, logger):
        """测试轨迹文件路径生成"""
        trajectory_file = logger.save_trajectory()
        expected_filename = f"{logger.metadata.instance_id}.traj"

        assert trajectory_file.endswith(expected_filename)
        assert os.path.exists(trajectory_file)
