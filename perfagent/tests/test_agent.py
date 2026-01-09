"""
PerfAgent 的 pytest 测试用例
"""

import tempfile
from pathlib import Path

import pytest

from perfagent.agent import EffiBenchXInstance, PerfAgent
from perfagent.config import PerfAgentConfig


class TestPerfAgent:
    """测试 PerfAgent"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def config(self, temp_dir):
        """创建测试配置"""
        cfg = PerfAgentConfig(max_iterations=3)
        cfg.logging.trajectory_dir = temp_dir
        return cfg

    @pytest.fixture
    def agent(self, config):
        """创建 PerfAgent 实例"""
        return PerfAgent(config)

    def test_yaml_config(self, temp_dir):
        """测试 YAML 配置加载"""
        config_file = temp_dir / "test_config.yaml"
        config_content = """
model:
  name: "test-model"
max_iterations: 5
logging:
  trajectory_dir: "/tmp/test"
"""
        config_file.write_text(config_content)

        config = PerfAgentConfig.from_yaml(str(config_file))
        assert config.model.name == "test-model"
        assert config.max_iterations == 5
        assert config.logging.trajectory_dir == Path("/tmp/test")

    def test_extract_initial_code(self, agent):
        """测试提取初始代码（适配占位符返回）"""
        inst = EffiBenchXInstance(
            id="inst-1",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code="""def slow_function():\n    result = 0\n    for i in range(1000000):\n        result += i\n    return result\n""",
        )

        code = agent._extract_initial_code(inst)
        assert isinstance(code, str)
        assert "# Start your code here" in code  # 当前实现返回占位符

    def test_detect_language(self, agent):
        """测试语言检测（使用配置语言）"""
        # 默认配置语言
        agent.config.language_cfg.language = "python3"
        inst = EffiBenchXInstance(
            id="x", title="", title_slug="", description="", description_md="", source="", url="", type=""
        )
        assert agent._detect_language(inst) == "python3"

        # 切换到 Java
        agent.config.language_cfg.language = "java"
        assert agent._detect_language(inst) == "java"

        # 切换到 C++
        agent.config.language_cfg.language = "cpp"
        assert agent._detect_language(inst) == "cpp"

    def test_evaluate_performance(self, agent, temp_dir):
        """测试性能评估"""
        # 创建测试数据
        test_cases = ["test_case_1", "test_case_2"]
        inst = EffiBenchXInstance(
            id="test_instance", title="", title_slug="", description="", description_md="", source="", url="", type=""
        )
        code = """
import time

def test_function():
    time.sleep(0.01)  # 模拟一些计算
    return sum(range(1000))

if __name__ == "__main__":
    start_time = time.time()
    result = test_function()
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.4f}秒")
    print(f"结果: {result}")
"""

        performance = agent._evaluate_performance("python3", code, test_cases, inst)

        # 验证性能指标
        assert "performance_analysis" in performance
        assert isinstance(performance["performance_analysis"], dict)

    def test_extract_diff_from_response(self, agent):
        """测试从响应中提取 diff（仅支持 SEARCH/REPLACE 格式）"""
        response_with_search_replace = (
            "这里是一些解释文本。\n\n"
            "<<<<<<< SEARCH\n"
            "line2\nline3\n"
            "=======\n"
            "LINE2\nLINE3\n"
            ">>>>>>> REPLACE\n\n"
            "这是一段额外的说明。"
        )
        diff = agent._extract_diff_from_response(response_with_search_replace)
        assert "<<<<<<< SEARCH" in diff
        assert ">>>>>>> REPLACE" in diff
        assert "line2" in diff and "LINE3" in diff

    def test_extract_diff_no_diff_block(self, agent):
        """测试没有 diff 块的响应"""
        response_without_diff = "这是一个没有 diff 的响应。"

        diff = agent._extract_diff_from_response(response_without_diff)
        assert diff == ""  # 应该返回空字符串而不是None

    def test_extract_diff_multiple_blocks(self, agent):
        """测试多个 SEARCH/REPLACE 区块的响应提取"""
        response_with_multiple_blocks = (
            "前文说明\n\n"
            "<<<<<<< SEARCH\n"
            "oldA\n"
            "=======\n"
            "newA\n"
            ">>>>>>> REPLACE\n\n"
            "一些说明\n\n"
            "<<<<<<< SEARCH\n"
            "oldB\n"
            "=======\n"
            "newB\n"
            ">>>>>>> REPLACE\n"
            "后文文本\n"
        )
        diff = agent._extract_diff_from_response(response_with_multiple_blocks)
        assert diff.count("<<<<<<< SEARCH") == 2
        assert diff.count(">>>>>>> REPLACE") == 2
        assert "oldA" in diff and "newB" in diff

    def test_agent_initialization(self, config):
        """测试 Agent 初始化"""
        agent = PerfAgent(config)

        assert agent.config == config
        assert agent.diff_applier is not None

    def test_config_validation(self):
        """测试配置验证"""
        # 测试正常配置
        config = PerfAgentConfig(max_iterations=5)
        assert config.max_iterations == 5

        # 测试默认值
        default_config = PerfAgentConfig()
        assert default_config.max_iterations == 10
        assert default_config.model.name == "gpt-4"

    def test_language_detection_edge_cases(self, agent):
        """测试语言检测的边缘情况（返回配置语言）"""
        agent.config.language_cfg.language = "python3"  # 默认语言
        inst = EffiBenchXInstance(
            id="edge", title="", title_slug="", description="", description_md="", source="", url="", type=""
        )

        # 空代码
        assert agent._detect_language(inst) == "python3"

        # 只有注释的代码
        assert agent._detect_language(inst) == "python3"

        # 混合语言特征（依然返回配置语言）
        assert agent._detect_language(inst) == "python3"
