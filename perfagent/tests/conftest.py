"""
pytest 共享配置和 fixtures
"""

import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

# Mock scipy to avoid import error
scipy = types.ModuleType("scipy")
scipy.stats = MagicMock()
sys.modules["scipy"] = scipy
sys.modules["scipy.stats"] = scipy.stats

# Mock openai to avoid pydantic dependency hell
openai = types.ModuleType("openai")
openai.OpenAI = MagicMock
openai.APIConnectionError = Exception
openai.APIError = Exception
openai.APITimeoutError = Exception
openai.BadRequestError = Exception
openai.RateLimitError = Exception
sys.modules["openai"] = openai

# Mock pydantic for backend_utils
pydantic = types.ModuleType("pydantic")
pydantic.BaseModel = MagicMock
pydantic.Field = MagicMock
sys.modules["pydantic"] = pydantic

# Mock other missing dependencies
fastapi = types.ModuleType("fastapi")
fastapi.FastAPI = MagicMock()
fastapi.HTTPException = MagicMock()
sys.modules["fastapi"] = fastapi

numpy = types.ModuleType("numpy")
sys.modules["numpy"] = numpy

import pytest

# 确保仓库根目录在 sys.path，以便使用绝对导入 'perfagent'
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def temp_session_dir():
    """会话级别的临时目录"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """函数级别的临时目录"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_python_code():
    """示例 Python 代码"""
    return """
def inefficient_function(n):
    result = 0
    for i in range(n):
        for j in range(i):
            result += 1
    return result

def main():
    import time
    start = time.time()
    result = inefficient_function(100)
    end = time.time()
    print(f"结果: {result}, 时间: {end - start:.4f}秒")

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def fake_llm():
    """提供一个假的 LLM 客户端，避免测试依赖真实接口。"""

    class FakeLLM:
        def call_llm(self, messages, temperature: float = 0.3, max_tokens=None) -> str:
            # 返回一个通用的 SEARCH/REPLACE 区块；若无法匹配则代码不变化
            return "<<<<<<< SEARCH\nold code snippet\n=======\nnew code snippet\n>>>>>>> REPLACE"

    return FakeLLM()


@pytest.fixture
def sample_java_code():
    """示例 Java 代码"""
    return """
public class SlowExample {
    public static int inefficientSum(int n) {
        int result = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                result++;
            }
        }
        return result;
    }
    
    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        int result = inefficientSum(1000);
        long end = System.currentTimeMillis();
        System.out.println("结果: " + result + ", 时间: " + (end - start) + "ms");
    }
}
"""


@pytest.fixture
def sample_cpp_code():
    """示例 C++ 代码"""
    return """
#include <iostream>
#include <chrono>

int inefficient_sum(int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            result++;
        }
    }
    return result;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    int result = inefficient_sum(1000);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "结果: " << result << ", 时间: " << duration.count() << "ms" << std::endl;
    
    return 0;
}
"""


@pytest.fixture
def default_config(temp_dir):
    """默认测试配置"""
    from perfagent.config import PerfAgentConfig

    return PerfAgentConfig(trajectory_dir=temp_dir / "trajectories", max_iterations=2)


@pytest.fixture
def sample_diff():
    """示例 diff"""
    return """@@ -1,5 +1,5 @@
 def function():
-    old_line = 1
+    new_line = 1
     unchanged_line = 2
-    another_old_line = 3
+    another_new_line = 3
     return result"""


def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "unit: 单元测试")


def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    # 为没有标记的测试添加 unit 标记
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # 为包含 "integration" 的测试添加 integration 标记
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # 为包含 "slow" 的测试添加 slow 标记
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
