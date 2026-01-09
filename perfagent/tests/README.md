# PerfAgent 测试套件

这个目录包含了 PerfAgent 的所有 pytest 测试用例。

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py             # 共享的 pytest fixtures 和配置
├── test_config.py          # 配置系统测试
├── test_diff_applier.py    # Diff 应用器测试
├── test_trajectory.py      # 轨迹记录器测试
├── test_agent.py          # PerfAgent 核心测试
├── test_integration.py    # 集成测试
└── README.md              # 本文件
```

## 运行测试

### 运行所有测试
```bash
cd perfagent
pytest
```

### 运行特定测试文件
```bash
pytest tests/test_config.py
pytest tests/test_agent.py
```

### 运行特定测试类或方法
```bash
pytest tests/test_config.py::TestPerfAgentConfig
pytest tests/test_agent.py::TestPerfAgent::test_detect_language
```

### 按标记运行测试
```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 跳过慢速测试
pytest -m "not slow"

# 运行配置相关测试
pytest -m config
```

### 详细输出
```bash
# 显示详细输出
pytest -v

# 显示测试覆盖率（需要安装 pytest-cov）
pytest --cov=perfagent --cov-report=html

# 显示失败的详细信息
pytest --tb=long
```

## 测试标记

- `unit`: 单元测试（默认）
- `integration`: 集成测试
- `slow`: 慢速测试，可能需要较长时间
- `config`: 配置相关测试

## 测试数据

测试使用临时目录和文件，会在测试结束后自动清理。共享的测试数据通过 `conftest.py` 中的 fixtures 提供。

## 添加新测试

1. 在适当的测试文件中添加新的测试方法
2. 使用适当的 pytest 标记
3. 利用 `conftest.py` 中的共享 fixtures
4. 确保测试是独立的，不依赖其他测试的状态

## 注意事项

- 集成测试可能需要较长时间运行
- 某些测试可能需要网络连接（如果涉及 LLM 调用）
- 使用 `-m "not slow"` 可以跳过耗时的测试