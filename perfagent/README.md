# PerfAgent - 代码性能优化工具

PerfAgent 是基于 SE-Agent 框架构建的代码性能优化工具，模仿 sweagent 的设计模式，专门用于迭代式优化代码效率。

## 功能特性

- **迭代优化**: 通过多轮迭代不断改进代码性能
- **性能评估**: 使用 EffiBench 基准测试评估代码性能
- **轨迹记录**: 完整记录优化过程，便于分析和复现
- **Diff 应用**: 自动解析和应用模型生成的代码修改
- **配置灵活**: 支持 YAML 配置文件和命令行参数
- **批量处理**: 支持单个实例和批量实例的优化

## 安装和设置

1. 确保在 SE-Agent 环境中运行
2. 安装依赖（如果需要）：
   ```bash
   pip install pyyaml
   ```

## 使用方法

### 基本用法

```bash
# 运行单个实例（推荐）
python -m perfagent.run --instance /path/to/instance.json --base-dir /path/to/output

# 批量运行（推荐）
python -m perfagent.run_batch --instances-dir /path/to/instances/ --base-dir /path/to/output

# 使用配置文件（批量）
python -m perfagent.run_batch --config config.yaml --instances-dir /path/to/instances/ --base-dir /path/to/output
```

### 命令行参数

- `--config`: 配置文件路径
- `--instance`: 单个实例文件路径
- `--instances-dir`: 实例目录路径（批量运行）
- `--output`: 结果输出文件路径
- `--base-dir`: 实例输出基目录（统一日志、轨迹与结果）
- `--max-iterations`: 最大迭代次数
- `--model`: 模型名称
- `--log-level`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `--trajectory-dir`: 轨迹保存目录
- `--log-dir`: 日志保存目录

### 配置文件

创建 YAML 配置文件来自定义 PerfAgent 的行为：

```yaml
# 基础配置
max_iterations: 10
time_limit: 300
memory_limit: 1024

# 模型配置
model_name: "gpt-4"
temperature: 0.1
max_tokens: 4000

# 性能评估配置
num_runs: 5
trim_ratio: 0.1
max_workers: 4

# 轨迹和日志配置
save_trajectory: true
trajectory_dir: "./trajectories"
log_dir: "./logs"
log_level: "INFO"
```

## 架构设计

### 核心组件

1. **PerfAgent**: 主要的优化代理类
2. **PerfAgentConfig**: 配置管理系统
3. **TrajectoryLogger**: 轨迹记录系统
4. **DiffApplier**: Diff 解析和应用工具
5. **ModelInterface**: 模型交互接口

### 优化流程

1. **初始化**: 加载配置和实例数据
2. **性能评估**: 评估初始代码性能
3. **迭代优化**:
   - 生成优化建议
   - 解析和应用 diff
   - 评估优化后性能
   - 记录优化历史
4. **结果输出**: 保存最佳代码和轨迹

### 可借鉴的 sweagent 设计

- **轨迹记录**: 完整记录每个步骤的输入输出
- **配置系统**: 灵活的 YAML 配置支持
- **模块化设计**: 清晰的组件分离和接口定义
- **错误处理**: 健壮的异常处理和恢复机制
- **日志系统**: 分级日志和文件输出

## 输出文件

### 轨迹文件

轨迹文件保存在 `<base_dir>/<task_name>/` 中，格式为 `<task_name>.traj`：

```json
{
  "metadata": {
    "instance_id": "test_001",
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:05:00",
    "total_iterations": 5,
    "success": true
  },
  "steps": [
    {
      "step_id": 1,
      "timestamp": "2024-01-01T10:00:00",
      "action": "initial_evaluation",
      "input_data": {...},
      "output_data": {...},
      "performance_metrics": {...}
    }
  ]
}
```

### 日志文件

日志文件保存在 `<base_dir>/<task_name>/perfagent.log` 中，包含详细的运行信息。

## 测试

运行测试用例：

```bash
python -m unittest perfagent.test_perfagent
```

## 示例

### 运行单个实例

```bash
python -m perfagent.run \
  --instance /mnt/d/workspace/SE-Agent/SE/instances/EffiBench-X/dataset/aizu_1444_yokohama-phenomena.json \
  --base-dir /mnt/d/workspace/SE-Agent/output \
  --max-iterations 5 \
  --output /mnt/d/workspace/SE-Agent/output/aizu_1444_yokohama-phenomena/result.json
```

### 批量运行 EffiBench-X

```bash
python -m perfagent.run_batch \
  --instances-dir /mnt/d/workspace/SE-Agent/SE/instances/EffiBench-X/dataset \
  --config perfagent/config_example.yaml \
  --base-dir /mnt/d/workspace/SE-Agent/output \
  --output /mnt/d/workspace/SE-Agent/output/summary.json
```

## 扩展和定制

### 自定义模型接口

继承 `ModelInterface` 类来集成不同的模型：

```python
from perfagent.agent import ModelInterface

class CustomModelInterface(ModelInterface):
    def query(self, prompt: str, max_tokens: int = 4000) -> str:
        # 实现自定义模型调用
        pass
```

### 自定义性能评估

修改 `_evaluate_performance` 方法来使用不同的评估标准。

### 自定义提示词

在配置文件中设置 `system_template` 和 `optimization_template` 来自定义提示词。

## 注意事项

1. 确保有足够的磁盘空间存储轨迹和日志文件
2. 根据实际情况调整 `time_limit` 和 `memory_limit`
3. 模型接口目前是简化实现，需要集成真实的 API
4. 性能评估依赖于 EffiBench 的 benchmark.py 函数

## 安全与配置建议

- 不要在仓库中保存明文 API Key。使用环境变量或本地 `.env` 文件，并在配置中引用，例如：
  - 将 `perfagent/config_example.yaml` 的 `model.api_key` 设置为 `${OPENROUTER_API_KEY}`，在运行前导出：
    - `export OPENROUTER_API_KEY=xxxxx`
- 启用请求与响应脱敏日志：
  - `--llm-log-io` 与 `--llm-log-sanitize` 会将 LLM I/O 记录到 `logs/llm_io.log`，并隐藏敏感端点信息。
- 推荐配置早停以避免无效迭代：
  - 通过 `--early-stop-no-improve N` 或在 YAML 中设置 `early_stop_no_improve: N` 控制连续未改进次数后停止。
- 日志重复与膨胀控制：
  - 工具会避免添加重复文件处理器；如需自定义路径请使用 `--log-dir`。

## 故障排除

### 常见问题

1. **无法找到实例文件**: 检查文件路径和权限
2. **性能评估失败**: 检查 benchmark.py 的依赖和配置
3. **模型调用失败**: 检查模型接口的实现和 API 配置
4. **轨迹保存失败**: 检查目录权限和磁盘空间

### 调试技巧

- 使用 `--log-level DEBUG` 获取详细日志
- 检查轨迹文件了解具体的执行步骤
- 使用 `--max-iterations 1` 进行快速测试