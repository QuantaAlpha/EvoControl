# PerfAgent - Code Performance Optimization Agent

PerfAgent is a code performance optimization tool built on top of the CSE framework, designed for iterative code efficiency optimization through LLM-driven refinement.

## 🎯 Features

- **Iterative Optimization**: Continuously improve code performance through multiple rounds
- **Performance Evaluation**: Evaluate code efficiency using EffiBench-X benchmark
- **Trajectory Recording**: Complete logging of optimization process for analysis and reproduction
- **Diff Application**: Automatic parsing and application of model-generated code modifications
- **Flexible Configuration**: Support for YAML configuration files and command-line arguments
- **Batch Processing**: Support for both single instance and batch instance optimization

## 📁 Directory Structure

```text
perfagent/
├── agent.py            # Main optimization agent class
├── config.py           # Configuration management system
├── trajectory.py       # Trajectory logging system
├── diff_applier.py     # Diff parsing and application tool
├── llm_client.py       # LLM interaction interface
├── run.py              # Single instance runner
├── run_batch.py        # Batch instance runner
├── effibench/          # EffiBench integration
│   ├── benchmark.py    # Benchmark execution
│   ├── backends/       # Execution backends
│   └── utils.py        # Utility functions
├── utils/              # Utility modules
│   └── log.py          # Logging utilities
└── tests/              # Test suite
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run single instance (recommended)
python -m perfagent.run \
    --instance /path/to/instance.json \
    --base-dir /path/to/output

# Batch run (recommended)
python -m perfagent.run_batch \
    --instances-dir /path/to/instances/ \
    --base-dir /path/to/output

# Use configuration file (batch)
python -m perfagent.run_batch \
    --config perfagent/config_example.yaml \
    --instances-dir /path/to/instances/ \
    --base-dir /path/to/output
```

### Command Line Arguments

| Argument           | Description                          |
| ------------------ | ------------------------------------ |
| `--config`         | Configuration file path              |
| `--instance`       | Single instance file path            |
| `--instances-dir`  | Instance directory path (batch run)  |
| `--output`         | Result output file path              |
| `--base-dir`       | Instance output base directory       |
| `--max-iterations` | Maximum number of iterations         |
| `--model`          | Model name                           |
| `--log-level`      | Log level (DEBUG/INFO/WARNING/ERROR) |
| `--trajectory-dir` | Trajectory save directory            |
| `--log-dir`        | Log save directory                   |

## ⚙️ Configuration

Create a YAML configuration file to customize PerfAgent behavior:

```yaml
# Basic configuration
max_iterations: 10
time_limit: 300
memory_limit: 1024

# Model configuration
model_name: "deepseek-chat"
temperature: 0.1
max_tokens: 4000

# Performance evaluation configuration
num_runs: 5
trim_ratio: 0.1
max_workers: 4

# Trajectory and log configuration
save_trajectory: true
trajectory_dir: "./trajectories"
log_dir: "./logs"
log_level: "INFO"
```

## 🏗️ Architecture

### Core Components

| Component          | Description                       |
| ------------------ | --------------------------------- |
| `PerfAgent`        | Main optimization agent class     |
| `PerfAgentConfig`  | Configuration management system   |
| `TrajectoryLogger` | Trajectory recording system       |
| `DiffApplier`      | Diff parsing and application tool |
| `LLMClient`        | Model interaction interface       |

### Optimization Flow

1. **Initialization**: Load configuration and instance data
2. **Performance Evaluation**: Evaluate initial code performance
3. **Iterative Optimization**:
   - Generate optimization suggestions
   - Parse and apply diff
   - Evaluate optimized performance
   - Record optimization history
4. **Result Output**: Save best code and trajectory

## 📊 Output Files

### Trajectory File

Trajectory files are saved in `<base_dir>/<task_name>/` as `<task_name>.traj`:

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

### Log File

Log files are saved in `<base_dir>/<task_name>/perfagent.log` with detailed runtime information.

## 🛠️ Testing

Run test cases:

```bash
# Run all tests
python -m pytest perfagent/tests/

# Run specific test
python -m pytest perfagent/tests/test_agent.py
```

## 📋 Examples

### Run Single Instance

```bash
python -m perfagent.run \
    --instance instances/aizu_1444_yokohama-phenomena.json \
    --base-dir output \
    --max-iterations 5 \
    --output output/aizu_1444_yokohama-phenomena/result.json
```

### Batch Run EffiBench-X

```bash
python -m perfagent.run_batch \
    --instances-dir instances/ \
    --config perfagent/config_example.yaml \
    --base-dir output \
    --output output/summary.json
```

## 🔒 Security & Configuration Tips

- **API Keys**: Do not store plaintext API keys in the repository. Use environment variables or local `.env` files:

  ```bash
  export OPENROUTER_API_KEY=xxxxx
  ```

  Then reference in config: `api_key: ${OPENROUTER_API_KEY}`

- **LLM Logging**: Enable request/response logging with sanitization:

  - `--llm-log-io` and `--llm-log-sanitize` log LLM I/O to `logs/llm_io.log` with sensitive endpoints hidden

- **Early Stopping**: Configure early stopping to avoid ineffective iterations:
  - Use `--early-stop-no-improve N` or set `early_stop_no_improve: N` in YAML

## 🔧 Extension & Customization

### Custom Model Interface

Inherit from the base class to integrate different models:

```python
from perfagent.llm_client import LLMClient

class CustomLLMClient(LLMClient):
    def query(self, prompt: str, max_tokens: int = 4000) -> str:
        # Implement custom model call
        pass
```

### Custom Performance Evaluation

Modify `_evaluate_performance` method to use different evaluation criteria.

### Custom Prompts

Set `system_template` and `optimization_template` in configuration file to customize prompts.

## ⚠️ Important Notes

1. Ensure sufficient disk space for trajectory and log files
2. Adjust `time_limit` and `memory_limit` based on actual requirements
3. EffiBench-X backend service must be running for performance evaluation
4. Check network connectivity for API calls

## 🆘 Troubleshooting

### Common Issues

| Issue                         | Solution                                         |
| ----------------------------- | ------------------------------------------------ |
| Instance file not found       | Check file path and permissions                  |
| Performance evaluation failed | Check EffiBench-X dependencies and configuration |
| Model call failed             | Check API key and network connection             |
| Trajectory save failed        | Check directory permissions and disk space       |

### Debugging Tips

- Use `--log-level DEBUG` for detailed logs
- Check trajectory files for execution steps
- Use `--max-iterations 1` for quick testing

## 🔗 Related Documentation

- [Main Project README](../README.md)
- [SE_Perf Documentation](../SE_Perf/README.md)
- [Configuration Examples](config_example.yaml)
