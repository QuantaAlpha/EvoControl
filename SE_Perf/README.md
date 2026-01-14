# SE_Perf - Core Evolution Framework

The core evolution framework module of CSE (Controlled Self-Evolution), implementing diversified planning initialization, controlled genetic evolution, and hierarchical memory systems.

## 🎯 Module Overview

SE_Perf is the core engine of EvoControl, responsible for:

- **Evolution Strategy Orchestration**: Coordinating multi-round iterative evolution
- **Operator System**: Implementing Plan, Mutation, Crossover, and other evolution operators
- **Memory System**: Managing Local Memory and Global Memory
- **Parallel Execution**: Supporting multi-instance parallel optimization

## 📁 Directory Structure

```text
SE_Perf/
├── instance_runner.py      # Main entry - multi-instance parallel executor
├── perf_run.py             # Single instance evolution runner
├── perf_config.py          # Configuration parser
├── core/                   # Core functionality modules
│   ├── swe_iterator.py     # Evolution iterator
│   ├── global_memory/      # Global memory system
│   │   ├── bank.py         # Memory bank management
│   │   ├── embeddings/     # Vector embeddings
│   │   └── memory/         # Memory storage
│   └── utils/              # Utility functions
├── operators/              # Evolution operator system
│   ├── base.py             # Operator base class
│   ├── registry.py         # Operator registration
│   ├── plan.py             # Diversified planning operator
│   ├── crossover.py        # Compositional crossover operator
│   ├── reflection_refine.py # Reflection refinement operator
│   ├── filter.py           # Filter operator
│   └── alternative_strategy.py # Alternative strategy operator
└── test/                   # Test suite
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run experiment (recommended entry point)
python SE_Perf/instance_runner.py \
    --config configs/Plan-Weighted-Local-Global-30.yaml \
    --max-parallel 10 \
    --mode execute

# Quick test (first 5 instances)
python SE_Perf/instance_runner.py \
    --config configs/Plan-Weighted-Local-Global-30.yaml \
    --limit 5 \
    --mode execute
```

### Single Instance Run

```bash
python SE_Perf/perf_run.py \
    --config configs/Plan-Weighted-Local-Global-30.yaml \
    --instance instances/aizu_1444_yokohama-phenomena.json \
    --output-dir trajectories_perf/test_run
```

## ⚙️ Configuration System

### Two-Layer Configuration Architecture

| Config Type         | File                                         | Purpose                                   |
| ------------------- | -------------------------------------------- | ----------------------------------------- |
| **Base Config**     | `configs/perf_configs/config_integral.yaml`  | Model parameters, runtime limits, prompts |
| **Strategy Config** | `configs/Plan-Weighted-Local-Global-30.yaml` | Evolution strategy orchestration          |

### Strategy Configuration Example

```yaml
# Model configuration
model:
  name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  api_key: "your-api-key"

# Operator model configuration
operator_models:
  name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  api_key: "your-api-key"

# Global memory configuration
global_memory_bank:
  enabled: true
  embedding_model:
    model: "text-embedding-3-small"
    api_key: "your-embedding-key"

# Strategy orchestration
strategy:
  iterations:
    - operator: "plan"
      num: 5
      trajectory_labels: ["iter1_sol1", "iter1_sol2", ...]
    - operator: "reflection_refine"
      trajectory_label: "iter1_sol6"
    - operator: "crossover"
      trajectory_label: "iter1_sol7"
```

## 🧬 Operator System

### Core Operators

| Operator               | Function                                          | Paper Component         |
| ---------------------- | ------------------------------------------------- | ----------------------- |
| `plan`                 | Generate diverse algorithmic strategies           | Diversified Planning    |
| `reflection_refine`    | Feedback-guided controlled mutation               | Controlled Mutation     |
| `crossover`            | Compositional crossover, merge solution strengths | Compositional Crossover |
| `filter`               | History-based solution filtering                  | Local Memory            |
| `alternative_strategy` | Explore alternative strategies                    | Strategy Exploration    |

### Custom Operators

```python
from SE_Perf.operators import TemplateOperator, register_operator

class MyOperator(TemplateOperator):
    def get_name(self):
        return "my_operator"

    def _generate_content(self, instance_info, problem_description, trajectory_data):
        # Implement custom generation logic
        return "Generated strategy content"

# Register operator
register_operator("my_operator", MyOperator)
```

> 📖 For detailed operator development guide, see [operators/README.md](operators/README.md)

## 🧠 Memory System

### Local Memory (Intra-task)

- Records success/failure experiences for current task
- Avoids repeated exploration of failed directions
- Guides optimization direction for subsequent iterations

### Global Memory (Inter-task)

- Extracts reusable optimization patterns from successful cases
- Retrieves relevant experiences based on semantic similarity
- Accelerates optimization process for new tasks

## 📊 Output Structure

```text
trajectories_perf/experiment_{timestamp}/
├── {instance_name}/
│   ├── iteration_{n}/
│   │   ├── result.json         # Evaluation results
│   │   └── *.traj              # Trajectory files
│   ├── final.json              # Best solution
│   ├── traj.pool               # Solution pool
│   ├── token_usage.jsonl       # Token usage log
│   └── se_framework.log        # Execution log
├── all_hist.json               # Aggregated history
├── final.json                  # All best solutions
└── total_token_usage.json      # Token statistics
```

## 🛠️ Development & Testing

```bash
# Run test suite
python SE_Perf/test/run_operator_tests.py

# Test specific operators
python SE_Perf/test/test_operators.py

# Test global memory
python SE_Perf/test/test_global_memory.py
```

## ⚠️ Important Notes

1. **Working Directory**: Commands must be executed from the project root
2. **API Configuration**: Valid API keys must be configured before running
3. **EffiBench-X Backend**: EffiBench-X evaluation service must be running
4. **Resource Limits**: Adjust `--max-parallel` based on machine capacity

## 🔗 Related Documentation

- [Main Project README](../README.md)
- [Operator Development Guide](operators/README.md)
- [PerfAgent Documentation](../perfagent/README.md)
