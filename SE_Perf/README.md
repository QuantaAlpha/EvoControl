# SE Framework 

## 🚀 Quick Start

The SE framework is a diversity experimental system based on SWE-agent that generates different solutions through multiple iterations and operator strategies.

### Immediate Use

```bash
# 1. Quick demo (recommended for first-time use)
python SE/basic_run.py --mode demo

# 2. Execute experiment (requires API key configuration)
python SE/basic_run.py --mode execute

# 3. Use custom configuration
python SE/basic_run.py --config SE/configs/se_configs/test_deepseek_se.yaml --mode execute
```

### Runtime Requirements

- **Working Directory**: Must be executed in the project root directory
- **API Configuration**: Valid API key must be set in configuration file
- **Dependencies**: SWE-agent and related dependencies must be installed

## 🎯 Core Features

### Multi-iteration Execution
- Execute multiple different solution attempts for each problem
- Each iteration uses different configurations and strategies
- Automatically generate timestamped directories to avoid conflicts

### Operator System
- **TemplateOperator**: Generate personalized system prompts
- **FilterOperator**: Generate historical filtering configurations
- Modular design, easy to extend new operators

### Intelligent Trajectory Processing
- Automatically compress trajectory files, saving 75%-87% storage space
- Intelligently extract problem descriptions as `.problem` files
- LLM-driven trajectory analysis and summarization

## 📁 Project Structure

```
SE/
├── basic_run.py              # Main entry - multi-iteration executor
├── configs/                  # Configuration files directory
│   ├── se_configs/           # SE main configurations
│   └── base_configs/         # SWE-agent base configurations
├── core/                     # Core functionality modules
│   ├── swe_iterator.py       # SWE-agent iteration runner
│   └── utils/               # Utility functions
├── operators/               # Operator system
│   ├── base.py              # Operator base class
│   └── registry.py          # Operator registration management
├── instances/               # Test instances
└── trajectories/            # Execution result output
```

## 🔧 Configuration Guide

### SE Main Configuration Files (se_configs/*.yaml)

```yaml
# Model configuration
model:
  name: "anthropic/claude-sonnet-4-20250514"
  api_base: "your_api_base"
  api_key: "your-api-key"

# Instance configuration
instances:
  json_file: "SE/instances/test.json"
  key: "instances"

# Output configuration
output_dir: "SE/trajectories/experiment_001"

# Strategy orchestration - define multiple iterations
strategy:
  iterations:
    - base_config: "test_claude"      # 1st iteration
      operator: null
    - base_config: "baseconfig1"      # 2nd iteration
      operator: null
    - base_config: "test_claude"      # 3rd iteration
      operator: "Crossover"
```

## 📊 Output Structure

Each run generates a unique output directory:

```
SE/trajectories/test_20250714_123456/
├── iteration_1/                    # First iteration
│   ├── instance_name/
│   │   ├── instance.traj           # Original trajectory
│   │   ├── instance.tra            # Compressed trajectory (saves 80%+)
│   │   ├── instance.problem        # Problem description
│   │   └── instance.pred           # Prediction results
│   └── preds.json                  # Batch results summary
├── iteration_2/                    # Second iteration
└── se_framework.log                # Framework logs
```

## 🛠️ Development Guide

### Testing System

```bash
# Run test suite
python SE/test/run_operator_tests.py

# Test specific operator
python SE/test/test_alternative_strategy.py

# Operator development testing
python SE/operator_dev.py
```

### Creating New Operators

```python
from SE.operators import TemplateOperator, register_operator

class MyOperator(TemplateOperator):
    def get_name(self):
        return "my_operator"
    
    def get_strategy_prefix(self):
        return "MY CUSTOM STRATEGY"
    
    def _generate_content(self, instance_info, problem_description, trajectory_data):
        # Implement generation logic
        return "Generated strategy content"

# Register operator
register_operator("my_operator", MyOperator)
```

## 📋 Usage Instructions

### First-time Use

1. **Run demo mode**: `python SE/basic_run.py --mode demo`
2. **Read output structure**: Understand the generated files and directories
3. **Configure API key**: Set valid API key in configuration file
4. **Execute experiment**: `python SE/basic_run.py --mode execute`

### Custom Experiments

1. **Create configuration file**: Copy and modify examples from `SE/configs/se_configs/`
2. **Configure instances**: Prepare test instances in `SE/instances/`
3. **Run experiment**: Use `--config` parameter to specify configuration file

## 🔗 Related Documentation

- Detailed development guide: `SE/test/README.md`
- Learning path: `SE/LEARNING_GUIDE.md`
- Development guide: `SE/DEVELOPMENT_GUIDE.md`
- Project architecture: `CLAUDE.md` in root directory

## ⚠️ Important Notes

- All SE-related commands must be executed in the project root directory
- Paths in configuration files are relative to the project root directory
- Valid API key configuration is required to execute experiments
- Demo mode does not consume API quota, suitable for testing

## 🆘 Troubleshooting

### Common Issues

1. **Module import errors**: Ensure running from project root directory
2. **API call failures**: Check API key and network connection
3. **Configuration file errors**: Verify YAML syntax and path correctness
4. **Permission issues**: Ensure write permissions for output directory

### Getting Help

- View log files: `SE/trajectories/*/se_framework.log`
- Run tests: `python SE/test/run_operator_tests.py`
- Consult detailed documentation: `SE/test/README.md`

---

*Start your diversity experimentation journey! 🚀*
