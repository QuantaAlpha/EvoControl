# SE Operators Development Guide

## Overview

SE Operators is a modular operator system for generating enhanced parameters between SWE-agent iterations. Based on Aeon generators design philosophy, it provides unified interfaces and powerful foundational functionality.

## Architecture Design

### System Components

1. **BaseOperator**: Base class for operators, providing common functionality
2. **TemplateOperator**: Template operator, generating system prompt templates
3. **FilterOperator**: Filter operator, generating historical filter configurations (to be implemented)
4. **OperatorRegistry**: Dynamic registration system

### Data Flow

```
Iteration 1 Complete → Iteration 2 Begins → Operator Processes Iteration 1 Data → Generate Enhanced Parameters → Iteration 2 Executes Using These
```

**Key Timing**: Operators are preprocessors that run before each iteration execution, analyzing previous iteration data to provide guidance for the current iteration.

**Configuration Example**:
```yaml
strategy:
  iterations:
    - base_config: "baseconfig1.yaml"
      operator: null                    # Iteration 1: Direct execution, no operator preprocessing
    - base_config: "baseconfig2.yaml"  
      operator: "alternative_strategy"  # Iteration 2: Use operator to process iteration 1 results before execution
    - base_config: "baseconfig3.yaml"
      operator: "traj_pool_summary"     # Iteration 3: Use operator to process iteration 1+2 results before execution
```

## Base Class Functionality Details

### BaseOperator Core Features

#### 1. LLM Integration Capability
```python
def _setup_model(self) -> None:
    """Automatically setup LLM model instance"""
    # Supports operator_models configuration, no cost limits
    # Uses sweagent's model infrastructure
```

#### 2. Instance Discovery
```python
def _discover_instances(self, workspace_dir: Path, current_iteration: int) -> List[Dict]:
    """Automatically discover processable instances"""
    # Find .tra files in iteration_{current_iteration-1} directory
    # Return list of instance information
```

#### 3. Data Extraction
```python
def _extract_problem_statement(self, trajectory_data: Dict) -> str:
    """Extract problem statement from trajectory data"""
    # Parse content within <pr_description> tags
```

#### 4. Multi-threaded Processing
```python
def process(self, workspace_dir: str, current_iteration: int, num_workers: int = 1):
    """Process multiple instances concurrently"""
    # Use ThreadPoolExecutor for parallel processing
```

#### 5. Error Handling and Logging
```python
self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")
# Complete exception catching and logging
```

### TemplateOperator Specialized Features

#### 1. Output Directory Management
```python
def _create_output_dir(self, workspace_dir: Path, current_iteration: int) -> Path:
    """Create iteration_{current_iteration}/system_prompt/ directory"""
```

#### 2. YAML Template Generation
```python
def _create_yaml_content(self, strategy_content: str) -> str:
    """Generate standard YAML format system prompts"""
    # Includes agent.templates.system_template structure
```

#### 3. Return Value Specification
```python
return {'instance_templates_dir': 'path/to/system_prompt/'}
```

## New Operator Development Guide

### Step 1: Choose Base Class

#### If generating system prompt templates → Inherit from `TemplateOperator`
```python
from SE.operators import TemplateOperator

class MyTemplateOperator(TemplateOperator):
    pass
```

#### If generating other configurations → Inherit from `BaseOperator`
```python
from SE.operators import BaseOperator

class MyCustomOperator(BaseOperator):
    pass
```

### Step 2: Implement Required Methods

#### For TemplateOperator

```python
class MyTemplateOperator(TemplateOperator):
    def get_name(self) -> str:
        """Return operator name"""
        return "my_template"
    
    def get_strategy_prefix(self) -> str:
        """Return strategy prefix identifier"""
        return "MY SOLUTION STRATEGY"
    
    def _generate_content(self, instance_info: Dict, problem_statement: str, trajectory_data: Dict) -> str:
        """Generate strategy content (core logic)"""
        # Implement your operator logic here
        # Can call self._call_llm_api() to use LLM
        # Can access self.logger for logging
        
        prompt = f"Generate solution strategy for the following problem:\n{problem_statement}"
        strategy = self._call_llm_api(prompt)
        return strategy
```

#### For BaseOperator

```python
class MyCustomOperator(BaseOperator):
    def get_name(self) -> str:
        return "my_custom"
    
    def _generate_content(self, instance_info: Dict, problem_statement: str, trajectory_data: Dict) -> str:
        """Generate content"""
        pass
    
    def process(self, workspace_dir: str, current_iteration: int, num_workers: int = 1) -> Optional[Dict[str, str]]:
        """Complete processing logic"""
        # Custom processing flow
        # Return corresponding parameter dictionary
        return {'custom_param': 'value'}
```

### Step 3: Register Operator

```python
from SE.operators import register_operator

# Register operator
register_operator("my_template", MyTemplateOperator)
```

### Step 4: Test Operator

```python
# Test using operator_dev.py
python SE/operator_dev.py --test-llm

# Or create custom tests
from SE.operators import create_operator

config = {...}  # Configuration dictionary
operator = create_operator("my_template", config)
result = operator.process(workspace_dir, current_iteration, num_workers)
```

## Specific Operator Implementation Examples

### Existing Operators Overview

| Operator Name | Function | Data Source | Output Prefix | Applicable Timing |
|---------------|----------|-------------|---------------|-------------------|
| `alternative_strategy` | Generate alternative solutions | Most recent failed attempt | ALTERNATIVE SOLUTION STRATEGY | Iteration 2 (based on iteration 1 failure) |
| `traj_pool_summary` | Risk-aware comprehensive analysis | All historical attempts | RISK-AWARE PROBLEM SOLVING GUIDANCE | Iteration 3+ (comprehensive historical analysis) |

### Operator Execution Timing Details

```
Timeline:
T1: Execute iteration 1 (baseconfig1.yaml, operator: null)
    → Produces iteration 1 trajectories and traj.pool data

T2: Operator preprocessing (alternative_strategy processes iteration 1 data)
    → Generate iteration_2/system_prompt/*.yaml
    → Execute iteration 2 (baseconfig2.yaml + operator-generated system prompts)
    → Produces iteration 2 trajectories and traj.pool data

T3: Operator preprocessing (traj_pool_summary processes iteration 1+2 data)
    → Generate iteration_3/system_prompt/*.yaml  
    → Execute iteration 3 (baseconfig3.yaml + operator-generated system prompts)
```

**Core Principles**:
- Operators run **before** iteration execution, as preprocessors
- Operators analyze data from **all previous iterations**
- Operators generate enhanced system prompts for **current iteration**

### Example 1: AlternativeStrategy Operator

Generate orthogonal alternative strategies based on most recent failed attempt:

```python
class AlternativeStrategyOperator(TemplateOperator):
    def get_name(self) -> str:
        return "alternative_strategy"
    
    def get_strategy_prefix(self) -> str:
        return "ALTERNATIVE SOLUTION STRATEGY"
    
    def _generate_content(self, instance_info, problem_statement, trajectory_data):
        # Load traj.pool to get failed methods
        instance_dir = instance_info['instance_dir']
        previous_iteration = instance_info['previous_iteration']
        
        traj_pool = self._load_traj_pool(instance_dir)
        previous_approach = traj_pool.get(str(previous_iteration), "")
        
        if not previous_approach:
            return ""
        
        # Generate alternative strategy
        return self._generate_alternative_strategy(problem_statement, previous_approach)
    
    def _load_traj_pool(self, instance_dir: Path) -> Dict[str, str]:
        """Load strategy pool"""
        # Implement traj.pool loading logic
        pass
    
    def _generate_alternative_strategy(self, problem_statement: str, previous_approach: str) -> str:
        """Use LLM to generate alternative strategy"""
        system_prompt = """You are a software engineering strategy expert..."""
        prompt = f"""Generate alternative strategy:\nProblem: {problem_statement}\nFailed method: {previous_approach}"""
        return self._call_llm_api(prompt, system_prompt)
```

### Example 2: TrajPoolSummary Operator

Generate risk-aware guidance based on all historical attempts:

```python
class TrajPoolSummaryOperator(TemplateOperator):
    def get_name(self) -> str:
        return "traj_pool_summary"
    
    def get_strategy_prefix(self) -> str:
        return "RISK-AWARE PROBLEM SOLVING GUIDANCE"
    
    def _generate_content(self, instance_info, problem_statement, trajectory_data):
        # Load all historical attempt data
        approaches_data = self._load_traj_pool(instance_info['instance_dir'])
        
        # Generate risk-aware guidance (keep within 200 words)
        guidance = self._generate_risk_aware_guidance(problem_statement, approaches_data)
        return guidance
    
    def _generate_risk_aware_guidance(self, problem_statement: str, approaches_data: Dict) -> str:
        """Generate concise risk-aware guidance"""
        # Use LLM to analyze historical failure patterns, generate blind spot identification and risk avoidance strategies
        # Output format: BLIND SPOTS TO AVOID + CRITICAL RISKS + STRATEGIC APPROACH
        pass
```

## Configuration Requirements

### operator_models Configuration

Add operator-specific model configuration in config file:

```yaml
# SE configuration file
operator_models:
  name: "anthropic/claude-sonnet-4-20250514"
  api_base: "your_api_base"
  api_key: "your-api-key"
  temperature: 0.0
  max_output_tokens: 4000

# Or use default model configuration
model:
  name: "openai/deepseek-chat"
  # ...
```

### Iteration Configuration

Use operators in strategy configuration:

```yaml
strategy:
  iterations:
    - base_config: "SE/configs/base_configs/baseconfig1.yaml"
      operator: null
    - base_config: "SE/configs/base_configs/baseconfig2.yaml"
      operator: "alternative_strategy"  # Use registered operator name
```

## Best Practices

### 1. Logging Usage
```python
self.logger.info("Starting instance processing")
self.logger.debug("Detailed debug information")
self.logger.warning("Warning information")
self.logger.error("Error information")
```

### 2. Error Handling
```python
try:
    result = self._some_operation()
except Exception as e:
    self.logger.error(f"Operation failed: {e}")
    return None
```

### 3. LLM Calls
```python
# Simple call
response = self._call_llm_api(prompt)

# Call with system prompt
response = self._call_llm_api(prompt, system_prompt)

# Check response
if not response:
    self.logger.warning("LLM call failed")
    return default_content
```

### 4. Performance Optimization
- Operators internally support multi-threading, no additional optimization needed
- LLM calls automatically reuse model instances
- Use logger.debug() for detailed information, avoid excessive printing

## Debugging Tips

### 1. Use Development Scripts
```bash
# Test operator basic functionality
python SE/operator_dev.py

# Test LLM connection
python SE/operator_dev.py --test-llm

# Specify configuration file
python SE/operator_dev.py --config custom_config.yaml
```

### 2. View Logs
```bash
# Log file location
SE/trajectories/operator_dev_test/test_*/se_framework.log
```

### 3. Check Output Files
```bash
# Template file location
SE/trajectories/*/iteration_*/system_prompt/*.yaml
```

## Common Issues

### Q: How to access historical strategy data?
A: Use `_load_traj_pool()` method to load traj.pool file, which contains all historical strategies.

### Q: How to handle LLM call failures?
A: `_call_llm_api()` returns empty string when failing, should provide default strategy or skip that instance.

### Q: How to customize output format?
A: Inherit from BaseOperator and override `process()` method, return custom parameter dictionary.

### Q: How to add new data sources?
A: Add custom data loading logic in `_generate_content()`.

## Extended Features

### 1. Custom Data Extraction
```python
def _extract_custom_data(self, trajectory_data: Dict) -> Any:
    """Extract custom data"""
    # Implement specific data extraction logic
    pass
```

### 2. Multi-stage Strategy Generation
```python
def _generate_content(self, instance_info, problem_statement, trajectory_data):
    # Stage 1: Analysis
    analysis = self._analyze_problem(problem_statement)
    
    # Stage 2: Strategy generation
    strategy = self._generate_strategy(analysis)
    
    # Stage 3: Optimization
    optimized_strategy = self._optimize_strategy(strategy)
    
    return optimized_strategy
```

### 3. Conditional Processing
```python
def _generate_content(self, instance_info, problem_statement, trajectory_data):
    if self._is_bug_fix(problem_statement):
        return self._generate_bug_fix_strategy(...)
    elif self._is_feature_request(problem_statement):
        return self._generate_feature_strategy(...)
    else:
        return self._generate_general_strategy(...)
```

This guide provides a complete operator development process and best practices. Based on this framework, you can quickly develop various specialized operators to enhance SWE-agent's iterative capabilities.