"""
PerfAgent 配置系统

模仿 sweagent 的配置结构，提供灵活的 agent 参数配置。
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# 内联所有配置类，移除独立的 config_components 目录


@dataclass
class ModelConfig:
    """模型配置"""

    name: str = "gpt-4"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_input_tokens: int = 4000
    max_output_tokens: int = 4000
    use_llm: bool = False
    # LLM 客户端增强参数
    request_timeout: float = 60.0
    max_retries: int = 10
    retry_delay: float = 2.0
    retry_backoff_factor: float = 2.0
    retry_jitter: float = 0.5
    log_inputs_outputs: bool = True
    log_sanitize: bool = True


@dataclass
class OptimizationConfig:
    """优化方向配置"""

    target: Literal["runtime", "memory", "integral"] = "runtime"  # 允许: "runtime" 或 "memory" 或 "integral"
    enable_memory_checks: bool = True
    enable_runtime_checks: bool = True
    adopt_only_if_improved: bool = False
    code_generation_mode: Literal["diff", "direct"] = "diff"
    # 是否在 metrics 中包含非 target 的其他指标（如 runtime, memory, integral）
    include_other_metrics_in_summary: bool = True


@dataclass
class RuntimeConfig:
    """运行时资源限制"""

    time_limit: int = 20  # 秒
    memory_limit: int = 1024  # MB
    max_workers: int = 4
    num_runs: int = 10
    trim_ratio: float = 0.1


@dataclass
class LoggingConfig:
    save_trajectory: bool = True
    trajectory_dir: Path = Path("./trajectories")
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"


@dataclass
class PromptConfig:
    system_template: str = ""
    optimization_template: str = ""
    # 额外的系统提示内容：用于格式化到模板中的 {additional_requirements}
    additional_requirements: str | None = None
    # 实例系统模板附加内容目录（包含若干 YAML 文件）（兼容旧流程）
    instance_templates_dir: Path | None = None
    include_all_history: bool = False
    local_memory: str | None = None
    global_memory: str | None = None


@dataclass
class LanguageConfig:
    language: str = "python3"
    supported_languages: list[str] = field(default_factory=lambda: ["python3", "cpp", "java", "javascript", "golang"])


@dataclass
class OverridesConfig:
    """可选的覆盖项配置

    - initial_code_dir: 指定每实例初始代码的目录（按实例名匹配文件）
    - initial_code_text: 直接提供初始代码文本（优先于目录）
    """

    initial_code_dir: Path | None = None
    initial_code_text: str | None = None


@dataclass
class PerfAgentConfig:
    """PerfAgent 配置类（模块化集成）"""

    # 迭代控制
    max_iterations: int = 10
    # 早停控制：连续未改进达到阈值后停止；0 表示不启用
    early_stop_no_improve: int = 0
    # 顶层并发控制（用于批量并行运行 run.py 子进程）
    max_workers: int = 4

    # 组件配置
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    language_cfg: LanguageConfig = field(default_factory=LanguageConfig)
    overrides: OverridesConfig = field(default_factory=OverridesConfig)

    def __post_init__(self):
        """初始化后处理，将旧字段同步到新组件，并验证目录与模板"""
        # 验证参数
        self._validate_config()

        # 确保目录存在
        self.logging.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)

        # 加载默认模板（组件字段）
        if not self.prompts.system_template:
            self.prompts.system_template = self._load_default_system_template()
        if not self.prompts.optimization_template:
            self.prompts.optimization_template = self._load_default_optimization_template()

    def apply_cli_overrides(self, args: Any) -> None:
        """根据 CLI 参数覆盖配置，仅在单一配置文件中完成所有映射。"""
        # 基础覆盖
        if getattr(args, "max_iterations", None) is not None:
            self.max_iterations = args.max_iterations
        if getattr(args, "max_workers", None) is not None:
            self.max_workers = args.max_workers
        if getattr(args, "model", None):
            self.model.name = args.model
        if getattr(args, "trajectory_dir", None):
            self.logging.trajectory_dir = args.trajectory_dir
        if getattr(args, "log_dir", None):
            self.logging.log_dir = args.log_dir
        if getattr(args, "log_level", None):
            self.logging.log_level = args.log_level

        # 语言与优化方向
        if getattr(args, "language", None):
            self.language_cfg.language = args.language
        if getattr(args, "opt_target", None):
            self.optimization.target = args.opt_target
        if getattr(args, "include_other_metrics", None) is not None:
            self.optimization.include_other_metrics_in_summary = args.include_other_metrics

        # LLM 客户端配置
        if getattr(args, "llm_use", False):
            self.model.use_llm = True
        if getattr(args, "llm_api_base", None):
            self.model.api_base = args.llm_api_base
        if getattr(args, "llm_api_key", None):
            self.model.api_key = args.llm_api_key
        if getattr(args, "llm_model", None):
            self.model.name = args.llm_model
        if getattr(args, "llm_temp", None) is not None:
            self.model.temperature = args.llm_temp
        if getattr(args, "llm_max_output", None) is not None:
            self.model.max_output_tokens = args.llm_max_output
        # LLM 请求控制与日志开关
        if getattr(args, "llm_timeout", None) is not None:
            self.model.request_timeout = args.llm_timeout
        if getattr(args, "llm_max_retries", None) is not None:
            self.model.max_retries = args.llm_max_retries
        if getattr(args, "llm_retry_delay", None) is not None:
            self.model.retry_delay = args.llm_retry_delay
        if getattr(args, "llm_retry_backoff", None) is not None:
            self.model.retry_backoff_factor = args.llm_retry_backoff
        if getattr(args, "llm_retry_jitter", None) is not None:
            self.model.retry_jitter = args.llm_retry_jitter
        if getattr(args, "llm_log_io", False):
            self.model.log_inputs_outputs = True
        if getattr(args, "llm_log_sanitize", False):
            self.model.log_sanitize = True
        # 早停
        if getattr(args, "early_stop_no_improve", None) is not None:
            self.early_stop_no_improve = args.early_stop_no_improve

        # 兼容：从 CLI 接收实例模板目录（迁移自 run.py），规范化为 Path
        if getattr(args, "instance_templates_dir", None) is not None and self.prompts:
            itd = args.instance_templates_dir
            if isinstance(itd, (str, Path)):
                self.prompts.instance_templates_dir = Path(itd)

        # 初始代码目录（按实例名匹配），规范化为 Path
        if getattr(args, "initial_code_dir", None) is not None and self.overrides is not None:
            icd = args.initial_code_dir
            if isinstance(icd, (str, Path)):
                self.overrides.initial_code_dir = Path(icd)

    def _validate_config(self):
        """验证配置参数"""
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if not (0.0 <= self.model.temperature <= 1.0):
            raise ValueError("model.temperature must be between 0.0 and 1.0")
        if self.runtime.num_runs < 1:
            raise ValueError("runtime.num_runs must be at least 1")
        # 优化方向校验
        if self.optimization.target not in ("runtime", "memory", "integral"):
            raise ValueError("optimization.target must be 'runtime' or 'memory' or 'integral'")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（嵌套组件序列化，并处理 Path）"""
        data = asdict(self)
        # 处理嵌套 Path 字段
        if "logging" in data:
            if isinstance(data["logging"].get("log_dir"), Path):
                data["logging"]["log_dir"] = str(data["logging"]["log_dir"])
            if isinstance(data["logging"].get("trajectory_dir"), Path):
                data["logging"]["trajectory_dir"] = str(data["logging"]["trajectory_dir"])
        # 处理 prompts 中的 Path 字段
        if "prompts" in data:
            itd = data["prompts"].get("instance_templates_dir")
            if isinstance(itd, Path):
                data["prompts"]["instance_templates_dir"] = str(itd)
            # additional_requirements 为普通字符串，无需特殊处理
        # 处理 overrides 中的 Path 字段
        if "overrides" in data and isinstance(data["overrides"], dict):
            icd = data["overrides"].get("initial_code_dir")
            if isinstance(icd, Path):
                data["overrides"]["initial_code_dir"] = str(icd)
        return data

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PerfAgentConfig":
        """从字典创建配置（严格使用嵌套组件键，不再支持旧顶层键）"""
        cfg = config_dict.copy() if config_dict else {}

        # model（仅从嵌套 model 读取）
        model_dict = cfg.get("model", {}) or {}
        # 兼容迁移：若旧配置在 model.include_all_history，则迁移到 prompts.include_all_history
        prompts_dict_migration = cfg.get("prompts", {}) or {}
        if "include_all_history" in model_dict and "include_all_history" not in prompts_dict_migration:
            try:
                prompts_dict_migration["include_all_history"] = bool(model_dict.pop("include_all_history"))
                cfg["prompts"] = prompts_dict_migration
            except Exception:
                model_dict.pop("include_all_history", None)
        model_cfg = ModelConfig(**model_dict)

        # runtime（仅从嵌套 runtime 读取）
        runtime_dict = cfg.get("runtime", {}) or {}
        runtime_cfg = RuntimeConfig(**runtime_dict)

        # logging（仅从嵌套 logging 读取，并处理路径类型）
        logging_dict = cfg.get("logging", {}) or {}
        if "trajectory_dir" in logging_dict and isinstance(logging_dict["trajectory_dir"], str):
            logging_dict["trajectory_dir"] = Path(logging_dict["trajectory_dir"])
        if "log_dir" in logging_dict and isinstance(logging_dict["log_dir"], str):
            logging_dict["log_dir"] = Path(logging_dict["log_dir"])
        logging_cfg = LoggingConfig(**logging_dict)

        # prompts（仅从嵌套 prompts 读取）
        prompts_dict = cfg.get("prompts", {}) or {}
        # 规范化路径类型
        if "instance_templates_dir" in prompts_dict and isinstance(prompts_dict["instance_templates_dir"], str):
            prompts_dict["instance_templates_dir"] = Path(prompts_dict["instance_templates_dir"])
        prompts_cfg = PromptConfig(**prompts_dict)

        # optimization（仅从嵌套 optimization 读取）
        optimization_dict = cfg.get("optimization", {}) or {}
        optimization_cfg = OptimizationConfig(**optimization_dict)

        # language（仅从嵌套 language_cfg 读取）
        language_dict = cfg.get("language_cfg", {}) or {}
        language_cfg = LanguageConfig(**language_dict)

        # overrides（可选嵌套）
        overrides_dict = cfg.get("overrides", {}) or {}
        if "initial_code_dir" in overrides_dict and isinstance(overrides_dict["initial_code_dir"], str):
            overrides_dict["initial_code_dir"] = Path(overrides_dict["initial_code_dir"])
        overrides_cfg = OverridesConfig(**overrides_dict)

        # 顶层允许的键
        max_iterations = cfg.get("max_iterations", 10)
        early_stop_no_improve = cfg.get("early_stop_no_improve", 0)
        max_workers = cfg.get("max_workers", 4)

        return cls(
            max_iterations=max_iterations,
            model=model_cfg,
            optimization=optimization_cfg,
            runtime=runtime_cfg,
            logging=logging_cfg,
            prompts=prompts_cfg,
            language_cfg=language_cfg,
            early_stop_no_improve=early_stop_no_improve,
            max_workers=max_workers,
            overrides=overrides_cfg,
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PerfAgentConfig":
        """从 YAML 文件加载配置（严格使用嵌套键）"""
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        return cls.from_dict(config_data)

    def to_yaml(self, config_path: Path) -> None:
        """保存配置到 YAML 文件（序列化嵌套组件）"""
        config_data = self.to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    def _load_default_system_template(self) -> str:
        """加载默认系统模板"""
        if self.optimization.code_generation_mode == "direct":
            return """你是一个专业的代码性能优化专家。你的任务是分析给定的代码，识别性能瓶颈，并提供优化建议。

你需要：
1. 仔细分析当前代码的性能问题
2. 考虑算法复杂度、数据结构选择、内存使用因素
3. 提供具体的优化方案，直接输出优化后的完整代码
4. 确保优化后的代码功能正确性不变
5. 优先考虑时间复杂度的改进

请始终保持代码的可读性和可维护性。

请输出完整代码，包含在 Markdown 代码块中（例如 ```python ... ```）。
"""
        return """你是一个专业的代码性能优化专家。你的任务是分析给定的代码，识别性能瓶颈，并提供优化建议。

你需要：
1. 仔细分析当前代码的性能问题
2. 考虑算法复杂度、数据结构选择、内存使用因素
3. 提供具体的优化方案，以 SEARCH/REPLACE 区块格式输出代码修改
4. 确保优化后的代码功能正确性不变
5. 优先考虑时间复杂度和空间复杂度的改进

请始终保持代码的可读性和可维护性。

严格输出如下格式的区块：
<<<<<<< SEARCH
（在原代码中需要完全匹配的连续片段，保持缩进与空格一致）
=======
（替换为的新代码片段）
>>>>>>> REPLACE
"""

    def _load_default_optimization_template(self) -> str:
        """加载默认优化模板"""
        if self.optimization.code_generation_mode == "direct":
            return """基于以下信息，请优化代码性能：

当前代码：
```{language}
{current_code}
```

性能分析结果：
{performance_analysis}

历史优化记录：
{optimization_history}

请提供优化方案，格式要求：
1. 简要说明优化思路（中文说明）
2. 直接输出优化后的完整代码，使用 Markdown 代码块包裹。

优化方案：
"""
        return """基于以下信息，请优化代码性能：

当前代码：
```{language}
{current_code}
```

性能分析结果：
{performance_analysis}

历史优化记录：
{optimization_history}

请提供优化方案，格式要求：
1. 简要说明优化思路（中文说明）
2. 仅输出一个或多个 SEARCH/REPLACE 区块，严格遵守以下格式：

<<<<<<< SEARCH
（原代码中完整存在的片段，必须精确匹配，包括换行与空格）
=======
（替换为的新代码片段）
>>>>>>> REPLACE

注意：
- 可以包含多个区块，按实际需要逐个给出；
- 不要输出 ```diff 或 @@ 格式；
- 不要在区块中加入额外解释文字；

优化方案：
"""


def load_config(config_path: Path | None = None) -> PerfAgentConfig:
    """加载配置文件"""
    if config_path and Path(config_path).exists():
        return PerfAgentConfig.from_yaml(Path(config_path))
    env_config_path = os.getenv("PERFAGENT_CONFIG")
    if env_config_path:
        env_path = Path(env_config_path)
        if env_path.exists():
            return PerfAgentConfig.from_yaml(env_path)
    return PerfAgentConfig()
