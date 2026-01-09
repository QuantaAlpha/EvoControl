"""
PerfAgent 核心类

实现代码性能优化的主要逻辑，包括迭代优化、diff 应用、性能评估等功能。
"""

import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PerfAgentConfig
from .diff_applier import DiffApplier
from .effibench.benchmark import run_performance_benchmark
from .effibench.utils import EFFIBENCH_REGISTRY
from .llm_client import LLMClient
from .trajectory import TrajectoryLogger
from .utils.log import get_se_logger


@dataclass
class EffiBenchXInstance:
    id: str
    title: str
    title_slug: str
    description: str
    description_md: str
    source: str
    url: str
    type: str
    starter_code: str | None = None
    solutions: dict[str, dict[str, str]] = field(default_factory=dict)
    language: str | None = None
    generated_tests: list[dict[str, Any]] = field(default_factory=list)
    evaluator: str | None = None
    test_runners: dict[str, str] = field(default_factory=dict)
    # 任务名（来源于实例文件名，不含扩展名）
    task_name: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EffiBenchXInstance":
        # Robustly parse generated_tests when it can be a list or a JSON string
        gt_raw = data.get("generated_tests", [])
        if isinstance(gt_raw, str):
            try:
                gt_parsed = json.loads(gt_raw)
            except Exception:
                gt_parsed = []
        elif isinstance(gt_raw, list):
            gt_parsed = gt_raw
        else:
            gt_parsed = []

        # Robustly parse test_runners when it can be a dict or a JSON string
        tr_raw = data.get("test_runners", {})
        if isinstance(tr_raw, str):
            try:
                tr_parsed = json.loads(tr_raw)
            except Exception:
                tr_parsed = {}
        elif isinstance(tr_raw, dict):
            tr_parsed = tr_raw
        else:
            tr_parsed = {}

        return EffiBenchXInstance(
            id=str(data.get("id", "unknown")),
            title=data.get("title", ""),
            title_slug=data.get("title_slug", ""),
            description=data.get("description", ""),
            description_md=data.get("description_md", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            type=data.get("type", ""),
            starter_code=data.get("starter_code"),
            solutions=data.get("solutions", {}),
            language=data.get("language"),
            generated_tests=gt_parsed,
            evaluator=data.get("evaluator"),
            test_runners=tr_parsed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "title_slug": self.title_slug,
            "description": self.description,
            "description_md": self.description_md,
            "source": self.source,
            "url": self.url,
            "type": self.type,
            "starter_code": self.starter_code,
            "solutions": self.solutions,
            "language": self.language,
            "generated_tests": self.generated_tests,
            "evaluator": self.evaluator,
            "test_runners": self.test_runners,
            "task_name": self.task_name,
        }


@dataclass
class RunContext:
    """保存单次运行的上下文状态"""

    instance: EffiBenchXInstance
    trajectory: TrajectoryLogger
    language: str
    optimization_target: str
    initial_code: str
    current_code: str
    best_code: str
    best_performance: float
    best_pass_rate: float
    current_benchmark_results: dict[str, Any]
    best_benchmark_results: dict[str, Any]
    optimization_history: list[dict[str, Any]]
    iter_offset: int
    no_improve_count: int = 0
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    initial_performance_value: float = float("inf")


class PerfAgent:
    """性能优化 Agent"""

    def __init__(self, config: PerfAgentConfig):
        self.config = config

        # 简化逻辑：凭据存在即初始化 LLMClient，无需 use_llm 标志
        self.llm_client = None
        if self.config.model.api_base and self.config.model.api_key:
            client_cfg = {
                "name": self.config.model.name,
                "api_base": self.config.model.api_base,
                "api_key": self.config.model.api_key,
                "max_output_tokens": self.config.model.max_output_tokens,
                "request_timeout": self.config.model.request_timeout,
                "max_retries": self.config.model.max_retries,
                "retry_delay": self.config.model.retry_delay,
                "retry_backoff_factor": getattr(self.config.model, "retry_backoff_factor", 2.0),
                "retry_jitter": getattr(self.config.model, "retry_jitter", 0.5),
                "log_inputs_outputs": self.config.model.log_inputs_outputs,
                "log_sanitize": self.config.model.log_sanitize,
            }
            # 将 LLM I/O 独立写入 log_dir/llm_io.log
            io_log_file = Path(self.config.logging.log_dir) / "llm_io.log"
            self.llm_client = LLMClient(
                client_cfg,
                io_log_path=io_log_file,
                log_inputs_outputs=self.config.model.log_inputs_outputs,
                log_sanitize=self.config.model.log_sanitize,
                request_timeout=self.config.model.request_timeout,
            )

        self.diff_applier = DiffApplier()

        # 设置日志：统一绑定到单一文件
        # 使用包含日志目录名的唯一 logger 名称，避免并发实例复用同名导致串写
        agent_logger_name = f"perfagent.agent.{Path(self.config.logging.log_dir).name}"
        get_se_logger(
            agent_logger_name,
            Path(self.config.logging.log_dir) / "perfagent.log",
            emoji="🔧",
            level=getattr(logging, self.config.logging.log_level.upper()),
            also_stream=False,
        )
        self.logger = logging.getLogger(agent_logger_name)

        # 优化历史
        self.optimization_history: list[dict[str, Any]] = []

        # 初始代码来源："default" | "text" | "dir"
        self._initial_code_source: str = "default"

    def _normalize_language(self, lang: str | None) -> str:
        # 标准化语言名称
        if not lang:
            return "python3"
        l = lang.lower()
        if l in ("python", "py", "python3"):
            return "python3"
        if l in ("cpp", "c++", "cxx"):
            return "cpp"
        if l in ("javascript", "js"):
            return "javascript"
        if l in ("java",):
            return "java"
        return l

    def _get_default_placeholder(self, language: str | None = None) -> str:
        """获取默认占位符代码（根据语言）"""
        lang = self._normalize_language(language or self.config.language_cfg.language)
        placeholder_map = {
            "python3": "# Start your code here\n",
            "cpp": "// Start your code here\n",
            "java": "// Start your code here\n",
            "javascript": "// Start your code here\n",
            "golang": "// Start your code here\n",
        }
        return placeholder_map.get(lang, "# Start your code here\n")

    def _extract_initial_code(
        self, instance: EffiBenchXInstance, language: str | None = None, optimization_target: str | None = None
    ) -> str:
        """从配置/文件系统注入或生成初始代码。

        优先级：
        1) 配置 overrides.initial_code_text（直接文本）
        2) 配置 overrides.initial_code_dir（按实例名匹配文件）
        3) 默认占位符代码（根据语言）
        """
        try:
            # 默认来源
            self._initial_code_source = "default"
            # 1) 直接文本覆盖
            override_text = getattr(getattr(self.config, "overrides", None), "initial_code_text", None)
            if isinstance(override_text, str) and override_text.strip():
                self._initial_code_source = "text"
                return override_text if override_text.endswith("\n") else override_text + "\n"

            # 2) 目录覆盖（按实例名匹配文件）
            code_dir = getattr(getattr(self.config, "overrides", None), "initial_code_dir", None)
            task_name = getattr(instance, "task_name", None) or getattr(instance, "id", None)
            if code_dir and task_name:
                lang = self._normalize_language(language or self.config.language_cfg.language)
                # 语言扩展映射
                ext_map = {
                    "python3": [".py"],
                    "cpp": [".cpp", ".cc", ".cxx"],
                    "java": [".java"],
                    "javascript": [".js", ".mjs"],
                    "golang": [".go"],
                }
                candidates: list[Path] = []
                for ext in ext_map.get(lang, []):
                    candidates.append(Path(code_dir) / f"{task_name}{ext}")
                # 退化：任意匹配同名文件（不区分扩展名）
                try:
                    for fp in Path(code_dir).iterdir():
                        if fp.is_file() and fp.stem == task_name and fp not in candidates:
                            candidates.append(fp)
                except Exception:
                    pass

                for fp in candidates:
                    try:
                        if fp.exists():
                            code = fp.read_text(encoding="utf-8")
                            if isinstance(code, str) and code.strip():
                                self.logger.info(f"使用覆盖初始代码: {fp}")
                                self._initial_code_source = "dir"
                                return code if code.endswith("\n") else code + "\n"
                    except Exception as e:
                        self.logger.warning(f"读取初始代码文件失败 {fp}: {e}")
        except Exception as e:
            # 覆盖流程失败则回退到占位符
            self.logger.warning(f"初始代码覆盖失败，使用默认占位符: {e}")

        # 3) 默认占位符（保持现有测试兼容）
        return self._get_default_placeholder(language)

    def _resolve_starter_code(self, instance: EffiBenchXInstance, language: str) -> str | None:
        sc = getattr(instance, "starter_code", None)
        if isinstance(sc, dict):
            try:
                return sc.get(language)
            except Exception:
                return None
        if isinstance(sc, str):
            return sc
        return None

    def _resolve_test_runner(self, instance: EffiBenchXInstance, language: str) -> str | None:
        trs = getattr(instance, "test_runners", None)
        if isinstance(trs, dict):
            try:
                lang_norm = self._normalize_language(language)
                val = trs.get(lang_norm)
                if isinstance(val, str) and val.strip():
                    return val
            except Exception:
                return None
        if isinstance(trs, str):
            try:
                parsed = json.loads(trs)
                if isinstance(parsed, dict):
                    val = parsed.get(self._normalize_language(language))
                    if isinstance(val, str) and val.strip():
                        return val
            except Exception:
                pass
        return None

    def _prepare_test_cases(self, instance: EffiBenchXInstance) -> list[dict[str, Any]]:
        """准备测试用例（实例仅为 dataclass）"""
        return instance.generated_tests or []

    def _detect_language(self, instance: EffiBenchXInstance) -> str:
        """检测编程语言（仅保留以兼容调用路径，但不使用）"""
        return self._normalize_language(self.config.language_cfg.language)

    def _create_empty_performance_metrics(self) -> dict[str, Any]:
        """创建一个空的性能分析指标结构"""
        return {
            "original_n": 0,
            "n": 0,
            "runtime": float("inf"),
            "memory": float("inf"),
            "integral": float("inf"),
            "pass_rate": 0.0,
            "passed": False,
            "analysis": {
                "runtime": self._create_empty_metric_analysis(),
                "memory": self._create_empty_metric_analysis(),
                "integral": self._create_empty_metric_analysis(),
            },
        }

    def _create_empty_metric_analysis(self) -> dict[str, Any]:
        """创建一个空的单项指标分析结构"""
        return {
            "original_n": 0,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }

    def _create_default_performance_result(self, consistent: bool = True) -> dict[str, Any]:
        """创建默认的性能评估结果结构"""
        return {
            "performance_analysis": self._create_empty_performance_metrics(),
            "first_run_details": [],
            "failed_test_details": [],
            "failed_submission_exit_codes": [],
            "pass_rates": [],
            "pass_rate_consistent": consistent,
        }

    def _evaluate_performance(
        self, language: str, code: str, test_cases: list[dict], instance: EffiBenchXInstance
    ) -> dict[str, Any]:
        """评估代码性能，保持参数兼容"""
        eval_start_time = time.time()
        self.logger.info(
            f"[性能评估开始] 时间: {datetime.now().strftime('%H:%M:%S')}, "
            f"测试用例数: {len(test_cases)}, 代码长度: {len(code)} 字符"
        )

        # 如果代码与占位符代码相同，返回默认失败结构
        if code == self._get_default_placeholder(language):
            self.logger.info("[性能评估跳过] 代码为占位符，返回默认失败结构")
            return self._create_default_performance_result(consistent=True)

        # 若 evaluator 或测试用例缺失/格式不合法，直接返回默认结构以避免长时间的后端调用
        evaluator = getattr(instance, "evaluator", None)
        tc_valid = bool(test_cases) and isinstance(test_cases, list) and isinstance(test_cases[0], dict)
        if not evaluator or not tc_valid:
            self.logger.warning(
                f"[性能评估跳过] 缺少必要组件 - evaluator: {bool(evaluator)}, test_cases有效: {tc_valid}"
            )
            return self._create_default_performance_result(consistent=True)
        test_runner = self._resolve_test_runner(instance, language)

        # 级联评估：先用 benchmark 进行一次运行（num_runs=1），若未全部通过则直接返回
        single_run_start = time.time()
        self.logger.info("[单次预运行开始] 验证代码正确性...")
        try:
            single_run_summary = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                num_runs=1,
                time_limit=self.config.runtime.time_limit,
                memory_limit=self.config.runtime.memory_limit,
                trim_ratio=self.config.runtime.trim_ratio,
                max_workers=self.config.runtime.max_workers,
            )
            single_run_elapsed = time.time() - single_run_start
            pass_rate = single_run_summary.get("performance_analysis", {}).get("pass_rate", 0)
            self.logger.info(f"[单次预运行完成] 耗时: {single_run_elapsed:.2f}s, 通过率: {pass_rate:.2%}")
        except Exception as e:
            single_run_elapsed = time.time() - single_run_start
            self.logger.warning(
                f"[单次预运行失败] 耗时: {single_run_elapsed:.2f}s, 错误类型: {type(e).__name__}, "
                f"错误信息: {e}\n{traceback.format_exc()}"
            )
            return self._create_default_performance_result(consistent=True)

        # 计算单次运行通过率（优先使用返回的 pass_rates）
        passed = single_run_summary.get("performance_analysis").get("passed", False)

        # 若未全部通过，直接返回单次运行的结果（不进行多次性能评估）
        if not passed:
            total_elapsed = time.time() - eval_start_time
            failed_count = len(single_run_summary.get("failed_test_details", []))
            self.logger.info(
                f"[性能评估提前结束] 代码未全部通过测试，失败用例数: {failed_count}, 总耗时: {total_elapsed:.2f}s"
            )
            return single_run_summary

        # 若重复运行次数为 1，直接返回单次运行的结果，无需进行级联评估
        if self.config.runtime.num_runs == 1:
            total_elapsed = time.time() - eval_start_time
            perf_analysis = single_run_summary.get("performance_analysis", {})
            self.logger.info(
                f"[性能评估完成] 单次运行模式, 总耗时: {total_elapsed:.2f}s, "
                f"runtime: {perf_analysis.get('runtime', 'N/A')}s, "
                f"memory: {perf_analysis.get('memory', 'N/A')}MB"
            )
            return single_run_summary

        # 所有测试用例通过，进行正式的多次性能评估
        multi_run_start = time.time()
        self.logger.info(
            f"[多次评估开始] 运行次数: {self.config.runtime.num_runs}, "
            f"time_limit: {self.config.runtime.time_limit}s, memory_limit: {self.config.runtime.memory_limit}MB"
        )
        try:
            result = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                num_runs=self.config.runtime.num_runs,
                time_limit=self.config.runtime.time_limit,
                memory_limit=self.config.runtime.memory_limit,
                trim_ratio=self.config.runtime.trim_ratio,
                max_workers=self.config.runtime.max_workers,
            )
            multi_run_elapsed = time.time() - multi_run_start
            total_elapsed = time.time() - eval_start_time
            perf_analysis = result.get("performance_analysis", {})
            self.logger.info(
                f"[多次评估完成] 多次运行耗时: {multi_run_elapsed:.2f}s, 总评估耗时: {total_elapsed:.2f}s\n"
                f"  - runtime: {perf_analysis.get('runtime', 'N/A')}s (trimmed_mean)\n"
                f"  - memory: {perf_analysis.get('memory', 'N/A')}MB (trimmed_mean)\n"
                f"  - integral: {perf_analysis.get('integral', 'N/A')}MB*s\n"
                f"  - pass_rate: {perf_analysis.get('pass_rate', 0):.2%}"
            )
            return result
        except Exception as e:
            multi_run_elapsed = time.time() - multi_run_start
            total_elapsed = time.time() - eval_start_time
            self.logger.error(
                f"[多次评估失败] 多次运行耗时: {multi_run_elapsed:.2f}s, 总耗时: {total_elapsed:.2f}s, "
                f"错误类型: {type(e).__name__}, 错误信息: {e}\n{traceback.format_exc()}"
            )
            return self._create_default_performance_result(consistent=False)

    def _extract_pass_rate(self, results: dict[str, Any]) -> float:
        """从评估结果中提取通过率"""
        # 1. 尝试直接获取 pass_rate 字段
        pass_rate = results.get("performance_analysis", {}).get("pass_rate")
        try:
            if pass_rate is not None and isinstance(pass_rate, (int, float)):
                return float(pass_rate)
        except Exception:
            pass

        # 2. 尝试从 pass_rates 列表获取（取最小值，保守策略）
        pr_list = results.get("pass_rates")
        try:
            if isinstance(pr_list, list) and pr_list:
                return float(min(float(p) for p in pr_list))
        except Exception:
            pass

        # 3. 尝试从 first_run_details 计算
        try:
            fr = results.get("first_run_details") or []
            total = len(fr)
            passed = sum(1 for tc in fr if tc.get("passed", False))
            return (passed / total) if total > 0 else 0.0
        except Exception:
            return 0.0

    def _clean_performance_value(self, val: Any) -> float:
        """清理性能指标值，转换为 float，处理 inf/nan"""
        if isinstance(val, (int, float)):
            return float(val)

        # 尝试处理 numpy 类型或 callable
        try:
            item_fn = getattr(val, "item", None)
            if callable(item_fn):
                val = item_fn()
        except Exception:
            pass

        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("inf", "+inf", "infinity", "+infinity"):
                return float("inf")
            elif s in ("-inf", "-infinity"):
                return float("-inf")
            elif s == "nan":
                return float("nan")
            else:
                try:
                    return float(val)
                except Exception:
                    return float("inf")
        return float(val) if isinstance(val, (int, float)) else float("inf")

    def run(self, instance: EffiBenchXInstance) -> dict[str, Any]:
        """运行性能优化流程（仅使用配置语言，实例为 dataclass）"""
        run_start_time = time.time()
        instance_id = getattr(instance, "task_name", None) or getattr(instance, "id", "unknown")

        self.logger.info(
            f"\n{'#' * 70}\n"
            f"# [PerfAgent 运行开始]\n"
            f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# 实例: {instance_id}\n"
            f"# 最大迭代次数: {self.config.max_iterations}\n"
            f"# 优化目标: {self.config.optimization.target}\n"
            f"# 模型: {self.config.model.name}\n"
            f"{'#' * 70}"
        )

        try:
            # 1. 初始化上下文
            init_start = time.time()
            ctx = self._init_run_context(instance)
            init_elapsed = time.time() - init_start
            self.logger.info(f"[上下文初始化完成] 耗时: {init_elapsed:.3f}s")

            # 2. 初始评估
            self._perform_initial_evaluation(ctx)

            # 3. 优化循环
            loop_start = time.time()
            self._process_optimization_loop(ctx)
            loop_elapsed = time.time() - loop_start
            self.logger.info(f"[优化循环完成] 总耗时: {loop_elapsed:.2f}s ({loop_elapsed / 60:.1f}分钟)")

            # 4. 完成并生成结果
            result = self._finalize_run(ctx)

            run_elapsed = time.time() - run_start_time
            self.logger.info(
                f"\n{'#' * 70}\n"
                f"# [PerfAgent 运行成功完成]\n"
                f"# 实例: {instance_id}\n"
                f"# 总耗时: {run_elapsed:.2f}s ({run_elapsed / 60:.1f}分钟)\n"
                f"# 成功: {result.get('success', False)}\n"
                f"{'#' * 70}"
            )
            return result

        except Exception as e:
            run_elapsed = time.time() - run_start_time
            self.logger.error(
                f"\n{'!' * 70}\n"
                f"! [PerfAgent 运行失败]\n"
                f"! 实例: {instance_id}\n"
                f"! 运行耗时: {run_elapsed:.2f}s ({run_elapsed / 60:.1f}分钟)\n"
                f"! 错误类型: {type(e).__name__}\n"
                f"! 错误信息: {e}\n"
                f"! 堆栈跟踪:\n{traceback.format_exc()}\n"
                f"{'!' * 70}"
            )
            # 尝试记录错误轨迹
            try:
                # 如果 ctx 存在，尝试用它来结束轨迹
                if "ctx" in locals():
                    ctx.trajectory.finalize(success=False, error_message=str(e), final_submission_code=ctx.best_code)
                    self.logger.info("[轨迹已保存] 错误轨迹记录完成")
            except Exception as traj_error:
                self.logger.warning(f"[轨迹保存失败] {type(traj_error).__name__}: {traj_error}")
            raise

    def _init_run_context(self, instance: EffiBenchXInstance) -> RunContext:
        """初始化运行上下文"""
        inst = instance
        instance_id = getattr(inst, "task_name", None) or getattr(inst, "id", "unknown")

        # 初始化轨迹记录器
        trajectory = TrajectoryLogger(
            instance_id,
            self.config.logging.trajectory_dir,
            log_dir=self.config.logging.log_dir,
        )

        language = self._normalize_language(self.config.language_cfg.language)
        trajectory.metadata.language = language
        trajectory.metadata.optimization_target = self.config.optimization.target

        # 记录 System Prompt
        system_prompt = self._build_system_prompt(
            language=language,
            optimization_target=self.config.optimization.target,
            task_description=inst.description_md,
            task_type=getattr(inst, "type", None),
            starter_code=self._resolve_starter_code(inst, language),
        )
        trajectory.add_history(role="system", content=system_prompt, message_type="system_prompt")

        # 提取初始代码
        initial_code = self._extract_initial_code(
            inst, language=language, optimization_target=self.config.optimization.target
        )
        if not initial_code:
            raise ValueError("无法提取初始代码")

        test_cases = self._prepare_test_cases(inst)
        iter_offset = 1 if self._initial_code_source in ("text", "dir") else 0

        # 初始化历史
        self.optimization_history = []

        return RunContext(
            instance=inst,
            trajectory=trajectory,
            language=language,
            optimization_target=self.config.optimization.target,
            initial_code=initial_code,
            current_code=initial_code,
            best_code=initial_code,
            best_performance=float("inf"),
            best_pass_rate=0.0,
            current_benchmark_results={},
            best_benchmark_results={},
            optimization_history=self.optimization_history,
            iter_offset=iter_offset,
            test_cases=test_cases,
        )

    def _perform_initial_evaluation(self, ctx: RunContext):
        """执行初始性能评估"""
        init_eval_start = time.time()
        self.logger.info(
            f"\n{'=' * 60}\n"
            f"[初始评估开始] 时间: {datetime.now().strftime('%H:%M:%S')}\n"
            f"  - 实例ID: {ctx.instance.id}\n"
            f"  - 语言: {ctx.language}\n"
            f"  - 优化目标: {ctx.optimization_target}\n"
            f"  - 初始代码来源: {self._initial_code_source}\n"
            f"  - 初始代码长度: {len(ctx.current_code)} 字符\n"
            f"  - 测试用例数: {len(ctx.test_cases)}\n"
            f"{'=' * 60}"
        )

        step_id = ctx.trajectory.start_step(
            "initial_evaluation", query="Evaluate the initial code performance.", code_snapshot=ctx.current_code
        )

        initial_performance = self._evaluate_performance(ctx.language, ctx.current_code, ctx.test_cases, ctx.instance)

        ctx.current_benchmark_results = initial_performance
        ctx.best_benchmark_results = initial_performance

        initial_evaluation_summary = {
            "performance_analysis": initial_performance.get("performance_analysis", {}),
            "failed_test_details": initial_performance.get("failed_test_details", [])[:3],
        }

        summary_text = self._build_summary_text(
            iteration=1 if ctx.iter_offset else 0,
            code_changed=False,
            diff_text=None,
            benchmark_results=initial_performance,
            current_program=ctx.current_code,
        )

        ctx.trajectory.end_step(
            step_id,
            response=summary_text,
            thought="收集初始性能基线以指导后续优化",
            code_changed=False,
            performance_metrics=initial_evaluation_summary,
            code_snapshot=ctx.current_code,
        )

        ctx.best_pass_rate = self._extract_pass_rate(initial_performance)
        init_metric = initial_performance.get("performance_analysis", {}).get(ctx.optimization_target, float("inf"))

        ctx.initial_performance_value = self._clean_performance_value(init_metric)

        if ctx.initial_performance_value <= ctx.best_performance:
            ctx.best_performance = ctx.initial_performance_value
            ctx.best_code = ctx.current_code

        init_eval_elapsed = time.time() - init_eval_start
        perf_analysis = initial_performance.get("performance_analysis", {})
        self.logger.info(
            f"\n[初始评估完成] 时间: {datetime.now().strftime('%H:%M:%S')}, 总耗时: {init_eval_elapsed:.2f}s\n"
            f"  📊 初始性能基线:\n"
            f"      - pass_rate: {ctx.best_pass_rate:.2%}\n"
            f"      - runtime: {perf_analysis.get('runtime', 'N/A')}s\n"
            f"      - memory: {perf_analysis.get('memory', 'N/A')}MB\n"
            f"      - integral: {perf_analysis.get('integral', 'N/A')}MB*s\n"
            f"      - {ctx.optimization_target} (优化目标): {ctx.initial_performance_value}"
        )

    def _process_optimization_loop(self, ctx: RunContext):
        """执行优化循环"""
        remaining_iterations = max(0, self.config.max_iterations - ctx.iter_offset)

        self.logger.info(
            f"\n[优化循环开始] 计划迭代次数: {remaining_iterations}, "
            f"iter_offset: {ctx.iter_offset}, max_iterations: {self.config.max_iterations}"
        )

        for iteration in range(remaining_iterations):
            current_iter_num = iteration + 1 + ctx.iter_offset

            should_stop = self._process_single_iteration(ctx, current_iter_num)
            if should_stop:
                self.logger.info(f"[优化循环提前终止] 在第 {current_iter_num} 次迭代后停止")
                break

        self.logger.info(f"[优化循环结束] 共执行 {len(ctx.optimization_history)} 次迭代")

    def _process_single_iteration(self, ctx: RunContext, iteration_num: int) -> bool:
        """处理单次迭代。返回 True 表示应该停止循环。"""
        iteration_start_time = time.time()
        self.logger.info(
            f"\n{'=' * 60}\n[迭代 {iteration_num} 开始] 时间: {datetime.now().strftime('%H:%M:%S')}\n{'=' * 60}"
        )

        # 1. 生成优化建议
        prompt_start = time.time()
        opt_prompt = self._build_optimization_prompt(
            current_program=ctx.current_code,
            language=ctx.language,
            benchmark_results=ctx.current_benchmark_results,
        )
        prompt_elapsed = time.time() - prompt_start
        self.logger.debug(f"[Prompt构建] 耗时: {prompt_elapsed:.3f}s, Prompt长度: {len(opt_prompt)} 字符")

        step_id = ctx.trajectory.start_step(
            "generate_optimization",
            query=opt_prompt,
            code_snapshot=ctx.current_code,
        )

        # 2. 调用 LLM（这里会有详细的 LLM 日志）
        llm_phase_start = time.time()
        optimization_response = self._call_llm_for_optimization(ctx, opt_prompt)
        llm_phase_elapsed = time.time() - llm_phase_start

        # 3. 提取和应用代码变更
        extract_start = time.time()
        diff_text = None
        optimized_code = None

        if self.config.optimization.code_generation_mode == "direct":
            optimized_code = self._extract_full_code_from_response(optimization_response)
            if not optimized_code:
                extract_elapsed = time.time() - extract_start
                self.logger.warning(
                    f"[代码提取失败] 耗时: {extract_elapsed:.3f}s, 模式: direct, "
                    f"响应长度: {len(optimization_response)} 字符"
                )
                self._handle_failed_code_extraction(
                    ctx, step_id, optimization_response, iteration_num, "无法从响应中提取有效的完整代码"
                )
                return False
        else:
            diff_text = self._extract_diff_from_response(optimization_response)
            if not diff_text:
                extract_elapsed = time.time() - extract_start
                self.logger.warning(
                    f"[Diff提取失败] 耗时: {extract_elapsed:.3f}s, 响应中未找到有效的 SEARCH/REPLACE 块"
                )
                self._handle_failed_code_extraction(
                    ctx, step_id, optimization_response, iteration_num, "无法从响应中提取有效的 diff"
                )
                return False

            # 应用 diff
            diff_apply_start = time.time()
            try:
                optimized_code = self.diff_applier.apply_diff(ctx.current_code, diff_text)
                diff_apply_elapsed = time.time() - diff_apply_start
                self.logger.info(f"[Diff应用成功] 耗时: {diff_apply_elapsed:.3f}s, diff长度: {len(diff_text)} 字符")
            except Exception as e:
                diff_apply_elapsed = time.time() - diff_apply_start
                self.logger.error(
                    f"[Diff应用失败] 耗时: {diff_apply_elapsed:.3f}s, 错误类型: {type(e).__name__}, 错误信息: {e}"
                )
                self._handle_failed_diff_application(
                    ctx, step_id, optimization_response, diff_text, iteration_num, str(e)
                )
                return False

        extract_elapsed = time.time() - extract_start

        # 4. 检查代码是否变化
        code_changed = optimized_code != ctx.current_code
        code_diff_lines = abs(len(optimized_code.splitlines()) - len(ctx.current_code.splitlines()))
        self.logger.info(
            f"[代码变更检查] 代码已变更: {code_changed}, "
            f"新代码长度: {len(optimized_code)} 字符, 行数变化: {code_diff_lines:+d}"
        )

        if not code_changed:
            self._handle_no_code_change(ctx, step_id, optimization_response, diff_text, iteration_num)
            ctx.no_improve_count += 1
            iteration_elapsed = time.time() - iteration_start_time
            self.logger.info(
                f"[迭代 {iteration_num} 结束] 代码未变更, 跳过评估\n"
                f"  - LLM调用耗时: {llm_phase_elapsed:.2f}s ({llm_phase_elapsed / 60:.1f}分钟)\n"
                f"  - 代码提取耗时: {extract_elapsed:.3f}s\n"
                f"  - 迭代总耗时: {iteration_elapsed:.2f}s ({iteration_elapsed / 60:.1f}分钟)\n"
                f"  - 连续未改进次数: {ctx.no_improve_count}"
            )
            if self.config.early_stop_no_improve and ctx.no_improve_count >= self.config.early_stop_no_improve:
                self.logger.info(f"[提前停止] 连续未改进达到阈值 {self.config.early_stop_no_improve}")
                return True
            return False

        # 5. 评估新代码（这里会有详细的评测日志）
        eval_phase_start = time.time()
        try:
            performance_result = self._evaluate_performance(ctx.language, optimized_code, ctx.test_cases, ctx.instance)
            eval_phase_elapsed = time.time() - eval_phase_start

            # 更新上下文状态
            improved = self._update_run_context_after_eval(
                ctx, optimized_code, performance_result, diff_text, iteration_num
            )

            # 记录步骤
            self._record_iteration_step(
                ctx,
                step_id,
                optimization_response,
                diff_text,
                optimized_code,
                performance_result,
                iteration_num,
                improved,
            )

            if improved:
                ctx.no_improve_count = 0
            else:
                ctx.no_improve_count += 1

            # 输出迭代总结
            iteration_elapsed = time.time() - iteration_start_time
            perf_analysis = performance_result.get("performance_analysis", {})
            self.logger.info(
                f"\n[迭代 {iteration_num} 完成] 时间: {datetime.now().strftime('%H:%M:%S')}\n"
                f"  ⏱️  时间分解:\n"
                f"      - LLM调用耗时: {llm_phase_elapsed:.2f}s ({llm_phase_elapsed / 60:.1f}分钟) "
                f"({llm_phase_elapsed / iteration_elapsed * 100:.1f}%)\n"
                f"      - 代码提取/应用耗时: {extract_elapsed:.3f}s\n"
                f"      - 性能评估耗时: {eval_phase_elapsed:.2f}s ({eval_phase_elapsed / 60:.1f}分钟) "
                f"({eval_phase_elapsed / iteration_elapsed * 100:.1f}%)\n"
                f"      - 迭代总耗时: {iteration_elapsed:.2f}s ({iteration_elapsed / 60:.1f}分钟)\n"
                f"  📊 性能指标:\n"
                f"      - pass_rate: {perf_analysis.get('pass_rate', 0):.2%}\n"
                f"      - runtime: {perf_analysis.get('runtime', 'N/A')}s\n"
                f"      - memory: {perf_analysis.get('memory', 'N/A')}MB\n"
                f"      - integral: {perf_analysis.get('integral', 'N/A')}MB*s\n"
                f"  ✅ 结果: {'性能改进，已采纳' if improved else '未改进'}\n"
                f"  📈 连续未改进次数: {ctx.no_improve_count}"
            )

            if self.config.early_stop_no_improve and ctx.no_improve_count >= self.config.early_stop_no_improve:
                self.logger.info(f"[提前停止] 连续未改进达到阈值 {self.config.early_stop_no_improve}")
                return True

        except Exception as e:
            eval_phase_elapsed = time.time() - eval_phase_start
            iteration_elapsed = time.time() - iteration_start_time
            self.logger.error(
                f"[迭代 {iteration_num} 异常] 评估阶段出错\n"
                f"  - 错误类型: {type(e).__name__}\n"
                f"  - 错误信息: {e}\n"
                f"  - LLM调用耗时: {llm_phase_elapsed:.2f}s\n"
                f"  - 评估耗时(至异常): {eval_phase_elapsed:.2f}s\n"
                f"  - 迭代总耗时: {iteration_elapsed:.2f}s\n"
                f"  - 堆栈跟踪:\n{traceback.format_exc()}"
            )
            self._handle_evaluation_error(ctx, step_id, optimization_response, diff_text, iteration_num, str(e))

        return False

    def _call_llm_for_optimization(self, ctx: RunContext, opt_prompt: str) -> str:
        """调用 LLM 获取优化建议"""
        system_prompt = self._build_system_prompt(
            language=ctx.language,
            optimization_target=self.config.optimization.target,
            task_description=ctx.instance.description_md,
            task_type=getattr(ctx.instance, "type", None),
            starter_code=self._resolve_starter_code(ctx.instance, ctx.language),
        )
        messages = self._build_messages(system_prompt, ctx.trajectory.history, opt_prompt)

        if self.llm_client:
            llm_start_time = time.time()
            self.logger.info(
                f"[LLM调用开始] 时间: {datetime.now().strftime('%H:%M:%S')}, 模型: {self.config.model.name}"
            )
            try:
                response = self.llm_client.call_llm(
                    messages,
                    temperature=self.config.model.temperature,
                    max_tokens=self.config.model.max_output_tokens,
                    usage_context="perfagent.optimize",
                )
                llm_elapsed = time.time() - llm_start_time
                self.logger.info(
                    f"[LLM调用完成] 耗时: {llm_elapsed:.2f}s ({llm_elapsed / 60:.1f}分钟), "
                    f"响应长度: {len(response)} 字符"
                )
                return response
            except TypeError:
                # 兼容旧版 API（不支持 usage_context 参数）
                response = self.llm_client.call_llm(
                    messages,
                    temperature=self.config.model.temperature,
                    max_tokens=self.config.model.max_output_tokens,
                )
                llm_elapsed = time.time() - llm_start_time
                self.logger.info(
                    f"[LLM调用完成] 耗时: {llm_elapsed:.2f}s ({llm_elapsed / 60:.1f}分钟), "
                    f"响应长度: {len(response)} 字符"
                )
                return response
            except Exception as e:
                llm_elapsed = time.time() - llm_start_time
                self.logger.error(
                    f"[LLM调用失败] 耗时: {llm_elapsed:.2f}s, 错误类型: {type(e).__name__}, "
                    f"错误信息: {e}\n{traceback.format_exc()}"
                )
                raise
        self.logger.warning("[LLM未配置] LLM 客户端未初始化，跳过本次优化")
        return "LLM 未配置或不可用，跳过本次优化建议。请检查 API 配置。"

    def _handle_failed_code_extraction(
        self, ctx: RunContext, step_id: str, response: str, iteration: int, error_msg: str
    ):
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=False,
            diff_text=None,
            benchmark_results=None,
            current_program=ctx.current_code,
            error_message=error_msg,
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="未能提取有效的代码/diff",
            code_changed=False,
            diff=None,
            error=error_msg,
            code_snapshot=ctx.current_code,
            summary=summary,
        )

    def _handle_failed_diff_application(
        self, ctx: RunContext, step_id: str, response: str, diff_text: str, iteration: int, error_msg: str
    ):
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=False,
            diff_text=diff_text,
            benchmark_results=None,
            current_program=ctx.current_code,
            error_message=f"应用 diff 失败: {error_msg}",
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="应用 diff 阶段发生异常",
            code_changed=None,
            diff=diff_text,
            performance_metrics=None,
            error=f"应用 diff 失败: {error_msg}",
            code_snapshot=ctx.current_code,
            summary=summary,
        )

    def _handle_no_code_change(self, ctx: RunContext, step_id: str, response: str, diff_text: str, iteration: int):
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=False,
            diff_text=diff_text,
            benchmark_results=ctx.current_benchmark_results,
            current_program=ctx.current_code,
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="diff 应用后代码未变化，跳过",
            code_changed=False,
            diff=diff_text,
            code_snapshot=ctx.current_code,
            summary=summary,
        )
        self.logger.warning("代码未发生变化，跳过此次迭代")

    def _update_run_context_after_eval(
        self, ctx: RunContext, optimized_code: str, performance_result: dict, diff_text: str | None, iteration: int
    ) -> bool:
        """更新上下文并判断是否改进"""
        current_performance = performance_result.get("performance_analysis", {}).get(
            ctx.optimization_target, float("inf")
        )
        current_pass_rate = self._extract_pass_rate(performance_result)

        improved = False
        if current_pass_rate == 1.0 and current_performance < ctx.best_performance:
            improved = True

        # 如果最大迭代次数为 1，强制视为改进（即总是保存生成代码）
        # 除非代码完全崩溃（这里可以根据需求调整，当前逻辑是只要运行了就保存）
        if self.config.max_iterations == 1 and not improved:
            improved = True
            self.logger.info("单次迭代模式：强制采纳生成代码作为最佳结果")

        # 记录历史
        ctx.optimization_history.append(
            {
                "iteration": iteration,
                "diff": diff_text,
                "performance_before": ctx.best_performance,
                "performance_after": current_performance,
                "improvement": ctx.best_performance - current_performance,
                "success": improved,
            }
        )

        if improved:
            ctx.best_pass_rate = current_pass_rate
            ctx.best_performance = current_performance
            ctx.best_code = optimized_code
            ctx.best_benchmark_results = performance_result
            self.logger.info(
                f"采用更优代码: pass_rate {ctx.best_pass_rate:.2f}, {ctx.optimization_target} {ctx.best_performance:.4f}"
            )
        else:
            self.logger.info(f"未改进: pass_rate {current_pass_rate:.2f} vs {ctx.best_pass_rate:.2f}")

        # 决定是否采用代码
        if self.config.optimization.adopt_only_if_improved:
            if improved:
                ctx.current_code = optimized_code
            else:
                ctx.current_code = ctx.best_code
        else:
            ctx.current_code = optimized_code

        ctx.current_benchmark_results = performance_result
        return improved

    def _record_iteration_step(
        self,
        ctx: RunContext,
        step_id: str,
        response: str,
        diff_text: str | None,
        optimized_code: str,
        performance_result: dict,
        iteration: int,
        improved: bool,
    ):
        adopted = improved if self.config.optimization.adopt_only_if_improved else True

        evaluation_summary = {
            "performance_analysis": performance_result.get("performance_analysis", {}),
            "failed_test_details": performance_result.get("failed_test_details", [])[:3],
            "pass_rates": performance_result.get("pass_rates", []),
            "pass_rate_consistent": performance_result.get("pass_rate_consistent", False),
        }

        summary_text = self._build_summary_text(
            iteration=iteration,
            code_changed=adopted,
            diff_text=diff_text,
            benchmark_results=performance_result,
            current_program=ctx.current_code,
        )

        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought=("应用 diff 并完成性能评估" if adopted else "评估未改进，未采用优化"),
            code_changed=adopted,
            diff=diff_text,
            performance_metrics=evaluation_summary,
            code_snapshot=ctx.current_code,
            summary=summary_text,
        )

    def _handle_evaluation_error(
        self, ctx: RunContext, step_id: str, response: str, diff_text: str | None, iteration: int, error_msg: str
    ):
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=True,
            diff_text=diff_text,
            benchmark_results=None,
            current_program=ctx.current_code,
            error_message=f"性能评估失败: {error_msg}",
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="性能评估阶段发生异常",
            code_changed=True,
            diff=diff_text,
            performance_metrics=None,
            error=f"性能评估失败: {error_msg}",
            code_snapshot=ctx.current_code,
            summary=summary,
        )

    def _finalize_run(self, ctx: RunContext) -> dict[str, Any]:
        """完成运行并生成最终结果"""
        finalize_start = time.time()
        self.logger.info(f"\n[结果汇总开始] 时间: {datetime.now().strftime('%H:%M:%S')}")

        initial_trimmed = ctx.initial_performance_value
        best_perf = self._clean_performance_value(ctx.best_performance)

        executed_iterations = len(ctx.optimization_history)
        # 初始代码 + 迭代次数
        total_iterations = (1 if self._initial_code_source in ("text", "dir") else 0) + executed_iterations

        optimized_code_final = ctx.best_code

        final_result = {
            "instance_id": ctx.instance.id,
            "initial_code": ctx.initial_code,
            "optimized_code": optimized_code_final,
            "initial_performance": initial_trimmed,
            "final_performance": best_perf,
            "total_iterations": total_iterations,
            "optimization_history": ctx.optimization_history,
            "success": bool(best_perf < initial_trimmed),
        }

        unit = (
            "s" if ctx.optimization_target == "runtime" else ("MB" if ctx.optimization_target == "memory" else "MB*s")
        )
        final_result["language"] = ctx.language
        final_result["optimization_target"] = ctx.optimization_target
        final_result["performance_unit"] = unit

        try:
            result_for_output = ctx.best_benchmark_results
            metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(result_for_output)
            metrics_md = self._format_metrics_md(metrics_dict)
            artifacts_md = self._format_artifacts_md(artifacts_dict)

            final_artifacts = "Current Metrics:\n" + metrics_md + "\n\nCurrent Artifacts:\n" + artifacts_md
            final_result["final_artifacts"] = final_artifacts
        except Exception as e:
            self.logger.warning(f"[构建最终指标失败] {type(e).__name__}: {e}")
            final_result["final_artifacts"] = None

        # 汇总最终三项指标
        try:
            perf_metrics = result_for_output.get("performance_analysis", {})
            final_result["final_metrics"] = {
                "runtime": perf_metrics.get("runtime", "Infinity"),
                "memory": perf_metrics.get("memory", "Infinity"),
                "integral": perf_metrics.get("integral", "Infinity"),
            }
        except Exception as e:
            self.logger.warning(f"[获取性能指标失败] {type(e).__name__}: {e}")
            final_result["final_metrics"] = {
                "runtime": "Infinity",
                "memory": "Infinity",
                "integral": "Infinity",
            }

        # 记录最终轨迹
        selected_value = final_result.get("final_metrics", {}).get(ctx.optimization_target)
        selected_value = self._clean_performance_value(selected_value)

        trajectory_file = ctx.trajectory.finalize(
            success=final_result["success"],
            final_performance={
                "target": ctx.optimization_target,
                "unit": unit,
                "value": selected_value if selected_value != float("inf") else best_perf,
                "runtime": final_result.get("final_metrics", {}).get("runtime"),
                "memory": final_result.get("final_metrics", {}).get("memory"),
                "integral": final_result.get("final_metrics", {}).get("integral"),
            },
            final_submission_code=optimized_code_final,
        )

        final_result["trajectory_file"] = trajectory_file

        # 计算改进幅度
        improvement_pct = 0.0
        if initial_trimmed != float("inf") and initial_trimmed > 0:
            improvement_pct = (initial_trimmed - best_perf) / initial_trimmed * 100

        # 统计优化历史
        successful_iterations = sum(1 for h in ctx.optimization_history if h.get("success", False))

        finalize_elapsed = time.time() - finalize_start
        self.logger.info(
            f"\n[优化结果总结]\n"
            f"  📋 基本信息:\n"
            f"      - 实例ID: {ctx.instance.id}\n"
            f"      - 语言: {ctx.language}\n"
            f"      - 优化目标: {ctx.optimization_target}\n"
            f"      - 执行迭代数: {executed_iterations}\n"
            f"      - 成功改进迭代数: {successful_iterations}\n"
            f"\n"
            f"  📈 性能变化:\n"
            f"      - 初始 {ctx.optimization_target}: {initial_trimmed} {unit}\n"
            f"      - 最终 {ctx.optimization_target}: {best_perf} {unit}\n"
            f"      - 改进幅度: {improvement_pct:.2f}%\n"
            f"      - 优化成功: {'✅ 是' if final_result['success'] else '❌ 否'}\n"
            f"\n"
            f"  📊 最终性能指标:\n"
            f"      - runtime: {final_result['final_metrics']['runtime']}s\n"
            f"      - memory: {final_result['final_metrics']['memory']}MB\n"
            f"      - integral: {final_result['final_metrics']['integral']}MB*s\n"
            f"      - pass_rate: {ctx.best_pass_rate:.2%}\n"
            f"\n"
            f"  📁 轨迹文件: {trajectory_file}\n"
            f"  ⏱️  结果汇总耗时: {finalize_elapsed:.3f}s"
        )

        return final_result

    def _build_optimization_prompt(
        self,
        current_program: str,
        language: str,
        benchmark_results: dict[str, Any],
    ) -> str:
        """构建优化提示词，填充当前程序、评估指标与构件(section)。"""
        if self.config.optimization.code_generation_mode == "direct":
            return self.config.prompts.optimization_template

        # diff-based prompt construction
        # 构造 metrics 与 artifacts
        metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(benchmark_results)
        # 以 Markdown 格式化，便于模型阅读
        current_metrics_str = self._format_metrics_md(metrics_dict)
        current_artifacts_str = self._format_artifacts_md(artifacts_dict)
        current_program_md = f"```\n{current_program}\n```"

        try:
            return self.config.prompts.optimization_template.format(
                current_program=current_program_md,
                current_metrics=current_metrics_str,
                current_artifacts_section=current_artifacts_str,
                language=language,
            )
        except Exception:
            # 若模板占位符不匹配，回退为一个通用提示
            return (
                "# Task\n"
                "请分析以下程序信息，并根据系统提示生成 `## Thinking` 与 `## Diffs`：\n\n"
                "## Current Program\n" + current_program_md + "\n\n"
                "## Current Metrics\n" + current_metrics_str + "\n\n"
                "## Current Artifacts\n" + current_artifacts_str
            )

    def _build_system_prompt(
        self,
        language: str,
        optimization_target: str,
        task_description: str,
        task_type: str | None = None,
        starter_code: str | None = None,
    ) -> str:
        tmpl = self.config.prompts.system_template
        additional = self.config.prompts.additional_requirements or ""
        local_memory = getattr(self.config.prompts, "local_memory", None) or ""
        global_memory = getattr(self.config.prompts, "global_memory", None) or ""
        allowed_imports_scope = EFFIBENCH_REGISTRY.get(language, {}).get("imports", "")
        is_functional = (task_type or "").lower() == "functional"
        if tmpl:
            try:
                base = tmpl.format(
                    language=language,
                    optimization_target=optimization_target,
                    task_description=task_description,
                    additional_requirements=additional,
                    local_memory=local_memory,
                    global_memory=global_memory,
                    allowed_imports_scope=allowed_imports_scope,
                )
            except Exception:
                base = tmpl
        else:
            base = (
                f"你是一个专业的代码性能优化专家。目标是提升 {optimization_target}。\n"
                f"当前语言：{language}。任务描述：{task_description}\n\n"
                f"附加要求：{additional}\n\n"
                f"本地记忆：{local_memory}\n\n"
                f"全局记忆：{global_memory}\n\n"
                f"允许使用的标准导入范围如下：\n"
                f"{allowed_imports_scope}"
            )
        if is_functional and starter_code:
            starter_section = (
                "\n\n## Starter Code\n"
                "Use the following starter code as the exact framework for your solution.\n\n"
                f"```{language}\n"
                f"{starter_code}\n"
                "```\n\n"
                "- Implement the function with the exact signature (name, parameters, etc.) "
                "specified in the starter code.\n"
            )
            return base + starter_section
        return base

    def _build_metrics_and_artifacts(
        self, benchmark_results: dict[str, Any], include_other_metrics: bool | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """根据基准评估结果构造 current_metrics 与 current_artifacts_section。"""
        performance_metrics = benchmark_results.get("performance_analysis", {})
        failed_test_details = benchmark_results.get("failed_test_details", []) or []

        # 失败情况：汇总失败信息并返回错误指标
        target = self.config.optimization.target

        # Determine which metrics to include
        if include_other_metrics is None:
            include_other_metrics = self.config.optimization.include_other_metrics_in_summary

        keys_to_include = {"runtime", "memory", "integral"}
        if not include_other_metrics:
            keys_to_include = {target}

        # 失败情况：汇总失败信息并返回错误指标
        passed = performance_metrics.get("passed", False)
        if not passed:
            num_failed = len(failed_test_details)
            num_total = len(benchmark_results.get("first_run_details", []))
            pass_rate = (num_total - num_failed) / num_total if num_total > 0 else 0

            representative_failures: dict[str, Any] = {}
            for failure in failed_test_details:
                status = failure.get("status", "unknown")
                if status not in representative_failures:
                    representative_failures[status] = failure

            failure_details_summary: list[str] = []
            for status, failure in representative_failures.items():
                text = failure.get("text", "No additional error text.")
                if isinstance(text, str) and len(text) > 300:
                    text = text[-300:] + "..."
                failure_details_summary.append(f"- Status: {status}, Details (last 300 chars of Output): {text}")

            failures_text = "\n".join(failure_details_summary)
            all_statuses = ", ".join(representative_failures.keys())

            error_artifacts = {
                "error_type": f"SolutionFailedTests (statuses: {all_statuses})",
                "error_message": (f"Solution passed {pass_rate:.2%} of test cases. Failure details:\n{failures_text}"),
                "suggestion": (
                    "Review the solution to ensure it correctly handles all test cases, including edge cases."
                ),
            }

            metrics = {
                "pass_rate": pass_rate,
                "target": target,
                "error": (
                    f"Solution failed {len(failed_test_details)} test case(s) with statuses: {all_statuses}. See artifacts for details."
                ),
            }
            for k in keys_to_include:
                metrics[k] = "Infinity"

            return metrics, error_artifacts

        # 成功情况：计算时间分数与综合分数
        pass_rate = 1.0

        metrics = {
            "pass_rate": pass_rate,
            "target": target,
        }
        for k in keys_to_include:
            metrics[k] = performance_metrics.get(k, "Infinity")

        artifacts = {"details": "All test cases passed."}
        return metrics, artifacts

    def _format_metrics_md(self, metrics: dict[str, Any]) -> str:
        """将性能指标格式化为 Markdown 文本。"""
        lines: list[str] = []

        pr = metrics.get("pass_rate")
        if pr is not None:
            try:
                pr_pct = f"{float(pr) * 100:.2f}%"
            except Exception:
                pr_pct = str(pr)
            lines.append(f"- Pass rate: {pr_pct}")

        def _fmt(val: Any, unit: str) -> str:
            if isinstance(val, (int, float)):
                if val == float("inf"):
                    return "Infinity"
                if val == float("-inf"):
                    return "-Infinity"
                return f"{float(val):.6f} {unit}"
            s = str(val).strip().lower()
            if s in ("inf", "+inf", "infinity", "+infinity"):
                return "Infinity"
            if s in ("-inf", "-infinity"):
                return "-Infinity"
            if s == "nan":
                return "NaN"
            try:
                return f"{float(val):.6f} {unit}"
            except Exception:
                return f"{val} {unit}"

        if "runtime" in metrics:
            lines.append(f"- Runtime: {_fmt(metrics.get('runtime'), 's')}")
        if "memory" in metrics:
            lines.append(f"- Memory: {_fmt(metrics.get('memory'), 'MB')}")
        if "integral" in metrics:
            lines.append(f"- Integral: {_fmt(metrics.get('integral'), 'MB*s')}")

        tgt = metrics.get("target")
        if tgt is not None:
            lines.append(f"- Target: {tgt}")

        err = metrics.get("error")
        if err:
            lines.append(f"- Error: {err}")

        return "\n".join(lines) if lines else "- No metrics available."

    def _format_artifacts_md(self, artifacts: dict[str, Any]) -> str:
        """将构件信息格式化为 Markdown 文本。"""
        if not artifacts:
            return "- No artifacts available."
        lines: list[str] = []
        for k, v in artifacts.items():
            if isinstance(v, str) and "\n" in v:
                indented = "\n  ".join(v.split("\n"))
                lines.append(f"- {k}: {indented}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _build_summary_text(
        self,
        iteration: int,
        code_changed: bool,
        diff_text: str | None,
        benchmark_results: dict[str, Any] | None,
        current_program: str | None = None,
        error_message: str | None = None,
    ) -> str:
        """构建一步迭代的 Markdown 摘要文本，包含程序更新、当前程序、指标与构件。

        - metrics/artifacts 由 `_build_metrics_and_artifacts` 生成并通过 `_format_*_md` 格式化。
        - 无评估或失败时，输出错误信息和占位构件。
        """
        # 构造指标与构件
        if benchmark_results:
            metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(benchmark_results)
        else:
            metrics_dict = {}
            artifacts_dict = {}
            if error_message:
                metrics_dict["error"] = error_message
                if not artifacts_dict:
                    artifacts_dict["details"] = "No evaluation due to error."

        metrics_md = self._format_metrics_md(metrics_dict)
        artifacts_md = self._format_artifacts_md(artifacts_dict)
        diff_size = len(diff_text) if diff_text else 0

        prog_text = current_program or ""

        return (
            "## Program Update\n"
            f"- Iteration: {iteration}\n"
            "## Current Program\n" + prog_text + "\n\n"
            "## Current Metrics\n" + metrics_md + "\n\n"
            "## Current Artifacts\n" + artifacts_md
        )

    def _extract_full_code_from_response(self, response: str) -> str:
        """从模型响应中提取完整代码（Markdown 代码块）。"""
        if not response:
            return ""
        # 匹配 ```language ... ```
        # 尝试匹配 python, cpp, java, etc. 或者不指定
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # 返回最后一个匹配的代码块，通常是最终代码
            return matches[-1].strip()
        return ""

    def _extract_diff_from_response(self, response: str) -> str:
        """从模型响应中提取 diff
        仅支持 SEARCH/REPLACE 区块格式。
        """
        if not response:
            return ""
        if "<<<<<<< SEARCH" in response and ">>>>>>> REPLACE" in response:
            try:
                start_idx = response.find("<<<<<<< SEARCH")
                end_idx = response.rfind(">>>>>>> REPLACE")
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    return response[start_idx : end_idx + len(">>>>>>> REPLACE")].strip()
            except Exception:
                return ""
        return ""

    def _build_messages(
        self, system_prompt: str, history: list[dict[str, Any]], user_prompt: str, limit: int = 200
    ) -> list[dict[str, str]]:
        use_all = bool(getattr(self.config.prompts, "include_all_history", False))
        if use_all:
            msgs: list[dict[str, str]] = []
            tail = history[-limit:] if len(history) > limit else history
            for h in tail:
                role = h.get("role")
                content = h.get("content", "")
                if role in ("system", "user", "assistant") and content:
                    msgs.append({"role": role, "content": content})
            return msgs
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
