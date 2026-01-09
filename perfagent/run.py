"""
PerfAgent 单实例运行脚本

提供命令行接口来运行单个性能优化任务。
统一使用 utils.log.get_file_logger 初始化所有日志器（带 emoji）。
"""

import argparse
import copy
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import EffiBenchXInstance, PerfAgent
from .config import PerfAgentConfig, load_config
from .utils.log import get_se_logger

# 不再需要全局初始化函数，直接在 main 中绑定文件日志器


def load_instance_data(instance_path: Path) -> EffiBenchXInstance:
    """加载实例数据为 EffiBenchXInstance dataclass"""
    with open(instance_path, encoding="utf-8") as f:
        data = json.load(f)
    inst = EffiBenchXInstance.from_dict(data)
    # 使用文件名（不含扩展名）作为任务名
    inst.task_name = instance_path.stem
    return inst


def run_single_instance(config: PerfAgentConfig, instance_path: Path, base_dir: Path | None = None) -> dict[str, Any]:
    """运行单个实例的优化"""
    # 初始绑定主日志器到 base_dir（或配置的 log_dir），后续在实例目录内绑定专属文件日志器
    try:
        pre_log_path = Path(base_dir) / "perfagent.log" if base_dir else Path(config.logging.log_dir) / "perfagent.log"
        get_se_logger(
            "perfagent.run_single.main",
            pre_log_path,
            emoji="🚀",
            level=getattr(logging, config.logging.log_level.upper()),
        )
    except Exception:
        # 回退到配置的 log_dir
        get_se_logger(
            "perfagent.run_single.main",
            Path(config.logging.log_dir) / "perfagent.log",
            emoji="🚀",
            level=getattr(logging, config.logging.log_level.upper()),
        )
    logger = logging.getLogger("perfagent.run_single.main")

    try:
        # 加载实例数据
        instance = load_instance_data(instance_path)
        # 同时输出 JSON 内 id 与文件名，以方便排查
        logger.info(f"加载实例: file={instance_path.stem}, json_id={getattr(instance, 'id', 'unknown')}")

        # 统一任务名与实例ID
        task_name = getattr(instance, "task_name", instance_path.stem)
        try:
            instance.id = task_name
        except Exception:
            pass

        # 计算并创建实例输出目录
        if base_dir:
            instance_output_dir = Path(base_dir) / task_name
        else:
            traj_dir = Path(config.logging.trajectory_dir)
            # 若 CLI 已传入以任务名为末级目录（来自 run_batch），避免重复嵌套
            if traj_dir.name == task_name:
                instance_output_dir = traj_dir
            else:
                instance_output_dir = traj_dir / task_name
        instance_output_dir.mkdir(parents=True, exist_ok=True)

        # 在实例目录内绑定专属日志文件（覆盖之前的主日志器用途）
        # 使用唯一的 logger 名称以避免并发复用导致串写
        instance_logger_name = f"perfagent.run_single.instance.{task_name}"
        get_se_logger(
            instance_logger_name,
            instance_output_dir / "perfagent.log",
            emoji="🎯",
            level=getattr(logging, config.logging.log_level.upper()),
            also_stream=False,
        )
        logger = logging.getLogger(instance_logger_name)

        # 为当前实例定制配置：将轨迹目录重定向到实例目录
        local_config = copy.deepcopy(config)
        local_config.logging.trajectory_dir = instance_output_dir
        local_config.logging.log_dir = instance_output_dir

        # 创建并运行 agent
        agent = PerfAgent(local_config)
        result = agent.run(instance)

        logger.info(f"优化完成: {result['instance_id']}")
        # 写出问题描述到 <instance_dir>/<task_name>.problem
        try:
            problem_text = (
                getattr(instance, "description_md", None)
                or getattr(instance, "description", None)
                or getattr(instance, "title", "")
            )
            if problem_text:
                problem_file = instance_output_dir / f"{task_name}.problem"
                with open(problem_file, "w", encoding="utf-8") as pf:
                    pf.write(problem_text)
                logger.info(f"写出问题描述: {problem_file}")
            else:
                logger.warning(f"实例 {task_name} 缺少问题描述字段，跳过写入 .problem")
        except Exception as e:
            logger.error(f"写出 .problem 失败: {e}")

        # 从轨迹 submission 生成 <instance_dir>/<task_name>.pred，并附加语言、优化目标与性能单位
        try:
            submission_code = ""
            traj_path = Path(result.get("trajectory_file", ""))
            info = {}
            if traj_path.exists():
                with open(traj_path, encoding="utf-8") as tf:
                    traj_json = json.load(tf)
                info = traj_json.get("info") or traj_json.get("metadata") or {}
                submission_code = (
                    info.get("final_best_code") or info.get("submission") or info.get("final_submission_code") or ""
                )
            pred_file = instance_output_dir / f"{task_name}.pred"
            with open(pred_file, "w", encoding="utf-8") as pf:
                pf.write((submission_code or ""))
            logger.info(f"写出预测结果: {pred_file}")
        except Exception as e:
            logger.error(f"写出 .pred 失败: {e}")

        return result

    except Exception as e:
        logger.error(f"运行实例失败: {e}")
        raise


def _json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-safe values.

    Handles:
    - Non-finite floats (incl. numpy scalars) -> "NaN"/"Infinity"/"-Infinity" strings
    - numpy scalars via `.item()` to native Python types
    - Path-like objects -> str
    - datetime -> ISO string
    - dict keys coerced to str; values sanitized recursively
    - lists/tuples/sets -> lists with sanitized items
    - Falls back to `str(obj)` for unknown objects
    """
    try:
        # Basic primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, float):
                if math.isfinite(obj):
                    return obj
                if math.isnan(obj):
                    return "NaN"
                return "-Infinity" if obj < 0 else "Infinity"
            return obj

        # numpy/scalar-like: try to convert using item()
        item_fn = getattr(obj, "item", None)
        if callable(item_fn):
            try:
                return _json_safe(item_fn())
            except Exception:
                pass

        # Path-like
        if hasattr(obj, "__fspath__"):
            try:
                return str(obj)
            except Exception:
                pass

        # datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Containers
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]

        # Fallback for other objects with __dict__
        if hasattr(obj, "__dict__"):
            try:
                return {k: _json_safe(v) for k, v in obj.__dict__.items()}
            except Exception:
                pass

        # Final fallback: string representation
        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


# 批量运行逻辑已迁移到 perfagent/run_batch.py，这里只支持单实例


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PerfAgent - 代码性能优化工具")

    # 基础参数
    parser.add_argument("--config", type=Path, help="配置文件路径")
    parser.add_argument("--instance", type=Path, help="单个实例文件路径")
    parser.add_argument("--output", type=Path, help="结果输出文件路径")
    parser.add_argument("--base-dir", type=Path, help="实例输出基目录（生成 .traj/.problem/.pred 的父目录）")

    # 配置覆盖参数（全部交由 PerfAgentConfig.apply_cli_overrides 处理）
    parser.add_argument("--max-iterations", type=int, help="最大迭代次数")
    parser.add_argument("--model", type=str, help="模型名称")
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别"
    )
    parser.add_argument("--trajectory-dir", type=Path, help="轨迹保存目录")
    parser.add_argument("--log-dir", type=Path, help="日志保存目录")

    # 新增参数：语言覆盖与优化方向
    parser.add_argument("--language", type=str, help="覆盖实例语言 (python3/cpp/java/javascript/golang)")
    parser.add_argument("--opt-target", type=str, choices=["runtime", "memory"], default="runtime", help="优化方向")

    # LLM 客户端配置（可选）
    parser.add_argument("--llm-use", action="store_true", help="启用LLM调用")
    parser.add_argument("--llm-api-base", type=str, help="LLM API 基础地址")
    parser.add_argument("--llm-api-key", type=str, help="LLM API 密钥")
    parser.add_argument("--llm-model", type=str, help="LLM 模型名称，例如 openai/deepseek-chat")
    parser.add_argument("--llm-temp", type=float, help="LLM 温度")
    parser.add_argument("--llm-max-output", type=int, help="LLM 最大输出 token")
    parser.add_argument("--llm-timeout", type=float, help="LLM 请求超时（秒）")
    parser.add_argument("--llm-max-retries", type=int, help="LLM 最大重试次数")
    parser.add_argument("--llm-retry-delay", type=float, help="LLM 重试初始等待（秒）")
    parser.add_argument("--llm-retry-backoff", type=float, help="LLM 重试指数退避因子")
    parser.add_argument("--llm-retry-jitter", type=float, help="LLM 重试抖动秒数上限")
    parser.add_argument("--llm-log-io", action="store_true", help="记录 LLM 输入与输出（可能包含代码）")
    parser.add_argument("--llm-log-sanitize", action="store_true", help="记录前进行敏感信息脱敏")
    parser.add_argument("--early-stop-no-improve", type=int, help="连续未改进次数达到阈值后提前停止")
    # 不再接受 instance-templates-dir，改由 prompts.additional_requirements 承载（SE 层负责生成）
    # 允许通过 CLI 指定 per-instance 初始代码目录（按实例名匹配）
    parser.add_argument("--initial-code-dir", type=Path, help="每实例初始代码目录（按实例文件名匹配）")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 统一由配置对象完成 CLI 覆盖
    config.apply_cli_overrides(args)

    # 绑定主运行日志器到实例目录（若提供 base_dir 与 instance），否则使用配置中的 log_dir
    try:
        if args.base_dir and args.instance:
            task_name = Path(args.instance).stem
            main_log_dir = Path(args.base_dir) / task_name
            main_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = main_log_dir / "perfagent.log"
        else:
            log_path = Path(config.logging.log_dir) / "perfagent.log"
    except Exception:
        log_path = Path(config.logging.log_dir) / "perfagent.log"

    get_se_logger(
        "perfagent.run_single.main",
        log_path,
        emoji="🚀",
        level=getattr(logging, config.logging.log_level.upper()),
    )
    logger = logging.getLogger("perfagent.run_single.main")
    logger.info("PerfAgent 启动")

    # 打印所有配置项
    logger.info(f"配置: {json.dumps(_json_safe(config.to_dict()), indent=2, ensure_ascii=False)}")

    try:
        # 仅支持单实例运行
        if not args.instance:
            logger.error("请指定 --instance 参数（run.py 仅支持单实例）")
            sys.exit(1)

        result = run_single_instance(config, args.instance, base_dir=args.base_dir)

        # 保存结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {args.output}")

        logger.info("PerfAgent 运行完成")

    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
