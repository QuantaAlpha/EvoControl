"""
PerfAgent 批量运行脚本（并行、独立）

目的：
- 通过并行线程直接调用 `run_single_instance` 实现批量运行（不再嵌套子进程）
- 每个任务将所有目录设置为：配置文件所在目录 / 任务名（文件名）
- 统一使用 utils.log.get_file_logger 进行批量日志记录

用法：
python -m perfagent.run_batch --instances-dir <path> [--config config.yaml] [--output summary.json]
"""

import argparse
import copy
import json
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import PerfAgentConfig, load_config
from .run import run_single_instance
from .utils.log import get_se_logger


def _json_safe(obj):
    """Recursively convert objects to JSON-safe values.

    Mirrors perfagent.run._json_safe and adds support for numpy scalars.
    """
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, float):
                if math.isfinite(obj):
                    return obj
                if math.isnan(obj):
                    return "NaN"
                return "-Infinity" if obj < 0 else "Infinity"
            return obj

        item_fn = getattr(obj, "item", None)
        if callable(item_fn):
            try:
                return _json_safe(item_fn())
            except Exception:
                pass

        if hasattr(obj, "__fspath__"):
            try:
                return str(obj)
            except Exception:
                pass

        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]

        if hasattr(obj, "__dict__"):
            try:
                return {k: _json_safe(v) for k, v in obj.__dict__.items()}
            except Exception:
                pass

        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dicts without overwriting nested keys.

    - For dict values: merge recursively.
    - For non-dict values (including lists/strings): replace with src.
    This avoids losing existing nested config like `prompts` when applying per-task templates.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _build_task_template_config(templates_root: Path | None, task_name: str, logger: logging.Logger) -> dict[str, Any]:
    """从模板根目录为指定任务读取单一 YAML 文件为 dict。

    约定：每个任务只有一个模板文件，位置为：
    - <root>/<task_name>.yaml 或 <root>/<task_name>.yml
    若两者同时存在，优先使用 .yaml。
    """
    if not templates_root:
        return {}
    try:
        root = Path(templates_root)
        if not root.exists():
            return {}
    except Exception:
        return {}

    yaml_fp = root / f"{task_name}.yaml"
    yml_fp = root / f"{task_name}.yml"

    selected: Path | None = None
    if yaml_fp.exists() and yml_fp.exists():
        logger.warning(f"任务 {task_name} 的 .yaml 与 .yml 同时存在，优先使用 .yaml，忽略 .yml")
        selected = yaml_fp
    elif yaml_fp.exists():
        selected = yaml_fp
    elif yml_fp.exists():
        selected = yml_fp
    else:
        logger.debug(f"任务 {task_name} 未找到模板文件，使用基础配置")
        return {}

    try:
        with open(selected, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
        logger.warning(f"模板 {selected} 格式非 dict，忽略该文件")
    except Exception as e:
        logger.warning(f"读取任务模板 {selected} 失败: {e}")

    return {}


def _create_task_temp_config(
    base_cfg_template: dict[str, Any],
    templates_root: Path | None,
    task_name: str,
    logger: logging.Logger,
) -> tuple[Path | None, bool]:
    """基于模板为指定任务创建临时配置文件。

    - 读取单一任务模板为 dict（通过 _build_task_template_config）。
    - 将模板 dict 顶层 update 合并到基础配置副本。
    - 写入系统临时文件（.yaml），返回其路径与是否需要清理。
    """
    if not templates_root:
        return None, False
    try:
        task_cfg_dict = _build_task_template_config(templates_root, task_name, logger)
        if not task_cfg_dict:
            logger.debug(f"任务 {task_name} 未找到模板内容，使用基础配置")
            return None, False

        logger.info(f"任务 {task_name} 模板内容：{task_cfg_dict}")
        merged_cfg = copy.deepcopy(base_cfg_template)
        # 使用递归合并，避免覆盖 prompts 等嵌套配置
        merged_cfg = _deep_update(merged_cfg, task_cfg_dict)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        logger.info(f"为任务 {task_name} 创建临时配置文件 {tmp.name}")
        try:
            yaml.safe_dump(merged_cfg, tmp, allow_unicode=True, sort_keys=False)
            tmp.flush()
            return Path(tmp.name), True
        finally:
            tmp.close()
    except Exception as ge:
        logger.warning(f"为任务 {task_name} 生成临时配置失败: {ge}")
        return None, False


def _run_instance_direct(
    instance_file: Path,
    config: PerfAgentConfig,
    config_path: Path | None,
    base_dir: Path,
    per_task_config_path: Path | None = None,
    cleanup_config: bool = False,
) -> tuple[str, dict[str, Any]]:
    """直接在当前进程运行单个实例，返回 (task_name, result_dict 或 error)。"""
    task_name = instance_file.stem
    instance_dir = base_dir / task_name
    instance_dir.mkdir(parents=True, exist_ok=True)
    output_file = instance_dir / "result.json"

    # 基于 per_task_config_path 或内存配置选择配置
    try:
        if per_task_config_path and Path(per_task_config_path).exists():
            # 若存在为当前任务生成的临时配置文件，则优先使用该配置
            local_config = load_config(per_task_config_path)
        else:
            # 否则直接使用已在主流程中应用了 CLI 覆盖项的内存配置，避免丢失 overrides.initial_code_dir
            local_config = config
    except Exception as e:
        return task_name, {"error": f"Failed to load config for task: {e}"}

    # 记录运行配置
    try:
        cfg_json = json.dumps(local_config.to_dict(), ensure_ascii=False, indent=2)
    except Exception:
        cfg_json = str(local_config)
    logging.getLogger("perfagent.run_batch").info(f"任务 {task_name} 运行配置:\n{cfg_json}")

    # 运行单实例
    try:
        result = run_single_instance(local_config, instance_file, base_dir=base_dir)
        # 与原子进程行为一致：写出 result.json
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
        except Exception as we:
            # 不影响返回，但记录为错误结构
            return task_name, {"error": f"Failed to write result.json: {we}", "result": result}
        return task_name, result
    except Exception as e:
        return task_name, {"error": str(e)}
    finally:
        if cleanup_config and per_task_config_path:
            try:
                p = Path(per_task_config_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def run_batch_instances(
    config: PerfAgentConfig,
    instances_dir: Path,
    config_path: Path | None,
    base_dir: Path,
    instance_templates_dir: Path | None = None,
) -> dict[str, Any]:
    get_se_logger(
        "perfagent.run_batch",
        base_dir / "perfagent.log",
        level=getattr(logging, config.logging.log_level.upper()),
    )
    logger = logging.getLogger("perfagent.run_batch")

    instance_files = list(instances_dir.glob("*.json"))
    if not instance_files:
        raise ValueError(f"在 {instances_dir} 中未找到实例文件")

    logger.info(f"找到 {len(instance_files)} 个实例文件，准备并行运行")

    results = {}
    successful = 0
    failed = 0
    # 并发度优先使用顶层配置的 max_workers（用户要求），否则回退到 runtime.max_workers
    max_workers = max(1, int(getattr(config, "max_workers", getattr(config.runtime, "max_workers", 4))))
    logger.info(f"并发度: max_workers={max_workers}")

    # 预读取基础配置（用于生成每任务配置）
    # 始终使用已加载并应用 CLI 覆盖项的内存配置，避免丢失 overrides.initial_code_dir 等动态覆盖。
    try:
        base_cfg_template = config.to_dict()
    except Exception as e:
        logger.warning(f"序列化内存配置失败: {e}，尝试回退为空配置")
        base_cfg_template = {}

    # 默认模板目录：优先使用传入参数，其次使用配置中的 prompts.instance_templates_dir
    templates_root = instance_templates_dir or getattr(getattr(config, "prompts", None), "instance_templates_dir", None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for f in instance_files:
            task_name = f.stem
            # 基于模板目录构建每任务临时配置（封装函数）
            per_task_cfg, cleanup_cfg = _create_task_temp_config(base_cfg_template, templates_root, task_name, logger)
            # 将 per-task 临时配置路径注入清晰日志，便于调试
            if per_task_cfg:
                logger.info(f"任务 {task_name} 使用临时配置: {per_task_cfg}")
            future = executor.submit(_run_instance_direct, f, config, config_path, base_dir, per_task_cfg, cleanup_cfg)
            future_map[future] = f
        for future in as_completed(future_map):
            inst_file = future_map[future]
            task_name = inst_file.stem
            try:
                tn, result = future.result()
                if "error" in result:
                    failed += 1
                    results[task_name] = result
                    logger.error(f"实例 {task_name} 优化失败: {result['error']}")
                else:
                    instance_id = result.get("instance_id", task_name)
                    results[instance_id] = result
                    successful += 1
                    logger.info(f"实例 {instance_id} 优化成功")
            except Exception as e:
                failed += 1
                results[task_name] = {"error": str(e)}
                logger.error(f"实例 {task_name} 优化失败: {e}")

    summary = {"total_instances": len(instance_files), "successful": successful, "failed": failed, "results": results}
    logger.info(f"批量优化完成: {successful}/{len(instance_files)} 成功")
    return summary


def main():
    parser = argparse.ArgumentParser(description="PerfAgent 批量运行脚本")
    parser.add_argument("--config", type=Path, help="配置文件路径")
    parser.add_argument("--instances-dir", type=Path, required=True, help="实例目录路径（批量运行）")
    parser.add_argument("--output", type=Path, help="结果输出文件路径")
    parser.add_argument("--base-dir", type=Path, help="基础目录路径，使用实例文件名作为子目录控制")
    parser.add_argument("--max-workers", type=int, help="并发工作线程（顶层配置）")
    parser.add_argument("--instance-templates-dir", type=Path, help="每任务模板根目录（按实例名匹配）")
    parser.add_argument("--initial-code-dir", type=Path, help="每实例初始代码目录（按实例名匹配文件）")
    args = parser.parse_args()

    # 加载并覆盖配置
    config = load_config(args.config)
    config.apply_cli_overrides(args)

    # 计算基础目录：优先使用 --base-dir；否则使用 配置文件所在目录；再否则使用 log_dir
    if args.base_dir:
        base_dir = args.base_dir
    elif args.config and Path(args.config).exists():
        base_dir = Path(args.config).parent
    else:
        base_dir = Path(config.logging.log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 顶层日志器绑定
    get_se_logger(
        "perfagent.run_batch.main",
        base_dir / "perfagent.log",
        level=getattr(logging, config.logging.log_level.upper()),
    )
    logger = logging.getLogger("perfagent.run_batch.main")
    logger.info("PerfAgent 批量运行启动")
    logger.info(f"基础目录: {base_dir}")

    try:
        result = run_batch_instances(
            config,
            args.instances_dir,
            args.config,
            base_dir,
            args.instance_templates_dir,
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {args.output}")
        logger.info("PerfAgent 批量运行完成")
    except Exception as e:
        logger.error(f"批量运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
