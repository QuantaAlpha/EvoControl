#!/usr/bin/env python3
"""
从实验目录中提取前 K 次迭代的最佳解决方案。

功能：
1. 遍历实验目录下的所有任务
2. 判断任务是否完成（有 traj.pool 文件）
3. 从 traj.pool 中提取前 K 次迭代中性能最优的解决方案
4. 输出为 final.json 格式

用法:
    python utils/extract_best_solutions.py <experiment_dir> -k <K> -o <output_path>

示例:
    # 提取前 10 次迭代的最佳解决方案
    python utils/extract_best_solutions.py trajectories_perf/deepseek-v3/Plan-Weighted-Local-Global-45its_20251220_074720 -k 10 -o output/final_k10.json

    # 提取前 20 次迭代的最佳解决方案
    python utils/extract_best_solutions.py trajectories_perf/deepseek-v3/Plan-Weighted-Local-Global-45its_20251220_074720 -k 20 -o output/final_k20.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple


class SolutionInfo(NamedTuple):
    """解决方案信息"""

    code: str
    performance: float
    iteration: int
    label: str


def parse_performance(perf_value) -> float:
    """解析性能值，返回浮点数（Infinity 返回 float('inf')）"""
    if perf_value is None:
        return float("inf")
    if isinstance(perf_value, str):
        if perf_value.lower() in ("infinity", "inf"):
            return float("inf")
        try:
            return float(perf_value)
        except ValueError:
            return float("inf")
    try:
        return float(perf_value)
    except (ValueError, TypeError):
        return float("inf")


def is_task_completed(task_dir: Path) -> bool:
    """
    判断任务是否完成。

    完成的标志：
    1. 存在 traj.pool 文件且不为空
    2. 或者存在 final.json 文件
    """
    final_json = task_dir / "final.json"

    if final_json.exists():
        try:
            with open(final_json, encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return True
        except Exception:
            pass

    return False


def extract_best_solution_from_traj_pool(
    traj_pool_path: Path, max_iterations: int, task_name: str
) -> SolutionInfo | None:
    """
    从 traj.pool 中提取前 max_iterations 次迭代中性能最优的解决方案。

    Args:
        traj_pool_path: traj.pool 文件路径
        max_iterations: 最大迭代次数
        task_name: 任务名称

    Returns:
        最优解决方案信息，如果没有有效解决方案则返回 None
    """
    if not traj_pool_path.exists():
        return None

    try:
        with open(traj_pool_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # 获取任务数据（可能是直接的任务名，也可能需要搜索）
    task_data = data.get(task_name)
    if not isinstance(task_data, dict):
        # 尝试获取第一个非空的任务
        for key, value in data.items():
            if isinstance(value, dict) and "problem" in value:
                task_data = value
                break

    if not task_data:
        return None

    # 收集所有在 max_iterations 范围内的解决方案
    candidates: list[SolutionInfo] = []

    for key, value in task_data.items():
        if key == "problem" or not isinstance(value, dict):
            continue

        iteration = value.get("iteration")
        performance = value.get("performance")
        code = value.get("code")

        if iteration is None or code is None:
            continue

        try:
            iter_num = int(iteration)
        except (ValueError, TypeError):
            continue

        # 只考虑前 max_iterations 次迭代
        if iter_num > max_iterations:
            continue

        perf_val = parse_performance(performance)

        # 跳过没有代码的解决方案
        if not code or not code.strip():
            continue

        candidates.append(
            SolutionInfo(
                code=code,
                performance=perf_val,
                iteration=iter_num,
                label=key,
            )
        )

    if not candidates:
        return None

    # 过滤掉 Infinity 的解决方案
    valid_candidates = [c for c in candidates if c.performance != float("inf")]

    if not valid_candidates:
        # 如果所有解决方案都是 Infinity，跳过该任务
        return None

    # 选择性能最优的（最小值）
    best = min(valid_candidates, key=lambda x: (x.performance, -x.iteration))
    return best


def load_fallback_data(
    report_file: Path | None,
    solution_file: Path | None,
) -> tuple[dict | None, dict | None]:
    """
    加载 fallback 数据：报告文件和解决方案文件。

    Args:
        report_file: 报告文件路径（如 report/deepseek-chat-v3-0324.json）
        solution_file: 解决方案文件路径（如 data/deepseek-chat-v3-0324-sol.json）

    Returns:
        (report_data, solution_data) 元组
    """
    report_data = None
    solution_data = None

    if report_file and report_file.exists():
        try:
            with open(report_file, encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception as e:
            print(f"Warning: 无法加载报告文件 {report_file}: {e}", file=sys.stderr)

    if solution_file and solution_file.exists():
        try:
            with open(solution_file, encoding="utf-8") as f:
                solution_data = json.load(f)
        except Exception as e:
            print(f"Warning: 无法加载解决方案文件 {solution_file}: {e}", file=sys.stderr)

    return report_data, solution_data


def get_fallback_solution(
    task_name: str,
    report_data: dict | None,
    solution_data: dict | None,
    lang: str = "python3",
) -> str | None:
    """
    从 fallback 数据中获取解决方案。

    如果任务在报告中显示通过（integral_score > 0），则从解决方案文件中提取代码。

    Args:
        task_name: 任务名称
        report_data: 报告数据
        solution_data: 解决方案数据
        lang: 编程语言（默认 python3）

    Returns:
        解决方案代码，如果不可用则返回 None
    """
    if not report_data or not solution_data:
        return None

    # 从报告中检查任务是否通过
    # 报告结构: {model_name: {per_task: {task_name: {lang: {integral_score: ...}}}}}
    per_task = None
    for _model_name, model_data in report_data.items():
        if isinstance(model_data, dict) and "per_task" in model_data:
            per_task = model_data.get("per_task", {})
            break

    if not per_task:
        return None

    task_report = per_task.get(task_name, {})
    lang_report = task_report.get(lang, {})
    integral_score = lang_report.get("integral_score", 0)

    # 如果 integral_score > 0，则任务通过
    if integral_score <= 0:
        return None

    # 从解决方案文件中提取代码
    task_solution = solution_data.get(task_name, {})
    code = task_solution.get(lang)

    if code and isinstance(code, str) and code.strip():
        return code

    return None


def extract_best_solutions(
    experiment_dir: Path,
    max_iterations: int,
    output_path: Path | None = None,
    verbose: bool = True,
    report_file: Path | None = None,
    solution_file: Path | None = None,
    lang: str = "python3",
) -> dict[str, str]:
    """
    从实验目录中提取所有已完成任务的最佳解决方案。

    Args:
        experiment_dir: 实验目录路径
        max_iterations: 最大迭代次数
        output_path: 输出文件路径（可选）
        verbose: 是否打印详细信息
        report_file: 报告文件路径，用于 fallback 检查（可选）
        solution_file: 解决方案文件路径，用于 fallback 提取（可选）
        lang: 编程语言（默认 python3）

    Returns:
        任务名到最佳代码的映射
    """
    if not experiment_dir.exists():
        print(f"Error: 实验目录不存在: {experiment_dir}", file=sys.stderr)
        return {}

    # 加载 fallback 数据
    report_data, solution_data = load_fallback_data(report_file, solution_file)
    if report_file and solution_file:
        if report_data and solution_data:
            if verbose:
                print(f"已加载 fallback 数据: report={report_file}, solution={solution_file}")
        else:
            if verbose:
                print("Warning: fallback 数据加载不完整")

    # 收集所有任务目录
    task_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    task_dirs.sort(key=lambda x: x.name)

    if verbose:
        print(f"发现 {len(task_dirs)} 个任务目录")
        print(f"提取前 {max_iterations} 次迭代的最佳解决方案...")

    results: dict[str, str] = {}
    stats = {
        "total": len(task_dirs),
        "completed": 0,
        "extracted": 0,
        "no_valid_solution": 0,
        "skipped": 0,
        "fallback_used": 0,
    }
    fallback_tasks: list[str] = []  # 记录使用 fallback 的任务名称

    for task_dir in task_dirs:
        task_name = task_dir.name

        # 检查任务是否完成
        if not is_task_completed(task_dir):
            stats["skipped"] += 1
            if verbose:
                print(f"  [跳过] {task_name}: 任务未完成")
            continue

        stats["completed"] += 1

        # 提取最佳解决方案
        traj_pool_path = task_dir / "traj.pool"
        best_solution = extract_best_solution_from_traj_pool(traj_pool_path, max_iterations, task_name)

        if best_solution is None:
            # 尝试 fallback
            fallback_code = get_fallback_solution(task_name, report_data, solution_data, lang)
            if fallback_code:
                stats["extracted"] += 1
                stats["fallback_used"] += 1
                fallback_tasks.append(task_name)
                results[task_name] = fallback_code
                if verbose:
                    print(f"  [fallback] {task_name}: 使用预生成解决方案")
            else:
                stats["no_valid_solution"] += 1
                if verbose:
                    print(f"  [跳过] {task_name}: 没有找到有效解决方案 (全为 Infinity 或无解)")
            continue

        stats["extracted"] += 1
        results[task_name] = best_solution.code

        if verbose:
            perf_str = f"{best_solution.performance:.6f}" if best_solution.performance != float("inf") else "Infinity"
            print(f"  [提取] {task_name}: 迭代 {best_solution.iteration}, 性能 {perf_str}")

    if verbose:
        print("\n统计:")
        print(f"  - 任务总数: {stats['total']}")
        print(f"  - 已完成: {stats['completed']}")
        print(f"  - 成功提取: {stats['extracted']}")
        if stats["fallback_used"] > 0:
            print(f"    - 其中 fallback: {stats['fallback_used']}")
            for fb_task in fallback_tasks:
                print(f"      • {fb_task}")
        print(f"  - 无有效解: {stats['no_valid_solution']}")
        print(f"  - 跳过 (未完成): {stats['skipped']}")

    # 输出到文件
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n已保存到: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="从实验目录中提取前 K 次迭代的最佳解决方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("experiment_dir", type=str, help="实验目录路径")
    parser.add_argument(
        "-k",
        "--max-iterations",
        type=int,
        required=True,
        help="最大迭代次数（只考虑前 K 次迭代）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出文件路径",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="静默模式，不打印详细信息",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="报告文件路径（用于 fallback 检查，如 report/deepseek-chat-v3-0324.json）",
    )
    parser.add_argument(
        "--solution-file",
        type=str,
        default=None,
        help="预生成解决方案文件路径（用于 fallback 提取，如 data/deepseek-chat-v3-0324-sol.json）",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="python3",
        help="编程语言（默认 python3，可选 cpp）",
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_path = Path(args.output)
    report_file = Path(args.report_file) if args.report_file else None
    solution_file = Path(args.solution_file) if args.solution_file else None

    results = extract_best_solutions(
        experiment_dir=experiment_dir,
        max_iterations=args.max_iterations,
        output_path=output_path,
        verbose=not args.quiet,
        report_file=report_file,
        solution_file=solution_file,
        lang=args.lang,
    )

    if not results:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
