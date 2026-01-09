#!/usr/bin/env python3
"""
异常任务删除脚本

功能：扫描指定目录下的任务文件夹，统计 se_framework.log 中 WARNING 的个数。
如果 WARNING 个数超过阈值，则删除整个任务目录。

使用方法：
    # 干运行模式（只显示统计，不删除）
    python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50 --dry-run

    # 实际删除模式
    python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50

    # 详细模式（显示所有任务统计）
    python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50 --verbose
"""

import argparse
import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple


def count_warnings_in_log(log_path: Path) -> int:
    """统计日志文件中 WARNING 的个数"""
    if not log_path.exists():
        return -1  # 文件不存在

    warning_count = 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # 匹配格式: "- WARNING -"
                if " - WARNING - " in line or "- WARNING -" in line:
                    warning_count += 1
    except Exception as e:
        print(f"  [错误] 读取文件 {log_path} 失败: {e}")
        return -1

    return warning_count


def scan_task_directories(base_dir: Path) -> List[Tuple[Path, int]]:
    """
    扫描基础目录下的所有任务目录，返回 (任务目录, WARNING数量) 列表
    """
    results = []

    # 获取所有直接子目录（任务目录）
    if not base_dir.exists():
        print(f"[错误] 目录不存在: {base_dir}")
        return results

    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            # 跳过特殊目录（如 global_memory）
            if item.name.startswith(".") or item.name == "global_memory":
                continue

            log_path = item / "se_framework.log"
            warning_count = count_warnings_in_log(log_path)
            results.append((item, warning_count))

    return results


def cleanup_tasks(base_dir: Path, threshold: int, dry_run: bool = True, verbose: bool = False):
    """
    清理异常任务

    Args:
        base_dir: 任务目录的基础路径
        threshold: WARNING 阈值，超过此值的任务将被删除
        dry_run: 如果为 True，只显示统计信息而不实际删除
        verbose: 如果为 True，显示所有任务的统计信息
    """
    print(f"=" * 60)
    print(f"异常任务清理工具")
    print(f"=" * 60)
    print(f"目录: {base_dir}")
    print(f"阈值: {threshold}")
    print(f"模式: {'干运行（不删除）' if dry_run else '实际删除'}")
    print(f"=" * 60)
    print()

    # 扫描任务目录
    results = scan_task_directories(base_dir)

    if not results:
        print("[警告] 未找到任何任务目录")
        return

    # 统计信息
    total_tasks = len(results)
    missing_log = 0
    normal_tasks = 0
    abnormal_tasks = 0
    tasks_to_delete = []

    print(f"{'任务名称':<60} {'WARNING数':<10} {'状态':<10}")
    print("-" * 80)

    for task_dir, warning_count in results:
        task_name = task_dir.name

        if warning_count < 0:
            status = "无日志"
            missing_log += 1
        elif warning_count > threshold:
            status = "⚠️ 异常"
            abnormal_tasks += 1
            tasks_to_delete.append((task_dir, warning_count))
        else:
            status = "✓ 正常"
            normal_tasks += 1

        # 根据 verbose 参数决定是否显示所有任务
        if verbose or warning_count > threshold or warning_count < 0:
            # 截断过长的任务名
            display_name = task_name[:58] + ".." if len(task_name) > 60 else task_name
            print(f"{display_name:<60} {warning_count:<10} {status:<10}")

    print("-" * 80)
    print()

    # 汇总统计
    print(f"统计汇总:")
    print(f"  总任务数:     {total_tasks}")
    print(f"  正常任务:     {normal_tasks}")
    print(f"  异常任务:     {abnormal_tasks}")
    print(f"  缺失日志:     {missing_log}")
    print()

    # 删除操作
    if tasks_to_delete:
        print(f"待删除任务列表 (WARNING > {threshold}):")
        for task_dir, warning_count in tasks_to_delete:
            print(f"  - {task_dir.name} (WARNING: {warning_count})")
        print()

        if dry_run:
            print(f"[干运行模式] 以上 {len(tasks_to_delete)} 个任务将被删除")
            print(f"若要实际删除，请移除 --dry-run 参数")
        else:
            print(f"正在删除 {len(tasks_to_delete)} 个异常任务...")
            deleted_count = 0
            for task_dir, warning_count in tasks_to_delete:
                try:
                    shutil.rmtree(task_dir)
                    print(f"  ✓ 已删除: {task_dir.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {task_dir.name} - {e}")

            print()
            print(f"删除完成: 成功删除 {deleted_count}/{len(tasks_to_delete)} 个任务")
    else:
        print("未发现需要删除的异常任务")


def main():
    parser = argparse.ArgumentParser(
        description="异常任务删除工具 - 根据 WARNING 数量清理任务目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 干运行模式，查看统计（推荐先执行）
  python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50 --dry-run

  # 实际删除异常任务
  python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50

  # 显示所有任务统计
  python cleanup_abnormal_tasks.py /path/to/trajectories --threshold 50 --verbose --dry-run
        """,
    )

    parser.add_argument("directory", type=str, help="包含任务目录的基础路径")

    parser.add_argument(
        "--threshold", "-t", type=int, default=50, help="WARNING 阈值，超过此值的任务将被删除 (默认: 50)"
    )

    parser.add_argument("--dry-run", "-n", action="store_true", help="干运行模式，只显示统计信息而不实际删除")

    parser.add_argument("--verbose", "-v", action="store_true", help="详细模式，显示所有任务的统计信息")

    args = parser.parse_args()

    base_dir = Path(args.directory).resolve()

    cleanup_tasks(base_dir=base_dir, threshold=args.threshold, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
