#!/usr/bin/env python3
"""
统计 trajectories_perf 目录中每个任务设置的成功任务数量。

成功的任务定义：在 final.json 中存在非空的解（即存在有 performance 的解）

用法:
    python count_task_success.py <目录路径> [选项]

选项:
    --show-failed    显示失败任务的详细列表
    --json           以 JSON 格式输出结果

示例:
    python count_task_success.py ./trajectories_perf/Claude-4.5-Sonnet
    python count_task_success.py ./trajectories_perf/Claude-4.5-Sonnet --show-failed
    python count_task_success.py ./trajectories_perf/Claude-4.5-Sonnet --json
"""

import json
import sys
from pathlib import Path


def analyze_final_json(final_json_path: Path) -> dict:
    """
    分析 final.json 文件，返回详细的成功/失败信息。

    返回: {
        'success_count': int,
        'total_count': int,
        'success_tasks': list[str],
        'failed_tasks': list[str]
    }
    """
    with open(final_json_path, encoding="utf-8") as f:
        data = json.load(f)

    success_tasks = [k for k, v in data.items() if v]
    failed_tasks = [k for k, v in data.items() if not v]

    return {
        "success_count": len(success_tasks),
        "total_count": len(data),
        "success_tasks": sorted(success_tasks),
        "failed_tasks": sorted(failed_tasks),
    }


def analyze_directory(target_dir: str, show_failed: bool = False, output_json: bool = False) -> dict:
    """
    分析目标目录中所有任务设置的成功率。

    返回: 分析结果字典
    """
    target_path = Path(target_dir)

    if not target_path.exists():
        print(f"错误: 目录不存在 - {target_dir}", file=sys.stderr)
        sys.exit(1)

    if not target_path.is_dir():
        print(f"错误: 不是目录 - {target_dir}", file=sys.stderr)
        sys.exit(1)

    # 收集所有任务设置目录
    task_settings = []
    for item in sorted(target_path.iterdir()):
        if item.is_dir():
            final_json = item / "final.json"
            if final_json.exists():
                task_settings.append((item.name, final_json))

    if not task_settings:
        print(f"警告: 在 {target_dir} 中未找到包含 final.json 的子目录", file=sys.stderr)
        return {}

    results = {"directory": str(target_dir), "settings": {}, "summary": {"total_success": 0, "total_tasks": 0}}

    for name, final_json in task_settings:
        try:
            analysis = analyze_final_json(final_json)
            results["settings"][name] = analysis
            results["summary"]["total_success"] += analysis["success_count"]
            results["summary"]["total_tasks"] += analysis["total_count"]
        except Exception as e:
            results["settings"][name] = {"error": str(e)}

    # JSON 输出模式
    if output_json:
        # 精简输出，不包含完整任务列表
        json_output = {
            "directory": results["directory"],
            "settings": {
                name: {
                    "success_count": data["success_count"],
                    "total_count": data["total_count"],
                    "success_rate": round(data["success_count"] / data["total_count"] * 100, 2)
                    if data["total_count"] > 0
                    else 0,
                    "failed_tasks": data.get("failed_tasks", []),
                }
                if "success_count" in data
                else {"error": data.get("error")}
                for name, data in results["settings"].items()
            },
            "summary": results["summary"],
        }
        print(json.dumps(json_output, ensure_ascii=False, indent=2))
        return results

    # 表格输出模式
    print(f"\n{'=' * 80}")
    print(f"目录: {target_dir}")
    print(f"{'=' * 80}")
    print(f"\n{'任务设置':<60} {'成功/总数':>10} {'成功率':>10}")
    print(f"{'-' * 80}")

    for name, data in results["settings"].items():
        if "error" in data:
            print(f"{name:<60} {'错误':>15}: {data['error']}")
        else:
            success = data["success_count"]
            total = data["total_count"]
            rate = (success / total * 100) if total > 0 else 0
            print(f"{name:<60} {success:>4}/{total:<5} {rate:>9.1f}%")

            # 显示失败任务详情
            if show_failed and data["failed_tasks"]:
                print(f"    └── 失败任务 ({len(data['failed_tasks'])} 个):")
                for task in data["failed_tasks"]:
                    print(f"        • {task}")

    # 打印汇总
    print(f"{'-' * 80}")
    total_success = results["summary"]["total_success"]
    total_tasks = results["summary"]["total_tasks"]
    overall_rate = (total_success / total_tasks * 100) if total_tasks > 0 else 0
    print(f"{'总计':<60} {total_success:>4}/{total_tasks:<5} {overall_rate:>9.1f}%")
    print()

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target_dir = sys.argv[1]
    show_failed = "--show-failed" in sys.argv
    output_json = "--json" in sys.argv

    analyze_directory(target_dir, show_failed=show_failed, output_json=output_json)


if __name__ == "__main__":
    main()
