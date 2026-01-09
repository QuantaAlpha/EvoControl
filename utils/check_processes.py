#!/usr/bin/env python3
"""
通用进程检查工具
可以检查僵尸进程、孤儿进程、资源使用情况等
支持自定义监控的进程名和阈值

使用方法：
# 检查所有 Python 进程（默认行为）
python utils/check_processes.py

# 检查特定进程
python utils/check_processes.py -p perf_run.py instance_runner.py

# 设置进程数量阈值
python utils/check_processes.py -p my_worker.py --threshold 10

# 调整资源使用阈值
python utils/check_processes.py --cpu-threshold 80 --mem-threshold 20

# 只检查僵尸进程
python utils/check_processes.py --zombies-only

# 显示指定进程的进程树
python utils/check_processes.py -p my_app.py --show-tree
"""

import argparse
import os
import subprocess
from datetime import datetime


# 终端颜色定义
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """禁用颜色（用于非终端输出）"""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ""
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ""


# 如果不是终端，禁用颜色
if not os.isatty(1):
    Colors.disable()


# 进程状态说明
STAT_DESCRIPTIONS = {
    "D": "不可中断睡眠（等待I/O）",
    "I": "空闲内核线程",
    "R": "运行中或可运行",
    "S": "可中断睡眠（等待事件）",
    "T": "被作业控制信号停止",
    "t": "被调试器停止",
    "W": "换页（2.6内核后无效）",
    "X": "已死亡（不应出现）",
    "Z": "僵尸进程",
    "<": "高优先级",
    "N": "低优先级",
    "L": "页面锁定在内存中",
    "s": "会话领导者",
    "l": "多线程",
    "+": "前台进程组",
}


def run_cmd(cmd: list[str]) -> str:
    """执行命令并返回输出"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def format_bytes(size_kb: int) -> str:
    """将 KB 格式化为人类可读的格式"""
    if size_kb < 1024:
        return f"{size_kb} KB"
    elif size_kb < 1024 * 1024:
        return f"{size_kb / 1024:.1f} MB"
    else:
        return f"{size_kb / (1024 * 1024):.2f} GB"


def get_stat_description(stat: str) -> str:
    """获取进程状态的详细描述"""
    main_stat = stat[0] if stat else "?"
    desc = STAT_DESCRIPTIONS.get(main_stat, "未知状态")
    extra = []
    for c in stat[1:]:
        if c in STAT_DESCRIPTIONS:
            extra.append(STAT_DESCRIPTIONS[c])
    if extra:
        desc += " (" + ", ".join(extra) + ")"
    return desc


def get_stat_color(stat: str) -> str:
    """根据进程状态返回对应颜色"""
    if not stat:
        return Colors.RESET
    main_stat = stat[0]
    if main_stat == "Z":
        return Colors.RED
    elif main_stat == "T" or main_stat == "t":
        return Colors.YELLOW
    elif main_stat == "R":
        return Colors.GREEN
    elif main_stat == "D":
        return Colors.YELLOW
    return Colors.RESET


def parse_ps_line(line: str) -> dict | None:
    """解析 ps aux 输出的一行"""
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        return {
            "user": parts[0],
            "pid": int(parts[1]),
            "cpu": float(parts[2]),
            "mem": float(parts[3]),
            "vsz": int(parts[4]),
            "rss": int(parts[5]),
            "tty": parts[6],
            "stat": parts[7],
            "start": parts[8],
            "time": parts[9],
            "cmd": " ".join(parts[10:]),
        }
    except (ValueError, IndexError):
        return None


def get_process_etime(pid: int) -> str:
    """获取进程的运行时间"""
    try:
        result = run_cmd(["ps", "-o", "etime=", "-p", str(pid)])
        return result.strip() if result else "N/A"
    except Exception:
        return "N/A"


def get_process_ppid(pid: int) -> int | None:
    """获取进程的父进程 ID"""
    try:
        result = run_cmd(["ps", "-o", "ppid=", "-p", str(pid)])
        return int(result.strip()) if result else None
    except Exception:
        return None


def print_process_detail(proc: dict, index: int = None, show_full_cmd: bool = False):
    """格式化打印进程详细信息"""
    c = Colors
    stat_color = get_stat_color(proc["stat"])

    # 标题行
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{c.BOLD}{c.CYAN}{prefix}PID: {proc['pid']}{c.RESET}")
    print("─" * 70)

    # 基本信息表格
    print(f"  {'用户:':<12} {c.BLUE}{proc['user']}{c.RESET}")
    print(f"  {'状态:':<12} {stat_color}{proc['stat']}{c.RESET} - {get_stat_description(proc['stat'])}")
    print(f"  {'CPU 使用率:':<12} {c.YELLOW if proc['cpu'] > 50 else ''}{proc['cpu']:.2f}%{c.RESET}")
    print(f"  {'内存使用率:':<12} {c.YELLOW if proc['mem'] > 10 else ''}{proc['mem']:.2f}%{c.RESET}")
    print(f"  {'虚拟内存:':<12} {format_bytes(proc['vsz'])}")
    print(f"  {'物理内存:':<12} {format_bytes(proc['rss'])}")
    print(f"  {'终端:':<12} {proc['tty']}")
    print(f"  {'启动时间:':<12} {proc['start']}")
    print(f"  {'CPU 时间:':<12} {proc['time']}")

    # 运行时长
    etime = get_process_etime(proc["pid"])
    print(f"  {'运行时长:':<12} {etime}")

    # 父进程
    ppid = get_process_ppid(proc["pid"])
    if ppid:
        print(f"  {'父进程 PID:':<12} {ppid}")

    # 命令行
    print(f"  {'命令:':<12}")
    if show_full_cmd or len(proc["cmd"]) <= 100:
        print(f"    {c.DIM}{proc['cmd']}{c.RESET}")
    else:
        print(f"    {c.DIM}{proc['cmd'][:100]}...{c.RESET}")
        print(f"    {c.DIM}(完整命令共 {len(proc['cmd'])} 字符){c.RESET}")


def print_section_header(title: str):
    """打印分节标题"""
    c = Colors
    print(f"\n{c.BOLD}{c.HEADER}{'═' * 70}{c.RESET}")
    print(f"{c.BOLD}{c.HEADER}  {title}{c.RESET}")
    print(f"{c.BOLD}{c.HEADER}{'═' * 70}{c.RESET}")


def print_process_summary_table(procs: list[dict], title: str = "进程列表"):
    """打印进程摘要表格"""
    if not procs:
        return

    c = Colors
    print(f"\n{c.BOLD}  {title} (共 {len(procs)} 个){c.RESET}")
    print("  " + "─" * 66)
    print(
        f"  {c.DIM}{'序号':<4} {'PID':<8} {'状态':<6} {'CPU%':<8} {'内存%':<8} {'物理内存':<10} {'运行时长':<12}{c.RESET}"
    )
    print("  " + "─" * 66)

    for i, p in enumerate(procs, 1):
        stat_color = get_stat_color(p["stat"])
        etime = get_process_etime(p["pid"])
        rss_str = format_bytes(p["rss"])
        print(
            f"  {i:<4} {p['pid']:<8} {stat_color}{p['stat']:<6}{c.RESET} "
            f"{p['cpu']:<8.2f} {p['mem']:<8.2f} {rss_str:<10} {etime:<12}"
        )
    print("  " + "─" * 66)


def check_zombies() -> list[dict]:
    """检查僵尸进程"""
    print_section_header("检查僵尸进程")
    output = run_cmd(["ps", "aux"])
    zombies = []
    for line in output.split("\n"):
        parts = line.split()
        if len(parts) > 7 and parts[7].startswith("Z"):
            proc = parse_ps_line(line)
            if proc:
                zombies.append(proc)

    c = Colors
    if not zombies:
        print(f"\n  {c.GREEN}✓ 未发现僵尸进程{c.RESET}")
    else:
        print(f"\n  {c.RED}⚠️  发现 {len(zombies)} 个僵尸进程！{c.RESET}")
        print_process_summary_table(zombies, "僵尸进程列表")
        # 显示每个僵尸进程的详细信息
        print(f"\n{c.BOLD}  详细信息：{c.RESET}")
        for i, z in enumerate(zombies, 1):
            print_process_detail(z, index=i, show_full_cmd=True)
    return zombies


def check_named_processes(process_names: list[str], threshold: int = None) -> dict[str, list[dict]]:
    """检查指定名称的进程

    Args:
        process_names: 要检查的进程名列表
        threshold: 进程数量阈值，超过则警告

    Returns:
        字典，键为进程名，值为对应的进程列表
    """
    results = {}
    output = run_cmd(["ps", "aux"])
    c = Colors

    for proc_name in process_names:
        print_section_header(f"检查 {proc_name} 进程")
        procs = []
        for line in output.split("\n"):
            if proc_name in line and "grep" not in line and "check_processes" not in line:
                proc = parse_ps_line(line)
                if proc:
                    procs.append(proc)

        results[proc_name] = procs

        if not procs:
            print(f"\n  {c.GREEN}✓ 未发现 {proc_name} 进程{c.RESET}")
        else:
            # 统计信息
            normal_count = sum(1 for p in procs if p["stat"][0] not in ["Z", "T", "t"])
            zombie_count = sum(1 for p in procs if p["stat"][0] == "Z")
            stopped_count = sum(1 for p in procs if p["stat"][0] in ["T", "t"])

            print(f"\n  发现 {c.BOLD}{len(procs)}{c.RESET} 个 {proc_name} 进程")
            print(f"    • 正常运行: {c.GREEN}{normal_count}{c.RESET}")
            if zombie_count > 0:
                print(f"    • 僵尸进程: {c.RED}{zombie_count}{c.RESET}")
            if stopped_count > 0:
                print(f"    • 已停止: {c.YELLOW}{stopped_count}{c.RESET}")

            if threshold and len(procs) > threshold:
                print(f"\n  {c.YELLOW}⚠️  进程数量 ({len(procs)}) 超过阈值 ({threshold}){c.RESET}")

            print_process_summary_table(procs, f"{proc_name} 进程列表")

            # 显示异常进程的详细信息
            abnormal = [p for p in procs if p["stat"][0] in ["Z", "T", "t"]]
            if abnormal:
                print(f"\n{c.BOLD}  异常进程详细信息：{c.RESET}")
                for i, p in enumerate(abnormal, 1):
                    print_process_detail(p, index=i, show_full_cmd=True)

    return results


def check_process_tree(pid: int):
    """检查进程树"""
    c = Colors
    print(f"\n{c.BOLD}{c.CYAN}  进程树 (PID: {pid}){c.RESET}")
    print("  " + "─" * 66)
    try:
        # 尝试使用 pstree
        tree = run_cmd(["pstree", "-p", "-a", str(pid)])
        if tree and "Error" not in tree:
            for line in tree.split("\n"):
                print(f"    {c.DIM}{line}{c.RESET}")
        else:
            # 回退到 ps，获取更详细的子进程信息
            children = run_cmd(["ps", "--ppid", str(pid), "-o", "pid,stat,etime,cmd"])
            if children:
                print(f"  {c.BOLD}子进程：{c.RESET}")
                for line in children.split("\n"):
                    print(f"    {c.DIM}{line}{c.RESET}")
            else:
                print(f"    {c.DIM}无子进程{c.RESET}")
    except Exception as e:
        print(f"    {c.RED}无法获取进程树: {e}{c.RESET}")


def check_long_running(process_keywords: list[str] = None, min_hours: int = 1) -> list[dict]:
    """检查长时间运行的进程

    Args:
        process_keywords: 要过滤的进程关键字列表，None 表示检查所有 Python 进程
        min_hours: 最小运行小时数
    """
    print_section_header("检查长时间运行的进程")
    output = run_cmd(["ps", "-eo", "pid,user,stat,etime,%cpu,%mem,rss,cmd"])
    long_running = []

    for line in output.split("\n"):
        # 根据关键字过滤
        if process_keywords:
            if not any(kw in line for kw in process_keywords):
                continue
        else:
            # 默认只检查 Python 进程
            if "python" not in line.lower():
                continue

        if "grep" in line or "check_processes" in line:
            continue

        parts = line.split()
        if len(parts) >= 4:
            etime = parts[3]  # elapsed time
            # 简单检查：如果包含天数分隔符或格式为 HH:MM:SS（超过1小时），可能是长时间运行
            if "-" in etime or (etime.count(":") >= 2 and not etime.startswith("00:")):
                try:
                    long_running.append(
                        {
                            "pid": int(parts[0]),
                            "user": parts[1],
                            "stat": parts[2],
                            "etime": etime,
                            "cpu": float(parts[4]) if len(parts) > 4 else 0.0,
                            "mem": float(parts[5]) if len(parts) > 5 else 0.0,
                            "rss": int(parts[6]) if len(parts) > 6 else 0,
                            "cmd": " ".join(parts[7:]) if len(parts) > 7 else "",
                        }
                    )
                except (ValueError, IndexError):
                    continue

    c = Colors
    if not long_running:
        print(f"\n  {c.GREEN}✓ 未发现异常长时间运行的进程{c.RESET}")
    else:
        print(f"\n  {c.YELLOW}⚠️  发现 {len(long_running)} 个长时间运行的进程：{c.RESET}")
        print("  " + "─" * 66)
        print(f"  {c.DIM}{'序号':<4} {'PID':<8} {'状态':<6} {'运行时长':<15} {'CPU%':<8} {'内存%':<8}{c.RESET}")
        print("  " + "─" * 66)

        for i, p in enumerate(long_running[:10], 1):
            stat_color = get_stat_color(p["stat"])
            print(
                f"  {i:<4} {p['pid']:<8} {stat_color}{p['stat']:<6}{c.RESET} "
                f"{p['etime']:<15} {p['cpu']:<8.2f} {p['mem']:<8.2f}"
            )
            # 显示命令行（截断）
            cmd_preview = p["cmd"][:70] + "..." if len(p["cmd"]) > 70 else p["cmd"]
            print(f"       {c.DIM}└─ {cmd_preview}{c.RESET}")

        print("  " + "─" * 66)
        if len(long_running) > 10:
            print(f"  {c.DIM}... 还有 {len(long_running) - 10} 个进程未显示{c.RESET}")
    return long_running


def check_resource_usage(
    cpu_threshold: float = 50.0, mem_threshold: float = 10.0, process_filter: str = None
) -> tuple[list[dict], list[dict]]:
    """检查资源使用情况

    Args:
        cpu_threshold: CPU 使用率阈值（百分比）
        mem_threshold: 内存使用率阈值（百分比）
        process_filter: 进程过滤关键字，None 表示检查所有 Python 进程

    Returns:
        (高 CPU 进程列表, 高内存进程列表)
    """
    print_section_header("检查资源使用情况")
    output = run_cmd(["ps", "aux"])
    high_cpu = []
    high_mem = []

    for line in output.split("\n"):
        # 过滤进程
        if process_filter:
            if process_filter not in line:
                continue
        else:
            if "python" not in line.lower():
                continue

        if "grep" in line or "check_processes" in line:
            continue

        proc = parse_ps_line(line)
        if proc:
            if proc["cpu"] > cpu_threshold:
                high_cpu.append(proc)
            if proc["mem"] > mem_threshold:
                high_mem.append(proc)

    c = Colors
    # 高 CPU 使用率进程
    print(f"\n{c.BOLD}  CPU 使用情况（阈值: >{cpu_threshold}%）{c.RESET}")
    print("  " + "─" * 66)
    if high_cpu:
        print(f"  {c.YELLOW}⚠️  发现 {len(high_cpu)} 个高 CPU 使用率的进程：{c.RESET}")
        print(f"  {c.DIM}{'序号':<4} {'PID':<8} {'用户':<10} {'CPU%':<10} {'状态':<6} {'运行时长':<12}{c.RESET}")
        print("  " + "─" * 66)
        for i, p in enumerate(sorted(high_cpu, key=lambda x: x["cpu"], reverse=True)[:10], 1):
            etime = get_process_etime(p["pid"])
            stat_color = get_stat_color(p["stat"])
            print(
                f"  {i:<4} {p['pid']:<8} {p['user']:<10} "
                f"{c.YELLOW}{p['cpu']:<10.2f}{c.RESET} {stat_color}{p['stat']:<6}{c.RESET} {etime:<12}"
            )
            # 显示命令行（截断）
            cmd_preview = p["cmd"][:70] + "..." if len(p["cmd"]) > 70 else p["cmd"]
            print(f"       {c.DIM}└─ {cmd_preview}{c.RESET}")
        print("  " + "─" * 66)
    else:
        print(f"  {c.GREEN}✓ 未发现异常高 CPU 使用的进程{c.RESET}")

    # 高内存使用率进程
    print(f"\n{c.BOLD}  内存使用情况（阈值: >{mem_threshold}%）{c.RESET}")
    print("  " + "─" * 66)
    if high_mem:
        print(f"  {c.YELLOW}⚠️  发现 {len(high_mem)} 个高内存使用率的进程：{c.RESET}")
        print(f"  {c.DIM}{'序号':<4} {'PID':<8} {'用户':<10} {'内存%':<8} {'物理内存':<12} {'状态':<6}{c.RESET}")
        print("  " + "─" * 66)
        for i, p in enumerate(sorted(high_mem, key=lambda x: x["mem"], reverse=True)[:10], 1):
            stat_color = get_stat_color(p["stat"])
            rss_str = format_bytes(p["rss"])
            print(
                f"  {i:<4} {p['pid']:<8} {p['user']:<10} "
                f"{c.YELLOW}{p['mem']:<8.2f}{c.RESET} {rss_str:<12} {stat_color}{p['stat']:<6}{c.RESET}"
            )
            # 显示命令行（截断）
            cmd_preview = p["cmd"][:70] + "..." if len(p["cmd"]) > 70 else p["cmd"]
            print(f"       {c.DIM}└─ {cmd_preview}{c.RESET}")
        print("  " + "─" * 66)
    else:
        print(f"  {c.GREEN}✓ 未发现异常高内存使用的进程{c.RESET}")

    return high_cpu, high_mem


def print_summary(zombies: list, named_procs: dict, long_running: list, process_threshold: int = None):
    """打印检查总结"""
    c = Colors
    print_section_header("检查总结")
    print()

    # 状态统计表格
    print(f"  {c.BOLD}{'检查项':<30} {'数量':<10} {'状态':<10}{c.RESET}")
    print("  " + "─" * 52)

    # 僵尸进程
    zombie_status = f"{c.GREEN}正常{c.RESET}" if len(zombies) == 0 else f"{c.RED}异常{c.RESET}"
    zombie_count_color = c.GREEN if len(zombies) == 0 else c.RED
    print(f"  {'僵尸进程':<28} {zombie_count_color}{len(zombies):<10}{c.RESET} {zombie_status}")

    # 指定进程
    for proc_name, procs in named_procs.items():
        if process_threshold:
            status = f"{c.GREEN}正常{c.RESET}" if len(procs) <= process_threshold else f"{c.YELLOW}超过阈值{c.RESET}"
            count_color = c.GREEN if len(procs) <= process_threshold else c.YELLOW
        else:
            status = f"{c.GREEN}无{c.RESET}" if len(procs) == 0 else f"{c.CYAN}运行中{c.RESET}"
            count_color = c.CYAN
        display_name = proc_name[:26] + ".." if len(proc_name) > 28 else proc_name
        print(f"  {display_name:<28} {count_color}{len(procs):<10}{c.RESET} {status}")

    # 长时间运行进程
    long_status = f"{c.GREEN}正常{c.RESET}" if len(long_running) == 0 else f"{c.YELLOW}需要关注{c.RESET}"
    long_count_color = c.GREEN if len(long_running) == 0 else c.YELLOW
    print(f"  {'长时间运行进程':<28} {long_count_color}{len(long_running):<10}{c.RESET} {long_status}")

    print("  " + "─" * 52)

    # 建议
    if zombies:
        print(f"\n  {c.YELLOW}💡 关于僵尸进程的建议：{c.RESET}")
        print("     1. 等待父进程清理（通常会自动清理）")
        print("     2. 如果父进程已死，僵尸进程会在 init 进程接管后自动清理")
        print("     3. 重启系统可以清理所有僵尸进程（最后手段）")

    for proc_name, procs in named_procs.items():
        if process_threshold and len(procs) > process_threshold:
            print(
                f"\n  {c.YELLOW}⚠️  警告：发现 {len(procs)} 个 {proc_name} 进程，超过阈值（{process_threshold}）{c.RESET}"
            )
            print("     可能存在进程泄漏，建议检查是否有进程卡住或未正确退出")

    if len(long_running) > 0:
        print(f"\n  {c.YELLOW}💡 关于长时间运行进程的建议：{c.RESET}")
        print("     检查这些进程是否正常工作，或者是否已卡住需要手动处理")

    # 最终状态
    has_issues = (
        len(zombies) > 0
        or len(long_running) > 0
        or any(process_threshold and len(procs) > process_threshold for procs in named_procs.values())
    )

    if not has_issues:
        print(f"\n  {c.GREEN}✅ 系统状态良好，未发现明显问题{c.RESET}")
    else:
        print(f"\n  {c.YELLOW}⚠️  发现一些需要关注的问题，请查看上方详情{c.RESET}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="通用进程检查工具 - 检查僵尸进程、资源使用情况等",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 检查所有（不指定进程名，检查所有 Python 进程）
  %(prog)s

  # 检查特定进程
  %(prog)s -p perf_run.py instance_runner.py

  # 设置进程数量阈值
  %(prog)s -p my_worker.py --threshold 10

  # 调整资源使用阈值
  %(prog)s --cpu-threshold 80 --mem-threshold 20

  # 只检查僵尸进程
  %(prog)s --zombies-only

  # 只检查资源使用
  %(prog)s --resources-only

  # 显示指定进程的进程树
  %(prog)s -p my_app.py --show-tree
        """,
    )

    parser.add_argument("-p", "--processes", nargs="+", default=[], help="要监控的进程名列表（支持部分匹配）")
    parser.add_argument("--threshold", type=int, default=None, help="进程数量阈值，超过则警告")
    parser.add_argument("--cpu-threshold", type=float, default=50.0, help="CPU 使用率阈值（百分比，默认 50）")
    parser.add_argument("--mem-threshold", type=float, default=10.0, help="内存使用率阈值（百分比，默认 10）")
    parser.add_argument("--zombies-only", action="store_true", help="只检查僵尸进程")
    parser.add_argument("--resources-only", action="store_true", help="只检查资源使用情况")
    parser.add_argument("--long-running-only", action="store_true", help="只检查长时间运行的进程")
    parser.add_argument("--show-tree", action="store_true", help="显示进程树")
    parser.add_argument("--no-color", action="store_true", help="禁用彩色输出")

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    c = Colors
    print(f"\n{c.BOLD}{c.BLUE}╔══════════════════════════════════════════════════════════════════════╗{c.RESET}")
    print(f"{c.BOLD}{c.BLUE}║              进程状态检查工具                                        ║{c.RESET}")
    print(f"{c.BOLD}{c.BLUE}╚══════════════════════════════════════════════════════════════════════╝{c.RESET}")
    print(f"\n  {c.DIM}检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{c.RESET}")

    zombies = []
    named_procs = {}
    long_running = []

    # 根据参数执行不同的检查
    if args.zombies_only:
        zombies = check_zombies()
    elif args.resources_only:
        check_resource_usage(args.cpu_threshold, args.mem_threshold)
    elif args.long_running_only:
        keywords = args.processes if args.processes else None
        long_running = check_long_running(keywords)
    else:
        # 完整检查
        zombies = check_zombies()

        if args.processes:
            named_procs = check_named_processes(args.processes, args.threshold)

        keywords = args.processes if args.processes else None
        long_running = check_long_running(keywords)

        process_filter = args.processes[0] if args.processes else None
        check_resource_usage(args.cpu_threshold, args.mem_threshold, process_filter)

        # 如果有指定进程且需要显示进程树
        if args.show_tree and named_procs:
            print_section_header("进程树视图")
            for proc_name, procs in named_procs.items():
                for proc in procs:
                    check_process_tree(proc["pid"])

        # 打印总结
        print_summary(zombies, named_procs, long_running, args.threshold)


if __name__ == "__main__":
    main()
