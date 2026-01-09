#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(_read_text(path) or "{}")
    except Exception:
        return {}


def _parse_runtime_from_result(result_json: dict[str, Any]) -> Any:
    try:
        return result_json.get("final_performance")
    except Exception:
        return None


def _collect_instance_history(root: Path, task_name: str) -> dict[str, Any]:
    instance_root = root / task_name
    data: dict[str, Any] = {"problem": "", "iteration": {}}

    # problem: prefer per-instance .problem, fallback to trajectory pool or instance description in result
    problem_path = instance_root / f"{task_name}.problem"
    if problem_path.exists():
        data["problem"] = _read_text(problem_path)
    else:
        # look for problem in top-level iteration dirs
        for iter_dir in sorted(root.glob("iteration_*")):
            candidate = iter_dir / task_name / f"{task_name}.problem"
            if candidate.exists():
                data["problem"] = _read_text(candidate)
                break

    # iterations: collect pred code and runtime per iteration
    entries: list[tuple[int, dict[str, Any]]] = []
    for iter_dir in root.glob("iteration_*"):
        m = re.search(r"(\d+)$", iter_dir.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        inst_dir = iter_dir / task_name
        pred_file = inst_dir / f"{task_name}.pred"
        traj_file = inst_dir / f"{task_name}.traj"
        result_file = inst_dir / "result.json"

        code = ""
        runtime = None

        if pred_file.exists():
            code = _read_text(pred_file)
            if not code:
                traj_json = _read_json(traj_file)
                info = traj_json.get("info") or {}
                sub = info.get("submission")
                if isinstance(sub, str):
                    code = sub
                elif isinstance(sub, dict):
                    code = sub.get("code", "")
        else:
            traj_json = _read_json(traj_file)
            info = traj_json.get("info") or {}
            sub = info.get("submission")
            if isinstance(sub, str):
                code = sub
            elif isinstance(sub, dict):
                code = sub.get("code", "")

        if result_file.exists():
            runtime = _parse_runtime_from_result(_read_json(result_file))

        entries.append((iter_num, {"code": code, "runtime": runtime}))

    entries.sort(key=lambda x: x[0])
    data["iteration"] = {str(k): v for k, v in entries}

    return data


def collect_folder(folder: str, task_name: str | None = None, output: str | None = None) -> dict[str, Any]:
    root = Path(folder)
    result: dict[str, Any] = {}

    if task_name:
        result[task_name] = _collect_instance_history(root, task_name)
    else:
        # discover tasks by scanning iteration_*/<task_name>/ directories
        tasks: set[str] = set()
        for iter_dir in sorted(root.glob("iteration_*")):
            for child in iter_dir.iterdir():
                if child.is_dir():
                    tasks.add(child.name)
        # also include any direct children that look like instance dirs
        for child in root.iterdir():
            if child.is_dir() and (child / f"{child.name}.traj").exists():
                tasks.add(child.name)
        for t in sorted(tasks):
            result[t] = _collect_instance_history(root, t)

    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def collect_openevolve_archive(folder: str, output: str | None = None) -> dict[str, Any]:
    root = Path(folder)
    result: dict[str, Any] = {}
    for checkpoints_dir in root.rglob("openevolve_output/run_*/checkpoints"):
        task_root = checkpoints_dir.parent.parent.parent
        task_name = task_root.name
        last_checkpoint = None
        max_num = -1
        for cp in checkpoints_dir.iterdir():
            if cp.is_dir():
                m = re.search(r"checkpoint_(\d+)$", cp.name)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                        last_checkpoint = cp
        if not last_checkpoint:
            continue
        programs_dir = last_checkpoint / "programs"
        if not programs_dir.exists():
            continue
        iteration_map: dict[str, dict[str, Any]] = {}
        idx = 0
        for prog_file in sorted(programs_dir.glob("*.json")):
            prog_json = _read_json(prog_file)
            code = prog_json.get("code") or ""
            metrics = prog_json.get("metrics") or {}
            runtime = None
            if isinstance(metrics, dict):
                runtime = metrics.get("trimmed_mean_runtime")
            it_found = prog_json.get("iteration_found")
            key = str(it_found) if isinstance(it_found, int) else None
            if key is None:
                idx += 1
                key = str(idx)
            iteration_map[key] = {"code": code, "runtime": runtime}
        sorted_items = sorted(((int(k), v) for k, v in iteration_map.items()), key=lambda x: x[0])
        data: dict[str, Any] = {"problem": "", "iteration": {str(k): v for k, v in sorted_items}}
        result[task_name] = data
    if output:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect per-iteration history for a task")
    parser.add_argument("folder", help="Root trajectory folder (e.g., SE/trajectories_perf/...)")
    parser.add_argument("task", nargs="?", help="Task name (instance_name). If omitted, collect all tasks")
    parser.add_argument("--output", help="Optional output json path")
    parser.add_argument("--openevolve", action="store_true")
    args = parser.parse_args()
    if args.openevolve:
        collect_openevolve_archive(args.folder, args.output)
    else:
        collect_folder(args.folder, args.task, args.output)


if __name__ == "__main__":
    main()
