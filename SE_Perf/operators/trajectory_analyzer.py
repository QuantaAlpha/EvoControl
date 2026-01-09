#!/usr/bin/env python3

"""
Trajectory Analyzer Operator

直接分析轨迹池中的条目，提取问题与轨迹快照，产出 YAML 形式的附加需求。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from operators.base import TemplateOperator


class TrajectoryAnalyzerOperator(TemplateOperator):
    def get_name(self) -> str:
        return "trajectory_analyzer"

    def get_strategy_prefix(self) -> str:
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("trajectory_analyzer", {}) if isinstance(pcfg, dict) else {}
        return str(opcfg.get("prefix") or pcfg.get("trajectory_analyzer_prefix") or "SOLUTION STRATEGY")

    def _build_additional_requirements(self, problem_statement: str, trajectory_snapshot: str) -> str:
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("trajectory_analyzer", {}) if isinstance(pcfg, dict) else {}
        header = str(opcfg.get("header") or pcfg.get("trajectory_analyzer_header") or "SOLUTION STRATEGY")
        prob = textwrap.indent(str(problem_statement).strip(), "  ")
        snap = textwrap.indent(str(trajectory_snapshot).strip(), "  ")
        body = (
            opcfg.get("guidance")
            or pcfg.get("trajectory_analyzer_guidance")
            or (
                "Guidance:\n"
                "1. Begin from an alternative entry point: runtime tracing, I/O profiling, or component isolation.\n"
                "2. Use a non-linear reasoning sequence with hypothesis→micro-test loops.\n"
                "3. Integrate unconventional techniques: targeted benchmarks, memory profiling, fuzzing.\n"
                "4. Prioritize overlooked aspects: performance metrics, boundary conditions, integration constraints.\n"
                "5. Keep changes minimal and testable; modify one module at a time with assertions.\n"
                "6. Explicitly validate assumptions to avoid repeating prior patterns.\n"
            )
        )
        return f"{header}\n\nPROBLEM:\n{prob}\n\nTRAJECTORY SNAPSHOT:\n{snap}\n\n{body}".strip()

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        all_instances = traj_pool_manager.get_all_trajectories()
        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return 0
                problem_statement = entry.get("problem")
                snapshot = self._format_entry(entry)
                if not problem_statement or not snapshot:
                    return 0
                content = self._build_additional_requirements(str(problem_statement), snapshot)
                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                return 1
            except Exception:
                return 0

        num = 1
        try:
            num = int(self.config.get("num_workers", 1))
        except Exception:
            num = 1

        written = 0
        with ThreadPoolExecutor(max_workers=max(1, num)) as ex:
            futures = [ex.submit(_work, (name, entry)) for name, entry in (all_instances or {}).items()]
            for fut in as_completed(futures):
                try:
                    written += int(fut.result() or 0)
                except Exception:
                    pass

        return {"instance_templates_dir": str(output_dir), "generated_count": written}


from .registry import register_operator

register_operator("trajectory_analyzer", TrajectoryAnalyzerOperator)
