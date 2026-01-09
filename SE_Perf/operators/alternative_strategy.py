#!/usr/bin/env python3
"""
Alternative Strategy Operator

基于指定的输入轨迹，生成一个全新的、策略上截然不同的解决方案。
此算子旨在跳出局部最优，从不同维度（例如，算法、数据结构、I/O模式）探索解空间。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class AlternativeStrategyOperator(BaseOperator):
    def get_name(self) -> str:
        return "alternative_strategy"

    """
    替代策略算子：
    根据 step_config 中指定的单个输入轨迹（input），
    生成一个策略迥异的新轨迹（output）。
    """

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return {"written": 0}
                problem_statement = entry.get("problem")
                previous_approach_summary = None
                used_labels: list[str] = []
                # 使用统一方法选择源标签（required_n=1）；若无则回退到全体加权采样
                chosen = self._select_source_labels(entry, step_config, required_n=1)
                if chosen:
                    sub = entry.get(chosen[0])
                    if isinstance(sub, dict):
                        previous_approach_summary = self._format_entry({str(chosen[0]): sub})
                        used_labels = [str(chosen[0])]
                else:
                    src_keys = self._weighted_select_labels(entry, k=1)
                    if src_keys:
                        sub = entry.get(src_keys[0])
                        if isinstance(sub, dict):
                            previous_approach_summary = self._format_entry({str(src_keys[0]): sub})
                            used_labels = [str(src_keys[0])]
                if not previous_approach_summary:
                    previous_approach_summary = self._format_entry(entry)
                if not problem_statement or not previous_approach_summary:
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": used_labels}
                content = self._build_additional_requirements(previous_approach_summary)
                if not content:
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": used_labels}
                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                return {"written": 1, "instance_name": instance_name, "source_entry_labels": used_labels}
            except Exception:
                return {"written": 0}

        num = self.config.get("num_workers", 1)
        try:
            max_workers = max(1, int(num))
        except Exception:
            max_workers = 1

        all_instances = traj_pool_manager.get_all_trajectories()
        written = 0
        per_instance_sources: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, (name, entry)) for name, entry in (all_instances or {}).items()]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if isinstance(res, dict):
                        try:
                            written += int(res.get("written", 0) or 0)
                        except Exception:
                            pass
                        inst = res.get("instance_name")
                        labs = res.get("source_entry_labels")
                        if isinstance(inst, str) and isinstance(labs, list):
                            per_instance_sources[inst] = [str(x) for x in labs]
                    else:
                        written += int(res or 0)
                except Exception:
                    pass

        return {
            "instance_templates_dir": str(output_dir),
            "generated_count": written,
            "source_entry_labels_per_instance": per_instance_sources,
        }

    def _build_additional_requirements(self, previous_approach: str) -> str:
        prev = textwrap.indent(previous_approach.strip(), "  ")
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("alternative_strategy", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("alternative_header")
            or "### STRATEGY MODE: ALTERNATIVE SOLUTION STRATEGY\nYou are explicitly instructed to abandon the current optimization trajectory and implement a FUNDAMENTALLY DIFFERENT approach."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("alternative_guidelines")
            or (
                """
### EXECUTION GUIDELINES
1. **Qualitative Shift**: You must NOT provide incremental refinements, micro-optimizations, or simple bugfixes to the code above.
2. **New Paradigm**: Switch the algorithmic paradigm or data structure entirely (e.g., if Greedy -> try DP; if List -> try Heap/Deque; if Iterative -> try Recursive).
3. **Shift Bottleneck Focus**: If the previous attempt focused heavily on Core Algorithmics, consider an I/O-centric technique (or vice versa).
4. **Target**: Aim for a better Big-O complexity (e.g., O(N) over O(N log N)) where feasible.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### PREVIOUS APPROACH SUMMARY\n" + prev)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("alternative_strategy", AlternativeStrategyOperator)
