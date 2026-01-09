#!/usr/bin/env python3
"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class CrossoverOperator(BaseOperator):
    """交叉算子：综合两条轨迹的优点，生成新的初始代码"""

    def get_name(self) -> str:
        return "crossover"

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        # 不直接初始化 input_label，统一用选择方法解析

        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return {"written": 0}
                # 当输入不足时，自适应选择两个源轨迹（不重复）
                chosen = self._select_source_labels(entry, step_config, required_n=2)
                pick1 = chosen[0] if len(chosen) >= 1 else None
                pick2 = chosen[1] if len(chosen) >= 2 else None
                if pick1 and pick2 and pick1 == pick2:
                    # 保障不重复，若重复则尝试再选一个不同的
                    extra = [l for l in self._weighted_select_labels(entry, k=3) if l != pick1]
                    if extra:
                        pick2 = extra[0]
                ref1 = entry.get(pick1) if pick1 else None
                ref2 = entry.get(pick2) if pick2 else None
                if not isinstance(ref1, dict) or not isinstance(ref2, dict):
                    return {
                        "written": 0,
                        "instance_name": instance_name,
                        "source_entry_labels": [str(pick1 or ""), str(pick2 or "")],
                    }
                problem_statement = entry.get("problem")
                summary1 = self._format_entry({str(pick1 or "iter1"): ref1}) if isinstance(ref1, dict) else ""
                summary2 = self._format_entry({str(pick2 or "iter2"): ref2}) if isinstance(ref2, dict) else ""
                if not problem_statement or not summary1 or not summary2:
                    used = [s for s in [pick1, pick2] if isinstance(s, str) and s]
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": [str(x) for x in used]}
                content = self._build_additional_requirements(summary1, summary2)
                if not content:
                    used = [s for s in [pick1, pick2] if isinstance(s, str) and s]
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": [str(x) for x in used]}
                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                used = [s for s in [pick1, pick2] if isinstance(s, str) and s]
                return {"written": 1, "instance_name": instance_name, "source_entry_labels": [str(x) for x in used]}
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

    def _build_additional_requirements(self, trajectory1: str, trajectory2: str) -> str:
        t1 = textwrap.indent(trajectory1.strip(), "  ")
        t2 = textwrap.indent(trajectory2.strip(), "  ")
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("crossover", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("crossover_header")
            or "### STRATEGY MODE: CROSSOVER STRATEGY\nYou are tasked with synthesizing a SUPERIOR hybrid solution by intelligently combining the best elements of two prior optimization trajectories described below."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("crossover_guidelines")
            or (
                """
### SYNTHESIS GUIDELINES
1. **Complementary Combination**: Actively combine specific strengths.
- Example: If T1 has a better Core Algorithm but slow I/O, and T2 has fast I/O but a naive algorithm, implement T1's algorithm using T2's I/O technique.
- Example: If T1 used a correct Stack logic but slow List, and T2 used a fast Array but had logic bugs, implement T1's logic using T2's structure.
2. **Avoid Shared Weaknesses**: If both trajectories failed at a specific sub-task, you must introduce a novel fix for that specific part.
3. **Seamless Integration**: Do not just concatenate code. The resulting logic must be a single, cohesive implementation.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### TRAJECTORY 1 SUMMARY\n" + t1)
        parts.append("\n### TRAJECTORY 2 SUMMARY\n" + t2)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("crossover", CrossoverOperator)
