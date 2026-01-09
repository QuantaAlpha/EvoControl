#!/usr/bin/env python3
"""
Reflection and Refine Operator

根据给定的源轨迹（source trajectory）进行反思与改进，生成更优的实现策略要求，
用于在下一次 PerfAgent 迭代中指导代码优化。该算子按实例并行处理，输出为每个实例的
`prompts.additional_requirements` 文本，调用方在对应的 iteration 目录下落盘。
"""

import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class ReflectionRefineOperator(BaseOperator):
    def get_name(self) -> str:
        return "reflection_refine"

    """
    反思与改进算子：
    输入：step_config.inputs 中给定的单个源轨迹标签，如 {"label": "sol1"}
    输出：为每个实例生成带有反思与具体改进指令的 additional_requirements 文本。
    """

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 不直接初始化 input_label，统一用选择方法解析

        def _work(args):
            instance_name, entry = args
            try:
                if not isinstance(entry, dict):
                    return {"written": 0}
                problem_statement = entry.get("problem")
                src_summary = None
                used_labels: list[str] = []
                # 若未提供输入标签，进行线性加权采样选择源轨迹
                chosen = self._select_source_labels(entry, step_config, required_n=1)
                if chosen:
                    sub = entry.get(chosen[0])
                    if isinstance(sub, dict):
                        src_summary = self._format_entry({str(chosen[0]): sub})
                        used_labels = [str(chosen[0])]
                else:
                    keys = self._weighted_select_labels(entry, k=1)
                    if keys:
                        sub = entry.get(keys[0])
                        if isinstance(sub, dict):
                            src_summary = self._format_entry({str(keys[0]): sub})
                            used_labels = [str(keys[0])]
                # 最后回退：使用最新条目摘要
                if not src_summary:
                    src_summary = self._format_entry(entry)

                if not problem_statement or not src_summary:
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": used_labels}

                content = self._build_additional_requirements(src_summary)
                if not content:
                    return {"written": 0, "instance_name": instance_name, "source_entry_labels": used_labels}

                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    import yaml

                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                return {"written": 1, "instance_name": instance_name, "source_entry_labels": used_labels}
            except Exception:
                return {"written": 0}

        # 并发度
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

    def _build_additional_requirements(self, source_summary: str) -> str:
        """
        构造带有反思与改进要求的 additional_requirements 文本。
        """
        src = textwrap.indent((source_summary or "").strip(), "  ")
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("reflection_refine", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("reflection_header")
            or "### STRATEGY MODE: REFLECTION AND REFINE STRATEGY\nYou must explicitly reflect on the previous trajectory and implement concrete improvements."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("reflection_guidelines")
            or (
                """
### REFINEMENT GUIDELINES
1. **Diagnose**: Identify the main shortcomings (correctness risks, bottlenecks, redundant work, I/O overhead).
2. **Fixes**: Propose targeted code-level changes (algorithmic upgrade, data structure replacement, caching/precomputation, I/O batching).
3. **Maintain Correctness**: Prioritize correctness; add guards/tests if necessary before optimizing runtime.
4. **Performance Goal**: Aim for measurable runtime improvement. Prefer asymptotic gains over micro-optimizations.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### SOURCE TRAJECTORY SUMMARY\n" + src)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("reflection_refine", ReflectionRefineOperator)
