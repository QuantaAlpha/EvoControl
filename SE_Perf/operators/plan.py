#!/usr/bin/env python3
"""
Plan Operator (LLM-based)

为每个实例使用LLM规划 K 种不同的实现方案（策略），并将这些方案按标签(sol1..solK)
组织为每次迭代的 system prompt 写入。该算子不直接写文件，而是返回每个标签下、
每个实例的附加需求字典，后续由运行器在各自的 iteration 目录中落盘。

特性：
- 并行按实例生成方案（线程池），参考 crossover 算子的并发实现
- 严格的 JSON 输出格式约束与校验，失败时重试；不足 K 条时使用回退策略补齐
- 清晰的注释与 docstring
"""

import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from operators.base import TemplateOperator

from .registry import register_operator


class PlanOperator(TemplateOperator):
    """LLM方案规划算子：为每个实例生成 K 条多样化策略。

    输出结构：
    {
      "plans": [
        {"label": "sol1", "per_instance_requirements": {"instA": "...", "instB": "..."}},
        {"label": "sol2", "per_instance_requirements": {"instA": "...", "instB": "..."}},
        ...
      ],
      "generated_count": N
    }

    其中 per_instance_requirements 的值将在 perf_run 中被写入各自 iteration 的
    system_prompt/<instance>.yaml。
    """

    def get_name(self) -> str:
        return "plan"

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        """执行算子：并行为每个实例生成 K 条策略，并按标签分组返回。"""

        model_cfg = self.config.get("operator_models", self.config.get("model", {})) or {}
        api_key = str(model_cfg.get("api_key", "")).strip()
        name_ok = bool(str(model_cfg.get("name", "")).strip())
        base_ok = bool(str(model_cfg.get("api_base", "")).strip())
        llm_enabled = name_ok and base_ok and api_key not in ("", "empty", "sk-PLACEHOLDER")

        num = step_config.get("num")
        try:
            num = int(num) if num is not None else 1
        except Exception:
            num = 1

        labels: list[str] = []
        if isinstance(step_config.get("trajectory_labels"), list):
            labels = [str(x) for x in step_config.get("trajectory_labels")]
        if not labels:
            labels = [f"sol{i}" for i in range(1, num + 1)]
        if len(labels) < num:
            labels = labels + [f"sol{i}" for i in range(len(labels) + 1, num + 1)]
        if len(labels) > num:
            labels = labels[:num]

        instances_dir = (self.config.get("instances") or {}).get("instances_dir")
        instances: list[str] = []
        problem_by_instance: dict[str, str] = {}
        if isinstance(instances_dir, str) and instances_dir.strip():
            try:
                inst_root = Path(instances_dir)
                files = list(inst_root.glob("*.json"))
                if files:
                    for f in files:
                        instances.append(f.stem)
                        try:
                            with open(f, encoding="utf-8") as jf:
                                jd = json.load(jf)
                                pd = jd.get("problem_description") or jd.get("description") or ""
                                if isinstance(pd, str) and pd.strip():
                                    problem_by_instance[f.stem] = pd.strip()
                        except Exception:
                            pass
                else:
                    for p in inst_root.iterdir():
                        if p.is_dir():
                            instances.append(p.name)
            except Exception:
                instances = []

        if instances:
            try:
                pool_data = traj_pool_manager.get_all_trajectories() or {}
                for inst in instances:
                    if inst not in problem_by_instance:
                        entry = pool_data.get(inst)
                        if isinstance(entry, dict):
                            prob = entry.get("problem")
                            if isinstance(prob, str) and prob.strip():
                                problem_by_instance[inst] = prob.strip()
            except Exception:
                pass

        def _build_prompts(problem_text: str, k: int) -> tuple[str, str]:
            """
            构建提示词，强调算法多样性、复杂度分析和严格的 JSON 格式。
            """
            pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
            plan_cfg = pcfg.get("plan", {}) if isinstance(pcfg, dict) else {}
            sys_prompt = (
                plan_cfg.get("system_prompt")
                or """You are a world-class Algorithm Engineer and Competitive Programmer. Your task is to design EXACTLY K distinct, high-performance algorithmic strategies for a given problem.

Guidelines:
1. **Diversity**: The strategies MUST differ in algorithmic paradigms (e.g., Dynamic Programming, Greedy, BFS/DFS, Two Pointers, Sliding Window, Bit Manipulation) or Data Structures.
2. **Performance**: Prioritize optimal Time and Space Complexity. Avoid naive brute-force unless unavoidable.
3. **Content**: Each strategy description must be a concise English paragraph including: 
- The core logic/heuristic.
- Key data structures.
- Expected Time Complexity (Big-O) and Space Complexity.
4. **Format**: Return the JSON object wrapped in a Markdown code block with the language tag 'json'.
- Structure:
    ```json
    {"strategies": [string, string, ...]}
    ```
- The array must contain exactly K strings.
- NO conversational text outside the code block."""
            )

            # 在 User Prompt 中再次强调 K，防止模型忽略
            up_template = plan_cfg.get("user_prompt_template") or (
                """
Instruction: Generate {k} diverse high-performance strategies.

Problem Description:
{problem_text}

Required Count: {k}
                """
            )
            user_prompt = up_template.format(k=k, problem_text=problem_text)

            return sys_prompt, user_prompt

        def _extract_json_fragment(text: str) -> str:
            """尽可能提取文本中的 JSON 片段（移除```json ... ``` 等包裹）。"""
            if not isinstance(text, str):
                return ""
            t = text.strip()
            # 去除三引号包裹
            fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
            if fence:
                t = fence.group(1).strip()
            # 截取最外层花括号
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                return t[start : end + 1]
            return t

        def _parse_strategies(message: str, k: int) -> list[str] | None:
            """解析并校验 JSON，确保数组长度为 k。"""
            frag = _extract_json_fragment(message)
            try:
                data = json.loads(frag)
            except Exception:
                return None
            arr = data.get("strategies") if isinstance(data, dict) else None
            if not isinstance(arr, list):
                return None
            vals = [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
            return vals if len(vals) >= k and all(v for v in vals) else None

        def _llm_strategies_with_retry(problem_text: str, k: int, max_attempts: int = 3) -> list[str]:
            """调用LLM并进行格式校验，失败则重试，最终不足则返回部分。"""
            if not llm_enabled:
                return []
            sys_prompt, user_prompt = _build_prompts(problem_text, k)
            last_vals: list[str] = []
            for attempt in range(1, max_attempts + 1):
                try:
                    msg = self._call_llm_api(prompt=user_prompt, system_prompt=sys_prompt)
                    vals = _parse_strategies(msg, k)
                    if vals:
                        return vals
                    # 若解析失败，增强系统提示后重试
                    sys_prompt = (
                        sys_prompt + " Strictly output valid JSON now. Do not include commentary or code fences."
                    )
                except Exception:
                    pass
            return last_vals

        def _fallback(idx: int) -> str:
            plan_cfg = (self.config.get("prompt_config", {}) or {}).get("plan", {})
            patterns = plan_cfg.get(
                "fallback_patterns",
                [
                    "Prefer DP/graph over greedy; restructure loops with memoization.",
                    "Use alternative data structures (heap/deque/ordered-set) to avoid linear scans.",
                    "Improve I/O throughput; batch parsing and reduce conversions.",
                    "Precompute invariants and cache expensive calls to eliminate repeated work.",
                    "Adopt divide-and-conquer or search; structure recursion/iteration for clarity and speed.",
                ],
            )
            try:
                body = str(patterns[(idx - 1) % len(patterns)])
            except Exception:
                body = "Diversify algorithmic approach; improve core performance pragmatically."
            header = f"DIVERSIFIED STRATEGY {idx}"
            return f"{header}\n{body}"

        def _build_additional_requirements(strategy_text: str) -> str:
            """按 crossover 风格包装为 additional_requirements 文本，加入 PLAN STRATEGY 头部。"""
            st = textwrap.indent((strategy_text or "").strip(), "  ")
            plan_cfg = (self.config.get("prompt_config", {}) or {}).get("plan", {})
            header = plan_cfg.get("strategy_header") or (
                """
### STRATEGY MODE: PLAN STRATEGY
You must strictly follow and implement the outlined approach below.
                """
            )
            parts = []
            if isinstance(header, str) and header.strip():
                parts.append(header.strip())
            parts.append("\n" + st)
            return "\n".join(parts)

        # 并发生成：每个实例生成 K 条策略
        def _work(inst_name: str) -> tuple[str, list[str]]:
            problem_text = problem_by_instance.get(inst_name, "")
            if not problem_text:
                # 无问题描述，直接回退
                vals = [_fallback(i) for i in range(1, num + 1)]
                return inst_name, vals
            vals = _llm_strategies_with_retry(problem_text, num) or []
            if len(vals) < num:
                # 使用回退补齐到 K 条
                for i in range(len(vals) + 1, num + 1):
                    vals.append(_fallback(i))
            return inst_name, vals

        cfg_workers = self.config.get("num_workers", 1)
        try:
            max_workers = max(1, int(cfg_workers))
        except Exception:
            max_workers = 1

        per_label_content: dict[str, dict[str, str]] = {str(lb): {} for lb in labels}
        total_generated = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, inst) for inst in instances]
            for fut in as_completed(futures):
                try:
                    inst_name, strategies = fut.result()
                except Exception:
                    inst_name, strategies = (None, None)
                if not inst_name or not isinstance(strategies, list):
                    continue
                for i, lb in enumerate(labels, 1):
                    try:
                        per_label_content[str(lb)][inst_name] = strategies[i - 1]
                        total_generated += 1
                    except Exception:
                        pass

        # 包装为 additional_requirements 形式
        plans: list[dict[str, Any]] = []
        for lb in labels:
            per_inst = per_label_content.get(str(lb), {})
            formatted: dict[str, str] = {}
            for inst_name, txt in (per_inst or {}).items():
                formatted[str(inst_name)] = _build_additional_requirements(str(txt))
            plans.append({"label": str(lb), "per_instance_requirements": formatted})

        return {"plans": plans, "generated_count": total_generated}


register_operator("plan", PlanOperator)
