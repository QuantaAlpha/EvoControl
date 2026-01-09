#!/usr/bin/env python3

"""
Local Memory Manager

管理短期工作记忆（Local Memory），用于在迭代优化过程中：
- 维护全局状态（当前代数、最佳性能、最佳解ID、当前方法）
- 记录尝试过的高层方向及其成败（direction board）
- 沉淀可迁移的成功/失败经验（reasoning_bank）

该模块参考 reasoningbank 的 Memory 设计思想，提供结构化的 JSON 存储与增量更新，
并在需要时调用 LLM 进行记忆提炼（Extraction）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .llm_client import LLMClient
from .se_logger import get_se_logger


class LocalMemoryManager:
    """
    本地记忆管理器（JSON 后端）

    存储结构（示例）：
    {
    "global_status": {
        "current_generation": 5,
        "current_solution_id": "Gen5_Sol_2",
        "best_solution_id": "Gen3_Sol_4"
    },

    "direction_board": [
        {
        "direction": "Use faster input/output instead of standard C++ streams.",
        "description": "For input-heavy C++ problems, replace cin/cout with faster I/O patterns such as scanf/printf or enabling ios::sync_with_stdio(false) and cin.tie(nullptr). This reduces per-call overhead and improves constant factors when reading or writing large volumes of data.",
        "status": "Success",               // Success | Failed | Neutral | Untried
        "success_count": 2,
        "failure_count": 1,
        "evidence": [
            {
            "solution_id": "Gen5_Sol_2",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%).",
            "code_change": "Replaced cin/cout with scanf/printf for all integer reads.",
            "context": "C++ solution with N up to 2e5 where input reading dominated runtime.",
            "step_outcome": "Success"
            }
        ]
        }
    ],

    "experience_library": [
        {
        "type": "Success",                 // Success | Failure | Neutral
        "title": "Bitwise modulo for power-of-two MOD",
        "description": "When MOD is a power of two, using x & (MOD-1) is faster than x % MOD and is mathematically equivalent.",
        "content": "- Only apply when MOD = 2^k.\n- Replacing division-based modulo with bitwise AND removes expensive division operations in tight loops.\n- This can significantly improve performance in DP transitions or frequency counting loops.\n- Must avoid using this trick when MOD can change or is not guaranteed to be a power of two.",
        "evidence": [
            {
            "solution_id": "Gen5_Sol_2",
            "code_change": "Changed dp[i] % 1024 -> dp[i] & 1023 in the main DP loop.",
            "metrics_delta": "Runtime: 150ms -> 120ms (-20%).",
            "context": "Hot DP loop with fixed MOD=1024, N up to 1e5."
            }
        ]
        }
    ]
    }
    """

    def __init__(
        self,
        memory_path: str | Path,
        llm_client: LLMClient | None = None,
        token_limit: int = 3000,
        format_mode: str = "short",
    ) -> None:
        """
        初始化本地记忆管理器。

        Args:
            memory_path: 记忆库 JSON 文件路径。
            llm_client: 可选的 LLM 客户端，用于进行记忆提炼。
            token_limit: 触发压缩的近似 token/字符阈值。
        """
        self.path = Path(memory_path)
        self.llm_client = llm_client
        self.token_limit = int(token_limit)
        self.logger = get_se_logger("local_memory", emoji="🧠")
        self.format_mode = str(format_mode or "short").lower()

    def _entry_include_keys(self) -> set[str] | None:
        try:
            if str(self.format_mode).lower() == "full":
                return None
        except Exception:
            pass
        return {"code", "perf_metrics"}

    def initialize(self) -> None:
        """确保记忆库文件存在，若不存在则创建空结构。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            empty = {
                "global_status": {
                    "current_generation": 0,
                    "current_solution_id": None,
                    "best_solution_id": None,
                },
                "direction_board": [],
                "experience_library": [],
            }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False, indent=2)
            self.logger.info(f"初始化本地记忆库: {self.path}")

    def load(self) -> dict[str, Any]:
        """加载记忆库 JSON。"""
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"global_status": {}, "direction_board": [], "experience_library": []}
        except Exception as e:
            self.logger.warning(f"加载本地记忆库失败: {e}")
            return {"global_status": {}, "direction_board": [], "experience_library": []}

    def save(self, memory: dict[str, Any]) -> None:
        """保存记忆库 JSON。"""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存本地记忆库失败: {e}")
            raise

    def render_as_markdown(self, memory: dict[str, Any]) -> str:
        """
        将结构化记忆渲染为简洁的 Markdown 文本，便于注入 System Prompt。
        """
        dirs = memory.get("direction_board") or []
        bank = memory.get("experience_library") or []

        lines: list[str] = []

        # 总体说明
        lines.append("## Local Memory (Evolution History)")
        lines.append("")
        lines.append("This is the accumulated knowledge from previous optimization attempts on THIS problem.")
        lines.append("**How to use this memory:**")
        lines.append("1. **Learn from successful patterns**: Apply or Improve insights from Success experiences.")
        lines.append("2. **Avoid repeated failures**: Do NOT retry directions that have failed multiple times.")
        lines.append(
            "3. **Explore new directions**: If existing directions are exhausted, try fundamentally different approaches."
        )
        lines.append("")

        # Tried Directions 部分
        lines.append("### Tried Directions (Strategy Board)")
        lines.append("")
        lines.append("These are high-level optimization strategies that have been attempted.")
        lines.append("- **[Success]**: This direction worked well. Consider building upon it.")
        lines.append("- **[Failed]**: This direction did NOT work. Do NOT retry the same approach.")
        lines.append("- **[Neutral]**: No effect; may be worth exploring with modifications.")
        lines.append("- **(✓N ✗M)**: N successful attempts, M failed attempts.")
        lines.append("")

        if dirs:
            for d in dirs:
                status = d.get("status", "Unknown")
                succ = d.get("success_count", 0)
                fail = d.get("failure_count", 0)
                lines.append(f"- [{status}] {d.get('direction', '')} (✓{succ} ✗{fail}) — {d.get('description', '')}")
        else:
            lines.append("- (No directions recorded yet)")
        lines.append("")

        # Learned Patterns 部分
        lines.append("### Learned Patterns (Experience Library)")
        lines.append("")
        lines.append("These are specific insights extracted from successful/failed attempts.")
        lines.append("- **✅ Apply**: Proven techniques that improve performance. Use or Improve Them!")
        lines.append("- **⚠️ Avoid**: Anti-patterns that caused failures. Do NOT repeat these mistakes!")
        lines.append("")

        if bank:
            for item in bank:
                item_type = str(item.get("type", "")).strip()
                title = str(item.get("title", "")).strip()
                description = str(item.get("description", "")).strip()
                content = item.get("content", "")

                # 根据类型格式化标题前缀，使成功/失败更清晰
                if item_type.lower() == "failure":
                    prefix = "⚠️ Avoid"
                    type_label = "Anti-pattern"
                elif item_type.lower() == "success":
                    prefix = "✅ Apply"
                    type_label = "Better Practice"
                else:
                    prefix = "📝"
                    type_label = "Observation"

                lines.append(f"#### {prefix}: {title}")
                lines.append(f"- ({type_label}) {description}")
                lines.append(f"- Detail: {content}")
                lines.append("")
        else:
            lines.append("- (No patterns learned yet)")

        return "\n".join(lines)

    def _estimate_chars(self, memory: dict[str, Any]) -> int:
        """粗略估计记忆体量（按字符计）。"""
        try:
            return len(self.render_as_markdown(memory))
        except Exception:
            return 0

    def _format_metrics_delta(self, perf_old: float | None, perf_new: float | None) -> str:
        """将性能变化格式化为易读字符串。"""
        try:
            if perf_old is None or perf_new is None:
                return "N/A"
            if math.isinf(perf_old) and not math.isinf(perf_new):
                return f"Runtime: inf -> {perf_new}"
            if math.isinf(perf_new):
                return f"Runtime: {perf_old} -> inf"
            delta = perf_new - perf_old
            pct = (delta / perf_old * 100.0) if perf_old and not math.isinf(perf_old) else None
            if pct is None:
                return f"Runtime: {perf_old} -> {perf_new}"
            sign = "+" if pct >= 0 else ""
            return f"Runtime: {perf_old} -> {perf_new} ({sign}{pct:.1f}%)"
        except Exception:
            return "N/A"

    def _build_extraction_prompts(
        self,
        problem_description: str | None,
        perf_old: float | None,
        perf_new: float | None,
        source_entries: list[dict[str, Any]] | None,
        current_entry: dict[str, Any] | None,
        best_entry: dict[str, Any] | None,
        current_directions: list[dict[str, Any]],
        language: str = "",
        optimization_target: str = "",
        current_solution_id: str | None = None,
    ) -> tuple[str, str]:
        """
        构造记忆提炼的 System/User 提示词。
        根据性能变化分流进入 Success 或 Failure 分支。

        对于初始解（无 perf_old）：
        - 如果 perf_new 不为 inf，视为 Success（基线建立成功）
        - 如果 perf_new 为 inf，视为 Failure（基线建立失败）
        """
        # 1. Metric Analysis
        perf_diff = 0.0

        if perf_old is not None and perf_new is not None:
            # Handle inf
            if math.isinf(perf_old) and not math.isinf(perf_new):
                perf_diff = float("inf")  # Improvement
            elif not math.isinf(perf_old) and math.isinf(perf_new):
                perf_diff = float("-inf")  # Regression
            elif math.isinf(perf_old) and math.isinf(perf_new):
                perf_diff = 0.0
            else:
                perf_diff = perf_old - perf_new
        elif perf_new is not None:
            # 初始解：根据 perf_new 是否为 inf 判断成功/失败
            if not math.isinf(perf_new):
                # 初始解成功（有有效性能数据），视为正向
                perf_diff = float("inf")  # 作为 Success 处理
            else:
                # 初始解失败（性能为 inf），视为负向
                perf_diff = float("-inf")  # 作为 Failure 处理

        # 2. Extraction Branch - 统一使用 Success/Failure 分支
        # 不再单独调用 _build_initial_prompt，初始解根据 perf_diff 归入相应分支
        if perf_diff > 0:
            return self._build_success_prompt(
                problem_description,
                perf_old,
                perf_new,
                perf_diff,
                source_entries,
                current_entry,
                best_entry,
                current_directions,
                language,
                optimization_target,
                current_solution_id,
            )
        else:
            return self._build_failure_prompt(
                problem_description,
                perf_old,
                perf_new,
                perf_diff,
                source_entries,
                current_entry,
                best_entry,
                current_directions,
                language,
                optimization_target,
                current_solution_id,
            )

    def _build_success_prompt(
        self,
        problem,
        perf_old,
        perf_new,
        perf_diff,
        source_entries,
        current_entry,
        best_entry,
        directions,
        language,
        target,
        current_solution_id,
    ) -> tuple[str, str]:
        # 1. System Prompt
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **attempted to optimize** a code solution and the **metrics show an improvement** (or at least not a clear regression).

Your job is NOT to log every tiny change. Your job is to maintain:
- a **high-level strategy board** (`direction_board`), and
- an **experience library** (`experience_library`)
that together guide future evolution.

---

## Goal

Given the previous and current solutions, you must:

1. Decide whether this step is truly a **Success**, or actually **Neutral** (e.g., noise, trivial refactor).
2. If (and only if) there are **strategy-level changes**, extract up to 3 new:
   - **Direction items**: reusable optimization strategies that can be tried again on other solutions.
   - **Memory items**: distilled reasoning patterns that explain *why* certain strategies work.

This memory is local to a single problem and will be shown to the model in later steps to encourage **diverse strategy exploration**, not to duplicate the same ideas.

---

## Definitions

- **Strategy-level change**:
  - Switching algorithms (e.g., brute force → two-pointer, BFS → Dijkstra, naive DP → optimized DP).
  - Changing core data structures (e.g., vector → bitset, list → array, unordered_map → array-based counter).
  - Applying a clear performance trick (e.g., fast I/O, precomputation, caching, reducing passes over the array).
  - Changing memory layout or loop structure in a way that affects asymptotics or constant factors in a hotspot.

- **Non-strategy changes (DO NOT create directions for these)**:
  - Renaming variables, reformatting, reordering independent statements.
  - Small cosmetic refactors that do not change complexity or memory access patterns.
  - Pure measurement noise: identical code with slightly different runtimes.

---

## Very Important Rules

1. **You may return ZERO new directions and ZERO new memories.**
   - This is the correct behavior when no strategy-level change happened.

2. **Do NOT create directions about measurement noise or “no change”.**
   - The following are explicitly forbidden as directions:
     - "No Change, OS Jitter"
     - "Measurement noise"
     - "Same code as previous solution"

3. **Noise vs Success vs Neutral**:
   - If the improvement is within typical measurement jitter, and there is *no* meaningful strategy change, treat the step as **Neutral**.
   - Only mark `"step_outcome": "Success"` when:
     - There is a real metric improvement **and**
     - You can tie it to a strategy-level code change.

4. **Rich, semantic content**:
   - `direction` should look like a clear strategy name that could appear on a "strategy board".
   - `description` should be 1–3 sentences explaining:
     - what the strategy does,
     - when to use it,
     - and potential trade-offs or risks.
   - For **Success** memory items:
     - `title`: Describe the successful technique (e.g., "Use rolling array DP for space optimization")
     - `content`: Explain **WHY** it works and **WHAT insight** makes it effective. Focus on the key reasoning.

5. **Cardinality constraints**:
   - At most 3 `new_direction_items`.
   - At most 3 `new_memory_items`.
   - Arrays can be empty (`[]`).

---

## Input Data Provided

You will be given:

1. **Optimization Target**: e.g., runtime, memory, integral.
2. **Language**: e.g., C++, Python.
3. **Problem Description**: The algorithmic problem being solved.
4. **Source Solutions**: Parent code(s), summaries and metrics before mutation.
5. **Current Solution**: Mutated code, summary and metrics after mutation.
6. **Best Solution**: The global best solution so far (for context).
7. **Current Directions**: The current snapshot of the strategy board for this problem.

Use the diffs between Source and Current solution to reason about what changed.

---

## Output Format

You must output a single JSON object **strictly** adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here.",

  "new_direction_items": [
    {
      "direction": "High-level strategy name.",
      "description": "1–3 sentences describing what was changed, why it is a reusable strategy, and when it applies.",
      "status": "Success | Neutral",
    }
  ],

  "new_memory_items": [
    {
      "type": "Success | Neutral",
      "title": "Concise title of the reasoning pattern.",
      "description": "One-sentence summary of the insight.",
      "content": "2–6 sentences explaining when to apply this, why it works, and any risks.",
    }
  ]
}
```

Notes:
- If there is no meaningful strategy-level change, set "step_outcome": "Neutral" and both arrays to [].
- Do not invent fake strategies just to fill the JSON.
        """
        # 判断是否是初始化场景（无 source entries）
        is_initial = not source_entries

        if is_initial:
            # 初始化场景：识别基线策略
            user_template = """
## Mode: BASELINE INITIALIZATION

This is the **initial solution** (baseline). There is no previous version to compare against.
Your task is to **identify the core algorithmic strategy** used in the Current Solution and record it as the baseline.

## Guidelines for Baseline Extraction

1. **Identify Strategy**: Analyze the code. What is the core algorithmic paradigm? (e.g., Dynamic Programming, Greedy, BFS, Binary Search, Simulation, or naive Brute Force).
2. **Establish Baseline**: Create a direction item describing this fundamental approach with status "Baseline" or "Success".
3. **No Comparison Needed**: Since there's no source to compare, focus on identifying WHAT strategy the code uses, not HOW it changed.

## Optimization Target

{optimization_target}

## Language

{language}

## Problem Description

{problem_description}

## Current Solution (Baseline)

{current_solution}

## Best Solution

{best_solution}

## Current Directions (Strategy Board Snapshot)

{directions}
        """
        else:
            # 变异场景：比较 source 和 current
            user_template = """
## Optimization Target

{optimization_target}

The optimization target is **integral**:  
- Interpret this as the **integral of memory usage over runtime** for all test cases, i.e., the **area under the memory–time curve**.
- Your performance judgments should consider **both** runtime and memory, focusing on how each slot affects this **memory–time integral**, not just speed or memory in isolation.
- A slot that is slightly slower but uses much less memory can be better if it reduces the overall integral, and vice versa.

## Language

{language}

## Problem Description

{problem_description}

## Source Solutions

{source_solutions}

## Current Solution

{current_solution}

## Best Solution

{best_solution}

## Current Directions (Strategy Board Snapshot)

{directions}
        """
        # Build formatted texts using TrajPoolManager.format_entry
        try:
            from .traj_pool_manager import TrajPoolManager
        except Exception:
            TrajPoolManager = None  # type: ignore

        def _fmt_entry_text(entry: dict | None) -> str:
            try:
                if TrajPoolManager and isinstance(entry, dict):
                    lbl = str(entry.get("label") or entry.get("solution_id") or "current")
                    return TrajPoolManager.format_entry({lbl: entry}, include_keys=self._entry_include_keys())
            except Exception:
                pass
            return "N/A"

        def _fmt_entries_text(entries: list[dict] | None) -> str:
            if not entries:
                return "N/A"
            texts: list[str] = []
            for e in entries:
                t = _fmt_entry_text(e)
                if t and t != "N/A":
                    texts.append(t)
            return "\n\n".join(texts) if texts else "N/A"

        source_solutions_text = _fmt_entries_text(source_entries)
        current_solution_text = _fmt_entry_text(current_entry)
        best_solution_text = _fmt_entry_text(best_entry)

        # 根据是否是初始化场景选择格式化参数
        format_kwargs = {
            "optimization_target": str(target or "Runtime"),
            "language": str(language or "Unknown"),
            "problem_description": str(problem or "N/A"),
            "current_solution": current_solution_text,
            "best_solution": best_solution_text,
            "directions": json.dumps(directions or [], ensure_ascii=False),
        }
        if not is_initial:
            format_kwargs["source_solutions"] = source_solutions_text

        user_prompt = user_template.format(**format_kwargs)

        return system_prompt, user_prompt

    def _build_failure_prompt(
        self,
        problem,
        perf_old,
        perf_new,
        perf_diff,
        source_entries,
        current_entry,
        best_entry,
        directions,
        language,
        target,
        current_solution_id,
    ) -> tuple[str, str]:
        # 1. System Prompt
        system_prompt = """You are an expert Algorithm Optimization Specialist. You have just observed an evolutionary step where an agent **attempted to optimize** a code solution and the **metrics show a regression or incorrectness**.

Your job is NOT to log every tiny change. Your job is to maintain:
- a **high-level strategy board** (`direction_board`), and
- an **experience library** (`experience_library`)
that warn future steps about bad ideas.

---

## Goal

Given the previous and current solutions, you must:

1. Decide whether this step is truly a **Failure**, or actually **Neutral** (e.g., noise, trivial refactor).
2. If (and only if) there are **strategy-level changes that caused the regression**, extract up to 3 new:
   - **Direction items**: strategies that should be marked as Failed or risky in the current context.
   - **Memory items**: warnings or anti-patterns explaining *why* this approach failed and when to avoid it.

---

## Definitions

- **Strategy-level change**:
  - Same as in the Success case: algorithm switch, data structure switch, clear performance trick, major loop or memory layout change.
- **Non-strategy changes (DO NOT create directions for these)**:
  - Formatting, renaming, minor refactors with no impact on complexity or memory access.
  - Pure measurement noise with identical code.

---

## Very Important Rules

1. **You may return ZERO new directions and ZERO new memories.**
   - This is the correct behavior when no strategy-level change caused the regression.

2. **Do NOT create directions about measurement noise or “no change”.**
   - Explicitly forbidden directions:
     - "No Change, OS Jitter"
     - "Measurement noise"
     - "Same code as previous solution"

3. **Noise vs Failure vs Neutral**:
   - If the regression is typical measurement jitter, and there is *no* meaningful strategy change, treat the step as **Neutral**.
   - Only mark `"step_outcome": "Failure"` when:
     - Runtime, memory, or correctness clearly got worse **and**
     - You can tie it to a strategy-level change (e.g., added redundant checks, switched to a slower algorithm, broke edge cases).

4. **Rich, semantic content**:
   - Directions should describe *what strategy went wrong* (e.g., "aggressive pruning without correctness proof", "using recursion with unbounded depth").
   - For **Failure** memory items:
     - `title`: Describe the **SPECIFIC mistake**, NOT just the approach name.
       - BAD: "BFS implementation" (too vague)
       - GOOD: "BFS without boundary check causes index out of bounds"
       - GOOD: "Recursive factorial without memoization causes TLE for large N"
     - `content`: Explain **WHY** it failed and **WHAT specific condition** triggered the failure.

5. **Cardinality constraints**:
   - At most 3 `new_direction_items`.
   - At most 3 `new_memory_items`.
   - Arrays can be empty (`[]`).

---

## Input Data Provided

Same as in the Success case:

1. **Optimization Target**
2. **Language**
3. **Problem Description**
4. **Source Solutions**
5. **Current Solution**
6. **Best Solution**
7. **Current Directions**

---

## Output Format

You must output a single JSON object **strictly** adhering to this schema:

```json
{
  "thought_process": "Briefly explain your reasoning here (max 2 sentences).",
  "step_outcome": "Failure | Neutral",

  "new_direction_items": [
    {
      "direction": "High-level description of the failed strategy.",
      "description": "1–3 sentences explaining what the strategy tried to do and why it is problematic in this context.",
      "status": "Failed | Neutral",

  "new_memory_items": [
    {
      "type": "Failure | Neutral",
      "title": "Start with 'Avoid ...' for Failure type (e.g., 'Avoid recursive solution without memoization').",
      "description": "One-sentence summary of why this approach is dangerous and should be avoided.",
      "content": "2–6 sentences explaining what went wrong, under what conditions it fails, and how to avoid it.",

}
```

Notes:
- If there is no meaningful strategy-level change, set "step_outcome": "Neutral" and both arrays to [].
- Do not mark previously successful strategies as failed just because one noisy run was slower.
        """
        # 判断是否是初始化场景（无 source entries）
        is_initial = not source_entries

        if is_initial:
            # 初始化失败场景：初始解就失败了（TLE/OOM/WA）
            user_template = """
## Mode: BASELINE INITIALIZATION FAILED

This is the **initial solution** (baseline), but it **failed** (TLE, OOM, WA, or other errors).
There is no previous version to compare against.

Your task is to:
1. **Identify what strategy the code attempted** (e.g., naive brute force, unoptimized DP, etc.)
2. **Record why it failed** as a warning for future iterations

## Guidelines for Failed Baseline

- Create a direction item with status "Failed" describing the attempted approach
- Create a memory item explaining why this approach doesn't work for this problem
- Focus on identifying the **root cause** of failure (time complexity too high? memory usage too large? edge case bug?)

## Optimization Target

{optimization_target}

## Language

{language}

## Problem Description

{problem_description}

## Current Solution (Failed Baseline)

{current_solution}

## Best Solution

{best_solution}

## Current Directions (Strategy Board Snapshot)

{directions}
        """
        else:
            # 变异失败场景：比较 source 和 current
            user_template = """    
## Optimization Target

{optimization_target}

The optimization target is **integral**:  
- Interpret this as the **integral of memory usage over runtime** for all test cases, i.e., the **area under the memory–time curve**.
- Your performance judgments should consider **both** runtime and memory, focusing on how each slot affects this **memory–time integral**, not just speed or memory in isolation.
- A slot that is slightly slower but uses much less memory can be better if it reduces the overall integral, and vice versa.


## Language

{language}

## Problem Description

{problem_description}

## Source Solutions

{source_solutions}

## Current Solution

{current_solution}

## Best Solution

{best_solution}

## Current Directions (Strategy Board Snapshot)

{directions}
        """
        # Build formatted texts using TrajPoolManager.format_entry
        try:
            from .traj_pool_manager import TrajPoolManager
        except Exception:
            TrajPoolManager = None  # type: ignore

        def _fmt_entry_text(entry: dict | None) -> str:
            try:
                if TrajPoolManager and isinstance(entry, dict):
                    lbl = str(entry.get("label") or entry.get("solution_id") or "current")
                    return TrajPoolManager.format_entry({lbl: entry}, include_keys=self._entry_include_keys())
            except Exception:
                pass
            return "N/A"

        def _fmt_entries_text(entries: list[dict] | None) -> str:
            if not entries:
                return "N/A"
            texts: list[str] = []
            for e in entries:
                t = _fmt_entry_text(e)
                if t and t != "N/A":
                    texts.append(t)
            return "\n\n".join(texts) if texts else "N/A"

        source_solutions_text = _fmt_entries_text(source_entries)
        current_solution_text = _fmt_entry_text(current_entry)
        best_solution_text = _fmt_entry_text(best_entry)

        # 根据是否是初始化场景选择格式化参数
        format_kwargs = {
            "optimization_target": str(target or "Runtime"),
            "language": str(language or "Unknown"),
            "problem_description": str(problem or "N/A"),
            "current_solution": current_solution_text,
            "best_solution": best_solution_text,
            "directions": json.dumps(directions or [], ensure_ascii=False),
        }
        if not is_initial:
            format_kwargs["source_solutions"] = source_solutions_text

        user_prompt = user_template.format(**format_kwargs)

        return system_prompt, user_prompt

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        """提取并解析 LLM 返回的 JSON 内容。"""
        content = (text or "").strip()
        if not content:
            msg = "空响应内容，无法解析为JSON"
            raise ValueError(msg)

        # 尝试直接解析完整JSON
        if content.startswith("{"):
            return json.loads(content)

        # 尝试提取JSON片段进行解析
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)

        # 未找到可解析的JSON片段
        msg = "响应中未找到可解析的JSON内容"
        raise ValueError(msg)

    def _validate_memory_response(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            msg = "响应数据必须为JSON对象"
            raise ValueError(msg)
        # 仅支持数组形式返回
        if "new_direction_items" not in data:
            msg = "响应格式缺少键: new_direction_items"
            raise ValueError(msg)
        nd = data.get("new_direction_items")
        if nd is not None and not isinstance(nd, list):
            msg = "new_direction_items必须为列表"
            raise ValueError(msg)
        if isinstance(nd, list):
            for it in nd:
                if not isinstance(it, dict):
                    msg = "new_direction_items的元素必须为对象"
                    raise ValueError(msg)

        if "new_memory_items" in data:
            nm = data.get("new_memory_items")
            if nm is not None and not isinstance(nm, list):
                msg = "new_memory_items必须为列表"
                raise ValueError(msg)
            if isinstance(nm, list):
                for it in nm:
                    if not isinstance(it, dict):
                        msg = "new_memory_items的元素必须为对象"
                        raise ValueError(msg)

    def _normalize_extraction_response(self, resp: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将LLM响应统一转换为列表形式。"""
        dirs: list[dict[str, Any]] = []
        mems: list[dict[str, Any]] = []
        try:
            single_dir = resp.get("new_direction_item")
            if isinstance(single_dir, dict):
                dirs.append(single_dir)
            multi_dir = resp.get("new_direction_items")
            if isinstance(multi_dir, list):
                dirs.extend([d for d in multi_dir if isinstance(d, dict)])
        except Exception:
            pass
        try:
            single_mem = resp.get("new_memory_item")
            if isinstance(single_mem, dict):
                mems.append(single_mem)
            multi_mem = resp.get("new_memory_items")
            if isinstance(multi_mem, list):
                mems.extend([m for m in multi_mem if isinstance(m, dict)])
        except Exception:
            pass
        return dirs, mems

    def _merge_direction_board(self, memory: dict[str, Any], new_items: list[dict[str, Any]]) -> None:
        """将提炼的方向项直接插入 direction_board。"""
        board: list[dict[str, Any]] = memory.get("direction_board") or []
        for raw in new_items:
            if not isinstance(raw, dict):
                continue
            direction = str(raw.get("direction") or "").strip()
            if not direction:
                continue
            description = str(raw.get("description") or "").strip()
            status = str(raw.get("status") or "Neutral").strip()
            evidence_src = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            evidence = [e for e in evidence_src if isinstance(e, dict)]

            # 根据 status 初始化计数（如果 LLM 没有返回计数）
            raw_success = raw.get("success_count")
            raw_failure = raw.get("failure_count")
            if raw_success is not None:
                success_count = int(raw_success)
            elif status.lower() in ("success", "baseline"):
                success_count = 1
            else:
                success_count = 0

            if raw_failure is not None:
                failure_count = int(raw_failure)
            elif status.lower() == "failed":
                failure_count = 1
            else:
                failure_count = 0

            board.append(
                {
                    "direction": direction,
                    "description": description,
                    "status": status,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "evidence": evidence,
                }
            )
        memory["direction_board"] = board

    def _merge_experience_library(self, memory: dict[str, Any], new_items: list[dict[str, Any]]) -> None:
        """将提炼的经验项直接插入 experience_library。"""
        library: list[dict[str, Any]] = memory.get("experience_library") or []
        for raw in new_items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            typ = str(raw.get("type") or "Neutral").strip()
            description = str(raw.get("description") or "").strip()
            content = raw.get("content")
            evidence_src = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            evidence = [e for e in evidence_src if isinstance(e, dict)]

            library.append(
                {
                    "type": typ,
                    "title": title,
                    "description": description,
                    "content": content,
                    "evidence": evidence,
                }
            )
        memory["experience_library"] = library

    def compress_if_needed(self, memory: dict[str, Any]) -> None:
        """
        如果记忆体量超过阈值，分别压缩 direction_board 和 experience_library。
        """
        try:
            if self._estimate_chars(memory) <= self.token_limit:
                return
            if not self.llm_client:
                self.logger.warning("LLM不可用，跳过记忆压缩")
                return

            # 分别压缩 direction_board 和 experience_library
            self._compress_direction_board(memory)
            self._compress_experience_library(memory)

            self.logger.info("LLM记忆压缩完成")
        except Exception as e:
            self.logger.warning(f"压缩记忆失败: {e}")

    def _compress_direction_board(self, memory: dict[str, Any]) -> None:
        """压缩 direction_board。"""
        direction_board = memory.get("direction_board") or []
        if len(direction_board) <= 3:
            return  # 太少，不需要压缩

        sys_prompt, user_prompt = self._build_compress_direction_board_prompts(direction_board)
        last_error: str | None = None

        for attempt in range(1, 4):
            try:
                llm_response = self.llm_client.call_with_system_prompt(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=None,  # 使用配置中的 max_output_tokens
                    usage_context="memory.compress_directions",
                )
                self.logger.debug(f"LLM原始响应 (压缩directions，第{attempt}次):\n{llm_response}")
                llm_response = self.llm_client.clean_think_tags(llm_response)
                parsed = self._parse_llm_json(llm_response)

                db = parsed.get("direction_board")
                if isinstance(db, list):
                    memory["direction_board"] = db
                    self.logger.info(f"direction_board 压缩成功: {len(direction_board)} -> {len(db)} 条")
                    return
            except ValueError as e:
                last_error = "invalid_response_format"
                self.logger.warning(f"direction_board 压缩解析失败 (第{attempt}次): {e}")
            except Exception as e:
                last_error = "llm_call_failed"
                self.logger.warning(f"direction_board 压缩调用失败 (第{attempt}次): {e}")

        if last_error:
            self.logger.error(f"direction_board 压缩最终失败: {last_error}")

    def _compress_experience_library(self, memory: dict[str, Any]) -> None:
        """压缩 experience_library。"""
        experience_library = memory.get("experience_library") or []
        if len(experience_library) <= 3:
            return  # 太少，不需要压缩

        sys_prompt, user_prompt = self._build_compress_experience_library_prompts(experience_library)
        last_error: str | None = None

        for attempt in range(1, 4):
            try:
                llm_response = self.llm_client.call_with_system_prompt(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=None,  # 使用配置中的 max_output_tokens
                    usage_context="memory.compress_experiences",
                )
                self.logger.debug(f"LLM原始响应 (压缩experiences，第{attempt}次):\n{llm_response}")
                llm_response = self.llm_client.clean_think_tags(llm_response)
                parsed = self._parse_llm_json(llm_response)

                el = parsed.get("experience_library")
                if isinstance(el, list):
                    memory["experience_library"] = el
                    self.logger.info(f"experience_library 压缩成功: {len(experience_library)} -> {len(el)} 条")
                    return
            except ValueError as e:
                last_error = "invalid_response_format"
                self.logger.warning(f"experience_library 压缩解析失败 (第{attempt}次): {e}")
            except Exception as e:
                last_error = "llm_call_failed"
                self.logger.warning(f"experience_library 压缩调用失败 (第{attempt}次): {e}")

        if last_error:
            self.logger.error(f"experience_library 压缩最终失败: {last_error}")

    def _build_compress_direction_board_prompts(self, direction_board: list[dict[str, Any]]) -> tuple[str, str]:
        """构建压缩 direction_board 的 prompt。"""
        system_prompt = """You are compressing the **direction_board** (Strategy Board) of an evolutionary coding agent.

## Task
Consolidate and compress the list of tried strategies while preserving useful information.

## Rules

1. **Merge semantically similar strategies**
   - If multiple entries describe the same idea (e.g., "Use fast I/O", "Replace cin/cout with scanf"), merge them.
   - Rewrite as a clear, unique strategy name.

2. **IMPORTANT: Do NOT merge strategies with DIFFERENT failure modes**
   - "Precompute factorials caused OOM" and "Iterative computation caused TLE" are DIFFERENT, keep them separate.
   - "Edge case N=0 failed" and "Large N caused overflow" are DIFFERENT, keep them separate.

3. **Aggregate counts when merging**
   - When merging similar strategies, SUM their success_count and failure_count.
   - Update status based on aggregated counts:
     - "Success" if success_count > failure_count.
     - "Failed" if failure_count > success_count.
     - "Neutral" if counts are equal or evidence is weak.

4. **Prune low-value directions**
   - Remove vague entries (e.g., "optimize code a bit").
   - Remove noise entries (e.g., "OS jitter", "no code change").
   - Keep roughly **5–10** useful directions.

## Output Format

```json
{
  "thought_process": "Brief explanation.",
  "direction_board": [
    {
      "direction": "Strategy name",
      "description": "1–3 sentences explaining the strategy.",
      "status": "Success | Failed | Neutral",
      "success_count": int,
      "failure_count": int
    }
  ]
}
```
"""
        user_prompt = f"""## Current Direction Board

{json.dumps(direction_board, indent=2, ensure_ascii=False)}

## Task
Compress and consolidate the direction_board above. Output ONLY the valid JSON object.
"""
        return system_prompt, user_prompt

    def _build_compress_experience_library_prompts(self, experience_library: list[dict[str, Any]]) -> tuple[str, str]:
        """构建压缩 experience_library 的 prompt。"""
        system_prompt = """You are compressing the **experience_library** of an evolutionary coding agent.

## Task
Consolidate and compress the list of learned experiences while preserving actionable insights.

## Rules

1. **Merge overlapping experiences**
   - If multiple entries describe the same lesson, merge them into one stronger experience.

2. **IMPORTANT: Do NOT merge experiences with DIFFERENT root causes**
   - "Avoid recursion without memoization (TLE)" and "Avoid large array allocation (OOM)" are DIFFERENT lessons, keep them separate.
   - "Use iterative DP" and "Use rolling array to save memory" are DIFFERENT techniques, keep them separate.

3. **Content Guidelines by Type**

   **For Success type:**
   - Title: Describe the successful technique/approach (e.g., "Use rolling array DP for space optimization")
   - Content: Explain WHY it works and WHAT insight makes it effective
   - Focus on: What was the key insight? Under what conditions does this work?

   **For Failure type:**
   - Title: Describe the SPECIFIC mistake, not just the approach (e.g., "BFS without boundary check causes index out of bounds", NOT just "BFS implementation")
   - Content: Explain WHY it failed and WHAT specific condition triggered the failure
   - Focus on: What exactly went wrong? What should be checked/avoided?

4. **Filter out trivial items**
   - Remove entries that only reflect measurement noise.
   - Remove entries with negligible effect and no actionable lesson.
   - Keep roughly **5–8** useful experiences.

## Output Format

```json
{
  "thought_process": "Brief explanation (1-2 sentences).",
  "experience_library": [
    {
      "type": "Success | Failure | Neutral",
      "title": "Specific, descriptive title",
      "description": "One-sentence summary of the insight/lesson.",
      "content": "2–6 sentences explaining when/why this works or fails."
    }
  ]
}
```
"""
        user_prompt = f"""## Current Experience Library

{json.dumps(experience_library, indent=2, ensure_ascii=False)}

## Task
Compress and consolidate the experience_library above. Output ONLY the valid JSON object.
"""
        return system_prompt, user_prompt

    def extract_and_update(
        self,
        instance_name: str,
        current_entry: dict[str, Any],
        source_entries: list[dict[str, Any]] | None = None,
        best_entry: dict[str, Any] | None = None,
        problem_description: str | None = None,
        language: str | None = None,
        optimization_target: str | None = None,
    ) -> None:
        """
        根据一次迭代的总结与性能数据，进行记忆提炼并更新本地记忆库。

        Args:
            instance_name: 实例名称。
            current_entry: 当前轨迹条目（包含 iteration, summary, code, perf_metrics 等）。
            source_entries: 来源轨迹条目列表（用于对比 diff 和性能变化）。
            best_entry: 当前最佳轨迹条目（用于参考）。
            problem_description: 问题描述。
            language: 编程语言。
            optimization_target: 优化目标（如 Runtime, Memory 等）。
        """
        memory = self.load()
        attempted = memory.get("direction_board") or []

        # Extract data from entries
        iteration = int(current_entry.get("iteration") or 0)
        perf_metrics = current_entry.get("perf_metrics")
        current_label = str(current_entry.get("label") or "")

        # 计算性能差异（old vs new）
        perf_old = None
        perf_new = None
        try:
            # New performance
            if perf_metrics:
                new_perf_val = perf_metrics.get("performance")
                perf_new = float(new_perf_val) if new_perf_val is not None else None
            if perf_new is None:
                # Fallback to top-level performance field
                new_perf_val = current_entry.get("performance")
                perf_new = float(new_perf_val) if new_perf_val is not None else None

            # Old performance: Compare against ALL source entries (Best/Min)
            source_perfs = []
            if source_entries:
                for entry in source_entries:
                    val = None
                    # Try perf_metrics
                    entry_perf_metrics = entry.get("perf_metrics")
                    if entry_perf_metrics:
                        perf_val = entry_perf_metrics.get("performance")
                        val = float(perf_val) if perf_val is not None else None
                    # Try top-level
                    if val is None:
                        perf_val = entry.get("performance")
                        val = float(perf_val) if perf_val is not None else None

                    if val is not None:
                        source_perfs.append(val)

            if source_perfs:
                # Assuming that Lower is Better, so we take the minimum of source entries
                perf_old = min(source_perfs)
        except Exception:
            pass

        # LLM 提炼：生成 Direction Item + 生成 Reasoning Item
        dir_items: list[dict[str, Any]] = []
        mem_items: list[dict[str, Any]] = []
        if self.llm_client:
            try:
                sys_prompt, user_prompt = self._build_extraction_prompts(
                    problem_description,
                    perf_old,
                    perf_new,
                    source_entries,
                    current_entry,
                    best_entry,
                    attempted,
                    language=language,
                    optimization_target=optimization_target,
                    current_solution_id=current_label,
                )
                last_error: str | None = None
                for attempt in range(1, 4):
                    try:
                        llm_response = self.llm_client.call_with_system_prompt(
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=0.7,
                            max_tokens=None,  # 使用配置中的 max_output_tokens
                            usage_context="local_memory.extract_and_update",
                        )
                        self.logger.debug(f"LLM原始响应 (第{attempt}次):\n{llm_response}")
                        llm_response = self.llm_client.clean_think_tags(llm_response)
                        self.logger.debug(f"LLM清理后响应 (第{attempt}次):\n{llm_response}")
                        parsed_response = self._parse_llm_json(llm_response)
                        self._validate_memory_response(parsed_response)
                        dir_items = [d for d in parsed_response.get("new_direction_items") or [] if isinstance(d, dict)]
                        mem_items = [m for m in parsed_response.get("new_memory_items") or [] if isinstance(m, dict)]
                        # 合并全部新项到内存结构
                        if dir_items:
                            self._merge_direction_board(memory, dir_items)
                        if mem_items:
                            self._merge_experience_library(memory, mem_items)
                        self.logger.info(f"LLM记忆提炼成功 (第{attempt}次)")
                        break
                    except ValueError as e:
                        last_error = "invalid_response_format"
                        self.logger.warning(f"LLM记忆提炼解析失败: 响应格式错误或无有效JSON片段 (第{attempt}次): {e}")
                    except Exception as e:
                        last_error = "llm_call_failed"
                        self.logger.warning(f"LLM记忆提炼调用失败 (第{attempt}次): {e}")
                if last_error:
                    self.logger.error(f"LLM记忆提炼最终失败: {last_error}")
            except Exception as e:
                self.logger.warning(f"LLM记忆提炼失败，使用规则回退: {e}")

        # 不再进行单项插入的兼容处理

        # 更新全局状态
        gs = memory.get("global_status") or {}
        gs["current_generation"] = int(iteration)
        try:
            current_solution_id = current_entry.get("label", "")
        except Exception:
            current_solution_id = None
        gs["current_solution_id"] = current_solution_id

        try:
            best_solution_id = best_entry.get("label", "")
        except Exception:
            best_solution_id = None
        gs["best_solution_id"] = best_solution_id

        memory["global_status"] = gs

        # 压缩（必要时）并保存
        self.compress_if_needed(memory)
        self.save(memory)
        self.logger.info(
            json.dumps(
                {
                    "memory_update": {
                        "instance": instance_name,
                        "iteration": iteration,
                        "label": current_label,
                        "current_generation": memory.get("global_status", {}).get("current_generation"),
                    }
                },
                ensure_ascii=False,
            )
        )
