#!/usr/bin/env python3
"""
SE Operators Base Classes

定义了所有算子的基类和核心接口。
算子是模块化的、可重用的组件，用于执行特定的轨迹操作，如生成、交叉或过滤。
"""

from __future__ import annotations

import abc
import random
import re
from typing import Any

from core.utils.llm_client import LLMClient
from core.utils.se_logger import get_se_logger
from core.utils.traj_pool_manager import TrajPoolManager


class BaseOperator(abc.ABC):
    """
    SE算子基类，定义通用功能和新的 `run` 接口。
    所有算子都应继承自此类。
    """

    def __init__(self, config: dict[str, Any]):
        """
        初始化算子。

        Args:
            config: 包含 `operator_models` 等模型配置信息。
        """
        self.config = config
        self.llm_client: LLMClient | None = None
        self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")

    def _setup_model(self) -> None:
        """设置LLM客户端实例。"""
        if self.llm_client is not None:
            return
        model_config_data = self.config.get("operator_models", self.config.get("model", {}))
        self.llm_client = LLMClient(model_config_data)
        self.logger.info(f"LLM客户端已初始化: {model_config_data.get('name')}")

    def _call_llm_api(self, prompt: str, system_prompt: str = "") -> str:
        """
        调用LLM API。

        Args:
            prompt: 用户提示。
            system_prompt: 系统提示。

        Returns:
            LLM生成的响应文本。
        """
        self._setup_model()
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        try:
            model_cfg = self.config.get("operator_models", self.config.get("model", {}))
            temp = model_cfg.get("temperature", 0.3)
            max_out = model_cfg.get("max_output_tokens")
            self.logger.debug(f"LLM系统提示词:\n{system_prompt}")
            self.logger.debug(f"LLM用户提示词:\n{prompt}")
            message = self.llm_client.call_llm(
                history,
                temperature=temp,
                max_tokens=max_out,
                usage_context=f"operator.{self.get_name()}",
            )
            self.logger.debug(f"LLM原始响应:\n{message}")
            if message:
                message = self.llm_client.clean_think_tags(message)
            self.logger.debug(f"LLM清理后响应:\n{message}")
            return message or ""
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            return ""

    def _extract_code_block_py(self, text: str) -> str | None:
        """从LLM输出中提取 ```py ... ``` 代码块内容。"""
        if not isinstance(text, str) or not text:
            return None
        pattern = re.compile(r"```(?:py|python)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
        m = pattern.search(text)
        if m:
            return m.group(1).strip() or None
        return None

    def _extract_code_text(self, text: str) -> str | None:
        """优先提取代码块，否则返回原始文本并尝试清理。"""
        if not isinstance(text, str) or not text.strip():
            return None
        block = self._extract_code_block_py(text)
        if isinstance(block, str) and block.strip():
            return block.strip()

        raw_code = text.strip()
        if raw_code.startswith("```") and raw_code.endswith("```"):
            try:
                raw_code = re.sub(r"^```(?:py|python)?\s*\n?", "", raw_code, flags=re.IGNORECASE)
                raw_code = re.sub(r"\n?```$", "", raw_code)
            except Exception:
                pass
        return raw_code.strip() or None

    def _require_py_block_with_retry(
        self,
        build_prompt_fn,
        max_retries: int = 2,
        temperature_override: float | None = None,
    ) -> str | None:
        """要求LLM以```py代码块```输出，若未满足则重试。"""
        self._setup_model()
        original_model_cfg = self.config.get("operator_models", self.config.get("model", {}))
        base_temp = original_model_cfg.get("temperature", 0.3)
        temp_to_use = base_temp if temperature_override is None else temperature_override

        for attempt in range(max_retries + 1):
            try:
                prompt, system_prompt = build_prompt_fn(attempt)
                pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
                common = pcfg.get("base_operator", {}) if isinstance(pcfg.get("base_operator"), dict) else {}
                enforce_tail = common.get(
                    "enforce_tail",
                    pcfg.get(
                        "operator_enforce_tail",
                        "\n\nSTRICT FORMAT: Wrap the entire solution inside a fenced code block starting with ```py and ending with ```.",
                    ),
                )
                import_blocks = common.get(
                    "imports_block",
                    pcfg.get(
                        "operator_imports_block",
                        """\n\nAllowed Imports Scope: You may only import libraries within the scope defined below.
```python
import re
from re import match, search, sub, split, findall, finditer
import sys
from sys import maxsize, stdin
import json
from json import loads
import math
from math import floor, ceil, factorial, sqrt, isqrt, inf, log2, log10, sin, cos, tan, pi, e, comb, perm, gcd, lcm
import copy
import pickle
import heapq
from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest
import bisect
from bisect import bisect_left, bisect_right
import string
from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, whitespace, punctuation, hexdigits
import random
import operator
import itertools
from itertools import combinations, permutations, product, groupby, chain, accumulate, zip_longest
import functools
from functools import lru_cache, cache, reduce
import collections
from collections import OrderedDict, defaultdict, Counter, deque
from typing import Set, Dict, List, Optional, Tuple
import sortedcontainers # pip install sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet
```""",
                    ),
                )
                optimization_target = common.get(
                    "optimization_target",
                    pcfg.get(
                        "operator_optimization_target",
                        """
CORE TASK
Your task is to iteratively improve a given program in python for the problem described below, aiming to increase its **runtime**.

GUIDING PRINCEPLES
Your core philosophy is **CORRECTNESS FIRST, THEN PERFORMANCE**.
1.  **Correctness Priority**: Your primary goal is to produce correct outputs for all required cases. Ensure any changes maintain or improve correctness *before* optimizing for performance.
2.  **Performance Focus**: Improve performance only *after* correctness is assured. Prefer algorithmic improvements over micro-optimizations.
3.  **Context Utilization**: You MUST leverage all provided information (evolution history in the chat, current metrics, artifacts etc.) to make informed optimization decisions.
4.  **Substantial Impact**: Focus on meaningful improvements that significantly impact the fitness score.
5.  **Code Quality**: Keep the code readable, robust, and maintainable. Avoid unnecessary refactors.
6.  **Diversity**: Explore alternative algorithms, data structures, or techniques (e.g., built-in operators, packages) when appropriate.
                        """,
                    ),
                )
                system_prompt_use = system_prompt or ""
                if isinstance(enforce_tail, str) and enforce_tail.strip():
                    system_prompt_use += enforce_tail
                if isinstance(import_blocks, str) and import_blocks.strip():
                    system_prompt_use += import_blocks
                if isinstance(optimization_target, str) and optimization_target.strip():
                    system_prompt_use += optimization_target

                history = [{"role": "system", "content": system_prompt_use}, {"role": "user", "content": prompt}]
                max_out = original_model_cfg.get("max_output_tokens")
                enable_thinking = None if attempt == 0 else False

                self.logger.info(f"第{attempt + 1}次尝试，温度={temp_to_use}")
                self.logger.debug(f"LLM系统提示词(重试第{attempt + 1}次)")
                self.logger.debug(f"LLM用户提示词(重试第{attempt + 1}次)")

                message = self.llm_client.call_llm(
                    history,
                    temperature=temp_to_use,
                    max_tokens=max_out,
                    enable_thinking=enable_thinking,
                    usage_context=f"operator.{self.get_name()}",
                )
                self.logger.debug(f"LLM原始响应(重试第{attempt + 1}次):\n{message}")
                if message:
                    message = self.llm_client.clean_think_tags(message)
                # self.logger.debug(f"LLM清理后响应(重试第{attempt + 1}次):\n{message}")

                code = self._extract_code_block_py(message or "")
                if code:
                    return code

                self.logger.warning("未检测到```py代码块，进行重试")
            except Exception as e:
                self.logger.error(f"格式化代码块生成失败: {e}")
                continue
        return None

    def _format_entry(self, approaches_data: dict[str, Any]) -> str:
        return TrajPoolManager.format_entry(approaches_data)

    def _weighted_select_labels(
        self, entry: dict[str, Any], k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        """基于 performance 的线性加权采样选择子标签，performance 越低权重越高。
        若提供 allowed_labels，则仅在该集合中进行采样（忽略不存在的标签）。
        """
        if not isinstance(entry, dict):
            return []
        items: list[tuple[str, float]] = []
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if allowed_labels is not None:
                lab = str(subkey)
                lab2 = str(subval.get("label")) if isinstance(subval.get("label"), str) else None
                if lab not in allowed_labels and (lab2 is None or lab2 not in allowed_labels):
                    continue
            perf = subval.get("performance")
            try:
                perf_val = float(perf) if perf is not None else 1.0
            except Exception:
                perf_val = 1.0
            items.append((str(subkey), perf_val))
        if not items:
            return []
        eps = 1e-9
        selected: list[str] = []
        remaining = items.copy()
        for _ in range(min(k, len(remaining))):
            weights = [max(0.001, 1.0 / max(eps, perf)) for _, perf in remaining]
            total = sum(weights)
            if total <= 0:
                choice = random.choice(remaining)[0]
            else:
                weights = [w / total for w in weights]
                r = random.random()
                s = 0.0
                choice = remaining[-1][0]
                for (label_key, perf), w in zip(remaining, weights):
                    s += w
                    if r <= s:
                        choice = label_key
                        break
            selected.append(choice)
            remaining = [it for it in remaining if it[0] != choice]
        return selected

    def _random_select_labels(
        self, entry: dict[str, Any], k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        if not isinstance(entry, dict):
            return []
        candidates: list[str] = []
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if allowed_labels is not None:
                lab = str(subkey)
                lab2 = str(subval.get("label")) if isinstance(subval.get("label"), str) else None
                if lab not in allowed_labels and (lab2 is None or lab2 not in allowed_labels):
                    continue
            candidates.append(str(subkey))
        if not candidates:
            return []
        k = min(k, len(candidates))
        try:
            return random.sample(candidates, k)
        except Exception:
            out: list[str] = []
            pool = candidates.copy()
            for _ in range(k):
                choice = random.choice(pool)
                out.append(choice)
                pool = [c for c in pool if c != choice]
                if not pool:
                    break
            return out

    def _get_selection_mode(self, step_config: dict[str, Any]) -> str:
        try:
            v = (step_config or {}).get("selection_mode")
            if isinstance(v, str) and v.strip():
                m = v.strip().lower()
                if m in ("weighted", "random"):
                    return m
            g = (self.config or {}).get("operator_selection_mode")
            if isinstance(g, str) and g.strip():
                m = g.strip().lower()
                if m in ("weighted", "random"):
                    return m
        except Exception:
            pass
        return "weighted"

    def _resolve_label_subkey(self, entry: dict[str, Any], label: str) -> str | None:
        """将外部提供的标签解析为 entry 的子键。
        优先匹配子键名，其次匹配子项内部的 `label` 字段。
        """
        if not isinstance(entry, dict):
            return None
        lab = str(label)
        if lab in entry and isinstance(entry.get(lab), dict):
            return lab
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if str(subval.get("label")) == lab:
                return str(subkey)
        return None

    def _select_source_labels(self, entry: dict[str, Any], step_config: dict[str, Any], required_n: int) -> list[str]:
        """统一选择源轨迹标签。
        规则：
        - 若 `inputs` 标签数目 == required_n：直接使用 `inputs`
        - 若 `inputs` 标签数目 >  required_n：在 `inputs` 范围内加权采样 required_n 个
        - 若 `inputs` 标签数目 <  required_n：先使用已有 `inputs`，剩余从整个 entry 中加权采样补齐
        返回 entry 子键名列表，唯一且最多 required_n 个。
        """
        if not isinstance(entry, dict):
            return []
        inputs = step_config.get("inputs") or []
        provided_labels = [str(i.get("label")) for i in inputs if isinstance(i, dict) and i.get("label")]
        # 解析为 entry 子键
        resolved = []
        seen = set()
        for lab in provided_labels:
            subkey = self._resolve_label_subkey(entry, lab)
            if subkey and subkey not in seen:
                resolved.append(subkey)
                seen.add(subkey)

        need = max(0, int(required_n))
        count = len(resolved)
        if count == need:
            return resolved
        if count > need:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled = self._random_select_labels(entry, k=need, allowed_labels=resolved)
            else:
                sampled = self._weighted_select_labels(entry, k=need, allowed_labels=resolved)
            # 去重并返回
            out = []
            used = set()
            for s in sampled:
                if s not in used:
                    out.append(s)
                    used.add(s)
            return out

        # count < need：先用已有，再补齐
        out = list(resolved)
        used = set(out)
        # 构建候选集合（排除已选）
        all_subkeys = [str(k) for k, v in entry.items() if k != "problem" and isinstance(v, dict)]
        remaining = [k for k in all_subkeys if k not in used]
        if remaining:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled_more = self._random_select_labels(entry, k=need - count, allowed_labels=remaining)
            else:
                sampled_more = self._weighted_select_labels(entry, k=need - count, allowed_labels=remaining)
            for s in sampled_more:
                if s not in used:
                    out.append(s)
                    used.add(s)
                if len(out) >= need:
                    break
        return out[:need]

    @abc.abstractmethod
    def get_name(self) -> str:
        """获取算子名称。"""
        pass

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        """
        执行算子的核心逻辑。

        Args:
            step_config: 当前步骤的配置，来自YAML文件。
            traj_pool_manager: 轨迹池管理器实例。
            workspace_dir: 工作区目录。

        Returns:
            一个包含算子执行结果的字典。
            例如: {"generated_code": "..."} 或 {"filtered_labels": ["..."]}
        """
        pass


class TemplateOperator(BaseOperator):
    """
    模板算子基类，用于为下一次 PerfAgent 运行生成初始代码。
    这类算子的 `run` 方法通常返回一个包含 `generated_code` 键的字典。
    """

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        pass


class EnhanceOperator(BaseOperator):
    """
    增强算子基类，用于为下一次 PerfAgent 运行生成增强历史配置。
    这类算子的 `run` 方法通常返回一个包含 `enhance_history_filter_json` 键的字典。
    """

    @abc.abstractmethod
    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        pass
