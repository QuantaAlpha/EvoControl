#!/usr/bin/env python3
"""
Trajectory Pool Manager (Label-based)

管理一个以“标签”为键的轨迹池。每个轨迹都是一个独立的实体，包含了执行摘要、
性能数据、代码路径等元信息。
"""

import copy
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.se_logger import get_se_logger


class TrajPoolManager:
    """
    轨迹池管理器 (基于标签)。
    负责加载、保存、查询和修改存储在 traj.pool 文件中的轨迹数据。
    轨迹池是一个以字符串标签为键的字典。
    """

    def __init__(
        self,
        pool_path: str,
        llm_client=None,
        num_workers: int | None = None,
        memory_manager: Optional["LocalMemoryManager"] = None,  # noqa: F821
        prompt_config: dict[str, Any] | None = None,
    ):
        """
        初始化轨迹池管理器。

        Args:
            pool_path: traj.pool 文件路径。
            llm_client: LLM 客户端实例，用于轨迹总结。
            num_workers: 并行生成总结的并发数。
        """
        self.pool_path = Path(pool_path)
        self.llm_client = llm_client
        # 并发控制（来自SE配置）；为空则使用默认策略
        self.num_workers = num_workers
        self.logger = get_se_logger("traj_pool", emoji="🏊")
        self.memory_manager = memory_manager
        self.prompt_config = prompt_config or {}
        self._best_labels: dict[str, str] = {}

    def initialize_pool(self) -> None:
        """初始化轨迹池文件。如果文件不存在，则创建一个空的 JSON 对象。"""
        try:
            # 确保目录存在
            self.pool_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果文件不存在，创建空的轨迹池
            if not self.pool_path.exists():
                with open(self.pool_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                self.logger.info(f"初始化空的轨迹池: {self.pool_path}")
            else:
                self.logger.info(f"轨迹池已存在: {self.pool_path}")
        except Exception as e:
            self.logger.error(f"初始化轨迹池失败: {e}")
            raise
        try:
            self.refresh_best_labels()
        except Exception:
            pass

    def load_pool(self) -> dict[str, Any]:
        """从文件加载整个轨迹池。"""
        try:
            if not self.pool_path.exists():
                self.logger.warning("轨迹池文件不存在，返回空池")
                return {}
            with open(self.pool_path, encoding="utf-8") as f:
                pool_data = json.load(f)
            self.logger.debug(f"加载了 {len(pool_data)} 条轨迹")
            return pool_data
        except Exception as e:
            self.logger.error(f"加载轨迹池失败: {e}")
            return {}

    def save_pool(self, pool_data: dict[str, Any]) -> None:
        """将轨迹池数据完整保存到文件。"""
        try:
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"保存了 {len(pool_data)} 条轨迹到轨迹池")
        except Exception as e:
            self.logger.error(f"保存轨迹池失败: {e}")
            raise

    def get_instance(self, instance_name: str) -> dict[str, Any] | None:
        """获取指定实例的所有轨迹数据。"""
        pool_data = self.load_pool()
        return pool_data.get(instance_name)

    def get_trajectory(self, label: str, instance_name: str | None = None) -> dict[str, Any] | None:
        """
        通过标签获取单个轨迹。

        Args:
            label: 轨迹标签。
            instance_name: (可选) 实例名称。如果提供，仅在该实例内查找。

        Returns:
            找到的轨迹字典，否则返回 None。
        """
        pool_data = self.load_pool()

        def _find_in_entry(entry_data: dict[str, Any]) -> dict[str, Any] | None:
            """在单个实例条目内查找轨迹。"""
            if not isinstance(entry_data, dict):
                return None
            # 优先匹配子键名
            if label in entry_data and isinstance(entry_data[label], dict):
                return entry_data[label]
            # 其次匹配子条目内的 "label" 字段
            for subkey, subval in entry_data.items():
                if subkey == "problem":
                    continue
                if isinstance(subval, dict) and str(subval.get("label")) == label:
                    return subval
            return None

        if instance_name:
            entry = pool_data.get(instance_name)
            return _find_in_entry(entry) if entry else None

        for entry in pool_data.values():
            found = _find_in_entry(entry)
            if found:
                return found
        return None

    def get_all_trajectories(self) -> dict[str, Any]:
        """获取池中所有的轨迹。"""
        return self.load_pool()

    def get_all_labels(self, instance_name: str | None = None) -> list[str]:
        """
        获取所有唯一的轨迹标签。

        Args:
            instance_name: (可选) 如果提供，仅返回该实例的标签。

        Returns:
            唯一的轨迹标签列表。
        """
        pool_data = self.load_pool()
        labels: set[str] = set()

        def _extract_labels_from_entry(entry: dict[str, Any]):
            if isinstance(entry, dict):
                for subkey, subval in entry.items():
                    if subkey == "problem":
                        continue
                    if isinstance(subval, dict):
                        # 优先使用 "label" 字段
                        if "label" in subval:
                            labels.add(str(subval["label"]))
                        # 否则使用子键名作为标签
                        else:
                            labels.add(subkey)

        if instance_name:
            entry = pool_data.get(instance_name)
            if entry:
                _extract_labels_from_entry(entry)
        else:
            for entry in pool_data.values():
                _extract_labels_from_entry(entry)

        return sorted(list(labels))

    def add_or_update_instance(self, instance_name: str, entry: dict[str, Any]) -> None:
        """
        向指定实例添加或更新一个轨迹条目。

        注意：此方法会立即加载和保存整个池，I/O 开销较大。
        对于批量操作，请使用 `summarize_and_add_trajectories`。

        Args:
            instance_name: 实例名称。
            entry: 要添加或更新的轨迹条目，必须包含 'label'。
        """
        pool_data = self.load_pool()
        inst_key = str(instance_name)
        existing = pool_data.get(inst_key) or {}

        # 保持顶层 "problem" 描述
        problem_text = entry.get("problem") or existing.get("problem")
        merged = {**existing}
        if problem_text is not None:
            merged["problem"] = problem_text

        # 将本次迭代的 "label" 作为子键，保存条目内容
        iter_label = entry.get("label")
        if not iter_label:
            raise ValueError("缺少 'label' 用于实例条目的子键")

        detail = entry.copy()
        detail.pop("problem", None)  # 避免在子条目中重复存储
        merged[str(iter_label)] = detail
        pool_data[inst_key] = merged

        self.save_pool(pool_data)
        self.logger.info(f"已更新实例 '{instance_name}' 的条目: {iter_label}")
        try:
            best = self._select_best_label(merged)
            if best:
                self._best_labels[inst_key] = best
        except Exception:
            pass

    def add_trajectory(self, label: str, traj_info: dict[str, Any], instance_name: str | None = None) -> None:
        """
        添加单条轨迹记录。

        Args:
            label: 轨迹标签。
            traj_info: 轨迹信息字典。
            instance_name: (可选) 实例名称。
        """
        inst_name = str(instance_name or traj_info.get("instance_name") or "")
        if not inst_name:
            raise ValueError("缺少 instance_name，无法添加轨迹")

        # 统一处理 trajectory_raw，确保其为 JSON 对象
        raw_content = traj_info.get("trajectory_raw")
        if isinstance(raw_content, str):
            try:
                trajectory_raw = json.loads(raw_content)
            except json.JSONDecodeError:
                self.logger.warning(f"无法将 trajectory_raw 解析为 JSON (标签: {label})，将作为原始文本存储。")
                trajectory_raw = {"_raw_text": raw_content}
        else:
            trajectory_raw = raw_content

        entry = {
            "problem": traj_info.get("problem_description") or traj_info.get("problem_statement"),
            "label": label,
            "summary": traj_info.get("summary") or {},
            "performance": traj_info.get("performance"),
            "source_dir": traj_info.get("source_dir"),
            "code": traj_info.get("patch_content") or traj_info.get("content"),
            "trajectory_raw": trajectory_raw,
            "iteration": traj_info.get("iteration"),
        }
        self.add_or_update_instance(inst_name, entry)

    def relabel(
        self,
        old_label: str,
        new_label: str,
        instance_name: str | None = None,
        operator_name: str | None = None,
        delete_old: bool = False,
    ) -> None:
        pool_data = self.load_pool()
        if instance_name:
            if instance_name not in pool_data:
                raise ValueError(f"实例 '{instance_name}' 不存在，无法重命名标签。")
            inst_entry = pool_data[instance_name]
            if old_label in inst_entry:
                old_entry = inst_entry.get(old_label)
                new_entry = copy.deepcopy(old_entry) if isinstance(old_entry, dict) else old_entry
                if isinstance(new_entry, dict):
                    # 更新 relabel 后 entry 相关的信息
                    new_entry["label"] = new_label
                    if operator_name is not None:
                        new_entry["operator_name"] = operator_name
                    new_entry["source_entry_labels"] = [old_label]
                inst_entry[str(new_label)] = new_entry
                if delete_old:
                    try:
                        del inst_entry[old_label]
                    except Exception:
                        pass
            inst_entry["label"] = new_label
        else:
            target_inst = None
            for inst_name, entry in pool_data.items():
                if isinstance(entry, dict) and entry.get("label") == old_label:
                    target_inst = inst_name
                    break
            if target_inst is None:
                raise ValueError(f"标签 '{old_label}' 不存在，无法重命名。")
            inst_entry = pool_data[target_inst]
            if old_label in inst_entry:
                old_entry = inst_entry.get(old_label)
                new_entry = copy.deepcopy(old_entry) if isinstance(old_entry, dict) else old_entry
                if isinstance(new_entry, dict):
                    new_entry["label"] = new_label
                    if operator_name is not None:
                        new_entry["operator_name"] = operator_name
                    new_entry["source_entry_labels"] = [old_label]
                inst_entry[str(new_label)] = new_entry
                if delete_old:
                    try:
                        del inst_entry[old_label]
                    except Exception:
                        pass
            inst_entry["label"] = new_label
        self.save_pool(pool_data)
        self.logger.info(f"重命名并更新算子 '{old_label}' -> '{new_label}'，operator={operator_name or 'unchanged'}。")

    def delete_trajectories(self, labels: list[str], instance_name: str | None = None) -> None:
        pool_data = self.load_pool()
        deleted_count = 0
        if instance_name:
            if instance_name in pool_data:
                inst_entry = pool_data[instance_name]
                # 删除匹配的子键，不删除整个实例
                for lb in labels:
                    if lb in inst_entry:
                        del inst_entry[lb]
                        deleted_count += 1
        else:
            to_delete = []
            for inst_name, entry in pool_data.items():
                if isinstance(entry, dict):
                    for lb in labels:
                        if lb in entry:
                            to_delete.append((inst_name, lb))
            for inst_name, lb in to_delete:
                try:
                    del pool_data[inst_name][lb]
                    deleted_count += 1
                    self.logger.debug(f"已从实例 '{inst_name}' 删除子条目 '{lb}'。")
                except Exception:
                    pass
        if deleted_count > 0:
            self.save_pool(pool_data)
        self.logger.info(f"从轨迹池中删除了 {deleted_count} 条轨迹。")

    def summarize_trajectory(
        self,
        trajectory_content: str,
        patch_content: str,
        iteration: int,
        label: str,
        problem_description: str | None = None,
        best_solution_text: str | None = None,
        target_solution_text: str | None = None,
    ) -> dict[str, Any]:
        """
        使用 LLM (或备用方法) 总结单条轨迹的内容。

        Args:
            trajectory_content: .tra 文件内容。
            patch_content: .patch/.pred 文件内容或 "FAILED_NO_PATCH"。
            iteration: 迭代号 (用于上下文)。
            label: 轨迹标签 (用于日志)。
            problem_description: 问题描述。

        Returns:
            轨迹总结字典。
        """
        from .llm_client import TrajectorySummarizer
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer()

        # 检查是否为失败实例
        is_failed = patch_content == "FAILED_NO_PATCH"

        try:
            if self.llm_client:
                traj_summarizer = TrajectorySummarizer(self.llm_client, prompt_config=self.prompt_config)
                summary = traj_summarizer.summarize_trajectory(
                    trajectory_content,
                    patch_content,
                    iteration,
                    problem_description=problem_description,
                    best_solution_text=best_solution_text,
                    target_solution_text=target_solution_text,
                )
                # 为失败实例添加特殊标记
                if is_failed:
                    summary["strategy_status"] = "FAILED"
                    summary["failure_reason"] = "No patch/prediction generated"
                self.logger.debug(f"LLM 轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
            else:
                self.logger.info(f"未配置 LLM 客户端，使用备用总结 (标签 '{label}')")
                summary = summarizer.create_fallback_summary(trajectory_content, patch_content, iteration)
                self.logger.debug(f"备用轨迹总结 (标签 '{label}'): {summary.get('approach_summary', 'N/A')}")
                return summary
        except Exception as e:
            self.logger.error(f"轨迹总结失败 (标签 '{label}'): {e}")
            return {
                "error": "summarization_failed",
                "details": str(e),
                "iteration": iteration,
                "label": label,
            }

    def _gather_memory_context(
        self, instance_name: str, res: dict[str, Any], pool_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        准备 Memory 模块所需的上下文信息。

        Args:
            instance_name: 实例名。
            res: 当前轨迹结果字典。
            pool_data: 整个轨迹池数据（用于查找 Source/Best）。

        Returns:
            包含 extract_and_update 所需参数的字典。
        """
        inst_entry = pool_data.get(str(instance_name)) or {}

        # 1. Source Entries (Old Code & Context)
        source_entries = []
        src_labels = res.get("source_entry_labels")
        if src_labels and isinstance(src_labels, list):
            for sl in src_labels:
                sl_str = str(sl)
                if sl_str in inst_entry and isinstance(inst_entry[sl_str], dict):
                    source_entries.append(inst_entry[sl_str])

        # 2. Best Entry (Best Code & Context)
        best_entry = None
        best_label = self._best_labels.get(str(instance_name))
        if not best_label:
            best_label = self._select_best_label(inst_entry)

        if best_label and str(best_label) in inst_entry:
            best_entry = inst_entry[str(best_label)]

        return {
            "instance_name": str(instance_name),
            "current_entry": res,
            "source_entries": source_entries,
            "best_entry": best_entry,
            "problem_description": inst_entry.get("problem"),
            "language": res.get("language"),
            "optimization_target": res.get("optimization_target"),
        }

    def _process_single_trajectory_summary(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """
        线程工作函数：总结单条轨迹并构建完整的 TrajectoryInfo 对象。
        """
        try:
            # 从 prompt_config.summarizer.enable_summary 读取是否执行LLM总结；默认 True
            do_summary = True
            try:
                summarizer_cfg = (
                    self.prompt_config.get("summarizer", {}) if isinstance(self.prompt_config, dict) else {}
                )
                flag = summarizer_cfg.get("enable_summary")
                if isinstance(flag, bool):
                    do_summary = flag
            except Exception:
                pass

            best_solution_text = ""
            try:
                inst = str(item.get("instance_name") or "")
                if inst:
                    best_label = self.get_best_label(inst)
                    if best_label:
                        pool_data = self.load_pool()
                        entry = pool_data.get(inst)
                        if isinstance(entry, dict):
                            cand = entry.get(str(best_label))
                            if isinstance(cand, dict):
                                best_solution_text = self.format_entry({str(best_label): cand})
            except Exception:
                best_solution_text = ""

            target_solution_text = ""
            try:
                lab = str(item.get("label") or "target")
                target_solution_text = self.format_entry(
                    {
                        lab: {
                            "label": lab,
                            "iteration": item.get("iteration"),
                            "code": item.get("patch_content") or "",
                            "perf_metrics": item.get("perf_metrics"),
                            "performance": item.get("performance"),
                            "operator_name": item.get("operator_name"),
                        }
                    }
                )
            except Exception:
                target_solution_text = str(item.get("patch_content") or "")

            summary = None
            if do_summary:
                summary = self.summarize_trajectory(
                    trajectory_content=item["trajectory_content"],
                    patch_content=item["patch_content"],
                    iteration=item["iteration"],
                    label=item["label"],
                    problem_description=item.get("problem_description"),
                    best_solution_text=best_solution_text,
                    target_solution_text=target_solution_text,
                )
            else:
                summary = {}

            # 解析 .tra 原始内容为 JSON 对象，如果失败则作为原始文本
            raw_content = item.get("trajectory_content")
            if isinstance(raw_content, str):
                try:
                    trajectory_raw_obj = json.loads(raw_content)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"无法将 trajectory_raw 解析为 JSON (标签: {item.get('label')})，将作为原始文本存储。"
                    )
                    trajectory_raw_obj = {"_raw_text": raw_content}
            else:
                trajectory_raw_obj = raw_content

            # 从 item 或全局配置中获取语言和优化目标
            cfg = self.prompt_config.get("summarizer", {}) if self.prompt_config else {}
            lang = item.get("language") or cfg.get("language") or "Unknown"
            target = item.get("optimization_target") or cfg.get("optimization_target") or "Runtime"

            return {
                "label": item["label"],
                "instance_name": item["instance_name"],
                "iteration": item["iteration"],
                "performance": item.get("performance"),
                "source_dir": item.get("source_dir"),
                "summary": summary,
                "problem_description": item.get("problem_description"),
                "code": item["patch_content"],
                "trajectory_raw": trajectory_raw_obj,
                "source_entry_labels": item.get("source_entry_labels"),
                "operator_name": item.get("operator_name"),
                "perf_metrics": item.get("perf_metrics"),
                "language": lang,
                "optimization_target": target,
                "meta": {"summary_enabled": bool(do_summary)},
            }
        except Exception as e:
            self.logger.error(f"并行轨迹总结任务失败 (标签 '{item.get('label')}'): {e}")
            return None

    def summarize_and_add_trajectories(
        self, trajectories_to_process: list[dict[str, Any]], num_workers: int | None = None
    ) -> int:
        """
        并行生成多条轨迹的总结，并一次性将它们作为新条目添加到轨迹池中。

        Args:
            trajectories_to_process: 待处理轨迹信息的列表。每个元素是一个字典，包含:
                - "label": str
                - "instance_name": str
                - "problem_description": str
                - "trajectory_content": str  (.tra 内容)
                - "patch_content": str       (.pred/.patch 文本)
                - "iteration": int
                - "perf_metrics": dict | None  包含:
                    - "passed": bool | None
                    - "performance": float | str | None
                    - "artifacts": str | None   (已格式化文本)
                - "performance": float | str | None  (兼容旧字段，若与 perf_metrics 同时存在，优先 perf_metrics.performance)
                - "source_dir": str
                - "operator_name": str | None
                - "source_entry_labels": list[str] | None
            num_workers: 并发数。

        Returns:
            成功处理并添加的轨迹数量。
        """
        if not trajectories_to_process:
            return 0

        try:
            cfg_workers = num_workers if num_workers is not None else self.num_workers
            max_workers = (
                max(1, int(cfg_workers)) if cfg_workers is not None else max(1, min(8, (os.cpu_count() or 4) * 2))
            )
            self.logger.debug(f"并行轨迹总结并发数: {max_workers}")

            newly_completed_trajectories = defaultdict(list)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_label = {
                    executor.submit(self._process_single_trajectory_summary, item): item["label"]
                    for item in trajectories_to_process
                }
                for future in as_completed(future_to_label):
                    label = future_to_label[future]
                    try:
                        if result := future.result():
                            if inst_name := result.get("instance_name"):
                                newly_completed_trajectories[inst_name].append(result)
                    except Exception as e:
                        self.logger.error(f"获取总结结果失败 (标签 '{label}'): {e}")

            if not newly_completed_trajectories:
                self.logger.warning("没有成功生成任何轨迹总结。")
                return 0

            # --- 批量写入 --- #
            pool_data = self.load_pool()
            written_count = 0
            for inst_name, results in newly_completed_trajectories.items():
                for res in results:
                    try:
                        inst_key = str(inst_name)
                        existing = pool_data.get(inst_key) or {}
                        problem_text = res.get("problem_description") or existing.get("problem")
                        merged = {**existing}
                        if problem_text is not None:
                            merged["problem"] = problem_text

                        iter_label = res.get("label")
                        if not iter_label:
                            self.logger.warning(f"跳过缺少 'label' 的轨迹: {res}")
                            continue

                        detail = res.copy()
                        detail.pop("problem_description", None)
                        merged[str(iter_label)] = detail
                        pool_data[inst_key] = merged
                        written_count += 1
                        try:
                            best = self._select_best_label(merged)
                            if best:
                                self._best_labels[inst_key] = best
                        except Exception:
                            pass

                        # 记忆提炼与更新
                        try:
                            if self.memory_manager:
                                ctx = self._gather_memory_context(inst_name, res, pool_data)
                                # 无论是否有 source entries，都进行记忆提炼与更新
                                # 初始解（无 source entries）会触发 initial prompt 分支
                                self.memory_manager.extract_and_update(**ctx)
                        except Exception as me:
                            self.logger.warning(
                                f"本地记忆提炼失败（实例 '{inst_name}' 标签 '{res.get('label')}'): {me}"
                            )
                    except Exception as we:
                        self.logger.error(f"准备写入轨迹池失败: 实例 '{inst_name}' 标签 '{res.get('label')}': {we}")

            if written_count > 0:
                self.save_pool(pool_data)

            self.logger.info(f"成功并行生成并向轨迹池添加了 {written_count} 条实例-迭代条目。")
            return written_count

        except Exception as e:
            self.logger.error(f"并行生成与批量写入轨迹总结失败: {e}")
            raise

    def _select_best_label(self, inst_entry: dict[str, Any]) -> str | None:
        candidates: list[tuple[str, float, int]] = []  # (label, perf, iteration)
        for k, v in inst_entry.items():
            if k == "problem" or not isinstance(v, dict):
                continue
            perf_val = None
            pm = v.get("perf_metrics")
            if isinstance(pm, dict) and pm.get("performance") is not None:
                perf_val = pm.get("performance")
            if perf_val is None:
                perf_val = v.get("performance")

            # parse performance to float
            try:
                if isinstance(perf_val, (int, float)):
                    val = float(perf_val)
                elif isinstance(perf_val, str):
                    s = perf_val.strip().lower()
                    if s in ("inf", "+inf", "infinity", "+infinity"):
                        val = float("inf")
                    elif s in ("-inf", "-infinity"):
                        val = float("inf")  # treat as non-finite for selection purposes
                    elif s == "nan":
                        val = float("inf")
                    else:
                        val = float(s)
                else:
                    val = float("inf")
            except Exception:
                val = float("inf")

            label_txt = str(v.get("label") or k)
            it_raw = v.get("iteration")
            try:
                iter_num = int(it_raw) if it_raw is not None else -1
            except Exception:
                iter_num = -1
            candidates.append((label_txt, val, iter_num))

        if not candidates:
            return None

        finite = [c for c in candidates if math.isfinite(c[1])]
        if finite:
            # choose min performance; tie-breaker: latest iteration
            finite.sort(key=lambda t: (t[1], -t[2]))
            return finite[0][0]
        # no finite performance: choose latest iteration
        candidates.sort(key=lambda t: (-t[2], t[0]))
        return candidates[0][0]

    def get_best_label(self, instance_name: str) -> str | None:
        inst_key = str(instance_name)
        lbl = self._best_labels.get(inst_key)
        if isinstance(lbl, str) and lbl:
            return lbl
        pool_data = self.load_pool()
        entry = pool_data.get(inst_key)
        if not isinstance(entry, dict):
            return None
        try:
            best = self._select_best_label(entry)
            if best:
                self._best_labels[inst_key] = best
            return best
        except Exception:
            return None

    def refresh_best_labels(self) -> None:
        self._best_labels = {}
        pool_data = self.load_pool()
        for inst_name, entry in pool_data.items():
            if isinstance(entry, dict):
                try:
                    best = self._select_best_label(entry)
                    if best:
                        self._best_labels[str(inst_name)] = best
                except Exception:
                    continue

    @staticmethod
    def format_entry(approaches_data: dict[str, Any], include_keys: set[str] | None = None) -> str:
        if not isinstance(approaches_data, dict) or not approaches_data:
            return ""

        def _parse_key_num(k: Any) -> int | None:
            if isinstance(k, str):
                if k.isdigit():
                    try:
                        return int(k)
                    except Exception:
                        return None
                import re

                m = re.search(r"(\d+)$", k)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return None
            return None

        candidates: list[int] = []
        mapping: dict[int, tuple[str, Any]] = {}
        for key, val in approaches_data.items():
            if key == "problem":
                continue
            key_num = _parse_key_num(key)
            iter_num = None
            if isinstance(val, dict):
                it = val.get("iteration")
                try:
                    if it is not None:
                        iter_num = int(it)
                except Exception:
                    iter_num = None
            use_num = iter_num if isinstance(iter_num, int) else key_num if isinstance(key_num, int) else -1
            candidates.append(use_num)
            mapping[use_num] = (str(key), val)

        if not candidates:
            return ""
        latest_iteration = max(candidates)
        latest_key, latest_data = mapping.get(latest_iteration, ("", {}))

        def indent_str(level: int) -> str:
            return "  " * level

        def fmt_value(val: Any, level: int) -> str:
            if val is None:
                return "null"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, str):
                if "\n" in val:
                    lines = val.splitlines()
                    pad = indent_str(level + 1)
                    return "|\n" + "\n".join(f"{pad}{line}" for line in lines)
                return val
            if isinstance(val, dict):
                lines: list[str] = []
                for k, v in val.items():
                    if str(k) in {"trajectory_raw", "source_dir"}:
                        continue
                    # include_keys 仅作用于顶层：当 level==0 时过滤；子层级全部格式化
                    if level == 0 and include_keys is not None and str(k) not in include_keys:
                        continue
                    key_line = f"{indent_str(level)}{k}:"
                    code_key = str(k) in {
                        "code",
                    }
                    if code_key and isinstance(v, str):
                        lines.append(key_line)
                        lines.append(f"```\n{v}\n```")
                    elif isinstance(v, (dict, list)) or (isinstance(v, str) and "\n" in v):
                        lines.append(key_line)
                        lines.append(fmt_value(v, level + 1))
                    else:
                        lines.append(f"{key_line} {fmt_value(v, 0)}")
                return "\n".join(lines)
            if isinstance(val, list):
                lines: list[str] = []
                for item in val:
                    if isinstance(item, (dict, list)) or (isinstance(item, str) and "\n" in item):
                        lines.append(f"{indent_str(level)}-")
                        lines.append(fmt_value(item, level + 1))
                    else:
                        lines.append(f"{indent_str(level)}- {fmt_value(item, 0)}")
                return "\n".join(lines)
            return str(val)

        chosen_label = latest_data.get("label") if isinstance(latest_data, dict) else None
        header = str(chosen_label or latest_key).strip()
        body = fmt_value(latest_data, 0)
        return f"{header}\n{body}".strip() if header else body

    def get_pool_stats(self) -> dict[str, Any]:
        """获取轨迹池的统计信息。"""
        try:
            pool_data = self.load_pool()
            stats = {
                "total_trajectories": len(pool_data),
                "labels": self.get_all_labels(),
            }
            self.logger.debug(f"轨迹池统计: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"获取轨迹池统计失败: {e}")
            return {"total_trajectories": 0, "labels": []}

    def _parse_perf(self, val: Any) -> float:
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("inf", "+inf", "infinity", "+infinity", "nan"):
                    return float("inf")
                return float(s)
            return float("inf")
        except Exception:
            return float("inf")

    def extract_steps(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        pool_data = self.load_pool()
        for inst_name, entry in pool_data.items():
            if not isinstance(entry, dict):
                continue
            for key, val in entry.items():
                if key == "problem" or not isinstance(val, dict):
                    continue
                opn = val.get("operator_name")
                if opn is None or opn in ["filter_trajectories", "plan"]:
                    continue
                src_labels = val.get("source_entry_labels")
                if not isinstance(src_labels, list) or not src_labels:
                    continue

                sources: list[tuple[str, dict, float]] = []
                for sl in src_labels:
                    sl_str = str(sl)
                    src = self.get_trajectory(sl_str, instance_name=str(inst_name))
                    if isinstance(src, dict):
                        pm_prev = src.get("perf_metrics")
                        perf_prev = self._parse_perf(
                            (pm_prev or {}).get("performance") if isinstance(pm_prev, dict) else src.get("performance")
                        )
                        if math.isfinite(perf_prev):
                            sources.append((sl_str, src, perf_prev))
                pm_curr = val.get("perf_metrics")
                perf_curr = self._parse_perf(
                    (pm_curr or {}).get("performance") if isinstance(pm_curr, dict) else val.get("performance")
                )
                if not math.isfinite(perf_curr) or not sources:
                    continue
                best_src = min(sources, key=lambda t: t[2])
                best_label_key, best_detail, perf_prev_best = best_src[0], best_src[1], best_src[2]
                improved = perf_curr < min(t[2] for t in sources)
                delta = perf_prev_best - perf_curr
                pct = (
                    (delta / perf_prev_best * 100.0) if perf_prev_best != 0 and math.isfinite(perf_prev_best) else None
                )
                prev_it = best_detail.get("iteration")
                try:
                    prev_iter = int(prev_it) if prev_it is not None else -1
                except Exception:
                    prev_iter = -1
                curr_it = val.get("iteration")
                try:
                    curr_iter = int(curr_it) if curr_it is not None else -1
                except Exception:
                    curr_iter = -1
                src_label_list = []
                for sl, _, _ in sources:
                    src_label_list.append(str(sl))
                steps.append(
                    {
                        "instance_name": str(inst_name),
                        "prev_label": str(best_detail.get("label") or best_label_key),
                        "curr_label": str(val.get("label") or key),
                        "prev_iter": int(prev_iter),
                        "curr_iter": int(curr_iter),
                        "perf_prev": perf_prev_best,
                        "perf_curr": perf_curr,
                        "delta": delta,
                        "pct": pct,
                        "prev_detail": best_detail,
                        "curr_detail": val,
                        "source_labels": src_label_list,
                        "operator_name": str(opn) if opn is not None else None,
                        "improved": bool(improved),
                    }
                )
        return steps
