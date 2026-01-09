#!/usr/bin/env python3
"""
Filter Trajectories Operator

根据指定的策略（如多样性、性能）对轨迹池中的一部分轨迹进行过滤、排序和选择。
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.utils.traj_pool_manager import TrajPoolManager

from operators.base import BaseOperator


class FilterTrajectoriesOperator(BaseOperator):
    """
    轨迹过滤算子：
    根据综合评分（多样性 + 性能）对输入的轨迹进行排序、过滤，
    并根据配置执行重标签和删除操作。
    """

    def get_name(self) -> str:
        return "filter_trajectories"

    def _calculate_kept_and_deleted_labels(
        self, inst_labels: list[str], entry: dict[str, Any], top_k: int | None = None
    ) -> tuple[list[str], list[str]]:
        n = len(inst_labels)
        self.logger.info(f"开始过滤轨迹，共 {n} 个标签，top_k={top_k}")

        if top_k is None or int(top_k) >= n:
            self.logger.info(f"top_k 为 None 或大于等于标签数量，保留所有 {n} 个标签")
            return inst_labels, []
        k = max(0, int(top_k))

        candidates: list[dict[str, Any]] = []
        for l in inst_labels:
            sub = entry.get(l) if isinstance(entry, dict) else None
            code_text = sub.get("code", "")

            perf_val = sub.get("performance", "")
            try:
                perf_num = float(perf_val) if perf_val is not None else None
            except Exception:
                perf_num = None
            score = 0.0
            if perf_num is not None and math.isfinite(perf_num) and perf_num > 0.0:
                score = 1.0 / perf_num
            candidates.append({"label": l, "text": code_text or "", "score": float(score)})

        if n <= k:
            kept_labels = [c["label"] for c in sorted(candidates, key=lambda x: x["score"], reverse=True)]
            self.logger.info(f"标签数量 {n} 小于等于 top_k {k}，按分数排序保留所有标签")
            return kept_labels, [l for l in inst_labels if l not in kept_labels]

        try:
            self.logger.info("使用聚类算法进行轨迹过滤")
            import numpy as np

            try:
                import Levenshtein

                self.logger.info("成功导入 Levenshtein 模块")
            except ImportError:
                Levenshtein = None
                self.logger.warning("未找到 Levenshtein 模块，将使用其他方法")
            from sklearn.cluster import AgglomerativeClustering

            texts = [c["text"] for c in candidates]
            dist = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    if Levenshtein:
                        sim = Levenshtein.ratio(texts[i], texts[j])
                    else:
                        raise ImportError("Levenshtein module is required for clustering.")
                    d = 1.0 - float(sim)
                    dist[i, j] = d
                    dist[j, i] = d

            num_clusters = min(k, n)
            self.logger.info(f"进行层次聚类，聚类数量: {num_clusters}")
            labels_arr = None
            try:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage="average")
                labels_arr = clustering.fit_predict(dist)
            except TypeError:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="average")
                labels_arr = clustering.fit_predict(dist)

            selected_indices: list[int] = []
            unique_cluster_ids = set(labels_arr)
            self.logger.info(f"聚类完成，实际聚类数量: {len(unique_cluster_ids)}")

            for cluster_id in unique_cluster_ids:
                indices = [idx for idx in range(n) if labels_arr[idx] == cluster_id]
                if not indices:
                    continue
                best_idx = max(indices, key=lambda idx: candidates[idx]["score"])
                selected_indices.append(best_idx)

            selected = [candidates[idx] for idx in selected_indices]
            selected.sort(key=lambda x: x["score"], reverse=True)

            kept_labels = [c["label"] for c in selected]
            self.logger.info(f"聚类过滤完成，保留 {len(kept_labels)} 个标签，删除 {n - len(kept_labels)} 个标签")

        except Exception as e:
            self.logger.warning(f"聚类算法失败: {e}，将使用分数排序作为备用方案")
            candidates.sort(key=lambda x: x["score"], reverse=True)
            kept_labels = [c["label"] for c in candidates[:k]]
            self.logger.info(f"使用分数排序备用方案，保留前 {k} 个标签")

        deleted = [l for l in inst_labels if l not in kept_labels]
        self.logger.info(f"过滤完成，最终保留 {len(kept_labels)} 个标签，删除 {len(deleted)} 个标签")
        return kept_labels, deleted

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager: TrajPoolManager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        input_labels = [item.get("label") for item in step_config.get("inputs", []) if item.get("label")]
        if not input_labels:
            input_labels = traj_pool_manager.get_all_labels()

        filter_strategy = step_config.get("strategy", {})
        top_k = filter_strategy.get("top_k")
        relabel_as: list[str] = []
        if isinstance(filter_strategy.get("relabel_as"), list):
            relabel_as = [str(x) for x in filter_strategy.get("relabel_as")]
        relabel_map: dict[str, str] = (
            filter_strategy.get("relabel", {}) if isinstance(filter_strategy.get("relabel"), dict) else {}
        )

        all_instances = traj_pool_manager.get_all_trajectories() or {}
        per_instance: dict[str, dict[str, list[str]]] = {}
        deleted_records_by_instance: dict[str, list[dict[str, Any]]] = {}

        def _process_instance(args):
            instance_name, entry = args
            if not isinstance(entry, dict):
                return None

            # 提取实例的标签
            inst_labels: list[str] = []
            for k, v in entry.items():
                if k == "problem":
                    continue
                if isinstance(v, dict):
                    if v.get("label"):
                        inst_labels.append(str(v.get("label")))
                    else:
                        inst_labels.append(str(k))

            if input_labels:
                inst_labels = [l for l in inst_labels if l in input_labels]
            if not inst_labels:
                return {
                    "instance_name": instance_name,
                    "final_kept": [],
                    "deleted": [],
                    "relabels": [],
                    "deleted_records": [],
                }

            # 使用封装的逻辑计算要保留和删除的标签
            kept, deleted = self._calculate_kept_and_deleted_labels(inst_labels, entry, top_k)

            # 应用重标签规则
            final_kept: list[str] = []
            relabel_ops: list[tuple[str, str]] = []
            if relabel_as:
                limit = min(len(relabel_as), len(kept))
                for idx in range(limit):
                    old = kept[idx]
                    new = relabel_as[idx]
                    if old != new:
                        relabel_ops.append((old, new))
                    final_kept.append(new)
                for rest in kept[limit:]:
                    final_kept.append(rest)
            else:
                for old in kept:
                    new = relabel_map.get(old, old)
                    if new != old:
                        relabel_ops.append((old, new))
                    final_kept.append(new)

            # 收集删除记录
            recs: list[dict[str, Any]] = []
            if deleted:
                for lb in deleted:
                    subentry = entry.get(lb) if isinstance(entry, dict) else None
                    recs.append(
                        {
                            "original_label": lb,
                            "relabel_target": relabel_map.get(lb),
                            "entry": subentry,
                        }
                    )

            return {
                "instance_name": instance_name,
                "final_kept": final_kept,
                "deleted": deleted,
                "relabels": relabel_ops,
                "deleted_records": recs,
            }

        num = self.config.get("num_workers", 1)
        try:
            max_workers = max(1, int(num))
        except Exception:
            max_workers = 1

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_instance, (name, entry)) for name, entry in (all_instances or {}).items()]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if isinstance(res, dict):
                        results.append(res)
                except Exception:
                    pass

        for res in results:
            inst = res.get("instance_name")
            final_kept = res.get("final_kept") or []
            deleted = res.get("deleted") or []
            per_instance[inst] = {"kept_labels": final_kept, "deleted_labels": deleted}
            recs = res.get("deleted_records") or []
            if deleted and recs:
                deleted_records_by_instance[inst] = recs

        for res in results:
            inst = res.get("instance_name")
            relabel_ops = res.get("relabels") or []
            for old, new in relabel_ops:
                try:
                    traj_pool_manager.relabel(
                        old,
                        new,
                        instance_name=inst,
                        operator_name=self.get_name(),
                        delete_old=False,
                    )
                except Exception:
                    pass
            deleted = res.get("deleted") or []

            # 仅当没有重标签策略时才删除
            has_relabel_strategy = bool(relabel_as) or bool(relabel_map)
            if deleted and not has_relabel_strategy:
                try:
                    traj_pool_manager.delete_trajectories(deleted, instance_name=inst)
                except Exception:
                    pass

        filtered_out_file = None
        out_path = Path(workspace_dir) / "filtered_out.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json

            payload = {"deleted": deleted_records_by_instance, "per_instance": per_instance}
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, ensure_ascii=False, indent=2)
            filtered_out_file = out_path
        except Exception:
            pass

        return {
            "per_instance": per_instance,
            "filtered_out_file": str(filtered_out_file) if filtered_out_file else None,
        }


# 注册算子
from .registry import register_operator

register_operator("filter_trajectories", FilterTrajectoriesOperator)
