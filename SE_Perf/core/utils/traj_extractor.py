#!/usr/bin/env python3
"""
轨迹数据提取器
从SWE-agent输出目录中提取实例数据，使用统一的Instance数据管理接口
"""

from pathlib import Path
from typing import Any

from core.utils.instance_data_manager import InstanceData, get_instance_data_manager
from core.utils.se_logger import get_se_logger


class TrajExtractor:
    """轨迹数据提取器 - 基于统一Instance数据管理"""

    def __init__(self):
        self.logger = get_se_logger("traj_extractor", emoji="📁")
        self.instance_manager = get_instance_data_manager()

    def _load_perf_metrics(self, instance_dir: Path) -> dict[str, Any] | None:
        """
        读取实例目录下的 result.json，提取关键性能指标。

        返回一个精简的字典，仅包含常用字段；若不存在或解析失败，返回None。
        """
        try:
            result_file = instance_dir / "result.json"
            if not result_file.exists():
                return None
            import json

            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)

            # 仅选择常用的字段，避免traj.pool体积膨胀
            keys = [
                "performance_before",
                "performance_after",
                "improvement",
                "success",
                "final_performance",
                "instance_id",
                "trajectory_file",
            ]
            metrics: dict[str, Any] = {}
            for k in keys:
                if k in data:
                    metrics[k] = data[k]

            # 兼容额外可能字段（例如运行时间、评估均值等）
            for k in ["runtime", "trimmed_mean", "optimized_code_len"]:
                if k in data:
                    metrics[k] = data[k]

            # 若无有效字段则返回None，避免冗余
            return metrics if metrics else None
        except Exception as e:
            self.logger.warning(f"读取性能指标失败: {instance_dir}/result.json: {e}")
            return None

    def extract_instance_data(
        self, iteration_dir: str, include_metrics: bool = False
    ) -> list[tuple[str, str | None, str, str]]:
        """
        从迭代目录中提取所有实例的数据

        Args:
            iteration_dir: 迭代目录路径
            include_metrics: 是否附带性能指标（来自result.json）

        Returns:
            List[Tuple[instance_name, problem_description, tra_content, patch_content]]
            如果 include_metrics=True，则额外在元组末尾附加 perf_metrics 字典

        Note:
            返回格式保持向后兼容，实际推荐使用extract_instances_structured()
        """
        instances = self.instance_manager.get_iteration_instances(iteration_dir)
        results = []

        for instance in instances:
            if instance.tra_content:
                # 如果有.tra文件，就包含这个实例（即使没有.patch文件）
                patch_content = instance.patch_content or "FAILED_NO_PATCH"
                if include_metrics:
                    perf_metrics = {
                        "passed": instance.passed,
                        "performance": instance.final_performance,
                        "performance_unit": instance.performance_unit,
                        "optimization_target": instance.optimization_target,
                        "language": instance.language,
                        "artifacts": instance.final_artifacts,
                    }
                    results.append(
                        (
                            instance.instance_name,
                            instance.problem_description,
                            instance.tra_content,
                            patch_content,
                            perf_metrics,
                        )
                    )
                else:
                    results.append(
                        (instance.instance_name, instance.problem_description, instance.tra_content, patch_content)
                    )
            else:
                # 没有.tra文件的实例才跳过
                self.logger.warning(f"实例 {instance.instance_name} 缺少.tra文件，跳过")

        if include_metrics:
            self.logger.info(f"从 {iteration_dir} 提取了 {len(results)} 个实例数据（含性能指标）")
        else:
            self.logger.info(f"从 {iteration_dir} 提取了 {len(results)} 个实例数据（包括失败实例）")
        return results

    def extract_instances_structured(self, iteration_dir: str) -> list[InstanceData]:
        """
        推荐的新接口：提取结构化的实例数据

        Args:
            iteration_dir: 迭代目录路径

        Returns:
            InstanceData对象列表
        """
        return self.instance_manager.get_iteration_instances(iteration_dir)

    def get_instance_completeness_report(self, iteration_dir: str) -> dict:
        """
        生成迭代目录中所有实例的完整性报告

        Args:
            iteration_dir: 迭代目录路径

        Returns:
            完整性报告字典
        """
        instances = self.instance_manager.get_iteration_instances(iteration_dir)

        report = {
            "total_instances": len(instances),
            "complete_instances": 0,
            "incomplete_instances": [],
            "file_availability": {"problem": 0, "tra": 0, "traj": 0, "patch": 0},
            "instances_detail": [],
        }

        for instance in instances:
            validation = self.instance_manager.validate_instance_completeness(instance)
            report["instances_detail"].append(validation)

            if validation["completeness_score"] == 100:
                report["complete_instances"] += 1
            else:
                report["incomplete_instances"].append(
                    {
                        "name": instance.instance_name,
                        "missing": validation["missing_data"],
                        "score": validation["completeness_score"],
                    }
                )

            # 统计文件可用性
            if validation["has_problem"]:
                report["file_availability"]["problem"] += 1
            if validation["has_tra"]:
                report["file_availability"]["tra"] += 1
            if validation["has_traj"]:
                report["file_availability"]["traj"] += 1
            if validation["has_patch"]:
                report["file_availability"]["patch"] += 1

        return report
