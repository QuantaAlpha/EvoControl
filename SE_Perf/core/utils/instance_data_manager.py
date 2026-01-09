#!/usr/bin/env python3
"""
Instance数据管理器
为operator提供统一的实例数据获取接口，包括problem、tra、patch、traj_pool等核心数据
"""

import json
from pathlib import Path
from typing import Any

from core.utils.problem_manager import get_problem_description
from core.utils.se_logger import get_se_logger


class InstanceData:
    """单个实例的完整数据封装"""

    def __init__(self, instance_name: str, instance_path: str):
        self.instance_name = instance_name
        self.instance_path = Path(instance_path)

        # 核心数据
        self.problem_description: str | None = None
        self.tra_content: str | None = None  # 压缩后的轨迹
        self.traj_content: str | None = None  # 原始轨迹
        self.patch_content: str | None = None  # 预测结果(.pred或.patch)

        # 评估数据
        self.passed: bool | None = None
        self.final_performance: float | str | None = None
        self.final_artifacts: str | None = None

        # 运行元数据
        self.language: str | None = None
        self.optimization_target: str | None = None
        self.performance_unit: str | None = None

        # 元数据
        self.available_files: list[str] = []
        self.data_sources: dict[str, str] = {}

    def __repr__(self):
        return f"InstanceData(name='{self.instance_name}', path='{self.instance_path}')"


class InstanceDataManager:
    """Instance数据管理器 - 为operator提供统一的数据获取接口"""

    def __init__(self):
        self.logger = get_se_logger("instance_data", emoji="📦")

    def get_instance_data(self, instance_path: str, load_all: bool = True) -> InstanceData:
        """
        获取实例的完整数据

        Args:
            instance_path: 实例目录路径
            load_all: 是否立即加载所有数据，False则按需加载

        Returns:
            InstanceData对象
        """
        instance_path = Path(instance_path)
        instance_name = instance_path.name

        instance_data = InstanceData(instance_name, str(instance_path))

        # 扫描可用文件
        instance_data.available_files = self._scan_available_files(instance_path, instance_name)

        if load_all:
            # 立即加载所有数据
            instance_data.problem_description = self._load_problem_description(instance_path)
            instance_data.tra_content = self._load_tra_content(instance_path, instance_name)
            instance_data.traj_content = self._load_traj_content(instance_path, instance_name)
            instance_data.patch_content = self._load_patch_content(instance_path, instance_name)
            self._load_result_metrics(instance_path, instance_name, instance_data)

        return instance_data

    def get_iteration_instances(self, iteration_dir: str) -> list[InstanceData]:
        """
        获取整个迭代目录中所有实例的数据

        Args:
            iteration_dir: 迭代目录路径

        Returns:
            InstanceData对象列表
        """
        iteration_path = Path(iteration_dir)
        instances = []

        if not iteration_path.exists():
            self.logger.error(f"迭代目录不存在: {iteration_dir}")
            return instances

        for instance_path in iteration_path.iterdir():
            if instance_path.is_dir():
                instance_data = self.get_instance_data(str(instance_path))
                instances.append(instance_data)

        self.logger.info(f"从 {iteration_dir} 获取了 {len(instances)} 个实例数据")
        return instances

    def get_traj_pool_data(self, traj_pool_path: str, instance_name: str) -> dict[str, Any] | None:
        """
        从轨迹池中获取特定实例的数据

        Args:
            traj_pool_path: traj.pool文件路径
            instance_name: 实例名称

        Returns:
            实例在轨迹池中的完整数据，包括problem和所有迭代总结
        """
        try:
            with open(traj_pool_path, encoding="utf-8") as f:
                pool_data = json.load(f)

            instance_pool_data = pool_data.get(instance_name)
            if instance_pool_data:
                self.logger.debug(f"从轨迹池获取实例数据: {instance_name}")
                return instance_pool_data
            else:
                self.logger.warning(f"轨迹池中未找到实例: {instance_name}")
                return None

        except Exception as e:
            self.logger.error(f"读取轨迹池失败: {e}")
            return None

    def get_instance_iteration_summary(
        self, traj_pool_path: str, instance_name: str, iteration: int | str
    ) -> dict[str, Any] | None:
        """
        获取实例特定迭代的总结数据

        Args:
            traj_pool_path: traj.pool文件路径
            instance_name: 实例名称
            iteration: 迭代编号

        Returns:
            特定迭代的总结数据
        """
        pool_data = self.get_traj_pool_data(traj_pool_path, instance_name)
        if pool_data:
            iteration_key = str(iteration)
            if iteration_key in pool_data:
                return pool_data[iteration_key]
            else:
                self.logger.warning(f"实例 {instance_name} 未找到迭代 {iteration}")
        return None

    def validate_instance_completeness(self, instance_data: InstanceData) -> dict[str, Any]:
        """
        验证实例数据的完整性

        Args:
            instance_data: 实例数据对象

        Returns:
            验证结果字典
        """
        result = {
            "instance_name": instance_data.instance_name,
            "has_problem": instance_data.problem_description is not None,
            "has_tra": instance_data.tra_content is not None,
            "has_traj": instance_data.traj_content is not None,
            "has_patch": instance_data.patch_content is not None,
            "available_files": instance_data.available_files,
            "completeness_score": 0,
            "missing_data": [],
        }

        # 计算完整性分数
        core_data = ["has_problem", "has_tra", "has_patch"]
        available_count = sum(1 for key in core_data if result[key])
        result["completeness_score"] = (available_count / len(core_data)) * 100

        # 记录缺失数据
        data_mapping = {
            "has_problem": "problem_description",
            "has_tra": "tra_content",
            "has_traj": "traj_content",
            "has_patch": "patch_content",
        }

        for key, data_name in data_mapping.items():
            if not result[key]:
                result["missing_data"].append(data_name)

        return result

    def _scan_available_files(self, instance_path: Path, instance_name: str) -> list[str]:
        """扫描实例目录中的可用文件"""
        extensions = [".problem", ".tra", ".traj", ".pred", ".patch"]
        available = []

        for ext in extensions:
            file_path = instance_path / f"{instance_name}{ext}"
            if file_path.exists():
                available.append(ext[1:])  # 去掉点号

        return available

    def _load_problem_description(self, instance_path: Path) -> str | None:
        """加载问题描述"""
        try:
            return get_problem_description(str(instance_path))
        except Exception as e:
            self.logger.error(f"加载问题描述失败: {e}")
            return None

    def _load_tra_content(self, instance_path: Path, instance_name: str) -> str | None:
        """加载.tra文件内容"""
        tra_file = instance_path / f"{instance_name}.tra"
        return self._read_file_safe(tra_file)

    def _load_traj_content(self, instance_path: Path, instance_name: str) -> str | None:
        """加载.traj文件内容"""
        traj_file = instance_path / f"{instance_name}.traj"
        return self._read_file_safe(traj_file)

    def _load_patch_content(self, instance_path: Path, instance_name: str) -> str | None:
        """加载预测结果内容 - 优先.patch，备选.pred"""
        # 优先级：.patch > .pred
        for ext in [".patch", ".pred"]:
            file_path = instance_path / f"{instance_name}{ext}"
            content = self._read_file_safe(file_path)
            if content is not None:
                self.logger.debug(f"加载预测内容: {file_path}")
                return content

        self.logger.warning(f"未找到预测文件: {instance_path}/{instance_name}.[patch|pred]")
        return None

    def _load_result_metrics(self, instance_path: Path, instance_name: str, instance_data: InstanceData) -> None:
        result_file = instance_path / "result.json"
        if not result_file.exists():
            return
        try:
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        perf = data.get("final_performance")
        artifacts = data.get("final_artifacts")

        instance_data.final_performance = perf
        instance_data.final_artifacts = artifacts if isinstance(artifacts, str) else None

        lang = data.get("language")
        opt = data.get("optimization_target")
        unit = data.get("performance_unit")
        instance_data.language = lang if isinstance(lang, str) else None
        instance_data.optimization_target = opt if isinstance(opt, str) else None
        instance_data.performance_unit = unit if isinstance(unit, str) else None

        try:
            import math

            if isinstance(perf, (int, float)):
                instance_data.passed = math.isfinite(float(perf))
            elif isinstance(perf, str):
                s = perf.strip().lower()
                instance_data.passed = s not in ("inf", "+inf", "infinity", "+infinity")
            else:
                instance_data.passed = False
        except Exception:
            instance_data.passed = False

    def _read_file_safe(self, file_path: Path) -> str | None:
        """安全读取文件内容"""
        if not file_path.exists():
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # 对于过长的内容进行截断处理
            max_length = 50000  # 设置合理的最大长度
            if len(content) > max_length:
                self.logger.debug(f"文件内容被截断: {file_path} ({len(content)} -> {max_length})")
                content = content[:max_length]

            return content

        except Exception as e:
            self.logger.error(f"读取文件失败 {file_path}: {e}")
            return None


# 全局实例
_instance_data_manager = None


def get_instance_data_manager() -> InstanceDataManager:
    """获取全局Instance数据管理器实例"""
    global _instance_data_manager
    if _instance_data_manager is None:
        _instance_data_manager = InstanceDataManager()
    return _instance_data_manager


def get_instance_data(instance_path: str, load_all: bool = True) -> InstanceData:
    """便捷函数：获取实例数据"""
    return get_instance_data_manager().get_instance_data(instance_path, load_all)


def get_iteration_instances(iteration_dir: str) -> list[InstanceData]:
    """便捷函数：获取迭代实例列表"""
    return get_instance_data_manager().get_iteration_instances(iteration_dir)


def get_traj_pool_data(traj_pool_path: str, instance_name: str) -> dict[str, Any] | None:
    """便捷函数：获取轨迹池实例数据"""
    return get_instance_data_manager().get_traj_pool_data(traj_pool_path, instance_name)
