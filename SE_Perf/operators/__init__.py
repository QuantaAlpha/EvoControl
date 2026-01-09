#!/usr/bin/env python3

"""
SE Operators Package

算子系统的统一入口，提供算子注册和访问功能。
"""

from .alternative_strategy import AlternativeStrategyOperator
from .base import BaseOperator, EnhanceOperator, TemplateOperator
from .crossover import CrossoverOperator
from .filter import FilterTrajectoriesOperator
from .plan import PlanOperator
from .reflection_refine import ReflectionRefineOperator
from .registry import create_operator, get_operator_class, get_registry, list_operators, register_operator

# 导入具体算子实现
from .traj_pool_summary import TrajPoolSummaryOperator
from .trajectory_analyzer import TrajectoryAnalyzerOperator

# 后续导入其他算子实现
# from .conclusion import ConclusionOperator
# from .summary_bug import SummaryBugOperator

__all__ = [
    "BaseOperator",
    "TemplateOperator",
    "EnhanceOperator",
    "register_operator",
    "get_operator_class",
    "create_operator",
    "list_operators",
    "get_registry",
    "TrajPoolSummaryOperator",
    "AlternativeStrategyOperator",
    "TrajectoryAnalyzerOperator",
    "CrossoverOperator",
    "FilterTrajectoriesOperator",
    "PlanOperator",
    "ReflectionRefineOperator",
]
