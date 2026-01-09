#!/usr/bin/env python3

"""
SE Framework Utils Package

SE框架工具模块，提供日志管理、轨迹处理等核心功能。
"""

from .instance_data_manager import (
    InstanceData,
    InstanceDataManager,
    get_instance_data,
    get_instance_data_manager,
    get_iteration_instances,
    get_traj_pool_data,
)
from .llm_client import LLMClient, TrajectorySummarizer
from .local_memory_manager import LocalMemoryManager
from .problem_manager import ProblemManager, get_problem_description, get_problem_manager, validate_problem_availability
from .se_logger import get_se_logger, setup_se_logging
from .traj_extractor import TrajExtractor
from .traj_pool_manager import TrajPoolManager
from .traj_summarizer import TrajSummarizer
from .trajectory_processor import TrajectoryProcessor, extract_problems_from_workspace, process_trajectory_files

__all__ = [
    # 日志系统
    "setup_se_logging",
    "get_se_logger",
    # 轨迹处理
    "TrajectoryProcessor",
    "process_trajectory_files",
    "extract_problems_from_workspace",
    "TrajPoolManager",
    "TrajSummarizer",
    "TrajExtractor",
    # LLM集成
    "LLMClient",
    "TrajectorySummarizer",
    "LocalMemoryManager",
    # 问题管理 (统一接口)
    "ProblemManager",
    "get_problem_manager",
    "get_problem_description",
    "validate_problem_availability",
    # Instance数据管理 (统一数据流转)
    "InstanceData",
    "InstanceDataManager",
    "get_instance_data_manager",
    "get_instance_data",
    "get_iteration_instances",
    "get_traj_pool_data",
]
