"""
PerfAgent - 代码效率优化 Agent

基于 SE-Agent 框架构建的代码性能优化工具，模仿 sweagent 的设计模式。
"""

from .agent import PerfAgent
from .config import PerfAgentConfig
from .trajectory import TrajectoryLogger

__all__ = ["PerfAgent", "PerfAgentConfig", "TrajectoryLogger"]
