"""
轨迹记录系统

模仿 sweagent 的轨迹记录方式，记录优化过程中的所有步骤和结果。
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils.log import get_se_logger


@dataclass
class OptimizationStep:
    """单个优化步骤的记录"""

    step_id: str
    timestamp: str
    action: str  # initial_evaluation, generate_optimization
    # 新增：多轮对话相关与融合后的字段
    query: str | None = None
    response: str | None = None
    thought: str | None = None
    code_snapshot: str | None = None
    code_changed: bool | None = None
    diff: str | None = None
    performance_metrics: dict[str, Any] | None = None
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class TrajectoryMetadata:
    """轨迹元数据"""

    instance_id: str
    start_time: str
    language: str | None = None
    optimization_target: str | None = None
    end_time: str | None = None
    total_iterations: int = 0
    success: bool = False
    final_performance: dict[str, Any] | None = None
    error_message: str | None = None
    final_best_code: str | None = None


class TrajectoryLogger:
    """轨迹记录器"""

    def __init__(
        self,
        instance_id: str,
        trajectory_dir: str = "./trajectories",
        log_dir: Path | None = None,
    ):
        """初始化轨迹记录器

        - 日志统一写入 `log_dir/perfagent.log`，同时输出到终端（由 get_se_logger 控制）
        - 若未提供 log_dir，则退回到 `trajectory_dir/{instance_id}.log` 以保持兼容
        """
        self.metadata = TrajectoryMetadata(instance_id=instance_id, start_time=datetime.now().isoformat())
        self.steps: list[OptimizationStep] = []
        self.history: list[dict[str, Any]] = []
        self.trajectory_dir = trajectory_dir

        # 确保目录存在
        os.makedirs(trajectory_dir, exist_ok=True)

        # 设置日志：优先使用统一 logs 目录
        if log_dir is not None:
            log_file = Path(log_dir) / "perfagent.log"
        else:
            # 兼容：未提供 log_dir 时，按实例单独写入轨迹目录
            log_file = Path(trajectory_dir) / f"{instance_id}.log"

        # 轨迹日志器使用唯一名称，并关闭终端输出，避免与实例日志竞争
        traj_logger_name = f"perfagent.trajectory.{instance_id}"
        self.logger = get_se_logger(traj_logger_name, log_file, also_stream=False)

    def add_history(self, role: str, content: str, message_type: str, agent: str | None = None) -> None:
        """记录对话历史（multi-turn chat）"""
        entry: dict[str, Any] = {"role": role, "content": content, "message_type": message_type}
        if agent:
            entry["agent"] = agent
        self.history.append(entry)

    def start_step(self, action: str, query: str | None = None, code_snapshot: str | None = None) -> str:
        """开始一个新的优化步骤"""
        step_id = f"step_{len(self.steps) + 1}"
        timestamp = datetime.now().isoformat()

        step = OptimizationStep(
            step_id=step_id,
            timestamp=timestamp,
            action=action,
            query=query,
            code_snapshot=code_snapshot,
        )

        self.steps.append(step)
        # 将用户查询记录入历史
        if query:
            self.add_history(role="user", content=query, message_type="query")
        return step_id

    def end_step(
        self,
        step_id: str,
        response: str | None = None,
        thought: str | None = None,
        code_changed: bool | None = None,
        diff: str | None = None,
        performance_metrics: dict[str, Any] | None = None,
        error: str | None = None,
        code_snapshot: str | None = None,
        summary: str | None = None,
    ) -> None:
        """结束当前步骤并记录结果"""
        # 找到对应的步骤
        step = None
        for s in self.steps:
            if s.step_id == step_id:
                step = s
                break

        if not step:
            self.logger.error(f"未找到步骤 {step_id}")
            return

        step.response = response
        step.thought = thought
        step.code_changed = code_changed
        step.diff = diff
        step.performance_metrics = performance_metrics
        step.error = error
        if code_snapshot is not None:
            step.code_snapshot = code_snapshot
        start_time = datetime.fromisoformat(step.timestamp)
        step.execution_time = time.time() - start_time.timestamp()

        # 将助手动作记录入历史
        if response is not None:
            self.add_history(role="assistant", content=response, message_type="action")

        # 摘要信息（由 agent 格式化并传入）
        if summary:
            self.add_history(role="user", content=summary, message_type="observation")

        if error:
            self.logger.error(f"步骤 {step_id} 失败: {error}")
        else:
            self.logger.info(f"步骤 {step_id} 完成，耗时 {step.execution_time:.2f}s")

        # 实时保存轨迹
        self.save_trajectory()

    def finalize(
        self,
        success: bool = True,
        final_performance: dict[str, Any] | None = None,
        error_message: str | None = None,
        final_submission_code: str | None = None,
    ) -> str:
        """完成轨迹记录"""
        self.metadata.end_time = datetime.now().isoformat()
        self.metadata.total_iterations = len(self.steps)
        self.metadata.success = success
        self.metadata.final_performance = final_performance
        self.metadata.error_message = error_message
        self.metadata.final_best_code = final_submission_code

        trajectory_file = self.save_trajectory()  # 调用 save_trajectory 而不是 _save_trajectory

        if success:
            self.logger.info(f"轨迹记录完成，共 {len(self.steps)} 步")
        else:
            self.logger.error(f"轨迹记录完成（失败），错误: {error_message}")

        return trajectory_file

    def save_trajectory(self) -> str:
        """保存轨迹到文件"""
        try:
            # 确保所有数据都可以 JSON 序列化
            def make_serializable(obj):
                """递归地将对象转换为可序列化的格式（鲁棒降级）。"""
                try:
                    # 基本类型（含原生布尔）
                    if obj is None or isinstance(obj, (str, int, float, bool)):
                        if isinstance(obj, float):
                            if math.isfinite(obj):
                                return obj
                            if math.isnan(obj):
                                return "NaN"
                            return "-Infinity" if obj < 0 else "Infinity"
                        return obj

                    # numpy/scalar 等：带 item() 的标量
                    item_fn = getattr(obj, "item", None)
                    if callable(item_fn):
                        try:
                            return make_serializable(item_fn())
                        except Exception:
                            pass

                    # 路径对象
                    if hasattr(obj, "__fspath__"):
                        return str(obj)

                    # datetime
                    from datetime import datetime as _dt

                    if isinstance(obj, _dt):
                        return obj.isoformat()

                    # 容器类型
                    if isinstance(obj, dict):
                        return {str(k): make_serializable(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [make_serializable(item) for item in obj]

                    # 具有 __dict__ 的一般对象
                    if hasattr(obj, "__dict__"):
                        return {k: make_serializable(v) for k, v in obj.__dict__.items()}

                    # 兜底：转字符串
                    return str(obj)
                except Exception:
                    try:
                        return str(obj)
                    except Exception:
                        return "<unserializable>"

            # 计算最终提交代码（submission）：优先使用 final_best_code，其次选择最后一个包含 code_snapshot 的步骤
            final_code_snapshot = None
            override_code = getattr(self.metadata, "final_best_code", None)
            if override_code:
                final_code_snapshot = override_code
            else:
                for s in reversed(self.steps):
                    cs = getattr(s, "code_snapshot", None)
                    if cs:
                        final_code_snapshot = cs
                        break

            info_dict = make_serializable(asdict(self.metadata))
            # 与 sweagent 兼容：在 info 中加入 submission 字段（完整代码而非 diff）
            info_dict["submission"] = final_code_snapshot or ""

            # 序列化所有数据
            trajectory_data = {
                "info": info_dict,
                "steps": [make_serializable(asdict(step)) for step in self.steps],
                "history": make_serializable(self.history),
            }

            # 输出为 <trajectory_dir>/<instance_id>.traj，便于兼容 sweagent 的提取脚本
            trajectory_file = os.path.join(self.trajectory_dir, f"{self.metadata.instance_id}.traj")

            with open(trajectory_file, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"轨迹已保存到 {trajectory_file}")
            return trajectory_file

        except Exception as e:
            self.logger.error(f"保存轨迹文件失败: {e}")
            raise

    def _calculate_total_duration(self) -> float:
        """计算总持续时间"""
        if not self.metadata.end_time:
            return 0.0

        start = datetime.fromisoformat(self.metadata.start_time)
        end = datetime.fromisoformat(self.metadata.end_time)
        return (end - start).total_seconds()

    @classmethod
    def load_trajectory(cls, trajectory_file: Path) -> "TrajectoryLogger":
        """从文件加载轨迹（兼容旧版metadata/steps结构）"""
        with open(trajectory_file, encoding="utf-8") as f:
            data = json.load(f)

        # 重建轨迹记录器
        info = data.get("info") or data.get("metadata") or {}
        instance_id = info.get("instance_id", "unknown")
        logger = cls(instance_id=instance_id, trajectory_dir=str(trajectory_file.parent))

        # 恢复元数据（兼容旧键名）
        logger.metadata = TrajectoryMetadata(**info)

        # 恢复步骤（兼容旧版字段）
        steps_data = data.get("steps", [])
        restored_steps: list[OptimizationStep] = []
        for step_data in steps_data:
            # 旧版可能包含 input_data/output_data/duration 字段
            # 将其映射到新字段
            restored = OptimizationStep(
                step_id=str(step_data.get("step_id", step_data.get("id", ""))),
                timestamp=step_data.get("timestamp", datetime.now().isoformat()),
                action=step_data.get("action", ""),
                query=step_data.get("query"),
                response=step_data.get("response") or step_data.get("output_data", {}).get("response"),
                thought=step_data.get("thought"),
                code_snapshot=step_data.get("code_snapshot"),
                code_changed=step_data.get("code_changed") or step_data.get("output_data", {}).get("code_changed"),
                diff=step_data.get("diff") or step_data.get("output_data", {}).get("diff"),
                performance_metrics=step_data.get("performance_metrics"),
                error=step_data.get("error") or step_data.get("output_data", {}).get("error"),
                execution_time=step_data.get("execution_time") or step_data.get("duration", 0.0),
            )
            restored_steps.append(restored)
        logger.steps = restored_steps

        # 历史（可能不存在于旧版）
        logger.history = data.get("history", [])

        return logger

    def get_trajectory_summary(self) -> dict[str, Any]:
        """获取轨迹摘要（兼容新结构与旧结构）"""
        successful_steps = [step for step in self.steps if not step.error]
        failed_steps = [step for step in self.steps if step.error]

        performance_improvements = []
        for step in self.steps:
            if step.performance_metrics:
                mm = step.performance_metrics.get("trimmed_mean")
                if mm is None:
                    perf = step.performance_metrics.get("performance_analysis", {})
                    tgt = self.metadata.optimization_target or "runtime"
                    an = perf.get("analysis", {})
                    mm = an.get(tgt, {}).get("trimmed_mean")
                if mm is not None:
                    performance_improvements.append(mm)

        performance_trend = "stable"
        if len(performance_improvements) >= 2:
            if performance_improvements[-1] < performance_improvements[0]:
                performance_trend = "improving"
            elif performance_improvements[-1] > performance_improvements[0]:
                performance_trend = "degrading"
        elif len(successful_steps) > len(failed_steps):
            performance_trend = "improving"

        return {
            "instance_id": self.metadata.instance_id,
            "total_steps": len(self.steps),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "performance_trend": performance_trend,
            "final_success": self.metadata.success,
            "total_duration": self._calculate_total_duration(),
        }
