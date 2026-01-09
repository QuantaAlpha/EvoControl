#!/usr/bin/env python3

"""
Trajectory Pool Summary Operator

分析轨迹池中的全部历史尝试，综合其优点与失败模式，
生成新的 Python 初始代码（仅代码），用于下一次迭代评估。
"""

import json
import textwrap
from pathlib import Path
from typing import Any

import yaml

from operators.base import TemplateOperator


class TrajPoolSummaryOperator(TemplateOperator):
    """轨迹池总结算子：综合历史轨迹，生成新的 Python 初始代码"""

    def get_name(self) -> str:
        return "traj_pool_summary"

    def get_strategy_prefix(self) -> str:
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("traj_pool_summary", {}) if isinstance(pcfg, dict) else {}
        return str(opcfg.get("prefix") or pcfg.get("traj_pool_summary_prefix") or "RISK-AWARE PROBLEM SOLVING GUIDANCE")

    def _discover_instances(self, workspace_dir: Path, current_iteration: int) -> list[dict[str, Any]]:
        """
        重写实例发现逻辑，直接查找工作目录中的traj.pool文件

        Args:
            workspace_dir: 工作目录路径
            current_iteration: 当前迭代号

        Returns:
            实例信息列表
        """
        instances = []

        # 通过父类方法加载 traj.pool 数据映射
        pool_data = self._load_traj_pool(workspace_dir)
        if not pool_data:
            return instances

        # 为每个实例创建实例信息
        for instance_name, instance_data in pool_data.items():
            if isinstance(instance_data, dict) and len(instance_data) > 0:
                # 检查是否有数字键（迭代数据）
                has_iteration_data = any(key.isdigit() for key in instance_data.keys())
                if has_iteration_data:
                    instances.append(
                        {
                            "instance_name": instance_name,
                            "instance_dir": workspace_dir,  # 使用工作目录作为实例目录
                            "trajectory_file": workspace_dir / "traj.pool",  # 指向 traj.pool 文件
                            "previous_iteration": current_iteration - 1,
                            "pool_data": instance_data,  # 附加池数据用于后续处理
                            "problem_description": instance_data.get("problem", {}),
                        }
                    )

        self.logger.info(f"发现 {len(instances)} 个可处理的实例")
        return instances

    def _extract_problem_statement(self, trajectory_data: dict[str, Any]) -> str:
        """
        重写问题陈述提取，返回占位符
        因为我们在_generate_content中直接使用pool_data中的问题陈述
        """
        return "placeholder"

    # 移除本地加载方法，统一由父类提供 _load_traj_pool

    def _format_approaches_data(self, approaches_data: dict[str, Any]) -> str:
        """格式化历史尝试数据为通用的嵌套文本结构。

        - 保留字典键原始顺序
        - 两空格缩进层级
        - 列表项以 "- " 前缀展示
        - 多行字符串块式缩进
        - 跳过空值/空列表/空字典
        """

        def _fmt(value: Any, indent: int) -> str:
            prefix = "  " * indent
            if isinstance(value, dict):
                lines: list[str] = []
                for k, v in value.items():
                    if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
                        continue
                    if isinstance(v, (int, float)):
                        lines.append(f"{prefix}{k}: {v}")
                    elif isinstance(v, bool):
                        lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
                    elif isinstance(v, str) and "\n" not in v:
                        lines.append(f"{prefix}{k}: {v}")
                    else:
                        lines.append(f"{prefix}{k}:")
                        child = _fmt(v, indent + 1)
                        if child:
                            lines.append(child)
                return "\n".join(lines)
            if isinstance(value, list):
                lines: list[str] = []
                for item in value:
                    if item is None or item == "":
                        continue
                    if isinstance(item, (int, float)):
                        lines.append(f"{prefix}- {item}")
                    elif isinstance(item, bool):
                        lines.append(f"{prefix}- {'true' if item else 'false'}")
                    elif isinstance(item, str) and "\n" not in item:
                        lines.append(f"{prefix}- {item}")
                    else:
                        child = _fmt(item, indent + 1)
                        if child:
                            child_lines = child.splitlines()
                            if child_lines:
                                child_prefix = "  " * (indent + 1)
                                first = child_lines[0]
                                if first.startswith(child_prefix):
                                    first = first[len(child_prefix) :]
                                lines.append(f"{prefix}- {first}")
                                for cl in child_lines[1:]:
                                    if cl.startswith(child_prefix):
                                        cl = cl[len(child_prefix) :]
                                    lines.append(f"{prefix}  {cl}")
                return "\n".join(lines)
            if isinstance(value, str):
                if "\n" in value:
                    return textwrap.indent(value, "  " * indent)
                return f"{prefix}{value}"
            if isinstance(value, bool):
                return f"{prefix}{'true' if value else 'false'}"
            if isinstance(value, (int, float)):
                return f"{prefix}{value}"
            return f"{prefix}{str(value)}"

        parts: list[str] = []
        for key, data in approaches_data.items():
            if key == "problem":
                continue
            if isinstance(data, dict):
                parts.append(f"ATTEMPT {key}:")
                body = _fmt(data, 0)
                if body:
                    parts.append(body)
        return "\n".join(parts)

    def _build_additional_requirements(self, problem_statement: str, approaches_data: dict[str, Any]) -> str:
        formatted_attempts = self._format_approaches_data(approaches_data)
        pcfg = self.config.get("prompt_config", {}) if isinstance(self.config, dict) else {}
        opcfg = pcfg.get("traj_pool_summary", {}) if isinstance(pcfg, dict) else {}
        header = str(
            opcfg.get("header") or pcfg.get("traj_pool_summary_header") or "RISK-AWARE PROBLEM SOLVING GUIDANCE"
        )
        prob = textwrap.indent(str(problem_statement).strip(), "  ")
        attempts = textwrap.indent(formatted_attempts.strip(), "  ")
        body = (
            opcfg.get("guidance")
            or pcfg.get("traj_pool_summary_guidance")
            or (
                "Guidance:\n"
                "1. Identify and avoid previously failed techniques and blind spots.\n"
                "2. Prefer robust alternatives with clearer performance characteristics.\n"
                "3. Integrate proven components across attempts when helpful.\n"
                "4. Keep code simple, correct, and maintainable.\n"
            )
        )
        return f"{header}\n\nPROBLEM:\n{prob}\n\nHISTORY COMPONENTS:\n{attempts}\n\n{body}".strip()

    def _generate_content(
        self, instance_info: dict[str, Any], problem_statement: str, trajectory_data: dict[str, Any]
    ) -> str:
        """生成轨迹池总结内容"""
        instance_name = instance_info["instance_name"]

        # 直接使用附加的池数据
        approaches_data = instance_info.get("pool_data", {})
        if not approaches_data:
            self.logger.warning(f"跳过 {instance_name}: 无轨迹池数据")
            return ""

        # 使用实例信息中的问题陈述，若为字典则转为简字符串
        pool_problem_raw = instance_info.get("problem_description", "N/A")
        try:
            pool_problem = (
                json.dumps(pool_problem_raw, ensure_ascii=False, indent=2)
                if isinstance(pool_problem_raw, (dict, list))
                else str(pool_problem_raw)
            )
        except Exception:
            pool_problem = str(pool_problem_raw)

        # 获取所有迭代数据（数字键）
        iteration_data = {k: v for k, v in approaches_data.items() if k.isdigit() and isinstance(v, dict)}

        if not iteration_data:
            self.logger.warning(f"跳过 {instance_name}: 无有效迭代数据")
            return ""

        self.logger.info(f"分析 {instance_name}: {len(iteration_data)} 个历史尝试")

        return self._build_additional_requirements(pool_problem, iteration_data)

    def run(
        self,
        step_config: dict[str, Any],
        traj_pool_manager,
        workspace_dir: str,
    ) -> dict[str, Any]:
        all_instances = traj_pool_manager.get_all_trajectories()
        output_dir = Path(workspace_dir) / "system_prompt"
        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for instance_name, instance_data in (all_instances or {}).items():
            try:
                if not isinstance(instance_data, dict):
                    continue
                problem_statement = instance_data.get("problem")
                approaches_data = {k: v for k, v in instance_data.items() if k != "problem" and isinstance(v, dict)}
                if not problem_statement or not approaches_data:
                    continue
                # 使用通用最新格式化作为融合提示的起点（但综合函数仍基于全部数据）
                latest_text = self._format_entry(approaches_data)
                content = self._build_additional_requirements(str(problem_statement), approaches_data)
                if not content:
                    continue
                data = {"prompts": {"additional_requirements": content}}
                file_path = output_dir / f"{instance_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                written += 1
            except Exception:
                continue

        return {"instance_templates_dir": str(output_dir), "generated_count": written}


# 注册算子
from .registry import register_operator

register_operator("traj_pool_summary", TrajPoolSummaryOperator)
