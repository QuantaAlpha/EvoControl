#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import pytest

# 添加SE目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import BaseOperator


class DummyOperator(BaseOperator):
    def get_name(self) -> str:
        return "dummy"

    def run(self, step_config, traj_pool_manager, workspace_dir):
        return {}


@pytest.fixture
def op():
    random.seed(42)
    return DummyOperator({"operator_models": {}})


@pytest.fixture
def entry():
    return {
        "problem": "p",
        "A": {"label": "A", "performance": 1.0},
        "B": {"label": "B", "performance": 0.5},
        "C": {"label": "C", "performance": 10.0},
        "INF": {"label": "INF", "performance": float("inf")},
    }


def test_equal_inputs_use_direct(op, entry):
    sc = {"inputs": [{"label": "B"}]}
    chosen = op._select_source_labels(entry, sc, required_n=1)
    assert chosen == ["B"]


def test_more_inputs_weighted_within_inputs(op, entry):
    sc = {"inputs": [{"label": "A"}, {"label": "B"}, {"label": "C"}]}
    counts = {}
    for _ in range(2000):
        chosen = tuple(op._select_source_labels(entry, sc, required_n=2))
        counts[chosen] = counts.get(chosen, 0) + 1
    # 较低performance的B参与的组合应更多
    assert sum(v for k, v in counts.items() if "B" in k) > sum(v for k, v in counts.items() if "B" not in k)


def test_less_inputs_fill_by_pool(op, entry):
    sc = {"inputs": [{"label": "A"}]}
    counts = {}
    for _ in range(2000):
        chosen = tuple(op._select_source_labels(entry, sc, required_n=2))
        counts[chosen] = counts.get(chosen, 0) + 1
    # B 应该经常被选为补齐项
    assert counts.get(("B", "A"), 0) + counts.get(("A", "B"), 0) > 500


def test_weighted_allowed_labels(op, entry):
    counts = {"A": 0, "B": 0}
    for _ in range(5000):
        lbls = op._weighted_select_labels(entry, k=1, allowed_labels=["A", "B"])
        counts[lbls[0]] += 1
    # B 的计数应显著大于 A
    assert counts["B"] > counts["A"]
