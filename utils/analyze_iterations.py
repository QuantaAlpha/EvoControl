#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any


def to_float(rt: Any) -> float:
    try:
        if rt is None:
            return float("inf")
        if isinstance(rt, (int, float)):
            return float(rt)
        if isinstance(rt, str):
            s = rt.strip().lower()
            if s in ("inf", "infinity", "nan"):
                return float("inf")
            return float(rt)
        return float("inf")
    except Exception:
        return float("inf")


def compute_metrics(entries: list[tuple[int, float]], Ks: list[int]) -> dict[int, dict[str, Any]]:
    entries.sort(key=lambda x: x[0])

    metrics: dict[int, dict[str, Any]] = {}
    for K in Ks:
        vals = [rt for k, rt in entries if k <= K]
        if not vals:
            metrics[K] = {
                "best": float("inf"),
                "pass": {
                    "finite": 0,
                    "total": 0,
                    "rate": 0.0,
                },
            }
            continue
        finite = sum(1 for rt in vals if math.isfinite(rt))
        best = min(vals)
        metrics[K] = {
            "best": best,
            "pass": {
                "finite": finite,
                "total": len(vals),
                "rate": (finite / len(vals)) if len(vals) > 0 else 0.0,
            },
        }
    has_finite = False
    for K in sorted(Ks):
        finite_cnt = metrics[K]["pass"]["finite"]
        if finite_cnt > 0:
            has_finite = True
        metrics[K]["best_pass_rate_upto"] = 1.0 if has_finite else 0.0
    return metrics


def sanitize_for_json(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize_for_json(v) for v in x]
    return x


def aggregate_limits(data: dict[str, Any], Ks: list[int]) -> dict[int, dict[str, float]]:
    agg: dict[int, dict[str, float]] = {}
    for K in Ks:
        agg[K] = {
            "pass_rate_mean": 0.0,
            "best_pass_rate_upto_mean": 0.0,
        }

    perK_pass_rates: dict[int, list[float]] = {K: [] for K in Ks}
    perK_best_pass_upto_rates: dict[int, list[float]] = {K: [] for K in Ks}

    for inst, info in data.items():
        sname = inst.strip().lower()
        if ("system_prompt" in sname) or ("system prompt" in sname):
            continue
        iter_map = info.get("iteration", {})
        entries: list[tuple[int, float]] = []
        for k_str, v in iter_map.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            rt = to_float(v.get("runtime"))
            entries.append((k, rt))
        if not entries:
            continue
        metrics = compute_metrics(entries, Ks)
        for K in Ks:
            rate = metrics[K]["pass"]["rate"]
            best_upto = metrics[K]["best_pass_rate_upto"]
            perK_pass_rates[K].append(rate)
            perK_best_pass_upto_rates[K].append(best_upto)

    for K in Ks:
        rates = perK_pass_rates[K]
        rates_upto = perK_best_pass_upto_rates[K]
        mean_rate = (sum(rates) / len(rates)) if rates else 0.0
        mean_rate_upto = (sum(rates_upto) / len(rates_upto)) if rates_upto else 0.0
        agg[K] = {
            "pass_rate_mean": mean_rate,
            "best_pass_rate_upto_mean": mean_rate_upto,
        }
    return agg


def main():
    parser = argparse.ArgumentParser(description="进化轮数分析：每个K下的best与pass")
    parser.add_argument("input", help="all_hist.json 路径")
    parser.add_argument("--k", type=int, default=5, help="间隔步长")
    parser.add_argument("--output", help="可选，输出汇总JSON路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    k_step = args.k
    max_iter = 0
    for inst, info in data.items():
        sname = inst.strip().lower()
        if ("system_prompt" in sname) or ("system prompt" in sname):
            continue
        iter_map = info.get("iteration", {})
        for k_str in iter_map.keys():
            try:
                kk = int(k_str)
            except Exception:
                continue
            if kk > max_iter:
                max_iter = kk
    Ks = list(range(k_step, max_iter + 1, k_step))
    if max_iter > 0 and (not Ks or Ks[-1] != max_iter):
        Ks.append(max_iter)

    instance_rows: list[dict[str, Any]] = []
    for inst, info in data.items():
        sname = inst.strip().lower()
        if ("system_prompt" in sname) or ("system prompt" in sname):
            continue
        iter_map = info.get("iteration", {})
        entries: list[tuple[int, float]] = []
        for k_str, v in iter_map.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            rt = to_float(v.get("runtime"))
            entries.append((k, rt))
        if not entries:
            continue
        metrics = compute_metrics(entries, Ks)
        instance_rows.append(
            {
                "instance": inst,
                "metrics": metrics,
            }
        )

    agg = aggregate_limits(data, Ks)

    summary = {
        "Ks": Ks,
        "aggregate": agg,
        "instances": instance_rows,
    }

    if args.output:
        cleaned = sanitize_for_json(summary)
        Path(args.output).write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    # 打印简要摘要
    print("=== 轮数上限聚合摘要 ===")
    for K in Ks:
        pr = summary["aggregate"][K]["pass_rate_mean"]
        pr_upto = summary["aggregate"][K]["best_pass_rate_upto_mean"]
        print(f"K={K}: pass_rate_mean={pr:.2%}, best_pass_rate_upto_mean={pr_upto:.2%}")

    # 列出前若干实例用于快速目视检查
    print("\n=== 示例实例指标（前10条） ===")
    for row in instance_rows[:10]:
        inst = row["instance"]
        parts = []
        for K in Ks:
            m = row["metrics"][K]
            b = m["best"]
            pr = m["pass"]["rate"]
            pr_upto = m["best_pass_rate_upto"]
            b_str = "inf" if not math.isfinite(b) else f"{b:.4f}"
            parts.append(f"K={K}: best={b_str}, pass={pr:.2%}, pass_upto={pr_upto:.2%}")
        print(f"- {inst}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
