#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def collect_outputs(root_dir: str | Path, instance_names: list[str] | None = None) -> dict[str, str]:
    root = Path(root_dir)
    agg_pool: dict[str, dict] = {}
    agg_hist: dict[str, dict] = {}
    agg_final: dict[str, str] = {}

    if instance_names is None:
        names = [p.name for p in root.iterdir() if p.is_dir()]
    else:
        names = [str(n) for n in instance_names]

    for name in names:
        inst_root = root / name
        pool_path = inst_root / "traj.pool"
        preds_path = inst_root / "preds.json"
        final_path = inst_root / "final.json"

        pool_data: dict[str, dict] = {}
        try:
            if pool_path.exists():
                pool_data = json.loads(pool_path.read_text(encoding="utf-8")) or {}
                for k, v in pool_data.items():
                    agg_pool[k] = v
        except Exception:
            pass

        try:
            if preds_path.exists():
                preds = json.loads(preds_path.read_text(encoding="utf-8")) or {}
                for inst_id, entries in preds.items():
                    iteration_map: dict[str, dict] = {}
                    for e in entries or []:
                        it = e.get("iteration")
                        code = e.get("code") or ""
                        performance = e.get("performance")
                        metrics = e.get("final_metrics") or {}
                        if it is None:
                            continue
                        iteration_map[str(it)] = {"code": code, "performance": performance, "final_metrics": metrics}
                    problem = None
                    try:
                        pinfo = pool_data.get(inst_id) or {}
                        problem = pinfo.get("problem")
                    except Exception:
                        problem = None
                    agg_hist[inst_id] = {"problem": problem, "iteration": iteration_map}
        except Exception:
            pass

        try:
            if final_path.exists():
                fdata = json.loads(final_path.read_text(encoding="utf-8")) or {}
                for inst_id, code in fdata.items():
                    agg_final[inst_id] = code
        except Exception:
            pass

    root.mkdir(parents=True, exist_ok=True)
    out_pool = root / "traj.pool"
    out_hist = root / "all_hist.json"
    out_final = root / "final.json"

    try:
        out_pool.write_text(json.dumps(agg_pool, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"聚合 traj.pool: {out_pool}")
    except Exception:
        print("聚合 traj.pool 失败")

    try:
        out_hist.write_text(json.dumps(agg_hist, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"聚合 all_hist.json: {out_hist}")
    except Exception:
        print("聚合 all_hist.json 失败")

    try:
        out_final.write_text(json.dumps(agg_final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"聚合 final.json: {out_final}")
    except Exception:
        print("聚合 final.json 失败")

    return {"traj_pool": str(out_pool), "all_hist": str(out_hist), "final": str(out_final)}


def main():
    parser = argparse.ArgumentParser(description="聚合每个实例输出，写入根目录")
    parser.add_argument("--root-dir", required=True, help="实例根目录")
    parser.add_argument("--names", nargs="*", help="要聚合的实例名称，可选")
    args = parser.parse_args()

    names = args.names if args.names else None
    result = collect_outputs(args.root_dir, names)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
