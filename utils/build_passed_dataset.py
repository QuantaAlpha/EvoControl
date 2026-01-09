import argparse
import json
from pathlib import Path


def build_passed(src_stats: Path, src_dataset: Path, out_dir: Path, lang: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(src_stats.read_text())
    copied = 0
    missing: list[str] = []
    for problem, langs in stats.items():
        if not isinstance(langs, dict):
            continue
        if lang:
            # 只检查指定语言是否通过
            rec = langs.get(lang)
            passed = isinstance(rec, dict) and rec.get("passed")
        else:
            # 任意语言通过即可
            passed = any(isinstance(rec, dict) and rec.get("passed") for rec in langs.values())
        if not passed:
            continue
        src_file = src_dataset / f"{problem}.json"
        if src_file.exists():
            out_file = out_dir / f"{problem}.json"
            out_file.write_text(src_file.read_text())
            copied += 1
        else:
            missing.append(str(src_file))
    print(f"copied={copied} out_dir={out_dir}")
    if missing:
        print("missing_count=", len(missing))
        for m in missing[:50]:
            print("missing:", m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/CodeEfficiency/benchmark/EffiBench-X/data/dataset"),
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--lang", type=str, help="Filter by language (e.g., python3, cpp)")
    args = p.parse_args()
    # 输出路径：如果指定了语言，则在路径中加上语言后缀
    suffix = f"_{args.lang}" if args.lang else ""
    out = args.out or args.dataset.parent / "dataset_passed" / f"{args.stats.stem}{suffix}"
    build_passed(args.stats, args.dataset, out, args.lang)


if __name__ == "__main__":
    main()
