import json
import argparse
from pathlib import Path


def generate_order(json_path: str, output_path: str, lang: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assume the first key is the model name
    model_name = list(data.keys())[0]
    per_task = data[model_name].get("per_task", {})

    passed_tasks = []
    failed_tasks = []

    for task_name, lang_data in per_task.items():
        if not lang_data:
            failed_tasks.append(task_name)
            continue

        # 获取指定语言的分数
        scores = lang_data.get(lang, {})
        integral_score = scores.get("integral_score", 0.0)

        if integral_score > 0:
            passed_tasks.append((task_name, integral_score))
        else:
            failed_tasks.append(task_name)

    # Sort passed tasks by integral_score descending
    passed_tasks.sort(key=lambda x: x[1], reverse=True)

    # Extract just the names
    passed_task_names = [t[0] for t in passed_tasks]

    # Combine
    final_order = passed_task_names + failed_tasks

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for task in final_order:
            f.write(f"{task}\n")

    print(f"Generated order file at {output_path}")
    print(f"Language: {lang}")
    print(f"Passed tasks: {len(passed_tasks)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print(f"Total tasks: {len(final_order)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task order from report JSON")
    parser.add_argument("json_path", help="Path to the report JSON file")
    parser.add_argument("--lang", required=True, help="Language to filter by (e.g., python3, cpp)")
    parser.add_argument("--output", "-o", help="Path to the output order file (default: auto-generated)")

    args = parser.parse_args()
    # 自动生成输出路径：在输入文件名后加上语言后缀
    if args.output:
        output_path = args.output
    else:
        json_path = Path(args.json_path)
        output_path = json_path.parent / f"{json_path.stem}_{args.lang}_order.txt"
    generate_order(args.json_path, str(output_path), args.lang)
