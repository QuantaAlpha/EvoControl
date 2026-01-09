import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

ROOT_DIR = os.environ.get(
    "VIZ_ROOT",
    "trajectories_perf/deepseek-v3.1",
)


def _safe_read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_llm_io(llm_path: Path):
    by_iter: dict[str, list[dict]] = {}
    if not llm_path.exists():
        return {"by_iteration": {}}
    try:
        optimize_counter = 0
        with llm_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                # Check keys in entry to debug
                # print(f"DEBUG: entry keys: {entry.keys()}")

                # Strictly use iteration_index when available, per user requirement
                it = entry.get("iteration_index")
                if it is None:
                    it = entry.get("iteration")

                # Fallback for missing iteration index
                if it is None:
                    context = entry.get("context")
                    if context == "perfagent.optimize":
                        # First optimize call is usually iteration 1
                        optimize_counter += 1
                        it = optimize_counter
                    else:
                        # For other contexts like summarizer, attach to current iteration
                        if optimize_counter > 0:
                            it = optimize_counter
                        else:
                            # Before any optimization
                            it = 0

                # If still None, skip this entry to avoid incorrect grouping
                if it is None:
                    # print(f"DEBUG: Skipping entry due to missing iteration: {entry.keys()}")
                    continue

                try:
                    it = int(it)
                except Exception:
                    continue

                simplified = {
                    "ts": entry.get("ts"),
                    "context": entry.get("context"),
                    "model": entry.get("model"),
                    "response": entry.get("response"),  # Added response field
                    # Keep messages available for future use, but frontend will only display context
                    "messages": [
                        {
                            "role": (m.get("role") if isinstance(m, dict) else None),
                            "content": (m.get("content") if isinstance(m, dict) else None),
                        }
                        for m in (entry.get("messages") or [])
                    ],
                }
                by_iter.setdefault(str(it), []).append(simplified)
    except Exception:
        return {"by_iteration": {}}
    return {"by_iteration": by_iter}


def _load_instance_data(root: Path, instance: str):
    if not root.exists() or not root.is_dir():
        return {}

    # Find the directory that might contain this instance
    # We need to search through subdirectories since we don't know which one contains the instance
    # However, the current structure seems to imply instances are keys in traj.pool files within subdirectories of 'root'
    # BUT, looking at previous _load_instances, it iterated over root.iterdir()
    # and for each subdir p, it read p/traj.pool.

    # To optimize, we first need to know which subdirectory contains the instance,
    # OR we just iterate like before but filter for the specific instance.
    # But the user wants to load *instances* list first.

    # Let's change approach:
    # 1. Get list of instances (and which subdir they belong to, if 'root' has multiple subdirs with traj.pool)
    #    Wait, 'root' in _load_instances was passed as Path(ROOT_DIR) / subdir from get_data.
    #    So 'root' here is the experiment directory (e.g. deepseek-v3.1-AutoSelect...).
    #    Inside this experiment dir, there are problem directories (e.g. aizu_1459...).
    #    And inside problem dir, there is traj.pool.

    # Let's look at LS output again.
    # /data/CodeEfficiency/SE-Agent/trajectories_perf/deepseek-v3.1/deepseek-v3.1-AutoSelect_20251204_165100/
    #   aizu_1459_e-circuit-is-now-on-sale/
    #   ...

    # So, for a given experiment dir (subdir):
    # Instances are likely the names of these problem directories?
    # Let's check _load_instances logic again.
    # for p in sorted(root.iterdir()):
    #    traj_path = p / "traj.pool"
    #    pool_data = _safe_read_json(traj_path)
    #    for inst, inst_data in pool_data.items(): ...

    # It seems 'inst' (instance name) comes from keys inside traj.pool.
    # And 'p' is a directory which might correspond to a problem.

    # So to get list of instances, we need to read all traj.pool files.
    # To get data for ONE instance, we need to find which traj.pool has it, and load only that.

    pass


def _get_instances_list(root: Path):
    instances = []
    if not root.exists() or not root.is_dir():
        return instances

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        traj_path = p / "traj.pool"
        if not traj_path.exists():
            continue

        # We can just assume the directory name is related, but the code reads keys from traj.pool
        # Reading all traj.pool files just to get keys might be slow if they are huge.
        # But we have to do it to get the exact instance keys as used before.
        try:
            # optimization: maybe we don't need to read the whole json if we just want keys?
            # But json.load reads all.
            # If files are large, this is slow.
            # However, previously we loaded ALL data for ALL instances.
            # So loading just keys (even by reading full file) is not worse than before,
            # but we want to avoid sending all data to frontend.

            pool_data = _safe_read_json(traj_path)
            if isinstance(pool_data, dict):
                for inst in pool_data.keys():
                    instances.append(inst)
        except Exception:
            continue

    return sorted(instances)


def _load_single_instance(root: Path, instance_id: str):
    if not root.exists() or not root.is_dir():
        return {}

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        traj_path = p / "traj.pool"
        llm_path = p / "llm_io.jsonl"
        if not traj_path.exists():
            continue

        pool_data = _safe_read_json(traj_path)
        if not isinstance(pool_data, dict):
            continue

        if instance_id in pool_data:
            inst_data = dict(pool_data[instance_id])

            # Debug logging
            print(f"Found instance {instance_id} in {p}")
            print(f"Checking LLM IO path: {llm_path}, exists: {llm_path.exists()}")

            inst_data["llm_io"] = _load_llm_io(llm_path)

            # Debug logging for loaded IO
            # print(f"Loaded LLM IO keys: {inst_data['llm_io'].keys()}")

            return {instance_id: inst_data}

    return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dirs")
def get_dirs():
    root = Path(ROOT_DIR)
    if not root.exists() or not root.is_dir():
        return jsonify([])
    dirs = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return jsonify(sorted(dirs))


@app.route("/api/instances")
def get_instances():
    subdir = request.args.get("subdir")
    if not subdir:
        return jsonify([])

    root = Path(ROOT_DIR) / subdir
    instances = _get_instances_list(root)
    return jsonify(instances)


@app.route("/api/data")
def get_data():
    subdir = request.args.get("subdir")
    instance = request.args.get("instance")
    if not subdir:
        return jsonify({})

    root = Path(ROOT_DIR) / subdir

    if instance:
        # Load specific instance data
        data = _load_single_instance(root, instance)
    else:
        # Keep backward compatibility or return empty?
        # If no instance specified, previously it loaded ALL.
        # But now we want to avoid that if possible, or maybe user wants all?
        # Given the request "load on select", we should probably not load anything if no instance.
        # But let's keep it safe: if no instance, return empty dict or maybe lightweight list?
        # Let's return empty dict to force selection.
        return jsonify({})

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
