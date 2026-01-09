#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import yaml


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.global_memory.bank import GlobalMemoryBank
from core.global_memory.utils.config import (
    ChromaBackendConfig,
    GlobalMemoryConfig,
    MemoryConfig,
    OpenAIEmbeddingConfig,
)


def load_gmb_config(config_path: Path, persist_dir: Path, collection_name: str | None) -> GlobalMemoryConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    gmb = raw.get("global_memory_bank") or {}
    em = gmb.get("embedding_model") or {}
    mem = gmb.get("memory") or {}
    chroma_raw = mem.get("chroma") or {}

    em_cfg = OpenAIEmbeddingConfig(
        provider=str(em.get("provider") or "openai"),
        api_base=em.get("api_base"),
        api_key=em.get("api_key"),
        model=em.get("model"),
        request_timeout=em.get("request_timeout"),
    )

    coll_name = str(collection_name or chroma_raw.get("collection_name") or "global_memory")
    chroma_cfg = ChromaBackendConfig(collection_name=coll_name, persist_path=str(persist_dir))
    mem_cfg = MemoryConfig(backend=str(mem.get("backend") or "chroma"), chroma=chroma_cfg)
    return GlobalMemoryConfig(enabled=True, embedding_model=em_cfg, memory=mem_cfg)


def format_experience_text(item: dict) -> str:
    lines = []

    lines.append(f"#### {item.get('type', '')} Experience: {item.get('title', '')} ")
    lines.append(f"- ({item.get('type', '')}) {item.get('title', '')} — {item.get('description', '')}")
    lines.append(f"Content:\n {item.get('content', '')}")
    return "\n".join(lines)


def ingest_memory_json(bank: GlobalMemoryBank, fp: Path) -> int:
    try:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    items = data.get("experience_library") or []
    if not isinstance(items, list) or not items:
        return 0
    docs: list[str] = [format_experience_text(it) for it in items]
    embeddings = bank.embedding_model.encode_batch(docs)
    payload = []
    for it, doc, emb in zip(items, docs, embeddings):
        meta = {
            "type": str(it.get("type") or "").strip(),
            "title": str(it.get("title") or "").strip(),
            "description": str(it.get("description") or "").strip(),
            "source": str(fp),
        }
        payload.append({"embedding": emb, "metadata": meta, "document": doc})
    bank.memory_backend.add(payload)
    return len(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--persist-dir", required=False)
    parser.add_argument("--collection", required=False, default="global_memory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    persist_dir = Path(args.persist_dir) if args.persist_dir else base_dir / "global_memory"
    persist_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_path)
    gmb_cfg = load_gmb_config(config_path, persist_dir, args.collection)
    bank = GlobalMemoryBank(config=gmb_cfg)

    memory_files = list(base_dir.rglob("memory.json"))
    total_files = len(memory_files)
    total_items = 0
    for fp in memory_files:
        try:
            added = ingest_memory_json(bank, fp)
            total_items += added
        except Exception:
            pass

    print(
        json.dumps(
            {
                "processed_files": total_files,
                "added_items": total_items,
                "persist_path": str(persist_dir),
                "collection": args.collection,
            }
        )
    )


if __name__ == "__main__":
    main()
