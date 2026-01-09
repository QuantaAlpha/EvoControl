#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.global_memory.bank import GlobalMemoryBank
from core.global_memory.embeddings.openai import OpenAIEmbeddingModel
from core.global_memory.memory.chroma import ChromaMemoryBackend


def run_embedding_tests(api_base: str, api_key: str, model: str) -> OpenAIEmbeddingModel:
    m = OpenAIEmbeddingModel(api_base=api_base, api_key=api_key, model=model)
    print("init_ok")
    v = m.encode("hello world")
    print("emb_len", len(v))
    vs = m.encode_batch(["hello", "world"])
    print("emb_batch_count", len(vs), "emb_batch_dim", len(vs[0]))
    return m


def run_chroma_tests(collection_name: str, emb_model: OpenAIEmbeddingModel) -> None:
    backend = ChromaMemoryBackend(collection_name=collection_name)
    exp = "Bitwise modulo optimization: use x & (MOD-1) when MOD = 2^k"
    meta = {"type": "Success", "title": "Bitwise modulo", "description": "Faster than division-based modulo"}
    item = {"embedding": emb_model.encode(exp), "metadata": meta, "document": exp}
    backend.add([item])
    q = "fast modulo optimization for dp loops"
    res = backend.query(emb_model.encode(q), k=1)
    print("query_top_count", len(res))
    print("query_top_meta_keys", sorted(list(res[0].keys())) if res else [])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    p.add_argument("--model", default="intfloat/e5-large-v2")
    p.add_argument("--collection", default="se_global_memory_test")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("missing --api-key or OPENROUTER_API_KEY")

    m = run_embedding_tests(args.api_base, args.api_key, args.model)
    run_chroma_tests(args.collection, m)

    cfg = {
        "memory": {"backend": "chroma", "chroma": {"collection_name": args.collection}},
        "embedding_model": {
            "provider": "openai",
            "api_base": args.api_base,
            "api_key": args.api_key,
            "model": args.model,
        },
    }
    bank = GlobalMemoryBank(config=cfg)
    exp_text = "Use faster I/O for heavy input in C++ by disabling sync and untie"
    bank.add_experience(
        exp_text,
        {
            "type": "Success",
            "title": "Faster C++ I/O",
            "description": "ios::sync_with_stdio(false); cin.tie(nullptr);",
        },
    )
    out = bank.retrieve_memories("speed up input reading in c++", k=1)
    print("bank_query_top_count", len(out))
    print("bank_query_top_meta_keys", sorted(list(out[0].keys())) if out else [])


if __name__ == "__main__":
    main()
