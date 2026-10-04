# Reads the parquet files in sorted order, drops docs shorter than min_chars, and tokenizes in batches. Each doc ends with <|endoftext|>.
# When about shard_tokens tokens are buffered, it writes a uint16 shard and rewrites meta.json atomically.
# Rerunning resumes from docs_consumed, and it stops at target_tokens.

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from sarthllm.data.tokenizer import EOT_ID, encode_docs


def iter_texts(paths, text_field, skip=0):
    """Yield document texts across parquet files in order, skipping the first `skip` docs."""
    seen = 0
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1000, columns=[text_field]):
            col = batch.column(0).to_pylist()
            if seen + len(col) <= skip:
                seen += len(col)
                continue
            yield from col[max(0, skip - seen):]
            seen += len(col)


def save_meta(path, meta):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written meta.json


def to_uint16(docs_ids):
    return np.concatenate([np.array(x, dtype=np.uint16) for x in docs_ids])


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cfg["raw_dir"]).rglob("*.parquet"))
    if not paths:
        raise SystemExit("No parquet files in raw_dir. Run download_raw first.")

    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "tokenizer": "gpt2", "dtype": "uint16", "eot_id": EOT_ID,
            "docs_consumed": 0, "shards": [], "total_tokens": 0,
        }

    target = cfg["target_tokens"]
    if meta["total_tokens"] >= target:
        print(f"already complete: {meta['total_tokens']:,} tokens")
        return

    def flush(chunks, docs):
        arr = np.concatenate(chunks)
        name = f"shard_{len(meta['shards']):03d}.bin"
        arr.tofile(out / name)
        meta["shards"].append({"file": name, "tokens": len(arr)})
        meta["total_tokens"] += len(arr)
        meta["docs_consumed"] = docs
        save_meta(meta_path, meta)
        return len(arr)

    docs = meta["docs_consumed"]
    chunks, n_buf, texts = [], 0, []
    pbar = tqdm(total=target, initial=meta["total_tokens"], unit="tok", unit_scale=True)

    for text in iter_texts(paths, cfg["text_field"], skip=docs):
        docs += 1
        if len(text) >= cfg["min_chars"]:
            texts.append(text)
        if len(texts) < cfg["batch_docs"]:
            continue

        chunks.append(to_uint16(encode_docs(texts, cfg["num_threads"])))
        texts = []
        n_buf += len(chunks[-1])
        if n_buf < cfg["shard_tokens"]:
            continue

        pbar.update(flush(chunks, docs))
        chunks, n_buf = [], 0
        if meta["total_tokens"] >= target:
            break
    else:  # raw data ran out before the target: write whatever is left
        if texts:
            chunks.append(to_uint16(encode_docs(texts, cfg["num_threads"])))
        if chunks:
            flush(chunks, docs)
        if meta["total_tokens"] < target:
            print("WARNING: raw data ran out before the target. Raise raw_files and download more.")

    print(f"done: {meta['total_tokens']:,} tokens in {len(meta['shards'])} shards")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    main(p.parse_args().config)