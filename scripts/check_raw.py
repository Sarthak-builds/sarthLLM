from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

files = sorted(Path("artifacts/raw/fineweb_edu").rglob("*.parquet"))
total = 0
for f in files:
    docs = pq.ParquetFile(f).metadata.num_rows
    toks = pc.sum(pq.read_table(f, columns=["token_count"])["token_count"]).as_py()
    total += toks
    print(f"{f.name}: {docs:,} docs, {toks:,} tokens, {f.stat().st_size / 1e9:.2f} GB")
print(f"TOTAL: {total:,} tokens")

sample = pq.ParquetFile(files[0]).read_row_group(0, columns=["text"])["text"][0].as_py()
print("--- sample doc ---\n", sample[:500])