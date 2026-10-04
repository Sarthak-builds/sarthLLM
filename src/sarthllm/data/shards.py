import json
from pathlib import Path

import numpy as np


class TokenShards:
    """Read-only view of the tokenized .bin shards as one long stream of token ids."""

    def __init__(self, data_dir, max_tokens=None):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        assert meta["dtype"] == "uint16"
        self.paths, self.sizes = [], []
        for s in meta["shards"]:
            if max_tokens is not None and sum(self.sizes) >= max_tokens:
                break
            path = d / s["file"]
            assert path.stat().st_size == s["tokens"] * 2, f"{path} is truncated"  # 2 bytes/token
            self.paths.append(path)
            self.sizes.append(s["tokens"])
        total = sum(self.sizes)
        self.total = total if max_tokens is None else min(total, max_tokens)
        self.starts = np.cumsum([0] + self.sizes[:-1])  # global start offset of each shard
        self._open = {}                                 # shard index -> memmap (opened lazily)

    def __len__(self):
        return self.total

    def _arr(self, k):
        if k not in self._open:
            self._open[k] = np.memmap(self.paths[k], dtype=np.uint16, mode="r")
        return self._open[k]

    def read(self, start, end):
        """Tokens [start, end) of the global stream as int64 (may span shards)."""
        end = min(end, self.total)
        parts = []
        k = int(np.searchsorted(self.starts, start, side="right")) - 1  # shard holding `start`
        while start < end:
            base = int(self.starts[k])
            stop = min(end, base + self.sizes[k])
            parts.append(self._arr(k)[start - base : stop - base])
            start, k = stop, k + 1
        return np.concatenate(parts).astype(np.int64)