from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("export_edges")
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("./results/viz/edges.csv"))
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    history = json.loads(args.history.read_text(encoding="utf-8"))
    # open CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["iteration", "edge_src", "edge_dst", "weight", "rank"])
        for i, entry in enumerate(history):
            mat = entry.get("avg_gating")
            if mat is None:
                continue
            arr = np.asarray(mat, dtype=np.float32)
            n = arr.shape[0]
            flat = []
            for a in range(n):
                for b in range(n):
                    flat.append((arr[a, b], a, b))
            flat.sort(reverse=True, key=lambda x: x[0])
            for rank, (w, a, b) in enumerate(flat[: args.top_k]):
                writer.writerow([i, int(a), int(b), float(w), rank + 1])
    print(f"Wrote edges CSV to {args.out}")


if __name__ == "__main__":
    main()
