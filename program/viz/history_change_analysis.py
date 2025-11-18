from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze pso_history.json: best_score changes & avg_gating diffs")
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./results/viz"))
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    history = json.loads(args.history.read_text(encoding="utf-8"))
    scores = [float(h.get("best_score", np.nan)) for h in history]
    mats = [np.asarray(h.get("avg_gating"), dtype=np.float32) if h.get("avg_gating") is not None else None for h in history]

    l2 = []
    for i in range(len(mats) - 1):
        a = mats[i]
        b = mats[i + 1]
        if a is None or b is None:
            l2.append(float("nan"))
            continue
        l2.append(float(np.linalg.norm(a - b)))

    score_changes = [i for i in range(1, len(scores)) if scores[i] != scores[i - 1]]
    gating_changes = [i for i, v in enumerate(l2) if not np.isnan(v) and v > 1e-7]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "history_changes.json").open("w", encoding="utf-8") as fp:
        json.dump({"score_changes": score_changes, "gating_changes": gating_changes, "l2": l2}, fp, indent=2)

    print(f"Saved history_changes.json to {out}")


if __name__ == "__main__":
    main()
