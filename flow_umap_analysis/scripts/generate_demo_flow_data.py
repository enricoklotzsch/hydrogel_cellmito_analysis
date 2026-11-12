#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)

    n = 6000
    c1 = rng.normal(loc=[6, 2, 1, 0], scale=[1.2, 0.7, 0.4, 0.5], size=(n // 2, 4))
    c2 = rng.normal(loc=[2, 6, 4, 2], scale=[0.9, 1.1, 0.6, 0.7], size=(n // 2, 4))
    x = np.vstack([c1, c2])

    df = pd.DataFrame(x, columns=["FSC-A", "SSC-A", "CD3", "CD4"])
    df["event_id"] = np.arange(len(df))

    out_path = Path("flow_umap_analysis/data/demo_flow.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
