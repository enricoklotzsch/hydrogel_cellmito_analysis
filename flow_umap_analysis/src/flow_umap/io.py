from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


def load_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".fcs":
        return _load_fcs(path)

    raise ValueError(f"Unsupported input format: {suffix}. Use .csv, .tsv, or .fcs")


def _load_fcs(path: Path) -> pd.DataFrame:
    try:
        import fcsparser
        import fcsparser.api as fcs_api
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "fcsparser is required for .fcs files. Install dependencies from requirements.txt"
        ) from exc

    def _patch_fcsparser_numpy2() -> bool:
        api_path = Path(fcsparser.__file__).resolve().parent / "api.py"
        if not api_path.exists():
            return False

        text = api_path.read_text(encoding="utf-8")
        old = "data = data.byteswap().newbyteorder()"
        if old not in text:
            return False
        new = "data = data.byteswap().view(data.dtype.newbyteorder())"
        api_path.write_text(text.replace(old, new), encoding="utf-8")
        return True

    major_version = int(str(np.__version__).split(".")[0])
    if major_version >= 2 and _patch_fcsparser_numpy2():
        importlib.reload(fcs_api)

    _, data = fcs_api.parse(str(path), reformat_meta=True)
    return data
