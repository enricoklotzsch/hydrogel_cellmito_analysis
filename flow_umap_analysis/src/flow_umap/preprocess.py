from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def choose_numeric_channels(df: pd.DataFrame, excluded_prefixes: Sequence[str] | None = None) -> list[str]:
    prefixes = tuple(excluded_prefixes or ["time", "event", "index", "sample"])
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    selected = []
    for col in numeric_cols:
        if col.lower().startswith(prefixes):
            continue
        selected.append(col)
    return selected


def arcsinh_transform(x: np.ndarray, cofactor: float) -> np.ndarray:
    return np.arcsinh(x / cofactor)


def prepare_matrix(
    df: pd.DataFrame,
    channels: Sequence[str],
    cofactor: float,
    use_arcsinh: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing channels in input data: {missing}")

    channel_df = df.loc[:, list(channels)].copy()
    channel_df = channel_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    x = channel_df.to_numpy(dtype=float)
    if use_arcsinh:
        x = arcsinh_transform(x, cofactor=cofactor)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, channel_df
