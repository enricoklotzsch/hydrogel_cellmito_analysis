from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import umap

from .io import load_table
from .preprocess import choose_numeric_channels


def _clean_channel_df(df: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing channels in input data: {missing}")

    channel_df = df.loc[:, list(channels)].copy()
    return channel_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")


def _transform_and_scale(channel_df: pd.DataFrame, cofactor: float, use_arcsinh: bool) -> np.ndarray:
    x = channel_df.to_numpy(dtype=float)
    if use_arcsinh:
        x = np.arcsinh(x / cofactor)

    scaler = StandardScaler()
    return scaler.fit_transform(x)


def _annotate_condition(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    donor = parts[0] if len(parts) > 0 else ""
    stim_state = parts[1] if len(parts) > 1 else ""
    condition_raw = parts[2] if len(parts) > 2 else stem
    condition = re.sub(r"-[A-H]\d+$", "", condition_raw)
    return {
        "sample_id": stem,
        "donor": donor,
        "stim_state": stim_state,
        "condition": condition,
    }


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    channels: Sequence[str] | None = None,
    sample_size: int | None = None,
    cofactor: float = 150.0,
    use_arcsinh: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    dbscan_eps: float = 0.7,
    dbscan_min_samples: int = 20,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)
    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    selected_channels = list(channels) if channels else choose_numeric_channels(df)
    if not selected_channels:
        raise ValueError("No numeric channels available. Provide channels with --channels.")

    clean_df = _clean_channel_df(df=df, channels=selected_channels)
    x = _transform_and_scale(clean_df, cofactor=cofactor, use_arcsinh=use_arcsinh)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(x)

    clusters = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(embedding)

    result = clean_df.copy()
    result["UMAP1"] = embedding[:, 0]
    result["UMAP2"] = embedding[:, 1]
    result["cluster"] = clusters

    csv_path = out_path / "umap_embedding.csv"
    result.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=result, x="UMAP1", y="UMAP2", hue="cluster", s=8, linewidth=0, ax=ax, palette="tab20")
    ax.set_title("UMAP embedding of flow data")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    plot_path = out_path / "umap_plot.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    metadata = {
        "input_path": str(input_path),
        "out_dir": str(out_path),
        "n_rows_after_cleaning": int(len(result)),
        "channels": selected_channels,
        "sample_size": sample_size,
        "use_arcsinh": use_arcsinh,
        "cofactor": cofactor,
        "umap": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        },
        "dbscan": {
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "n_clusters_including_noise": int(len(set(clusters))),
            "noise_points": int((clusters == -1).sum()),
        },
        "outputs": {
            "embedding_csv": str(csv_path),
            "plot_png": str(plot_path),
        },
    }

    with (out_path / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def run_directory_pipeline(
    input_dir: str | Path,
    out_dir: str | Path,
    channels: Sequence[str] | None = None,
    sample_size_per_file: int | None = 1500,
    cofactor: float = 150.0,
    use_arcsinh: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    dbscan_eps: float = 0.7,
    dbscan_min_samples: int = 20,
) -> dict:
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fcs_files = sorted(input_path.glob("*.fcs"))
    if not fcs_files:
        raise ValueError(f"No .fcs files found in {input_path}")

    result, selected_channels, clusters = build_embedding_dataframe_from_fcs_files(
        fcs_files=fcs_files,
        channels=channels,
        sample_size_per_file=sample_size_per_file,
        cofactor=cofactor,
        use_arcsinh=use_arcsinh,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    csv_path = out_path / "umap_embedding_combined.csv"
    result.to_csv(csv_path, index=False)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=result, x="UMAP1", y="UMAP2", hue="cluster", s=6, linewidth=0, ax=ax1, palette="tab20")
    ax1.set_title("Combined UMAP across FCS files (colored by cluster)")
    ax1.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig1.tight_layout()
    cluster_plot_path = out_path / "umap_cluster_plot.png"
    fig1.savefig(cluster_plot_path, dpi=220)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=result, x="UMAP1", y="UMAP2", hue="condition", s=6, linewidth=0, ax=ax2)
    ax2.set_title("Combined UMAP across FCS files (colored by condition)")
    ax2.legend(title="condition", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, ncols=1)
    fig2.tight_layout()
    condition_plot_path = out_path / "umap_condition_plot.png"
    fig2.savefig(condition_plot_path, dpi=220)
    plt.close(fig2)

    condition_counts = (
        result.groupby(["donor", "stim_state", "condition", "source_file"], as_index=False)
        .size()
        .rename(columns={"size": "events_used"})
    )
    counts_path = out_path / "condition_event_counts.csv"
    condition_counts.to_csv(counts_path, index=False)

    metadata = {
        "input_dir": str(input_path),
        "out_dir": str(out_path),
        "n_files_total": len(fcs_files),
        "n_files_used": int(result["source_file"].nunique()),
        "n_rows_after_cleaning": int(len(result)),
        "channels": selected_channels,
        "sample_size_per_file": sample_size_per_file,
        "use_arcsinh": use_arcsinh,
        "cofactor": cofactor,
        "umap": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        },
        "dbscan": {
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "n_clusters_including_noise": int(len(set(clusters))),
            "noise_points": int((clusters == -1).sum()),
        },
        "outputs": {
            "embedding_csv": str(csv_path),
            "cluster_plot_png": str(cluster_plot_path),
            "condition_plot_png": str(condition_plot_path),
            "condition_counts_csv": str(counts_path),
        },
    }

    with (out_path / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def build_embedding_dataframe_from_fcs_files(
    fcs_files: Sequence[str | Path],
    channels: Sequence[str] | None = None,
    sample_size_per_file: int | None = 1500,
    cofactor: float = 150.0,
    use_arcsinh: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    dbscan_eps: float = 0.7,
    dbscan_min_samples: int = 20,
    sample_metadata_by_file: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    selected_channels: list[str] | None = list(channels) if channels else None
    parts: list[pd.DataFrame] = []

    for fp in fcs_files:
        fcs_file = Path(fp)
        df = load_table(fcs_file)

        if sample_size_per_file is not None and sample_size_per_file > 0 and len(df) > sample_size_per_file:
            df = df.sample(n=sample_size_per_file, random_state=random_state)

        if selected_channels is None:
            selected_channels = choose_numeric_channels(df)
            if not selected_channels:
                raise ValueError("No numeric channels available. Provide channels with --channels.")

        clean_df = _clean_channel_df(df=df, channels=selected_channels)
        if clean_df.empty:
            continue

        ann = _annotate_condition(fcs_file.stem)
        if sample_metadata_by_file:
            ws_ann = (
                sample_metadata_by_file.get(str(fcs_file.resolve()))
                or sample_metadata_by_file.get(fcs_file.name)
                or sample_metadata_by_file.get(fcs_file.stem)
            )
            if ws_ann:
                ann.update({k: v for k, v in ws_ann.items() if isinstance(v, str) and v})
        for k, v in ann.items():
            clean_df[k] = v
        clean_df["source_file"] = fcs_file.name
        parts.append(clean_df)

    if not parts:
        raise ValueError("No usable events after cleaning across FCS files.")

    result = pd.concat(parts, ignore_index=True)
    x = _transform_and_scale(
        result.loc[:, selected_channels],
        cofactor=cofactor,
        use_arcsinh=use_arcsinh,
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(x)
    clusters = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(embedding)

    result["UMAP1"] = embedding[:, 0]
    result["UMAP2"] = embedding[:, 1]
    result["cluster"] = clusters
    return result, selected_channels, clusters
