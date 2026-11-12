from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.colors as pc
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_umap.pipeline import build_embedding_dataframe_from_fcs_files
from flow_umap.wsp import parse_flowjo_workspace, resolve_workspace_files, workspace_sample_metadata_map


DEFAULT_DATA = Path("flow_umap_analysis/results/exp02_24h_combined/umap_embedding_combined.csv")
DEFAULT_WSP = Path(
    "/Users/enricoklotzsch/Downloads/Exp02_20261006_Tcells_LNP_Substrates_sim_act_transf_24h/"
    "Exp02_20261006_Tcells_LNP_Substrates_sim_act_transf_24h_new.zip.wsp"
)


def _load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_data_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _column_choices(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        c
        for c in df.columns
        if df[c].dtype == object or str(df[c].dtype).startswith("category")
    ]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return categorical, numeric


def _parse_channel_list(text_value: str) -> list[str] | None:
    raw = [x.strip() for x in text_value.split(",") if x.strip()]
    return raw if raw else None


def _discover_fcs_dirs(base_dir: Path, max_depth: int = 2) -> list[Path]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    candidates: set[Path] = set()
    if any(base_dir.glob("*.fcs")):
        candidates.add(base_dir)

    for fcs_file in base_dir.rglob("*.fcs"):
        try:
            rel = fcs_file.parent.relative_to(base_dir)
        except ValueError:
            continue
        if len(rel.parts) <= max_depth:
            candidates.add(fcs_file.parent)

    return sorted(candidates)


def _workspace_tokens(wsp_path: Path) -> list[str]:
    stem = wsp_path.stem
    tokens = [t for t in stem.replace("-", "_").split("_") if t]
    out = [stem]
    out.extend(tokens)
    return list(dict.fromkeys(out))


def _candidate_fcs_roots_for_workspace(wsp_path: Path, fcs_root: Path) -> list[Path]:
    if not fcs_root.exists() or not fcs_root.is_dir():
        return []

    tokens = [t.lower() for t in _workspace_tokens(wsp_path)]
    candidates: list[Path] = []

    # Direct match with workspace stem.
    direct = fcs_root / wsp_path.stem
    if direct.exists() and direct.is_dir():
        candidates.append(direct)

    # Case-insensitive or token-based folder matching under root.
    for child in sorted([p for p in fcs_root.iterdir() if p.is_dir()]):
        lname = child.name.lower()
        if any(tok and tok in lname for tok in tokens):
            candidates.append(child)

    # Keep order while removing duplicates.
    uniq: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve())
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def _resolve_lut(color_is_numeric: bool, lut_name: str) -> dict[str, list[str] | str]:
    def seq(name: str, fallback: list[str]) -> list[str]:
        return getattr(px.colors.sequential, name, fallback)

    def div(name: str, fallback: list[str]) -> list[str]:
        return getattr(px.colors.diverging, name, fallback)

    def cyc(name: str, fallback: list[str]) -> list[str]:
        return getattr(px.colors.cyclical, name, fallback)

    def qual(name: str, fallback: list[str]) -> list[str]:
        return getattr(px.colors.qualitative, name, fallback)

    viridis = seq("Viridis", ["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"])
    jet = seq("Jet", ["#00007F", "#007FFF", "#7FFF7F", "#FF7F00", "#7F0000"])

    continuous = {
        "Viridis": viridis,
        "Plasma": seq("Plasma", viridis),
        "Turbo": seq("Turbo", viridis),
        "Cividis": seq("Cividis", viridis),
        "Jet": jet,
        "Inferno": seq("Inferno", viridis),
        "Magma": seq("Magma", viridis),
        "IceFire": div("IceFire", viridis),
        "RdBu": div("RdBu", viridis),
        "Rainbow": cyc("Rainbow", viridis),
    }
    categorical = {
        "Plotly": qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]),
        "D3": qual("D3", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Set1": qual("Set1", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Dark24": qual("Dark24", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Alphabet": qual("Alphabet", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Safe": qual("Safe", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Prism": qual("Prism", qual("Plotly", ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])),
        "Jet": pc.sample_colorscale(jet, [i / 19 for i in range(20)]),
    }

    if color_is_numeric:
        scale = continuous.get(lut_name, px.colors.sequential.Viridis)
        return {"color_continuous_scale": scale}

    scale = categorical.get(lut_name, px.colors.qualitative.Plotly)
    return {"color_discrete_sequence": scale}


def _get_workspace_info(wsp_path: Path) -> dict[str, Any]:
    return parse_flowjo_workspace(wsp_path)


@st.cache_data(show_spinner=False)
def _get_workspace_info_cached(wsp_path_str: str) -> dict[str, Any]:
    return parse_flowjo_workspace(Path(wsp_path_str))


@st.cache_data(show_spinner=True)
def _build_embedding_cached(
    file_paths: tuple[str, ...],
    channels_csv: str,
    sample_size_per_file: int,
    cofactor: float,
    use_arcsinh: bool,
    n_neighbors: int,
    min_dist: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
    metadata_json: str,
) -> pd.DataFrame:
    channel_list = [c for c in channels_csv.split("||") if c] if channels_csv else None
    metadata_map = json.loads(metadata_json) if metadata_json else None
    result_df, _, _ = build_embedding_dataframe_from_fcs_files(
        fcs_files=[Path(p) for p in file_paths],
        channels=channel_list,
        sample_size_per_file=sample_size_per_file,
        cofactor=cofactor,
        use_arcsinh=use_arcsinh,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        sample_metadata_by_file=metadata_map,
    )
    return result_df


def main() -> None:
    st.set_page_config(page_title="Flow UMAP Explorer", layout="wide")
    st.title("Flow UMAP Explorer")

    st.sidebar.header("Data")
    source_mode = st.sidebar.radio(
        "Source mode",
        options=["UMAP CSV", "FlowJo WSP"],
        index=1,
    )

    if source_mode == "UMAP CSV":
        uploaded = st.sidebar.file_uploader(
            "Load UMAP CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
            help="Load one or multiple UMAP CSV files (for multi-dataset overlays).",
        )

        if uploaded:
            frames: list[pd.DataFrame] = []
            invalid_files: list[str] = []
            for up in uploaded:
                dfi = pd.read_csv(up)
                if not {"UMAP1", "UMAP2"}.issubset(dfi.columns):
                    invalid_files.append(up.name)
                    continue
                dfi = dfi.copy()
                dfi["dataset"] = Path(up.name).stem
                if "experiment_id" not in dfi.columns:
                    dfi["experiment_id"] = dfi["dataset"]
                frames.append(dfi)

            if invalid_files:
                st.error(
                    "These files are missing required UMAP1/UMAP2 columns: "
                    + ", ".join(invalid_files)
                )
                st.stop()

            if not frames:
                st.error("No valid CSV files loaded.")
                st.stop()

            df = pd.concat(frames, ignore_index=True, sort=False)
            data_label = f"Uploaded datasets: {len(frames)}"
        else:
            if not DEFAULT_DATA.exists():
                st.error(
                    "No default file found. Run the pipeline first or upload a CSV in the sidebar."
                )
                st.stop()
            df = _load_data_cached(str(DEFAULT_DATA))
            if "dataset" not in df.columns:
                df = df.copy()
                df["dataset"] = DEFAULT_DATA.stem
            if "experiment_id" not in df.columns:
                df = df.copy()
                df["experiment_id"] = df["dataset"] if "dataset" in df.columns else DEFAULT_DATA.stem
            data_label = str(DEFAULT_DATA)
    else:
        st.sidebar.subheader("Workspace input")
        wsp_mode = st.sidebar.radio(
            "Workspace selection",
            options=["Auto-detect in folder", "Manual path", "Upload .wsp"],
            index=0,
        )

        wsp_folder_default = str(DEFAULT_WSP.parent if DEFAULT_WSP.exists() else Path.cwd())
        wsp_scan_dir = Path(
            st.sidebar.text_input(
                "Workspace folder",
                value=wsp_folder_default,
                help="Folder to scan for .wsp workspace files.",
            )
        )

        discovered_wsp: list[Path] = []
        if wsp_scan_dir.exists() and wsp_scan_dir.is_dir():
            discovered_wsp = sorted(wsp_scan_dir.glob("*.wsp"))

        selected_wsp_paths: list[Path] = []
        if wsp_mode == "Auto-detect in folder":
            if discovered_wsp:
                labels = [p.name for p in discovered_wsp]
                default_selection = [DEFAULT_WSP.name] if DEFAULT_WSP.name in labels else labels[:1]
                selected_names = st.sidebar.multiselect(
                    "Select workspace(s)",
                    options=labels,
                    default=default_selection,
                    help="Select one or multiple workspace files to aggregate into a single UMAP.",
                )
                selected_wsp_paths = [p for p in discovered_wsp if p.name in selected_names]
            else:
                st.sidebar.warning("No .wsp files found in the selected folder.")
        elif wsp_mode == "Upload .wsp":
            uploaded_wsps = st.sidebar.file_uploader("Upload WSP file(s)", type=["wsp"], accept_multiple_files=True)
            if uploaded_wsps:
                cache_dir = Path("flow_umap_analysis/tmp_uploaded_wsp")
                cache_dir.mkdir(parents=True, exist_ok=True)
                for uploaded_wsp in uploaded_wsps:
                    target = cache_dir / uploaded_wsp.name
                    target.write_bytes(uploaded_wsp.getvalue())
                    selected_wsp_paths.append(target)
        else:
            wsp_paths_text = st.sidebar.text_area(
                "WSP file path(s)",
                value=str(DEFAULT_WSP),
                help="Enter one path per line to load multiple workspaces.",
            )
            selected_wsp_paths = [Path(line.strip()) for line in wsp_paths_text.splitlines() if line.strip()]

        if not selected_wsp_paths:
            st.error("WSP file not selected. Choose at least one workspace file.")
            st.stop()
        for path in selected_wsp_paths:
            if not path.exists():
                st.error(f"WSP file not found: {path}")
                st.stop()

        fcs_base_dir = selected_wsp_paths[0].parent
        st.sidebar.subheader("FCS folder selection")
        fcs_mode = st.sidebar.radio(
            "FCS folder prompt",
            options=["Auto-detect near workspace", "Manual path"],
            index=0,
        )
        fcs_dir_path: Path | None = None
        if fcs_mode == "Auto-detect near workspace":
            fcs_scan_root = Path(
                st.sidebar.text_input(
                    "FCS scan root",
                    value=str(fcs_base_dir),
                    help="Folder to scan for subfolders containing .fcs files.",
                )
            )
            discovered_fcs_dirs = _discover_fcs_dirs(fcs_scan_root)
            if discovered_fcs_dirs:
                fcs_options = [str(p) for p in discovered_fcs_dirs]
                default_fcs = str(fcs_base_dir) if str(fcs_base_dir) in fcs_options else fcs_options[0]
                selected_fcs_dir = st.sidebar.selectbox("Select FCS folder", options=fcs_options, index=fcs_options.index(default_fcs))
                fcs_dir_path = Path(selected_fcs_dir)
            else:
                st.sidebar.warning("No folders with .fcs files found in scan root.")
        else:
            fcs_dir_text = st.sidebar.text_input("FCS directory", value=str(fcs_base_dir))
            fcs_dir_path = Path(fcs_dir_text)

        if fcs_dir_path is None or not fcs_dir_path.exists() or not fcs_dir_path.is_dir():
            st.error("FCS directory not selected or not found. Choose a valid FCS folder.")
            st.stop()

        st.sidebar.subheader("Workspace-to-FCS mapping")
        mapping_mode = st.sidebar.radio(
            "Mapping mode",
            options=["Auto-map by workspace name", "Manual map each workspace"],
            index=1 if len(selected_wsp_paths) > 1 else 0,
            help="Use manual mode when different experiments reuse the same FCS filenames.",
        )

        mapping_scan_root = fcs_dir_path
        if mapping_mode == "Manual map each workspace":
            mapping_root_default = fcs_dir_path.parent if fcs_dir_path.parent.exists() else fcs_dir_path
            mapping_scan_root = Path(
                st.sidebar.text_input(
                    "Mapping root folder",
                    value=str(mapping_root_default),
                    help="Scan this root to find candidate experiment folders for each workspace.",
                )
            )

        discovered_map_dirs = _discover_fcs_dirs(mapping_scan_root, max_depth=4)
        if mapping_scan_root.exists() and mapping_scan_root.is_dir() and mapping_scan_root not in discovered_map_dirs:
            discovered_map_dirs = [mapping_scan_root] + discovered_map_dirs

        workspace_fcs_roots: dict[str, Path] = {}
        if mapping_mode == "Manual map each workspace":
            st.sidebar.caption("Assign one FCS folder per workspace (strict matching).")
            folder_options = [str(p) for p in discovered_map_dirs]
            for i, wsp_path in enumerate(selected_wsp_paths):
                auto_candidates = _candidate_fcs_roots_for_workspace(wsp_path, fcs_dir_path)
                default_dir = auto_candidates[0] if auto_candidates else fcs_dir_path
                if folder_options:
                    default_idx = folder_options.index(str(default_dir)) if str(default_dir) in folder_options else 0
                    chosen = st.sidebar.selectbox(
                        f"FCS folder for {wsp_path.name}",
                        options=folder_options,
                        index=default_idx,
                        key=f"map_wsp_{i}",
                    )
                    workspace_fcs_roots[str(wsp_path)] = Path(chosen)
                else:
                    manual_dir = st.sidebar.text_input(
                        f"FCS folder for {wsp_path.name}",
                        value=str(default_dir),
                        key=f"map_wsp_text_{i}",
                    )
                    workspace_fcs_roots[str(wsp_path)] = Path(manual_dir)
        else:
            for wsp_path in selected_wsp_paths:
                workspace_fcs_roots[str(wsp_path)] = fcs_dir_path

        for wsp_path in selected_wsp_paths:
            assigned = workspace_fcs_roots.get(str(wsp_path), fcs_dir_path)
            if not assigned.exists() or not assigned.is_dir():
                st.error(f"Assigned FCS folder is invalid for {wsp_path.name}: {assigned}")
                st.stop()

        workspace_infos: list[tuple[Path, dict[str, Any]]] = [
            (wsp_path, _get_workspace_info_cached(str(wsp_path)))
            for wsp_path in selected_wsp_paths
        ]

        all_group_sets = [set(info.get("groups", {}).keys()) for _, info in workspace_infos if info.get("groups")]
        common_groups = sorted(set.intersection(*all_group_sets)) if all_group_sets else []
        if common_groups:
            default_group = "All Samples" if "All Samples" in common_groups else common_groups[0]
            selected_groups = st.sidebar.multiselect(
                "Workspace group(s)",
                options=common_groups,
                default=[default_group],
                help="Groups are applied to all selected workspaces and then merged.",
            )
        else:
            selected_groups = []
            st.sidebar.info("No common group names across selected workspaces. All samples from each workspace will be used.")

        st.sidebar.subheader("Embedding settings")
        sample_size_per_file = st.sidebar.slider("Sample size per FCS", min_value=100, max_value=3000, value=600, step=100)
        channel_text = st.sidebar.text_area(
            "Optional channels (comma-separated)",
            value="",
            help="Leave empty to auto-select numeric channels from the data.",
        )
        cofactor = st.sidebar.number_input("Arcsinh cofactor", min_value=1.0, max_value=10000.0, value=150.0, step=10.0)
        use_arcsinh = st.sidebar.checkbox("Use arcsinh transform", value=True)
        n_neighbors = st.sidebar.slider("UMAP n_neighbors", min_value=5, max_value=100, value=30)
        min_dist = st.sidebar.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
        dbscan_eps = st.sidebar.slider("DBSCAN eps", min_value=0.1, max_value=2.0, value=0.7, step=0.05)
        dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=5, max_value=100, value=20)

        build_clicked = st.sidebar.button("Load from WSP and build UMAP", type="primary")

        if "wsp_loaded_df" not in st.session_state:
            st.session_state["wsp_loaded_df"] = None
            st.session_state["wsp_loaded_label"] = ""

        if build_clicked:
            with st.spinner("Parsing workspace and computing UMAP from linked FCS files..."):
                fcs_dir = fcs_dir_path
                files: list[Path] = []
                file_to_groups: dict[str, list[str]] = {}
                file_to_experiments: dict[str, str] = {}
                metadata_map: dict[str, dict[str, str]] = {}
                workspace_match_report: list[dict[str, str | int]] = []
                for wsp_path, info in workspace_infos:
                    experiment_id = wsp_path.stem
                    workspace_groups = selected_groups if selected_groups else [None]
                    workspace_md = workspace_sample_metadata_map(info, None)
                    matched_for_workspace: list[Path] = []
                    matched_roots: list[str] = []
                    workspace_root = workspace_fcs_roots.get(str(wsp_path), fcs_dir)
                    if mapping_mode == "Manual map each workspace":
                        candidate_roots = [workspace_root]
                    else:
                        candidate_roots = _candidate_fcs_roots_for_workspace(wsp_path, workspace_root)
                        if not candidate_roots:
                            candidate_roots = [workspace_root]

                    for group_name in workspace_groups:
                        group_files: list[Path] = []
                        for root_candidate in candidate_roots:
                            found = resolve_workspace_files(info, fcs_dir=root_candidate, group_name=group_name)
                            if found:
                                group_files = found
                                matched_roots.append(str(root_candidate))
                                break
                        if (
                            not group_files
                            and mapping_mode != "Manual map each workspace"
                            and fcs_dir not in candidate_roots
                        ):
                            found_global = resolve_workspace_files(info, fcs_dir=fcs_dir, group_name=group_name)
                            if found_global:
                                group_files = found_global
                                matched_roots.append(str(fcs_dir))

                        for fp in group_files:
                            matched_for_workspace.append(fp)
                            fp_key = str(fp.resolve())
                            files.append(fp)
                            if group_name:
                                file_to_groups.setdefault(fp_key, []).append(group_name)
                            file_to_experiments[fp_key] = experiment_id

                            merged_md = dict(
                                workspace_md.get(fp.name)
                                or workspace_md.get(fp.stem)
                                or {}
                            )
                            merged_md["experiment_id"] = experiment_id
                            merged_md["dataset"] = experiment_id
                            if group_name:
                                merged_md["workspace_group"] = group_name
                            metadata_map[fp_key] = merged_md
                            metadata_map[fp.name] = merged_md
                            metadata_map[fp.stem] = merged_md

                    workspace_match_report.append(
                        {
                            "workspace": wsp_path.name,
                            "experiment_id": experiment_id,
                            "files_matched": len({str(p.resolve()) for p in matched_for_workspace}),
                            "assigned_fcs_root": str(workspace_root),
                            "search_root_used": " | ".join(sorted(set(matched_roots))) if matched_roots else "none",
                        }
                    )

                # Deduplicate while preserving order.
                uniq: list[Path] = []
                seen: set[str] = set()
                for fp in files:
                    key = str(fp.resolve())
                    if key not in seen:
                        uniq.append(fp)
                        seen.add(key)
                files = uniq

                if not files:
                    st.error("No matching FCS files were found for the selected workspaces/groups in the chosen FCS directory.")
                    st.stop()

                match_df = pd.DataFrame(workspace_match_report)
                if not match_df.empty and (match_df["files_matched"] == 0).any():
                    st.warning("Some workspaces did not match any FCS files with the selected root. See load report below.")
                if not match_df.empty:
                    st.dataframe(match_df, width="stretch")

                parsed_channels = _parse_channel_list(channel_text)
                files_key = tuple(str(p) for p in files)
                channels_key = "||".join(parsed_channels) if parsed_channels else ""
                metadata_json = json.dumps(metadata_map, sort_keys=True)
                result_df = _build_embedding_cached(
                    file_paths=files_key,
                    channels_csv=channels_key,
                    sample_size_per_file=sample_size_per_file,
                    cofactor=cofactor,
                    use_arcsinh=use_arcsinh,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_samples=dbscan_min_samples,
                    metadata_json=metadata_json,
                )
                if "experiment_id" not in result_df.columns:
                    result_df = result_df.copy()
                    result_df["experiment_id"] = result_df["source_file"].map(
                        lambda x: file_to_experiments.get(str((fcs_dir / str(x)).resolve()), "")
                    )
                if "workspace_group" not in result_df.columns:
                    if "experiment_id" not in result_df.columns:
                        result_df = result_df.copy()
                    result_df["workspace_group"] = result_df["source_file"].map(
                        lambda x: " | ".join(sorted(set(file_to_groups.get(str((fcs_dir / str(x)).resolve()), []))))
                    )
                if "dataset" not in result_df.columns:
                    result_df["dataset"] = result_df.get("experiment_id", "combined")
                if "experiment_id" not in result_df.columns:
                    result_df["experiment_id"] = result_df["dataset"]

                st.session_state["wsp_loaded_df"] = result_df
                st.session_state["wsp_loaded_label"] = (
                    f"WSPs: {len(selected_wsp_paths)} | Groups: {len(selected_groups) if selected_groups else 'all'} | "
                    f"Files: {len(files)} | Rows: {len(result_df):,}"
                )

        if st.session_state["wsp_loaded_df"] is None:
            st.info("Click 'Load from WSP and build UMAP' in the sidebar to load this workspace directly.")
            st.stop()

        df = st.session_state["wsp_loaded_df"]
        data_label = st.session_state["wsp_loaded_label"]

    required = {"UMAP1", "UMAP2"}
    if not required.issubset(df.columns):
        st.error("CSV must contain UMAP1 and UMAP2 columns.")
        st.stop()

    # Normalize experiment labeling so faceting/filtering works consistently across sources.
    if "Experiment" not in df.columns:
        if "experiment_id" in df.columns:
            df = df.copy()
            df["Experiment"] = df["experiment_id"].astype(str)
        elif "dataset" in df.columns:
            df = df.copy()
            df["Experiment"] = df["dataset"].astype(str)

    categorical_cols, numeric_cols = _column_choices(df)

    st.sidebar.header("Plot options")
    color_options = categorical_cols + numeric_cols
    default_color = "cluster" if "cluster" in df.columns else color_options[0]
    color_by = st.sidebar.selectbox("Color by", options=color_options, index=color_options.index(default_color))
    color_is_numeric = color_by in numeric_cols

    if color_is_numeric:
        lut_options = ["Viridis", "Plasma", "Turbo", "Cividis", "Jet", "Inferno", "Magma", "IceFire", "RdBu", "Rainbow"]
    else:
        lut_options = ["Plotly", "D3", "Set1", "Dark24", "Alphabet", "Safe", "Prism", "Jet"]
    lut_name = st.sidebar.selectbox("LUT / colormap", options=lut_options, index=lut_options.index("Jet") if "Jet" in lut_options else 0)
    lut_kwargs = _resolve_lut(color_is_numeric=color_is_numeric, lut_name=lut_name)

    point_size = st.sidebar.slider("Point size", min_value=2, max_value=12, value=5)
    point_alpha = st.sidebar.slider("Point opacity", min_value=0.1, max_value=1.0, value=0.65, step=0.05)

    max_points = min(len(df), 50000)
    if max_points <= 1000:
        subsample_n = max_points
        st.sidebar.caption(f"Points to draw: {subsample_n} (all points)")
    else:
        min_points = 1000
        default_points = min(max_points, 15000)
        range_width = max_points - min_points
        step_points = 1000 if range_width >= 1000 else max(1, range_width // 5)
        subsample_n = st.sidebar.slider(
            "Points to draw",
            min_value=min_points,
            max_value=max_points,
            value=default_points,
            step=step_points,
        )

    filtered = df.copy()

    if "donor" in df.columns:
        donor_values = sorted([str(x) for x in df["donor"].dropna().unique()])
        selected_donors = st.sidebar.multiselect("Donor filter", donor_values, default=donor_values)
        if selected_donors:
            filtered = filtered[filtered["donor"].astype(str).isin(selected_donors)]

    if "stim_state" in df.columns:
        stim_values = sorted([str(x) for x in df["stim_state"].dropna().unique()])
        selected_stim = st.sidebar.multiselect("Stim-state filter", stim_values, default=stim_values)
        if selected_stim:
            filtered = filtered[filtered["stim_state"].astype(str).isin(selected_stim)]

    if "condition" in df.columns:
        cond_values = sorted([str(x) for x in df["condition"].dropna().unique()])
        selected_conditions = st.sidebar.multiselect("Condition filter", cond_values, default=cond_values)
        if selected_conditions:
            filtered = filtered[filtered["condition"].astype(str).isin(selected_conditions)]

    if "Experiment" in df.columns:
        exp_values = sorted([str(x) for x in df["Experiment"].dropna().unique()])
        selected_experiments = st.sidebar.multiselect("Experiment filter", exp_values, default=exp_values)
        if selected_experiments:
            filtered = filtered[filtered["Experiment"].astype(str).isin(selected_experiments)]

    design_skip = {
        "sample_id",
        "source_file",
        "sample_name",
        "annotation",
        "file_basename",
        "file_stem",
        "__highlight__",
    }
    design_factors = [
        c
        for c in categorical_cols
        if c not in design_skip and c not in {"donor", "stim_state", "condition"}
    ]
    design_factors = [
        c
        for c in design_factors
        if 1 < df[c].dropna().nunique() <= 40
    ]

    if design_factors:
        st.sidebar.header("Experimental design")
        for factor in design_factors:
            values = sorted([str(x) for x in df[factor].dropna().unique()])
            selected_values = st.sidebar.multiselect(
                f"{factor} filter",
                values,
                default=values,
            )
            if selected_values:
                filtered = filtered[filtered[factor].astype(str).isin(selected_values)]

    if filtered.empty:
        st.warning("No rows left after filtering. Adjust filters.")
        st.stop()

    if len(filtered) > subsample_n:
        plot_df = filtered.sample(n=subsample_n, random_state=42)
    else:
        plot_df = filtered

    cluster_summary = None
    if "cluster" in plot_df.columns and len(plot_df["cluster"].dropna().unique()) > 0:
        summary_numeric = [
            c for c in numeric_cols if c not in {"UMAP1", "UMAP2", "cluster"}
        ]
        if summary_numeric:
            cluster_means = plot_df.groupby("cluster")[summary_numeric].mean()
            overall_means = plot_df[summary_numeric].mean()
            stds = plot_df[summary_numeric].std(ddof=0).replace(0, np.nan)
            diff = cluster_means.sub(overall_means, axis=1).div(stds, axis=1).fillna(0)
            cluster_summary = diff.reset_index().rename(columns={"cluster": "cluster"})
            cluster_summary = cluster_summary.melt(
                id_vars="cluster",
                var_name="parameter",
                value_name="z_vs_global",
            )
            cluster_summary["direction"] = np.where(cluster_summary["z_vs_global"] > 0, "high", "low")

    st.sidebar.header("Highlight population")
    marker_candidates = [
        c
        for c in numeric_cols
        if any(k in c.lower() for k in ["cd", "gfp", "egfp", "mcherry", "pd1", "ccr7", "cd69", "cd4"])
    ]
    default_markers = [
        c
        for c in marker_candidates
        if any(k in c.lower() for k in ["cd4", "cd69", "gfp", "egfp"])
    ][:3]

    enable_highlight = st.sidebar.checkbox("Enable highlight", value=False)
    highlight_markers = st.sidebar.multiselect(
        "Positive marker channels",
        options=marker_candidates if marker_candidates else numeric_cols,
        default=default_markers,
        help="Choose channels to define positivity, e.g. CD4, CD69, eGFP.",
    )
    pos_percentile = st.sidebar.slider(
        "Positivity threshold percentile",
        min_value=50,
        max_value=99,
        value=85,
        step=1,
        help="Events above this percentile are considered marker-positive.",
    )

    highlight_mask = pd.Series(False, index=plot_df.index)
    highlight_label = ""
    if enable_highlight and highlight_markers:
        per_marker_masks: dict[str, pd.Series] = {}
        for marker in highlight_markers:
            thr = float(plot_df[marker].quantile(pos_percentile / 100.0))
            per_marker_masks[marker] = plot_df[marker] >= thr

        option_names = [f"{m}+" for m in highlight_markers]
        option_names.extend(["All selected (AND)", "Any selected (OR)"])
        highlight_target = st.sidebar.selectbox("Highlight target", options=option_names, index=max(len(option_names) - 2, 0))

        if highlight_target.endswith("+") and highlight_target[:-1] in per_marker_masks:
            chosen_marker = highlight_target[:-1]
            highlight_mask = per_marker_masks[chosen_marker]
            highlight_label = highlight_target
        elif highlight_target == "All selected (AND)":
            highlight_mask = pd.concat(per_marker_masks.values(), axis=1).all(axis=1)
            highlight_label = " AND ".join([f"{m}+" for m in highlight_markers])
        else:
            highlight_mask = pd.concat(per_marker_masks.values(), axis=1).any(axis=1)
            highlight_label = " OR ".join([f"{m}+" for m in highlight_markers])

        plot_df = plot_df.copy()
        plot_df["__highlight__"] = np.where(highlight_mask, "Highlighted", "Other")

    use_highlight_coloring = enable_highlight and highlight_markers and st.sidebar.checkbox(
        "Color by highlight status",
        value=True,
        help="If enabled, highlighted events are colored and all others are gray.",
    )

    st.sidebar.header("Faceting")
    preferred_facets = ["condition", "donor", "stim_state", "timepoint", "well", "cluster"]
    if "Experiment" in plot_df.columns:
        preferred_facets = ["Experiment"] + preferred_facets
    dynamic_facets = [c for c in design_factors if c in plot_df.columns]
    facet_values = []
    for col in preferred_facets + dynamic_facets:
        if col in plot_df.columns and col not in facet_values:
            facet_values.append(col)
    facet_options = ["None"] + facet_values
    facet_by = st.sidebar.selectbox("Facet by", options=facet_options)

    title = st.sidebar.text_input("Plot title", value="UMAP view")

    effective_color_by = "__highlight__" if use_highlight_coloring else color_by

    if use_highlight_coloring:
        lut_kwargs = {
            "color_discrete_map": {
                "Highlighted": "#FF3B30",
                "Other": "#BFBFBF",
            }
        }

    if facet_by == "None":
        fig = px.scatter(
            plot_df,
            x="UMAP1",
            y="UMAP2",
            color=effective_color_by,
            opacity=point_alpha,
            title=title,
            hover_data=[c for c in ["source_file", "condition", "donor", "stim_state", "cluster"] if c in plot_df.columns],
            render_mode="webgl",
            **lut_kwargs,
        )
    else:
        fig = px.scatter(
            plot_df,
            x="UMAP1",
            y="UMAP2",
            color=effective_color_by,
            facet_col=facet_by,
            facet_col_wrap=4,
            opacity=point_alpha,
            title=title,
            hover_data=[c for c in ["source_file", "condition", "donor", "stim_state", "cluster"] if c in plot_df.columns],
            render_mode="webgl",
            **lut_kwargs,
        )

    fig.update_traces(marker={"size": point_size, "line": {"width": 0}})
    fig.update_layout(height=780, legend_title_text=color_by)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Rows loaded", f"{len(df):,}")
    c2.metric("Rows after filters", f"{len(filtered):,}")
    c3.caption(f"Data source: {data_label}")
    if use_highlight_coloring and highlight_label:
        st.caption(f"Highlight active: {highlight_label}")

    st.plotly_chart(fig, width="stretch")

    if cluster_summary is not None:
        st.subheader("Cluster marker summary")
        cluster_ids = sorted(plot_df["cluster"].dropna().astype(str).unique())
        cluster_selection = st.selectbox("Inspect cluster", options=cluster_ids)

        cluster_feature_df = cluster_summary[cluster_summary["cluster"].astype(str) == cluster_selection].copy()
        cluster_feature_df = cluster_feature_df.sort_values("z_vs_global", ascending=False)
        cluster_feature_df = cluster_feature_df.head(12)
        cluster_feature_df = cluster_feature_df.rename(
            columns={"parameter": "feature", "z_vs_global": "z-score vs global mean"}
        )

        high_markers = cluster_feature_df.head(6)
        low_markers = cluster_feature_df.tail(6).sort_values("z-score vs global mean", ascending=True)

        h_col, l_col = st.columns(2)
        with h_col:
            st.caption(f"Highest in cluster {cluster_selection}")
            st.dataframe(high_markers[["feature", "z-score vs global mean"]], width="stretch")
        with l_col:
            st.caption(f"Lowest in cluster {cluster_selection}")
            st.dataframe(low_markers[["feature", "z-score vs global mean"]], width="stretch")

        marker_heatmap = plot_df.groupby("cluster")[numeric_cols].mean().T
        if "cluster" in marker_heatmap.index:
            marker_heatmap = marker_heatmap.drop(index=["cluster"], errors="ignore")

        if marker_heatmap.shape[1] > 0:
            overall_means = plot_df[numeric_cols].mean()
            stds = plot_df[numeric_cols].std(ddof=0).replace(0, np.nan)
            heatmap_values = (
                plot_df.groupby("cluster")[numeric_cols]
                .mean()
                .sub(overall_means, axis=1)
                .div(stds, axis=1)
                .fillna(0)
            )
            heatmap_values = heatmap_values.reindex(sorted(cluster_ids))
            heatmap_fig = px.imshow(
                heatmap_values,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-3,
                zmax=3,
                labels={"x": "Marker", "y": "Cluster", "color": "z-score vs global"},
            )
            heatmap_fig.update_layout(height=max(350, 45 * max(2, heatmap_values.shape[0])))
            st.plotly_chart(heatmap_fig, width="stretch")

    st.subheader("Data preview")
    show_preview = st.checkbox("Show preview table", value=False)
    if show_preview:
        preview_rows = min(120, len(plot_df))
        st.caption(f"Showing first {preview_rows} rows in plain-text preview.")
        st.code(plot_df.head(preview_rows).to_string(index=False), language="text")


if __name__ == "__main__":
    main()
