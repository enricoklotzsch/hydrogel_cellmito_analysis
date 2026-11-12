# Flow UMAP Analysis

Small Python project for UMAP analysis of flow cytometry data from CSV/TSV/FCS files.

## What it does

- Loads flow data from `.csv`, `.tsv`, or `.fcs`.
- Selects channels (features) automatically or from user input.
- Applies optional arcsinh transform (common for flow data).
- Scales features, runs UMAP, then clusters with DBSCAN.
- Exports per-event embedding/cluster table and a UMAP plot.

## Project layout

- `src/flow_umap/`: core package code.
- `scripts/run_umap.py`: CLI entrypoint.
- `data/`: place input files here (or pass absolute path).
- `results/`: output files.

## Setup

Use your existing Python 3.11 environment:

```bash
/Users/enricoklotzsch/Documents/Python/Image_Analysis_Raphael/.venv311/bin/python -m pip install -r flow_umap_analysis/requirements.txt
```

## Run

### Auto-select numeric channels

```bash
/Users/enricoklotzsch/Documents/Python/Image_Analysis_Raphael/.venv311/bin/python flow_umap_analysis/scripts/run_umap.py \
  --input flow_umap_analysis/data/example.csv \
  --out-dir flow_umap_analysis/results
```

### Specify channels manually

```bash
/Users/enricoklotzsch/Documents/Python/Image_Analysis_Raphael/.venv311/bin/python flow_umap_analysis/scripts/run_umap.py \
  --input flow_umap_analysis/data/example.fcs \
  --channels FSC-A SSC-A CD3 CD4 CD8 \
  --sample-size 50000 \
  --cofactor 150 \
  --out-dir flow_umap_analysis/results
```

### Run a full experiment folder of FCS files

```bash
/Users/enricoklotzsch/Documents/Python/Image_Analysis_Raphael/.venv311/bin/python flow_umap_analysis/scripts/run_umap.py \
  --input-dir /path/to/fcs_folder \
  --sample-size-per-file 1500 \
  --out-dir flow_umap_analysis/results/experiment_run
```

Notes for directory mode:

- `--sample-size-per-file` controls per-file balancing before the combined UMAP.
- Condition labels are inferred from file names and exported with each event.
- Outputs include both cluster-colored and condition-colored UMAP plots.

## Outputs

- `umap_embedding.csv`: event-level table with `UMAP1`, `UMAP2`, and `cluster`.
- `umap_plot.png`: scatter plot colored by cluster.
- `run_metadata.json`: run parameters, selected channels, counts.

Directory mode outputs:

- `umap_embedding_combined.csv`
- `umap_cluster_plot.png`
- `umap_condition_plot.png`
- `condition_event_counts.csv`
- `run_metadata.json`

## GUI Explorer

Interactive plotting app for exploring UMAP outputs:

```bash
/Users/enricoklotzsch/Documents/Python/Image_Analysis_Raphael/.venv311/bin/python -m streamlit run flow_umap_analysis/gui/umap_explorer.py
```

GUI features:

- Upload a UMAP CSV or use the default combined output.
- Load a FlowJo `.wsp` directly, select workspace group, and auto-load linked FCS files.
- Color by cluster, condition, donor, or marker channels.
- Filter donor / stimulation state / condition.
- Adjust point size, opacity, and point count.
- Optional faceting by condition, donor, stimulation state, or cluster.

## Notes

- For large files, use `--sample-size` first to iterate quickly.
- If FCS parsing fails for a specific file, convert to CSV and rerun.
