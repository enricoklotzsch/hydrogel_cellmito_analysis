# Nikon Mitochondrial Morphology Analysis

This workspace is set up for mitochondrial morphology analysis from Nikon microscopy datasets, including `.nd`, `.nd2`, `.tif`, and `.ome.tif` inputs. It covers three pieces of the workflow:

- reading Nikon datasets and viewing them in napari
- segmenting nuclei, cells, and mitochondria
- exporting mitochondria morphology and skeleton-network measurements

## Dependencies

The environment now includes:

- Nikon readers: `nd2`, `bioio`, `bioio-bioformats`, `bioio-tifffile`
- interactive viewing: `napari[all]`, `magicgui`
- segmentation and measurement: `numpy`, `scipy`, `scikit-image`, `tifffile`, `pandas`, `openpyxl`, `skan`
- the vendored mitochondria analysis plugin: `-e ./napari-mito-hcs`

Install or refresh the environment with:

```bash
source .venv311/bin/activate
python -m pip install -r requirements.txt
```

For `.nd` files, Bio-Formats may require a local Java runtime.

## Main Workflow

Use the Nikon-aware entrypoint:

```bash
source .venv311/bin/activate
python scripts/nikon_nd_mito_analysis.py --show-viewer
```

If you do not pass an input path, the script opens a file chooser. Outputs are written under `results/<input-stem>/` by default.

When napari opens, the viewer now provides:

- full 3D channel layers for Z-stack browsing
- 2D analysis layers based on either max projection, middle slice, or a selected Z slice
- a live mitochondria threshold slider and minimum-size control
- optional 3D watershed splitting of the mitochondria mask
- a `Save current analysis` button to export the currently tuned segmentation and morphology tables

Example non-interactive run:

```bash
source .venv311/bin/activate
python scripts/nikon_nd_mito_analysis.py data/example.nd2 --output-dir results/example --time-index 0 --show-viewer
```

To run a built-in mitochondria parameter sweep on one dataset and rank plausible settings:

```bash
source .venv311/bin/activate
python scripts/nikon_nd_mito_analysis.py data/example.nd2 --run-sweep --sweep-top-k 5 --output-dir results/example_sweep
```

This writes:

- `mitochondria_sweep_results.csv` with all tested settings
- `mitochondria_sweep_top_ranked.csv` with the top-ranked settings
- `mitochondria_sweep_recommendation.txt` with the current best starting point

To batch-process multiple files and compare the `MVG` and `VLVG` conditions by filename prefix:

```bash
source .venv311/bin/activate
python scripts/nikon_nd_mito_analysis.py /Users/enricoklotzsch/Downloads --conditions MVG VLVG --output-dir results/mvg_vlvg_batch
```

If you also want a live QC window while the batch is running, add `--show-viewer`. In batch mode this opens a lightweight matplotlib window that updates file by file with the cell hull, mitochondria mask, and mitochondria labels.

Example with a path that contains spaces:

```bash
source .venv311/bin/activate
python scripts/nikon_nd_mito_analysis.py "/Volumes/Extreme SSD/Enrico/2026-09-08_Mitotracker Red" --conditions MVG VLVG --show-viewer --output-dir results/mvg_vlvg_batch
```

This writes:

- `batch_file_summary.csv` with one summary row per image
- `batch_per_cell_summary.csv` with per-cell measurements across all files
- `batch_mito_objects.csv` with per-object mitochondria measurements across all files
- `batch_condition_summary.csv` with aggregated per-condition means, medians, and standard deviations
- `batch_condition_comparison.csv` with direct `MVG` vs `VLVG` mean differences and ratios
- `batch_segmentation_qc_contact_sheet.png` with QC panels for the processed files
- one `segmentation_qc.png` inside each per-file output folder

Generated outputs include:

- `raw_timepoint_czyx.tif` with the selected 3D timepoint
- projected channel images as TIFF
- nuclei, cell, and mitochondria label masks
- `mitochondria_objects.csv` with per-object morphology and skeleton measurements
- `mitochondria_per_cell_summary.csv` with per-cell aggregate mitochondrial morphology
- `mitochondria_skeleton.csv` with branch-level skeleton metrics
- `analysis_summary.csv` with one-row aggregate network statistics

## Structure

- `data/`: raw microscopy inputs
- `scripts/`: analysis entrypoints and helper scripts
- `notebooks/`: exploratory notebook work
- `results/`: exported masks and measurements
- `visualization/`: downstream figures and plotting scripts
- `napari-mito-hcs/`: bundled napari plugin for interactive mitochondria feature analysis

## Clone Notes

This repository includes `napari-mito-hcs` as a Git submodule.

- Clone with `git clone --recurse-submodules https://github.com/enricoklotzsch/hydrogel_cellmito_analysis.git`
- If you already cloned the repository, run `git submodule update --init --recursive`
