from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from bioio import BioImage
from bioio_bioformats import Reader as BioformatsReader
from magicgui import magicgui
import nd2
from scipy import ndimage as ndi
from skan.csr import Skeleton, summarize as summarize_skeleton
from skimage import exposure, feature as skfeature, filters, measure, morphology, segmentation
from skimage.restoration import rolling_ball


DEFAULT_PIXEL_SIZE_UM = 0.0866
SUPPORTED_IMAGE_SUFFIXES = (".nd", ".nd2", ".tif", ".tiff", ".ome.tif", ".ome.tiff")
BATCH_PLOT_METRICS = [
    ("mitochondria_count", "Mito count per cell"),
    ("total_mito_volume_um3", "Total mito volume per cell (um^3)"),
    ("mito_volume_fraction", "Mito volume fraction per cell"),
    ("total_branch_length_um", "Total branch length per cell (um)"),
    ("junction_count", "Junction count per cell"),
    ("connectivity_index", "Connectivity index per cell"),
]

warnings.filterwarnings(
    "ignore",
    message="separator in column name will change to _ in version 0.13.*",
)
warnings.filterwarnings(
    "ignore",
    message="Failed to get convex hull image.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in scalar divide",
    category=RuntimeWarning,
)


@dataclass
class ImageDataset:
    data: np.ndarray
    channel_names: list[str]
    spacing_zyx: tuple[float, float, float]


def _find_branch_length_column(dataframe: pd.DataFrame) -> str:
    for candidate in ("branch-distance", "branch_distance"):
        if candidate in dataframe.columns:
            return candidate
    raise KeyError(
        "skan summary did not provide a branch length column; expected one of "
        "'branch-distance' or 'branch_distance'."
    )


def _normalize_channel_names(channel_names: list[str], channel_count: int) -> list[str]:
    normalized = [str(name) if str(name).strip() else f"channel_{index}" for index, name in enumerate(channel_names)]
    for index in range(len(normalized), channel_count):
        normalized.append(f"channel_{index}")
    return normalized[:channel_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze mitochondrial morphology from Nikon .nd/.nd2 and TIFF microscopy datasets."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Input microscopy file (.nd, .nd2, .tif, .tiff, .ome.tif, .ome.tiff).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported masks and CSV results. Defaults to results/<input-stem>.",
    )
    parser.add_argument("--time-index", type=int, default=0, help="Time index to analyze.")
    parser.add_argument(
        "--skip-leading-z-slices",
        type=int,
        default=2,
        help="Exclude this many leading Z slices from 3D analysis to reduce noise.",
    )
    parser.add_argument(
        "--z-mode",
        choices=("max", "middle", "index"),
        default="max",
        help="How to build the 2D preview image in napari and on disk.",
    )
    parser.add_argument("--z-index", type=int, default=0, help="Used when --z-mode index.")
    parser.add_argument("--nuclei-channel", type=int, default=0, help="Nuclei channel index.")
    parser.add_argument("--cell-channel", type=int, default=1, help="Cell channel index.")
    parser.add_argument("--mito-channel", type=int, default=2, help="Mitochondria channel index.")
    parser.add_argument(
        "--cell-threshold-method",
        choices=("triangle", "otsu", "li", "manual"),
        default="li",
        help="Thresholding strategy for the single-cell hull on channel 2.",
    )
    parser.add_argument(
        "--cell-threshold",
        type=float,
        default=None,
        help="Manual absolute threshold on the prepared cell projection in the range [0, 0.1].",
    )
    parser.add_argument(
        "--cell-threshold-offset",
        type=float,
        default=0.0,
        help="Offset added to the automatic cell threshold.",
    )
    parser.add_argument(
        "--cell-min-size",
        type=int,
        default=2000,
        help="Minimum projected cell area in pixels.",
    )
    parser.add_argument(
        "--cell-closing-radius",
        type=int,
        default=7,
        help="Morphological closing radius used on the projected cell hull.",
    )
    parser.add_argument(
        "--mito-threshold-method",
        choices=("local", "slice_otsu", "otsu", "manual"),
        default="local",
        help="Thresholding strategy for mitochondria segmentation.",
    )
    parser.add_argument(
        "--mito-threshold",
        type=float,
        default=None,
        help="Manual absolute threshold on the preprocessed mitochondria image in the range [0, 1].",
    )
    parser.add_argument(
        "--mito-threshold-offset",
        type=float,
        default=0.0,
        help="Offset added to the automatic threshold. Useful for tuning local or Otsu thresholding.",
    )
    parser.add_argument(
        "--mito-local-block-size",
        type=int,
        default=51,
        help="Odd block size for local mitochondria thresholding.",
    )
    parser.add_argument(
        "--mito-min-size",
        type=int,
        default=30,
        help="Minimum mitochondria component size in voxels or pixels.",
    )
    parser.add_argument(
        "--mito-watershed-min-distance",
        type=int,
        default=6,
        help="Minimum distance in voxels between watershed seeds when splitting 3D mitochondria.",
    )
    parser.add_argument(
        "--disable-mito-watershed",
        action="store_true",
        help="Disable the 3D watershed split and keep plain connected components.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=DEFAULT_PIXEL_SIZE_UM,
        help="Fallback XY pixel size in micrometers when metadata is unavailable.",
    )
    parser.add_argument(
        "--z-step-um",
        type=float,
        default=None,
        help="Fallback Z step size in micrometers when metadata is unavailable.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Open the raw stack, preview images, and labels in napari after analysis.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["MVG", "VLVG"],
        help="Condition prefixes to include and compare during batch processing.",
    )
    parser.add_argument(
        "--run-sweep",
        action="store_true",
        help="Run a built-in parameter sweep for mitochondria segmentation instead of a single analysis run.",
    )
    parser.add_argument(
        "--sweep-top-k",
        type=int,
        default=10,
        help="How many top-ranked sweep settings to save in the summary outputs.",
    )
    return parser.parse_args()


def select_input_file() -> Path:
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select Nikon or TIFF dataset",
        filetypes=[
            ("Microscopy files", "*.nd *.nd2 *.tif *.tiff *.ome.tif *.ome.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit("No input file selected.")
    return Path(selected)


def normalize_axes(array: np.ndarray, axes: str) -> np.ndarray:
    axes_list = [axis.upper() for axis in axes]
    data = np.asarray(array)

    for axis in "TCZYX":
        if axis not in axes_list:
            data = np.expand_dims(data, axis=data.ndim)
            axes_list.append(axis)

    order = [axes_list.index(axis) for axis in "TCZYX"]
    return np.transpose(data, order)


def load_with_tifffile(path: Path) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tif:
        array = tif.asarray()
        axes = getattr(tif.series[0], "axes", "YX")
    return array, axes


def load_with_nd2(path: Path) -> ImageDataset:
    with nd2.ND2File(path) as handle:
        array = handle.asarray()
        axes = "".join(handle.sizes.keys())
        metadata_channels = getattr(handle.metadata, "channels", [])
        channel_names = []
        for index, channel in enumerate(metadata_channels):
            channel_name = getattr(getattr(channel, "channel", None), "name", None)
            channel_names.append(str(channel_name) if channel_name is not None else f"channel_{index}")
        voxel_size = handle.voxel_size()
        spacing_zyx = (
            float(voxel_size.z) if voxel_size.z is not None else DEFAULT_PIXEL_SIZE_UM,
            float(voxel_size.y) if voxel_size.y is not None else DEFAULT_PIXEL_SIZE_UM,
            float(voxel_size.x) if voxel_size.x is not None else DEFAULT_PIXEL_SIZE_UM,
        )
    data = normalize_axes(array, axes)
    return ImageDataset(
        data=data,
        channel_names=_normalize_channel_names(channel_names, data.shape[1]),
        spacing_zyx=spacing_zyx,
    )


def load_with_bioformats(path: Path, fallback_spacing_zyx: tuple[float, float, float]) -> ImageDataset:
    image = BioImage(path, reader=BioformatsReader)
    data = image.get_image_data("TCZYX")
    channel_names = getattr(image, "channel_names", None) or []
    channel_names = _normalize_channel_names([str(name) for name in channel_names], data.shape[1])
    pixel_sizes = getattr(image, "physical_pixel_sizes", None)
    if pixel_sizes is not None:
        spacing_zyx = (
            float(getattr(pixel_sizes, "Z", fallback_spacing_zyx[0]) or fallback_spacing_zyx[0]),
            float(getattr(pixel_sizes, "Y", fallback_spacing_zyx[1]) or fallback_spacing_zyx[1]),
            float(getattr(pixel_sizes, "X", fallback_spacing_zyx[2]) or fallback_spacing_zyx[2]),
        )
    else:
        spacing_zyx = fallback_spacing_zyx
    return ImageDataset(data=data, channel_names=channel_names, spacing_zyx=spacing_zyx)


def load_dataset(path: Path, fallback_xy_um: float, fallback_z_um: float | None) -> ImageDataset:
    fallback_spacing_zyx = (fallback_z_um or fallback_xy_um, fallback_xy_um, fallback_xy_um)
    lower_name = path.name.lower()
    if lower_name.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        array, axes = load_with_tifffile(path)
        data = normalize_axes(array, axes)
        return ImageDataset(
            data=data,
            channel_names=_normalize_channel_names([], data.shape[1]),
            spacing_zyx=fallback_spacing_zyx,
        )
    if lower_name.endswith(".nd2"):
        try:
            return load_with_nd2(path)
        except Exception as exc:
            if "Invalid ChunkMap signature" in str(exc):
                raise RuntimeError(f"Unreadable ND2 file for direct loader: {path}") from exc
            return load_with_bioformats(path, fallback_spacing_zyx)
    if lower_name.endswith(".nd"):
        return load_with_bioformats(path, fallback_spacing_zyx)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def is_supported_image_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in SUPPORTED_IMAGE_SUFFIXES)


def infer_condition(path: Path, conditions: list[str]) -> str | None:
    stem_upper = path.stem.upper()
    for condition in conditions:
        normalized = condition.upper()
        if stem_upper.startswith(normalized):
            return normalized
    return None


def collect_batch_files(input_path: Path, conditions: list[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path] if infer_condition(input_path, conditions) is not None else []

    files = []
    for candidate in sorted(input_path.iterdir()):
        if not candidate.is_file() or not is_supported_image_file(candidate):
            continue
        if infer_condition(candidate, conditions) is None:
            continue
        files.append(candidate)
    return files


def choose_volume_plane(volume_zyx: np.ndarray, z_mode: str, z_index: int) -> np.ndarray:
    if volume_zyx.ndim != 3 or volume_zyx.shape[0] == 1:
        return volume_zyx[0] if volume_zyx.ndim == 3 else volume_zyx
    if z_mode == "max":
        return np.max(volume_zyx, axis=0)
    if z_mode == "middle":
        return volume_zyx[volume_zyx.shape[0] // 2]
    clamped = max(0, min(z_index, volume_zyx.shape[0] - 1))
    return volume_zyx[clamped]


def choose_channel_preview(timepoint_czyx: np.ndarray, z_mode: str, z_index: int) -> np.ndarray:
    previews = [choose_volume_plane(channel_volume, z_mode, z_index) for channel_volume in timepoint_czyx]
    return np.stack(previews, axis=0)


def trim_leading_z_slices(timepoint_czyx: np.ndarray, skip_leading_z_slices: int) -> np.ndarray:
    skip_count = max(0, int(skip_leading_z_slices))
    if skip_count == 0 or timepoint_czyx.ndim < 4:
        return timepoint_czyx
    z_count = timepoint_czyx.shape[1]
    if z_count <= skip_count:
        return timepoint_czyx[:, -1:, ...]
    return timepoint_czyx[:, skip_count:, ...]


def prepare_image(image: np.ndarray, background_radius: int = 0) -> np.ndarray:
    data = image.astype(np.float32)
    if background_radius > 0:
        background = rolling_ball(data, radius=background_radius)
        data = data - background
        data[data < 0] = 0

    low, high = np.percentile(data, (1, 99.8))
    if high > low:
        data = np.clip(data, low, high)
        data = (data - low) / (high - low)
    else:
        data = np.zeros_like(data)
    return exposure.equalize_adapthist(data)


def prepare_volume(volume: np.ndarray, background_radius: int = 0) -> np.ndarray:
    if volume.ndim == 2:
        return prepare_image(volume, background_radius=background_radius)
    prepared_slices = [prepare_image(slice_image, background_radius=background_radius) for slice_image in volume]
    return np.stack(prepared_slices, axis=0)


def _footprint(ndim: int, radius: int) -> np.ndarray:
    if ndim == 2:
        return morphology.disk(radius)
    return morphology.ball(radius)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if np.max(labels) == 0:
        return np.zeros_like(mask, dtype=bool)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))
    return labels == largest_label


def fill_z_between_extents(mask3d: np.ndarray) -> np.ndarray:
    filled = np.zeros_like(mask3d, dtype=bool)
    if mask3d.ndim != 3:
        return mask3d.astype(bool)
    height, width = mask3d.shape[1], mask3d.shape[2]
    for y_index in range(height):
        for x_index in range(width):
            z_indices = np.flatnonzero(mask3d[:, y_index, x_index])
            if z_indices.size == 0:
                continue
            filled[z_indices[0] : z_indices[-1] + 1, y_index, x_index] = True
    return filled


def segment_nuclei(nuclei_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prepared = prepare_volume(nuclei_image)
    threshold = filters.threshold_otsu(prepared)
    mask = prepared > threshold
    mask = morphology.remove_small_objects(mask, min_size=400 if nuclei_image.ndim == 3 else 150)
    mask = morphology.remove_small_holes(mask, area_threshold=300 if nuclei_image.ndim == 3 else 200)
    mask = morphology.binary_opening(mask, _footprint(nuclei_image.ndim, 1 if nuclei_image.ndim == 3 else 2))
    mask = ndi.binary_fill_holes(mask)
    return measure.label(mask), prepared


def segment_cells(
    cell_image: np.ndarray,
    nuclei_labels: np.ndarray,
    threshold_method: str,
    threshold: float | None,
    threshold_offset: float,
    min_size: int,
    closing_radius: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    prepared = prepare_volume(cell_image, background_radius=40)
    if cell_image.ndim == 3:
        projection = np.max(prepared, axis=0)
        if threshold_method == "manual" and threshold is None:
            threshold = 0.35
        if threshold is not None:
            resolved_threshold = float(threshold)
            resolved_method = "manual"
        elif threshold_method == "otsu":
            resolved_threshold = float(filters.threshold_otsu(projection) + threshold_offset)
            resolved_method = "otsu"
        elif threshold_method == "li":
            resolved_threshold = float(filters.threshold_li(projection) + threshold_offset)
            resolved_method = "li"
        else:
            resolved_threshold = float(filters.threshold_triangle(projection) + threshold_offset)
            resolved_method = "triangle"
        mask = prepared > resolved_threshold
        mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_size)))
        mask = ndi.binary_fill_holes(mask)
        mask = keep_largest_component(mask)

        hull_slices = []
        for slice_mask in mask:
            if np.any(slice_mask):
                filled_slice = ndi.binary_fill_holes(slice_mask)
                hull_slices.append(morphology.convex_hull_image(filled_slice))
            else:
                hull_slices.append(np.zeros_like(slice_mask, dtype=bool))
        mask = np.stack(hull_slices, axis=0)
        mask = fill_z_between_extents(mask)
        mask = morphology.binary_closing(mask, morphology.ball(max(1, int(closing_radius) // 2)))
    else:
        if threshold_method == "manual" and threshold is None:
            threshold = 0.35
        if threshold is not None:
            resolved_threshold = float(threshold)
            resolved_method = "manual"
        elif threshold_method == "otsu":
            resolved_threshold = float(filters.threshold_otsu(prepared) + threshold_offset)
            resolved_method = "otsu"
        elif threshold_method == "li":
            resolved_threshold = float(filters.threshold_li(prepared) + threshold_offset)
            resolved_method = "li"
        else:
            resolved_threshold = float(filters.threshold_triangle(prepared) + threshold_offset)
            resolved_method = "triangle"
        mask = prepared > resolved_threshold
        mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_size)))
        mask = morphology.binary_closing(mask, morphology.disk(max(1, int(closing_radius))))
        mask = ndi.binary_fill_holes(mask)
        mask = keep_largest_component(mask)
        if np.any(mask):
            mask = morphology.convex_hull_image(mask)
        mask = morphology.binary_closing(mask, morphology.disk(max(1, int(closing_radius))))
    mask = ndi.binary_fill_holes(mask)

    labels = np.zeros_like(mask, dtype=np.int32)
    labels[mask] = 1
    return labels, prepared, resolved_threshold, resolved_method


def relabel_mito_by_cell(mito_mask: np.ndarray, cell_labels: np.ndarray) -> np.ndarray:
    relabeled = np.zeros_like(cell_labels, dtype=np.int32)
    offset = 0
    for cell_id in sorted(int(value) for value in np.unique(cell_labels) if value > 0):
        local_labels = measure.label(mito_mask & (cell_labels == cell_id))
        if np.max(local_labels) == 0:
            continue
        local_labels[local_labels > 0] += offset
        relabeled[local_labels > 0] = local_labels[local_labels > 0]
        offset = int(np.max(relabeled))
    return relabeled


def split_mito_components_by_watershed(
    mito_mask: np.ndarray,
    cell_labels: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    relabeled = np.zeros_like(cell_labels, dtype=np.int32)
    next_label = 1
    minimum_distance = max(1, int(min_distance))

    for cell_id in sorted(int(value) for value in np.unique(cell_labels) if value > 0):
        cell_mask = mito_mask & (cell_labels == cell_id)
        if not np.any(cell_mask):
            continue

        distance = ndi.distance_transform_edt(cell_mask)
        coordinates = skfeature.peak_local_max(
            distance,
            labels=cell_mask.astype(np.uint8),
            min_distance=minimum_distance,
            exclude_border=False,
        )

        if len(coordinates) < 2:
            local_labels = measure.label(cell_mask)
        else:
            markers = np.zeros(cell_mask.shape, dtype=np.int32)
            for marker_id, coordinate in enumerate(coordinates, start=1):
                markers[tuple(coordinate)] = marker_id
            local_labels = segmentation.watershed(-distance, markers=markers, mask=cell_mask)

        if np.max(local_labels) == 0:
            continue
        local_labels = local_labels.astype(np.int32)
        local_labels[local_labels > 0] += next_label - 1
        relabeled[local_labels > 0] = local_labels[local_labels > 0]
        next_label = int(np.max(relabeled)) + 1
    return relabeled


def segment_mitochondria(
    mito_image: np.ndarray,
    cell_labels: np.ndarray,
    threshold_method: str,
    threshold: float | None,
    threshold_offset: float,
    local_block_size: int,
    min_size: int,
    apply_watershed_split: bool,
    watershed_min_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    prepared = prepare_volume(mito_image, background_radius=20)
    block_size = max(3, int(local_block_size))
    if block_size % 2 == 0:
        block_size += 1

    if threshold_method == "manual" and threshold is None:
        threshold = 0.50

    if threshold is not None:
        binary = prepared > threshold
        resolved_threshold = float(threshold)
        resolved_method = "manual"
    elif threshold_method == "local":
        if prepared.ndim == 2:
            local_threshold = filters.threshold_local(prepared, block_size=block_size)
            binary = prepared > (local_threshold + threshold_offset)
            resolved_threshold = float(np.mean(local_threshold + threshold_offset))
        else:
            binary_slices = []
            threshold_means = []
            for slice_image in prepared:
                local_threshold = filters.threshold_local(slice_image, block_size=block_size)
                cutoff = local_threshold + threshold_offset
                binary_slices.append(slice_image > cutoff)
                threshold_means.append(float(np.mean(cutoff)))
            binary = np.stack(binary_slices, axis=0)
            resolved_threshold = float(np.mean(threshold_means)) if threshold_means else 0.0
        resolved_method = "local"
    elif threshold_method == "slice_otsu":
        if prepared.ndim == 2:
            cutoff = float(filters.threshold_otsu(prepared) + threshold_offset)
            binary = prepared > cutoff
            resolved_threshold = cutoff
        else:
            binary_slices = []
            thresholds = []
            for slice_image in prepared:
                cutoff = float(filters.threshold_otsu(slice_image) + threshold_offset)
                binary_slices.append(slice_image > cutoff)
                thresholds.append(cutoff)
            binary = np.stack(binary_slices, axis=0)
            resolved_threshold = float(np.mean(thresholds)) if thresholds else 0.0
        resolved_method = "slice_otsu"
    else:
        cutoff = float(filters.threshold_otsu(prepared) + threshold_offset)
        binary = prepared > cutoff
        resolved_threshold = cutoff
        resolved_method = "otsu"

    binary = morphology.remove_small_objects(binary, min_size=max(1, int(min_size)))
    binary = morphology.binary_opening(binary, _footprint(mito_image.ndim, 1))
    binary &= cell_labels > 0
    unsplit_labels = relabel_mito_by_cell(binary, cell_labels)
    if apply_watershed_split and mito_image.ndim == 3:
        split_labels = split_mito_components_by_watershed(binary, cell_labels, watershed_min_distance)
        unsplit_count = int(np.max(unsplit_labels))
        split_count = int(np.max(split_labels))
        # Avoid pathological over-splitting on low-signal datasets where the
        # watershed turns one mitochondrial network into hundreds of fragments.
        if unsplit_count > 0 and split_count > max(unsplit_count * 4, unsplit_count + 100):
            mito_labels = unsplit_labels
            resolved_method = f"{resolved_method}_no_watershed_fallback"
        else:
            mito_labels = split_labels
    else:
        mito_labels = unsplit_labels
    return mito_labels, binary.astype(np.uint8), prepared, resolved_threshold, resolved_method


def summarize_object_skeleton(local_mask: np.ndarray, spacing: tuple[float, ...]) -> tuple[int, float, float, int, int, pd.DataFrame]:
    skeleton_mask = morphology.skeletonize(local_mask)
    skeleton_pixels = int(np.count_nonzero(skeleton_mask))
    if skeleton_pixels < 2:
        return 0, 0.0, 0.0, 0, 0, pd.DataFrame()

    try:
        skeleton = Skeleton(skeleton_mask, spacing=spacing)
        skeleton_df = summarize_skeleton(skeleton)
    except ValueError:
        return 0, 0.0, 0.0, 0, 0, pd.DataFrame()

    if skeleton_df.empty:
        return 0, 0.0, 0.0, 0, 0, pd.DataFrame()

    branch_length_column = _find_branch_length_column(skeleton_df)
    degrees = np.asarray(skeleton.degrees)
    branch_count = int(len(skeleton_df))
    total_branch_length = float(skeleton_df[branch_length_column].sum())
    mean_branch_length = float(skeleton_df[branch_length_column].mean())
    junction_count = int(np.sum(degrees >= 3))
    endpoint_count = int(np.sum(degrees == 1))
    return branch_count, total_branch_length, mean_branch_length, junction_count, endpoint_count, skeleton_df


def safe_region_metric(region: measure._regionprops.RegionProperties, attribute: str, default: float = 0.0) -> float:
    try:
        value = getattr(region, attribute)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return default
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric_value):
        return default
    return numeric_value


def analyze_mitochondria(
    mito_labels: np.ndarray,
    mito_image: np.ndarray,
    cell_labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    measurement_spacing = spacing_zyx if mito_labels.ndim == 3 else spacing_zyx[1:]
    voxel_volume = float(np.prod(measurement_spacing))
    object_rows = []
    skeleton_frames = []

    for region in measure.regionprops(mito_labels, intensity_image=mito_image):
        local_mask = region.image
        overlap = cell_labels[region.slice][local_mask]
        positive_overlap = overlap[overlap > 0]
        parent_cell_id = int(np.bincount(positive_overlap).argmax()) if positive_overlap.size else 0

        branch_count, total_branch_length, mean_branch_length, junction_count, endpoint_count, object_skeleton_df = summarize_object_skeleton(
            local_mask,
            measurement_spacing,
        )
        if not object_skeleton_df.empty:
            object_skeleton_df = object_skeleton_df.copy()
            object_skeleton_df.insert(0, "mito_id", int(region.label))
            object_skeleton_df.insert(1, "cell_id", parent_cell_id)
            skeleton_frames.append(object_skeleton_df)

        try:
            inertia_eigvals = np.sort(np.asarray(region.inertia_tensor_eigvals, dtype=float))[::-1]
        except (ValueError, ZeroDivisionError, FloatingPointError, TypeError):
            inertia_eigvals = np.asarray([], dtype=float)
        equivalent_diameter_um = safe_region_metric(region, "equivalent_diameter_area")
        major_axis_um = safe_region_metric(region, "axis_major_length")
        minor_axis_um = safe_region_metric(region, "axis_minor_length")
        row = {
            "mito_id": int(region.label),
            "cell_id": parent_cell_id,
            "voxel_count": float(region.area),
            "volume_um3": float(region.area) * voxel_volume,
            "equivalent_diameter_um": equivalent_diameter_um,
            "major_axis_um": major_axis_um,
            "minor_axis_um": minor_axis_um,
            "aspect_ratio": major_axis_um / max(minor_axis_um, 1e-6),
            "extent": safe_region_metric(region, "extent"),
            "solidity": safe_region_metric(region, "solidity"),
            "euler_number": int(safe_region_metric(region, "euler_number")),
            "mean_intensity": safe_region_metric(region, "intensity_mean"),
            "max_intensity": safe_region_metric(region, "max_intensity"),
            "min_intensity": safe_region_metric(region, "min_intensity"),
            "skeleton_branch_count": branch_count,
            "skeleton_total_branch_length_um": total_branch_length,
            "skeleton_mean_branch_length_um": mean_branch_length,
            "junction_count": junction_count,
            "endpoint_count": endpoint_count,
            "largest_inertia_eigval": float(inertia_eigvals[0]) if inertia_eigvals.size > 0 else 0.0,
            "middle_inertia_eigval": float(inertia_eigvals[1]) if inertia_eigvals.size > 1 else 0.0,
            "smallest_inertia_eigval": float(inertia_eigvals[2]) if inertia_eigvals.size > 2 else 0.0,
        }
        for index, value in enumerate(region.centroid):
            row[f"centroid_axis_{index}"] = float(value)
        for index, value in enumerate(region.bbox):
            row[f"bbox_{index}"] = int(value)
        object_rows.append(row)

    mito_df = pd.DataFrame(object_rows)

    if skeleton_frames:
        skeleton_df = pd.concat(skeleton_frames, ignore_index=True)
        if "branch-type" in skeleton_df.columns:
            skeleton_df = skeleton_df.rename(columns={"branch-type": "branch_type"})
        if "branch-distance" in skeleton_df.columns:
            skeleton_df = skeleton_df.rename(columns={"branch-distance": "branch_length_um"})
        if "euclidean-distance" in skeleton_df.columns:
            skeleton_df = skeleton_df.rename(columns={"euclidean-distance": "euclidean_length_um"})
        if "skeleton-id" in skeleton_df.columns:
            skeleton_df = skeleton_df.rename(columns={"skeleton-id": "skeleton_id"})
    else:
        skeleton_df = pd.DataFrame()

    per_cell_rows = []
    for cell_id in sorted(int(value) for value in np.unique(cell_labels) if value > 0):
        cell_mask = cell_labels == cell_id
        cell_volume_um3 = float(np.sum(cell_mask)) * voxel_volume
        cell_objects = mito_df[mito_df["cell_id"] == cell_id].copy() if not mito_df.empty else pd.DataFrame()
        total_mito_volume = float(cell_objects["volume_um3"].sum()) if not cell_objects.empty else 0.0
        component_count = int(len(cell_objects))
        junction_count = int(cell_objects["junction_count"].sum()) if not cell_objects.empty else 0
        largest_fraction = float(cell_objects["volume_um3"].max() / total_mito_volume) if total_mito_volume > 0 else 0.0
        per_cell_rows.append(
            {
                "cell_id": cell_id,
                "cell_volume_um3": cell_volume_um3,
                "mitochondria_count": component_count,
                "connected_component_count": component_count,
                "total_mito_volume_um3": total_mito_volume,
                "mean_mito_volume_um3": float(cell_objects["volume_um3"].mean()) if not cell_objects.empty else 0.0,
                "median_mito_volume_um3": float(cell_objects["volume_um3"].median()) if not cell_objects.empty else 0.0,
                "mito_volume_fraction": total_mito_volume / cell_volume_um3 if cell_volume_um3 > 0 else 0.0,
                "total_branch_length_um": float(cell_objects["skeleton_total_branch_length_um"].sum()) if not cell_objects.empty else 0.0,
                "mean_branch_length_um": float(cell_objects["skeleton_mean_branch_length_um"].mean()) if not cell_objects.empty else 0.0,
                "branch_count": int(cell_objects["skeleton_branch_count"].sum()) if not cell_objects.empty else 0,
                "junction_count": junction_count,
                "endpoint_count": int(cell_objects["endpoint_count"].sum()) if not cell_objects.empty else 0,
                "mean_aspect_ratio": float(cell_objects["aspect_ratio"].mean()) if not cell_objects.empty else 0.0,
                "mean_solidity": float(cell_objects["solidity"].mean()) if not cell_objects.empty else 0.0,
                "largest_component_fraction": largest_fraction,
                "connectivity_index": float(junction_count / max(component_count, 1)),
            }
        )
    per_cell_df = pd.DataFrame(per_cell_rows)

    summary_df = pd.DataFrame(
        [
            {
                "mitochondria_count": int(len(mito_df)),
                "cell_count": int(len(per_cell_df)),
                "total_mito_volume_um3": float(mito_df["volume_um3"].sum()) if not mito_df.empty else 0.0,
                "mean_mito_volume_um3": float(mito_df["volume_um3"].mean()) if not mito_df.empty else 0.0,
                "mean_aspect_ratio": float(mito_df["aspect_ratio"].mean()) if not mito_df.empty else 0.0,
                "mean_solidity": float(mito_df["solidity"].mean()) if not mito_df.empty else 0.0,
                "total_branch_length_um": float(mito_df["skeleton_total_branch_length_um"].sum()) if not mito_df.empty else 0.0,
                "mean_branch_length_um": float(mito_df["skeleton_mean_branch_length_um"].mean()) if not mito_df.empty else 0.0,
                "total_junction_count": int(mito_df["junction_count"].sum()) if not mito_df.empty else 0,
                "total_endpoint_count": int(mito_df["endpoint_count"].sum()) if not mito_df.empty else 0,
                "mean_mito_volume_fraction_per_cell": float(per_cell_df["mito_volume_fraction"].mean()) if not per_cell_df.empty else 0.0,
                "mean_connectivity_index_per_cell": float(per_cell_df["connectivity_index"].mean()) if not per_cell_df.empty else 0.0,
            }
        ]
    )
    return mito_df, skeleton_df, per_cell_df, summary_df


def run_analysis(
    timepoint: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    nuclei_channel: int,
    cell_channel: int,
    mito_channel: int,
    skip_leading_z_slices: int,
    z_mode: str,
    z_index: int,
    cell_threshold_method: str,
    cell_threshold: float | None,
    cell_threshold_offset: float,
    cell_min_size: int,
    cell_closing_radius: int,
    threshold_method: str,
    threshold: float | None,
    threshold_offset: float,
    local_block_size: int,
    min_size: int,
    apply_watershed_split: bool,
    watershed_min_distance: int,
) -> dict[str, object]:
    analysis_timepoint = trim_leading_z_slices(timepoint, skip_leading_z_slices)
    preview_channels = choose_channel_preview(analysis_timepoint, z_mode, z_index)
    nuclei_image = analysis_timepoint[nuclei_channel]
    cell_image = analysis_timepoint[cell_channel]
    mito_image = analysis_timepoint[mito_channel]

    nuclei_labels, nuclei_prepared = segment_nuclei(nuclei_image)
    cell_labels, cell_prepared, cell_resolved_threshold, cell_resolved_method = segment_cells(
        cell_image,
        nuclei_labels,
        threshold_method=cell_threshold_method,
        threshold=cell_threshold,
        threshold_offset=cell_threshold_offset,
        min_size=cell_min_size,
        closing_radius=cell_closing_radius,
    )
    mito_labels, mito_binary, mito_prepared, resolved_threshold, resolved_method = segment_mitochondria(
        mito_image,
        cell_labels,
        threshold_method=threshold_method,
        threshold=threshold,
        threshold_offset=threshold_offset,
        local_block_size=local_block_size,
        min_size=min_size,
        apply_watershed_split=apply_watershed_split,
        watershed_min_distance=watershed_min_distance,
    )
    mito_df, skeleton_df, per_cell_df, summary_df = analyze_mitochondria(
        mito_labels,
        mito_image,
        cell_labels,
        spacing_zyx,
    )
    summary_df = summary_df.copy()
    summary_df["threshold_method"] = resolved_method
    summary_df["threshold_value"] = resolved_threshold
    summary_df["cell_threshold_method"] = cell_resolved_method
    summary_df["cell_threshold_value"] = cell_resolved_threshold
    summary_df["preview_mode"] = z_mode
    summary_df["preview_z_index"] = z_index
    summary_df["skip_leading_z_slices"] = int(max(0, skip_leading_z_slices))
    summary_df["watershed_split"] = bool(apply_watershed_split and mito_image.ndim == 3)
    summary_df["watershed_min_distance"] = int(watershed_min_distance)
    return {
        "analysis_timepoint": analysis_timepoint,
        "preview_channels": preview_channels,
        "nuclei_prepared": nuclei_prepared,
        "cell_prepared": cell_prepared,
        "mito_prepared": mito_prepared,
        "nuclei_labels": nuclei_labels,
        "cell_labels": cell_labels,
        "mito_labels": mito_labels,
        "mito_binary": mito_binary,
        "mito_df": mito_df,
        "skeleton_df": skeleton_df,
        "per_cell_df": per_cell_df,
        "summary_df": summary_df,
        "resolved_threshold": resolved_threshold,
        "resolved_method": resolved_method,
    }


def run_mito_analysis_with_fixed_context(
    mito_image: np.ndarray,
    cell_labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    threshold_method: str,
    threshold: float | None,
    threshold_offset: float,
    local_block_size: int,
    min_size: int,
    apply_watershed_split: bool,
    watershed_min_distance: int,
) -> dict[str, object]:
    mito_labels, mito_binary, mito_prepared, resolved_threshold, resolved_method = segment_mitochondria(
        mito_image,
        cell_labels,
        threshold_method=threshold_method,
        threshold=threshold,
        threshold_offset=threshold_offset,
        local_block_size=local_block_size,
        min_size=min_size,
        apply_watershed_split=apply_watershed_split,
        watershed_min_distance=watershed_min_distance,
    )
    mito_df, skeleton_df, per_cell_df, summary_df = analyze_mitochondria(
        mito_labels,
        mito_image,
        cell_labels,
        spacing_zyx,
    )
    summary_df = summary_df.copy()
    summary_df["threshold_method"] = resolved_method
    summary_df["threshold_value"] = resolved_threshold
    summary_df["watershed_split"] = bool(apply_watershed_split and mito_image.ndim == 3)
    summary_df["watershed_min_distance"] = int(watershed_min_distance)
    return {
        "mito_labels": mito_labels,
        "mito_binary": mito_binary,
        "mito_prepared": mito_prepared,
        "mito_df": mito_df,
        "skeleton_df": skeleton_df,
        "per_cell_df": per_cell_df,
        "summary_df": summary_df,
        "resolved_threshold": resolved_threshold,
        "resolved_method": resolved_method,
    }


def score_sweep_result(per_cell_df: pd.DataFrame, mito_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, float]:
    row = summary_df.iloc[0]
    max_largest_component_fraction = (
        float(per_cell_df["largest_component_fraction"].max())
        if not per_cell_df.empty and "largest_component_fraction" in per_cell_df
        else 1.0
    )
    mean_largest_component_fraction = (
        float(per_cell_df["largest_component_fraction"].mean())
        if not per_cell_df.empty and "largest_component_fraction" in per_cell_df
        else 1.0
    )
    mitochondria_count = int(row["mitochondria_count"])
    total_volume = float(row["total_mito_volume_um3"])
    connectivity = float(row["mean_connectivity_index_per_cell"])
    branch_length = float(row["total_branch_length_um"])
    cells_with_mito = int((per_cell_df["mitochondria_count"] > 0).sum()) if not per_cell_df.empty else 0
    return {
        "mitochondria_count": mitochondria_count,
        "total_mito_volume_um3": total_volume,
        "mean_connectivity_index_per_cell": connectivity,
        "total_branch_length_um": branch_length,
        "max_largest_component_fraction": max_largest_component_fraction,
        "mean_largest_component_fraction": mean_largest_component_fraction,
        "cells_with_mito": cells_with_mito,
        "mean_object_volume_um3": float(mito_df["volume_um3"].mean()) if not mito_df.empty else 0.0,
    }


def run_parameter_sweep(
    timepoint: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    nuclei_channel: int,
    cell_channel: int,
    mito_channel: int,
    skip_leading_z_slices: int,
    z_mode: str,
    z_index: int,
    cell_threshold_method: str,
    cell_threshold: float | None,
    cell_threshold_offset: float,
    cell_min_size: int,
    cell_closing_radius: int,
    top_k: int,
    apply_watershed_split: bool,
    watershed_min_distance: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_timepoint = trim_leading_z_slices(timepoint, skip_leading_z_slices)
    nuclei_image = analysis_timepoint[nuclei_channel]
    cell_image = analysis_timepoint[cell_channel]
    mito_image = analysis_timepoint[mito_channel]

    nuclei_labels, _ = segment_nuclei(nuclei_image)
    cell_labels, _, _, _ = segment_cells(
        cell_image,
        nuclei_labels,
        threshold_method=cell_threshold_method,
        threshold=cell_threshold,
        threshold_offset=cell_threshold_offset,
        min_size=cell_min_size,
        closing_radius=cell_closing_radius,
    )

    settings = []
    for threshold_offset in (0.02, 0.05, 0.08):
        for local_block_size in (31, 51):
            for min_size in (30, 60):
                settings.append({
                    "threshold_method": "local",
                    "threshold": None,
                    "threshold_offset": threshold_offset,
                    "local_block_size": local_block_size,
                    "min_size": min_size,
                })
    for threshold_method in ("slice_otsu", "otsu"):
        for threshold_offset in (0.02, 0.05, 0.08):
            for min_size in (30, 60):
                settings.append({
                    "threshold_method": threshold_method,
                    "threshold": None,
                    "threshold_offset": threshold_offset,
                    "local_block_size": 31,
                    "min_size": min_size,
                })

    sweep_rows = []
    for setting in settings:
        result = run_mito_analysis_with_fixed_context(
            mito_image,
            cell_labels,
            spacing_zyx=spacing_zyx,
            threshold_method=setting["threshold_method"],
            threshold=setting["threshold"],
            threshold_offset=setting["threshold_offset"],
            local_block_size=setting["local_block_size"],
            min_size=setting["min_size"],
            apply_watershed_split=apply_watershed_split,
            watershed_min_distance=watershed_min_distance,
        )
        metrics = score_sweep_result(result["per_cell_df"], result["mito_df"], result["summary_df"])
        sweep_rows.append({
            "threshold_method": setting["threshold_method"],
            "threshold_offset": setting["threshold_offset"],
            "local_block_size": setting["local_block_size"],
            "mito_min_size": setting["min_size"],
            "resolved_threshold": float(result["resolved_threshold"]),
            "watershed_split": bool(apply_watershed_split and mito_image.ndim == 3),
            "watershed_min_distance": int(watershed_min_distance),
            **metrics,
        })

    sweep_df = pd.DataFrame(sweep_rows)
    if sweep_df.empty:
        return sweep_df, sweep_df

    count_max = max(float(sweep_df["mitochondria_count"].max()), 1.0)
    volume_max = max(float(sweep_df["total_mito_volume_um3"].max()), 1.0)
    branch_max = max(float(sweep_df["total_branch_length_um"].max()), 1.0)
    cells_max = max(float(sweep_df["cells_with_mito"].max()), 1.0)

    sweep_df["count_ratio"] = sweep_df["mitochondria_count"] / count_max
    sweep_df["volume_ratio"] = sweep_df["total_mito_volume_um3"] / volume_max
    sweep_df["branch_ratio"] = sweep_df["total_branch_length_um"] / branch_max
    sweep_df["cells_ratio"] = sweep_df["cells_with_mito"] / cells_max
    sweep_df["dominance_penalty"] = 0.65 * sweep_df["max_largest_component_fraction"] + 0.35 * sweep_df["mean_largest_component_fraction"]
    sweep_df["plausibility_score"] = (
        0.55 * sweep_df["dominance_penalty"]
        - 0.20 * sweep_df["count_ratio"]
        - 0.10 * sweep_df["volume_ratio"]
        - 0.10 * sweep_df["branch_ratio"]
        - 0.05 * sweep_df["cells_ratio"]
    )

    filtered_df = sweep_df[(sweep_df["count_ratio"] >= 0.15) & (sweep_df["volume_ratio"] >= 0.10)].copy()
    if filtered_df.empty:
        filtered_df = sweep_df.copy()

    ranked_df = filtered_df.sort_values(
        ["plausibility_score", "max_largest_component_fraction", "mean_largest_component_fraction", "mitochondria_count"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    ranked_df.insert(0, "rank", np.arange(1, len(ranked_df) + 1))
    return sweep_df, ranked_df.head(max(1, top_k)).copy()


def save_sweep_outputs(output_dir: Path, sweep_df: pd.DataFrame, top_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_dir / "mitochondria_sweep_results.csv", index=False)
    top_df.to_csv(output_dir / "mitochondria_sweep_top_ranked.csv", index=False)
    with (output_dir / "mitochondria_sweep_recommendation.txt").open("w", encoding="utf-8") as handle:
        if top_df.empty:
            handle.write("No valid sweep results were produced.\n")
            return
        best = top_df.iloc[0]
        handle.write("Top-ranked mitochondria segmentation setting\n")
        handle.write("=========================================\n\n")
        handle.write(
            f"threshold_method={best['threshold_method']}\n"
            f"threshold_offset={best['threshold_offset']:.2f}\n"
            f"local_block_size={int(best['local_block_size'])}\n"
            f"mito_min_size={int(best['mito_min_size'])}\n"
            f"resolved_threshold={best['resolved_threshold']:.4f}\n\n"
        )
        handle.write("Ranking favors lower dominant-component fraction while keeping count, volume, branch length, and cell coverage plausible.\n")


def summarize_condition_metrics(per_cell_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mitochondria_count",
        "total_mito_volume_um3",
        "mean_mito_volume_um3",
        "mito_volume_fraction",
        "total_branch_length_um",
        "mean_branch_length_um",
        "branch_count",
        "junction_count",
        "endpoint_count",
        "mean_aspect_ratio",
        "mean_solidity",
        "largest_component_fraction",
        "connectivity_index",
    ]
    existing_metrics = [column for column in metric_columns if column in per_cell_df.columns]
    rows = []
    for condition, group in per_cell_df.groupby("condition"):
        row = {"condition": condition, "cell_count": int(len(group))}
        for metric in existing_metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_conditions(condition_summary_df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    if condition_summary_df.empty:
        return pd.DataFrame()
    normalized = [condition.upper() for condition in conditions]
    available = {row["condition"]: row for _, row in condition_summary_df.iterrows()}
    if not all(condition in available for condition in normalized[:2]):
        return pd.DataFrame()
    left = available[normalized[0]]
    right = available[normalized[1]]
    comparison_rows = []
    for column in condition_summary_df.columns:
        if column == "condition" or column == "cell_count" or not column.endswith("_mean"):
            continue
        comparison_rows.append(
            {
                "metric": column[:-5],
                f"{normalized[0]}_mean": float(left[column]),
                f"{normalized[1]}_mean": float(right[column]),
                "difference": float(left[column]) - float(right[column]),
                "ratio": float(left[column]) / float(right[column]) if float(right[column]) != 0 else np.nan,
            }
        )
    return pd.DataFrame(comparison_rows)


def save_batch_condition_boxplots(per_cell_df: pd.DataFrame, output_dir: Path, conditions: list[str]) -> list[Path]:
    if per_cell_df.empty or "condition" not in per_cell_df.columns:
        return []

    ordered_conditions = [condition.upper() for condition in conditions if condition.upper() in set(per_cell_df["condition"].astype(str))]
    if not ordered_conditions:
        ordered_conditions = sorted(set(per_cell_df["condition"].astype(str)))

    available_metrics = [(metric, label) for metric, label in BATCH_PLOT_METRICS if metric in per_cell_df.columns]
    if not available_metrics:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths = []
    colors = {condition: color for condition, color in zip(ordered_conditions, ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"])}

    ncols = 2
    nrows = int(np.ceil(len(available_metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.default_rng(42)
    for ax, (metric, label) in zip(axes, available_metrics):
        grouped_values = []
        for condition in ordered_conditions:
            values = per_cell_df.loc[per_cell_df["condition"] == condition, metric].astype(float).dropna().to_numpy()
            grouped_values.append(values)

        positions = np.arange(1, len(ordered_conditions) + 1)
        ax.boxplot(
            grouped_values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.5},
            boxprops={"edgecolor": "#444444", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 1.1},
            capprops={"color": "#555555", "linewidth": 1.1},
        )

        for patch, condition in zip(ax.artists, ordered_conditions):
            patch.set_facecolor(colors[condition])
            patch.set_alpha(0.35)

        for idx, (condition, values) in enumerate(zip(ordered_conditions, grouped_values), start=1):
            if len(values) == 0:
                continue
            x = rng.normal(loc=idx, scale=0.05, size=len(values))
            ax.scatter(x, values, s=18, alpha=0.7, color=colors[condition], edgecolors="white", linewidths=0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_conditions)
        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(available_metrics):]:
        ax.axis("off")

    fig.suptitle("Mitochondrial per-cell metrics by condition", fontsize=16)
    combined_path = output_dir / "batch_condition_boxplots.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)
    created_paths.append(combined_path)

    for metric, label in available_metrics:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        grouped_values = []
        positions = np.arange(1, len(ordered_conditions) + 1)
        for condition in ordered_conditions:
            values = per_cell_df.loc[per_cell_df["condition"] == condition, metric].astype(float).dropna().to_numpy()
            grouped_values.append(values)

        ax.boxplot(
            grouped_values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.5},
            boxprops={"edgecolor": "#444444", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 1.1},
            capprops={"color": "#555555", "linewidth": 1.1},
        )
        for patch, condition in zip(ax.artists, ordered_conditions):
            patch.set_facecolor(colors[condition])
            patch.set_alpha(0.35)
        for idx, (condition, values) in enumerate(zip(ordered_conditions, grouped_values), start=1):
            if len(values) == 0:
                continue
            x = rng.normal(loc=idx, scale=0.05, size=len(values))
            ax.scatter(x, values, s=20, alpha=0.75, color=colors[condition], edgecolors="white", linewidths=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_conditions)
        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        safe_name = metric.replace("/", "_")
        metric_path = output_dir / f"boxplot_{safe_name}.png"
        fig.savefig(metric_path, dpi=200)
        plt.close(fig)
        created_paths.append(metric_path)

    return created_paths


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low = float(np.percentile(data, 1))
    high = float(np.percentile(data, 99.5))
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, low, high)
    return (data - low) / (high - low)


def project_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return np.max(image, axis=0)
    return image


def save_segmentation_qc_figure(
    output_dir: Path,
    file_label: str,
    raw_timepoint: np.ndarray,
    cell_channel: int,
    mito_channel: int,
    cell_labels: np.ndarray,
    mito_binary: np.ndarray,
    mito_labels: np.ndarray,
) -> Path:
    cell_image = project_for_display(raw_timepoint[cell_channel])
    mito_image = project_for_display(raw_timepoint[mito_channel])
    cell_mask = project_for_display(cell_labels > 0)
    mito_mask = project_for_display(mito_binary > 0)
    mito_label_projection = project_for_display(mito_labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(normalize_for_display(cell_image), cmap="gray")
    axes[0].contour(cell_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
    axes[0].set_title("Cell hull on channel 2")

    axes[1].imshow(normalize_for_display(mito_image), cmap="magma")
    axes[1].contour(cell_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
    axes[1].contour(mito_mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.9)
    axes[1].set_title("Mito mask inside hull")

    axes[2].imshow(normalize_for_display(mito_image), cmap="gray")
    axes[2].imshow(np.ma.masked_where(mito_label_projection <= 0, mito_label_projection), cmap="nipy_spectral", alpha=0.75)
    axes[2].contour(cell_mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
    axes[2].set_title("Mito labels")

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(f"Segmentation QC: {file_label}", fontsize=15)
    output_path = output_dir / "segmentation_qc.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


class BatchProgressViewer:
    def __init__(self) -> None:
        backend = str(plt.get_backend()).lower()
        self.active = not backend.startswith("agg")
        self.fig = None
        self.axes = None
        if not self.active:
            return
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    def update(
        self,
        file_label: str,
        raw_timepoint: np.ndarray,
        cell_channel: int,
        mito_channel: int,
        cell_labels: np.ndarray,
        mito_binary: np.ndarray,
        mito_labels: np.ndarray,
        summary_df: pd.DataFrame,
        current_index: int,
        total_count: int,
    ) -> None:
        if not self.active or self.fig is None or self.axes is None:
            return

        cell_image = project_for_display(raw_timepoint[cell_channel])
        mito_image = project_for_display(raw_timepoint[mito_channel])
        cell_mask = project_for_display(cell_labels > 0)
        mito_mask = project_for_display(mito_binary > 0)
        mito_label_projection = project_for_display(mito_labels)

        for ax in self.axes:
            ax.clear()

        self.axes[0].imshow(normalize_for_display(cell_image), cmap="gray")
        self.axes[0].contour(cell_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        self.axes[0].set_title("Cell hull on channel 2")

        self.axes[1].imshow(normalize_for_display(mito_image), cmap="magma")
        self.axes[1].contour(cell_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
        self.axes[1].contour(mito_mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.9)
        self.axes[1].set_title("Mito mask inside hull")

        self.axes[2].imshow(normalize_for_display(mito_image), cmap="gray")
        self.axes[2].imshow(np.ma.masked_where(mito_label_projection <= 0, mito_label_projection), cmap="nipy_spectral", alpha=0.75)
        self.axes[2].contour(cell_mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
        self.axes[2].set_title("Mito labels")

        for ax in self.axes:
            ax.set_axis_off()

        summary_row = summary_df.iloc[0] if not summary_df.empty else {}
        title = f"Batch QC {current_index}/{total_count}: {file_label}"
        metrics_line = ""
        if len(summary_df):
            metrics_line = (
                f"count={int(summary_row['mitochondria_count'])} | "
                f"volume={float(summary_row['total_mito_volume_um3']):.2f} um^3 | "
                f"branch={float(summary_row['total_branch_length_um']):.2f} um"
            )
        self.fig.suptitle(title + ("\n" + metrics_line if metrics_line else ""), fontsize=14)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        if not self.active or self.fig is None:
            return
        plt.ioff()
        plt.close(self.fig)


def save_batch_qc_contact_sheet(batch_file_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if batch_file_df.empty or "file_stem" not in batch_file_df.columns:
        return None

    qc_paths = []
    labels = []
    for _, row in batch_file_df.iterrows():
        qc_path = output_dir / str(row["file_stem"]) / "segmentation_qc.png"
        if qc_path.is_file():
            qc_paths.append(qc_path)
            labels.append(str(row.get("file_name", row["file_stem"])))

    if not qc_paths:
        return None

    ncols = 3
    nrows = int(np.ceil(len(qc_paths) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, qc_path, label in zip(axes, qc_paths, labels):
        image = plt.imread(qc_path)
        ax.imshow(image)
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()

    for ax in axes[len(qc_paths):]:
        ax.axis("off")

    fig.suptitle("Batch segmentation QC", fontsize=16)
    output_path = output_dir / "batch_segmentation_qc_contact_sheet.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_batch_analysis(
    input_path: Path,
    output_dir: Path,
    conditions: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = collect_batch_files(input_path, conditions)
    if not files:
        raise SystemExit(f"No supported image files matching conditions {conditions} were found under {input_path}")

    print(f"Found {len(files)} files for conditions {', '.join(conditions)} under {input_path}", flush=True)

    file_rows = []
    per_cell_frames = []
    per_object_frames = []
    failure_rows = []
    progress_viewer = BatchProgressViewer() if args.show_viewer else None

    for file_index, image_file in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] Processing {image_file.name}", flush=True)
        condition = infer_condition(image_file, conditions)
        try:
            dataset = load_dataset(image_file, fallback_xy_um=args.pixel_size_um, fallback_z_um=args.z_step_um)
            if args.time_index < 0 or args.time_index >= dataset.data.shape[0]:
                raise RuntimeError(f"time-index {args.time_index} is outside dataset range for {image_file.name}")
            timepoint = dataset.data[args.time_index]
            required_index = max(args.nuclei_channel, args.cell_channel, args.mito_channel)
            if required_index >= timepoint.shape[0]:
                raise RuntimeError(f"Dataset {image_file.name} has only {timepoint.shape[0]} channels, requested {required_index}")

            result = run_analysis(
                timepoint,
                spacing_zyx=dataset.spacing_zyx,
                nuclei_channel=args.nuclei_channel,
                cell_channel=args.cell_channel,
                mito_channel=args.mito_channel,
                skip_leading_z_slices=args.skip_leading_z_slices,
                z_mode=args.z_mode,
                z_index=args.z_index,
                cell_threshold_method=args.cell_threshold_method,
                cell_threshold=args.cell_threshold,
                cell_threshold_offset=args.cell_threshold_offset,
                cell_min_size=args.cell_min_size,
                cell_closing_radius=args.cell_closing_radius,
                threshold_method=args.mito_threshold_method,
                threshold=args.mito_threshold,
                threshold_offset=args.mito_threshold_offset,
                local_block_size=args.mito_local_block_size,
                min_size=args.mito_min_size,
                apply_watershed_split=not args.disable_mito_watershed,
                watershed_min_distance=args.mito_watershed_min_distance,
            )
        except Exception as exc:
            print(f"    Skipping {image_file.name}: {exc}", flush=True)
            failure_rows.append(
                {
                    "condition": condition,
                    "file_name": image_file.name,
                    "file_stem": image_file.stem,
                    "error": str(exc),
                }
            )
            continue

        file_output_dir = output_dir / image_file.stem
        save_outputs(
            file_output_dir,
            result["preview_channels"],
            timepoint,
            result["nuclei_labels"],
            result["cell_labels"],
            result["mito_labels"],
            result["mito_binary"],
            result["mito_df"],
            result["skeleton_df"],
            result["per_cell_df"],
            result["summary_df"],
        )
        save_segmentation_qc_figure(
            file_output_dir,
            image_file.name,
            timepoint,
            cell_channel=args.cell_channel,
            mito_channel=args.mito_channel,
            cell_labels=result["cell_labels"],
            mito_binary=result["mito_binary"],
            mito_labels=result["mito_labels"],
        )
        if progress_viewer is not None:
            progress_viewer.update(
                image_file.name,
                timepoint,
                cell_channel=args.cell_channel,
                mito_channel=args.mito_channel,
                cell_labels=result["cell_labels"],
                mito_binary=result["mito_binary"],
                mito_labels=result["mito_labels"],
                summary_df=result["summary_df"],
                current_index=file_index,
                total_count=len(files),
            )

        summary_row = result["summary_df"].iloc[0]
        print(
            "    "
            + f"mitos={int(summary_row['mitochondria_count'])} "
            + f"volume_um3={float(summary_row['total_mito_volume_um3']):.2f} "
            + f"branch_um={float(summary_row['total_branch_length_um']):.2f}",
            flush=True,
        )

        summary_row_dict = summary_row.to_dict()
        summary_row_dict.update(
            {
                "condition": condition,
                "file_name": image_file.name,
                "file_stem": image_file.stem,
                "channel_names": ",".join(dataset.channel_names),
                "spacing_z_um": dataset.spacing_zyx[0],
                "spacing_y_um": dataset.spacing_zyx[1],
                "spacing_x_um": dataset.spacing_zyx[2],
            }
        )
        file_rows.append(summary_row_dict)

        per_cell_df = result["per_cell_df"].copy()
        per_cell_df.insert(0, "condition", condition)
        per_cell_df.insert(1, "file_name", image_file.name)
        per_cell_frames.append(per_cell_df)

        per_object_df = result["mito_df"].copy()
        per_object_df.insert(0, "condition", condition)
        per_object_df.insert(1, "file_name", image_file.name)
        per_object_frames.append(per_object_df)

    batch_file_df = pd.DataFrame(file_rows).sort_values(["condition", "file_name"]).reset_index(drop=True)
    batch_per_cell_df = pd.concat(per_cell_frames, ignore_index=True) if per_cell_frames else pd.DataFrame()
    batch_per_object_df = pd.concat(per_object_frames, ignore_index=True) if per_object_frames else pd.DataFrame()
    condition_summary_df = summarize_condition_metrics(batch_per_cell_df) if not batch_per_cell_df.empty else pd.DataFrame()
    condition_comparison_df = compare_conditions(condition_summary_df, conditions)

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_file_df.to_csv(output_dir / "batch_file_summary.csv", index=False)
    batch_per_cell_df.to_csv(output_dir / "batch_per_cell_summary.csv", index=False)
    batch_per_object_df.to_csv(output_dir / "batch_mito_objects.csv", index=False)
    condition_summary_df.to_csv(output_dir / "batch_condition_summary.csv", index=False)
    condition_comparison_df.to_csv(output_dir / "batch_condition_comparison.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(output_dir / "batch_failures.csv", index=False)
    save_batch_condition_boxplots(batch_per_cell_df, output_dir, conditions)
    save_batch_qc_contact_sheet(batch_file_df, output_dir)
    if progress_viewer is not None:
        progress_viewer.close()
    print(f"Batch analysis finished. Outputs saved to {output_dir}", flush=True)
    return batch_file_df, batch_per_cell_df, condition_summary_df, condition_comparison_df


def save_outputs(
    output_dir: Path,
    preview_channels: np.ndarray,
    raw_timepoint: np.ndarray,
    nuclei_labels: np.ndarray,
    cell_labels: np.ndarray,
    mito_labels: np.ndarray,
    mito_binary: np.ndarray,
    mito_df: pd.DataFrame,
    skeleton_df: pd.DataFrame,
    per_cell_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_dir / "raw_timepoint_czyx.tif", raw_timepoint.astype(np.float32))
    tifffile.imwrite(output_dir / "preview_channels.tif", preview_channels.astype(np.float32))
    tifffile.imwrite(output_dir / "nuclei_labels.tif", nuclei_labels.astype(np.uint16))
    tifffile.imwrite(output_dir / "cell_labels.tif", cell_labels.astype(np.uint16))
    tifffile.imwrite(output_dir / "mitochondria_labels.tif", mito_labels.astype(np.uint16))
    tifffile.imwrite(output_dir / "mitochondria_binary.tif", mito_binary.astype(np.uint8))
    mito_df.to_csv(output_dir / "mitochondria_objects.csv", index=False)
    skeleton_df.to_csv(output_dir / "mitochondria_skeleton.csv", index=False)
    per_cell_df.to_csv(output_dir / "mitochondria_per_cell_summary.csv", index=False)
    summary_df.to_csv(output_dir / "analysis_summary.csv", index=False)


def show_in_napari(
    timepoint: np.ndarray,
    channel_names: list[str],
    spacing_zyx: tuple[float, float, float],
    output_dir: Path,
    nuclei_channel: int,
    cell_channel: int,
    mito_channel: int,
    initial_skip_leading_z_slices: int,
    initial_z_mode: str,
    initial_z_index: int,
    initial_cell_threshold_method: str,
    initial_cell_threshold: float | None,
    initial_cell_threshold_offset: float,
    initial_cell_min_size: int,
    initial_cell_closing_radius: int,
    initial_threshold_method: str,
    initial_threshold: float | None,
    initial_threshold_offset: float,
    initial_local_block_size: int,
    initial_mito_min_size: int,
    initial_apply_watershed_split: bool,
    initial_watershed_min_distance: int,
) -> None:
    import napari

    state = {
        "skip_leading_z_slices": initial_skip_leading_z_slices,
        "z_mode": initial_z_mode,
        "z_index": initial_z_index,
        "cell_threshold_method": initial_cell_threshold_method,
        "cell_threshold": initial_cell_threshold,
        "cell_threshold_offset": initial_cell_threshold_offset,
        "cell_min_size": initial_cell_min_size,
        "cell_closing_radius": initial_cell_closing_radius,
        "threshold_method": initial_threshold_method,
        "threshold": initial_threshold,
        "threshold_offset": initial_threshold_offset,
        "local_block_size": initial_local_block_size,
        "mito_min_size": initial_mito_min_size,
        "apply_watershed_split": initial_apply_watershed_split,
        "watershed_min_distance": initial_watershed_min_distance,
    }
    result = run_analysis(
        timepoint,
        spacing_zyx=spacing_zyx,
        nuclei_channel=nuclei_channel,
        cell_channel=cell_channel,
        mito_channel=mito_channel,
        skip_leading_z_slices=state["skip_leading_z_slices"],
        z_mode=state["z_mode"],
        z_index=state["z_index"],
        cell_threshold_method=state["cell_threshold_method"],
        cell_threshold=state["cell_threshold"],
        cell_threshold_offset=state["cell_threshold_offset"],
        cell_min_size=state["cell_min_size"],
        cell_closing_radius=state["cell_closing_radius"],
        threshold_method=state["threshold_method"],
        threshold=state["threshold"],
        threshold_offset=state["threshold_offset"],
        local_block_size=state["local_block_size"],
        min_size=state["mito_min_size"],
        apply_watershed_split=state["apply_watershed_split"],
        watershed_min_distance=state["watershed_min_distance"],
    )

    viewer = napari.Viewer()
    scale = spacing_zyx if timepoint[nuclei_channel].ndim == 3 else spacing_zyx[1:]

    viewer.add_image(timepoint[nuclei_channel], name=f"{channel_names[nuclei_channel]}_3d", scale=scale, blending="additive")
    viewer.add_image(timepoint[cell_channel], name=f"{channel_names[cell_channel]}_3d", scale=scale, blending="additive", visible=False)
    viewer.add_image(timepoint[mito_channel], name=f"{channel_names[mito_channel]}_3d", scale=scale, blending="additive")
    viewer.add_image(result["preview_channels"][nuclei_channel], name="nuclei_preview_2d", blending="additive", visible=False)
    viewer.add_image(result["preview_channels"][cell_channel], name="cell_preview_2d", blending="additive", visible=False)
    viewer.add_image(result["preview_channels"][mito_channel], name="mitochondria_preview_2d", blending="additive")
    viewer.add_image(result["mito_prepared"], name="mitochondria_preprocessed", scale=scale, blending="additive", visible=False)
    viewer.add_labels(result["nuclei_labels"], name="nuclei_labels", scale=scale)
    viewer.add_labels(result["cell_labels"], name="cell_labels", scale=scale)
    viewer.add_labels(result["mito_labels"], name="mitochondria_labels", scale=scale)
    viewer.add_labels(result["mito_binary"], name="mitochondria_binary", scale=scale, visible=False)

    def refresh_layers(updated: dict[str, object]) -> None:
        viewer.layers["nuclei_preview_2d"].data = updated["preview_channels"][nuclei_channel]
        viewer.layers["cell_preview_2d"].data = updated["preview_channels"][cell_channel]
        viewer.layers["mitochondria_preview_2d"].data = updated["preview_channels"][mito_channel]
        viewer.layers["mitochondria_preprocessed"].data = updated["mito_prepared"]
        viewer.layers["nuclei_labels"].data = updated["nuclei_labels"]
        viewer.layers["cell_labels"].data = updated["cell_labels"]
        viewer.layers["mitochondria_labels"].data = updated["mito_labels"]
        viewer.layers["mitochondria_binary"].data = updated["mito_binary"]
        print(updated["summary_df"].to_string(index=False))

    max_z = max(0, trim_leading_z_slices(timepoint, state["skip_leading_z_slices"]).shape[1] - 1)

    @magicgui(
        auto_call=True,
        skip_leading_z_slices={"widget_type": "SpinBox", "min": 0, "max": max(0, timepoint.shape[1] - 1), "step": 1},
        preview_mode={"choices": [("Max projection", "max"), ("Middle slice", "middle"), ("Z slice", "index")]},
        z_index={"widget_type": "Slider", "min": 0, "max": max_z},
        cell_threshold_method={
            "choices": [
                ("Cell triangle", "triangle"),
                ("Cell Otsu", "otsu"),
                ("Cell Li", "li"),
                ("Cell manual", "manual"),
            ]
        },
        cell_manual_threshold={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01},
        cell_threshold_offset={"widget_type": "FloatSlider", "min": -0.25, "max": 0.25, "step": 0.01},
        cell_min_size={"widget_type": "SpinBox", "min": 1, "max": 200000, "step": 1},
        cell_closing_radius={"widget_type": "SpinBox", "min": 1, "max": 50, "step": 1},
        threshold_method={
            "choices": [
                ("Local per slice", "local"),
                ("Per-slice Otsu", "slice_otsu"),
                ("Global Otsu", "otsu"),
                ("Manual absolute", "manual"),
            ]
        },
        manual_threshold={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01},
        threshold_offset={"widget_type": "FloatSlider", "min": -0.25, "max": 0.25, "step": 0.01},
        local_block_size={"widget_type": "SpinBox", "min": 3, "max": 255, "step": 2},
        mito_min_size={"widget_type": "SpinBox", "min": 1, "max": 500000, "step": 1},
        apply_watershed_split={"widget_type": "CheckBox"},
        watershed_min_distance={"widget_type": "SpinBox", "min": 1, "max": 20, "step": 1},
    )
    def analysis_controls(
        skip_leading_z_slices: int = state["skip_leading_z_slices"],
        preview_mode: str = state["z_mode"],
        z_index: int = max(0, min(state["z_index"], max_z)),
        cell_threshold_method: str = state["cell_threshold_method"],
        cell_manual_threshold: float = 0.35 if state["cell_threshold"] is None else float(state["cell_threshold"]),
        cell_threshold_offset: float = state["cell_threshold_offset"],
        cell_min_size: int = state["cell_min_size"],
        cell_closing_radius: int = state["cell_closing_radius"],
        threshold_method: str = state["threshold_method"],
        manual_threshold: float = float(result["resolved_threshold"]),
        threshold_offset: float = state["threshold_offset"],
        local_block_size: int = state["local_block_size"],
        mito_min_size: int = state["mito_min_size"],
        apply_watershed_split: bool = state["apply_watershed_split"],
        watershed_min_distance: int = state["watershed_min_distance"],
    ):
        state["skip_leading_z_slices"] = skip_leading_z_slices
        state["z_mode"] = preview_mode
        state["z_index"] = z_index
        state["cell_threshold_method"] = cell_threshold_method
        state["cell_threshold"] = cell_manual_threshold if cell_threshold_method == "manual" else None
        state["cell_threshold_offset"] = cell_threshold_offset
        state["cell_min_size"] = cell_min_size
        state["cell_closing_radius"] = cell_closing_radius
        state["threshold_method"] = threshold_method
        state["threshold"] = manual_threshold if threshold_method == "manual" else None
        state["threshold_offset"] = threshold_offset
        state["local_block_size"] = local_block_size
        state["mito_min_size"] = mito_min_size
        state["apply_watershed_split"] = apply_watershed_split
        state["watershed_min_distance"] = watershed_min_distance

        updated = run_analysis(
            timepoint,
            spacing_zyx=spacing_zyx,
            nuclei_channel=nuclei_channel,
            cell_channel=cell_channel,
            mito_channel=mito_channel,
            skip_leading_z_slices=skip_leading_z_slices,
            z_mode=preview_mode,
            z_index=z_index,
            cell_threshold_method=cell_threshold_method,
            cell_threshold=state["cell_threshold"],
            cell_threshold_offset=cell_threshold_offset,
            cell_min_size=cell_min_size,
            cell_closing_radius=cell_closing_radius,
            threshold_method=threshold_method,
            threshold=state["threshold"],
            threshold_offset=threshold_offset,
            local_block_size=local_block_size,
            min_size=mito_min_size,
            apply_watershed_split=apply_watershed_split,
            watershed_min_distance=watershed_min_distance,
        )
        refresh_layers(updated)
        return updated["mito_labels"]

    @magicgui(call_button="Save current analysis")
    def save_current_analysis() -> None:
        updated = run_analysis(
            timepoint,
            spacing_zyx=spacing_zyx,
            nuclei_channel=nuclei_channel,
            cell_channel=cell_channel,
            mito_channel=mito_channel,
            skip_leading_z_slices=state["skip_leading_z_slices"],
            z_mode=state["z_mode"],
            z_index=state["z_index"],
            cell_threshold_method=state["cell_threshold_method"],
            cell_threshold=state["cell_threshold"],
            cell_threshold_offset=state["cell_threshold_offset"],
            cell_min_size=state["cell_min_size"],
            cell_closing_radius=state["cell_closing_radius"],
            threshold_method=state["threshold_method"],
            threshold=state["threshold"],
            threshold_offset=state["threshold_offset"],
            local_block_size=state["local_block_size"],
            min_size=state["mito_min_size"],
            apply_watershed_split=state["apply_watershed_split"],
            watershed_min_distance=state["watershed_min_distance"],
        )
        save_outputs(
            output_dir,
            updated["preview_channels"],
            timepoint,
            updated["nuclei_labels"],
            updated["cell_labels"],
            updated["mito_labels"],
            updated["mito_binary"],
            updated["mito_df"],
            updated["skeleton_df"],
            updated["per_cell_df"],
            updated["summary_df"],
        )
        print(f"Saved analysis outputs to {output_dir}")

    viewer.window.add_dock_widget(analysis_controls, area="right")
    viewer.window.add_dock_widget(save_current_analysis, area="right")
    napari.run()


def main() -> int:
    args = parse_args()
    input_path = args.input_path or select_input_file()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    normalized_conditions = [condition.upper() for condition in args.conditions]
    if input_path.is_dir():
        output_dir = args.output_dir or input_path / "mito_analysis_results"
        batch_file_df, batch_per_cell_df, condition_summary_df, condition_comparison_df = run_batch_analysis(
            input_path,
            output_dir,
            normalized_conditions,
            args,
        )
        print(f"Saved batch outputs to {output_dir}")
        print(batch_file_df.to_string(index=False))
        if not condition_summary_df.empty:
            print(condition_summary_df.to_string(index=False))
        if not condition_comparison_df.empty:
            print(condition_comparison_df.to_string(index=False))
        return 0

    dataset = load_dataset(input_path, fallback_xy_um=args.pixel_size_um, fallback_z_um=args.z_step_um)
    if args.time_index < 0 or args.time_index >= dataset.data.shape[0]:
        raise SystemExit(f"time-index {args.time_index} is outside dataset range 0..{dataset.data.shape[0] - 1}")

    timepoint = dataset.data[args.time_index]
    channel_count = timepoint.shape[0]
    required_index = max(args.nuclei_channel, args.cell_channel, args.mito_channel)
    if required_index >= channel_count:
        raise SystemExit(
            f"Dataset has {channel_count} channels, but channel index {required_index} was requested."
        )

    output_dir = args.output_dir or (input_path.parent / f"{input_path.stem}_mito_analysis")

    if args.run_sweep:
        sweep_df, top_df = run_parameter_sweep(
            timepoint,
            spacing_zyx=dataset.spacing_zyx,
            nuclei_channel=args.nuclei_channel,
            cell_channel=args.cell_channel,
            mito_channel=args.mito_channel,
            skip_leading_z_slices=args.skip_leading_z_slices,
            z_mode=args.z_mode,
            z_index=args.z_index,
            cell_threshold_method=args.cell_threshold_method,
            cell_threshold=args.cell_threshold,
            cell_threshold_offset=args.cell_threshold_offset,
            cell_min_size=args.cell_min_size,
            cell_closing_radius=args.cell_closing_radius,
            top_k=args.sweep_top_k,
            apply_watershed_split=not args.disable_mito_watershed,
            watershed_min_distance=args.mito_watershed_min_distance,
        )
        save_sweep_outputs(output_dir, sweep_df, top_df)
        print(f"Channel names: {dataset.channel_names}")
        print(f"Voxel spacing z/y/x (um): {dataset.spacing_zyx}")
        print(f"Saved sweep outputs to {output_dir}")
        if not top_df.empty:
            print(top_df.to_string(index=False))
        return 0

    result = run_analysis(
        timepoint,
        spacing_zyx=dataset.spacing_zyx,
        nuclei_channel=args.nuclei_channel,
        cell_channel=args.cell_channel,
        mito_channel=args.mito_channel,
        skip_leading_z_slices=args.skip_leading_z_slices,
        z_mode=args.z_mode,
        z_index=args.z_index,
        cell_threshold_method=args.cell_threshold_method,
        cell_threshold=args.cell_threshold,
        cell_threshold_offset=args.cell_threshold_offset,
        cell_min_size=args.cell_min_size,
        cell_closing_radius=args.cell_closing_radius,
        threshold_method=args.mito_threshold_method,
        threshold=args.mito_threshold,
        threshold_offset=args.mito_threshold_offset,
        local_block_size=args.mito_local_block_size,
        min_size=args.mito_min_size,
        apply_watershed_split=not args.disable_mito_watershed,
        watershed_min_distance=args.mito_watershed_min_distance,
    )

    save_outputs(
        output_dir,
        result["preview_channels"],
        timepoint,
        result["nuclei_labels"],
        result["cell_labels"],
        result["mito_labels"],
        result["mito_binary"],
        result["mito_df"],
        result["skeleton_df"],
        result["per_cell_df"],
        result["summary_df"],
    )

    print(f"Channel names: {dataset.channel_names}")
    print(f"Voxel spacing z/y/x (um): {dataset.spacing_zyx}")
    print(f"Saved analysis outputs to {output_dir}")
    print(result["summary_df"].to_string(index=False))

    if args.show_viewer:
        show_in_napari(
            timepoint,
            channel_names=dataset.channel_names,
            spacing_zyx=dataset.spacing_zyx,
            output_dir=output_dir,
            nuclei_channel=args.nuclei_channel,
            cell_channel=args.cell_channel,
            mito_channel=args.mito_channel,
            initial_skip_leading_z_slices=args.skip_leading_z_slices,
            initial_z_mode=args.z_mode,
            initial_z_index=args.z_index,
            initial_cell_threshold_method=args.cell_threshold_method,
            initial_cell_threshold=args.cell_threshold,
            initial_cell_threshold_offset=args.cell_threshold_offset,
            initial_cell_min_size=args.cell_min_size,
            initial_cell_closing_radius=args.cell_closing_radius,
            initial_threshold_method=args.mito_threshold_method,
            initial_threshold=args.mito_threshold,
            initial_threshold_offset=args.mito_threshold_offset,
            initial_local_block_size=args.mito_local_block_size,
            initial_mito_min_size=args.mito_min_size,
            initial_apply_watershed_split=not args.disable_mito_watershed,
            initial_watershed_min_distance=args.mito_watershed_min_distance,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())