"""
Full pipeline for 3D cell and mitochondria analysis:
- Loads 3D TIFF stack (multi-channel)
- Segments nuclei (DAPI), actin, and mitochondria
- Computes morphometric features (volume, shape)
- Performs mitochondria network analysis
"""
import os
import numpy as np
import tifffile
from skimage import filters, measure, morphology, segmentation, exposure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# --- CONFIG ---
DAPI_CHANNEL = 0  # index for DAPI
ACTIN_CHANNEL = 1 # index for actin
MITO_CHANNEL = 2  # index for mitochondria

# --- LOAD DATA ---
def load_stack(tiff_path):
    """Load a 3-channel 3D TIFF stack as (Z, Y, X, C)."""
    arr = tifffile.imread(tiff_path)
    if arr.ndim == 4:
        # Already (Z, Y, X, C)
        return arr
    elif arr.ndim == 3:
        # Assume (C, Z, Y, X) or (Z, Y, X)
        if arr.shape[0] == 3:
            return np.moveaxis(arr, 0, -1)
        else:
            raise ValueError("TIFF stack shape not recognized.")
    else:
        raise ValueError("TIFF stack shape not recognized.")

# --- SEGMENTATION ---
def segment_nuclei(dapi_stack):
    """Segment nuclei in 3D using DAPI channel."""
    # Enhance contrast
    dapi_eq = exposure.equalize_adapthist(dapi_stack)
    # Threshold
    thresh = filters.threshold_otsu(dapi_eq)
    binary = dapi_eq > thresh
    # Remove small objects
    binary = morphology.remove_small_objects(binary, min_size=500)
    # Fill holes
    binary = ndi.binary_fill_holes(binary)
    # Label nuclei
    labels = measure.label(binary)
    return labels

def segment_actin(actin_stack):
    """Segment actin cytoskeleton (optional, for morphometry)."""
    actin_eq = exposure.equalize_adapthist(actin_stack)
    thresh = filters.threshold_otsu(actin_eq)
    binary = actin_eq > thresh
    binary = morphology.remove_small_objects(binary, min_size=1000)
    binary = ndi.binary_fill_holes(binary)
    labels = measure.label(binary)
    return labels

def segment_mito(mito_stack):
    """Segment mitochondria network."""
    mito_eq = exposure.equalize_adapthist(mito_stack)
    thresh = filters.threshold_otsu(mito_eq)
    binary = mito_eq > thresh
    binary = morphology.remove_small_objects(binary, min_size=50)
    return binary

# --- MORPHOMETRIC ANALYSIS ---
def analyze_nuclei(labels):
    """Compute volume and shape for each nucleus."""
    props = measure.regionprops_table(labels, properties=['label', 'area', 'bbox', 'eccentricity', 'solidity'])
    return props

def analyze_mito_network(mito_binary, nuclei_labels):
    """Analyze mitochondria network per cell (simple: count, size, connectivity)."""
    mito_labeled = measure.label(mito_binary)
    mito_props = measure.regionprops_table(mito_labeled, properties=['label', 'area', 'eccentricity', 'solidity'])
    # Optionally: assign mito objects to nearest nucleus
    return mito_props

# --- PIPELINE ---
def run_pipeline(tiff_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stack = load_stack(tiff_path)
    dapi = stack[..., DAPI_CHANNEL]
    actin = stack[..., ACTIN_CHANNEL]
    mito = stack[..., MITO_CHANNEL]
    # Segment
    nuclei_labels = segment_nuclei(dapi)
    actin_labels = segment_actin(actin)
    mito_binary = segment_mito(mito)
    # Save masks
    tifffile.imwrite(os.path.join(out_dir, 'nuclei_labels.tif'), nuclei_labels.astype(np.uint16))
    tifffile.imwrite(os.path.join(out_dir, 'actin_labels.tif'), actin_labels.astype(np.uint16))
    tifffile.imwrite(os.path.join(out_dir, 'mito_binary.tif'), mito_binary.astype(np.uint8))
    # Analyze
    nuclei_props = analyze_nuclei(nuclei_labels)
    mito_props = analyze_mito_network(mito_binary, nuclei_labels)
    # Save results
    import pandas as pd
    pd.DataFrame(nuclei_props).to_csv(os.path.join(out_dir, 'nuclei_morphometry.csv'), index=False)
    pd.DataFrame(mito_props).to_csv(os.path.join(out_dir, 'mito_network.csv'), index=False)
    print('Analysis complete. Results saved to', out_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python segment_and_analyze.py <tiff_path> <out_dir>')
        sys.exit(1)
    run_pipeline(sys.argv[1], sys.argv[2])
