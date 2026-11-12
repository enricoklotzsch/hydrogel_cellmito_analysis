import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology

# Parameters for dataset structure
N_CHANNELS = 3
N_Z = 100
N_TILES = 4
N_CONDITIONS = 2
FRAMES_PER_STACK = 2400
FRAME_SIZE_MB = 4

# Helper to memory-map large TIFFs
def open_virtual_stack(path):
    return tifffile.memmap(path)

# Example main script for image analysis


def split_stack(stack):
    # stack shape: (frames, height, width) or (frames, ...)
    # Reshape to (conditions, tiles, z, channels, height, width)
    frames, *im_shape = stack.shape
    assert frames == N_CHANNELS * N_Z * N_TILES * N_CONDITIONS, (
        f"Unexpected number of frames: {frames}")
    arr = stack.reshape((N_CONDITIONS, N_TILES, N_Z, N_CHANNELS, *im_shape))
    return arr


def analyze_channel(channel_stack):
    # channel_stack: (z, height, width)
    # Example: threshold and label max projection
    max_proj = np.max(channel_stack, axis=0)
    threshold = filters.threshold_otsu(max_proj)
    binary = max_proj > threshold
    labeled = measure.label(binary)
    return labeled


def main(data_dir='data'):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    for fname in os.listdir(data_dir):
        if fname.endswith('.tif') or fname.endswith('.tiff'):
            path = os.path.join(data_dir, fname)
            print(f"Processing {fname} as virtual stack...")
            stack = open_virtual_stack(path)
            arr = split_stack(stack)
            for cond in range(N_CONDITIONS):
                for tile in range(N_TILES):
                    for ch in range(N_CHANNELS):
                        # arr shape: (conditions, tiles, z, channels, height, width)
                        channel_stack = arr[cond, tile, :, ch]
                        labeled = analyze_channel(channel_stack)
                        out_name = f"{fname}_cond{cond}_tile{tile}_ch{ch}_analysis.png"
                        plt.imshow(labeled)
                        plt.title(f"{fname} C{cond} T{tile} Ch{ch}")
                        plt.savefig(os.path.join(results_dir, out_name))
                        plt.close()

if __name__ == '__main__':
    import sys
    # Allow user to specify a data directory as a command-line argument
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
