
# --- All imports at top ---
import os
import glob
import tifffile
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from xml_mapping import parse_xlsx_mapping
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Slider as SliderWidget, Button
from skimage import filters, measure, morphology, segmentation, exposure
from scipy import ndimage as ndi
import pandas as pd

# --- SEGMENTATION AND ANALYSIS SUBROUTINE (from segment_and_analyze.py) ---
def segment_and_analyze_stack(stack, out_dir):
    """Run segmentation and morphometric analysis on a 3-channel stack (Z, Y, X, C)."""
    os.makedirs(out_dir, exist_ok=True)
    dapi = stack[..., 0]
    actin = stack[..., 1]
    mito = stack[..., 2]
    # Normalize to [0, 1]
    def normalize01(img):
        img = img.astype(np.float32)
        vmin = np.min(img)
        vmax = np.max(img)
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        else:
            return np.zeros_like(img)
    dapi_n = normalize01(dapi)
    actin_n = normalize01(actin)
    mito_n = normalize01(mito)
    # Segment
    dapi_eq = exposure.equalize_adapthist(dapi_n)
    thresh = filters.threshold_otsu(dapi_eq)
    binary = dapi_eq > thresh
    binary = morphology.remove_small_objects(binary, min_size=500)
    binary = ndi.binary_fill_holes(binary)
    nuclei_labels = measure.label(binary)
    actin_eq = exposure.equalize_adapthist(actin_n)
    thresh_a = filters.threshold_otsu(actin_eq)
    binary_a = actin_eq > thresh_a
    binary_a = morphology.remove_small_objects(binary_a, min_size=1000)
    binary_a = ndi.binary_fill_holes(binary_a)
    actin_labels = measure.label(binary_a)
    mito_eq = exposure.equalize_adapthist(mito_n)
    thresh_m = filters.threshold_otsu(mito_eq)
    binary_m = mito_eq > thresh_m
    binary_m = morphology.remove_small_objects(binary_m, min_size=50)
    mito_binary = binary_m
    # Save masks
    tifffile.imwrite(os.path.join(out_dir, 'nuclei_labels.tif'), nuclei_labels.astype(np.uint16))
    tifffile.imwrite(os.path.join(out_dir, 'actin_labels.tif'), actin_labels.astype(np.uint16))
    tifffile.imwrite(os.path.join(out_dir, 'mito_binary.tif'), mito_binary.astype(np.uint8))
    # Analyze
    # 'eccentricity' is not supported for 3D images
    nuclei_props = measure.regionprops_table(nuclei_labels, properties=['label', 'area', 'bbox', 'solidity'])
    mito_labeled = measure.label(mito_binary)
    mito_props = measure.regionprops_table(mito_labeled, properties=['label', 'area', 'solidity'])
    pd.DataFrame(nuclei_props).to_csv(os.path.join(out_dir, 'nuclei_morphometry.csv'), index=False)
    pd.DataFrame(mito_props).to_csv(os.path.join(out_dir, 'mito_network.csv'), index=False)
    print('Analysis complete. Results saved to', out_dir)


import os
import glob
import tifffile
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from xml_mapping import parse_xlsx_mapping


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Slider as SliderWidget, Button

# --- Analysis pipeline dependencies ---
from skimage import filters, measure, morphology, exposure
from scipy import ndimage as ndi
import pandas as pd


# File name format: 1008-136-510-0-211-15-413-610.tiff
# 1008, 136: tile position; 0: z slice; 413, 414, 415: channels

def parse_filename(fname):
    # Returns dict with tile_x, tile_y, z, channel, etc.
    base = os.path.basename(fname)
    parts = base.split('-')
    if len(parts) < 7:
        return None
    tile_x = int(parts[0])
    tile_y = int(parts[1])
    z = int(parts[3])
    channel = int(parts[6])
    return {'tile_x': tile_x, 'tile_y': tile_y, 'z': z, 'channel': channel, 'fname': fname}

def collect_files(folder):
    files = glob.glob(os.path.join(folder, '*.tif*'))
    if not files:
        print(f'No TIFF files found in {folder}')
    parsed = [parse_filename(f) for f in files]
    parsed = [p for p in parsed if p]
    if not parsed:
        print('No valid TIFF files parsed. Check file naming convention.')
    return parsed

def get_unique(parsed, key):
    return sorted(set(p[key] for p in parsed))

def get_overlay(parsed, tile_x, tile_y, z, channel_ids):
    # channel_ids: list of 3 channel numbers (e.g. [413,414,415])
    imgs = []
    for ch in channel_ids:
        match = [p for p in parsed if p['tile_x']==tile_x and p['tile_y']==tile_y and p['z']==z and p['channel']==ch]
        if match:
            try:
                print(f"Loading TIFF: {match[0]['fname']}")
                img = tifffile.imread(match[0]['fname'])
                imgs.append(img.astype(np.float32))
            except Exception as e:
                print(f"Error loading {match[0]['fname']}: {e}")
                imgs.append(np.zeros_like(tifffile.imread(parsed[0]['fname']), dtype=np.float32))
        else:
            print(f"No match for channel {ch}, using zeros.")
            imgs.append(np.zeros_like(tifffile.imread(parsed[0]['fname']), dtype=np.float32))
    try:
        stacked = np.stack(imgs, axis=-1)
    except Exception as e:
        print(f"Error stacking images: {e}")
        return None
    return stacked

# --- Analysis pipeline ---
def run_analysis_pipeline(folder, channel_ids, out_dir):
    print(f"[ANALYSIS] Running segmentation and analysis on folder: {folder}")
    # Collect all z for the first tile_x, tile_y
    parsed = collect_files(folder)
    tile_xs = get_unique(parsed, 'tile_x')
    tile_ys = get_unique(parsed, 'tile_y')
    zs = get_unique(parsed, 'z')
    if not tile_xs or not tile_ys or not zs:
        print('[ANALYSIS] No valid tiles/z found.')
        return
    tile_x = tile_xs[0]
    tile_y = tile_ys[0]
    # Load full 3D stack for the selected tile
    stack = []
    for z in zs:
        overlay = get_overlay(parsed, tile_x, tile_y, z, channel_ids)
        stack.append(overlay)
    stack = np.stack(stack, axis=0)  # (Z, Y, X, C)
    dapi = stack[..., 0]
    actin = stack[..., 1]
    mito = stack[..., 2]
    # --- SEGMENTATION ---
    def normalize01(img):
        img = img.astype(np.float32)
        vmin = np.min(img)
        vmax = np.max(img)
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        else:
            return np.zeros_like(img)
    def segment_nuclei(dapi_stack):
        dapi_norm = normalize01(dapi_stack)
        dapi_eq = exposure.equalize_adapthist(dapi_norm)
        thresh = filters.threshold_otsu(dapi_eq)
        binary = dapi_eq > thresh
        binary = morphology.remove_small_objects(binary, min_size=500)
        binary = ndi.binary_fill_holes(binary)
        labels = measure.label(binary)
        return labels
    def segment_actin(actin_stack, thresh=None, min_size=1000):
        actin_norm = normalize01(actin_stack)
        actin_eq = exposure.equalize_adapthist(actin_norm)
        if thresh is None:
            thresh = filters.threshold_otsu(actin_eq)
        binary = actin_eq > thresh
        binary = morphology.remove_small_objects(binary, min_size=int(min_size))
        binary = ndi.binary_fill_holes(binary)
        labels = measure.label(binary)
        return labels
    def segment_mito(mito_stack, thresh=None, min_size=50):
        mito_norm = normalize01(mito_stack)
        mito_eq = exposure.equalize_adapthist(mito_norm)
        if thresh is None:
            thresh = filters.threshold_otsu(mito_eq)
        binary = mito_eq > thresh
        binary = morphology.remove_small_objects(binary, min_size=int(min_size))
        return binary
    # --- ANALYSIS ---
    def analyze_nuclei(labels):
        props = measure.regionprops_table(labels, properties=['label', 'area', 'bbox', 'solidity'])
        return props
    def analyze_mito_network(mito_binary, nuclei_labels):
        mito_labeled = measure.label(mito_binary)
        mito_props = measure.regionprops_table(mito_labeled, properties=['label', 'area', 'solidity'])
        return mito_props
    # --- RUN ---
    os.makedirs(out_dir, exist_ok=True)
    # Use provided parameters or defaults
    import inspect
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    actin_thresh = values.get('actin_thresh', None)
    actin_min_size = values.get('actin_min_size', 1000)
    mito_thresh = values.get('mito_thresh', None)
    mito_min_size = values.get('mito_min_size', 50)
    nuclei_labels = segment_nuclei(dapi)
    actin_labels = segment_actin(actin, thresh=actin_thresh, min_size=actin_min_size)
    mito_binary = segment_mito(mito, thresh=mito_thresh, min_size=mito_min_size)
    # Ensure correct types and value ranges
    nuclei_labels = np.nan_to_num(nuclei_labels).astype(np.uint16)
    actin_labels = np.nan_to_num(actin_labels).astype(np.uint16)
    mito_binary = (mito_binary > 0).astype(np.uint8)
    # Save masks
    tifffile.imwrite(os.path.join(out_dir, 'nuclei_labels.tif'), nuclei_labels)
    tifffile.imwrite(os.path.join(out_dir, 'actin_labels.tif'), actin_labels)
    tifffile.imwrite(os.path.join(out_dir, 'mito_binary.tif'), mito_binary)
    # Analyze
    nuclei_props = analyze_nuclei(nuclei_labels)
    mito_props = analyze_mito_network(mito_binary, nuclei_labels)
    pd.DataFrame(nuclei_props).to_csv(os.path.join(out_dir, 'nuclei_morphometry.csv'), index=False)
    pd.DataFrame(mito_props).to_csv(os.path.join(out_dir, 'mito_network.csv'), index=False)
    print(f'[ANALYSIS] Complete. Results in {out_dir}')

def show_overlay_viewer(folder, channel_ids=[413,414,415], xlsx_path=None):
    # --- Place Mito Threshold Adjust button after overlays and plt are defined ---
    # This must be after overlays is assigned
    # --- Add Mito-Hacker 3D analysis button ---
    ax_button_mitohacker = plt.axes([0.05, 0.45, 0.15, 0.05])
    btn_mitohacker = Button(ax_button_mitohacker, 'Mito Analysis (Mito-Hacker)')
    def mito_hacker_analysis(event):
        # 3D stack: overlays shape (Z, Y, X, C), mitochondria in overlays[..., 2]
        mito_stack = overlays[..., 2]
        # Normalize stack
        mito_norm = (mito_stack - np.min(mito_stack)) / (np.max(mito_stack) - np.min(mito_stack)) if np.max(mito_stack) > np.min(mito_stack) else np.zeros_like(mito_stack)
        # Adaptive thresholding (Mito-Hacker inspired, per-slice Otsu + optional 3D median)
        from skimage import filters, morphology, measure
        from scipy import ndimage as ndi
        binary_stack = np.zeros_like(mito_norm, dtype=bool)
        for z in range(mito_norm.shape[0]):
            slice_img = mito_norm[z]
            thresh = filters.threshold_otsu(slice_img)
            binary = slice_img > thresh
            binary = morphology.remove_small_objects(binary, min_size=30)
            binary = ndi.binary_fill_holes(binary)
            binary_stack[z] = binary
        # Optionally smooth in 3D
        binary_stack = morphology.remove_small_objects(binary_stack, min_size=100)
        # Label 3D mitochondria
        mito_labels = measure.label(binary_stack)
        mito_props = measure.regionprops(mito_labels)
        # Summary: count, volume per object
        voxel_volume = 0.095 * 0.095 * 0.135  # μm³
        results = []
        for prop in mito_props:
            label = prop.label
            voxels = prop.area
            volume = voxels * voxel_volume
            results.append((label, volume, voxels))
        msg = f"Mitochondria (3D) Analysis (Mito-Hacker style):\nTotal objects: {len(results)}\nLabel\tVolume (μm³)\tVoxels\n"
        for label, volume, voxels in results:
            msg += f"{label}\t{volume:.2f}\t{voxels}\n"
        print(msg)
        import tkinter.messagebox as mbox
        mbox.showinfo("Mitochondria 3D Analysis", msg)
        # Optionally save labeled stack
        from pathlib import Path
        import tifffile
        import datetime
        out_dir = Path('results') / f"{selected_well}_{tile}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'mito_labels_3d.tif'
        tifffile.imwrite(str(out_path), mito_labels.astype(np.uint16))
        print(f"Saved 3D mitochondria labels to {out_path}")
    btn_mitohacker.on_clicked(mito_hacker_analysis)
    # ...existing code...
    from functools import partial
    import datetime
    print(f"Matplotlib backend in use: {matplotlib.get_backend()}")
    print("[DEBUG] Collecting files...")

    # --- XLSX mapping integration ---
    if not xlsx_path or not os.path.exists(xlsx_path):
        print("[ERROR] You must provide a valid XLSX file for mapping.")
        return
    mapping, df = parse_xlsx_mapping(xlsx_path)
    wells = sorted(mapping.keys())
    print(f"[DEBUG] Mapping structure:")
    for well in wells:
        print(f"  Well {well}: tiles {list(mapping[well].keys())}")

    # Helper: get all unique (tile_x, tile_y) pairs for a given well
    def get_tiles_for_well(well=None, mapping=None):
        if mapping and well and well in mapping:
            tile_keys = sorted(mapping[well].keys())
            print(f"[DEBUG] Extracted tile_keys for well {well}: {tile_keys}")
            return tile_keys
        return []


    # Tkinter selectors for well and tile
    root = tk.Tk()
    root.title('TIFF Viewer - Select Well and Tile')
    well_var = tk.StringVar(value=wells[0] if wells else '')
    tile_var = tk.StringVar()

    def update_tiles(*args):
        selected_well = well_var.get()
        tile_keys = get_tiles_for_well(selected_well, mapping)
        tile_strs = tile_keys
        print(f"[DEBUG] Available wells: {wells}")
        print(f"[DEBUG] Tiles for well {selected_well}: {tile_strs}")
        tile_menu['values'] = tile_strs
        if tile_strs:
            tile_var.set(tile_strs[0])
        else:
            tile_var.set('')
            messagebox.showerror('No Tiles', f'No tiles found for well {selected_well}. Exiting.')
            root.after(100, root.destroy)
            return

    tk.Label(root, text='Well:').grid(row=0, column=0)
    well_menu = ttk.Combobox(root, textvariable=well_var, values=wells, state='readonly')
    well_menu.grid(row=0, column=1)
    tk.Label(root, text='Tile:').grid(row=1, column=0)
    tile_menu = ttk.Combobox(root, textvariable=tile_var, values=[], state='readonly')
    tile_menu.grid(row=1, column=1)
    well_var.trace('w', update_tiles)
    update_tiles()
    def on_select():
        root.destroy()
    select_btn = tk.Button(root, text='Load', command=on_select)
    select_btn.grid(row=2, column=0, columnspan=2, pady=10)
    root.mainloop()

    selected_well = well_var.get()
    tile_str = tile_var.get()
    if tile_str:
        tile = tile_str
    else:
        tile = None

    from pathlib import Path
    overlays = []
    zs = []
    tiff_order = []
    # Use new mapping for TIFF loading
    if mapping and selected_well in mapping and tile in mapping[selected_well]:
        z_dict = mapping[selected_well][tile]
        first_value = next(iter(z_dict.values()), None)
        if isinstance(first_value, dict):
            nested_value = next(iter(first_value.values()), None)
            if isinstance(nested_value, dict):
                timepoints = sorted(z_dict.keys(), key=lambda x: int(x[1:]) if isinstance(x, str) and len(x) > 1 and x[1:].isdigit() else x)
                selected_timepoint = timepoints[0] if timepoints else ''
                print(f"[DEBUG] Using timepoint {selected_timepoint or '(no timepoint)'} for well {selected_well}, tile {tile}")
                z_dict = z_dict[selected_timepoint]
        zs = sorted(z_dict.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else int(x))
        for zslice in zs:
            print(f"[DEBUG] z-slice: {zslice}, available keys: {list(z_dict[zslice].keys())}")
            ch_files = []
            for ch in channel_ids:
                ch_key = f"R{ch}"
                orig_ch_key = ch_key
                if ch_key not in z_dict[zslice]:
                    if ch == 413:
                        ch_key = "R1"
                    elif ch == 414:
                        ch_key = "R2"
                    elif ch == 415:
                        ch_key = "R3"
                print(f"[DEBUG] Looking for channel {ch} (key tried: {orig_ch_key} -> {ch_key}) in z-slice {zslice}")
                tiff_file = z_dict[zslice].get(ch_key)
                print(f"[DEBUG] TIFF file for channel {ch_key}: {tiff_file}")
                if tiff_file:
                    tiff_path = os.path.join(folder, tiff_file)
                    print(f"[DEBUG] TIFF path: {tiff_path}, exists: {os.path.exists(tiff_path)}")
                    if os.path.exists(tiff_path):
                        img = tifffile.imread(tiff_path).astype(np.float32)
                        ch_files.append(img)
                    else:
                        print(f"[WARN] TIFF not found: {tiff_path}")
                        ch_files.append(np.zeros_like(tifffile.imread(list(Path(folder).glob('*.tif*'))[0]), dtype=np.float32))
                else:
                    print(f"[WARN] No TIFF file found for channel {ch_key} in z-slice {zslice}")
                    ch_files.append(np.zeros_like(tifffile.imread(list(Path(folder).glob('*.tif*'))[0]), dtype=np.float32))
            try:
                overlay = np.stack(ch_files, axis=-1)
            except Exception as e:
                print(f"Error stacking images: {e}")
                overlay = None
            overlays.append(overlay)
            tiff_order.append([z_dict[zslice].get(f"R{ch}") if f"R{ch}" in z_dict[zslice] else z_dict[zslice].get(f"R{1 if ch==413 else 2 if ch==414 else 3 if ch==415 else ch}") for ch in channel_ids])
        print("[DEBUG] TIFF order for selected well/tile:")
        for i, files in enumerate(tiff_order):
            print(f"Z {zs[i]}: {files}")
        overlays = np.array(overlays)
    else:
        print('No z-slices found for selected tile.')
        return

    # --- Matplotlib viewer with z-slice and max projection ---

    # --- Matplotlib viewer with z-slice and max projection ---
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    # Load all overlays for the selected tile_x, tile_y
    overlays = np.array(overlays)  # overlays already created above if XML mapping is used
    # --- Mito Threshold Adjust button (after overlays is defined) ---
    mito_stack = overlays[..., 2] if overlays is not None and overlays.shape[-1] > 2 else None
    ax_button_mitothresh = plt.axes([0.05, 0.52, 0.15, 0.05])
    btn_mitothresh = Button(ax_button_mitothresh, 'Mito Threshold Adjust')
    def mito_threshold_adjust(event):
        import matplotlib.pyplot as plt2
        from matplotlib.widgets import Slider as Slider2
        from skimage import filters, morphology
        if mito_stack is None:
            print("[ERROR] mito_stack is not available.")
            return
        z = mito_stack.shape[0] // 2  # middle slice
        img = mito_stack[z]
        fig2, ax2 = plt2.subplots()
        plt2.subplots_adjust(left=0.25, bottom=0.25)
        vmin, vmax = float(np.min(img)), float(np.max(img))
        thresh_init = filters.threshold_otsu(img)
        binary = img > thresh_init
        binary = morphology.remove_small_objects(binary, min_size=30)
        im2 = ax2.imshow(img, cmap='gray')
        overlay2 = ax2.imshow(np.ma.masked_where(~binary, binary), cmap='spring', alpha=0.5)
        ax2.set_title(f'Mitochondria Threshold Adjust (Z={z})')
        axcolor = 'lightgoldenrodyellow'
        axthresh = plt2.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        sthresh = Slider2(axthresh, 'Threshold', vmin, vmax, valinit=thresh_init)
        def update(val):
            t = sthresh.val
            binary = img > t
            binary = morphology.remove_small_objects(binary, min_size=30)
            overlay2.set_data(np.ma.masked_where(~binary, binary))
            fig2.canvas.draw_idle()
        sthresh.on_changed(update)
        plt2.show()
    btn_mitothresh.on_clicked(mito_threshold_adjust)
    # Initial display: first z-slice
    current_z = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Per-channel min/max sliders
    min_vals = [float(np.nanmin(overlays[..., c])) for c in range(3)]
    max_vals = [float(np.nanmax(overlays[..., c])) for c in range(3)]
    min_init = min_vals.copy()
    max_init = max_vals.copy()

    # Sliders for z
    ax_z = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    s_z = Slider(ax_z, 'Z', 0, overlays.shape[0]-1, valinit=current_z, valstep=1)

    # Per-channel min/max sliders
    ax_min_r = plt.axes([0.25, 0.13, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    ax_max_r = plt.axes([0.25, 0.08, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    ax_min_g = plt.axes([0.48, 0.13, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    ax_max_g = plt.axes([0.48, 0.08, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    ax_min_b = plt.axes([0.71, 0.13, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    ax_max_b = plt.axes([0.71, 0.08, 0.18, 0.03], facecolor='lightgoldenrodyellow')
    s_min_r = Slider(ax_min_r, 'Min R', min_vals[0], max_vals[0], valinit=min_init[0])
    s_max_r = Slider(ax_max_r, 'Max R', min_vals[0], max_vals[0], valinit=max_init[0])
    s_min_g = Slider(ax_min_g, 'Min G', min_vals[1], max_vals[1], valinit=min_init[1])
    s_max_g = Slider(ax_max_g, 'Max G', min_vals[1], max_vals[1], valinit=max_init[1])
    s_min_b = Slider(ax_min_b, 'Min B', min_vals[2], max_vals[2], valinit=min_init[2])
    s_max_b = Slider(ax_max_b, 'Max B', min_vals[2], max_vals[2], valinit=max_init[2])

    # Max projection button
    ax_button = plt.axes([0.05, 0.03, 0.15, 0.05])
    btn_max = Button(ax_button, 'Max Projection')
    is_max_proj = [False]

    def normalize_channel(img, vmin, vmax):
        img = np.clip(img, vmin, vmax)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)
        return img

    def get_rgb_overlay(zidx):
        rgb = np.zeros_like(overlays[zidx])
        rgb[..., 0] = normalize_channel(overlays[zidx, ..., 0], s_min_r.val, s_max_r.val)
        rgb[..., 1] = normalize_channel(overlays[zidx, ..., 1], s_min_g.val, s_max_g.val)
        rgb[..., 2] = normalize_channel(overlays[zidx, ..., 2], s_min_b.val, s_max_b.val)
        return rgb

    def get_rgb_maxproj():
        rgb = np.zeros_like(overlays[0])
        rgb[..., 0] = normalize_channel(np.nanmax(overlays[..., 0], axis=0), s_min_r.val, s_max_r.val)
        rgb[..., 1] = normalize_channel(np.nanmax(overlays[..., 1], axis=0), s_min_g.val, s_max_g.val)
        rgb[..., 2] = normalize_channel(np.nanmax(overlays[..., 2], axis=0), s_min_b.val, s_max_b.val)
        return rgb

    im = ax.imshow(get_rgb_overlay(current_z))
    ax.set_title(f'Tile {tile} Z {zs[current_z]}')

    def update(val=None):
        if is_max_proj[0]:
            im.set_data(get_rgb_maxproj())
            ax.set_title(f'Tile {tile} [MAX PROJ]')
        else:
            zidx = int(s_z.val)
            im.set_data(get_rgb_overlay(zidx))
            ax.set_title(f'Tile {tile} Z {zs[zidx]}')
        fig.canvas.draw_idle()

    s_z.on_changed(update)
    s_min_r.on_changed(update)
    s_max_r.on_changed(update)
    s_min_g.on_changed(update)
    s_max_g.on_changed(update)
    s_min_b.on_changed(update)
    s_max_b.on_changed(update)

    def toggle_max(event):
        is_max_proj[0] = not is_max_proj[0]
        update()
    btn_max.on_clicked(toggle_max)
    # --- Add analysis button (after overlays and plt are defined) ---
    ax_button_analysis = plt.axes([0.05, 0.10, 0.15, 0.05])
    btn_analysis = Button(ax_button_analysis, 'Run Analysis')
    def run_analysis(event):
        out_dir = os.path.join('results', f"{selected_well}_{tile}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"[ANALYSIS] Running segmentation and morphometry for {selected_well} {tile}, output: {out_dir}")
        segment_and_analyze_stack(overlays, out_dir)
    btn_analysis.on_clicked(run_analysis)

    # --- Add nucleus volume analysis button ---
    ax_button_nucvol = plt.axes([0.05, 0.17, 0.15, 0.05])
    btn_nucvol = Button(ax_button_nucvol, 'Nucleus Volume')
    def analyze_nucleus_volume(event):
        # Red channel assumed to be overlays[..., 0]
        dapi = overlays[..., 0]
        # Normalize and segment as in main analysis
        dapi_n = (dapi - np.min(dapi)) / (np.max(dapi) - np.min(dapi)) if np.max(dapi) > np.min(dapi) else np.zeros_like(dapi)
        from skimage import exposure, filters, morphology, measure
        from scipy import ndimage as ndi
        dapi_eq = exposure.equalize_adapthist(dapi_n)
        thresh = filters.threshold_otsu(dapi_eq)
        binary = dapi_eq > thresh
        binary = morphology.remove_small_objects(binary, min_size=500)
        binary = ndi.binary_fill_holes(binary)
        nuclei_labels = measure.label(binary)
        voxel_volume = 0.095 * 0.095 * 0.135  # μm³
        labels = np.unique(nuclei_labels)
        labels = labels[labels != 0]  # Exclude background
        results = []
        for label in labels:
            voxels = np.sum(nuclei_labels == label)
            volume = voxels * voxel_volume
            results.append((label, volume, voxels))
        # Prepare message
        msg = "Nucleus Volumes (per object):\n"
        msg += "Label\tVolume (μm³)\tVoxels\n"
        for label, volume, voxels in results:
            msg += f"{label}\t{volume:.2f}\t{voxels}\n"
        print(msg)
        import tkinter.messagebox as mbox
        mbox.showinfo("Nucleus Volumes", msg)
    btn_nucvol.on_clicked(analyze_nucleus_volume)

    # --- Add cell volume analysis button (after overlays and plt are defined) ---
    ax_button_cellvol = plt.axes([0.05, 0.24, 0.15, 0.05])
    btn_cellvol = Button(ax_button_cellvol, 'Cell Volume')
    def analyze_cell_volume(event):
        # Sum normalized intensities from all 3 channels
        stack = overlays.astype(np.float32)
        # Normalize each channel to [0, 1] before summing
        norm_stack = np.zeros_like(stack)
        for c in range(3):
            ch = stack[..., c]
            vmin, vmax = np.min(ch), np.max(ch)
            if vmax > vmin:
                norm_stack[..., c] = (ch - vmin) / (vmax - vmin)
            else:
                norm_stack[..., c] = 0
        summed = np.sum(norm_stack, axis=-1)
        # Segment cell volume by thresholding the sum
        from skimage import filters, morphology, measure
        from scipy import ndimage as ndi
        thresh = filters.threshold_otsu(summed)
        binary = summed > thresh
        binary = morphology.remove_small_objects(binary, min_size=2000)
        binary = ndi.binary_fill_holes(binary)
        cell_labels = measure.label(binary)
        voxel_volume = 0.095 * 0.095 * 0.135  # μm³
        labels = np.unique(cell_labels)
        labels = labels[labels != 0]
        results = []
        for label in labels:
            voxels = np.sum(cell_labels == label)
            volume = voxels * voxel_volume
            results.append((label, volume, voxels))
        # Prepare message
        msg = "Cell Volumes (per object):\n"
        msg += "Label\tVolume (μm³)\tVoxels\n"
        for label, volume, voxels in results:
            msg += f"{label}\t{volume:.2f}\t{voxels}\n"
        print(msg)
        import tkinter.messagebox as mbox
        mbox.showinfo("Cell Volumes", msg)
    btn_cellvol.on_clicked(analyze_cell_volume)

    # --- Add mitochondria count per cell button ---
    ax_button_mitocount = plt.axes([0.05, 0.31, 0.15, 0.05])
    btn_mitocount = Button(ax_button_mitocount, 'Mito Count')
    def analyze_mito_per_cell(event):
        # Segment cells (sum of normalized channels)
        stack = overlays.astype(np.float32)
        norm_stack = np.zeros_like(stack)
        for c in range(3):
            ch = stack[..., c]
            vmin, vmax = np.min(ch), np.max(ch)
            if vmax > vmin:
                norm_stack[..., c] = (ch - vmin) / (vmax - vmin)
            else:
                norm_stack[..., c] = 0
        summed = np.sum(norm_stack, axis=-1)
        from skimage import filters, morphology, measure
        from scipy import ndimage as ndi
        cell_thresh = filters.threshold_otsu(summed)
        cell_binary = summed > cell_thresh
        cell_binary = morphology.remove_small_objects(cell_binary, min_size=2000)
        cell_binary = ndi.binary_fill_holes(cell_binary)
        cell_labels = measure.label(cell_binary)
        # Segment mitochondria (blue channel)
        mito_stack = overlays[..., 2]
        mito_norm = (mito_stack - np.min(mito_stack)) / (np.max(mito_stack) - np.min(mito_stack)) if np.max(mito_stack) > np.min(mito_stack) else np.zeros_like(mito_stack)
        mito_eq = exposure.equalize_adapthist(mito_norm)
        mito_thresh = filters.threshold_otsu(mito_eq)
        mito_binary = mito_eq > mito_thresh
        mito_binary = morphology.remove_small_objects(mito_binary, min_size=50)
        mito_binary = ndi.binary_fill_holes(mito_binary)
        mito_labels = measure.label(mito_binary)
        # For each cell, count mitochondria objects inside
        results = []
        cell_ids = np.unique(cell_labels)
        cell_ids = cell_ids[cell_ids != 0]
        for cell_id in cell_ids:
            cell_mask = (cell_labels == cell_id)
            mito_in_cell = mito_labels * cell_mask
            mito_ids = np.unique(mito_in_cell)
            mito_ids = mito_ids[(mito_ids != 0)]
            mito_count = len(mito_ids)
            results.append((cell_id, mito_count))
        # Prepare message
        msg = "Mitochondria Count per Cell:\nCell\tMito Count\n"
        for cell_id, mito_count in results:
            msg += f"{cell_id}\t{mito_count}\n"
        print(msg)
        import tkinter.messagebox as mbox
        mbox.showinfo("Mitochondria Count per Cell", msg)
    btn_mitocount.on_clicked(analyze_mito_per_cell)

    # --- Add save mito z-stack button ---
    ax_button_savemito = plt.axes([0.05, 0.38, 0.15, 0.05])
    btn_savemito = Button(ax_button_savemito, 'Save Mito Z-Stack')
    def save_mito_zstack(event):
        from pathlib import Path
        import tifffile
        import datetime
        mito_stack = overlays[..., 2]
        out_dir = Path('results') / f"{selected_well}_{tile}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'mito_zstack.tif'
        tifffile.imwrite(str(out_path), mito_stack.astype(np.float32))
        import tkinter.messagebox as mbox
        mbox.showinfo("Saved", f"Mitochondria z-stack saved to:\n{out_path}")
    btn_savemito.on_clicked(save_mito_zstack)

    plt.show()

# --- Add main block to run viewer ---
if __name__ == '__main__':
    # Usage: python viewer.py <folder> <xlsx_path> [ch1 ch2 ch3]
    import sys
    default_folder = "/Users/enricoklotzsch/Documents/Raphael/dataset-f68ff1b2-1d51-11f1-9126-02420a000421/Original/Images/"
    default_xlsx = "/Users/enricoklotzsch/Documents/Raphael/dataset-f68ff1b2-1d51-11f1-9126-02420a000421/Original/Images/image_id_to_tiff_mapping.xlsx"
    folder = default_folder
    xlsx_path = default_xlsx
    channel_ids = [413, 414, 415]
    if len(sys.argv) >= 3:
        folder = sys.argv[1]
        xlsx_path = sys.argv[2]
        if len(sys.argv) >= 6:
            channel_ids = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
    show_overlay_viewer(folder, channel_ids, xlsx_path)