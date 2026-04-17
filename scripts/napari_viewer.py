import os
import sys
if sys.version_info < (3, 11):
    print("\n[ERROR] This script must be run with Python 3.11 or newer (e.g., your .venv311 environment).\nActivate the correct environment with:\n\n    source .venv311/bin/activate\n    python scripts/napari_viewer.py\n\nCurrent Python: {}.{}.{}\n".format(*sys.version_info[:3]))
    sys.exit(1)
import numpy as np
import tifffile
import pandas as pd
import dask.array as da
from dask import delayed
import tkinter as tk
from tkinter import filedialog, ttk
from xml_mapping import parse_xlsx_mapping
from skimage import filters, morphology, measure, exposure
from scipy import ndimage as ndi


def sort_mapping_key(value: str):
    if value is None or value == '':
        return -1
    return int(value[1:]) if isinstance(value, str) and len(value) > 1 and value[1:].isdigit() else value


def format_timepoint_label(timepoint):
    return timepoint if timepoint else '(no timepoint)'


def get_tiff_shape(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        return tif.series[0].shape


def read_tiff_float32(tiff_path):
    return tifffile.imread(tiff_path).astype(np.float32)


def load_tiff_lazy(tiff_path, shape):
    return da.from_delayed(
        delayed(read_tiff_float32)(tiff_path),
        shape=shape,
        dtype=np.float32,
    )


def get_timepoints(mapping, well, tile):
    return sorted(mapping[well][tile].keys(), key=sort_mapping_key)


def select_well_tile(mapping):
    wells = sorted([str(w) for w in mapping.keys()])
    tiles = sorted([str(t) for t in mapping[wells[0]].keys()]) if wells else []
    root = tk.Tk()
    root.title('Select Well and Tile')
    tk.Label(root, text='Select Well:').grid(row=0, column=0, padx=10, pady=5)
    well_listbox = tk.Listbox(root, exportselection=False, height=min(10, len(wells)))
    for w in wells:
        well_listbox.insert(tk.END, w)
    well_listbox.grid(row=1, column=0, padx=10, pady=5)
    well_listbox.selection_set(0)
    tk.Label(root, text='Select Tile:').grid(row=0, column=1, padx=10, pady=5)
    tile_listbox = tk.Listbox(root, exportselection=False, height=min(10, len(tiles)))
    for t in tiles:
        tile_listbox.insert(tk.END, t)
    tile_listbox.grid(row=1, column=1, padx=10, pady=5)
    tile_listbox.selection_set(0)

    def update_tiles_on_well_select(event):
        selection = well_listbox.curselection()
        if not selection:
            return
        selected_well = wells[selection[0]]
        new_tiles = sorted([str(t) for t in mapping[selected_well].keys()], key=sort_mapping_key)
        tile_listbox.delete(0, tk.END)
        for t in new_tiles:
            tile_listbox.insert(tk.END, t)
        if new_tiles:
            tile_listbox.selection_set(0)

    well_listbox.bind('<<ListboxSelect>>', update_tiles_on_well_select)
    selection = {'well': None, 'tile': None}

    def on_select():
        sel = well_listbox.curselection()
        if sel:
            selection['well'] = wells[sel[0]]
        else:
            selection['well'] = wells[0]
        tile_sel = tile_listbox.curselection()
        if tile_sel:
            selection['tile'] = tile_listbox.get(tile_sel[0])
        else:
            selection['tile'] = tile_listbox.get(0)
        root.quit()
        root.destroy()
    select_btn = tk.Button(root, text='Load', command=on_select)
    select_btn.grid(row=2, column=0, columnspan=2, pady=10)
    root.mainloop()
    return selection['well'], selection['tile']

def load_stack_from_mapping(folder, mapping, well, tile, channel_keys):
    timepoints = get_timepoints(mapping, well, tile)
    zs = sorted({zslice for timepoint in timepoints for zslice in mapping[well][tile][timepoint].keys()}, key=sort_mapping_key)
    overlays = []
    ref_shape = None
    # Find a reference shape
    for timepoint in timepoints:
        z_dict = mapping[well][tile][timepoint]
        for zslice in zs:
            for ch in channel_keys:
                tiff_file = z_dict.get(zslice, {}).get(ch)
                tiff_path = os.path.join(folder, tiff_file) if tiff_file else None
                if tiff_file and tiff_path and os.path.exists(tiff_path):
                    ref_shape = get_tiff_shape(tiff_path)
                    break
            if ref_shape is not None:
                break
        if ref_shape is not None:
            break
    if ref_shape is None:
        raise RuntimeError("No reference image found for fallback shape. Please check your mapping and TIFF files.")
    for timepoint in timepoints:
        z_dict = mapping[well][tile][timepoint]
        timepoint_overlays = []
        for zslice in zs:
            ch_files = []
            for ch in channel_keys:
                tiff_file = z_dict.get(zslice, {}).get(ch)
                tiff_path = os.path.join(folder, tiff_file) if tiff_file else None
                if tiff_file and tiff_path and os.path.exists(tiff_path):
                    try:
                        img = load_tiff_lazy(tiff_path, ref_shape)
                        ch_files.append(img)
                    except Exception as e:
                        print(f"[ERROR] Could not read {tiff_path}: {e}")
                        ch_files.append(da.zeros(ref_shape, dtype=np.float32, chunks='auto'))
                else:
                    print(f"[WARN] Missing TIFF for timepoint={format_timepoint_label(timepoint)} slice={zslice} ch={ch}, using zeros.")
                    ch_files.append(da.zeros(ref_shape, dtype=np.float32, chunks='auto'))
            try:
                overlay = da.stack(ch_files, axis=-1)
            except Exception as e:
                print(f"[ERROR] Could not stack images for timepoint={format_timepoint_label(timepoint)} slice={zslice}: {e}")
                overlay = da.zeros((*ref_shape, len(channel_keys)), dtype=np.float32, chunks='auto')
            timepoint_overlays.append(overlay)
        overlays.append(da.stack(timepoint_overlays, axis=0))
    overlays = da.stack(overlays, axis=0)
    return overlays, timepoints, zs


def get_channel_keys(mapping, well, tile):
    channel_keys = set()
    for timepoint in get_timepoints(mapping, well, tile):
        z_dict = mapping[well][tile][timepoint]
        for channels in z_dict.values():
            channel_keys.update(channels.keys())
    return sorted(channel_keys, key=sort_mapping_key)

def mitochondria_analysis(mito_stack, threshold=0.5, min_size=50):
    if hasattr(mito_stack, 'compute'):
        mito_stack = mito_stack.compute()
    mito_norm = (mito_stack - np.min(mito_stack)) / (np.max(mito_stack) - np.min(mito_stack)) if np.max(mito_stack) > np.min(mito_stack) else np.zeros_like(mito_stack)
    mito_eq = exposure.equalize_adapthist(mito_norm)
    binary = mito_eq > threshold
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = ndi.binary_fill_holes(binary)
    mito_labels = measure.label(binary)
    mito_props = measure.regionprops_table(mito_labels, properties=['label', 'area', 'solidity'])
    return mito_labels, mito_props

def main():
    # --- Select XLSX mapping file ---
    root = tk.Tk()
    root.withdraw()
    xlsx_path = filedialog.askopenfilename(title='Select XLSX mapping file', filetypes=[('Excel files', '*.xlsx')])
    if not xlsx_path:
        print('No XLSX file selected. Exiting.')
        sys.exit(1)
    mapping, _ = parse_xlsx_mapping(xlsx_path)

    # --- Select image folder ---
    folder = filedialog.askdirectory(title='Select folder with TIFF files')
    if not folder:
        print('No folder selected. Exiting.')
        sys.exit(1)

    # --- Select well and tile ---
    well, tile = select_well_tile(mapping)
    print(f"[DEBUG] Selected well: {well}, tile: {tile}")

    # --- Load all available channels, timepoints and z-slices for the selected well/tile ---
    channel_keys = get_channel_keys(mapping, well, tile)
    if not channel_keys:
        print('No channels found for the selected well/tile. Exiting.')
        sys.exit(1)
    overlays, timepoints, zs = load_stack_from_mapping(folder, mapping, well, tile, channel_keys)
    print(f"[DEBUG] Loaded timepoints: {[format_timepoint_label(timepoint) for timepoint in timepoints]}")
    print(f"[DEBUG] Loaded channels: {channel_keys}")

    mito_channel_key = 'R3' if 'R3' in channel_keys else channel_keys[-1]
    mito_channel_index = channel_keys.index(mito_channel_key)

    def get_selected_mito_stack():
        if overlays.shape[0] > 1:
            timepoint_index = int(viewer.dims.current_step[0])
        else:
            timepoint_index = 0
        return overlays[timepoint_index, ..., mito_channel_index], timepoint_index

    # --- Launch napari ---
    import napari
    from magicgui import magicgui
    viewer = napari.Viewer()
    for i, channel_key in enumerate(channel_keys):
        viewer.add_image(overlays[..., i], name=f'{channel_key} Image', scale=(1, 1, 1, 1), blending='additive', cache=True)

    # --- Interactive threshold slider ---
    @magicgui(call_button='Update Mitochondria Mask', threshold={'widget_type': 'FloatSlider', 'min': 0.0, 'max': 1.0, 'step': 0.01}, min_size={'widget_type': 'SpinBox', 'min': 1, 'max': 1000, 'step': 1})
    def mito_threshold_widget(threshold: float = 0.5, min_size: int = 50):
        mito_stack, timepoint_index = get_selected_mito_stack()
        mito_labels, mito_props = mitochondria_analysis(mito_stack, threshold=threshold, min_size=min_size)
        # Remove previous mask layer if exists
        if 'Mito Mask' in [l.name for l in viewer.layers]:
            viewer.layers['Mito Mask'].data = mito_labels
        else:
            viewer.add_labels(mito_labels, name='Mito Mask')
        print(f"[INFO] Updated mitochondria mask for {format_timepoint_label(timepoints[timepoint_index])}")
        print(pd.DataFrame(mito_props))
        return mito_labels
    viewer.window.add_dock_widget(mito_threshold_widget, area='right')

    # --- Analyze Mitochondria Button ---
    @magicgui(call_button='Analyze Mitochondria')
    def analyze_mitochondria(threshold: float = 0.5, min_size: int = 50):
        mito_stack, timepoint_index = get_selected_mito_stack()
        mito_labels, mito_props = mitochondria_analysis(mito_stack, threshold=threshold, min_size=min_size)
        # Save results
        result_name = f"mito_{well}_{tile}"
        timepoint = timepoints[timepoint_index]
        if timepoint:
            result_name = f"{result_name}_{timepoint}"
        out_dir = os.path.join('results', result_name)
        os.makedirs(out_dir, exist_ok=True)
        # tifffile.imwrite(os.path.join(out_dir, 'mito_labels.tif'), mito_labels.astype(np.uint16))
        pd.DataFrame(mito_props).to_csv(os.path.join(out_dir, 'mito_morphometry.csv'), index=False)
        print(f"[INFO] Saved mitochondria labels and morphometry to {out_dir} using {mito_channel_key}")
        return mito_labels
    viewer.window.add_dock_widget(analyze_mitochondria, area='right')

    napari.run()

if __name__ == '__main__':
    main()
