import os
import sys
if sys.version_info < (3, 11):
    print("\n[ERROR] This script must be run with Python 3.11 or newer (e.g., your .venv311 environment).\nActivate the correct environment with:\n\n    source .venv311/bin/activate\n    python scripts/napari_viewer.py\n\nCurrent Python: {}.{}.{}\n".format(*sys.version_info[:3]))
    sys.exit(1)
import numpy as np
import tifffile
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from xml_mapping import parse_xlsx_mapping
from skimage import filters, morphology, measure, exposure
from scipy import ndimage as ndi

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
        new_tiles = sorted([str(t) for t in mapping[selected_well].keys()])
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
    z_dict = mapping[well][tile]
    # Sort z-slices naturally (e.g., P1, P2, ..., P10)
    zs = sorted(z_dict.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    overlays = []
    ref_shape = None
    # Find a reference shape
    for zslice in zs:
        for ch in channel_keys:
            tiff_file = z_dict[zslice].get(ch)
            tiff_path = os.path.join(folder, tiff_file) if tiff_file else None
            if tiff_file and tiff_path and os.path.exists(tiff_path):
                ref_shape = tifffile.imread(tiff_path).shape
                break
        if ref_shape is not None:
            break
    if ref_shape is None:
        raise RuntimeError("No reference image found for fallback shape. Please check your mapping and TIFF files.")
    for zslice in zs:
        ch_files = []
        for ch in channel_keys:
            tiff_file = z_dict[zslice].get(ch)
            tiff_path = os.path.join(folder, tiff_file) if tiff_file else None
            if tiff_file and tiff_path and os.path.exists(tiff_path):
                try:
                    img = tifffile.imread(tiff_path).astype(np.float32)
                    ch_files.append(img)
                except Exception as e:
                    print(f"[ERROR] Could not read {tiff_path}: {e}")
                    ch_files.append(np.zeros(ref_shape, dtype=np.float32))
            else:
                print(f"[WARN] Missing TIFF for slice={zslice} ch={ch}, using zeros.")
                ch_files.append(np.zeros(ref_shape, dtype=np.float32))
        try:
            overlay = np.stack(ch_files, axis=-1)
        except Exception as e:
            print(f"[ERROR] Could not stack channels for slice={zslice}: {e}")
            overlay = np.zeros((*ref_shape, len(channel_keys)), dtype=np.float32)
        overlays.append(overlay)
    overlays = np.array(overlays)
    return overlays, zs

def mitochondria_analysis(mito_stack, threshold=0.5, min_size=50):
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

    # --- Only load DAPI and Mitochondria channels ---
    # Assume DAPI is R1, Mito is R3 (adjust if needed)
    channel_keys = ['R1', 'R3']
    overlays, zs = load_stack_from_mapping(folder, mapping, well, tile, channel_keys)

    # --- Launch napari ---
    import napari
    from magicgui import magicgui
    viewer = napari.Viewer()
    viewer.add_image(overlays[..., 0], name='Nucl Image', scale=(1, 1, 1), blending='additive', cache=True)
    mito_layer = viewer.add_image(overlays[..., 1], name='Mito Image', scale=(1, 1, 1), blending='additive', cache=True)

    # --- Interactive threshold slider ---
    @magicgui(call_button='Update Mitochondria Mask', threshold={'widget_type': 'FloatSlider', 'min': 0.0, 'max': 1.0, 'step': 0.01}, min_size={'widget_type': 'SpinBox', 'min': 1, 'max': 1000, 'step': 1})
    def mito_threshold_widget(threshold: float = 0.5, min_size: int = 50):
        mito_stack = overlays[..., 1]
        mito_labels, mito_props = mitochondria_analysis(mito_stack, threshold=threshold, min_size=min_size)
        # Remove previous mask layer if exists
        if 'Mito Mask' in [l.name for l in viewer.layers]:
            viewer.layers['Mito Mask'].data = mito_labels
        else:
            viewer.add_labels(mito_labels, name='Mito Mask')
        print(pd.DataFrame(mito_props))
        return mito_labels
    viewer.window.add_dock_widget(mito_threshold_widget, area='right')

    # --- Analyze Mitochondria Button ---
    @magicgui(call_button='Analyze Mitochondria')
    def analyze_mitochondria(threshold: float = 0.5, min_size: int = 50):
        mito_stack = overlays[..., 1]
        mito_labels, mito_props = mitochondria_analysis(mito_stack, threshold=threshold, min_size=min_size)
        # Save results
        out_dir = os.path.join('results', f"mito_{well}_{tile}")
        os.makedirs(out_dir, exist_ok=True)
        # tifffile.imwrite(os.path.join(out_dir, 'mito_labels.tif'), mito_labels.astype(np.uint16))
        pd.DataFrame(mito_props).to_csv(os.path.join(out_dir, 'mito_morphometry.csv'), index=False)
        print(f"[INFO] Saved mitochondria labels and morphometry to {out_dir}")
        return mito_labels
    viewer.window.add_dock_widget(analyze_mitochondria, area='right')

    napari.run()

if __name__ == '__main__':
    main()

