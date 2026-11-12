import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import tifffile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skimage import filters, measure, morphology, exposure
from skimage.restoration import rolling_ball
from scipy.ndimage import binary_fill_holes


# ======================================================
# DEFAULT PARAMETERS
# ======================================================
pixel_size_um = 0.0866

cell_channel = 1       # second channel
nucleus_channel = 2    # third channel

z_slice_index = None   # None = max projection

min_nucleus_size_px = 100
min_cell_size_px = 2000
# ======================================================


def load_ome_as_tczyx(filepath):
    with tifffile.TiffFile(filepath) as tif:
        img = tif.asarray()
        axes = tif.series[0].axes

    if "T" not in axes:
        img = np.expand_dims(img, axis=0)
        axes = "T" + axes

    if "Z" not in axes:
        y_index = axes.index("Y")
        img = np.expand_dims(img, axis=y_index)
        axes = axes[:y_index] + "Z" + axes[y_index:]

    required_axes = "TCZYX"
    order = [axes.index(ax) for ax in required_axes]
    img = np.transpose(img, order)

    return img, axes


def get_2d_image(stack):
    if z_slice_index is None:
        return np.max(stack, axis=0), "max_projection"
    return stack[z_slice_index], z_slice_index


def normalize_image(img):
    img = img.astype(float)
    p1, p99 = np.percentile(img, (1, 99.8))
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img


def subtract_background(img, radius):
    if radius <= 0:
        return img
    bg = rolling_ball(img, radius=radius)
    corrected = img - bg
    corrected[corrected < 0] = 0
    return corrected


def segment_cells(cell_img, threshold, rb_radius):
    img = subtract_background(cell_img, rb_radius)
    img = normalize_image(img)
    img = exposure.equalize_adapthist(img)

    cell_mask = img > threshold

    cell_mask = morphology.binary_opening(cell_mask, morphology.disk(5))
    cell_mask = morphology.binary_closing(cell_mask, morphology.disk(7))
    cell_mask = morphology.remove_small_objects(cell_mask, min_size=min_cell_size_px)
    cell_mask = morphology.remove_small_holes(cell_mask, area_threshold=5000)
    cell_mask = binary_fill_holes(cell_mask)

    cells_labeled = measure.label(cell_mask)
    return cells_labeled, cell_mask, img


def segment_nuclei(nuc_img, threshold, rb_radius):
    img = subtract_background(nuc_img, rb_radius)
    img = normalize_image(img)
    img = exposure.equalize_adapthist(img)

    nuc_mask = img > threshold

    nuc_mask = morphology.remove_small_objects(nuc_mask, min_size=min_nucleus_size_px)
    nuc_mask = morphology.remove_small_holes(nuc_mask, area_threshold=200)
    nuc_mask = morphology.binary_opening(nuc_mask, morphology.disk(2))

    nuclei_labeled = measure.label(nuc_mask)
    return nuclei_labeled, nuc_mask, img


class CellNucleusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell / Nucleus Size Over Time")

        self.filepath = None
        self.img = None
        self.T = 0
        self.C = 0
        self.Z = 0

        self.cell_thresholds = {}
        self.nucleus_thresholds = {}
        self.cell_rb_radii = {}
        self.nucleus_rb_radii = {}

        self.current_t = tk.IntVar(value=0)

        self.setup_gui()

    def setup_gui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(
            control_frame,
            text="Load OME-TIFF",
            command=self.load_file
        ).pack(fill=tk.X, pady=5)

        self.file_label = tk.Label(control_frame, text="No file loaded", wraplength=250)
        self.file_label.pack(pady=5)

        tk.Label(control_frame, text="Time point").pack()
        self.time_slider = tk.Scale(
            control_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_time_change
        )
        self.time_slider.pack(fill=tk.X)

        tk.Label(control_frame, text="Cell threshold").pack()
        self.cell_thresh_slider = tk.Scale(
            control_frame,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change
        )
        self.cell_thresh_slider.set(0.35)
        self.cell_thresh_slider.pack(fill=tk.X)

        tk.Label(control_frame, text="Nucleus threshold").pack()
        self.nuc_thresh_slider = tk.Scale(
            control_frame,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change
        )
        self.nuc_thresh_slider.set(0.35)
        self.nuc_thresh_slider.pack(fill=tk.X)

        tk.Label(control_frame, text="Cell rolling-ball radius").pack()
        self.cell_rb_slider = tk.Scale(
            control_frame,
            from_=0,
            to=200,
            resolution=1,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change
        )
        self.cell_rb_slider.set(50)
        self.cell_rb_slider.pack(fill=tk.X)

        tk.Label(control_frame, text="Nucleus rolling-ball radius").pack()
        self.nuc_rb_slider = tk.Scale(
            control_frame,
            from_=0,
            to=200,
            resolution=1,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change
        )
        self.nuc_rb_slider.set(30)
        self.nuc_rb_slider.pack(fill=tk.X)

        tk.Button(
            control_frame,
            text="Auto Otsu thresholds for current frame",
            command=self.auto_threshold_current
        ).pack(fill=tk.X, pady=5)

        tk.Button(
            control_frame,
            text="Apply current settings to all time points",
            command=self.apply_current_to_all
        ).pack(fill=tk.X, pady=5)

        tk.Button(
            control_frame,
            text="Analyze and Export CSV + Plot",
            command=self.analyze_and_export
        ).pack(fill=tk.X, pady=15)

        self.status_label = tk.Label(control_frame, text="", wraplength=250)
        self.status_label.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select OME-TIFF",
            filetypes=[
                ("OME-TIFF", "*.ome.tif *.ome.tiff"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not filepath:
            return

        self.filepath = filepath
        self.img, axes = load_ome_as_tczyx(filepath)

        self.T, self.C, self.Z, Y, X = self.img.shape

        if cell_channel >= self.C or nucleus_channel >= self.C:
            messagebox.showerror(
                "Channel error",
                f"File has only {self.C} channels. "
                f"cell_channel={cell_channel}, nucleus_channel={nucleus_channel}"
            )
            return

        self.file_label.config(text=Path(filepath).name)
        self.time_slider.config(to=self.T - 1)

        for t in range(self.T):
            self.cell_thresholds[t] = 0.35
            self.nucleus_thresholds[t] = 0.35
            self.cell_rb_radii[t] = 50
            self.nucleus_rb_radii[t] = 30

        self.status_label.config(
            text=f"Loaded: T={self.T}, C={self.C}, Z={self.Z}"
        )

        self.update_preview()

    def save_current_parameters(self):
        if self.img is None:
            return

        t = self.time_slider.get()

        self.cell_thresholds[t] = self.cell_thresh_slider.get()
        self.nucleus_thresholds[t] = self.nuc_thresh_slider.get()
        self.cell_rb_radii[t] = self.cell_rb_slider.get()
        self.nucleus_rb_radii[t] = self.nuc_rb_slider.get()

    def load_current_parameters(self):
        t = self.time_slider.get()

        self.cell_thresh_slider.set(self.cell_thresholds.get(t, 0.35))
        self.nuc_thresh_slider.set(self.nucleus_thresholds.get(t, 0.35))
        self.cell_rb_slider.set(self.cell_rb_radii.get(t, 50))
        self.nuc_rb_slider.set(self.nucleus_rb_radii.get(t, 30))

    def on_time_change(self, event=None):
        self.load_current_parameters()
        self.update_preview()

    def on_parameter_change(self, event=None):
        self.save_current_parameters()
        self.update_preview()

    def get_current_images(self, t):
        cell_stack = self.img[t, cell_channel]
        nuc_stack = self.img[t, nucleus_channel]

        cell_img, z_used = get_2d_image(cell_stack)
        nuc_img, _ = get_2d_image(nuc_stack)

        return cell_img, nuc_img, z_used

    def update_preview(self):
        if self.img is None:
            return

        t = self.time_slider.get()

        cell_img, nuc_img, _ = self.get_current_images(t)

        cells_labeled, _, cell_display = segment_cells(
            cell_img,
            self.cell_thresholds[t],
            self.cell_rb_radii[t]
        )

        nuclei_labeled, _, _ = segment_nuclei(
            nuc_img,
            self.nucleus_thresholds[t],
            self.nucleus_rb_radii[t]
        )

        self.ax.clear()
        self.ax.imshow(cell_display, cmap="gray")
        self.ax.contour(cells_labeled, colors="lime", linewidths=1)
        self.ax.contour(nuclei_labeled, colors="red", linewidths=1)

        self.ax.set_title(
            f"Time {t + 1}/{self.T} | "
            f"Cell thr={self.cell_thresholds[t]:.2f}, "
            f"Nuc thr={self.nucleus_thresholds[t]:.2f}"
        )
        self.ax.axis("off")

        self.canvas.draw()

    def auto_threshold_current(self):
        if self.img is None:
            return

        t = self.time_slider.get()

        cell_img, nuc_img, _ = self.get_current_images(t)

        cell_corr = subtract_background(cell_img, self.cell_rb_slider.get())
        nuc_corr = subtract_background(nuc_img, self.nuc_rb_slider.get())

        cell_norm = exposure.equalize_adapthist(normalize_image(cell_corr))
        nuc_norm = exposure.equalize_adapthist(normalize_image(nuc_corr))

        self.cell_thresholds[t] = filters.threshold_otsu(cell_norm)
        self.nucleus_thresholds[t] = filters.threshold_otsu(nuc_norm)

        self.load_current_parameters()
        self.update_preview()

    def apply_current_to_all(self):
        if self.img is None:
            return

        t = self.time_slider.get()

        cell_thr = self.cell_thresholds[t]
        nuc_thr = self.nucleus_thresholds[t]
        cell_rb = self.cell_rb_radii[t]
        nuc_rb = self.nucleus_rb_radii[t]

        for frame in range(self.T):
            self.cell_thresholds[frame] = cell_thr
            self.nucleus_thresholds[frame] = nuc_thr
            self.cell_rb_radii[frame] = cell_rb
            self.nucleus_rb_radii[frame] = nuc_rb

        self.status_label.config(text="Current settings applied to all time points.")

    def analyze_and_export(self):
        if self.img is None:
            return

        input_path = Path(self.filepath)
        base = input_path.stem

        output_csv = input_path.parent / f"{base}_cell_nucleus_stats.csv"
        summary_csv = input_path.parent / f"{base}_size_over_time.csv"
        thresholds_csv = input_path.parent / f"{base}_thresholds.csv"
        output_png = input_path.parent / f"{base}_size_over_time.png"

        results = []
        threshold_records = []

        for t in range(self.T):
            self.status_label.config(text=f"Analyzing time point {t + 1}/{self.T}")
            self.root.update_idletasks()

            cell_img, nuc_img, z_used = self.get_current_images(t)

            cells_labeled, _, _ = segment_cells(
                cell_img,
                self.cell_thresholds[t],
                self.cell_rb_radii[t]
            )

            nuclei_labeled, _, _ = segment_nuclei(
                nuc_img,
                self.nucleus_thresholds[t],
                self.nucleus_rb_radii[t]
            )

            threshold_records.append({
                "File": input_path.name,
                "Time index": t,
                "Time point": t + 1,
                "Cell threshold": self.cell_thresholds[t],
                "Nucleus threshold": self.nucleus_thresholds[t],
                "Cell rolling-ball radius": self.cell_rb_radii[t],
                "Nucleus rolling-ball radius": self.nucleus_rb_radii[t],
                "Z used": z_used
            })

            nucleus_props = measure.regionprops(nuclei_labeled)

            for nuc in nucleus_props:
                ny, nx = nuc.centroid
                cell_label = cells_labeled[int(ny), int(nx)]

                if cell_label == 0:
                    continue

                nucleus_region = nuclei_labeled == nuc.label
                cell_region = cells_labeled == cell_label

                nucleus_area_px = np.sum(nucleus_region)
                cell_area_px = np.sum(cell_region)

                nucleus_area_um2 = nucleus_area_px * pixel_size_um**2
                cell_area_um2 = cell_area_px * pixel_size_um**2

                results.append({
                    "File": input_path.name,
                    "Time index": t,
                    "Time point": t + 1,
                    "Z used": z_used,
                    "Nucleus ID": nuc.label,
                    "Matched Cell Label": cell_label,
                    "Nucleus centroid Y": ny,
                    "Nucleus centroid X": nx,
                    "Nucleus area px²": nucleus_area_px,
                    "Nucleus area µm²": nucleus_area_um2,
                    "Cell area px²": cell_area_px,
                    "Cell area µm²": cell_area_um2,
                    "Nucleus/Cell area ratio": (
                        nucleus_area_um2 / cell_area_um2
                        if cell_area_um2 > 0 else np.nan
                    ),
                    "Cell threshold": self.cell_thresholds[t],
                    "Nucleus threshold": self.nucleus_thresholds[t],
                    "Cell rolling-ball radius": self.cell_rb_radii[t],
                    "Nucleus rolling-ball radius": self.nucleus_rb_radii[t]
                })

        df = pd.DataFrame(results)
        threshold_df = pd.DataFrame(threshold_records)

        df.to_csv(output_csv, index=False)
        threshold_df.to_csv(thresholds_csv, index=False)

        if len(df) == 0:
            messagebox.showwarning(
                "No objects",
                "No nucleus-cell matches were found. Try lowering thresholds."
            )
            return

        summary_df = (
            df.groupby("Time point")
            .agg(
                Number_of_matched_cells=("Matched Cell Label", "nunique"),
                Mean_cell_area_um2=("Cell area µm²", "mean"),
                SD_cell_area_um2=("Cell area µm²", "std"),
                Mean_nucleus_area_um2=("Nucleus area µm²", "mean"),
                SD_nucleus_area_um2=("Nucleus area µm²", "std"),
                Mean_nucleus_cell_ratio=("Nucleus/Cell area ratio", "mean")
            )
            .reset_index()
        )

        summary_df = summary_df.rename(columns={
            "Number_of_matched_cells": "Number of matched cells",
            "Mean_cell_area_um2": "Mean cell area µm²",
            "SD_cell_area_um2": "SD cell area µm²",
            "Mean_nucleus_area_um2": "Mean nucleus area µm²",
            "SD_nucleus_area_um2": "SD nucleus area µm²",
            "Mean_nucleus_cell_ratio": "Mean nucleus/cell area ratio"
        })

        summary_df.to_csv(summary_csv, index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(
            summary_df["Time point"],
            summary_df["Mean cell area µm²"],
            marker="o",
            label="Mean cell area"
        )
        plt.plot(
            summary_df["Time point"],
            summary_df["Mean nucleus area µm²"],
            marker="o",
            label="Mean nucleus area"
        )
        plt.xlabel("Time point")
        plt.ylabel("Area [µm²]")
        plt.title(f"{base}: cell and nucleus size over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.show()

        self.status_label.config(text="Analysis complete.")

        messagebox.showinfo(
            "Done",
            f"Saved:\n\n"
            f"{output_csv}\n"
            f"{summary_csv}\n"
            f"{thresholds_csv}\n"
            f"{output_png}"
        )


def main():
    root = tk.Tk()
    app = CellNucleusGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()