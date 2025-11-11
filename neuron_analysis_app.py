# -*- coding: utf-8 -*-
"""
Neuron Tracker - PyQt GUI

Modernized PyQt-based GUI providing full feature parity with the original
Tkinter neuron_tracker_gui:

- Single-animal pipeline (ntc)
- ROI drawing & filtering
- Atlas registration & warp parameter generation
- Warp application & warp inspection
- Group-level visualizations

Requires:
    PyQt5
    matplotlib
    numpy, scipy, scikit-image, scikit-learn
    colorcet
    neuron_tracker_core.py (same API as original GUI)
"""

import os
import sys
import json
import numpy as np

import pandas as pd

from numpy import pi, arctan2
from scipy import signal
from scipy.stats import circmean
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from scipy.ndimage import generic_filter

import skimage.io
import colorcet as cet

from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.widgets import Slider, RadioButtons

from PyQt5 import QtCore, QtWidgets

import neuron_tracker_core as ntc
import cosinor as csn

# ------------------------------------------------------------
# Centralized State Management
# ------------------------------------------------------------
class AnalysisState:
    """A simple class to hold all application state."""
    def __init__(self):
        self.reset()

    def reset(self):
        # File paths
        self.input_movie_path = ""
        self.output_basename = ""
        self.atlas_roi_path = ""
        
        # Lists for workflow panels
        self.target_roi_paths = []
        self.warp_param_paths = []
        self.group_data_paths = []
        
        # In-memory data
        self.unfiltered_data = {}
        self.loaded_data = {}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clear_layout(layout: QtWidgets.QLayout):
    """Remove all widgets/items from a layout without replacing the layout object."""
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)


def add_mpl_to_tab(tab: QtWidgets.QWidget):
    """
    Ensure the tab has a QVBoxLayout, clear it, and add a FigureCanvas+Toolbar.
    Returns (fig, canvas).
    """
    layout = tab.layout()
    if layout is None:
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
    clear_layout(layout)

    fig = Figure()
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, tab)

    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    return fig, canvas


# ------------------------------------------------------------
# ROI Drawer
# ------------------------------------------------------------

class ROIDrawerDialog(QtWidgets.QDialog):
    """
    ROI drawing dialog: include/exclude polygons, writes:
      - <basename>_anatomical_roi.json
      - <basename>_roi_filtered.csv
    Calls callback(filtered_indices, rois_dict_list)
    """

    def __init__(self, parent, bg_image, roi_data, output_basename, callback,
                 vmin=None, vmax=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced ROI Definition Tool")
        self.resize(900, 900)

        self.bg_image = bg_image
        self.roi_data = roi_data
        self.output_basename = output_basename
        self.callback = callback

        self.rois = []
        self.current_vertices = []
        self.current_line = None
        self.mode = "Include"

        main_layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Select mode, then left-click to draw polygon vertices.\n"
            "Right-click to remove last point. Click 'Finish Polygon' to close."
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

        # Matplotlib
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(bg_image, cmap="gray", vmin=vmin, vmax=vmax)

        if roi_data is not None and len(roi_data) > 0:
            self.ax.plot(roi_data[:, 0], roi_data[:, 1],
                         ".", color="gray", markersize=2, alpha=0.5)

        self.ax.set_title("Click to Define ROI Polygon")
        self.canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)

        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)

        # Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.include_btn = QtWidgets.QRadioButton("Include")
        self.exclude_btn = QtWidgets.QRadioButton("Exclude")
        self.include_btn.setChecked(True)
        self.include_btn.toggled.connect(self.update_mode)
        btn_layout.addWidget(self.include_btn)
        btn_layout.addWidget(self.exclude_btn)

        finish_btn = QtWidgets.QPushButton("Finish Polygon")
        finish_btn.clicked.connect(self.finish_polygon)
        btn_layout.addWidget(finish_btn)

        confirm_btn = QtWidgets.QPushButton("Confirm All ROIs")
        confirm_btn.clicked.connect(self.confirm_rois)
        btn_layout.addWidget(confirm_btn)

        main_layout.addLayout(btn_layout)

    def update_mode(self):
        self.mode = "Include" if self.include_btn.isChecked() else "Exclude"

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # left-click
        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            self.current_vertices.append((float(event.xdata), float(event.ydata)))
            self.update_plot()
        # right-click: remove last
        elif event.button == 3:
            if self.current_vertices:
                self.current_vertices.pop()
                self.update_plot()

    def update_plot(self):
        if self.current_line is not None:
            for artist in self.current_line:
                artist.remove()
        self.current_line = None

        if len(self.current_vertices) > 1:
            xs, ys = zip(*self.current_vertices)
            color = "g-" if self.mode == "Include" else "r-"
            self.current_line = self.ax.plot(xs, ys, color)
        elif self.current_vertices:
            x, y = self.current_vertices[0]
            color = "g+" if self.mode == "Include" else "r+"
            self.current_line = self.ax.plot(x, y, color)

        self.canvas.draw_idle()

    def finish_polygon(self):
        if len(self.current_vertices) > 2:
            # close loop
            self.current_vertices.append(self.current_vertices[0])
            self.update_plot()
            self.rois.append({
                "path_vertices": list(self.current_vertices),
                "mode": self.mode,
            })
            self.current_vertices = []
            self.current_line = None
            self.ax.set_title(f"{len(self.rois)} ROI(s) defined. Draw another or Confirm.")
            self.canvas.draw_idle()

    def confirm_rois(self):
        # Save anatomical ROI JSON
        if self.rois and self.output_basename:
            filepath = f"{self.output_basename}_anatomical_roi.json"
            try:
                serializable = [
                    {"path_vertices": roi["path_vertices"], "mode": roi["mode"]}
                    for roi in self.rois
                ]
                with open(filepath, "w") as f:
                    json.dump(serializable, f, indent=4)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Save Error",
                    f"Error saving ROI file:\n{e}"
                )

        if not self.rois:
            # no ROI → clear filter
            if self.callback:
                self.callback(None, None)
        else:
            # build paths in memory
            for roi in self.rois:
                roi["path"] = Path(roi["path_vertices"])
            final_mask = np.zeros(len(self.roi_data), dtype=bool)

            include_rois = [r for r in self.rois if r["mode"] == "Include"]
            if include_rois:
                for roi in include_rois:
                    final_mask |= roi["path"].contains_points(self.roi_data)
            else:
                final_mask[:] = True

            for roi in self.rois:
                if roi["mode"] == "Exclude":
                    final_mask &= ~roi["path"].contains_points(self.roi_data)

            filtered_indices = np.where(final_mask)[0]

            # Save filtered ROI CSV
            if self.output_basename:
                try:
                    filtered = self.roi_data[filtered_indices]
                    np.savetxt(
                        f"{self.output_basename}_roi_filtered.csv",
                        filtered,
                        delimiter=",",
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self, "Save Error",
                        f"Error saving filtered ROI CSV:\n{e}"
                    )

            if self.callback:
                self.callback(filtered_indices, self.rois)

        self.accept()


# ------------------------------------------------------------
# Registration Window
# ------------------------------------------------------------

class RegistrationWindow(QtWidgets.QDialog):
    """
    Landmark-based TPS registration between atlas and each target.
    Writes *_warp_parameters.json exactly as original.
    """

    def __init__(self, parent, state, log_callback):
        super().__init__(parent)
        self.setWindowTitle("Atlas Registration Tool")
        self.resize(1200, 700)

        self.state = state
        self.atlas_path = self.state.atlas_roi_path
        # Make a copy so we can pop items without affecting the main state
        self.target_paths = list(self.state.target_roi_paths)
        self.log_callback = log_callback

        with open(self.atlas_path, "r") as f:
            self.atlas_rois = json.load(f)

        self.source_landmarks = []
        self.dest_landmarks = []
        self.warp_params = None

        main_layout = QtWidgets.QVBoxLayout(self)

        # Matplotlib
        self.fig = Figure()
        self.ax_atlas = self.fig.add_subplot(1, 2, 1)
        self.ax_target = self.fig.add_subplot(1, 2, 2)

        self.canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.calc_btn = QtWidgets.QPushButton("Calculate Warp")
        self.calc_btn.clicked.connect(self.calculate_warp)
        btn_layout.addWidget(self.calc_btn)

        self.save_btn = QtWidgets.QPushButton("Save Warp & Next")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_and_next)
        btn_layout.addWidget(self.save_btn)

        reset_btn = QtWidgets.QPushButton("Reset Landmarks")
        reset_btn.clicked.connect(self.reset_landmarks)
        btn_layout.addWidget(reset_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        main_layout.addLayout(btn_layout)

        # MPL events
        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)

        self.load_next_target()

    def load_next_target(self):
        if not self.target_paths:
            self.log_callback("All targets have been registered.")
            self.accept()
            return

        self.current_target_path = self.target_paths.pop(0)
        self.log_callback(f"Registering: {os.path.basename(self.current_target_path)}")
        with open(self.current_target_path, "r") as f:
            self.target_rois = json.load(f)

        self.reset_landmarks()
        self.update_plots()
        self.setWindowTitle(
            f"Atlas Registration - {os.path.basename(self.current_target_path)} "
            f"({len(self.target_paths)} remaining)"
        )

    def reset_landmarks(self):
        self.source_landmarks = []
        self.dest_landmarks = []
        self.warp_params = None
        self.save_btn.setEnabled(False)
        self.update_plots()

    def on_click(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        # Left figure: atlas landmarks
        if event.inaxes == self.ax_atlas:
            if len(self.source_landmarks) > len(self.dest_landmarks):
                # force pairing in order
                return
            self.source_landmarks.append((event.xdata, event.ydata))
        # Right figure: target landmarks
        elif event.inaxes == self.ax_target:
            if len(self.dest_landmarks) >= len(self.source_landmarks):
                return
            self.dest_landmarks.append((event.xdata, event.ydata))

        self.update_plots()

    def update_plots(self, preview_shapes=None, warp_vectors=None):
        self.ax_atlas.clear()
        self.ax_target.clear()
        self.ax_atlas.set_title("Atlas SCN")
        self.ax_target.set_title("Target SCN")
        for roi in self.atlas_rois:
            xs, ys = zip(*roi["path_vertices"])
            self.ax_atlas.plot(xs, ys, color="black")
        for roi in getattr(self, "target_rois", []):
            xs, ys = zip(*roi["path_vertices"])
            self.ax_target.plot(xs, ys, color="black")
        for i, (x, y) in enumerate(self.source_landmarks):
            self.ax_atlas.text(x, y, str(i + 1), color="red", ha="center", va="center", weight="bold")
        for i, (x, y) in enumerate(self.dest_landmarks):
            self.ax_target.text(x, y, str(i + 1), color="red", ha="center", va="center", weight="bold")
        if preview_shapes:
            for shape in preview_shapes:
                xs, ys = shape[:, 0], shape[:, 1]
                self.ax_atlas.plot(xs, ys, color="cyan", linestyle="--")
        if warp_vectors:
            ox, oy, dx, dy = warp_vectors
            self.ax_target.quiver(ox, oy, dx, dy, color="cyan", angles="xy", scale_units="xy", scale=1)
        for ax in (self.ax_atlas, self.ax_target):
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.autoscale_view()
        l1, r1 = self.ax_atlas.get_xlim()
        b1, t1 = self.ax_atlas.get_ylim()
        l2, r2 = self.ax_target.get_xlim()
        b2, t2 = self.ax_target.get_ylim()
        final_l, final_r = min(l1, l2), max(r1, r2)
        final_b, final_t = max(b1, b2), min(t1, t2)
        self.ax_atlas.set_xlim(final_l, final_r)
        self.ax_atlas.set_ylim(final_b, final_t)
        self.ax_target.set_xlim(final_l, final_r)
        self.ax_target.set_ylim(final_b, final_t)
        self.canvas.draw_idle()

    def calculate_warp(self):
        if len(self.source_landmarks) < 3 or len(self.source_landmarks) != len(self.dest_landmarks):
            self.log_callback("Error: Need ≥3 matched landmark pairs.")
            return
        source_pts = np.array(self.source_landmarks)
        dest_pts = np.array(self.dest_landmarks)
        sc = source_pts.mean(axis=0)
        dc = dest_pts.mean(axis=0)
        ssd = np.sqrt(np.mean(np.sum((source_pts - sc) ** 2, axis=1)))
        dsd = np.sqrt(np.mean(np.sum((dest_pts - dc) ** 2, axis=1)))
        if ssd == 0 or dsd == 0:
            self.log_callback("Error: Degenerate landmark configuration.")
            return
        source_norm = (source_pts - sc) / ssd
        dest_norm = (dest_pts - dc) / dsd
        from skimage.transform import ThinPlateSplineTransform
        tps = ThinPlateSplineTransform()
        tps.estimate(dest_norm, source_norm)
        self.warp_params = {
            "source_centroid": sc.tolist(), "dest_centroid": dc.tolist(),
            "source_scale": float(ssd), "dest_scale": float(dsd),
            "source_landmarks_norm": source_norm.tolist(), "dest_landmarks_norm": dest_norm.tolist(),
            "source_landmarks": self.source_landmarks, "destination_landmarks": self.dest_landmarks,
        }
        preview = []
        for roi in self.target_rois:
            verts = np.array(roi["path_vertices"])
            verts_norm = (verts - dc) / dsd
            warped_norm = tps(verts_norm)
            warped = warped_norm * ssd + sc
            preview.append(warped)
        dx = source_pts[:, 0] - dest_pts[:, 0]
        dy = source_pts[:, 1] - dest_pts[:, 1]
        warp_vectors = (dest_pts[:, 0], dest_pts[:, 1], dx, dy)
        self.update_plots(preview_shapes=preview, warp_vectors=warp_vectors)
        self.save_btn.setEnabled(True)
        self.log_callback("Warp computed; review preview and click 'Save Warp & Next'.")

    def save_and_next(self):
        if not self.warp_params:
            self.log_callback("Error: compute warp before saving.")
            return
        out_path = self.current_target_path.replace("_anatomical_roi.json", "_warp_parameters.json")
        try:
            with open(out_path, "w") as f:
                json.dump(self.warp_params, f, indent=4)
            self.log_callback(f"Saved warp parameters: {os.path.basename(out_path)}")
        except Exception as e:
            self.log_callback(f"Error saving warp file: {e}")
            return
        self.load_next_target()


# ------------------------------------------------------------
# Warp Inspector
# ------------------------------------------------------------

class WarpInspectorWindow(QtWidgets.QDialog):
    """
    Diagnostic visualization: original target vs warped (normalized atlas space),
    with option to overlay normalized original.
    """

    def __init__(self, parent, atlas_roi_path, target_roi_path,
                 original_points, warped_points,
                 original_points_norm, warped_points_norm,
                 warp_params, title):
        super().__init__(parent)
        self.setWindowTitle(f"Warp Inspector - {title}")
        self.resize(1200, 700)

        self.atlas_roi_path = atlas_roi_path
        self.target_roi_path = target_roi_path
        self.original_points = original_points
        self.warped_points = warped_points
        self.original_points_norm = original_points_norm
        self.warped_points_norm = warped_points_norm
        self.warp_params = warp_params
        self.overlay_visible = False
        self.overlay_artists = []
        main_layout = QtWidgets.QVBoxLayout(self)
        self.fig = Figure(figsize=(12, 6))
        self.ax_before = self.fig.add_subplot(1, 2, 1)
        self.ax_after = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)
        btn_layout = QtWidgets.QHBoxLayout()
        self.overlay_btn = QtWidgets.QCheckBox("Show Normalized Original Data as Overlay")
        self.overlay_btn.stateChanged.connect(self.toggle_overlay)
        btn_layout.addWidget(self.overlay_btn)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)
        self.draw_plots()

    def draw_plots(self):
        self.ax_before.clear()
        self.ax_before.set_title("Before Warp: Original Target")
        with open(self.target_roi_path, "r") as f: trois = json.load(f)
        for roi in trois:
            xs, ys = zip(*roi["path_vertices"])
            self.ax_before.plot(xs, ys, color="black", linewidth=2)
        self.ax_before.scatter(self.original_points[:, 0], self.original_points[:, 1], s=10, alpha=0.7, c="blue")
        self.ax_before.set_aspect("equal", adjustable="box")
        self.ax_before.invert_yaxis()
        self.ax_after.clear()
        self.ax_after.set_title("After Warp: Normalized Atlas Space")
        sc = np.array(self.warp_params["source_centroid"])
        ss = self.warp_params["source_scale"]
        with open(self.atlas_roi_path, "r") as f: arois = json.load(f)
        for roi in arois:
            verts = (np.array(roi["path_vertices"]) - sc) / ss
            self.ax_after.plot(verts[:, 0], verts[:, 1], color="black", linewidth=2)
        self.ax_after.scatter(self.warped_points_norm[:, 0], self.warped_points_norm[:, 1], s=10, alpha=0.7, c="red")
        self.ax_after.set_aspect("equal", adjustable="box")
        self.ax_after.invert_yaxis()
        self.toggle_overlay()
        self.canvas.draw_idle()

    def toggle_overlay(self):
        for a in self.overlay_artists: a.remove()
        self.overlay_artists.clear()
        if self.overlay_btn.isChecked():
            dc = np.array(self.warp_params["dest_centroid"])
            ds = self.warp_params["dest_scale"]
            with open(self.target_roi_path, "r") as f: trois = json.load(f)
            for roi in trois:
                verts_norm = (np.array(roi["path_vertices"]) - dc) / ds
                line = self.ax_after.plot(verts_norm[:, 0], verts_norm[:, 1], color="blue", linestyle="--", linewidth=1, alpha=0.5)[0]
                self.overlay_artists.append(line)
            scatter = self.ax_after.scatter(self.original_points_norm[:, 0], self.original_points_norm[:, 1], s=10, alpha=0.3, c="blue", marker="x")
            self.overlay_artists.append(scatter)
        self.ax_after.autoscale_view()
        self.canvas.draw_idle()

# ------------------------------------------------------------
# Visualization Viewers
# ------------------------------------------------------------

class HeatmapViewer:
    def __init__(self, fig, loaded_data, filtered_indices, phases, rhythm_scores, is_emphasized, rhythm_sort_desc):
        self.fig = fig
        self.loaded_data = loaded_data
        self.traces_data = loaded_data["traces"]
        self.roi_data = loaded_data["roi"]
        self.filtered_indices = filtered_indices
        self.is_emphasized = is_emphasized
        self.rhythm_mask = None
        self.emphasis_overlay = None
        self.rhythm_sort_desc = rhythm_sort_desc
        self.last_sort_indices = np.arange(len(self.roi_data))

        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax_heatmap = self.fig.add_subplot(gs[0])
        self.ax_trace = self.fig.add_subplot(gs[1])
        
        self._prepare_normalized_data()
        
        self.image_artist = self.ax_heatmap.imshow(self.normalized_data.T, aspect="auto", cmap="viridis", interpolation="nearest")
        self.ax_heatmap.set_title("Intensity Heatmap")
        self.cbar = self.fig.colorbar(self.image_artist, ax=self.ax_heatmap, label="Normalized Intensity")

        ax_radio = self.fig.add_axes([0.01, 0.7, 0.15, 0.15])
        sort_options = ["Y-coordinate", "Phase", "Rhythmicity"]
        if phases is None: sort_options.remove("Phase")
        if rhythm_scores is None: sort_options.remove("Rhythmicity")
        
        self.radio_buttons = RadioButtons(ax_radio, sort_options)
        self.radio_buttons.on_clicked(self.on_sort_change)
        
        self.ax_trace.set_xlabel("Time (frames)")
        self.ax_trace.set_ylabel("Intensity")
        self.ax_trace.set_title("Selected Cell Trace")
        self.trace_line = self.ax_trace.plot([], [])[0]

        if rhythm_scores is not None:
            main_win = self.fig.canvas.parent().window()
            thr = float(main_win.phase_params["rhythm_threshold"][0].text())
            if self.rhythm_sort_desc: self.rhythm_mask = rhythm_scores >= thr
            else: self.rhythm_mask = rhythm_scores <= thr

        self.update_phase_data(phases, rhythm_scores, self.rhythm_mask, self.rhythm_sort_desc)
        
        if sort_options: self.on_sort_change(sort_options[0])

    def _prepare_normalized_data(self):
        intensities = self.traces_data[:, 1:]
        if intensities.size == 0: self.normalized_data = np.zeros((1, 1)); return
        mins, maxs = intensities.min(axis=0), intensities.max(axis=0)
        denom = maxs - mins; denom[denom == 0] = 1
        self.normalized_data = (intensities - mins) / denom

    def on_sort_change(self, label):
        if self.normalized_data.size == 0: return
        sort_values = self.sort_values.get(label)
        if sort_values is None: return
        is_descending = self.rhythm_sort_desc if label == "Rhythmicity" else False
        if is_descending: sort_values = -sort_values
        if self.is_emphasized and self.rhythm_mask is not None:
            final_indices = np.lexsort((sort_values, ~self.rhythm_mask))
        else:
            final_indices = np.argsort(sort_values)
        self.image_artist.set_data(self.normalized_data[:, final_indices].T)
        self.ax_heatmap.set_ylabel(f"Cells (sorted by {label})")
        if self.emphasis_overlay: self.emphasis_overlay.remove(); self.emphasis_overlay = None
        if self.is_emphasized and self.rhythm_mask is not None:
            num_rhythmic = np.sum(self.rhythm_mask)
            total_cells = len(self.rhythm_mask)
            if num_rhythmic < total_cells:
                from matplotlib.patches import Rectangle
                height = total_cells - num_rhythmic
                y_start = num_rhythmic - 0.5 
                self.emphasis_overlay = Rectangle(xy=(-0.5, y_start), width=self.normalized_data.shape[0], height=height, facecolor='black', alpha=0.6, edgecolor='none', zorder=10)
                self.ax_heatmap.add_patch(self.emphasis_overlay)
        self.last_sort_indices = final_indices
        self.fig.canvas.draw_idle()

    def update_phase_data(self, phases, rhythm_scores, rhythm_mask=None, sort_desc=True):
        self.phases, self.rhythm_scores, self.rhythm_mask, self.rhythm_sort_desc = phases, rhythm_scores, rhythm_mask, sort_desc
        self.sort_values = {"Y-coordinate": self.roi_data[:, 1]}
        if self.phases is not None: self.sort_values["Phase"] = self.phases
        if self.rhythm_scores is not None: self.sort_values["Rhythmicity"] = self.rhythm_scores
        current_sort = self.radio_buttons.value_selected
        if current_sort in self.sort_values: self.on_sort_change(current_sort)

    def update_rhythm_emphasis(self, rhythm_mask, is_emphasized):
        self.is_emphasized, self.rhythm_mask = is_emphasized, rhythm_mask
        self.on_sort_change(self.radio_buttons.value_selected)

    def update_selected_trace(self, original_index):
        if self.filtered_indices is not None:
            try: current_index = np.where(self.filtered_indices == original_index)[0][0]
            except IndexError:
                self.trace_line.set_data([], []); self.ax_trace.set_title("Selected Cell Trace (Not in current filter)")
                self.ax_trace.relim(); self.ax_trace.autoscale_view(); self.fig.canvas.draw_idle()
                return
        else: current_index = original_index
        if 0 <= current_index < self.traces_data.shape[1] - 1:
            time, intensity = self.traces_data[:, 0], self.traces_data[:, current_index + 1]
            self.trace_line.set_data(time, intensity)
            self.ax_trace.relim(); self.ax_trace.autoscale_view()
            self.ax_trace.set_title(f"Trace for ROI {original_index + 1}")
        else:
            self.trace_line.set_data([], []); self.ax_trace.set_title("Selected Cell Trace")
        self.fig.canvas.draw_idle()

    def get_export_data(self):
        if self.normalized_data.size == 0: return None, ""
        sorted_data = self.normalized_data[:, self.last_sort_indices].T
        if self.filtered_indices is not None:
            sorted_original_indices = self.filtered_indices[self.last_sort_indices]
        else:
            sorted_original_indices = self.last_sort_indices
        df = pd.DataFrame(sorted_data)
        df.columns = [f"Frame_{i}" for i in range(sorted_data.shape[1])]
        df.insert(0, "Cell_ID", sorted_original_indices + 1)
        return df, "heatmap_data.csv"

class ContrastViewer:
    def __init__(self, fig, ax, bg_image, com_points, on_change_callback, on_select_callback):
        self.fig = fig
        self.ax = ax
        self.com_points = com_points
        self.on_change_callback = on_change_callback
        self.on_select_callback = on_select_callback
        self.highlight_artist = None
        self.scatter_artists = []

        self.image_artist = ax.imshow(bg_image, cmap="gray")
        ax.set_title("Center of Mass (Click to Select Trajectory)")

        ax_contrast = fig.add_axes([0.25, 0.06, 0.65, 0.03])
        ax_brightness = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        min_val, max_val = float(bg_image.min()), float(bg_image.max())
        self.contrast_slider = Slider(ax=ax_contrast, label="Contrast", valmin=min_val, valmax=max_val, valinit=max_val)
        self.brightness_slider = Slider(ax=ax_brightness, label="Brightness", valmin=min_val, valmax=max_val, valinit=min_val)
        self.contrast_slider.on_changed(self.update)
        self.brightness_slider.on_changed(self.update)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update(None)
        self._draw_scatter() # Initial drawing

    def _draw_scatter(self, rhythm_mask=None, is_emphasized=False):
        for artist in self.scatter_artists:
            artist.remove()
        self.scatter_artists.clear()

        if len(self.com_points) == 0:
            self.fig.canvas.draw_idle()
            return

        if is_emphasized and rhythm_mask is not None:
            rhythmic_pts = self.com_points[rhythm_mask]
            non_rhythmic_pts = self.com_points[~rhythm_mask]
            
            if len(rhythmic_pts) > 0:
                s1 = self.ax.plot(rhythmic_pts[:, 0], rhythmic_pts[:, 1], ".", color="red", markersize=5, alpha=0.8)[0]
                self.scatter_artists.append(s1)
            if len(non_rhythmic_pts) > 0:
                s2 = self.ax.plot(non_rhythmic_pts[:, 0], non_rhythmic_pts[:, 1], ".", color="gray", markersize=3, alpha=0.4)[0]
                self.scatter_artists.append(s2)
        else:
            s = self.ax.plot(self.com_points[:, 0], self.com_points[:, 1], ".", color="red", markersize=5, alpha=0.8)[0]
            self.scatter_artists.append(s)
        
        self.fig.canvas.draw_idle()

    def update_rhythm_emphasis(self, rhythm_mask, is_emphasized):
        self._draw_scatter(rhythm_mask, is_emphasized)

    def on_click(self, event):
        if event.inaxes != self.ax or len(self.com_points) == 0:
            return
        distances = np.sqrt((self.com_points[:, 0] - event.xdata)**2 + (self.com_points[:, 1] - event.ydata)**2)
        selected_index = np.argmin(distances)
        if distances[selected_index] < 20:
            self.highlight_point(selected_index)
            if self.on_select_callback:
                self.on_select_callback(selected_index)

    def highlight_point(self, index):
        if self.highlight_artist:
            self.highlight_artist.remove()
            self.highlight_artist = None
        if index is not None and 0 <= index < len(self.com_points):
            point = self.com_points[index]
            self.highlight_artist = self.ax.plot(point[0], point[1], 'o', markersize=12, markerfacecolor='none', markeredgecolor='cyan', markeredgewidth=2)[0]
        self.fig.canvas.draw_idle()

    def update(self, _):
        vmin = self.brightness_slider.val
        vmax = self.contrast_slider.val
        if vmax <= vmin: vmax = vmin + 1e-6
        self.image_artist.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()
        if self.on_change_callback:
            self.on_change_callback(vmin, vmax)


class TrajectoryInspector:
    def __init__(self, fig, ax, trajectories, movie_stack):
        self.fig = fig
        self.ax = ax
        self.trajectories = trajectories
        self.movie_stack = movie_stack
        self.num_frames = len(movie_stack)
        self.num_trajectories = len(trajectories)
        self.index = 0
        self.vmin = None
        self.vmax = None
        self.bg_artist = None

        # --- Create Widgets ---
        # Previous/Next buttons for trajectory index
        ax_prev = fig.add_axes([0.7, 0.05, 0.1, 0.04])
        ax_next = fig.add_axes([0.81, 0.05, 0.1, 0.04])
        from matplotlib.widgets import Button
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(self.prev_trajectory)
        self.btn_next.on_clicked(self.next_trajectory)

        # Slider for frame navigation
        ax_slider = fig.add_axes([0.15, 0.01, 0.7, 0.03])
        self.frame_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1
        )
        self.frame_slider.on_changed(self.on_frame_change)
        
        self.update()

    def on_frame_change(self, frame_index):
        """Callback for when the frame slider is moved."""
        self.update()

    def set_trajectory(self, index):
        """Public method to set the currently displayed trajectory by its index."""
        if 0 <= index < self.num_trajectories:
            self.index = index
            self.update()

    def update_contrast(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        if self.bg_artist:
            self.bg_artist.set_clim(vmin, vmax)
            self.fig.canvas.draw_idle()

    def next_trajectory(self, _):
        if self.num_trajectories > 0:
            self.index = (self.index + 1) % self.num_trajectories
            self.update()

    def prev_trajectory(self, _):
        if self.num_trajectories > 0:
            self.index = (self.index - 1 + self.num_trajectories) % self.num_trajectories
            self.update()

    def update(self):
        self.ax.clear()
        current_frame = int(self.frame_slider.val)

        self.bg_artist = self.ax.imshow(
            self.movie_stack[current_frame],
            cmap="gray",
            vmin=self.vmin,
            vmax=self.vmax,
        )
        
        if self.num_trajectories > 0:
            traj = self.trajectories[self.index]
            
            # Plot the full trajectory path as a faint line
            self.ax.plot(
                traj[:, 1], traj[:, 0], '-', color='cyan', linewidth=1, alpha=0.7
            )
            
            # Plot a prominent but non-obscuring marker at the current frame's position
            current_pos = traj[current_frame]
            self.ax.plot(
                current_pos[1], current_pos[0], 'o', 
                markersize=10, 
                markerfacecolor=(1, 1, 0, 0.5), # Yellow with 50% transparency
                markeredgecolor='yellow',
                markeredgewidth=1.5
            )
            
            self.ax.set_title(
                f"Trajectory {self.index + 1} / {self.num_trajectories} (Frame {current_frame + 1}/{self.num_frames})"
            )
        else:
            self.ax.set_title("No Trajectories to Display")

        self.ax.set_xlim(0, self.movie_stack[0].shape[1])
        self.ax.set_ylim(self.movie_stack[0].shape[0], 0)
        self.fig.canvas.draw_idle()


class PhaseMapViewer:
    def __init__(self, fig, ax, bg_image, rhythmic_df, on_select_callback, vmin=None, vmax=None):
        self.fig = fig
        self.ax = ax
        self.rhythmic_df = rhythmic_df
        self.roi_data = self.rhythmic_df[['X_Position', 'Y_Position']].values if not self.rhythmic_df.empty else np.array([])
        self.period_hours = self.rhythmic_df['Period_Hours'].iloc[0] if not self.rhythmic_df.empty else 24
        self.on_select_callback = on_select_callback
        self.highlight_artist = None
        self.bg_artist = ax.imshow(bg_image, cmap="gray", vmin=vmin, vmax=vmax)
        if not self.rhythmic_df.empty:
            self.scatter = ax.scatter(
                self.rhythmic_df['X_Position'], self.rhythmic_df['Y_Position'],
                c=self.rhythmic_df['Relative_Phase_Hours'],
                cmap=cet.cm.cyclic_mygbm_30_95_c78, s=25, edgecolor="black", linewidth=0.5,
            )
            self.cbar = fig.colorbar(self.scatter, ax=ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("Relative Peak Time (hours)", fontsize=10)
            ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim)
            self.update_clim(max_range)
        ax.set_title("Spatiotemporal Phase Map (Click to Select Trajectory)")
        ax.set_xticks([]); ax.set_yticks([])
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax or self.rhythmic_df.empty: return
        distances = np.sqrt((self.roi_data[:, 0] - event.xdata)**2 + (self.roi_data[:, 1] - event.ydata)**2)
        selected_index = np.argmin(distances)
        if distances[selected_index] < 20:
            self.highlight_point(selected_index)
            if self.on_select_callback: self.on_select_callback(selected_index)

    def highlight_point(self, index):
        if self.highlight_artist: self.highlight_artist.remove(); self.highlight_artist = None
        if index is not None and 0 <= index < len(self.roi_data):
            point = self.roi_data[index]
            self.highlight_artist = self.ax.plot(point[0], point[1], 'o', markersize=15, markerfacecolor='none', markeredgecolor='white', markeredgewidth=2)[0]
        self.fig.canvas.draw_idle()

    def update_contrast(self, vmin, vmax):
        if self.bg_artist is not None: self.bg_artist.set_clim(vmin, vmax); self.fig.canvas.draw_idle()

    def update_clim(self, val):
        if hasattr(self, "scatter"): self.scatter.set_clim(-val, val); self.fig.canvas.draw_idle()

    def get_export_data(self):
        return self.rhythmic_df, "phase_map_data.csv"

class GroupScatterViewer:
    def __init__(self, fig, ax, group_df):
        self.fig = fig
        self.ax = ax
        self.group_df = group_df
        self.period_hours = self.group_df['Period_Hours'].iloc[0] if not self.group_df.empty else 24
        ax.set_title("Group Phase Distribution")
        if not self.group_df.empty:
            self.scatter = ax.scatter(
                self.group_df['Warped_X'], self.group_df['Warped_Y'],
                c=self.group_df['Relative_Phase_Hours'], cmap=cet.cm.cyclic_mygbm_30_95_c78, s=10, alpha=0.8,
            )
            cbar = fig.colorbar(self.scatter, ax=ax)
            cbar.set_label("Mean Relative Peak Time (hours)")
            ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim)
            self.update_clim(max_range)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

    def update_clim(self, val):
        if hasattr(self, "scatter"): self.scatter.set_clim(-val, val); self.fig.canvas.draw_idle()

    def get_export_data(self):
        return self.group_df, "group_scatter_data.csv"

class GroupAverageMapViewer:
    def __init__(self, fig, ax, group_binned_df, group_scatter_df, grid_res, do_smooth):
        self.fig = fig
        self.ax = ax
        self.group_binned_df = group_binned_df
        self.group_scatter_df = group_scatter_df
        self.period_hours = self.group_scatter_df['Period_Hours'].iloc[0] if not self.group_scatter_df.empty else 24
        ax.set_title("Group Average Phase Map")
        
        if self.group_binned_df.empty:
            ax.text(0.5, 0.5, "No data to display.", ha='center', va='center')
            self.fig.canvas.draw_idle()
            return

        binned_grid = np.full((grid_res, grid_res), np.nan)
        for _, row in self.group_binned_df.iterrows():
            if 0 <= row['Grid_Y_Index'] < grid_res and 0 <= row['Grid_X_Index'] < grid_res:
                binned_grid[int(row['Grid_Y_Index']), int(row['Grid_X_Index'])] = row['Relative_Phase_Hours']

        if do_smooth:
            pass

        x_min, x_max = self.group_scatter_df['Warped_X'].min(), self.group_scatter_df['Warped_X'].max()
        y_min, y_max = self.group_scatter_df['Warped_Y'].min(), self.group_scatter_df['Warped_Y'].max()

        self.im = ax.imshow(binned_grid, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap=cet.cm.cyclic_mygbm_30_95_c78)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(self.im, ax=ax)
        cbar.set_label("Mean Relative Peak Time (hours)")
        ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim)
        self.update_clim(max_range)
        self.fig.canvas.draw_idle()

    def update_clim(self, val):
        if hasattr(self, "im"): self.im.set_clim(-val, val); self.fig.canvas.draw_idle()

    def get_export_data(self):
        return self.group_scatter_df, "group_binned_details_data.csv"


class InterpolatedMapViewer:
    def __init__(self, fig, ax, roi_data, relative_phases,
                 period_hours, grid_resolution, rois=None):
        self.fig = fig
        self.ax = ax
        self.period_hours = period_hours

        ax.set_title("Interpolated Spatiotemporal Phase Map")

        if len(roi_data) < 4:
            ax.text(
                0.5, 0.5,
                "Not enough data points (<4) for interpolation.",
                ha="center", va="center",
            )
            return

        phase_angles_rad = (relative_phases / (period_hours / 2.0)) * pi
        x_comp = np.cos(phase_angles_rad)
        y_comp = np.sin(phase_angles_rad)

        xs = roi_data[:, 0]
        ys = roi_data[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_buf = (x_max - x_min) * 0.05
        y_buf = (y_max - y_min) * 0.05

        grid_x, grid_y = np.mgrid[
            x_min - x_buf:x_max + x_buf:complex(grid_resolution),
            y_min - y_buf:y_max + y_buf:complex(grid_resolution),
        ]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        rbf_x = RBFInterpolator(roi_data, x_comp, kernel="linear", smoothing=1.0)
        rbf_y = RBFInterpolator(roi_data, y_comp, kernel="linear", smoothing=1.0)

        gx = rbf_x(grid_points)
        gy = rbf_y(grid_points)

        angles = arctan2(gy, gx)
        grid_z = (angles / pi) * (period_hours / 2.0)
        grid_z = grid_z.reshape(grid_x.shape)

        # Mask outside ROIs/hull
        if rois:
            final_mask = np.zeros(grid_x.shape, dtype=bool)
            include_paths = [r["path"] for r in rois if r["mode"] == "Include"]
            if include_paths:
                for path in include_paths:
                    final_mask |= path.contains_points(
                        grid_points
                    ).reshape(grid_x.shape)
            else:
                if len(roi_data) > 2:
                    hull = ConvexHull(roi_data)
                    hpath = Path(roi_data[hull.vertices])
                    final_mask = hpath.contains_points(
                        grid_points
                    ).reshape(grid_x.shape)
            for roi in rois:
                if roi["mode"] == "Exclude":
                    final_mask &= ~roi["path"].contains_points(
                        grid_points
                    ).reshape(grid_x.shape)
            grid_z[~final_mask] = np.nan
        elif len(roi_data) > 2:
            hull = ConvexHull(roi_data)
            hpath = Path(roi_data[hull.vertices])
            mask = hpath.contains_points(grid_points).reshape(grid_x.shape)
            grid_z[~mask] = np.nan

        im = ax.imshow(
            grid_z.T,
            origin="upper",
            extent=[
                x_min - x_buf,
                x_max + x_buf,
                y_min - y_buf,
                y_max + y_buf,
            ],
            cmap=cet.cm.cyclic_mygbm_30_95_c78,
            interpolation="bilinear",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04
        )
        cbar.set_label("Relative Peak Time (hours)", fontsize=10)

        ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(
            ax=ax_slider,
            label="Phase Range (+/- hrs)",
            valmin=1.0,
            valmax=max_range,
            valinit=max_range,
        )
        self.range_slider.on_changed(self.update_clim)
        self.update_clim(max_range)

    def update_clim(self, val):
        # assumes image exists
        for im in self.ax.images:
            im.set_clim(-val, val)
        self.fig.canvas.draw_idle()


# ------------------------------------------------------------
# Phase calculation (unchanged)
# ------------------------------------------------------------

def calculate_phases_fft(traces_data, minutes_per_frame,
                         period_min=None, period_max=None):
    intensities = traces_data[:, 1:]
    if intensities.shape[1] < 2: # Need at least 2 time points
        return np.array([]), 0.0, np.array([])

    hours_per_frame = minutes_per_frame / 60.0
    sampling_rate = 1.0 / hours_per_frame

    nyquist = 0.5 * sampling_rate
    cutoff = 1.0 / 40.0 # High-pass filter cutoff in Hz (1 cycle per 40 hours)
    if cutoff >= nyquist:
        # If cutoff is too high for the sampling rate, use a smaller fraction
        cutoff = nyquist / 2.0
        
    b, a = signal.butter(3, cutoff / nyquist, btype="high", analog=False)
    filtered = signal.filtfilt(b, a, intensities, axis=0)

    mean_signal = filtered.mean(axis=1)
    fft_pop = np.fft.fft(mean_signal)
    freqs = np.fft.fftfreq(len(mean_signal), d=hours_per_frame)

    pos_idx = np.where(freqs > 0)
    pos_freqs = freqs[pos_idx]
    pos_fft = fft_pop[pos_idx]
    power_spectrum_pop = np.abs(pos_fft)

    if period_min and period_max:
        f_low, f_high = 1.0 / period_max, 1.0 / period_min
        mask = (pos_freqs >= f_low) & (pos_freqs <= f_high)
        if not np.any(mask):
            raise ValueError("No FFT frequencies found in specified period range.")
        search_power = power_spectrum_pop[mask]
        sub_idx = np.argmax(search_power)
        peak_idx = np.where(mask)[0][sub_idx]
    else:
        peak_idx = np.argmax(power_spectrum_pop)

    dominant_freq = pos_freqs[peak_idx]
    if dominant_freq == 0:
        raise ValueError("Could not determine dominant frequency.")

    period_hours = 1.0 / dominant_freq

    phases = []
    rhythm_snr_scores = []

    for i in range(filtered.shape[1]):
        fft_cell = np.fft.fft(filtered[:, i])
        cell_fft_pos = fft_cell[pos_idx]
        
        # Calculate phase (remains the same)
        phase_angle = np.angle(cell_fft_pos[peak_idx])
        phase_hours = -phase_angle / (2 * np.pi * dominant_freq)
        phases.append(phase_hours)
        
        # --- New Rhythmicity Score: Signal-to-Noise Ratio ---
        power_spectrum_cell = np.abs(cell_fft_pos)
        
        # Define signal band: +/- 1 frequency bin around the peak
        signal_bins = slice(max(0, peak_idx - 1), peak_idx + 2)
        signal_power = np.mean(power_spectrum_cell[signal_bins])
        
        # Define noise band: +/- 10 bins, excluding the signal band
        noise_start = max(0, peak_idx - 10)
        noise_end = peak_idx + 11
        
        # Create a mask to exclude the signal band from the noise calculation
        noise_mask = np.ones(len(power_spectrum_cell), dtype=bool)
        noise_mask[signal_bins] = False
        
        noise_indices = np.arange(len(power_spectrum_cell))[noise_start:noise_end]
        valid_noise_indices = noise_indices[noise_mask[noise_start:noise_end]]
        
        if len(valid_noise_indices) > 0:
            noise_power = np.mean(power_spectrum_cell[valid_noise_indices])
        else:
            noise_power = 1e-9 # Avoid division by zero if spectrum is too narrow

        if noise_power > 0:
            snr = signal_power / noise_power
        else:
            snr = np.inf # Effectively a perfect rhythm
            
        rhythm_snr_scores.append(snr)

    return np.array(phases), period_hours, np.array(rhythm_snr_scores)


# ------------------------------------------------------------
# Analysis worker (QThread)
# ------------------------------------------------------------

class AnalysisWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(float)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)  # success, error_msg

    def __init__(self, input_file, output_basename, args):
        super().__init__()
        self.input_file = input_file
        self.output_basename = output_basename
        self.args = args

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.message.emit(f"Loading data from {self.input_file}...")
            data = skimage.io.imread(self.input_file)
            data = ntc.rescale(data, 0.0, 1.0)

            self.message.emit("Stage 1/4: Detecting features...")
            ims, ids, trees, blob_lists = ntc.process_frames(
                data,
                sigma1=self.args["sigma1"],
                sigma2=self.args["sigma2"],
                blur_sigma=self.args["blur_sigma"],
                max_features=self.args["max_features"],
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.25)

            self.message.emit("Stage 2/4: Building trajectories...")
            graph, subgraphs = ntc.build_trajectories(
                blob_lists,
                trees,
                ids,
                search_range=self.args["search_range"],
                cone_radius_base=self.args["cone_radius_base"],
                cone_radius_multiplier=self.args["cone_radius_multiplier"],
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.5)

            self.message.emit("Stage 3/4: Pruning trajectories...")
            pruned_subgraphs, reverse_ids = ntc.prune_trajectories(
                graph,
                subgraphs,
                ids,
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.7)

            self.message.emit("Stage 4/4: Extracting traces...")
            com, traj, lines = ntc.extract_and_interpolate_data(
                ims,
                pruned_subgraphs,
                reverse_ids,
                min_trajectory_length=self.args["min_trajectory_length"],
                sampling_box_size=self.args["sampling_box_size"],
                sampling_sigma=self.args["sampling_sigma"],
                max_interpolation_distance=self.args["max_interpolation_distance"],
                progress_callback=self.message.emit,
            )

            if len(lines) == 0:
                self.message.emit(
                    "Processing complete, but no valid trajectories were found."
                )
            else:
                self.message.emit(
                    f"Processing complete. Found {len(lines)} valid trajectories."
                )

                # ROI centers (x, y)
                np.savetxt(
                    f"{self.output_basename}_roi.csv",
                    np.column_stack((com[:, 1], com[:, 0])),
                    delimiter=",",
                )
                self.message.emit("Saved center-of-mass data.")

                # Traces: time + intensities
                np.savetxt(
                    f"{self.output_basename}_traces.csv",
                    np.column_stack((lines[0, :, 0], lines[:, :, 1].T)),
                    delimiter=",",
                )
                self.message.emit("Saved intensity traces.")

                # Full trajectories
                np.save(f"{self.output_basename}_trajectories.npy", traj)
                self.message.emit("Saved full trajectory data.")

            self.progress.emit(1.0)
            self.finished.emit(True, "")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.message.emit(f"--- ANALYSIS FAILED ---\nError: {e}")
            self.message.emit(tb)
            self.finished.emit(False, str(e))

class MovieLoaderWorker(QtCore.QObject):
    """Worker to load a movie file in a background thread."""
    finished = QtCore.pyqtSignal(object)  # Emits the loaded numpy array
    error = QtCore.pyqtSignal(str)

    def __init__(self, movie_path):
        super().__init__()
        self.movie_path = movie_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if not self.movie_path or not os.path.exists(self.movie_path):
                raise FileNotFoundError("Movie file not found.")
            data = skimage.io.imread(self.movie_path)
            self.finished.emit(data)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Failed to load movie: {e}\n{tb}")
            
# ------------------------------------------------------------
# Main Window
# ------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    """
    PyQt main window implementing the full original workflow.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuron Analysis Workspace (PyQt)")
        self.resize(1400, 900)

        # Centralized state object
        self.state = AnalysisState()

        self.params = {}
        self.phase_params = {}
        self.filtered_indices = None
        self.rois = None
        self.vmin = None
        self.vmax = None
        self.visualization_widgets = {}

        self.workflow_state = {
            "has_input": False,
            "has_results": False,
            "has_anatomical_roi": False,
            "has_warp": False,
            "has_group_data": False,
        }

        self._analysis_thread = None
        self._analysis_worker = None
        self._movie_loader_thread = None
        self._movie_loader_worker = None

        self._build_ui()
        self._connect_signals()

    # ------------ UI construction ------------

    def export_current_data(self):
        current_tab = self.vis_tabs.currentWidget()
        viewer = self.visualization_widgets.get(current_tab)

        if not viewer or not hasattr(viewer, 'get_export_data'):
            self.log_message("No data export available for the current tab.")
            return

        try:
            df, default_filename = viewer.get_export_data()
            if df is None or df.empty:
                self.log_message("No data to export.")
                return
        except Exception as e:
            self.log_message(f"Error preparing data for export: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return

        base = self.state.output_basename
        if current_tab in [self.group_scatter_tab, self.group_avg_tab]:
            base = "group_analysis" # Use a generic name for group exports
        
        default_filename = f"{os.path.basename(base)}_{default_filename}" if base else default_filename
        suggested_path = os.path.join(os.path.dirname(base) if base else ".", default_filename)

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", suggested_path, "CSV files (*.csv)"
        )

        if not path:
            return
        
        try:
            df.to_csv(path, index=False)
            self.log_message(f"Data exported successfully to {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"Error saving data file: {e}")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter, 1)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.splitter.addWidget(left_widget)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll, 3)
        self.ctrl_container = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl_container)
        scroll.setWidget(self.ctrl_container)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        self.splitter.addWidget(right_widget)
        self.vis_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.vis_tabs)
        log_group = QtWidgets.QGroupBox("Execution Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.progress_bar = QtWidgets.QProgressBar()
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.progress_bar)
        left_layout.addWidget(log_group, 1)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self._build_workflow_section()
        self.mode_stack = QtWidgets.QStackedWidget()
        self._build_mode_section()
        self.ctrl_layout.addWidget(self.mode_stack)
        self._build_mode_panels()

        # --- Create and add the Execution box to the main control layout ---
        exec_box = QtWidgets.QGroupBox("Execution")
        exec_layout = QtWidgets.QVBoxLayout(exec_box)
        self.btn_run_analysis = QtWidgets.QPushButton("Run Full Analysis")
        self.btn_run_analysis.setEnabled(False)
        self.btn_load_results = QtWidgets.QPushButton("Load Existing Results")
        self.btn_load_results.setEnabled(False)
        self.btn_export_data = QtWidgets.QPushButton("Export Current Data...")
        self.btn_export_data.setEnabled(False)
        self.btn_export_plot = QtWidgets.QPushButton("Export Current Plot...")
        self.btn_export_plot.setEnabled(False)
        exec_layout.addWidget(self.btn_run_analysis)
        exec_layout.addWidget(self.btn_load_results)
        exec_layout.addWidget(self.btn_export_data)
        exec_layout.addWidget(self.btn_export_plot)
        self.ctrl_layout.addWidget(exec_box) # Add to self.ctrl_layout
        self.ctrl_layout.addStretch(1) # Add stretch to push it up

        self._build_vis_tabs()

    def _build_workflow_section(self):
        box = QtWidgets.QGroupBox("Workflow")
        layout = QtWidgets.QVBoxLayout(box)
        self.step_labels = {}
        items = [
            ("single", "1. Single Animal Analysis"),
            ("register", "2. Atlas Registration"),
            ("apply_warp", "3. Apply Warp to Data"),
            ("group_view", "4. Group Data Viewer"),
        ]
        for key, text in items:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color: #666666;")
            layout.addWidget(lbl)
            self.step_labels[key] = lbl
        self.ctrl_layout.addWidget(box)

    def _build_mode_section(self):
        box = QtWidgets.QGroupBox("Active Panel")
        layout = QtWidgets.QVBoxLayout(box)
        self.mode_buttons = {}
        self.mode_group = QtWidgets.QButtonGroup(self)
        def add_mode(key, label, enabled=True):
            btn = QtWidgets.QRadioButton(label)
            btn.setEnabled(enabled)
            self.mode_group.addButton(btn)
            self.mode_buttons[key] = btn
            layout.addWidget(btn)
            btn.toggled.connect(
                lambda checked, k=key: checked and self._switch_mode(k)
            )
        add_mode("single", "Single Animal", True)
        add_mode("register", "Atlas Registration", False)
        add_mode("apply_warp", "Apply Warp to Data", False)
        add_mode("group_view", "Group Data Viewer", True)
        self.mode_buttons["single"].setChecked(True)
        self.ctrl_layout.addWidget(box)

    def _build_mode_panels(self):
        self.single_panel = QtWidgets.QWidget()
        self._build_single_panel(self.single_panel)
        self.mode_stack.addWidget(self.single_panel)
        self.register_panel = QtWidgets.QWidget()
        self._build_register_panel(self.register_panel)
        self.mode_stack.addWidget(self.register_panel)
        self.apply_panel = QtWidgets.QWidget()
        self._build_apply_panel(self.apply_panel)
        self.mode_stack.addWidget(self.apply_panel)
        self.group_panel = QtWidgets.QWidget()
        self._build_group_panel(self.group_panel)
        self.mode_stack.addWidget(self.group_panel)

    def _build_single_panel(self, panel):
        layout = QtWidgets.QVBoxLayout(panel)
        io_box = QtWidgets.QGroupBox("File I/O")
        io_layout = QtWidgets.QGridLayout(io_box)
        self.btn_load_movie = QtWidgets.QPushButton("Load Movie...")
        io_layout.addWidget(self.btn_load_movie, 0, 0, 1, 2)
        io_layout.addWidget(QtWidgets.QLabel("Input File:"), 1, 0)
        self.input_file_edit = QtWidgets.QLineEdit()
        self.input_file_edit.setReadOnly(True)
        io_layout.addWidget(self.input_file_edit, 1, 1)
        io_layout.addWidget(QtWidgets.QLabel("Output Basename:"), 2, 0)
        self.output_base_edit = QtWidgets.QLineEdit()
        io_layout.addWidget(self.output_base_edit, 2, 1)
        self.status_traces_label = QtWidgets.QLabel("Traces: —")
        self.status_roi_label = QtWidgets.QLabel("ROI: —")
        self.status_traj_label = QtWidgets.QLabel("Trajectories: —")
        s_layout = QtWidgets.QHBoxLayout()
        s_layout.addWidget(self.status_traces_label)
        s_layout.addWidget(self.status_roi_label)
        s_layout.addWidget(self.status_traj_label)
        io_layout.addLayout(s_layout, 3, 0, 1, 2)
        layout.addWidget(io_box)
        roi_box = QtWidgets.QGroupBox("Region of Interest (ROI)")
        roi_layout = QtWidgets.QHBoxLayout(roi_box)
        self.btn_define_roi = QtWidgets.QPushButton("Define Anatomical ROI...")
        self.btn_define_roi.setEnabled(False)
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI Filter")
        self.btn_clear_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_define_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addWidget(roi_box)
        param_box = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QVBoxLayout(param_box)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_params = QtWidgets.QPushButton("Save Params...")
        self.btn_load_params = QtWidgets.QPushButton("Load Params...")
        btn_row.addWidget(self.btn_save_params)
        btn_row.addWidget(self.btn_load_params)
        param_layout.addLayout(btn_row)
        tabs = QtWidgets.QTabWidget()
        param_layout.addWidget(tabs)
        det_tab = QtWidgets.QWidget()
        det_layout = QtWidgets.QFormLayout(det_tab)
        self._add_param_field(det_layout, "sigma1", 3.0)
        self._add_param_field(det_layout, "sigma2", 20.0)
        self._add_param_field(det_layout, "blur_sigma", 2.0)
        self._add_param_field(det_layout, "max_features", 200)
        tabs.addTab(det_tab, "Detection")
        tr_tab = QtWidgets.QWidget()
        tr_layout = QtWidgets.QFormLayout(tr_tab)
        self._add_param_field(tr_layout, "search_range", 50)
        self._add_param_field(tr_layout, "cone_radius_base", 1.5)
        self._add_param_field(tr_layout, "cone_radius_multiplier", 0.125)
        tabs.addTab(tr_tab, "Tracking")
        fl_tab = QtWidgets.QWidget()
        fl_layout = QtWidgets.QFormLayout(fl_tab)
        self._add_param_field(fl_layout, "min_trajectory_length", 0.08)
        self._add_param_field(fl_layout, "sampling_box_size", 15)
        self._add_param_field(fl_layout, "sampling_sigma", 2.0)
        self._add_param_field(fl_layout, "max_interpolation_distance", 5.0)
        tabs.addTab(fl_tab, "Filtering")
        layout.addWidget(param_box)
        phase_box = QtWidgets.QGroupBox("Phase Map Parameters")
        phase_layout = QtWidgets.QFormLayout(phase_box)
        self.analysis_method_combo = QtWidgets.QComboBox()
        self.analysis_method_combo.addItems(["FFT (SNR)", "Cosinor (p-value)"])
        phase_layout.addRow("Analysis Method:", self.analysis_method_combo)
        self._add_phase_field(phase_layout, "minutes_per_frame", 15.0)
        self.discovered_period_edit = QtWidgets.QLineEdit("N/A")
        self.discovered_period_edit.setReadOnly(True)
        phase_layout.addRow("Discovered Period (hrs):", self.discovered_period_edit)
        self._add_phase_field(phase_layout, "period_min", 22.0)
        self._add_phase_field(phase_layout, "period_max", 28.0)
        self._add_phase_field(phase_layout, "grid_resolution", 100, int)
        _, self.rhythm_threshold_label = self._add_phase_field(phase_layout, "rhythm_threshold", 2.0)
        rsquared_le, rsquared_label = self._add_phase_field(phase_layout, "r_squared_threshold", 0.3)
        self.rsquared_widgets = (rsquared_label, rsquared_le)
        self.emphasize_rhythm_check = QtWidgets.QCheckBox("Emphasize rhythmic cells in all plots")
        phase_layout.addRow(self.emphasize_rhythm_check)
        self.btn_regen_phase = QtWidgets.QPushButton("Update Plots")
        self.btn_regen_phase.setEnabled(False)
        phase_layout.addRow(self.btn_regen_phase)
        layout.addWidget(phase_box)
        layout.addStretch(1)
        self._on_analysis_method_changed(0)

    def _add_param_field(self, layout, name, default):
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(QtWidgets.QLabel(f"{name}:"), le)
        self.params[name] = (le, type(default))

    def _add_phase_field(self, layout, name, default, typ=float):
        label = QtWidgets.QLabel(f"{name}:")
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(label, le)
        self.phase_params[name] = (le, typ)
        return le, label # Return both the line edit and the label

    def _build_register_panel(self, panel):
        layout = QtWidgets.QVBoxLayout(panel)
        box = QtWidgets.QGroupBox("Atlas Registration Setup")
        b = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.btn_select_atlas = QtWidgets.QPushButton("Select Atlas...")
        self.atlas_path_edit = QtWidgets.QLineEdit()
        self.atlas_path_edit.setReadOnly(True)
        row.addWidget(self.btn_select_atlas)
        row.addWidget(self.atlas_path_edit)
        b.addLayout(row)
        self.target_list = QtWidgets.QListWidget()
        b.addWidget(self.target_list)
        row_btn = QtWidgets.QHBoxLayout()
        self.btn_add_targets = QtWidgets.QPushButton("Add Target(s)...")
        self.btn_remove_target = QtWidgets.QPushButton("Remove Selected")
        row_btn.addWidget(self.btn_add_targets)
        row_btn.addWidget(self.btn_remove_target)
        b.addLayout(row_btn)
        self.btn_begin_reg = QtWidgets.QPushButton("Begin Registration...")
        self.btn_begin_reg.setEnabled(False)
        b.addWidget(self.btn_begin_reg)
        layout.addWidget(box)
        layout.addStretch(1)

    def _build_apply_panel(self, panel):
        layout = QtWidgets.QVBoxLayout(panel)
        box = QtWidgets.QGroupBox("Apply Warp Setup")
        b = QtWidgets.QVBoxLayout(box)
        self.warp_list = QtWidgets.QListWidget()
        self.warp_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        b.addWidget(self.warp_list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add_warp = QtWidgets.QPushButton("Add Warp Parameter File(s)...")
        self.btn_remove_warp = QtWidgets.QPushButton("Remove Selected")
        row.addWidget(self.btn_add_warp)
        row.addWidget(self.btn_remove_warp)
        b.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        self.btn_inspect_warp = QtWidgets.QPushButton("Inspect Selected Warp...")
        self.btn_inspect_warp.setEnabled(False)
        self.btn_apply_warp = QtWidgets.QPushButton("Apply All Warp(s)")
        self.btn_apply_warp.setEnabled(False)
        row2.addWidget(self.btn_inspect_warp)
        row2.addWidget(self.btn_apply_warp)
        b.addLayout(row2)
        layout.addWidget(box)
        layout.addStretch(1)

    def _build_group_panel(self, panel):
        layout = QtWidgets.QVBoxLayout(panel)
        box = QtWidgets.QGroupBox("Group Data Setup")
        b = QtWidgets.QVBoxLayout(box)
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        b.addWidget(self.group_list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton("Add Warped ROI File(s)...")
        self.btn_remove_group = QtWidgets.QPushButton("Remove Selected")
        row.addWidget(self.btn_add_group)
        row.addWidget(self.btn_remove_group)
        b.addLayout(row)
        form = QtWidgets.QFormLayout()
        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth to fill empty bins")
        form.addRow("Grid Resolution:", self.group_grid_res_edit)
        form.addRow(self.group_smooth_check)
        b.addLayout(form)
        self.btn_view_group = QtWidgets.QPushButton("Generate Group Visualizations")
        self.btn_view_group.setEnabled(False)
        b.addWidget(self.btn_view_group)
        layout.addWidget(box)
        layout.addStretch(1)

    def _build_vis_tabs(self):
        self.heatmap_tab = QtWidgets.QWidget()
        self.com_tab = QtWidgets.QWidget()
        self.traj_tab = QtWidgets.QWidget()
        self.phase_tab = QtWidgets.QWidget()
        self.interp_tab = QtWidgets.QWidget()
        self.group_scatter_tab = QtWidgets.QWidget()
        self.group_avg_tab = QtWidgets.QWidget()
        tabs_to_add = [
            (self.heatmap_tab, "Heatmap"),
            (self.com_tab, "Center of Mass"),
            (self.traj_tab, "Trajectory Inspector"),
            (self.phase_tab, "Phase Map"),
            (self.interp_tab, "Interpolated Map"),
            (self.group_scatter_tab, "Group Scatter"),
            (self.group_avg_tab, "Group Average Map"),
        ]
        for tab, name in tabs_to_add:
            self.vis_tabs.addTab(tab, name)
            layout = QtWidgets.QVBoxLayout(tab)
            label = QtWidgets.QLabel(
                f"{name} will appear here after analysis."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
        for i in range(self.vis_tabs.count()):
            self.vis_tabs.setTabEnabled(i, False)

    # ------------ Signals ------------

    def _connect_signals(self):
        self.btn_load_movie.clicked.connect(self.load_movie)
        self.output_base_edit.textChanged.connect(self.update_output_basename)
        self.btn_define_roi.clicked.connect(self.open_roi_tool)
        self.btn_clear_roi.clicked.connect(self.clear_roi_filter)
        self.btn_save_params.clicked.connect(self.save_parameters)
        self.btn_load_params.clicked.connect(self.load_parameters)
        self.btn_run_analysis.clicked.connect(self.start_analysis)
        self.btn_load_results.clicked.connect(self.load_results)
        self.btn_export_plot.clicked.connect(self.export_current_plot)
        self.btn_export_data.clicked.connect(self.export_current_data)
        self.btn_regen_phase.clicked.connect(self.regenerate_phase_maps)
        self.emphasize_rhythm_check.stateChanged.connect(self.regenerate_phase_maps)
        
        # Connect the dropdown to the UI handler
        self.analysis_method_combo.currentIndexChanged.connect(self._on_analysis_method_changed)

        self.btn_select_atlas.clicked.connect(self.select_atlas)
        self.btn_add_targets.clicked.connect(self.add_targets)
        self.btn_remove_target.clicked.connect(self.remove_target)
        self.btn_begin_reg.clicked.connect(self.begin_registration)
        self.btn_add_warp.clicked.connect(self.add_warp_files)
        self.btn_remove_warp.clicked.connect(self.remove_warp_file)
        self.warp_list.itemSelectionChanged.connect(self.check_apply_warp_buttons_state)
        self.btn_apply_warp.clicked.connect(self.apply_warps)
        self.btn_inspect_warp.clicked.connect(self.inspect_warp)
        self.btn_add_group.clicked.connect(self.add_group_files)
        self.btn_remove_group.clicked.connect(self.remove_group_file)
        self.btn_view_group.clicked.connect(self.generate_group_visualizations)

    # ------------ Mode & workflow helpers ------------

    def _reset_state(self):
        self.log_message("Resetting workspace for new analysis...")
        self.state.reset()
        self.filtered_indices = None
        self.rois = None
        self.vmin = None
        self.vmax = None
        self.visualization_widgets.clear()
        for i in range(self.vis_tabs.count()):
            tab = self.vis_tabs.widget(i)
            layout = tab.layout()
            if layout is not None:
                clear_layout(layout)
            else:
                layout = QtWidgets.QVBoxLayout(tab)
            label = QtWidgets.QLabel(
                f"{self.vis_tabs.tabText(i)} will appear here after analysis."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
            self.vis_tabs.setTabEnabled(i, False)
        self.btn_define_roi.setEnabled(False)
        self.btn_clear_roi.setEnabled(False)
        self.btn_regen_phase.setEnabled(False)
        self.btn_export_plot.setEnabled(False)
        self.btn_run_analysis.setEnabled(False)
        self.btn_load_results.setEnabled(False)
        self.status_traces_label.setText("Traces: —")
        self.status_roi_label.setText("ROI: —")
        self.status_traj_label.setText("Trajectories: —")
        self.progress_bar.setValue(0)

    def log_message(self, text: str):
        self.log_text.appendPlainText(text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _switch_mode(self, mode_name: str):
        mode_map = {
            "single": 0, "register": 1, "apply_warp": 2, "group_view": 3,
        }
        idx = mode_map.get(mode_name)
        if idx is not None:
            self.mode_stack.setCurrentIndex(idx)
            self.log_message(f"Switched to '{mode_name}' panel.")
        else:
            self.log_message(f"Error: Unknown mode '{mode_name}'")

    def _set_mode_enabled(self, mode_key, enabled: bool):
        btn = self.mode_buttons.get(mode_key)
        if btn:
            btn.setEnabled(enabled)

    def _mark_step_ready(self, mode_key):
        lbl = self.step_labels.get(mode_key)
        if lbl:
            lbl.setStyleSheet("color: #107010;")

    # ------------ Single-animal flow ------------

    def load_movie(self):
        self._reset_state()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Movie", "", "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if not path:
            self.input_file_edit.clear()
            self.output_base_edit.clear()
            return
        self.state.input_movie_path = path
        self.input_file_edit.setText(path)
        base, _ = os.path.splitext(path)
        self.state.output_basename = base
        self.output_base_edit.setText(base)
        self.workflow_state["has_input"] = True
        self.log_message(f"Loaded movie: {os.path.basename(path)}")
        self.update_workflow_from_files()

    def update_output_basename(self, text):
        self.state.output_basename = text
        self.update_workflow_from_files()

    def update_workflow_from_files(self):
        basename = self.state.output_basename
        if not basename:
            return
        input_present = bool(self.state.input_movie_path)
        if input_present:
            self.workflow_state["has_input"] = True
            self._mark_step_ready("single")
            self.btn_run_analysis.setEnabled(True)
        traces_path = f"{basename}_traces.csv"
        roi_path = f"{basename}_roi.csv"
        traj_path = f"{basename}_trajectories.npy"
        has_traces = os.path.exists(traces_path)
        has_roi = os.path.exists(roi_path)
        has_traj = os.path.exists(traj_path)
        self.status_traces_label.setText(f"Traces: {'found' if has_traces else 'missing'}")
        self.status_roi_label.setText(f"ROI: {'found' if has_roi else 'missing'}")
        self.status_traj_label.setText(f"Trajectories: {'found' if has_traj else 'missing'}")
        has_results = has_traces and has_roi and has_traj
        self.workflow_state["has_results"] = has_results
        self.btn_load_results.setEnabled(has_results)
        if has_results:
            self._mark_step_ready("single")
            self._set_mode_enabled("register", True)
            self.btn_define_roi.setEnabled(True)
            self.btn_regen_phase.setEnabled(True)
            self.btn_export_plot.setEnabled(True)
        anatomical_roi_path = f"{basename}_anatomical_roi.json"
        has_anatomical_roi = os.path.exists(anatomical_roi_path)
        self.workflow_state["has_anatomical_roi"] = has_anatomical_roi
        if has_anatomical_roi:
            self._mark_step_ready("register")
            self._set_mode_enabled("register", True)
            self._set_mode_enabled("apply_warp", True)
        dirname = os.path.dirname(basename) or "."
        warp_files = [f for f in os.listdir(dirname) if f.endswith("_warp_parameters.json")]
        warped_roi_files = [f for f in os.listdir(dirname) if f.endswith("_roi_warped.csv")]
        has_warp = len(warp_files) > 0
        has_group = len(warped_roi_files) > 0
        self.workflow_state["has_warp"] = has_warp
        self.workflow_state["has_group_data"] = has_group
        if has_warp:
            self._mark_step_ready("apply_warp")
            self._set_mode_enabled("apply_warp", True)
        if has_group:
            self._mark_step_ready("group_view")
            self._set_mode_enabled("group_view", True)
            self.btn_view_group.setEnabled(True)

    def start_analysis(self):
        if not self.state.input_movie_path or not self.state.output_basename:
            self.log_message("Error: input file and output basename required.")
            return
        try:
            args = {name: t(le.text()) for name, (le, t) in self.params.items()}
        except ValueError as e:
            self.log_message(f"Error in analysis parameters: {e}")
            return
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.btn_run_analysis.setEnabled(False)
        self.btn_load_results.setEnabled(False)
        self._analysis_worker = AnalysisWorker(self.state.input_movie_path, self.state.output_basename, args)
        self._analysis_thread = QtCore.QThread(self)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.message.connect(self.log_message)
        self._analysis_worker.progress.connect(lambda v: self.progress_bar.setValue(int(v * 100)))
        def done(success, msg):
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._analysis_worker = None
            self._analysis_thread = None
            if success:
                self.log_message("Analysis finished. Loading results...")
                self.load_results()
            else:
                self.log_message("Analysis failed.")
            self.update_workflow_from_files()
        self._analysis_worker.finished.connect(done)
        self._analysis_thread.start()

    def load_results(self):
        basename = self.state.output_basename
        if not basename:
            self.log_message("Error: Output basename not set.")
            return

        # Prevent starting a new load if one is already running
        if self._movie_loader_thread is not None and self._movie_loader_thread.isRunning():
            self.log_message("Movie is already being loaded.")
            return

        try:
            # Load small/fast files on the main thread
            self.state.unfiltered_data["traces"] = np.loadtxt(f"{basename}_traces.csv", delimiter=",")
            self.state.unfiltered_data["roi"] = np.loadtxt(f"{basename}_roi.csv", delimiter=",")
            self.state.unfiltered_data["trajectories"] = np.load(f"{basename}_trajectories.npy")
            self.log_message("Loaded ROI and trace data.")
        except Exception as e:
            self.log_message(f"Error loading result files: {e}")
            return

        # Disable UI to prevent user interaction during load
        self.btn_load_movie.setEnabled(False)
        self.btn_load_results.setEnabled(False)
        self.log_message("Loading movie in background... The UI will remain responsive.")

        # Setup and start the background thread for the slow I/O operation
        self._movie_loader_worker = MovieLoaderWorker(self.state.input_movie_path)
        self._movie_loader_thread = QtCore.QThread(self)
        self._movie_loader_worker.moveToThread(self._movie_loader_thread)

        self._movie_loader_thread.started.connect(self._movie_loader_worker.run)
        self._movie_loader_worker.finished.connect(self._on_movie_loaded)
        self._movie_loader_worker.error.connect(self._on_movie_load_error)
        
        # Clean up the thread when it's done
        self._movie_loader_worker.finished.connect(self._movie_loader_thread.quit)
        self._movie_loader_thread.finished.connect(self._movie_loader_thread.deleteLater)
        self._movie_loader_worker.finished.connect(self._movie_loader_worker.deleteLater)

        self._movie_loader_thread.start()
    
    @QtCore.pyqtSlot(object)
    def _on_movie_loaded(self, movie_data):
        """Receives the loaded movie data from the worker thread."""
        self.log_message("Movie loaded successfully.")
        
        # Store the entire movie stack for the trajectory inspector
        self.state.unfiltered_data["movie"] = movie_data
        # Also keep a single background frame for other, static plots
        self.state.unfiltered_data["background"] = movie_data[len(movie_data) // 2]
        
        # Re-enable UI
        self.btn_load_movie.setEnabled(True)
        self.btn_load_results.setEnabled(True)

        # Now that all data is loaded, proceed with visualization
        self.apply_roi_filter(None, None)

        # Clear thread handles
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    @QtCore.pyqtSlot(str)
    def _on_movie_load_error(self, error_message):
        """Receives an error message from the worker thread."""
        self.log_message(f"--- MOVIE LOAD FAILED ---\n{error_message}")
        
        # Re-enable UI
        self.btn_load_movie.setEnabled(True)
        self.btn_load_results.setEnabled(True)

        # Clean up the failed thread
        if self._movie_loader_thread:
            self._movie_loader_thread.quit()
            self._movie_loader_thread.wait()
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    def open_roi_tool(self):
        if "background" not in self.state.unfiltered_data:
            self.log_message("Error: Load data before defining an ROI.")
            return
        if not self.state.output_basename:
            self.log_message("Error: Output Basename required.")
            return
        dlg = ROIDrawerDialog(
            self,
            self.state.unfiltered_data["background"],
            self.state.unfiltered_data["roi"],
            self.state.output_basename,
            self.apply_roi_filter,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        dlg.exec_()
        self.update_workflow_from_files()

    def apply_roi_filter(self, indices, rois):
        self.filtered_indices = indices
        self.rois = rois
        if indices is None:
            self.state.loaded_data = dict(self.state.unfiltered_data)
            self.log_message("ROI filter cleared. Showing all data.")
            self.btn_clear_roi.setEnabled(False)
        else:
            self.state.loaded_data["roi"] = self.state.unfiltered_data["roi"][indices]
            self.state.loaded_data["trajectories"] = self.state.unfiltered_data["trajectories"][indices]
            trace_indices = np.concatenate(([0], indices + 1))
            self.state.loaded_data["traces"] = self.state.unfiltered_data["traces"][:, trace_indices]
            self.log_message(f"ROI filter applied. {len(indices)} cells selected.")
            self.btn_clear_roi.setEnabled(True)
        self.populate_visualizations()

    def clear_roi_filter(self):
        self.apply_roi_filter(None, None)

    def save_parameters(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON files (*.json)")
        if not path: return
        data = {name: le.text() for name, (le, _) in self.params.items()}
        with open(path, "w") as f: json.dump(data, f, indent=4)
        self.log_message(f"Parameters saved to {os.path.basename(path)}")

    def load_parameters(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON files (*.json)")
        if not path: return
        with open(path, "r") as f: data = json.load(f)
        for name, value in data.items():
            if name in self.params: self.params[name][0].setText(str(value))
        self.log_message(f"Parameters loaded from {os.path.basename(path)}")

    def export_current_plot(self):
        widget = self.vis_tabs.currentWidget()
        viewer = self.visualization_widgets.get(widget)
        if not viewer or not hasattr(viewer, "fig"):
            self.log_message("No active figure to export.")
            return
        fig = viewer.fig
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)")
        if not path: return
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            self.log_message(f"Plot saved to {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"Error saving plot: {e}")

    # ------------ Visualization population ------------

    def on_roi_selected(self, original_index):
        """Callback for when a point is clicked in a spatial plot."""
        self.log_message(f"ROI {original_index + 1} selected.")
        
        # Update the trajectory inspector
        traj_viewer = self.visualization_widgets.get(self.traj_tab)
        if traj_viewer:
            traj_viewer.set_trajectory(original_index)
            self.vis_tabs.setCurrentWidget(self.traj_tab)

        # Highlight the point in other relevant viewers
        com_viewer = self.visualization_widgets.get(self.com_tab)
        if com_viewer:
            com_viewer.highlight_point(original_index)
        
        # Update the heatmap's selected trace plot
        heatmap_viewer = self.visualization_widgets.get(self.heatmap_tab)
        if heatmap_viewer:
            heatmap_viewer.update_selected_trace(original_index)

        phase_viewer = self.visualization_widgets.get(self.phase_tab)
        if phase_viewer:
            try:
                rhythm = calculate_phases_fft(self.state.loaded_data["traces"], minutes_per_frame=float(self.phase_params["minutes_per_frame"][0].text()))[2]
                thr = float(self.phase_params["rhythm_threshold"][0].text())
                rhythmic_indices_relative = np.where(rhythm >= thr)[0]

                if self.filtered_indices is not None:
                    rhythmic_indices_original = self.filtered_indices[rhythmic_indices_relative]
                else:
                    rhythmic_indices_original = rhythmic_indices_relative
                
                match = np.where(rhythmic_indices_original == original_index)[0]
                if len(match) > 0:
                    phase_map_index = match[0]
                    phase_viewer.highlight_point(phase_map_index)
                else:
                    phase_viewer.highlight_point(None) 
            except Exception:
                pass

    def on_contrast_change(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        
        # Look up viewers by their widget object key, not by string
        traj_viewer = self.visualization_widgets.get(self.traj_tab)
        if traj_viewer:
            traj_viewer.update_contrast(vmin, vmax)
            
        phase_viewer = self.visualization_widgets.get(self.phase_tab)
        if phase_viewer:
            phase_viewer.update_contrast(vmin, vmax)

    def populate_visualizations(self):
        if not self.state.loaded_data or "background" not in self.state.unfiltered_data:
            return
        self.log_message("Generating interactive plots...")
        bg = self.state.unfiltered_data["background"]
        movie = self.state.unfiltered_data.get("movie")
        if movie is None:
            self.log_message("Error: Full movie stack not found in state.")
            return

        self.vmin, self.vmax = float(bg.min()), float(bg.max())
        single_animal_tabs = [self.heatmap_tab, self.com_tab, self.traj_tab, self.phase_tab, self.interp_tab]
        group_tabs = [self.group_scatter_tab, self.group_avg_tab]
        for tab in single_animal_tabs: self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(tab), True)
        for tab in group_tabs: self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(tab), False)
        
        try:
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = self._calculate_rhythms()
        except Exception as e:
            self.log_message(f"Could not calculate rhythms: {e}")
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = None, None, None, None, True

        is_emphasized = self.emphasize_rhythm_check.isChecked()
        
        fig_h, _ = add_mpl_to_tab(self.heatmap_tab)
        viewer_h = HeatmapViewer(fig_h, self.state.loaded_data, self.filtered_indices, phases, sort_scores, is_emphasized, rhythm_sort_desc)
        self.visualization_widgets[self.heatmap_tab] = viewer_h
        
        fig_c, _ = add_mpl_to_tab(self.com_tab)
        viewer_c = ContrastViewer(fig_c, fig_c.add_subplot(111), bg, self.state.loaded_data["roi"], self.on_contrast_change, self.on_roi_selected)
        self.visualization_widgets[self.com_tab] = viewer_c
        
        fig_t, _ = add_mpl_to_tab(self.traj_tab)
        viewer_t = TrajectoryInspector(fig_t, fig_t.add_subplot(111), self.state.loaded_data["trajectories"], movie)
        self.visualization_widgets[self.traj_tab] = viewer_t
        
        self.regenerate_phase_maps()
        self.btn_export_plot.setEnabled(True)
        self.btn_export_data.setEnabled(True)

    def _calculate_rhythms(self):
        method = self.analysis_method_combo.currentText()
        self.log_message(f"Calculating rhythms using {method} method...")
        phase_args = {name: t(w.text()) for name, (w, t) in self.phase_params.items() if w.text() and name not in ("grid_resolution", "rhythm_threshold", "r_squared_threshold")}
        if not phase_args.get("minutes_per_frame"): raise ValueError("Minutes per frame is required.")
        
        _, discovered_period, _ = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
        self.discovered_period_edit.setText(f"{discovered_period:.2f}")

        if "FFT" in method:
            phases, period, snr_scores = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
            return phases, period, snr_scores, snr_scores, True
        elif "Cosinor" in method:
            traces = self.state.loaded_data["traces"]
            time_points_hours = traces[:, 0] * (phase_args["minutes_per_frame"] / 60.0)
            phases, p_values, r_squareds = [], [], []
            for i in range(1, traces.shape[1]):
                intensity = traces[:, i]
                result = csn.cosinor_analysis(intensity, time_points_hours, period=discovered_period)
                phases.append(result['acrophase'])
                p_values.append(result['p_value'])
                r_squareds.append(result['r_squared'])
            return np.array(phases), discovered_period, np.array(r_squareds), np.array(p_values), True
        return None, None, None, None, True

    @QtCore.pyqtSlot(int)
    def _on_analysis_method_changed(self, index):
        """Updates the UI when the analysis method dropdown changes."""
        method = self.analysis_method_combo.currentText()
        thresh_edit = self.phase_params["rhythm_threshold"][0]

        if "FFT" in method:
            self.rhythm_threshold_label.setText("Rhythm SNR Threshold (>=):")
            thresh_edit.setText("2.0")
            if hasattr(self, 'rsquared_widgets'):
                for widget in self.rsquared_widgets:
                    widget.hide()
        elif "Cosinor" in method:
            self.rhythm_threshold_label.setText("Rhythm p-value (<=):")
            thresh_edit.setText("0.05")
            if hasattr(self, 'rsquared_widgets'):
                for widget in self.rsquared_widgets:
                    widget.show()

    def regenerate_phase_maps(self):
        if not self.state.loaded_data or "traces" not in self.state.loaded_data:
            if self.emphasize_rhythm_check.isChecked(): self.log_message("Load data before enabling rhythm emphasis.")
            return
        self.log_message("Updating plots based on phase parameters...")
        for tab in (self.phase_tab, self.interp_tab): clear_layout(tab.layout())
        try:
            phases, period, sort_scores, filter_scores, sort_desc = self._calculate_rhythms()
            if phases is None: raise ValueError("Rhythm calculation failed.")
            method = self.analysis_method_combo.currentText()
            thresh = float(self.phase_params["rhythm_threshold"][0].text())
            if "Cosinor" in method:
                r_thresh = float(self.phase_params["r_squared_threshold"][0].text())
                rhythm_mask = (filter_scores <= thresh) & (sort_scores >= r_thresh)
                self.log_message(f"Applying Cosinor filter: p <= {thresh} AND R² >= {r_thresh}")
            else:
                rhythm_mask = filter_scores >= thresh
                self.log_message(f"Applying FFT filter: SNR >= {thresh}")
            is_emphasized = self.emphasize_rhythm_check.isChecked()
            heatmap_viewer = self.visualization_widgets.get(self.heatmap_tab)
            if heatmap_viewer:
                heatmap_viewer.update_phase_data(phases, sort_scores, rhythm_mask, sort_desc)
                heatmap_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)
            com_viewer = self.visualization_widgets.get(self.com_tab)
            if com_viewer:
                com_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)
            rhythmic_indices_relative = np.where(rhythm_mask)[0]
            self.log_message(f"{len(rhythmic_indices_relative)} cells pass rhythmicity threshold(s).")
            if self.filtered_indices is not None:
                rhythmic_indices_original = self.filtered_indices[rhythmic_indices_relative]
            else:
                rhythmic_indices_original = rhythmic_indices_relative
            if len(rhythmic_indices_relative) == 0:
                for t in [self.phase_tab, self.interp_tab]:
                    fig, canvas = add_mpl_to_tab(t)
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, "No cells passed the rhythmicity filter.", ha='center', va='center')
                    canvas.draw()
                self.log_message("No cells passed rhythmicity filter.")
                # Create an empty dataframe for the viewer to avoid crashing on export
                empty_df = pd.DataFrame()
                viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], empty_df, None, vmin=self.vmin, vmax=self.vmax)
                self.visualization_widgets[self.phase_tab] = viewer_p
                return

            ph_rad = (phases[rhythmic_indices_relative] % period) * (2 * np.pi / period)
            mean_rad = circmean(ph_rad)
            mean_h = mean_rad * (period / (2 * np.pi))
            rel_phases = (phases[rhythmic_indices_relative] - mean_h + period / 2) % period - period / 2
            
            # --- DataFrame Creation with Dynamic Column Names ---
            df_data = {
                'Original_ROI_Index': rhythmic_indices_original + 1,
                'X_Position': self.state.loaded_data['roi'][rhythmic_indices_relative, 0],
                'Y_Position': self.state.loaded_data['roi'][rhythmic_indices_relative, 1],
                'Phase_Hours': phases[rhythmic_indices_relative],
                'Relative_Phase_Hours': rel_phases,
                'Period_Hours': period
            }
            if "Cosinor" in method:
                df_data['R_Squared'] = sort_scores[rhythmic_indices_relative]
                df_data['P_Value'] = filter_scores[rhythmic_indices_relative]
            else: # FFT
                df_data['SNR'] = sort_scores[rhythmic_indices_relative]
            
            rhythmic_df = pd.DataFrame(df_data)
            
            def phase_map_callback(selected_phase_index):
                original_index = rhythmic_df['Original_ROI_Index'].iloc[selected_phase_index] - 1
                self.on_roi_selected(original_index)
            fig_p, _ = add_mpl_to_tab(self.phase_tab)
            viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], rhythmic_df, phase_map_callback, vmin=self.vmin, vmax=self.vmax)
            self.visualization_widgets[self.phase_tab] = viewer_p
            grid_res = int(self.phase_params["grid_resolution"][0].text())
            fig_i, _ = add_mpl_to_tab(self.interp_tab)
            viewer_i = InterpolatedMapViewer(fig_i, fig_i.add_subplot(111), rhythmic_df[['X_Position', 'Y_Position']].values, rhythmic_df['Relative_Phase_Hours'].values, period, grid_res, rois=self.rois)
            self.visualization_widgets[self.interp_tab] = viewer_i
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.phase_tab), True)
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.interp_tab), True)
            self.btn_regen_phase.setEnabled(True)
            self.log_message("Phase-based plots updated.")
        except Exception as e:
            self.log_message(f"Could not update plots: {e}")
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.phase_tab), False)
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.interp_tab), False)

    # ------------ Registration panel actions ------------

    def select_atlas(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Atlas ROI File", "", "Anatomical ROI files (*_anatomical_roi.json)")
        if not path: return
        self.state.atlas_roi_path = path
        self.atlas_path_edit.setText(path)
        self._update_reg_button_state()

    def add_targets(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Target ROI Files", "", "Anatomical ROI files (*_anatomical_roi.json)")
        for f in files:
            if f not in self.state.target_roi_paths:
                self.state.target_roi_paths.append(f)
                self.target_list.addItem(f)
        self._update_reg_button_state()

    def remove_target(self):
        for item in self.target_list.selectedItems():
            row = self.target_list.row(item)
            path = self.target_list.item(row).text()
            if path in self.state.target_roi_paths:
                self.state.target_roi_paths.remove(path)
            self.target_list.takeItem(row)
        self._update_reg_button_state()

    def _update_reg_button_state(self):
        has_atlas = bool(self.state.atlas_roi_path)
        has_targets = len(self.state.target_roi_paths) > 0
        self.btn_begin_reg.setEnabled(has_atlas and has_targets)

    def begin_registration(self):
        if not self.state.atlas_roi_path or not self.state.target_roi_paths: return
        dlg = RegistrationWindow(self, self.state, self.log_message)
        dlg.exec_()
        self.update_workflow_from_files()

    # ------------ Apply warp panel actions ------------

    def add_warp_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Warp Parameter Files", "", "Warp Parameters (*_warp_parameters.json)")
        for f in files:
            if f not in self.state.warp_param_paths:
                self.state.warp_param_paths.append(f)
                self.warp_list.addItem(f)
        self.check_apply_warp_buttons_state()

    def remove_warp_file(self):
        for item in self.warp_list.selectedItems():
            row = self.warp_list.row(item)
            path = self.warp_list.item(row).text()
            if path in self.state.warp_param_paths:
                self.state.warp_param_paths.remove(path)
            self.warp_list.takeItem(row)
        self.check_apply_warp_buttons_state()

    def check_apply_warp_buttons_state(self):
        total = len(self.state.warp_param_paths)
        selected = len(self.warp_list.selectedItems())
        self.btn_apply_warp.setEnabled(total > 0)
        self.btn_inspect_warp.setEnabled(selected == 1)

    def apply_warps(self):
        if not self.state.warp_param_paths:
            self.log_message("No warp files selected.")
            return
        self.log_message(f"Applying {len(self.state.warp_param_paths)} warp(s)...")
        for warp_file in self.state.warp_param_paths:
            try:
                with open(warp_file, "r") as f: warp = json.load(f)
                roi_file = warp_file.replace("_warp_parameters.json", "_roi_filtered.csv")
                if not os.path.exists(roi_file):
                    self.log_message(f"Missing _roi_filtered.csv for {os.path.basename(warp_file)}")
                    continue
                roi_pts = np.loadtxt(roi_file, delimiter=",")
                from skimage.transform import ThinPlateSplineTransform
                tps = ThinPlateSplineTransform()
                tps.estimate(np.array(warp["dest_landmarks_norm"]), np.array(warp["source_landmarks_norm"]))
                dc = np.array(warp["dest_centroid"])
                ds = warp["dest_scale"]
                sc = np.array(warp["source_centroid"])
                ss = warp["source_scale"]
                pts_norm = (roi_pts - dc) / ds
                warped_norm = tps(pts_norm)
                warped_pts = warped_norm * ss + sc
                out = roi_file.replace("_roi_filtered.csv", "_roi_warped.csv")
                np.savetxt(out, warped_pts, delimiter=",")
                self.log_message(f"Created {os.path.basename(out)}")
            except Exception as e:
                self.log_message(f"Failed to process {os.path.basename(warp_file)}: {e}")
        self.log_message("Warp application complete.")
        self.update_workflow_from_files()

    def inspect_warp(self):
        items = self.warp_list.selectedItems()
        if not items:
            self.log_message("Select a warp file to inspect.")
            return
        warp_file = items[0].text()
        try:
            if not self.state.atlas_roi_path or not os.path.exists(self.state.atlas_roi_path):
                # Attempt to find a logical atlas file if not set
                base_dir = os.path.dirname(warp_file)
                potential_atlas = os.path.join(base_dir, "atlas_anatomical_roi.json") # A guess
                if os.path.exists(potential_atlas):
                    self.state.atlas_roi_path = potential_atlas
                    self.atlas_path_edit.setText(potential_atlas)
                    self.log_message(f"Auto-detected atlas: {os.path.basename(potential_atlas)}")
                else:
                    raise ValueError("Atlas file not set. Please select one in the 'Atlas Registration' panel.")
            
            target_roi_filtered = warp_file.replace("_warp_parameters.json", "_roi_filtered.csv")
            target_anatomical = warp_file.replace("_warp_parameters.json", "_anatomical_roi.json")
            if not os.path.exists(target_roi_filtered): raise FileNotFoundError(f"Missing _roi_filtered.csv for {os.path.basename(warp_file)}")
            if not os.path.exists(target_anatomical): raise FileNotFoundError(f"Missing _anatomical_roi.json for {os.path.basename(warp_file)}")
            
            with open(warp_file, "r") as f: warp = json.load(f)
            orig_pts = np.loadtxt(target_roi_filtered, delimiter=",")
            from skimage.transform import ThinPlateSplineTransform
            tps = ThinPlateSplineTransform()
            tps.estimate(np.array(warp["dest_landmarks_norm"]), np.array(warp["source_landmarks_norm"]))
            dc = np.array(warp["dest_centroid"])
            ds = warp["dest_scale"]
            sc = np.array(warp["source_centroid"])
            ss = warp["source_scale"]
            pts_norm = (orig_pts - dc) / ds
            warped_norm = tps(pts_norm)
            warped_pts = warped_norm * ss + sc
            
            dlg = WarpInspectorWindow(
                self, self.state.atlas_roi_path, target_anatomical,
                orig_pts, warped_pts, pts_norm, warped_norm, warp,
                title=os.path.basename(target_roi_filtered)
            )
            dlg.exec_()
        except Exception as e:
            self.log_message(f"Error during warp inspection: {e}")

    # ------------ Group panel actions ------------

    def add_group_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Warped ROI Files", "", "Warped ROI files (*_roi_warped.csv)")
        for f in files:
            if f not in self.state.group_data_paths:
                self.state.group_data_paths.append(f)
                self.group_list.addItem(f)
        self._update_group_view_button()

    def remove_group_file(self):
        for item in self.group_list.selectedItems():
            row = self.group_list.row(item)
            path = self.group_list.item(row).text()
            if path in self.state.group_data_paths:
                self.state.group_data_paths.remove(path)
            self.group_list.takeItem(row)
        self._update_group_view_button()

    def _update_group_view_button(self):
        self.btn_view_group.setEnabled(len(self.state.group_data_paths) > 0)

    def generate_group_visualizations(self):
        if not self.state.group_data_paths:
            self.log_message("No group files selected.")
            return
        self.log_message("Loading and processing group data...")
        single_animal_tabs = [self.heatmap_tab, self.com_tab, self.traj_tab, self.phase_tab, self.interp_tab]
        group_tabs = [self.group_scatter_tab, self.group_avg_tab]
        try:
            valid_fft_keys = ["minutes_per_frame", "period_min", "period_max"]
            phase_args = {}
            for key in valid_fft_keys:
                if key in self.phase_params:
                    widget, type_caster = self.phase_params[key]
                    value_str = widget.text().strip()
                    if value_str: phase_args[key] = type_caster(value_str)
            if "minutes_per_frame" not in phase_args: raise ValueError("Minutes per Frame must be set for group analysis.")
            
            all_dfs = []
            for i, roi_file in enumerate(self.state.group_data_paths):
                self.log_message(f"  [{i + 1}/{len(self.state.group_data_paths)}] {os.path.basename(roi_file)}")
                traces_file = roi_file.replace("_roi_warped.csv", "_traces.csv")
                unfiltered_roi_file = roi_file.replace("_roi_warped.csv", "_roi.csv")
                filtered_roi_file = roi_file.replace("_roi_warped.csv", "_roi_filtered.csv")
                required_files = [traces_file, unfiltered_roi_file, filtered_roi_file, roi_file]
                if not all(os.path.exists(f) for f in required_files):
                    self.log_message("    Warning: missing one or more required files; skipping.")
                    continue
                warped_rois = np.atleast_2d(np.loadtxt(roi_file, delimiter=","))
                filtered_rois = np.atleast_2d(np.loadtxt(filtered_roi_file, delimiter=","))
                if warped_rois.shape[0] != filtered_rois.shape[0]:
                    self.log_message(f"    ERROR: Data inconsistency detected. Warped file has {warped_rois.shape[0]} cells, but filtered file has {filtered_rois.shape[0]}. Skipping.")
                    continue
                traces = np.loadtxt(traces_file, delimiter=",")
                unfiltered_rois = np.loadtxt(unfiltered_roi_file, delimiter=",")
                indices = []
                for point in filtered_rois:
                    diff = np.sum((unfiltered_rois - point)**2, axis=1)
                    idx = np.argmin(diff)
                    if diff[idx] < 1e-9: indices.append(idx)
                if len(indices) != filtered_rois.shape[0]:
                    self.log_message("    Warning: Could not match filtered to unfiltered ROIs; skipping.")
                    continue
                trace_indices_to_keep = np.array(indices) + 1
                filtered_traces_data = traces[:, np.concatenate(([0], trace_indices_to_keep))]
                phases, period, _ = calculate_phases_fft(filtered_traces_data, **phase_args)
                ph_rad = (phases % period) * (2 * np.pi / period)
                mean_rad = circmean(ph_rad)
                mean_h = mean_rad * (period / (2 * np.pi))
                rel_phases = (phases - mean_h + period / 2) % period - period / 2
                animal_df = pd.DataFrame({
                    'Source_Animal': os.path.basename(roi_file).replace('_roi_warped.csv', ''),
                    'Warped_X': warped_rois[:, 0],
                    'Warped_Y': warped_rois[:, 1],
                    'Relative_Phase_Hours': rel_phases,
                    'Period_Hours': period
                })
                all_dfs.append(animal_df)
            
            if not all_dfs: raise ValueError("No valid and consistent group data was loaded.")
            
            group_scatter_df = pd.concat(all_dfs, ignore_index=True)
            period = group_scatter_df['Period_Hours'].iloc[0]
            grid_res = int(self.group_grid_res_edit.text())
            do_smooth = self.group_smooth_check.isChecked()
            x_min, x_max = group_scatter_df['Warped_X'].min(), group_scatter_df['Warped_X'].max()
            y_min, y_max = group_scatter_df['Warped_Y'].min(), group_scatter_df['Warped_Y'].max()
            grid_x_bins = np.linspace(x_min, x_max, grid_res + 1)
            grid_y_bins = np.linspace(y_min, y_max, grid_res + 1)
            group_scatter_df['Grid_X_Index'] = pd.cut(group_scatter_df['Warped_X'], bins=grid_x_bins, labels=False, include_lowest=True).fillna(-1).astype(int)
            group_scatter_df['Grid_Y_Index'] = pd.cut(group_scatter_df['Warped_Y'], bins=grid_y_bins, labels=False, include_lowest=True).fillna(-1).astype(int)

            def circmean_phase(series):
                rad = (series / (period / 2.0)) * np.pi
                mean_rad = circmean(rad)
                return (mean_rad / np.pi) * (period / 2.0)
            group_binned_df = group_scatter_df[group_scatter_df['Grid_X_Index'] >= 0].groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
            
            fig_s, _ = add_mpl_to_tab(self.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), group_scatter_df)
            self.visualization_widgets[self.group_scatter_tab] = viewer_s
            fig_g, _ = add_mpl_to_tab(self.group_avg_tab)
            viewer_g = GroupAverageMapViewer(fig_g, fig_g.add_subplot(111), group_binned_df, group_scatter_df, grid_res, do_smooth)
            self.visualization_widgets[self.group_avg_tab] = viewer_g
            for tab in group_tabs: self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(tab), True)
            for tab in single_animal_tabs: self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(tab), False)
            self.vis_tabs.setCurrentWidget(self.group_scatter_tab)
            self._mark_step_ready("group_view")
            self.log_message("Group visualizations generated.")
            self.btn_export_data.setEnabled(True)
        except Exception as e:
            import traceback
            self.log_message(f"Error generating group visualizations: {e}")
            self.log_message(traceback.format_exc())
            for tab in group_tabs: self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(tab), False)

    # --------------------------------------------------------


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
