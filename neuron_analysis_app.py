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
from scipy.signal import medfilt
from scipy.signal import find_peaks

import skimage.io
import colorcet as cet

from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Polygon, Rectangle

from PyQt5 import QtCore, QtWidgets, QtGui

import neuron_tracker_core as ntc
import cosinor as csn

# ------------------------------------------------------------
# Centralized State Management
# ------------------------------------------------------------

# ------------------------------------------------------------
# Tooltip Helper
# ------------------------------------------------------------
class Tooltip(QtCore.QObject):
    """A custom tooltip that can be styled and supports rich text."""
    _instance = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QtWidgets.QLabel(
            None,
            flags=QtCore.Qt.ToolTip | QtCore.Qt.BypassWindowManagerHint,
        )
        self._label.setWindowOpacity(0.95)
        self._label.setStyleSheet("""
            QLabel {
                border: 1px solid #555;
                padding: 5px;
                background-color: #ffffe0; /* Light yellow */
                color: #000;
                border-radius: 3px;
            }
        """)
        self._label.setWordWrap(True)
        self._label.setAlignment(QtCore.Qt.AlignLeft)
        self._timer = QtCore.QTimer(self, singleShot=True)
        self._timer.setInterval(750)  # ms delay
        self._timer.timeout.connect(self._show)
        self._text = ""

    def _show(self):
        if not self._text:
            return
        self._label.setText(self._text)
        self._label.adjustSize()
        pos = QtGui.QCursor.pos()
        x = pos.x() + 20
        y = pos.y() + 20
        
        screen_geo = QtWidgets.QApplication.desktop().availableGeometry(pos)
        if x + self._label.width() > screen_geo.right():
            x = pos.x() - self._label.width() - 20
        if y + self._label.height() > screen_geo.bottom():
            y = pos.y() - self._label.height() - 20
            
        self._label.move(x, y)
        self._label.show()

    def show_tooltip(self, text):
        self._text = text
        self._timer.start()

    def hide_tooltip(self):
        self._timer.stop()
        self._label.hide()

    @classmethod
    def install(cls, widget, text):
        if cls._instance is None:
            # Ensure the instance has a parent to be properly managed
            parent = widget.window() if widget.window() else QtWidgets.QApplication.instance()
            cls._instance = cls(parent)
        
        widget.setMouseTracking(True)
        
        class Filter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QtCore.QEvent.Enter:
                        cls._instance.show_tooltip(text)
                    elif event.type() == QtCore.QEvent.Leave:
                        cls._instance.hide_tooltip()
                return False

        if not hasattr(widget, '_tooltip_filter'):
            widget._tooltip_filter = Filter(widget)
            widget.installEventFilter(widget._tooltip_filter)

class AnalysisState:
    """A simple class to hold all application state."""
    def __init__(self):
        self.reset()

    def reset(self):
        # File paths
        self.project_path = "" # Path to the .ntp project file
        self.input_movie_path = ""
        self.output_basename = ""
        self.atlas_roi_path = ""
        self.reference_roi_path = ""

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

    def __init__(self, parent, bg_image, roi_data, output_basename, callback, vmin=None, vmax=None):
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
        self.finished_artists = []  # Track artists for finished polygons
        self.mode = "Include"

        # Attempt to load existing anatomical ROI file
        if self.output_basename:
            json_path = f"{self.output_basename}_anatomical_roi.json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        self.rois = json.load(f)
                except Exception as e:
                    print(f"Error loading existing ROIs: {e}")

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
        
        self.ref_btn = QtWidgets.QRadioButton("Phase Reference")
        self.ref_btn.setStyleSheet("color: blue;")
        
        self.include_btn.setChecked(True)

        self.include_btn.toggled.connect(self.update_mode)
        self.exclude_btn.toggled.connect(self.update_mode)
        self.ref_btn.toggled.connect(self.update_mode)

        btn_layout.addWidget(self.include_btn)
        btn_layout.addWidget(self.exclude_btn)
        btn_layout.addWidget(self.ref_btn)

        finish_btn = QtWidgets.QPushButton("Finish Polygon")
        finish_btn.clicked.connect(self.finish_polygon)
        btn_layout.addWidget(finish_btn)

        confirm_btn = QtWidgets.QPushButton("Confirm All ROIs")
        confirm_btn.clicked.connect(self.confirm_rois)
        btn_layout.addWidget(confirm_btn)

        main_layout.addLayout(btn_layout)
        
        # Draw any loaded ROIs immediately
        self.redraw_finished_rois()

    def redraw_finished_rois(self):
        # Remove old artists to avoid duplicates/ghosts
        for artist in self.finished_artists:
            artist.remove()
        self.finished_artists = []

        style_map = {
            "Include": {"color": "lime", "linestyle": "-"},
            "Exclude": {"color": "red", "linestyle": "-"},
            "Phase Reference": {"color": "blue", "linestyle": "--"}
        }

        for roi in self.rois:
            mode = roi.get("mode", "Include")
            verts = roi.get("path_vertices", [])
            if not verts:
                continue
            
            style = style_map.get(mode, style_map["Include"])
            
            poly = Polygon(
                verts, 
                closed=True, 
                fill=False, 
                edgecolor=style["color"], 
                linestyle=style["linestyle"], 
                linewidth=2
            )
            self.ax.add_patch(poly)
            self.finished_artists.append(poly)
        
        self.canvas.draw_idle()

    def update_mode(self):
        if self.include_btn.isChecked():
            self.mode = "Include"
        elif self.exclude_btn.isChecked():
            self.mode = "Exclude"
        else:
            self.mode = "Phase Reference"
        self.update_plot()

    def update_plot(self):
        # Handles the CURRENTLY DRAWING line (not finished ones)
        if self.current_line is not None:
            for artist in self.current_line:
                artist.remove()
        self.current_line = None

        # --- COLOR LOGIC ---
        color_map = {
            "Include": ("g-", "g+"),
            "Exclude": ("r-", "r+"),
            "Phase Reference": ("b-", "b+")
        }
        line_color, point_color = color_map[self.mode]

        if len(self.current_vertices) > 1:
            xs, ys = zip(*self.current_vertices)
            self.current_line = self.ax.plot(xs, ys, line_color)
        elif self.current_vertices:
            x, y = self.current_vertices[0]
            self.current_line = self.ax.plot(x, y, point_color)

        self.canvas.draw_idle()

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
                
    def finish_polygon(self):
        if len(self.current_vertices) > 2:
            # close loop for data consistency
            self.current_vertices.append(self.current_vertices[0])
            
            # Save to memory
            self.rois.append({
                "path_vertices": list(self.current_vertices),
                "mode": self.mode,
            })
            
            # Reset current drawing state
            self.current_vertices = []
            if self.current_line is not None:
                for artist in self.current_line:
                    artist.remove()
                self.current_line = None
            
            # Redraw all finished ROIs so the new one appears permanently
            self.redraw_finished_rois()
            
            self.ax.set_title(f"{len(self.rois)} ROI(s) defined. Draw another or Confirm.")

    def confirm_rois(self):
        # Separate the ROIs by their purpose
        anatomical_rois = [r for r in self.rois if r["mode"] in ("Include", "Exclude")]
        phase_ref_rois = [r for r in self.rois if r["mode"] == "Phase Reference"]

        # 1. Save the ANATOMICAL (warping) ROI JSON
        # Check if ANY ROIs exist (including Phase Ref), not just anatomical ones.
        all_rois_to_save = anatomical_rois + phase_ref_rois
        
        if all_rois_to_save and self.output_basename:
            filepath = f"{self.output_basename}_anatomical_roi.json"
            try:
                serializable = [
                    {"path_vertices": roi["path_vertices"], "mode": roi["mode"]}
                    for roi in all_rois_to_save
                ]
                with open(filepath, "w") as f:
                    json.dump(serializable, f, indent=4)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving anatomical ROI file:\n{e}")

        # 2. Calculate the FILTERING based on Include/Exclude ROIs
        if not anatomical_rois:
            # No filter defined, pass back all data
            if self.callback:
                self.callback(None, None, phase_ref_rois) # Still pass back phase ref
        else:
            # Build paths and calculate the final mask for filtering
            final_mask = np.zeros(len(self.roi_data), dtype=bool)
            include_paths = [Path(r["path_vertices"]) for r in anatomical_rois if r["mode"] == "Include"]
            if include_paths:
                for path in include_paths:
                    final_mask |= path.contains_points(self.roi_data)
            else:
                final_mask[:] = True # If only exclude ROIs are present, start with all points selected

            for roi in anatomical_rois:
                if roi["mode"] == "Exclude":
                    final_mask &= ~Path(roi["path_vertices"]).contains_points(self.roi_data)

            filtered_indices = np.where(final_mask)[0]

            # Save the filtered ROI CSV
            if self.output_basename:
                try:
                    np.savetxt(f"{self.output_basename}_roi_filtered.csv", self.roi_data[filtered_indices], delimiter=",")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving filtered ROI CSV:\n{e}")

            # 3. Callback with ALL THREE pieces of information
            if self.callback:
                self.callback(filtered_indices, anatomical_rois, phase_ref_rois)

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
    def __init__(self, fig, loaded_data, filtered_indices, phases, rhythm_scores, is_emphasized, rhythm_sort_desc, 
                 period=None, minutes_per_frame=None, reference_phase=None, trend_window_hours=None):
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
        
        # Track the currently displayed trace so we can refresh it when params change
        self.current_selected_index = None 
        
        # Analysis Parameters for On-Demand Visualization
        self.period = period
        self.minutes_per_frame = minutes_per_frame
        self.reference_phase = reference_phase
        self.trend_window_hours = trend_window_hours or 36.0

        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.45) 
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
        self.ax_trace.set_ylabel("Detrended Intensity")
        self.ax_trace.set_title("Selected Cell Trace")
        
        self.trace_line = self.ax_trace.plot([], [], 'b-', alpha=0.6, label="Data")[0]
        self.fit_line = self.ax_trace.plot([], [], 'r-', linewidth=2, label="Fit")[0]
        self.ax_trace.legend(loc="upper right", fontsize="small")

        if rhythm_scores is not None:
            main_win = self.fig.canvas.parent().window()
            try:
                thr = float(main_win.phase_params["rhythm_threshold"][0].text())
            except:
                thr = 0.0
            if self.rhythm_sort_desc: self.rhythm_mask = rhythm_scores >= thr
            else: self.rhythm_mask = rhythm_scores <= thr

        self.update_phase_data(phases, rhythm_scores, self.rhythm_mask, self.rhythm_sort_desc, 
                               period, minutes_per_frame, reference_phase, trend_window_hours)
        
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

    def update_phase_data(self, phases, rhythm_scores, rhythm_mask=None, sort_desc=True, 
                          period=None, minutes_per_frame=None, reference_phase=None, trend_window_hours=None):
        self.phases = phases
        self.rhythm_scores = rhythm_scores
        self.rhythm_mask = rhythm_mask
        self.rhythm_sort_desc = sort_desc
        
        # Update analysis params if provided
        if period is not None: self.period = period
        if minutes_per_frame is not None: self.minutes_per_frame = minutes_per_frame
        self.reference_phase = reference_phase 
        if trend_window_hours is not None: self.trend_window_hours = trend_window_hours

        self.sort_values = {"Y-coordinate": self.roi_data[:, 1]}
        if self.phases is not None: self.sort_values["Phase"] = self.phases
        if self.rhythm_scores is not None: self.sort_values["Rhythmicity"] = self.rhythm_scores
        
        current_sort = self.radio_buttons.value_selected
        if current_sort in self.sort_values: self.on_sort_change(current_sort)

        # Automatically refresh the selected trace with new params (e.g. new Ref Phase)
        if self.current_selected_index is not None:
            self.update_selected_trace(self.current_selected_index)

    def update_rhythm_emphasis(self, rhythm_mask, is_emphasized):
        self.is_emphasized, self.rhythm_mask = is_emphasized, rhythm_mask
        self.on_sort_change(self.radio_buttons.value_selected)

    def update_selected_trace(self, original_index):
        # FIX: Store the current selection state
        self.current_selected_index = original_index

        # Clear previous vertical lines
        while len(self.ax_trace.lines) > 2:
            self.ax_trace.lines[-1].remove()
            
        if self.filtered_indices is not None:
            try: current_index = np.where(self.filtered_indices == original_index)[0][0]
            except IndexError:
                self.trace_line.set_data([], [])
                self.fit_line.set_data([], [])
                self.ax_trace.set_title("Selected Cell Trace (Not in current filter)")
                self.fig.canvas.draw_idle()
                return
        else: current_index = original_index

        if 0 <= current_index < self.traces_data.shape[1] - 1:
            # 1. Get Data
            time_frames = self.traces_data[:, 0]
            raw_intensity = self.traces_data[:, current_index + 1]
            
            # 2. Detrend 
            mpf = self.minutes_per_frame or 15.0
            trend_win = self.trend_window_hours or 36.0
            win_frames = compute_median_window_frames(mpf, trend_win, len(raw_intensity))
            detrended = preprocess_for_rhythmicity(raw_intensity, method="running_median", median_window_frames=win_frames)
            
            self.trace_line.set_data(time_frames, detrended)
            
            # 3. Calculate and Draw Fit
            title_text = f"Trace for ROI {original_index + 1}"
            
            if self.period and self.minutes_per_frame:
                time_hours = time_frames * (self.minutes_per_frame / 60.0)
                res = csn.cosinor_analysis(detrended, time_hours, self.period)
                
                if not np.isnan(res['amplitude']):
                    # Model: M + A * cos(w * (t - acrophase))
                    w = 2 * np.pi / self.period
                    model = res['mesor'] + res['amplitude'] * np.cos(w * (time_hours - res['acrophase']))
                    self.fit_line.set_data(time_frames, model)
                    
                    # 4. Draw Markers and Stats
                    cell_phase = res['acrophase']
                    
                    # Draw vertical line at first occurrence
                    phase_frame = (cell_phase / (self.minutes_per_frame / 60.0))
                    self.ax_trace.axvline(phase_frame, color='r', linestyle='-', alpha=0.8, label='Cell Peak')
                    
                    # Reference Line
                    ref_text = ""
                    if self.reference_phase is not None:
                        ref_phase_frame = (self.reference_phase / (self.minutes_per_frame / 60.0))
                        self.ax_trace.axvline(ref_phase_frame, color='k', linestyle='--', alpha=0.8, label='Ref Peak')
                        
                        # Calculate Delta
                        diff = (cell_phase - self.reference_phase + self.period/2) % self.period - self.period/2
                        sign = "+" if diff > 0 else ""
                        ref_text = f" | Ref: {self.reference_phase:.1f}h | Δ: {sign}{diff:.1f}h"

                    title_text += f" | Phase: {cell_phase:.1f}h{ref_text}"
                else:
                     self.fit_line.set_data([], [])
            else:
                self.fit_line.set_data([], [])

            self.ax_trace.set_title(title_text, fontsize=10)
            self.ax_trace.relim()
            self.ax_trace.autoscale_view()
        else:
            self.trace_line.set_data([], [])
            self.fit_line.set_data([], [])
            self.ax_trace.set_title("Selected Cell Trace")
            
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
    def __init__(self, fig, ax, bg_image, com_points, on_change_callback, on_select_callback, filtered_indices=None, rois=None):
        self.fig = fig
        self.ax = ax
        self.com_points = com_points
        self.on_change_callback = on_change_callback
        self.on_select_callback = on_select_callback
        self.filtered_indices = filtered_indices
        self.rois = rois or [] # Store ROIs
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
        
        self._draw_rois()    # Draw polygons
        self._draw_scatter() # Initial drawing

    def _draw_rois(self):
        """Draws the Include/Exclude/Reference polygons on the axes."""
        if not self.rois:
            return

        style_map = {
            "Include": {"color": "lime", "linestyle": "-", "linewidth": 1.5},
            "Exclude": {"color": "red", "linestyle": "-", "linewidth": 1.5},
            "Phase Reference": {"color": "cyan", "linestyle": "--", "linewidth": 1.5}
        }

        for roi in self.rois:
            mode = roi.get("mode", "Include")
            verts = roi.get("path_vertices", [])
            if not verts:
                continue
            
            style = style_map.get(mode, style_map["Include"])
            
            # Create polygon patch
            # fill=False keeps the inside empty so you can see the cells
            poly = Polygon(
                verts, 
                closed=True, 
                fill=False, 
                edgecolor=style["color"], 
                linestyle=style["linestyle"], 
                linewidth=style["linewidth"],
                label=mode
            )
            self.ax.add_patch(poly)
        
        # Optional: Add a simple legend if ROIs exist, handling duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', framealpha=0.5)

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
        selected_index = np.argmin(distances) # This is the local/filtered index

        if distances[selected_index] < 20:
            # Highlight the point in this plot (uses the local index)
            # Calculate the Original Index FIRST, then call highlight with it.
            if self.filtered_indices is not None:
                original_index = self.filtered_indices[selected_index]
            else:
                original_index = selected_index
            
            # This ensures the highlighting logic is consistent regardless of who called it
            self.highlight_point(original_index)

            if self.on_select_callback:
                # Send the CORRECT, original index to the main window
                self.on_select_callback(original_index)

    def highlight_point(self, index):
        """
        Highlights a point. 
        'index' is treated as the GLOBAL (Original) index.
        We must convert it to the Local index to plot it on the filtered view.
        """
        if self.highlight_artist:
            self.highlight_artist.remove()
            self.highlight_artist = None
        
        local_index = None
        if index is not None:
            if self.filtered_indices is not None:
                # Find where the global index exists in the filtered array
                # This handles the case where the global index might have been filtered out
                matches = np.where(self.filtered_indices == index)[0]
                if len(matches) > 0:
                    local_index = matches[0]
                else:
                    local_index = None
            else:
                # No filter active, 1:1 mapping
                local_index = index

        if local_index is not None and 0 <= local_index < len(self.com_points):
            point = self.com_points[local_index]
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
# Phase calculation
# ------------------------------------------------------------

RHYTHM_TREND_WINDOW_HOURS = 36.0  # default detrend timescale for circadian work

def compute_median_window_frames(minutes_per_frame, trend_window_hours, T=None):
    """
    Compute an odd-length median filter window in frames for a given trend window in hours.

    Parameters:
        minutes_per_frame : float
            Sampling interval in minutes.
        trend_window_hours : float
            Desired width of the detrending window in hours.
        T : int or None
            Optional number of frames in the trace, used to cap the window.

    Returns:
        median_window_frames : int (odd, >= 3)
    """
    if minutes_per_frame <= 0:
        # Fallback: use a small default window in frames if sampling is invalid
        frames = 3
    else:
        hours_per_frame = minutes_per_frame / 60.0
        if hours_per_frame <= 0:
            frames = 3
        else:
            frames = int(round(trend_window_hours / hours_per_frame))
            if frames < 3:
                frames = 3

    # Enforce odd window size
    if frames % 2 == 0:
        frames += 1

    # Cap by T if provided and meaningful
    if T is not None and T > 0:
        if frames > T:
            frames = T if (T % 2 == 1) else (T - 1)
            if frames < 3:
                frames = 3

    return frames

def preprocess_for_rhythmicity(trace, method="running_median",
                               median_window_frames=None,
                               poly_order=2):
    """
    Preprocess a raw fluorescence trace before rhythmicity analysis.
    Returns a detrended trace suitable for FFT or cosinor.

    Parameters:
        trace: 1D numpy array
        method: "running_median", "polynomial", or "none"
        median_window_frames: int or None
            If None and method == "running_median", caller must have computed a
            suitable window; this function will fall back to a small default.
        poly_order: polynomial order for polynomial detrending

    Returns:
        detrended_trace: 1D numpy array
    """
    trace = np.asarray(trace, dtype=float)

    if method == "none":
        return trace.copy()

    x = np.arange(len(trace))

    if method == "running_median":
        # If the caller did not provide a window, fall back to a minimal safe default.
        if median_window_frames is None:
            median_window_frames = 3
        if median_window_frames % 2 == 0:
            median_window_frames += 1
        baseline = medfilt(trace, kernel_size=median_window_frames)
        detrended = trace - baseline
        return detrended

    if method == "polynomial":
        # Global polynomial baseline
        if trace.size < (poly_order + 1):
            return trace.copy()
        coeffs = np.polyfit(x, trace, poly_order)
        baseline = np.polyval(coeffs, x)
        detrended = trace - baseline
        return detrended

    # Fallback
    return trace.copy()

def estimate_cycle_count_from_trace(
    trace,
    minutes_per_frame,
    period_hours,
    detrend_method="running_median",
    median_window_frames=None,
    trend_window_hours=RHYTHM_TREND_WINDOW_HOURS,
    smoothing_window_hours=2.0,
    min_prominence_fraction=0.2,
):
    """
    Estimate how many cycles a single trace contains around a target period.

    Parameters:
        trace : 1D array-like
            Raw fluorescence trace for one cell.
        minutes_per_frame : float
            Sampling interval in minutes.
        period_hours : float
            Target period (e.g., 24.0 for circadian).
        detrend_method : str
            Method passed to preprocess_for_rhythmicity ("running_median", "polynomial", or "none").
        median_window_frames : int or None
            If None and detrend_method == "running_median", window is derived from
            minutes_per_frame and trend_window_hours.
        trend_window_hours : float
            Trend window in hours for dynamic median-window computation.
        smoothing_window_hours : float
            Width of the smoothing window (in hours) for peak detection.
        min_prominence_fraction : float
            Minimum peak prominence as a fraction of the smoothed trace amplitude range.

    Returns:
        n_cycles : int
            Estimated number of cycles (roughly number of usable peaks minus 1).
        peak_indices : 1D np.ndarray of ints
            Indices of peaks used for counting.
    """
    trace = np.asarray(trace, dtype=float)
    T = trace.shape[0]
    if T < 3:
        return 0, np.array([], dtype=int)

    # Choose median window in frames if needed
    if detrend_method == "running_median" and median_window_frames is None:
        median_window_frames = compute_median_window_frames(
            minutes_per_frame,
            trend_window_hours,
            T=T,
        )

    detr = preprocess_for_rhythmicity(
        trace,
        method=detrend_method,
        median_window_frames=median_window_frames,
    )

    hours_per_frame = minutes_per_frame / 60.0
    if hours_per_frame <= 0:
        return 0, np.array([], dtype=int)

    window_frames = int(round(smoothing_window_hours / hours_per_frame))
    if window_frames < 1:
        window_frames = 1
    if window_frames % 2 == 0:
        window_frames += 1

    if window_frames > 1:
        kernel = np.ones(window_frames, dtype=float) / float(window_frames)
        smooth = np.convolve(detr, kernel, mode="same")
    else:
        smooth = detr

    amp_range = float(smooth.max() - smooth.min())
    if amp_range <= 0:
        return 0, np.array([], dtype=int)

    prominence = amp_range * float(min_prominence_fraction)
    if prominence <= 0:
        prominence = amp_range * 0.1

    peaks, _ = find_peaks(smooth, prominence=prominence)
    if peaks.size == 0:
        return 0, peaks

    expected_frames = period_hours / hours_per_frame
    if expected_frames <= 0:
        return int(peaks.size), peaks

    min_dist = 0.5 * expected_frames
    max_dist = 1.5 * expected_frames

    good_peaks = []
    prev_peak = None
    for idx in peaks:
        if prev_peak is None:
            good_peaks.append(idx)
            prev_peak = idx
        else:
            d = idx - prev_peak
            if min_dist <= d <= max_dist:
                good_peaks.append(idx)
                prev_peak = idx
            elif d > max_dist:
                good_peaks.append(idx)
                prev_peak = idx
            else:
                continue

    good_peaks = np.array(good_peaks, dtype=int)
    n_cycles = max(int(good_peaks.size - 1), 0)
    return n_cycles, good_peaks

def strict_cycle_mask(
    traces_data,
    minutes_per_frame,
    period_hours,
    base_mask,
    min_cycles=2,
    detrend_method="running_median",
    median_window_frames=None,
    trend_window_hours=RHYTHM_TREND_WINDOW_HOURS,
    smoothing_window_hours=2.0,
    min_prominence_fraction=0.2,
):
    """
    Refine a base rhythmicity mask by requiring a minimum number of cycles.

    Parameters:
        traces_data : 2D array-like, shape (T, N+1)
            First column is time, remaining columns are per-cell traces.
        minutes_per_frame : float
            Sampling interval in minutes.
        period_hours : float
            Target period (e.g., 24.0 for circadian).
        base_mask : 1D boolean array, length N
            Initial rhythmicity decision per cell (e.g., Cosinor/FFT filter).
        min_cycles : int
            Minimum number of cycles required to keep a cell as "rhythmic".
        detrend_method, median_window_frames, trend_window_hours,
        smoothing_window_hours, min_prominence_fraction :
            Parameters passed through to estimate_cycle_count_from_trace.

    Returns:
        strict_mask : 1D boolean np.ndarray, length N
            Refined mask that incorporates both base_mask and cycle-count requirement.
    """
    traces_data = np.asarray(traces_data)
    base_mask = np.asarray(base_mask, dtype=bool)

    if traces_data.ndim != 2 or traces_data.shape[1] < 2:
        raise ValueError("traces_data must have shape (T, N+1).")

    T, cols = traces_data.shape
    N = cols - 1

    if base_mask.shape[0] != N:
        raise ValueError("base_mask length must equal number of cells (N).")

    strict_mask = base_mask.copy()
    if not np.any(base_mask):
        return strict_mask

    intensities = traces_data[:, 1:]

    for i in range(N):
        if not base_mask[i]:
            continue

        trace = intensities[:, i]
        n_cycles, _ = estimate_cycle_count_from_trace(
            trace,
            minutes_per_frame=minutes_per_frame,
            period_hours=period_hours,
            detrend_method=detrend_method,
            median_window_frames=median_window_frames,
            trend_window_hours=trend_window_hours,
            smoothing_window_hours=smoothing_window_hours,
            min_prominence_fraction=min_prominence_fraction,
        )

        if n_cycles < min_cycles:
            strict_mask[i] = False

    return strict_mask

def calculate_phases_fft(traces_data,
                         minutes_per_frame,
                         period_min=None,
                         period_max=None,
                         detrend_method="running_median",
                         detrend_window_frames=None,
                         detrend_window_hours=RHYTHM_TREND_WINDOW_HOURS):
    """
    Compute per-cell FFT phases and a sideband-based SNR rhythmicity score.

    Parameters:
        traces_data: 2D array of shape (T, N+1) where first column is time
                     and remaining columns are fluorescence traces.
        minutes_per_frame: float
        period_min, period_max: optional bounds on allowed periods (hours)
        detrend_method: "running_median", "polynomial", or "none"
        detrend_window_frames: int or None
            If None and detrend_method == "running_median", the window will be
            derived from minutes_per_frame and detrend_window_hours.
        detrend_window_hours: float
            Trend window width in hours for dynamic median-window computation.

    Returns:
        phases: array of shape (N,)
        period_hours: float, dominant period
        rhythm_snr_scores: array of shape (N,)
    """

    intensities = traces_data[:, 1:]        # (T, N)
    T, N = intensities.shape

    # Time → frequency mapping
    dt_hours = minutes_per_frame / 60.0
    freqs = np.fft.rfftfreq(T, d=dt_hours)

    # Choose median window in frames if needed
    if detrend_method == "running_median" and detrend_window_frames is None:
        detrend_window_frames = compute_median_window_frames(
            minutes_per_frame,
            detrend_window_hours,
            T=T,
        )

    # Detrend ALL traces before spectrum analysis
    detrended = np.zeros_like(intensities)
    for i in range(N):
        detrended[:, i] = preprocess_for_rhythmicity(
            intensities[:, i],
            method=detrend_method,
            median_window_frames=detrend_window_frames,
        )

    # Mean trace used to estimate dominant period
    mean_signal = detrended.mean(axis=1)
    fft_mean = np.fft.rfft(mean_signal)
    power_mean = np.abs(fft_mean) ** 2

    # Determine allowed frequency range for peak search
    if period_min is not None and period_max is not None:
        fmin = 1.0 / period_max
        fmax = 1.0 / period_min
        mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        mask = np.ones_like(freqs, dtype=bool)

    masked_power = power_mean.copy()
    masked_power[~mask] = 0
    peak_idx = np.argmax(masked_power)
    peak_freq = freqs[peak_idx]
    period_hours = 1.0 / peak_freq if peak_freq > 0 else np.inf

    # Sideband SNR: signal band = ±1 bins; noise = ±10 bins outside that
    signal_band = []
    for off in (-1, 0, 1):
        idx = peak_idx + off
        if 0 <= idx < len(freqs):
            signal_band.append(idx)
    signal_band = np.array(signal_band, dtype=int)

    noise_band = []
    for off in range(2, 11):
        for sign in (-1, +1):
            idx = peak_idx + sign * off
            if 0 <= idx < len(freqs):
                noise_band.append(idx)
    noise_band = np.array(noise_band, dtype=int)

    phases = np.zeros(N)
    rhythm_snr_scores = np.zeros(N)

    for i in range(N):
        trace = detrended[:, i]
        fft_vals = np.fft.rfft(trace)
        power_vals = np.abs(fft_vals) ** 2

        complex_val = fft_vals[peak_idx]
        phases[i] = np.angle(complex_val)

        signal_power = power_vals[signal_band].sum()
        noise_power = power_vals[noise_band].mean() if len(noise_band) > 0 else 1e-9
        snr = signal_power / (noise_power + 1e-9)
        rhythm_snr_scores[i] = snr

    return phases, period_hours, rhythm_snr_scores

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
        self.setWindowTitle("Neuron Analysis Workspace")
        self.resize(1400, 900)

        # Centralized state object
        self.state = AnalysisState()

        self.params = {}
        self.phase_params = {}
        self.filtered_indices = None
        self.rois = None
        self.phase_reference_rois = None
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
        self._build_menu() # Call the new menu builder
        self._connect_signals()

    # ------------ UI construction ------------

    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QtWidgets.QAction("&Open Project...", self)
        open_action.triggered.connect(self.load_project)
        file_menu.addAction(open_action)

        save_action = QtWidgets.QAction("&Save Project", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        save_as_action = QtWidgets.QAction("Save Project &As...", self)
        save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        file_menu.addAction(save_as_action)
        
    def save_project(self, save_as=False):
        """Saves the current session state to a project file."""
        project_path = self.state.project_path
        if not project_path or save_as:
            start_dir = self._get_last_dir()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Project", start_dir, "Neuron Tracker Project (*.ntp)"
            )
            if not path:
                return
            project_path = path
        
        try:
            self._sync_state_from_ui()
            state_to_save = {
                'atlas_roi_path': self.state.atlas_roi_path,
                'target_roi_paths': self.state.target_roi_paths,
                'warp_param_paths': self.state.warp_param_paths,
                'group_data_paths': self.state.group_data_paths,
            }
            with open(project_path, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            
            self._set_last_dir(project_path)
            self.state.project_path = project_path
            self.log_message(f"Project saved to {os.path.basename(self.state.project_path)}")
            self.setWindowTitle(f"{os.path.basename(self.state.project_path)} - Neuron Analysis Workspace (PyQt)")

        except Exception as e:
            self.log_message(f"Error saving project: {e}")


    def load_project(self):
        """Loads a session state from a project file."""
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", start_dir, "Neuron Tracker Project (*.ntp)"
        )
        if not path:
            return
        
        self._set_last_dir(path)
        self._reset_state()
        
        try:
            with open(path, 'r') as f:
                loaded_state = json.load(f)
            
            self.state.project_path = path
            self.state.atlas_roi_path = loaded_state.get('atlas_roi_path', "")
            self.state.target_roi_paths = loaded_state.get('target_roi_paths', [])
            self.state.warp_param_paths = loaded_state.get('warp_param_paths', [])
            self.state.group_data_paths = loaded_state.get('group_data_paths', [])
            
            self.log_message(f"Loaded project: {os.path.basename(path)}")
            self._update_ui_from_state()
            self.setWindowTitle(f"{os.path.basename(path)} - Neuron Analysis Workspace (PyQt)")

        except Exception as e:
            self.log_message(f"Error loading project: {e}")

    def _update_ui_from_state(self):
        """Updates all UI list widgets to match the current state object."""
        self.atlas_path_edit.setText(self.state.atlas_roi_path)
        
        self.target_list.clear()
        self.target_list.addItems(self.state.target_roi_paths)
        
        self.warp_list.clear()
        self.warp_list.addItems(self.state.warp_param_paths)
        
        self.group_list.clear()
        self.group_list.addItems(self.state.group_data_paths)
        
        # Trigger updates for button enabled/disabled states
        self._update_reg_button_state()
        self.check_apply_warp_buttons_state()
        self._update_group_view_button()
        self.log_message("UI updated from loaded project.")

    def _sync_state_from_ui(self):
        """Updates the state object from the current UI list widgets."""
        self.state.atlas_roi_path = self.atlas_path_edit.text()
        
        self.state.target_roi_paths = [self.target_list.item(i).text() for i in range(self.target_list.count())]
        self.state.warp_param_paths = [self.warp_list.item(i).text() for i in range(self.warp_list.count())]
        self.state.group_data_paths = [self.group_list.item(i).text() for i in range(self.group_list.count())]
        self.log_message("Internal state synchronized from UI.")

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
            base = "group_analysis"
        
        default_filename = f"{os.path.basename(base)}_{default_filename}" if base else default_filename
        
        start_dir = self._get_last_dir()
        suggested_path = os.path.join(start_dir, default_filename)

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", suggested_path, "CSV files (*.csv)"
        )

        if not path:
            return
        
        self._set_last_dir(path)
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
        Tooltip.install(self.btn_run_analysis, "Performs the complete cell tracking and trace extraction pipeline on the loaded movie. This may take several minutes and will overwrite existing result files for this movie.")
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
        self.ctrl_layout.addWidget(exec_box)
        self.ctrl_layout.addStretch(1)

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
        add_mode("register", "Atlas Registration", True)
        add_mode("apply_warp", "Apply Warp to Data", True)
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
        Tooltip.install(self.btn_define_roi, "Launches the ROI drawing tool. This is required to create the `_anatomical_roi.json` file, which is a prerequisite for the Atlas Registration workflow.")
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI Filter")
        self.btn_clear_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_define_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addWidget(roi_box)
        
        param_tooltips = {
            "sigma1": "<b>What it is:</b> The size (pixels) of the smaller Gaussian blur, roughly the radius of your cells.<br><b>How to tune it:</b> Decrease for smaller cells, increase for larger cells.<br><b>Trade-off:</b> Too small detects noise; too large merges cells.",
            "sigma2": "<b>What it is:</b> The size (pixels) of the larger Gaussian blur for background subtraction.<br><b>How to tune it:</b> Increase for large, uneven background brightness.<br><b>Trade-off:</b> Too small subtracts cells; too large is ineffective.",
            "blur_sigma": "<b>What it is:</b> A small blur applied before ranking features by brightness.<br><b>How to tune it:</b> A small value (1-2 pixels) makes brightness measurement more stable.<br><b>Trade-off:</b> 0 is fine, but slight blurring is often more robust.",
            "max_features": "<b>What it is:</b> The max number of brightest features to consider in any single frame.<br><b>How to tune it:</b> Set higher than the max number of cells you expect to see.<br><b>Trade-off:</b> Too low misses cells; too high includes noise and slows analysis.",
            "search_range": "<b>What it is:</b> Max number of frames to look backward in time to link a track.<br><b>How to tune it:</b> Increase if cells can disappear for long periods.<br><b>Trade-off:</b> Too high increases incorrect matches and slows tracking.",
            "cone_radius_base": "<b>What it is:</b> The initial search radius (pixels) for linking a cell to the very next frame.<br><b>How to tune it:</b> Increase if cells move rapidly between frames.<br><b>Trade-off:</b> Too small fails to track fast cells; too large risks mismatches with neighbors.",
            "cone_radius_multiplier": "<b>What it is:</b> A factor allowing the search radius to grow for linking across larger time gaps.<br><b>How to tune it:</b> Advanced. Leave at default unless movement is very erratic.<br><b>Trade-off:</b> Helps track fast cells over gaps, but increases mismatch risk.",
            "min_trajectory_length": "<b>What it is:</b> The minimum track length as a fraction of total movie length (e.g., 0.08 = 8%).<br><b>How to tune it:</b> Increase to be stricter; decrease to include transient cells.<br><b>Trade-off:</b> Too high discards valid cells; too low includes noisy false-positives.",
            "sampling_box_size": "<b>What it is:</b> Side length (pixels) of the square box used to measure cell brightness.<br><b>How to tune it:</b> Should be large enough to encompass the entire cell (e.g., 2-3x cell diameter).<br><b>Trade-off:</b> Too small misses signal; too large includes background noise.",
            "sampling_sigma": "<b>What it is:</b> Blur applied within the sampling box for a weighted measurement.<br><b>How to tune it:</b> Should be close to the cell's radius. A value of 0 is a simple average.<br><b>Trade-off:</b> A good value (e.g., 2.0) makes the measurement robust to tracking jitter.",
            "max_interpolation_distance": "<b>What it is:</b> A safety check. Tracks with a frame-to-frame jump larger than this (in pixels) are discarded.<br><b>How to tune it:</b> Set slightly larger than the max plausible distance a cell could move in one frame.<br><b>Trade-off:</b> Too low discards valid fast cells; too high fails to catch errors.",
            "minutes_per_frame": "<b>REQUIRED:</b> The sampling interval of your recording in minutes.",
            "period_min": "<b>Optional:</b> Constrains the rhythm search to periods LONGER than this value (in hours).",
            "period_max": "<b>Optional:</b> Constrains the rhythm search to periods SHORTER than this value (in hours).",
            "grid_resolution": "<b>What it is:</b> The resolution of the grid used for interpolated and group average maps.<br><b>How to tune it:</b> Higher values produce a smoother, higher-resolution image but can take longer to compute.",
            "rhythm_threshold": "<b>What it is:</b> The cutoff for considering a cell 'rhythmic'. Its meaning depends on the selected Analysis Method.<br><b>FFT (SNR):</b> A signal-to-noise ratio. A value >= 2.0 is a good start.<br><b>Cosinor (p-value):</b> The statistical significance. A value less than or equal to 0.05 is standard.",
            "r_squared_threshold": "<b>What it is:</b> (Cosinor only) The 'goodness-of-fit'. Filters for cells where the cosine model explains at least this much of the data's variance.<br><b>How to tune it:</b> A value >= 0.3 is a reasonable starting point for finding well-fit rhythms."
        }
        
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
        self._add_param_field(det_layout, "sigma1", 3.0, param_tooltips)
        self._add_param_field(det_layout, "sigma2", 20.0, param_tooltips)
        self._add_param_field(det_layout, "blur_sigma", 2.0, param_tooltips)
        self._add_param_field(det_layout, "max_features", 200, param_tooltips)
        tabs.addTab(det_tab, "Detection")
        tr_tab = QtWidgets.QWidget()
        tr_layout = QtWidgets.QFormLayout(tr_tab)
        self._add_param_field(tr_layout, "search_range", 50, param_tooltips)
        self._add_param_field(tr_layout, "cone_radius_base", 1.5, param_tooltips)
        self._add_param_field(tr_layout, "cone_radius_multiplier", 0.125, param_tooltips)
        tabs.addTab(tr_tab, "Tracking")
        fl_tab = QtWidgets.QWidget()
        fl_layout = QtWidgets.QFormLayout(fl_tab)
        self._add_param_field(fl_layout, "min_trajectory_length", 0.08, param_tooltips)
        self._add_param_field(fl_layout, "sampling_box_size", 15, param_tooltips)
        self._add_param_field(fl_layout, "sampling_sigma", 2.0, param_tooltips)
        self._add_param_field(fl_layout, "max_interpolation_distance", 5.0, param_tooltips)
        tabs.addTab(fl_tab, "Filtering")
        layout.addWidget(param_box)
        phase_box = QtWidgets.QGroupBox("Phase Map Parameters")
        phase_layout = QtWidgets.QFormLayout(phase_box)
        self.analysis_method_combo = QtWidgets.QComboBox()
        self.analysis_method_combo.addItems(["FFT (SNR)", "Cosinor (p-value)"])
        phase_layout.addRow("Analysis Method:", self.analysis_method_combo)
        
        self.use_subregion_ref_check = QtWidgets.QCheckBox("Use Drawn Sub-Region as Phase Reference")
        Tooltip.install(
            self.use_subregion_ref_check,
            "<b>What it does:</b> If checked, the mean phase of only the cells inside your drawn 'Phase Reference' polygon will be used as the 'zero point' for the phase map."
        )
        self.use_subregion_ref_check.setEnabled(False) # Disabled until a ref ROI is drawn
        phase_layout.addRow(self.use_subregion_ref_check)        

        # Core rhythmicity parameters
        self._add_phase_field(phase_layout, "minutes_per_frame", 15.0, tooltips=param_tooltips)
        self.discovered_period_edit = QtWidgets.QLineEdit("N/A")
        self.discovered_period_edit.setReadOnly(True)
        phase_layout.addRow("Discovered Period (hrs):", self.discovered_period_edit)
        self._add_phase_field(phase_layout, "period_min", 22.0, tooltips=param_tooltips)
        self._add_phase_field(phase_layout, "period_max", 28.0, tooltips=param_tooltips)

        # Detrend window in hours (controls RHYTHM_TREND_WINDOW_HOURS behavior)
        self._add_phase_field(phase_layout, "trend_window_hours", 36.0, tooltips=param_tooltips)

        self._add_phase_field(phase_layout, "grid_resolution", 100, int, param_tooltips)
        _, self.rhythm_threshold_label = self._add_phase_field(
            phase_layout,
            "rhythm_threshold",
            2.0,
            tooltips=param_tooltips,
        )
        rsquared_le, rsquared_label = self._add_phase_field(
            phase_layout,
            "r_squared_threshold",
            0.3,
            tooltips=param_tooltips,
        )
        self.rsquared_widgets = (rsquared_label, rsquared_le)

        # Rhythm emphasis checkbox
        self.emphasize_rhythm_check = QtWidgets.QCheckBox("Emphasize rhythmic cells in all plots")
        Tooltip.install(
            self.emphasize_rhythm_check,
            "<b>What it does:</b> de-emphasizes non-rhythmic cells in all visualizations "
            "(Heatmap, Center of Mass, Phase Maps). Rhythmic cells remain fully visible.",
        )
        phase_layout.addRow(self.emphasize_rhythm_check)

        # Strict cycle-count checkbox
        self.strict_cycle_check = QtWidgets.QCheckBox("Require >= 2 cycles (strict filter)")
        Tooltip.install(
            self.strict_cycle_check,
            "<b>What it does:</b> after FFT/Cosinor thresholds, only keeps cells that show at "
            "least two cycles near the target period, based on peak counting.",
        )
        phase_layout.addRow(self.strict_cycle_check)

        self.btn_regen_phase = QtWidgets.QPushButton("Update Plots")
        Tooltip.install(
            self.btn_regen_phase,
            "Re-calculates all relevant plots (Heatmap, Center of Mass, Phase Maps).",
        )
        self.btn_regen_phase.setEnabled(False)
        phase_layout.addRow(self.btn_regen_phase)

        layout.addWidget(phase_box)
        layout.addStretch(1)
        self._on_analysis_method_changed(0)

    def _add_param_field(self, layout, name, default, tooltips=None):
        label = QtWidgets.QLabel(f"{name}:")
        if tooltips and name in tooltips:
            Tooltip.install(label, tooltips[name])
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(label, le)
        self.params[name] = (le, type(default))
        return le, label

    def _add_phase_field(self, layout, name, default, typ=float, tooltips=None):
        label = QtWidgets.QLabel(f"{name}:")
        if tooltips and name in tooltips:
            Tooltip.install(label, tooltips[name])
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(label, le)
        self.phase_params[name] = (le, typ)
        return le, label

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
        
        
        # Group box for analysis parameters
        param_box = QtWidgets.QGroupBox("Group Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_box)

        # Grid Resolution and Smoothing
        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth to fill empty bins")
        Tooltip.install(self.group_smooth_check, "<b>What it is:</b> A 3x3 circular mean smoothing filter.<br><b>How it works:</b> For each empty bin in the Group Average Map, it calculates the circular mean of its 8 neighbors. If any neighbors have data, the empty bin is filled with that mean value.<br><b>Trade-off:</b> Creates a visually smoother map but interpolates data. The unsmoothed map is a more direct representation of the raw data.")
        param_layout.addRow("Grid Resolution:", self.group_grid_res_edit)
        param_layout.addRow(self.group_smooth_check)

        # Normalization Method Radio Buttons
        norm_label = QtWidgets.QLabel("Phase Normalization Method:")
        self.norm_global_radio = QtWidgets.QRadioButton("Global Mean (per animal)")
        self.norm_anatomical_radio = QtWidgets.QRadioButton("Anatomical Reference ROI")
        self.norm_method_group = QtWidgets.QButtonGroup(self)
        self.norm_method_group.addButton(self.norm_global_radio)
        self.norm_method_group.addButton(self.norm_anatomical_radio)
        self.norm_global_radio.setChecked(True) # Default option
        
        param_layout.addRow(norm_label)
        param_layout.addRow(self.norm_global_radio)
        param_layout.addRow(self.norm_anatomical_radio)

        # Conditional widgets for Anatomical Reference
        self.ref_roi_widget = QtWidgets.QWidget()
        ref_roi_layout = QtWidgets.QHBoxLayout(self.ref_roi_widget)
        ref_roi_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select_ref_roi = QtWidgets.QPushButton("Load Reference ROI...")
        self.ref_roi_path_edit = QtWidgets.QLineEdit()
        self.ref_roi_path_edit.setReadOnly(True)
        ref_roi_layout.addWidget(self.btn_select_ref_roi)
        ref_roi_layout.addWidget(self.ref_roi_path_edit)
        param_layout.addRow(self.ref_roi_widget)
        self.ref_roi_widget.hide() # Hidden by default

        b.addWidget(param_box)        
        
        self.btn_view_group = QtWidgets.QPushButton("Generate Group Visualizations")
        Tooltip.install(self.btn_view_group, "Loads all specified warped ROI and trace files, calculates phases, and generates the Group Scatter and Group Average Map plots.")
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
        self.use_subregion_ref_check.stateChanged.connect(self.regenerate_phase_maps)
        
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
        
        self.norm_anatomical_radio.toggled.connect(self._on_norm_method_changed)
        self.btn_select_ref_roi.clicked.connect(self._select_reference_roi)

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

    def _get_last_dir(self):
        """Retrieves the last used directory from settings."""
        settings = QtCore.QSettings()
        return settings.value("last_dir", "")

    def _set_last_dir(self, path):
        """Saves a directory to settings for future use."""
        if not path:
            return
        settings = QtCore.QSettings()
        # If path is a file, get its directory; otherwise, assume it's a directory
        directory = os.path.dirname(path) if os.path.isfile(path) else path
        settings.setValue("last_dir", directory)

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
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Movie", start_dir, "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if not path:
            self.input_file_edit.clear()
            self.output_base_edit.clear()
            return
        
        self._set_last_dir(path)
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
            traces = np.loadtxt(f"{basename}_traces.csv", delimiter=",")
            roi = np.loadtxt(f"{basename}_roi.csv", delimiter=",")
            traj = np.load(f"{basename}_trajectories.npy")

            # --- Ensure consistent shapes, even for single-cell datasets ---
            if isinstance(roi, np.ndarray) and roi.ndim == 1:
                roi = roi.reshape(1, -1)
            if isinstance(traces, np.ndarray) and traces.ndim == 1:
                traces = traces.reshape(1, -1)

            self.state.unfiltered_data["traces"] = traces
            self.state.unfiltered_data["roi"] = roi
            self.state.unfiltered_data["trajectories"] = traj

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
        self.apply_roi_filter(None, None, None)

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

    def apply_roi_filter(self, indices, rois, phase_ref_rois):
        self.filtered_indices = indices
        
        # Generate Path objects for BOTH anatomical and phase reference ROIs
        # This ensures downstream viewers can use 'path' consistently
        if rois:
            for r in rois:
                if "path_vertices" in r:
                    r["path"] = Path(r["path_vertices"])
        
        if phase_ref_rois:
            for r in phase_ref_rois:
                if "path_vertices" in r:
                    r["path"] = Path(r["path_vertices"])

        self.rois = rois
        self.phase_reference_rois = phase_ref_rois 

        # Enable/disable the checkbox based on whether a phase ref ROI was drawn
        self.use_subregion_ref_check.setEnabled(bool(phase_ref_rois))
        if not phase_ref_rois:
            self.use_subregion_ref_check.setChecked(False)

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
        self.apply_roi_filter(None, None, None)

    def save_parameters(self):
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        self._set_last_dir(path)
        data = {name: le.text() for name, (le, _) in self.params.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        self.log_message(f"Parameters saved to {os.path.basename(path)}")

    def load_parameters(self):
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Parameters", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        self._set_last_dir(path)
        with open(path, "r") as f:
            data = json.load(f)
        for name, value in data.items():
            if name in self.params:
                self.params[name][0].setText(str(value))
        self.log_message(f"Parameters loaded from {os.path.basename(path)}")

    def export_current_plot(self):
        widget = self.vis_tabs.currentWidget()
        viewer = self.visualization_widgets.get(widget)
        if not viewer or not hasattr(viewer, "fig"):
            self.log_message("No active figure to export.")
            return
        fig = viewer.fig
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot", start_dir,
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        if not path:
            return
        self._set_last_dir(path)
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            self.log_message(f"Plot saved to {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"Error saving plot: {e}")

    # ------------ Visualization population ------------

    def on_roi_selected(self, original_index):
        """Callback for when a point is clicked in a spatial plot."""
        self.log_message(f"ROI {original_index + 1} selected.")
        
        # Calculate the Local Index for the Trajectory Inspector.
        # The Inspector displays 'loaded_data', which is the FILTERED subset.
        # We must map the Global Original Index -> Local Filtered Index.
        local_index = original_index
        if self.filtered_indices is not None:
            matches = np.where(self.filtered_indices == original_index)[0]
            if len(matches) > 0:
                local_index = matches[0]
            else:
                # The selected global index is not in the current filter view.
                # This implies a state mismatch, but we should just ignore it for the trajectory viewer.
                local_index = -1

        # Update the trajectory inspector
        traj_viewer = self.visualization_widgets.get(self.traj_tab)
        if traj_viewer and local_index != -1:
            traj_viewer.set_trajectory(local_index) # Pass the LOCAL index
            self.vis_tabs.setCurrentWidget(self.traj_tab)

        # Highlight the point in other relevant viewers
        # (ContrastViewer and HeatmapViewer handle the Global Index -> Local Index conversion internally)
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
        
        # Initialize HeatmapViewer with new signature defaults
        fig_h, _ = add_mpl_to_tab(self.heatmap_tab)
        viewer_h = HeatmapViewer(fig_h, self.state.loaded_data, self.filtered_indices, phases, sort_scores, is_emphasized, rhythm_sort_desc,
                                 period=period, minutes_per_frame=None, reference_phase=None)
        self.visualization_widgets[self.heatmap_tab] = viewer_h
        
        fig_c, _ = add_mpl_to_tab(self.com_tab)
        
        current_rois = []
        if self.rois: 
            current_rois.extend(self.rois)
        if self.phase_reference_rois: 
            current_rois.extend(self.phase_reference_rois)

        viewer_c = ContrastViewer(
            fig_c, 
            fig_c.add_subplot(111), 
            bg, 
            self.state.loaded_data["roi"], 
            self.on_contrast_change, 
            self.on_roi_selected, 
            filtered_indices=self.filtered_indices,
            rois=current_rois
        )
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

        # Build phase_args from GUI fields, excluding non-analytic controls
        phase_args = {
            name: t(w.text())
            for name, (w, t) in self.phase_params.items()
            if w.text() and name not in ("grid_resolution", "rhythm_threshold", "r_squared_threshold")
        }
        if not phase_args.get("minutes_per_frame"):
            raise ValueError("Minutes per frame is required.")

        # Map GUI trend_window_hours -> detrend_window_hours argument
        if "trend_window_hours" in phase_args:
            phase_args["detrend_window_hours"] = phase_args.pop("trend_window_hours")

        # Use FFT on detrended mean signal to discover dominant period
        _, discovered_period, _ = calculate_phases_fft(
            self.state.loaded_data["traces"],
            **phase_args,
        )
        self.discovered_period_edit.setText(f"{discovered_period:.2f}")

        if "FFT" in method:
            phases, period, snr_scores = calculate_phases_fft(
                self.state.loaded_data["traces"],
                **phase_args,
            )
            return phases, period, snr_scores, snr_scores, True

        elif "Cosinor" in method:
            traces = self.state.loaded_data["traces"]
            time_points_hours = traces[:, 0] * (phase_args["minutes_per_frame"] / 60.0)

            phases = []
            p_values = []
            r_squareds = []

            detrend_method = "running_median"
            minutes_per_frame = phase_args["minutes_per_frame"]
            T = traces.shape[0]
            # Use the same trend window hours that FFT is using
            trend_window_hours = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
            median_window_frames = compute_median_window_frames(
                minutes_per_frame,
                trend_window_hours,
                T=T,
            )

            for i in range(1, traces.shape[1]):
                raw_intensity = traces[:, i]
                intensity = preprocess_for_rhythmicity(
                    raw_intensity,
                    method=detrend_method,
                    median_window_frames=median_window_frames,
                )

                result = csn.cosinor_analysis(
                    intensity,
                    time_points_hours,
                    period=discovered_period,
                )

                phases.append(result["acrophase"])
                p_values.append(result["p_value"])
                r_squareds.append(result["r_squared"])

            return (
                np.array(phases),
                discovered_period,
                np.array(r_squareds),
                np.array(p_values),
                True,
            )
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

            # Get extra params for visualization
            mpf = float(self.phase_params["minutes_per_frame"][0].text())
            try:
                trend_win = float(self.phase_params["trend_window_hours"][0].text())
            except:
                trend_win = 36.0

            if "Cosinor" in method:
                r_thresh = float(self.phase_params["r_squared_threshold"][0].text())
                rhythm_mask = (filter_scores <= thresh) & (sort_scores >= r_thresh)
                self.log_message(f"Applying Cosinor filter: p <= {thresh} AND R² >= {r_thresh}")
                # Cosinor returns phases in HOURS
                phases_hours = phases
            else:
                rhythm_mask = filter_scores >= thresh
                self.log_message(f"Applying FFT filter: SNR >= {thresh}")
                # FFT returns phases in RADIANS. Convert to Hours for consistency.
                # Phase (rad) / 2pi * Period = Phase (hours)
                phases_hours = (phases / (2 * np.pi)) * period

            if self.strict_cycle_check.isChecked():
                rhythm_mask = strict_cycle_mask(
                    self.state.loaded_data["traces"],
                    minutes_per_frame=mpf,
                    period_hours=period,
                    base_mask=rhythm_mask,
                    min_cycles=2,
                    trend_window_hours=trend_win,
                )
                self.log_message("Strict cycle filter applied: requiring >= 2 cycles.")

            is_emphasized = self.emphasize_rhythm_check.isChecked()
            
            # We will update the HeatmapViewer AFTER we calculate mean_h below
            
            com_viewer = self.visualization_widgets.get(self.com_tab)
            if com_viewer:
                com_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)
            
            rhythmic_indices_relative = np.where(rhythm_mask)[0]
            self.log_message(f"{len(rhythmic_indices_relative)} cells pass rhythmicity threshold(s).")
            
            if len(rhythmic_indices_relative) == 0:
                for t in [self.phase_tab, self.interp_tab]:
                    fig, canvas = add_mpl_to_tab(t)
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, "No cells passed the rhythmicity filter.", ha='center', va='center')
                    canvas.draw()
                self.log_message("No cells passed rhythmicity filter.")
                
                empty_df = pd.DataFrame()
                fig_p, _ = add_mpl_to_tab(self.phase_tab) 
                viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], empty_df, None, vmin=self.vmin, vmax=self.vmax)
                self.visualization_widgets[self.phase_tab] = viewer_p
                
                # Even if no cells, update Heatmap to clear emphasis/lines if needed
                heatmap_viewer = self.visualization_widgets.get(self.heatmap_tab)
                if heatmap_viewer:
                     heatmap_viewer.update_phase_data(phases_hours, sort_scores, rhythm_mask, sort_desc,
                                                 period=period, minutes_per_frame=mpf, 
                                                 reference_phase=None, trend_window_hours=trend_win)
                return

            mean_h = 0.0
            # Helper to convert hours array to circular mean in hours
            def calc_circ_mean_hours(h_vals, p):
                rads = (h_vals % p) * (2 * np.pi / p)
                m_rad = circmean(rads)
                # Map back to -period/2 to period/2 or 0 to period
                m_h = m_rad * (p / (2 * np.pi))
                return m_h

            if self.use_subregion_ref_check.isChecked() and self.phase_reference_rois:
                self.log_message("Using drawn sub-region as phase reference.")
                
                rhythmic_coords = self.state.loaded_data['roi'][rhythmic_indices_relative]
                
                ref_mask = np.zeros(len(rhythmic_coords), dtype=bool)
                for roi in self.phase_reference_rois:
                    # Use the 'path' object we ensured exists in apply_roi_filter
                    if "path" in roi:
                        path = roi["path"]
                    else:
                        path = Path(roi['path_vertices'])
                    ref_mask |= path.contains_points(rhythmic_coords)
                
                ref_indices_in_rhythmic_array = np.where(ref_mask)[0]
                
                self.log_message(f"  -> Found {len(ref_indices_in_rhythmic_array)} rhythmic cells inside reference ROI.")

                if len(ref_indices_in_rhythmic_array) > 0:
                    ref_phases = phases_hours[rhythmic_indices_relative][ref_indices_in_rhythmic_array]
                    mean_h = calc_circ_mean_hours(ref_phases, period)
                    self.log_message(f"  -> Reference Mean Phase: {mean_h:.2f} hours")
                else:
                    self.log_message("  -> Warning: Reference ROI is empty of rhythmic cells. Falling back to global mean.")
                    mean_h = calc_circ_mean_hours(phases_hours[rhythmic_indices_relative], period)
                    self.log_message(f"  -> Global Mean Phase: {mean_h:.2f} hours")
            else:
                if self.use_subregion_ref_check.isChecked():
                    self.log_message("Sub-region reference selected, but no sub-region is defined. Using global mean.")
                
                mean_h = calc_circ_mean_hours(phases_hours[rhythmic_indices_relative], period)
                self.log_message(f"Global Mean Phase: {mean_h:.2f} hours")
            
            # --- UPDATE HEATMAP VIEWER WITH CALCULATED METRICS ---
            heatmap_viewer = self.visualization_widgets.get(self.heatmap_tab)
            if heatmap_viewer:
                 heatmap_viewer.update_phase_data(phases_hours, sort_scores, rhythm_mask, sort_desc,
                                             period=period, minutes_per_frame=mpf, 
                                             reference_phase=mean_h, trend_window_hours=trend_win)
                 heatmap_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)

            # Calculate relative phases centered on the mean
            rel_phases = (phases_hours[rhythmic_indices_relative] - mean_h + period / 2) % period - period / 2
            
            if self.filtered_indices is not None:
                rhythmic_indices_original = self.filtered_indices[rhythmic_indices_relative]
            else:
                rhythmic_indices_original = rhythmic_indices_relative

            df_data = {
                'Original_ROI_Index': rhythmic_indices_original + 1,
                'X_Position': self.state.loaded_data['roi'][rhythmic_indices_relative, 0],
                'Y_Position': self.state.loaded_data['roi'][rhythmic_indices_relative, 1],
                'Phase_Hours': phases_hours[rhythmic_indices_relative],
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
            import traceback
            self.log_message(traceback.format_exc())
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.phase_tab), False)
            self.vis_tabs.setTabEnabled(self.vis_tabs.indexOf(self.interp_tab), False)

    # ------------ Registration panel actions ------------

    def select_atlas(self):
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Atlas ROI File",
            start_dir,
            "Anatomical ROI files (*_anatomical_roi.json)",
        )
        if not path:
            return
        self._set_last_dir(path)
        self.state.atlas_roi_path = path
        self.atlas_path_edit.setText(path)
        self._update_reg_button_state()

    def add_targets(self):
        start_dir = self._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Target ROI Files",
            start_dir,
            "Anatomical ROI files (*_anatomical_roi.json)",
        )
        if not files:
            return
        self._set_last_dir(files[0])
        for f in files:
            if f not in self.state.target_roi_paths:
                self.state.target_roi_paths.append(f)
                self.target_list.addItem(f)
        self._update_reg_button_state()

    def remove_target(self):
        for item in self.target_list.selectedItems():
            row = self.target_list.row(item)
            path = self.target_list.item(row).text()
            # Directly remove from state object
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

    @QtCore.pyqtSlot(bool)
    def _on_norm_method_changed(self, checked):
        """Shows or hides the reference ROI loader based on radio button selection."""
        if self.norm_anatomical_radio.isChecked():
            self.ref_roi_widget.show()
        else:
            self.ref_roi_widget.hide()
            
    def _select_reference_roi(self):
        """Opens a file dialog to select the anatomical reference ROI."""
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Anatomical Reference ROI File",
            start_dir,
            "JSON files (*.json)",
        )
        if not path:
            return
        self._set_last_dir(path)
        self.state.reference_roi_path = path
        self.ref_roi_path_edit.setText(path)
        self.log_message(f"Loaded anatomical reference ROI: {os.path.basename(path)}")

    # ------------ Apply warp panel actions ------------

    def add_warp_files(self):
        start_dir = self._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Warp Parameter Files",
            start_dir,
            "Warp Parameters (*_warp_parameters.json)",
        )
        if not files:
            return
        self._set_last_dir(files[0])
        for f in files:
            if f not in self.state.warp_param_paths:
                self.state.warp_param_paths.append(f)
                self.warp_list.addItem(f)
        self.check_apply_warp_buttons_state()

    def remove_warp_file(self):
        for item in self.warp_list.selectedItems():
            row = self.warp_list.row(item)
            path = self.warp_list.item(row).text()
            # Directly remove from state object
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
        start_dir = self._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Warped ROI Files",
            start_dir,
            "Warped ROI files (*_roi_warped.csv)",
        )
        if not files:
            return
        self._set_last_dir(files[0])
        for f in files:
            if f not in self.state.group_data_paths:
                self.state.group_data_paths.append(f)
                self.group_list.addItem(f)
        self._update_group_view_button()

    def remove_group_file(self):
        for item in self.group_list.selectedItems():
            row = self.group_list.row(item)
            path = self.group_list.item(row).text()
            # Directly remove from state object
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
            # --- Phase Calculation Setup ---
            valid_fft_keys = ["minutes_per_frame", "period_min", "period_max"]
            phase_args = {}
            for key in valid_fft_keys:
                if key in self.phase_params:
                    widget, type_caster = self.phase_params[key]
                    value_str = widget.text().strip()
                    if value_str: phase_args[key] = type_caster(value_str)
            if "minutes_per_frame" not in phase_args: raise ValueError("Minutes per Frame must be set for group analysis.")
            
            # --- Load Master Reference ROI (if enabled) ---
            master_ref_rois = []
            if self.norm_anatomical_radio.isChecked():
                if not self.state.reference_roi_path or not os.path.exists(self.state.reference_roi_path):
                    QtWidgets.QMessageBox.warning(self, "Error", "Anatomical Reference method selected, but no valid reference ROI file has been loaded.")
                    return
                with open(self.state.reference_roi_path, 'r') as f:
                    master_ref_rois = json.load(f)

            all_dfs = []
            
            # --- Processing Loop ---
            for i, roi_file in enumerate(self.state.group_data_paths):
                self.log_message(f"  [{i + 1}/{len(self.state.group_data_paths)}] {os.path.basename(roi_file)}")
                
                # 1. Identify Files
                traces_file = roi_file.replace("_roi_warped.csv", "_traces.csv")
                unfiltered_roi_file = roi_file.replace("_roi_warped.csv", "_roi.csv")
                filtered_roi_file = roi_file.replace("_roi_warped.csv", "_roi_filtered.csv")
                
                required_files = [traces_file, unfiltered_roi_file, filtered_roi_file, roi_file]
                if not all(os.path.exists(f) for f in required_files):
                    self.log_message("    Warning: missing one or more required files; skipping.")
                    continue
                
                # 2. Load Data
                warped_rois = np.atleast_2d(np.loadtxt(roi_file, delimiter=","))
                # We load the filtered native ROIs to ensure 1:1 index alignment with warped_rois
                native_filtered_rois = np.atleast_2d(np.loadtxt(filtered_roi_file, delimiter=",")) 
                
                if warped_rois.shape[0] != native_filtered_rois.shape[0]:
                    self.log_message(f"    ERROR: Data inconsistency (warped vs filtered count). Skipping.")
                    continue
                
                # 3. Match Traces 
                # We must find which columns in the raw traces file correspond to our filtered cells.
                traces = np.loadtxt(traces_file, delimiter=",")
                unfiltered_rois = np.loadtxt(unfiltered_roi_file, delimiter=",")
                
                indices = []
                # Match native_filtered back to unfiltered to get the original trace indices
                for point in native_filtered_rois:
                    diff = np.sum((unfiltered_rois - point)**2, axis=1)
                    idx = np.argmin(diff)
                    if diff[idx] < 1e-9: indices.append(idx)
                
                if len(indices) != native_filtered_rois.shape[0]:
                    self.log_message("    Warning: Could not match filtered to unfiltered ROIs; skipping.")
                    continue
                
                trace_indices_to_keep = np.array(indices) + 1 # +1 because col 0 is time
                filtered_traces_data = traces[:, np.concatenate(([0], trace_indices_to_keep))]
                
                # 4. Calculate Phases (Raw)
                phases, period, _ = calculate_phases_fft(filtered_traces_data, **phase_args)

                # 5. Determine Reference Mean (Hybrid Logic)
                mean_h = 0.0
                
                if self.norm_global_radio.isChecked():
                     # Method A: Global Mean
                     ph_rad = (phases % period) * (2 * np.pi / period)
                     mean_rad = circmean(ph_rad)
                     mean_h = mean_rad * (period / (2 * np.pi))
                     self.log_message(f"    -> Norm: Global Mean ({mean_h:.2f}h)")

                else:
                    # Method B: Anatomical Reference (Hybrid)
                    
                    # Check for Individual Override (Native Space)
                    individual_roi_path = roi_file.replace("_roi_warped.csv", "_anatomical_roi.json")
                    individual_ref_poly = []
                    
                    if os.path.exists(individual_roi_path):
                        with open(individual_roi_path, 'r') as f:
                            ind_rois = json.load(f)
                            # Look specifically for "Phase Reference" polygons
                            individual_ref_poly = [r for r in ind_rois if r.get("mode") == "Phase Reference"]

                    ref_indices = []
                    
                    if individual_ref_poly:
                        # CASE 1: Use Individual Native ROI
                        self.log_message("    -> Norm: Individual 'Phase Reference' ROI (Native Space)")
                        ref_mask = np.zeros(len(native_filtered_rois), dtype=bool)
                        for roi in individual_ref_poly:
                            path = Path(roi['path_vertices'])
                            ref_mask |= path.contains_points(native_filtered_rois)
                        ref_indices = np.where(ref_mask)[0]
                        
                    else:
                        # CASE 2: Fallback to Master Atlas ROI (Warped Space)
                        self.log_message("    -> Norm: Master Atlas ROI (Warped Space)")
                        # Look specifically for "Phase Reference" polygons in the master atlas
                        master_ref_poly = [r for r in master_ref_rois if r.get("mode") == "Phase Reference"]
                        
                        if not master_ref_poly:
                            self.log_message("    ERROR: No 'Phase Reference' polygon found in Master Atlas file. Cannot normalize.")
                            continue
                            
                        ref_mask = np.zeros(len(warped_rois), dtype=bool)
                        for roi in master_ref_poly:
                            path = Path(roi['path_vertices'])
                            ref_mask |= path.contains_points(warped_rois)
                        ref_indices = np.where(ref_mask)[0]

                    if len(ref_indices) == 0:
                        self.log_message("    Warning: No rhythmic cells found in reference region. Skipping.")
                        continue

                    ref_phases = phases[ref_indices]
                    ref_rad = (ref_phases % period) * (2 * np.pi / period)
                    ref_mean_rad = circmean(ref_rad)
                    mean_h = ref_mean_rad * (period / (2 * np.pi))
                    self.log_message(f"       Ref Mean: {mean_h:.2f}h (n={len(ref_indices)})")

                # 6. Normalize
                rel_phases = (phases - mean_h + period / 2) % period - period / 2

                animal_df = pd.DataFrame({
                    'Source_Animal': os.path.basename(roi_file).replace('_roi_warped.csv', ''),
                    'Warped_X': warped_rois[:, 0],
                    'Warped_Y': warped_rois[:, 1],
                    'Relative_Phase_Hours': rel_phases,
                    'Period_Hours': period
                })
                all_dfs.append(animal_df)

            # --- Downstream Processing (Visualizations) ---
            if not all_dfs: 
                self.log_message("No valid group data generated.")
                return
            
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
    
    # Set application details for QSettings
    QtCore.QCoreApplication.setOrganizationName("HerzogLab")
    QtCore.QCoreApplication.setApplicationName("NeuronAnalysis")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
