import json
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from gui.theme import get_icon

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
        close_btn = QtWidgets.QPushButton(get_icon('fa5s.times'), "Close")
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
