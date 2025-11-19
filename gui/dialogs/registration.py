import os
import json
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from skimage.transform import ThinPlateSplineTransform

from gui.theme import get_icon

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
        self.calc_btn = QtWidgets.QPushButton(get_icon('fa5s.calculator'), "Calculate Warp")
        self.calc_btn.clicked.connect(self.calculate_warp)
        btn_layout.addWidget(self.calc_btn)

        self.save_btn = QtWidgets.QPushButton(get_icon('fa5s.save'), "Save Warp & Next")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_and_next)
        btn_layout.addWidget(self.save_btn)

        reset_btn = QtWidgets.QPushButton(get_icon('fa5s.undo'), "Reset Landmarks")
        reset_btn.clicked.connect(self.reset_landmarks)
        btn_layout.addWidget(reset_btn)

        close_btn = QtWidgets.QPushButton(get_icon('fa5s.times'), "Close")
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
