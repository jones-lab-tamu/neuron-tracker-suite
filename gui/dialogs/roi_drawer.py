import os
import json
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.lines import Line2D

from gui.theme import get_icon

class RegionAttributesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region Attributes")
        self.layout = QtWidgets.QFormLayout(self)
        
        self.zone_id_spin = QtWidgets.QSpinBox()
        self.zone_id_spin.setRange(1, 20)
        self.layout.addRow("Zone ID (1=Dorsal, etc):", self.zone_id_spin)
        
        self.lobe_combo = QtWidgets.QComboBox()
        self.lobe_combo.addItems(["Left", "Right"])
        self.layout.addRow("Lobe:", self.lobe_combo)
        
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Dorsal Shell")
        self.layout.addRow("Name (Optional):", self.name_edit)
        
        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        self.layout.addWidget(self.btns)

    def get_data(self):
        return {
            "zone_id": self.zone_id_spin.value(),
            "lobe": self.lobe_combo.currentText(),
            "name": self.name_edit.text()
        }


        
def select_projection_frames(n_frames, max_frames=300):
    if n_frames <= max_frames:
        return np.arange(n_frames)
    # Evenly spaced indices (deterministic)
    indices = np.linspace(0, n_frames - 1, max_frames).round().astype(int)
    return np.unique(indices)

def compute_projection(frames, mode, max_frames=300):
    import time
    t0 = time.time()
    n_total = frames.shape[0]
    
    
    if "Median" in mode or "P95" in mode:
        indices = select_projection_frames(n_total, max_frames)
        stack_for_proj = frames[indices].astype(np.float32, copy=False)
        n_used = len(indices)
    else:

        stack_for_proj = frames
        n_used = n_total

    if "Mean" in mode:
        img = np.nanmean(stack_for_proj, axis=0)
    elif "Median" in mode:
        img = np.nanmedian(stack_for_proj, axis=0)
    elif "Std" in mode:
        img = np.nanstd(stack_for_proj, axis=0)
    elif "Max" in mode:
        img = np.nanmax(stack_for_proj, axis=0)
    elif "P95" in mode:
        img = np.nanpercentile(stack_for_proj, 95, axis=0)
    else:
        img = None

    dt = time.time() - t0
    return img, n_used, n_total, dt

class BgComputeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, str, float, int, int, object, int) # (image, mode, seconds, n_used, n_total, dataset_key, max_frames)
    error = QtCore.pyqtSignal(str)

    def __init__(self, frames, mode, max_frames=300, dataset_key=None):
        super().__init__()
        self.frames = frames
        self.mode = mode
        self.max_frames = max_frames
        self.dataset_key = dataset_key

    def run(self):
        try:
            img, n_used, n_total, dt = compute_projection(
                self.frames, self.mode, self.max_frames
            )
            self.finished.emit(img, self.mode, dt, n_used, n_total, self.dataset_key, self.max_frames)
        except Exception as e:
            self.error.emit(str(e))

class ROIDrawerDialog(QtWidgets.QDialog):
    """
    ROI drawing dialog: include/exclude polygons, writes:
      - <basename>_anatomical_roi.json
      - <basename>_roi_filtered.csv
    Calls callback(filtered_indices, rois_dict_list)
    """

    def __init__(self, parent, bg_image, roi_data, output_basename, callback, vmin=None, vmax=None, is_region_mode=False, movie_data=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced ROI Definition Tool")
        self.resize(1100, 800)

        self.bg_image = bg_image
        self.roi_data = roi_data
        self.output_basename = output_basename
        self.callback = callback
        self.is_region_mode = is_region_mode
        self.movie_data = movie_data
        
        # --- Init-Safety Defaults ---

        self.movie_available = False
        self.movie_frames = None
        self.total_frames = 0
        self.current_frame_idx = 1
        self.current_bg_mode = "Static"
        self.bg_image = None
        
        # --- Movie Data Processing & Cache Init ---
        self._bg_cache = {} # Key: (dataset_key, mode, max_frames)
        # Key now derived dynamically from movie_frames
        self._bg_cache = {} # Key: (dataset_key, mode, max_frames)
        # Key now derived dynamically from movie_frames
        self.roi_projection_max_frames = 150
        self._is_computing = False
        self._compute_thread = None
        self._compute_worker = None
        
        self.movie_frames = None
        self.total_frames = 0
        self.current_frame_idx = 1 # 1-based index
        self._last_bg_state = None  # (mode, frame_idx) for log deduplication
        
        self.static_bg_image = bg_image
        self.current_bg_image = bg_image

        # Robust Axis Inference
        valid_movie = False
        if self.movie_data is not None and not self.is_region_mode and self.static_bg_image is not None:
             if self.movie_data.ndim == 3:
                 H, W = self.static_bg_image.shape
                 shape = self.movie_data.shape
                 
                 candidates = []
                 # Check all 3 axes as potential time axis
                 for axis in range(3):
                     # Temporarily move axis to front: (T, Y, X)
                     frames = np.moveaxis(self.movie_data, axis, 0)
                     spatial = frames.shape[1:]
                     
                     if spatial == (H, W):
                         candidates.append(frames) # No transpose needed
                     elif spatial == (W, H):
                         candidates.append(frames.transpose(0, 2, 1)) # Transpose spatial
                 
                 if len(candidates) == 1:
                     self.movie_frames = candidates[0]
                     valid_movie = True
                 elif len(candidates) == 0:
                     warn = f"ROI Drawer Warning: No axis in movie shape {shape} matches background {(H, W)}."
                 else:
                     warn = f"ROI Drawer Warning: Ambiguous axes in movie shape {shape} for background {(H, W)}."
                 
                 if not valid_movie and 'warn' in locals() and parent and hasattr(parent, 'mw'):
                      parent.mw.log_message(warn)
             
        if valid_movie:
             self.movie_available = True
             self.total_frames = self.movie_frames.shape[0]
             self.current_frame_idx = max(1, min((self.total_frames // 2) + 1, self.total_frames))
             
             # Smart Default Mode:
             settings = QtCore.QSettings("NeuronTracker", "ROIDrawer")
             last_mode = settings.value("roi_drawer/bg_mode", "Mean (time)")
             if "Median" in str(last_mode) or "P95" in str(last_mode):
                 last_mode = "Mean (time)"
             
             self.current_bg_mode = last_mode
        else:
             self.movie_available = False
             self.current_bg_mode = "Static"
             self.movie_frames = None

        # Log initial state
        if parent and hasattr(parent, 'mw') and hasattr(parent.mw, 'log_message'):
             state_desc = "Static (Region)" if self.is_region_mode else self.current_bg_mode
             parent.mw.log_message(f"ROI Tool opened. Mode={state_desc}, Frames={self.total_frames}")




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
                    if parent and hasattr(parent, 'mw') and hasattr(parent.mw, 'log_message'):
                         parent.mw.log_message(f"Error loading existing ROIs: {e}")

        # --- Layout Setup ---
        main_layout = QtWidgets.QHBoxLayout(self)

        # LEFT: Plot Area
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        # Store persistent handle
        self.bg_im = self.ax.imshow(bg_image, cmap="gray", vmin=vmin, vmax=vmax)

        if roi_data is not None and len(roi_data) > 0:
            self.ax.plot(roi_data[:, 0], roi_data[:, 1],
                         ".", color="gray", markersize=2, alpha=0.5)

        self.ax.set_title("Click to Define ROI Polygon")
        self.canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        
        main_layout.addWidget(plot_widget, stretch=3)

        # RIGHT: Control Panel
        ctrl_widget = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_widget)
        
        # Instructions
        info = QtWidgets.QLabel(
            "<b>Instructions:</b><br>"
            "1. Select Mode below.<br>"
            "2. Left-click on image to draw.<br>"
            "3. Right-click to remove last point.<br>"
            "4. Click 'Finish Polygon' to save.<br><br>"
            "<b>To Delete:</b> Select from list and click Delete."
        )
        info.setWordWrap(True)
        ctrl_layout.addWidget(info)

        # --- Background Controls ---
        bg_group = QtWidgets.QGroupBox("Background")
        self.bg_group = bg_group
        bg_layout = QtWidgets.QVBoxLayout(bg_group)
        
        # Mode Selector
        self.bg_mode_combo = QtWidgets.QComboBox()
        
        if self.movie_frames is not None and not self.is_region_mode:
             modes = ["Single frame", "Mean (time)", "Median (time)", "Std (time)", "Max (time)", "P95 (time)"]
             self.bg_mode_combo.addItems(modes)
             idx = self.bg_mode_combo.findText(self.current_bg_mode)
             if idx >= 0: 
                 self.bg_mode_combo.blockSignals(True)
                 self.bg_mode_combo.setCurrentIndex(idx)
                 self.bg_mode_combo.blockSignals(False)
             self.bg_mode_combo.setEnabled(True)

        else:
             # Static or Region Mode: disable mode selector
             self.bg_mode_combo.addItem("Static")
             self.bg_mode_combo.blockSignals(True)
             self.bg_mode_combo.setCurrentIndex(0)
             self.bg_mode_combo.blockSignals(False)
             self.bg_mode_combo.setEnabled(False)
        
        self.bg_mode_combo.currentTextChanged.connect(self._on_bg_mode_changed)
        bg_layout.addWidget(QtWidgets.QLabel("Mode:"))
        bg_layout.addWidget(self.bg_mode_combo)
        
        # Frame Control (Slider + Spin)
        self.curr_frame_widget = QtWidgets.QWidget()
        cf_layout = QtWidgets.QHBoxLayout(self.curr_frame_widget)
        cf_layout.setContentsMargins(0,0,0,0)
        
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(1, max(1, self.total_frames))
        self.frame_spin.setValue(self.current_frame_idx)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(1, max(1, self.total_frames))
        self.frame_slider.setValue(self.current_frame_idx)
        
        self.frame_spin.valueChanged.connect(self.frame_slider.setValue)
        self.frame_slider.valueChanged.connect(self.frame_spin.setValue)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        
        cf_layout.addWidget(QtWidgets.QLabel("Frame:"))
        cf_layout.addWidget(self.frame_slider)
        cf_layout.addWidget(self.frame_spin)
        
        # Enable frame widget steps logic
        # Only enable if NOT region mode, HAVE movie, Total > 1, and in Single Frame mode
        should_enable_frames = (not self.is_region_mode) and (self.movie_frames is not None) and (self.total_frames > 1) and (self.current_bg_mode == "Single frame")
        self.curr_frame_widget.setEnabled(should_enable_frames)
        
        bg_layout.addWidget(self.curr_frame_widget)
        
        # Auto Contrast
        self.btn_auto_contrast = QtWidgets.QPushButton("Auto contrast")
        self.btn_auto_contrast.clicked.connect(self.apply_auto_contrast)
        bg_layout.addWidget(self.btn_auto_contrast)
        
        # Status Label
        self.lbl_bg_status = QtWidgets.QLabel("Background: Static")
        self.lbl_bg_status.setStyleSheet("color: gray; font-size: 10px;")
        bg_layout.addWidget(self.lbl_bg_status)
        
        # Initialize UI state
        self._apply_bg_controls_state()
        
        ctrl_layout.addWidget(bg_group)


        # Mode Selection
        mode_box = QtWidgets.QGroupBox("Drawing Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        self.include_btn = QtWidgets.QRadioButton("Include (Green)")
        self.exclude_btn = QtWidgets.QRadioButton("Exclude (Red)")
        self.ref_btn = QtWidgets.QRadioButton("Phase Ref (Blue)")
        self.ref_btn.setStyleSheet("color: blue; font-weight: bold;")
        
        # Phase Axis Button
        self.axis_btn = QtWidgets.QRadioButton("Phase Axis (Arrow)")
        self.axis_btn.setStyleSheet("color: magenta; font-weight: bold;")
        
        self.include_btn.setChecked(True)
        self.include_btn.toggled.connect(self.update_mode)
        self.exclude_btn.toggled.connect(self.update_mode)
        self.ref_btn.toggled.connect(self.update_mode)
        self.axis_btn.toggled.connect(self.update_mode)
        
        mode_layout.addWidget(self.include_btn)
        mode_layout.addWidget(self.exclude_btn)
        mode_layout.addWidget(self.ref_btn)
        mode_layout.addWidget(self.axis_btn)
        ctrl_layout.addWidget(mode_box)

        # Action Buttons
        self.finish_btn = QtWidgets.QPushButton(get_icon('fa5s.check'), "Finish Polygon")
        self.finish_btn.clicked.connect(self.finish_polygon)
        ctrl_layout.addWidget(self.finish_btn)

        # ROI List Management
        list_label = QtWidgets.QLabel("Defined ROIs:")
        ctrl_layout.addWidget(list_label)
        
        self.roi_list_widget = QtWidgets.QListWidget()
        self.roi_list_widget.itemSelectionChanged.connect(self.highlight_selected_roi)
        ctrl_layout.addWidget(self.roi_list_widget)
        
        self.delete_btn = QtWidgets.QPushButton(get_icon('fa5s.trash'), "Delete Selected ROI")
        self.delete_btn.clicked.connect(self.delete_selected_roi)
        ctrl_layout.addWidget(self.delete_btn)
        
        ctrl_layout.addStretch()
        
        # Final Confirm
        confirm_btn = QtWidgets.QPushButton(get_icon('fa5s.save'), "Save & Confirm All")
        confirm_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        confirm_btn.clicked.connect(self.confirm_rois)
        ctrl_layout.addWidget(confirm_btn)

        main_layout.addWidget(ctrl_widget, stretch=1)

        self.cid = self.canvas.mpl_connect("button_press_event", self.on_click)
        
        
        # Initial Background Compute & Render
        # Initial Background Compute & Render (instant static first)
        # Guard against init-order issues
        if hasattr(self, "bg_im"):
            img0 = getattr(self, "static_bg_image", None)
            if img0 is None:
                img0 = getattr(self, "bg_image", None)
            if img0 is not None:
                self.bg_im.set_data(img0)
        if hasattr(self, "lbl_bg_status"):
            self.lbl_bg_status.setText("Background: Static (loading...)")
        self._apply_bg_controls_state()

        # Force current_bg_mode to match UI selection if movie is available
        if self.movie_available and hasattr(self, "bg_mode_combo"):
             self.current_bg_mode = self.bg_mode_combo.currentText()
        else:
             self.current_bg_mode = "Static"

        # Defer ONE background update to the next event loop tick
        QtCore.QTimer.singleShot(0, self.update_background)
        
        # Draw initial state
        self.redraw_finished_rois()

    def closeEvent(self, event):
        # Clean up thread if running
        if self._compute_thread and self._compute_thread.isRunning():
            self._compute_thread.quit()
            self._compute_thread.wait()
        
        # Explicit cleanup
        if self._compute_worker:
            self._compute_worker.deleteLater()
        if self._compute_thread:
            self._compute_thread.deleteLater()
            
        super().closeEvent(event)

    def _get_dataset_key(self):
        """Returns a robust key tuple for the current frame stack."""
        if self.movie_frames is None:
            return None
        # Track identity, shape, and dtype to be safe against in-place modifications or replacements
        return (id(self.movie_frames), getattr(self.movie_frames, "shape", None), getattr(self.movie_frames, "dtype", None))

    def _apply_bg_controls_state(self):
        """
        Robustly enable/disable background controls based on current state.
        Call this whenever state changes or a compute finishes.
        """
        # Guards for init-order safety
        if not hasattr(self, "bg_group") or not hasattr(self, "bg_mode_combo") or \
           not hasattr(self, "curr_frame_widget") or not hasattr(self, "btn_auto_contrast"):
             return

        # A) Region Mode -> Everything disabled
        if self.is_region_mode:
            self.bg_group.setEnabled(False)
            return

        # B) Not region mode -> Group enabled
        self.bg_group.setEnabled(True)

        # B1) No Movie -> Static only
        if self.movie_frames is None:
            self.bg_mode_combo.setEnabled(False)
            self.curr_frame_widget.setEnabled(False)
            self.btn_auto_contrast.setEnabled(True)
            return

        # B2) Computing -> Controls disabled
        if self._is_computing:
            self.bg_mode_combo.setEnabled(False)
            self.curr_frame_widget.setEnabled(False)
            self.btn_auto_contrast.setEnabled(False)
            return

        # B3) Movie Available & Idle
        self.bg_mode_combo.setEnabled(True)
        self.btn_auto_contrast.setEnabled(True)
        
        mode = self.bg_mode_combo.currentText()
        if mode == "Single frame" and self.total_frames > 1:
            self.curr_frame_widget.setEnabled(True)
        else:
            self.curr_frame_widget.setEnabled(False)

    def _on_frame_slider_changed(self, val):
        self.current_frame_idx = val
        self.update_background()

    def compute_background(self, mode):
        if not getattr(self, "movie_available", False) or self.movie_frames is None:
            return getattr(self, "bg_image", None)
            
        if mode == "Single frame":
            # No caching for single frames, dynamic slice
            # existing bg_image might be a fallback if movie not valid, but movie_available checks that
            idx = self.current_frame_idx - 1 # 0-based
            idx = np.clip(idx, 0, self.total_frames - 1)
            return self.movie_data[idx]
        
        # Time Summary Modes
        if mode in self._bg_cache:
            return self._bg_cache[mode]
            
        # Compute
        img = None
        if "Mean" in mode:
            img = np.nanmean(self.movie_data, axis=0)
        elif "Median" in mode:
            img = np.nanmedian(self.movie_data, axis=0)
        elif "Std" in mode:
            img = np.nanstd(self.movie_data, axis=0)
        elif "Max" in mode:
            img = np.nanmax(self.movie_data, axis=0)
        elif "P95" in mode:
            img = np.nanpercentile(self.movie_data, 95, axis=0)
        
        if img is not None:
            self._bg_cache[mode] = img
            return img
            
        return self.bg_image # Fallback

    def _on_bg_mode_changed(self, mode):
        settings = QtCore.QSettings("NeuronTracker", "ROIDrawer")
        settings.setValue("roi_drawer/bg_mode", mode)
        self.current_bg_mode = mode
        self.update_background()

    def update_background(self):
        # Read mode from UI if present, else fallback to current_bg_mode
        mode = getattr(self, "current_bg_mode", "Static")
        if hasattr(self, "bg_mode_combo"):
            mode = self.bg_mode_combo.currentText()
            
        # Defensive check for early calls
        is_avail = getattr(self, "movie_available", False)
        frames = getattr(self, "movie_frames", None)
        
        if (not is_avail) or (frames is None):
            # Static mode (safe even during init)
            if hasattr(self, "lbl_bg_status"):
                self.lbl_bg_status.setText("Background: Static")
            if hasattr(self, "curr_frame_widget"):
                self.curr_frame_widget.setEnabled(False)
            
            # Prefer static_bg_image if available, else bg_image
            img = getattr(self, "static_bg_image", None)
            if img is None:
                img = getattr(self, "bg_image", None)
        else:
            self.current_bg_mode = mode
            
            # NOTE: do NOT write QSettings here. That is handled by _on_bg_mode_changed.
            if mode == "Single frame":
                if hasattr(self, "curr_frame_widget"):
                    self.curr_frame_widget.setEnabled(True)
                if hasattr(self, "lbl_bg_status"):
                    self.lbl_bg_status.setText(f"Background: Single frame (frame {self.current_frame_idx} of {self.total_frames})")
            else:
                if hasattr(self, "curr_frame_widget"):
                    self.curr_frame_widget.setEnabled(False)
                if hasattr(self, "lbl_bg_status"):
                    self.lbl_bg_status.setText(f"Background: {mode}")
            
            img = self.compute_background(mode)
            
        # If we have no image yet, exit cleanly
        if img is None:
             return

        # CRITICAL: matplotlib widgets may not exist yet if update_background fired during init
        if not hasattr(self, "ax") or not hasattr(self, "canvas"):
             # Store it for later, but do not attempt to draw yet
             self.bg_image = img
             return

        # Update stored image and redraw
        self.bg_image = img
        artists = self.ax.get_images()
        if artists:
            artists[0].set_data(img)
        else:
            self.ax.imshow(img, cmap="gray")
        
        self.canvas.draw_idle()

    def apply_auto_contrast(self):
        artists = self.ax.get_images()
        if not artists: return
        im = artists[0]
        data = im.get_array()
        
        if data is None: return
        
        flat = data[np.isfinite(data)]
        if len(flat) == 0: return
        
        vmin = np.percentile(flat, 2)
        vmax = np.percentile(flat, 98)
        
        if vmax <= vmin:
            vmax = vmin + 1e-6
            
        im.set_clim(vmin, vmax)
        self.canvas.draw_idle()
        
        if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
             self.parent().mw.log_message(f"ROI AutoContrast: vmin={vmin:.2f}, vmax={vmax:.2f}")

    def redraw_finished_rois(self):
        for artist in self.finished_artists:
            artist.remove()
        self.finished_artists = []
        
        self.roi_list_widget.blockSignals(True)
        self.roi_list_widget.clear()

        style_map = {
            "Include": {"color": "lime", "linestyle": "-"},
            "Exclude": {"color": "red", "linestyle": "-"},
            "Phase Reference": {"color": "cyan", "linestyle": "--"},
            "Phase Axis": {"color": "magenta", "linestyle": "-", "marker": ">"} 
        }

        for i, roi in enumerate(self.rois):
            mode = roi.get("mode", "Include")
            verts = roi.get("path_vertices", [])
            if not verts:
                continue
            
            style = style_map.get(mode, style_map["Include"])
            
            if mode == "Phase Axis":
                xs, ys = zip(*verts)
                line = Line2D(
                    xs, ys, 
                    color=style["color"], 
                    linestyle=style["linestyle"], 
                    linewidth=2,
                    marker=style.get("marker"),
                    markersize=6,
                    picker=True
                )
                self.ax.add_artist(line)
                self.finished_artists.append(line)
            else:
                poly = Polygon(
                    verts, 
                    closed=True, 
                    fill=False, 
                    edgecolor=style["color"], 
                    linestyle=style["linestyle"], 
                    linewidth=2,
                    picker=True 
                )
                self.ax.add_patch(poly)
                self.finished_artists.append(poly)
            
            # Show Zone/Lobe in list
            label = f"{i+1}. {mode}"
            if "zone_id" in roi:
                label += f" (Zone {roi['zone_id']} {roi['lobe']})"
            item = QtWidgets.QListWidgetItem(label)
            
            self.roi_list_widget.addItem(item)
        
        self.roi_list_widget.blockSignals(False)
        self.canvas.draw_idle()

    def highlight_selected_roi(self):
        """Highlights the ROI selected in the list."""
        selected_rows = [x.row() for x in self.roi_list_widget.selectedIndexes()]
        idx = selected_rows[0] if selected_rows else -1
        
        style_map = {
            "Include": {"color": "lime", "linestyle": "-"},
            "Exclude": {"color": "red", "linestyle": "-"},
            "Phase Reference": {"color": "cyan", "linestyle": "--"},
            "Phase Axis": {"color": "magenta", "linestyle": "-", "marker": ">"}
        }
        
        for i, artist in enumerate(self.finished_artists):
            if i >= len(self.rois): 
                break
                
            roi = self.rois[i]
            mode = roi.get("mode", "Include")
            base_style = style_map.get(mode, style_map["Include"])
            
            # Check if artist is a Line2D (Axis) or Polygon (ROI)
            is_line = isinstance(artist, Line2D)
            
            if i == idx:
                # Highlight style
                if is_line:
                    artist.set_color("yellow")
                    artist.set_linewidth(3.5)
                else:
                    artist.set_edgecolor("yellow")
                    artist.set_linewidth(3.5)
                    
                artist.set_linestyle("-")
                artist.set_zorder(10)
            else:
                # Normal style
                if is_line:
                    artist.set_color(base_style["color"])
                    artist.set_linewidth(2)
                else:
                    artist.set_edgecolor(base_style["color"])
                    artist.set_linewidth(2)
                    
                artist.set_linestyle(base_style["linestyle"])
                artist.set_zorder(1)
                
        self.canvas.draw_idle()

    def delete_selected_roi(self):
        selected_rows = [x.row() for x in self.roi_list_widget.selectedIndexes()]
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Info", "Please select an ROI from the list to delete.")
            return
        
        idx = selected_rows[0]
        
        # Remove from data
        self.rois.pop(idx)
        
        # Update UI
        self.redraw_finished_rois()

    def update_mode(self):
        if self.include_btn.isChecked():
            self.mode = "Include"
        elif self.exclude_btn.isChecked():
            self.mode = "Exclude"
        elif self.ref_btn.isChecked():
            self.mode = "Phase Reference"
        else:
            self.mode = "Phase Axis"
        self.update_plot()

    def update_plot(self):
        if self.current_line is not None:
            for artist in self.current_line:
                artist.remove()
        self.current_line = None

        color_map = {
            "Include": ("g-", "g+"),
            "Exclude": ("r-", "r+"),
            "Phase Reference": ("b-", "b+")
        }
        line_color, point_color = color_map.get(self.mode, ("g-", "g+"))

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
        if event.button == 1: # Left click
            if event.xdata is None or event.ydata is None:
                return
            self.current_vertices.append((float(event.xdata), float(event.ydata)))
            self.update_plot()
        elif event.button == 3: # Right click
            if self.current_vertices:
                self.current_vertices.pop()
                self.update_plot()

    def finish_polygon(self):
        # Logic: If it's a Polygon, we need > 2 points. If it's an Axis (Line), 2 points is enough.
        min_points = 2 if self.mode == "Phase Axis" else 3
        
        if len(self.current_vertices) >= min_points:
            
            # Region Tagging
            attributes = {}
            if self.is_region_mode and self.mode == "Include":
                # Pop up the dialog
                dlg = RegionAttributesDialog(self)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    attributes = dlg.get_data()
                else:
                    # User cancelled, abort polygon creation
                    return 

            # Only close the loop if it is NOT a Phase Axis
            if self.mode != "Phase Axis":
                self.current_vertices.append(self.current_vertices[0]) # Close loop
            
            roi_data = {
                "path_vertices": list(self.current_vertices),
                "mode": self.mode,
            }
            # Merge attributes (Zone ID, Lobe) into the ROI data
            roi_data.update(attributes)
            
            self.rois.append(roi_data)
            
            self.current_vertices = []
            if self.current_line is not None:
                for artist in self.current_line:
                    artist.remove()
                self.current_line = None
            
            self.redraw_finished_rois()
            self.ax.set_title(f"{len(self.rois)} ROI(s) defined.")

    def confirm_rois(self):
        # Separate the ROIs by their purpose
        anatomical_rois = [r for r in self.rois if r["mode"] in ("Include", "Exclude")]
        # Include Phase Axis here so it is passed back to the saver
        phase_ref_rois = [r for r in self.rois if r["mode"] in ("Phase Reference", "Phase Axis")]

        # Save ALL ROIs (Standard Mode only)
        all_rois_to_save = anatomical_rois + phase_ref_rois
        
        if all_rois_to_save and self.output_basename:
            filepath = f"{self.output_basename}_anatomical_roi.json"
            try:
                serializable = []
                for r in all_rois_to_save:

                    serializable.append({
                        "path_vertices": r["path_vertices"], 
                        "mode": r["mode"]
                    })
                    
                with open(filepath, "w") as f:
                    json.dump(serializable, f, indent=4)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving anatomical ROI file:\n{e}")

        # Calculate Filtering (Only if we have cell data)
        filtered_indices = None
        
        # Check if roi_data exists
        if self.roi_data is not None:
            if anatomical_rois:
                final_mask = np.zeros(len(self.roi_data), dtype=bool)
                include_paths = [Path(r["path_vertices"]) for r in anatomical_rois if r["mode"] == "Include"]
                
                if include_paths:
                    for path in include_paths:
                        final_mask |= path.contains_points(self.roi_data)
                else:
                    # If only Excludes exist, start with everything included
                    final_mask[:] = True 

                for roi in anatomical_rois:
                    if roi["mode"] == "Exclude":
                        final_mask &= ~Path(roi["path_vertices"]).contains_points(self.roi_data)

                filtered_indices = np.where(final_mask)[0]

                if self.output_basename:
                    try:
                        np.savetxt(f"{self.output_basename}_roi_filtered.csv", self.roi_data[filtered_indices], delimiter=",")
                        np.savetxt(f"{self.output_basename}_roi_filtered_ids.csv", filtered_indices + 1, header="Original_ROI_Index", comments='', fmt='%d')
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving filtered ROI CSV:\n{e}")

        # Return results via callback
        if self.callback:
            self.callback(filtered_indices, anatomical_rois, phase_ref_rois)

        self.accept()

    def _on_frame_slider_changed(self, val):
        self.current_frame_idx = val
        self.update_background()

    def _on_compute_finished(self, img, mode, dt, n_used, n_total, dataset_key, max_frames):
        self._is_computing = False
        self.unsetCursor()
        
        # Restore controls based on logic
        self._apply_bg_controls_state()
        
        if img is not None:
             # Cache with dataset_key from the worker to avoid race conditions
             cache_key = (dataset_key, mode, max_frames)
             self._bg_cache[cache_key] = img
             
             self.lbl_bg_status.setText(f"Background: {mode} ready ({n_used}/{n_total} frames, {dt:.2f}s)")
             if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
                  self.parent().mw.log_message(f"ROI background ready: {mode}, Used={n_used}/{n_total}, seconds={dt:.2f}")
             
             # Only update display if we are still in that mode
             if self.bg_mode_combo.currentText() == mode:
                  self.current_bg_image = img
                  self.bg_im.set_data(img)
                  self.canvas.draw_idle()
        
        # Cleanup thread
        if self._compute_thread:
             self._compute_thread.quit()
             self._compute_thread.wait()
             self._compute_worker.deleteLater()
             self._compute_thread.deleteLater()
             self._compute_thread = None
             self._compute_worker = None

    def _on_compute_error(self, err_msg):
        self._is_computing = False
        self.unsetCursor()
        
        # Restore controls
        self._apply_bg_controls_state()
        
        mode = self.bg_mode_combo.currentText()
        
        self.lbl_bg_status.setText(f"Background: {mode} failed, using previous view")
        if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
             self.parent().mw.log_message(f"ROI background compute error: {err_msg}")

        if self._compute_thread:
             self._compute_thread.quit()
             self._compute_thread.wait()
             self._compute_worker.deleteLater()
             self._compute_thread.deleteLater()
             self._compute_thread = None
             self._compute_worker = None

    def compute_background_async(self, mode):
        # Launch QThread
        self._is_computing = True
        self.setCursor(QtCore.Qt.WaitCursor)
        self.lbl_bg_status.setText(f"Background: Computing {mode} over T={self.total_frames} (please wait...)")
        
        # Disable controls immediately using centralized logic
        self._apply_bg_controls_state()
        
        if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
             self.parent().mw.log_message(f"Starting async background compute: {mode}")

        self._compute_thread = QtCore.QThread()
        # Compute dataset key NOW to bind it to this specific computation
        dataset_key = self._get_dataset_key()
        self._compute_worker = BgComputeWorker(
            self.movie_frames,
            mode, 
            max_frames=self.roi_projection_max_frames,
            dataset_key=dataset_key
        )
        self._compute_worker.moveToThread(self._compute_thread)
        
        self._compute_thread.started.connect(self._compute_worker.run)
        self._compute_worker.finished.connect(self._on_compute_finished)
        self._compute_worker.error.connect(self._on_compute_error)
        
        self._compute_thread.start()

    def update_background(self):
        # Ignore re-entrant calls if busy
        if getattr(self, '_is_computing', False):
             return

        # Initialize img to safe default
        img = None

        # 1. Validate Mode
        mode = self.bg_mode_combo.currentText()
        
        if self.movie_available:
             settings = QtCore.QSettings("NeuronTracker", "ROIDrawer")
             settings.setValue("roi_drawer/bg_mode", mode)
        
        # 2. Check Static / Region Conditions
        if self.movie_frames is None or self.is_region_mode or mode == "Static":
            img = self.static_bg_image
            
            if self.is_region_mode:
                 self.lbl_bg_status.setText("Background: Static (Region mode)")
            else:
                 self.lbl_bg_status.setText("Background: Static")
                 
            self._apply_bg_controls_state()
        
        elif mode == "Single frame":
            self.current_bg_mode = mode
            
            self.lbl_bg_status.setText(f"Background: Single frame (frame {self.current_frame_idx} of {self.total_frames})")
            
            # Synchronous
            idx = self.current_frame_idx - 1 
            idx = np.clip(idx, 0, self.total_frames - 1)
            img = self.movie_frames[idx]
            
            self._apply_bg_controls_state()
            
        else:
            # Time Summary Modes
            self.current_bg_mode = mode
            
            # Check cache
            cache_key = (self._get_dataset_key(), mode, self.roi_projection_max_frames)
            if cache_key in self._bg_cache:
                img = self._bg_cache[cache_key]
                self.lbl_bg_status.setText(f"Background: {mode} (Cached)")
                self._apply_bg_controls_state()
            else:
                # Need compute
                self._apply_bg_controls_state()
                self.compute_background_async(mode)
                return # Async will update display when done

        if img is not None and self.bg_im is not None:
             self.current_bg_image = img
             self.bg_im.set_data(img)
             self.canvas.draw_idle()

    def apply_auto_contrast(self):
        if self.bg_im is None: return
        data = self.bg_im.get_array()
        
        if data is None: return
        
        flat = data[np.isfinite(data)]
        if len(flat) == 0: return
        
        vmin = np.percentile(flat, 2)
        vmax = np.percentile(flat, 98)
        
        if vmax <= vmin:
            vmax = vmin + 1e-6
            
        self.bg_im.set_clim(vmin, vmax)
        self.canvas.draw_idle()
        
        if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
             self.parent().mw.log_message(f"ROI AutoContrast: vmin={vmin:.2f}, vmax={vmax:.2f}")
