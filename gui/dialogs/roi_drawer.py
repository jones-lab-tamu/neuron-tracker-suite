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
        
        # --- Movie Data Processing & Cache Init ---
        self._bg_cache = {}
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
             self.total_frames = self.movie_frames.shape[0]
             self.current_frame_idx = max(1, min((self.total_frames // 2) + 1, self.total_frames))
             self.current_bg_mode = "Median (time)"
        else:
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
        
        # --- Background Controls ---
        bg_group = QtWidgets.QGroupBox("Background")
        bg_layout = QtWidgets.QVBoxLayout(bg_group)
        
        # Mode Selector
        self.bg_mode_combo = QtWidgets.QComboBox()
        modes = ["Single frame", "Mean (time)", "Median (time)", "Std (time)", "Max (time)", "P95 (time)"]
        self.bg_mode_combo.addItems(modes)
        
        if self.movie_frames is not None and not self.is_region_mode:
             modes = ["Single frame", "Mean (time)", "Median (time)", "Std (time)", "Max (time)", "P95 (time)"]
             self.bg_mode_combo.addItems(modes)
             idx = self.bg_mode_combo.findText(self.current_bg_mode)
             if idx >= 0: self.bg_mode_combo.setCurrentIndex(idx)
             self.bg_mode_combo.setEnabled(True)
        else:
             # Static or Region Mode: disable mode selector
             self.bg_mode_combo.addItem("Static")
             self.bg_mode_combo.setCurrentIndex(0)
             self.bg_mode_combo.setEnabled(False)
        
        self.bg_mode_combo.currentIndexChanged.connect(self.update_background)
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
        # Status Label
        if self.is_region_mode:
             state_text = "Static (Region mode)"
        elif self.movie_frames is None:
             state_text = "Static"
        else:
             state_text = self.current_bg_mode
             
        self.lbl_bg_status = QtWidgets.QLabel(f"Background: {state_text}")
        self.lbl_bg_status.setStyleSheet("color: gray; font-size: 10px;")
        bg_layout.addWidget(self.lbl_bg_status)
        
        # If region mode or static, disable entire group except maybe auto-contrast?
        if self.is_region_mode:
             bg_group.setEnabled(False)
        elif self.movie_frames is None:
             # If just static (no movie), combo/slider disabled above.
             pass
        
        ctrl_layout.addWidget(bg_group)

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
        self.update_background()
        
        # Draw initial state
        self.redraw_finished_rois()

    def _on_frame_slider_changed(self, val):
        self.current_frame_idx = val
        self.update_background()

    def compute_background(self, mode):
        if not self.movie_available:
            return self.bg_image
            
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

    def update_background(self):
        mode = self.bg_mode_combo.currentText()
        if not self.movie_available:
            # Static mode
            self.lbl_bg_status.setText("Background: Static")
            self.curr_frame_widget.setEnabled(False)
            img = self.bg_image
        else:
            self.current_bg_mode = mode
            if mode == "Single frame":
                self.curr_frame_widget.setEnabled(True)
                self.lbl_bg_status.setText(f"Background: Single frame (frame {self.current_frame_idx} of {self.total_frames})")
                
                # Log change if parent available
                if self.parent() and hasattr(self.parent(), 'mw'):
                     # Only log if sender is user interaction to avoid spam? Or just log.
                     pass 
            else:
                self.curr_frame_widget.setEnabled(False)
                self.lbl_bg_status.setText(f"Background: {mode}")
            
            img = self.compute_background(mode)
            
            # Log
            if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
                 # Simple deduplication could be added here if needed, but per-click logging is fine
                 pass

        if img is not None:
             self.bg_image = img
             # Update imshow
             artists = self.ax.get_images()
             if artists:
                 im = artists[0]
                 im.set_data(img)
                 # Don't reset vmin/vmax automatically here unless it's first load?
                 # Actually user might want stable contrast when scrubbing. 
                 # Let's keep existing vmin/vmax of the artist unless they are None.
                 # The init passes vmin/vmax.
             else:
                 self.ax.imshow(img, cmap="gray") # Should not happen given init
             
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
                    # Handle Line2D vs Polygon data structures if necessary
                    # But self.rois usually stores raw dicts, so this is fine.
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

    def compute_background(self, mode):
        # Fallback to static checks
        if self.movie_frames is None or self.is_region_mode:
            return self.static_bg_image
            
        if mode == "Single frame":
            # 1-based index conversion
            idx = self.current_frame_idx - 1 
            idx = np.clip(idx, 0, self.total_frames - 1)
            return self.movie_frames[idx]
        
        # Time Summary Modes
        if mode in self._bg_cache:
            return self._bg_cache[mode]
            
        # Compute
        img = None
        if "Mean" in mode:
            img = np.nanmean(self.movie_frames, axis=0)
        elif "Median" in mode:
            img = np.nanmedian(self.movie_frames, axis=0)
        elif "Std" in mode:
            img = np.nanstd(self.movie_frames, axis=0)
        elif "Max" in mode:
            img = np.nanmax(self.movie_frames, axis=0)
        elif "P95" in mode:
            img = np.nanpercentile(self.movie_frames, 95, axis=0)
        
        if img is not None:
            self._bg_cache[mode] = img
            return img
            
        return self.static_bg_image # Fallback

    def update_background(self):
        mode = self.bg_mode_combo.currentText()
        if self.movie_frames is None or self.is_region_mode or mode == "Static":
            # Static mode
            img = self.static_bg_image
            self.curr_frame_widget.setEnabled(False)
            
            if self.is_region_mode:
                 self.lbl_bg_status.setText("Background: Static (Region mode)")
            else:
                 self.lbl_bg_status.setText("Background: Static")
        else:
            self.current_bg_mode = mode
            is_single = (mode == "Single frame")
            
            # Update Frame Widget State
            should_enable_frames = is_single and (self.total_frames > 1)
            self.curr_frame_widget.setEnabled(should_enable_frames)
            
            # Status Label
            if is_single:
                 txt = f"Background: Single frame (frame {self.current_frame_idx} of {self.total_frames})"
            else:
                 txt = f"Background: {mode}"
            self.lbl_bg_status.setText(txt)
            
            img = self.compute_background(mode)
            
            # Dedicated Logging with De-duplication
            new_state = (mode, self.current_frame_idx if is_single else -1)
            if new_state != self._last_bg_state:
                if self.parent() and hasattr(self.parent(), 'mw') and hasattr(self.parent().mw, 'log_message'):
                     msg = f"ROI Background changed: {mode}"
                     if is_single:
                         msg += f" ({self.current_frame_idx}/{self.total_frames})"
                     self.parent().mw.log_message(msg)
                self._last_bg_state = new_state

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
