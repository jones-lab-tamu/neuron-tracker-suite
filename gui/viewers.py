import numpy as np
import pandas as pd
import colorcet as cet
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from numpy import pi, arctan2
from scipy.stats import circmean, circstd, linregress, sem

from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cosinor as csn
from scipy.ndimage import label as nd_label, gaussian_laplace, gaussian_filter, binary_fill_holes
from gui.analysis import compute_median_window_frames, preprocess_for_rhythmicity

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
        
        self.current_selected_index = None 
        
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
            try:
                main_win = self.fig.canvas.parent().window()
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
        
        if period is not None: self.period = period
        if minutes_per_frame is not None: self.minutes_per_frame = minutes_per_frame
        self.reference_phase = reference_phase 
        if trend_window_hours is not None: self.trend_window_hours = trend_window_hours

        self.sort_values = {"Y-coordinate": self.roi_data[:, 1]}
        if self.phases is not None: self.sort_values["Phase"] = self.phases
        if self.rhythm_scores is not None: self.sort_values["Rhythmicity"] = self.rhythm_scores
        
        current_sort = self.radio_buttons.value_selected
        if current_sort in self.sort_values: self.on_sort_change(current_sort)

        if self.current_selected_index is not None:
            self.update_selected_trace(self.current_selected_index)

    def update_rhythm_emphasis(self, rhythm_mask, is_emphasized):
        self.is_emphasized, self.rhythm_mask = is_emphasized, rhythm_mask
        self.on_sort_change(self.radio_buttons.value_selected)

    def update_selected_trace(self, original_index):
        self.current_selected_index = original_index

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
            time_frames = self.traces_data[:, 0]
            raw_intensity = self.traces_data[:, current_index + 1]
            
            mpf = self.minutes_per_frame or 15.0
            trend_win = self.trend_window_hours or 36.0
            win_frames = compute_median_window_frames(mpf, trend_win, len(raw_intensity))
            detrended = preprocess_for_rhythmicity(raw_intensity, method="running_median", median_window_frames=win_frames)
            
            self.trace_line.set_data(time_frames, detrended)
            
            title_text = f"Trace for ROI {original_index + 1}"
            
            if self.period and self.minutes_per_frame:
                time_hours = time_frames * (self.minutes_per_frame / 60.0)
                res = csn.cosinor_analysis(detrended, time_hours, self.period)
                
                if not np.isnan(res['amplitude']):
                    w = 2 * np.pi / self.period
                    model = res['mesor'] + res['amplitude'] * np.cos(w * (time_hours - res['acrophase']))
                    self.fit_line.set_data(time_frames, model)
                    
                    cell_phase_hours = res['acrophase']
                    period_frames = self.period / (self.minutes_per_frame / 60.0)
                    cell_peak_frame_base = (cell_phase_hours / (self.minutes_per_frame / 60.0))
                    
                    ref_text = ""
                    final_peak_frame = cell_peak_frame_base
                    
                    if self.reference_phase is not None:
                        ref_phase_frame = (self.reference_phase / (self.minutes_per_frame / 60.0))
                        self.ax_trace.axvline(ref_phase_frame, color='k', linestyle='--', alpha=0.8, label='Ref Peak')
                        
                        delta = ref_phase_frame - cell_peak_frame_base
                        cycles_shift = round(delta / period_frames)
                        candidate_frame = cell_peak_frame_base + (cycles_shift * period_frames)
                        
                        max_frame = time_frames[-1]
                        if 0 <= candidate_frame <= max_frame:
                            final_peak_frame = candidate_frame
                        else:
                            if 0 <= cell_peak_frame_base <= max_frame:
                                final_peak_frame = cell_peak_frame_base
                            else:
                                if cell_peak_frame_base < 0:
                                    final_peak_frame = cell_peak_frame_base + period_frames
                        
                        diff = (cell_phase_hours - self.reference_phase + self.period/2) % self.period - self.period/2
                        sign = "+" if diff > 0 else ""
                        ref_text = f" | Ref: {self.reference_phase:.1f}h | Δ: {sign}{diff:.1f}h"
                    
                    self.ax_trace.axvline(final_peak_frame, color='r', linestyle='-', alpha=0.8, label='Cell Peak')
                    title_text += f" | Phase: {cell_phase_hours:.1f}h{ref_text}"
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
        self.rois = rois or []
        self.highlight_artist = None
        self.scatter_artists = []

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        self.image_artist = ax.imshow(bg_image, cmap="gray")
        ax.set_title("Center of Mass (Click to Select Trajectory)")

        ax_contrast = fig.add_axes([0.25, 0.10, 0.60, 0.03])
        ax_brightness = fig.add_axes([0.25, 0.05, 0.60, 0.03])
        
        min_val, max_val = float(bg_image.min()), float(bg_image.max())
        self.contrast_slider = Slider(ax=ax_contrast, label="Contrast", valmin=min_val, valmax=max_val, valinit=max_val)
        self.brightness_slider = Slider(ax=ax_brightness, label="Brightness", valmin=min_val, valmax=max_val, valinit=min_val)
        self.contrast_slider.on_changed(self.update)
        self.brightness_slider.on_changed(self.update)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update(None)
        
        self._draw_rois()    
        self._draw_scatter() 

    def _draw_rois(self):
        if not self.rois:
            return

        style_map = {
            "Include": {"color": "lime", "linestyle": "-", "linewidth": 1.5},
            "Exclude": {"color": "red", "linestyle": "-", "linewidth": 1.5},
            "Phase Reference": {"color": "cyan", "linestyle": "--", "linewidth": 1.5},
            "Phase Axis": {"color": "magenta", "linestyle": "-", "linewidth": 2.0}
        }

        for roi in self.rois:
            mode = roi.get("mode", "Include")
            verts = roi.get("path_vertices", [])
            if not verts:
                continue
            
            style = style_map.get(mode, style_map["Include"])
            
            # Draw Line instead of Polygon for Axis
            if mode == "Phase Axis":
                xs, ys = zip(*verts)
                self.ax.plot(xs, ys, **style, label=mode)
            else:
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
        selected_index = np.argmin(distances) 

        if distances[selected_index] < 20:
            if self.filtered_indices is not None:
                original_index = self.filtered_indices[selected_index]
            else:
                original_index = selected_index
            
            self.highlight_point(original_index)

            if self.on_select_callback:
                self.on_select_callback(original_index)

    def highlight_point(self, index):
        if self.highlight_artist:
            self.highlight_artist.remove()
            self.highlight_artist = None
        
        local_index = None
        if index is not None:
            if self.filtered_indices is not None:
                matches = np.where(self.filtered_indices == index)[0]
                if len(matches) > 0:
                    local_index = matches[0]
                else:
                    local_index = None
            else:
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

def _compute_cellness_features_for_patch_view(patch, center_rc, log_sigma, thr_k, inner_r, annulus_r0, annulus_r1,
                                            area_min=6, area_max=500, ecc_max=0.85, ratio_min=1.1, logz_min=2.0):
    """
    Compute per-frame features for display in the patch view.
    Returns a dict of features + labelled image/mask info.
    """
    features = {
        "logz": 0.0, "area": 0.0, "ecc": 1.0, "center_annulus_ratio": 0.0,
        "blob_ok": False, "thr": 0.0, "thr_k": thr_k,
        "labeled": None, "target_label": 0
    }
    
    h, w = patch.shape
    patch_float = patch.astype(float)
    
    # 1. LoG z-score (Peak vs Dist)
    peak_im = -gaussian_laplace(patch_float, sigma=log_sigma)
    center_val = peak_im[center_rc]
    med_peak = np.median(peak_im)
    mad_peak = np.median(np.abs(peak_im - med_peak))
    features["logz"] = (center_val - med_peak) / (1.4826 * mad_peak + 1e-12)
    
    # 2. Thresholding & Blob Analysis
    mean_val = np.mean(patch_float)
    std_val = np.std(patch_float)
    features["thr"] = mean_val + thr_k * std_val
    
    mask = patch_float > features["thr"]
    labeled, _ = nd_label(mask)
    target_lbl = labeled[center_rc]
    features["labeled"] = labeled
    features["target_label"] = target_lbl
    
    if target_lbl > 0:
        # Area
        coords = np.argwhere(labeled == target_lbl)
        area = len(coords)
        features["area"] = area
        
        # Eccentricity
        if area <= 2:
            features["ecc"] = 1.0
        else:
            cov = np.cov(coords.T)
            # handle singular cases
            try:
                eigvals = np.linalg.eigvalsh(cov)
                l2 = float(eigvals[0])
                l1 = float(eigvals[-1]) # max
                if l1 > 0:
                    features["ecc"] = np.sqrt(max(0.0, 1 - (l2 / (l1 + 1e-12))))
                else:
                    features["ecc"] = 1.0
            except:
                features["ecc"] = 1.0
    else:
        features["area"] = 0.0
        features["ecc"] = 1.0

    # 3. Center/Annulus Ratio
    # Simple grid based distance
    y, x = np.ogrid[:h, :w]
    cy, cx = center_rc
    dists = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    inner_mask = dists <= inner_r
    annulus_mask = (dists >= annulus_r0) & (dists <= annulus_r1)
    
    inner_mean = np.mean(patch_float[inner_mask]) if np.any(inner_mask) else 0.0
    annulus_mean = np.mean(patch_float[annulus_mask]) if np.any(annulus_mask) else 0.0
    
    features["center_annulus_ratio"] = inner_mean / (annulus_mean + 1e-12)

    # 4. Blob OK Logic
    area_ok = area_min <= features["area"] <= area_max
    ecc_ok = features["ecc"] <= ecc_max
    ratio_ok = features["center_annulus_ratio"] >= ratio_min
    logz_ok = features["logz"] >= logz_min
    
    features["blob_ok"] = area_ok and ecc_ok and ratio_ok and logz_ok
    
    return features


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
        
        # Marker Settings Defaults
        self.marker_style = "Ring" # Ring, Crosshair, Dot, Off
        self.marker_alpha = 0.45
        self.marker_radius = 8
        self.marker_lw = 1
        
        # Patch Settings Defaults
        self.patch_size = 31
        self.show_footprint = False
        self.footprint_k = 1.0
        
        # Cellness Feature Defaults (Display Only)
        self.cellness_log_sigma = 1.5
        self.cellness_area_min = 6
        self.cellness_area_max = 500
        self.cellness_ecc_max = 0.85
        self.cellness_ratio_min = 1.10
        self.cellness_logz_min = 2.0
        self.cellness_inner_r = 3
        self.cellness_annulus_r0 = 6
        self.cellness_annulus_r1 = 10

        # Layout: Increase bottom margin for controls and right margin for inset
        self.fig.subplots_adjust(left=0.05, bottom=0.35, right=0.80, top=0.9)

        # Map to store tooltip text for controls
        self.controls_tooltips = {}

        # --- CONTROLS ---
        # 1. Transport (Frame Slider & Nav) - Row 1 (Top of controls)
        ax_slider = fig.add_axes([0.10, 0.25, 0.50, 0.03])
        self.frame_slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=self.num_frames - 1, valinit=0, valstep=1)
        self.frame_slider.on_changed(self.on_frame_change)
        self.controls_tooltips[ax_slider] = "Frame: Select which movie frame is displayed for the currently inspected trajectory."
        
        ax_prev = fig.add_axes([0.62, 0.25, 0.08, 0.04])
        from matplotlib.widgets import Button
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_prev.on_clicked(self.prev_trajectory)
        self.controls_tooltips[ax_prev] = "Previous: Switch to the previous trajectory in the list."
        
        ax_next = fig.add_axes([0.71, 0.25, 0.08, 0.04])
        self.btn_next = Button(ax_next, "Next >")
        self.btn_next.on_clicked(self.next_trajectory)
        self.controls_tooltips[ax_next] = "Next: Switch to the next trajectory in the list."

        # 2. Marker Controls - Row 2
        # Style (Cycle Button)
        ax_style = fig.add_axes([0.05, 0.15, 0.12, 0.04])
        self.btn_style = Button(ax_style, f"Marker: {self.marker_style}")
        self.btn_style.on_clicked(self.cycle_marker_style)
        self.controls_tooltips[ax_style] = "Marker: Controls how the tracked position is drawn on the full-frame view."

        # Alpha
        ax_alpha = fig.add_axes([0.22, 0.15, 0.20, 0.03])
        self.slider_alpha = Slider(ax=ax_alpha, label='Alpha', valmin=0.05, valmax=1.0, valinit=self.marker_alpha)
        self.slider_alpha.on_changed(self.on_marker_param_change)
        self.slider_alpha.label.set_size(9)
        self.controls_tooltips[ax_alpha] = "Alpha: Marker transparency. Lower values occlude less."

        # Radius
        ax_radius = fig.add_axes([0.48, 0.15, 0.20, 0.03])
        self.slider_radius = Slider(ax=ax_radius, label='Radius', valmin=2, valmax=30, valinit=self.marker_radius, valstep=1)
        self.slider_radius.on_changed(self.on_marker_param_change)
        self.slider_radius.label.set_size(9)
        self.controls_tooltips[ax_radius] = "Radius: Marker size in pixels (data coordinates)."
        
        # LineWidth
        ax_lw = fig.add_axes([0.74, 0.15, 0.15, 0.03])
        self.slider_lw = Slider(ax=ax_lw, label='LW', valmin=1, valmax=4, valinit=self.marker_lw, valstep=1)
        self.slider_lw.on_changed(self.on_marker_param_change)
        self.slider_lw.label.set_size(9)
        self.controls_tooltips[ax_lw] = "LineWidth: Marker outline thickness in pixels."

        # 3. Patch Controls - Row 3
        # Patch Size
        ax_psize = fig.add_axes([0.15, 0.05, 0.25, 0.03])
        self.slider_psize = Slider(ax=ax_psize, label='Patch Size', valmin=11, valmax=101, valinit=self.patch_size, valstep=2)
        self.slider_psize.on_changed(self.on_patch_param_change)
        self.controls_tooltips[ax_psize] = "Patch: Size of the zoomed patch centered on the tracked position (odd number of pixels)."

        # Footprint Checkbox
        ax_chk = fig.add_axes([0.45, 0.05, 0.15, 0.04])
        self.chk_footprint = CheckButtons(ax_chk, ["Footprint"], [self.show_footprint])
        self.chk_footprint.on_clicked(self.toggle_footprint)
        self.controls_tooltips[ax_chk] = "Show footprint: Overlay a threshold-based outline of the center-connected blob in the patch (display only)."

        # k Factor
        ax_k = fig.add_axes([0.65, 0.05, 0.20, 0.03])
        self.slider_k = Slider(ax=ax_k, label='k (Thr)', valmin=0.0, valmax=5.0, valinit=self.footprint_k, valstep=0.1)
        self.slider_k.on_changed(self.on_patch_param_change)
        self.controls_tooltips[ax_k] = "k: Threshold = mean + k·std for the footprint overlay (display only)."

        # --- INSET AXES ---
        # Right side panel for Cell Patch
        self.ax_patch = self.fig.add_axes([0.82, 0.45, 0.15, 0.15]) # [left, bottom, width, height]
        self.ax_patch.set_title("Cell Patch", fontsize=9)
        self.ax_patch.set_xticks([])
        self.ax_patch.set_yticks([])
        
        # Metrics Panel (Below Patch)
        self.ax_patch_metrics = self.fig.add_axes([0.82, 0.25, 0.15, 0.18])
        self.ax_patch_metrics.set_xticks([]); self.ax_patch_metrics.set_yticks([])
        self.ax_patch_metrics.axis('off')
        
        # Tooltip Hover Event
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        
        self.update()
    
    def _on_hover(self, event):
        """Implement real Qt tooltips by checking which axes the mouse is over."""
        msg = ""
        if event.inaxes and event.inaxes in self.controls_tooltips:
            msg = self.controls_tooltips[event.inaxes]
        
        # Access the Qt Widget (FigureCanvasQTAgg inherits from QWidget)
        try:
            self.fig.canvas.setToolTip(msg)
        except Exception:
            pass # Fallback if backend is weird, but standard is fine.

    def on_frame_change(self, val):
        self.update()

    def on_marker_param_change(self, val):
        self.marker_alpha = self.slider_alpha.val
        self.marker_radius = int(self.slider_radius.val)
        self.marker_lw = self.slider_lw.val
        self.update()

    def cycle_marker_style(self, event):
        modes = ["Ring", "Crosshair", "Dot", "Off"]
        try:
            curr_idx = modes.index(self.marker_style)
            next_idx = (curr_idx + 1) % len(modes)
        except ValueError:
            next_idx = 0
        self.marker_style = modes[next_idx]
        self.btn_style.label.set_text(f"Marker: {self.marker_style}")
        self.update()

    def on_patch_param_change(self, val):
        self.patch_size = int(self.slider_psize.val)
        # Ensure odd
        if self.patch_size % 2 == 0: self.patch_size += 1
        self.footprint_k = self.slider_k.val
        self.update()

    def toggle_footprint(self, label):
        self.show_footprint = not self.show_footprint
        self.update()

    def set_trajectory(self, index):
        if 0 <= index < self.num_trajectories:
            self.index = index
            self.update()

    def _lw_px_to_points(self, lw_px: float) -> float:
        # Convert pixel thickness to points using figure DPI
        # points = pixels * 72 / dpi
        dpi = float(self.fig.dpi) if getattr(self.fig, "dpi", None) else 100.0
        return float(lw_px) * 72.0 / dpi

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
        self.ax_patch.clear()
        self.ax_patch_metrics.clear()
        self.ax_patch_metrics.axis('off')
        
        self.ax_patch.set_xticks([])
        self.ax_patch.set_yticks([]) 
        self.ax_patch.set_facecolor("black")
        
        current_frame = int(self.frame_slider.val)
        
        # --- MAIN VIEW ---
        self.bg_artist = self.ax.imshow(
            self.movie_stack[current_frame],
            cmap="gray",
            vmin=self.vmin,
            vmax=self.vmax,
        )
        
        if self.num_trajectories > 0:
            traj = self.trajectories[self.index]
            # Draw Path
            self.ax.plot(
                traj[:, 1], traj[:, 0], '-', color='cyan', linewidth=1, alpha=0.5
            )
            
            # Draw Marker (Pixel-True Geometry)
            current_pos = traj[current_frame] # [y, x]
            y, x = current_pos[0], current_pos[1]
            
            if self.marker_style != "Off":
                alpha = self.marker_alpha
                rad = self.marker_radius
                lw = self.marker_lw
                lw_pts = self._lw_px_to_points(lw)
                
                if self.marker_style == "Ring":
                    # Circle patch is in data coordinates (pixels)
                    c = Circle((x, y), radius=rad, fill=False, edgecolor='yellow', linewidth=lw_pts, alpha=alpha)
                    self.ax.add_patch(c)
                elif self.marker_style == "Crosshair":
                    # Plot lines in data coordinates
                    L = max(3, rad)
                    self.ax.plot([x - L, x + L], [y, y], '-', color='yellow', linewidth=lw_pts, alpha=alpha)
                    self.ax.plot([x, x], [y - L, y + L], '-', color='yellow', linewidth=lw_pts, alpha=alpha)
                elif self.marker_style == "Dot":
                    # Small circle for Dot
                    c = Circle((x, y), radius=1.0, color='yellow', alpha=alpha)
                    self.ax.add_patch(c)

            self.ax.set_title(
                f"Trajectory {self.index + 1} / {self.num_trajectories} (Frame {current_frame + 1}/{self.num_frames})"
            )
            
            # --- PATCH VIEW ---
            # Extract Patch
            img = self.movie_stack[current_frame]
            h, w = img.shape
            half = self.patch_size // 2
            iy, ix = int(round(y)), int(round(x))
            
            y0, y1 = iy - half, iy + half + 1
            x0, x1 = ix - half, ix + half + 1
            
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                self.ax_patch.text(0.5, 0.5, "Out of Bounds", ha='center', va='center', color='red', transform=self.ax_patch.transAxes)
                self.ax_patch.set_facecolor('black')
                self.ax_patch_metrics.text(0.5, 0.5, "No metrics\n(out of bounds)", ha='center', va='center', transform=self.ax_patch_metrics.transAxes, fontsize=8)
            else:
                patch = img[y0:y1, x0:x1]
                # Contrast for Patch (Robust Percentile)
                p_vmin, p_vmax = np.percentile(patch, 5), np.percentile(patch, 99)
                if p_vmax <= p_vmin: p_vmax = p_vmin + 1
                
                self.ax_patch.imshow(patch, cmap="gray", vmin=p_vmin, vmax=p_vmax, origin='upper', interpolation='nearest')
                
                # Crosshair in patch (fixed center)
                cx, cy = half, half
                self.ax_patch.plot(cx, cy, '+', color='cyan', alpha=0.5, markersize=8)
                
                # Compute Per-Frame Features
                feats = _compute_cellness_features_for_patch_view(
                    patch, center_rc=(cy, cx),
                    log_sigma=self.cellness_log_sigma,
                    thr_k=self.footprint_k,
                    inner_r=self.cellness_inner_r,
                    annulus_r0=self.cellness_annulus_r0,
                    annulus_r1=self.cellness_annulus_r1,
                    area_min=self.cellness_area_min,
                    area_max=self.cellness_area_max,
                    ecc_max=self.cellness_ecc_max,
                    ratio_min=self.cellness_ratio_min,
                    logz_min=self.cellness_logz_min
                )
                
                # Display Text Overlay
                txt = (
                    f"LoGz: {feats['logz']:.2f}\n"
                    f"Area: {feats['area']:.0f}\n"
                    f"Ecc: {feats['ecc']:.2f}\n"
                    f"C/A: {feats['center_annulus_ratio']:.2f}\n"
                    f"Thr: {feats['thr']:.2f}\n"
                    f"OK: {feats['blob_ok']}"
                )
                
                # Display in separated metrics panel
                self.ax_patch_metrics.text(0.02, 0.98, txt, transform=self.ax_patch_metrics.transAxes,
                                   fontsize=8, verticalalignment='top', horizontalalignment='left')
                
                # Footprint Overlay
                if self.show_footprint:
                    # Reuse valid calculation if available
                    lbl = feats['labeled']
                    target_label = feats['target_label']
                    
                    if target_label > 0:
                         # Draw contour
                        self.ax_patch.contour(lbl == target_label, levels=[0.5], colors='lime', linewidths=1, alpha=0.7)

            self.ax_patch.set_title(f"Patch {self.patch_size}x{self.patch_size}", fontsize=8, color='white')

        else:
            self.ax.set_title("No Trajectories to Display")
            self.ax_patch.text(0.5, 0.5, "No Data", ha='center', va='center', transform=self.ax_patch.transAxes)
            self.ax_patch_metrics.text(0.5, 0.5, "No Data", ha='center', va='center', transform=self.ax_patch_metrics.transAxes)

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
        
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        self.bg_artist = ax.imshow(bg_image, cmap="gray", vmin=vmin, vmax=vmax)
        
        if not self.rhythmic_df.empty:
            self.scatter = ax.scatter(
                self.rhythmic_df['X_Position'], self.rhythmic_df['Y_Position'],
                c=self.rhythmic_df['Relative_Phase_Hours'],
                cmap=self.cmap_cyclic, s=25, edgecolor="black", linewidth=0.5,
            )
            cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
            self.cbar = fig.colorbar(self.scatter, cax=cax)
            self.cbar.set_label("Relative Peak Time (hours)", fontsize=10)
            
            ax_slider = fig.add_axes([0.25, 0.10, 0.60, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim)
            
            ax_button = fig.add_axes([0.25, 0.05, 0.15, 0.04])
            self.cmap_btn = Button(ax_button, "Mode: Cyclic")
            self.cmap_btn.on_clicked(self.toggle_cmap)
            
            self.update_clim(max_range)

        ax.set_title("Spatiotemporal Phase Map (Click to Select Trajectory)")
        ax.set_xticks([]); ax.set_yticks([])
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.scatter.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or self.rhythmic_df.empty: return
        distances = np.sqrt((self.roi_data[:, 0] - event.xdata)**2 + (self.roi_data[:, 1] - event.ydata)**2)
        selected_index = np.argmin(distances) 
        
        if distances[selected_index] < 20:
            global_index = int(self.rhythmic_df.iloc[selected_index]['Original_ROI_Index'] - 1)
            self.highlight_point(global_index)
            if self.on_select_callback: self.on_select_callback(selected_index)

    def highlight_point(self, original_index):
        if self.highlight_artist: self.highlight_artist.remove(); self.highlight_artist = None
        
        if original_index is not None and not self.rhythmic_df.empty:
            matches = self.rhythmic_df.index[self.rhythmic_df['Original_ROI_Index'] == (original_index + 1)].tolist()
            if matches:
                row_idx = matches[0]
                point = self.roi_data[row_idx]
                self.highlight_artist = self.ax.plot(point[0], point[1], 'o', markersize=15, markerfacecolor='none', markeredgecolor='white', markeredgewidth=2)[0]
                
        self.fig.canvas.draw_idle()

    def update_contrast(self, vmin, vmax):
        if self.bg_artist is not None: self.bg_artist.set_clim(vmin, vmax); self.fig.canvas.draw_idle()

    def update_clim(self, val):
        if hasattr(self, "scatter"): self.scatter.set_clim(-val, val); self.fig.canvas.draw_idle()

    def get_export_data(self):
        return self.rhythmic_df, "phase_map_data.csv"
        
class GroupScatterViewer:
    def __init__(self, fig, ax, group_df, grid_bins=None):
        self.fig = fig
        self.ax = ax
        self.full_group_df = group_df # Store all data
        self.grid_bins = grid_bins
        self.period_hours = self.full_group_df['Period_Hours'].iloc[0] if not self.full_group_df.empty else 24
        
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)
        ax.set_title("Group Phase Distribution")

        # Radio Buttons for Group Filter (Positioned Middle Bottom)
        ax_radio = fig.add_axes([0.4, 0.05, 0.2, 0.1])
        self.group_radio = RadioButtons(ax_radio, ['All', 'Control', 'Experiment'])
        self.group_radio.on_clicked(self.update_filter)

        # Scatter setup (initially empty)
        self.scatter = ax.scatter([], [], c=[], cmap=self.cmap_cyclic, s=25, edgecolor="black", linewidth=0.5, alpha=1.0, zorder=10)
        
        cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
        self.cbar = fig.colorbar(self.scatter, cax=cax)
        self.cbar.set_label("Relative Peak Time (Circadian Hours, CT)")
        
        ax_slider = fig.add_axes([0.25, 0.15, 0.60, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label="Phase Range", valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim)
        
        ax_button = fig.add_axes([0.10, 0.05, 0.15, 0.04])
        self.cmap_btn = Button(ax_button, "Mode: Cyclic")
        self.cmap_btn.on_clicked(self.toggle_cmap)

        # Draw Grid
        if not self.full_group_df.empty:
            if grid_bins is not None:
                xbins, ybins = grid_bins
                grid_style = {'color': '#999999', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.4, 'zorder': 0}
                for x in xbins: ax.axvline(x, **grid_style)
                for y in ybins: ax.axhline(y, **grid_style)
            
            # Set Limits based on FULL data
            xs, ys = self.full_group_df['Warped_X'], self.full_group_df['Warped_Y']
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            max_range = max(xs.max() - xs.min(), ys.max() - ys.min())
            half = (max_range * 1.15) / 2 
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

        self.annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w", alpha=0.9), arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        # Initial Plot
        self.update_filter('All')

    def update_filter(self, label):
        if self.full_group_df.empty: return
        
        if label == 'All':
            self.current_df = self.full_group_df
        else:
            self.current_df = self.full_group_df[self.full_group_df['Group'] == label]
            
        # Update Scatter Data
        # Note: set_offsets takes (N, 2), set_array takes (N,)
        self.scatter.set_offsets(self.current_df[['Warped_X', 'Warped_Y']].values)
        self.scatter.set_array(self.current_df['Relative_Phase_Hours'].values)
        
        # Reset clim to force refresh
        val = self.range_slider.val
        self.scatter.set_clim(-val, val)
        
        self.fig.canvas.draw_idle()

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.scatter.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
        self.fig.canvas.draw_idle()

    def update_clim(self, val):
        self.scatter.set_clim(-val, val)
        self.fig.canvas.draw_idle()

    def update_annot(self, ind):
        if len(ind["ind"]) == 0: return
        idx = ind["ind"][0]
        # Map back to dataframe row (reset_index logic might be needed if df is filtered? 
        # No, current_df matches the scatter points 1:1 in order)
        row = self.current_df.iloc[idx]
        pos = self.scatter.get_offsets()[idx]
        self.annot.xy = pos
        text = f"Animal: {row['Source_Animal']}\nPhase: {row['Relative_Phase_Hours']:.2f} h\nX: {row['Warped_X']:.1f}, Y: {row['Warped_Y']:.1f}"
        self.annot.set_text(text)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis: self.annot.set_visible(False); self.fig.canvas.draw_idle()

    def get_export_data(self):
        return self.current_df, "group_scatter_data.csv"

def smooth_circular_phase_grid(phase_grid, count_grid, period_hours, mask=None, sigma_bins=2.0, min_weight=1e-3):
    half_range = period_hours / 2.0
    
    # 2) Build weights
    w = np.asarray(count_grid, dtype=float)
    # Defensive: ensure w itself is finite
    w[~np.isfinite(w)] = 0.0
    # Also enforce w=0 where phase is invalid (prevents weight-only support)
    w[~np.isfinite(phase_grid)] = 0.0
    
    if mask is not None:
        w[~mask] = 0.0
        
    # 3) Identify Valid Cells (Phase finite AND Weight > 0)
    #    We must not evaluate cos/sin on NaNs.
    valid = np.isfinite(phase_grid) & (w > 0)
    if mask is not None:
        valid &= mask
        
    # 4) Initialize u, v as zeros (safe against NaNs)
    u = np.zeros_like(w, dtype=float)
    v = np.zeros_like(w, dtype=float)
    
    # 5) Compute components ONLY on valid entries
    if np.any(valid):
        # theta in [-pi, pi]
        theta_valid = (phase_grid[valid] / half_range) * np.pi
        u[valid] = np.cos(theta_valid) * w[valid]
        v[valid] = np.sin(theta_valid) * w[valid]

    # Sanity Guard
    if not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(w).all()):
        raise ValueError("smooth_circular_phase_grid: non-finite values reached gaussian_filter inputs")

    # 6) Gaussian smooth
    u_f = gaussian_filter(u, sigma=sigma_bins, mode="nearest")
    v_f = gaussian_filter(v, sigma=sigma_bins, mode="nearest")
    w_f = gaussian_filter(w, sigma=sigma_bins, mode="nearest")
    
    # 7) Normalize
    eps = 1e-12
    u_s = u_f / (w_f + eps)
    v_s = v_f / (w_f + eps)
    
    mag = np.sqrt(u_s*u_s + v_s*v_s)
    ok = mag > eps
    u_s[ok] /= mag[ok]
    v_s[ok] /= mag[ok]
    
    # 8) Recover phase
    theta_s = np.arctan2(v_s, u_s) # [-pi, pi]
    phase_s = (theta_s / np.pi) * half_range
    
    # 9) Confidence & Mask cuts
    phase_s[w_f < min_weight] = np.nan
    
    if mask is not None:
        phase_s[~mask] = np.nan
        
    return phase_s

class GroupAverageMapViewer:
    def __init__(self, fig, ax, group_binned_df, group_scatter_df, grid_dims, do_smooth):
        self.fig = fig
        self.ax = ax
        self.full_binned = group_binned_df # Unused in dynamic, but kept for ref
        self.full_scatter = group_scatter_df
        self.grid_dims = grid_dims
        self.do_smooth = do_smooth
        self.period_hours = self.full_scatter['Period_Hours'].iloc[0] if not self.full_scatter.empty else 24
        
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)
        ax.set_title("Group Average Phase Map")

        # Radio Buttons
        ax_radio = fig.add_axes([0.4, 0.01, 0.2, 0.08])
        self.group_radio = RadioButtons(ax_radio, ['All', 'Control', 'Experiment'])
        self.group_radio.on_clicked(self.update_filter)

        # Smoothness Slider
        self.smooth_scale = 0.02
        ax_smooth = fig.add_axes([0.25, 0.10, 0.60, 0.03])
        self.smooth_slider = Slider(ax=ax_smooth, label="Smooth", valmin=0.0, valmax=0.05, valinit=self.smooth_scale)
        self.smooth_slider.on_changed(self._on_smooth_change)

        # Initial placeholder image
        self.im = ax.imshow(np.zeros((grid_dims[1], grid_dims[0])), origin="lower", cmap=self.cmap_cyclic)

        # Controls
        cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
        self.cbar = fig.colorbar(self.im, cax=cax)
        self.cbar.set_label("Relative Peak Time (Circadian Hours, CT)")
        
        ax_slider = fig.add_axes([0.25, 0.15, 0.60, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label="Phase Range", valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim)
        
        ax_button = fig.add_axes([0.10, 0.05, 0.15, 0.04])
        self.cmap_btn = Button(ax_button, "Mode: Cyclic")
        self.cmap_btn.on_clicked(self.toggle_cmap)

        if not self.full_scatter.empty:
            # Fixed Limits based on FULL data to ensure alignment doesn't jump
            xs, ys = self.full_scatter['Warped_X'], self.full_scatter['Warped_Y']
            self.extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            max_range = max(xs.max() - xs.min(), ys.max() - ys.min())
            half = (max_range * 1.15) / 2
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            self.im.set_extent(self.extent)
            
            # Calc initial view
            self.update_filter('All')
        
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

        self.annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w", alpha=0.9), arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def _on_smooth_change(self, val):
        self.smooth_scale = float(val)
        # Recompute current selection
        self.update_filter(self.group_radio.value_selected)

    def update_filter(self, group_label):
        if self.full_scatter.empty: return
        
        if group_label == 'All':
            df = self.full_scatter
        else:
            df = self.full_scatter[self.full_scatter['Group'] == group_label]
            
        # Recalculate Binned Grid for this subset
        nx, ny = self.grid_dims
        binned_grid = np.full((ny, nx), np.nan)
        self.count_grid = np.zeros((ny, nx), dtype=int)
        
        if df.empty:
            self.im.set_data(binned_grid)
            self.fig.canvas.draw_idle()
            return

        # Recalculate Means
        def circmean_phase(series):
            rad = (series / (self.period_hours/2.0)) * np.pi 
            mean_rad = circmean(rad, low=-np.pi, high=np.pi)
            return (mean_rad / np.pi) * (self.period_hours/2.0)

        grouped = df.groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
        
        for _, row in grouped.iterrows():
            ix, iy = int(row['Grid_X_Index']), int(row['Grid_Y_Index'])
            if 0 <= iy < ny and 0 <= ix < nx:
                binned_grid[iy, ix] = row['Relative_Phase_Hours']
                
        # Metadata (Counts)
        counts = df.groupby(['Grid_X_Index', 'Grid_Y_Index']).size().reset_index(name='count')
        for _, row in counts.iterrows():
            ix, iy = int(row['Grid_X_Index']), int(row['Grid_Y_Index'])
            if 0 <= iy < ny and 0 <= ix < nx:
                self.count_grid[iy, ix] = row['count']

        # Smoothing
        if self.do_smooth:
            # 1) Calculate Sigma (retained)
            ny, nx = binned_grid.shape
            sigma_bins = self.smooth_scale * max(nx, ny)
            sigma_bins = max(1.0, sigma_bins)
            sigma_bins = min(6.0, sigma_bins)

            # 2) Define occupancy in GRID space
            occupied = (self.count_grid > 0)

            # 3) Label connected components
            # Explicit 4-connectivity to avoid bridging diagonals
            structure = np.array([[0,1,0],
                                  [1,1,1],
                                  [0,1,0]], dtype=bool)
            lbl, ncomp = nd_label(occupied, structure=structure)

            # 4) Initialize output
            smoothed = np.full_like(binned_grid, np.nan, dtype=float)

            # 5) Process each component
            for i in range(1, ncomp + 1):
                comp_occ = (lbl == i)

                # Optional: fill enclosed holes only, DO NOT dilate outward
                comp_mask = binary_fill_holes(comp_occ)

                comp_phase = smooth_circular_phase_grid(
                    phase_grid=binned_grid,
                    count_grid=self.count_grid,
                    period_hours=self.period_hours,
                    mask=comp_mask,
                    sigma_bins=sigma_bins,
                    min_weight=1e-3,
                )

                smoothed[comp_mask] = comp_phase[comp_mask]
            
            binned_grid = smoothed

        self.im.set_data(binned_grid)
        
        # Reset clim
        val = self.range_slider.val
        self.im.set_clim(-val, val)
        self.fig.canvas.draw_idle()

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.im.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
        self.fig.canvas.draw_idle()

    def update_clim(self, val):
        self.im.set_clim(-val, val)
        self.fig.canvas.draw_idle()
        
    def hover(self, event):
        if event.inaxes != self.ax:
            if self.annot.get_visible(): self.annot.set_visible(False); self.fig.canvas.draw_idle()
            return
        
        # [Existing hover logic uses self.im extent]
        extent = self.im.get_extent() 
        arr = self.im.get_array()
        ny, nx = arr.shape
        xmin, xmax, ymin, ymax = extent
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        c = int((event.xdata - xmin) / dx)
        r = int((event.ydata - ymin) / dy)
        
        if 0 <= r < ny and 0 <= c < nx:
            val = arr[r, c]
            if np.ma.is_masked(val) or np.isnan(val):
                self.annot.set_visible(False); self.fig.canvas.draw_idle(); return
            
            count = self.count_grid[r, c]
            text = f"Phase: {val:.2f} h\nN={count} cells"
            if count == 0: text += "\n(Interpolated)"
            
            self.annot.xy = (event.xdata, event.ydata)
            self.annot.set_text(text)
            self.annot.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            if self.annot.get_visible(): self.annot.set_visible(False); self.fig.canvas.draw_idle()

    def get_export_data(self):
        # Return currently filtered data
        # We don't store the filtered binned df, so let's just return full scatter
        return self.full_scatter, "group_data.csv"

class InterpolatedMapViewer:
    def __init__(self, fig, ax, roi_data, relative_phases,
                 period_hours, grid_resolution, rois=None):
        self.fig = fig
        self.ax = ax
        self.period_hours = period_hours
        
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        ax.set_title("Interpolated Spatiotemporal Phase Map")

        if len(roi_data) < 4:
            ax.text(0.5, 0.5, "Not enough data points (<4) for interpolation.", ha="center", va="center")
            return

        phase_angles_rad = (relative_phases / (period_hours / 2.0)) * pi
        x_comp = np.cos(phase_angles_rad)
        y_comp = np.sin(phase_angles_rad)

        xs = roi_data[:, 0]
        ys = roi_data[:, 1]
        bounds_x_min, bounds_x_max = xs.min(), xs.max()
        bounds_y_min, bounds_y_max = ys.min(), ys.max()
        
        if rois:
            include_verts = []
            for r in rois:
                if isinstance(r.get("mode"), str) and r.get("mode").strip().lower() == "include" and "path_vertices" in r:
                    include_verts.append(np.array(r["path_vertices"]))
            
            if include_verts:
                all_verts = np.vstack(include_verts)
                roi_x_min, roi_x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
                roi_y_min, roi_y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
                bounds_x_min = min(bounds_x_min, roi_x_min)
                bounds_x_max = max(bounds_x_max, roi_x_max)
                bounds_y_min = min(bounds_y_min, roi_y_min)
                bounds_y_max = max(bounds_y_max, roi_y_max)

        x_buf = (bounds_x_max - bounds_x_min) * 0.05
        y_buf = (bounds_y_max - bounds_y_min) * 0.05
        grid_x_min, grid_x_max = bounds_x_min - x_buf, bounds_x_max + x_buf
        grid_y_min, grid_y_max = bounds_y_min - y_buf, bounds_y_max + y_buf

        grid_x, grid_y = np.mgrid[
            grid_x_min:grid_x_max:complex(grid_resolution),
            grid_y_min:grid_y_max:complex(grid_resolution),
        ]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        rbf_x = RBFInterpolator(roi_data, x_comp, kernel="linear", smoothing=1.0)
        rbf_y = RBFInterpolator(roi_data, y_comp, kernel="linear", smoothing=1.0)
        
        gx = rbf_x(grid_points)
        gy = rbf_y(grid_points)
        angles = arctan2(gy, gx)
        grid_z = (angles / pi) * (period_hours / 2.0)
        grid_z = grid_z.reshape(grid_x.shape)

        if rois:
            final_mask = np.zeros(grid_x.shape, dtype=bool)

            def _mode_lower(r):
                m = r.get("mode", "")
                return m.strip().lower() if isinstance(m, str) else ""

            def _as_path(r):
                # Prefer prebuilt Path, otherwise build from vertices
                p = r.get("path", None)
                if isinstance(p, Path):
                    return p
                verts = r.get("path_vertices", None)
                if verts is None:
                    return None
                try:
                    return Path(verts)
                except Exception:
                    return None

            include_paths = []
            exclude_paths = []

            for r in rois:
                mode = _mode_lower(r)
                p = _as_path(r)
                if p is None:
                    continue
                if mode == "include":
                    include_paths.append(p)
                elif mode == "exclude":
                    exclude_paths.append(p)

            if include_paths:
                for p in include_paths:
                    final_mask |= p.contains_points(grid_points).reshape(grid_x.shape)
            else:
                # Fallback: convex hull of the rhythmic ROI points
                if len(roi_data) > 2:
                    hull = ConvexHull(roi_data)
                    hpath = Path(roi_data[hull.vertices])
                    final_mask = hpath.contains_points(grid_points).reshape(grid_x.shape)

            for p in exclude_paths:
                final_mask &= ~p.contains_points(grid_points).reshape(grid_x.shape)

            grid_z[~final_mask] = np.nan

        else:
            # No ROIs provided, fall back to convex hull of data points
            if len(roi_data) > 2:
                hull = ConvexHull(roi_data)
                hpath = Path(roi_data[hull.vertices])
                mask = hpath.contains_points(grid_points).reshape(grid_x.shape)
                grid_z[~mask] = np.nan

        self.im = ax.imshow(
            grid_z.T,
            origin="upper",
            extent=[grid_x_min, grid_x_max, grid_y_max, grid_y_min], 
            cmap=self.cmap_cyclic,
            interpolation="bilinear",
        )

        cx = (bounds_x_min + bounds_x_max) / 2
        cy = (bounds_y_min + bounds_y_max) / 2
        range_x = bounds_x_max - bounds_x_min
        range_y = bounds_y_max - bounds_y_min
        max_range = max(range_x, range_y)
        half_span = (max_range * 1.15) / 2 
        
        ax.set_xlim(cx - half_span, cx + half_span)
        ax.set_ylim(cy + half_span, cy - half_span) 

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
        self.cbar = fig.colorbar(self.im, cax=cax)
        self.cbar.set_label("Relative Peak Time (hours)", fontsize=10)

        ax_slider = fig.add_axes([0.25, 0.10, 0.60, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(
            ax=ax_slider,
            label='Phase Range (+/- hrs)',
            valmin=1.0,
            valmax=max_range,
            valinit=max_range,
        )
        self.range_slider.on_changed(self.update_clim)
        
        ax_button = fig.add_axes([0.25, 0.05, 0.15, 0.04])
        self.cmap_btn = Button(ax_button, "Mode: Cyclic")
        self.cmap_btn.on_clicked(self.toggle_cmap)
        
        self.update_clim(max_range)

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.im.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
        self.fig.canvas.draw_idle()

    def update_clim(self, val):
        for im in self.ax.images:
            im.set_clim(-val, val)
        self.fig.canvas.draw_idle()

class PhaseGradientViewer:
    def __init__(self, fig, ax, gradient_data):
        self.fig = fig
        self.ax = ax
        self.gradient_data = gradient_data
        
        self.fig.subplots_adjust(left=0.15, bottom=0.20, right=0.95, top=0.9)
        ax.set_title("Dorsoventral Phase Gradient")
        ax.set_xlabel("Anatomical Position (s)\n(0.0 = Dorsal/Start, 1.0 = Ventral/End)")
        ax.set_ylabel("Relative Phase (h, wrapped to [-12, +12])")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        gd = getattr(self, 'gradient_data', None)
        
        if not gd:
            ax.text(0.5, 0.5, "No gradient data available.", ha='center')
            self.fig.canvas.draw_idle()
            return

        # Prepare colors
        unique_groups = sorted(list(set(d['group'] for d in gd)))
        group_colors = {'Control': 'blue', 'Experiment': 'red'} 
        # Fallback for other groups
        import matplotlib.colors as mcolors
        fallback_colors = list(mcolors.TABLEAU_COLORS.values())
        for i, g in enumerate(unique_groups):
            if g not in group_colors:
                group_colors[g] = fallback_colors[i % len(fallback_colors)]
        
        # Plot lines
        for d in gd:
            s = d['s']
            ph = d['phases']
            grp = d['group']
            color = group_colors.get(grp, 'gray')
            alpha = 0.5
            lw = 1.0
            
            # Label includes Animal ID and Slope
            label = f"{d['animal']} ({grp})"
            if np.isfinite(d.get('slope_hours', np.nan)):
                 label += f" [Slope: {d['slope_hours']:+.2f}h/u]"
            
            ax.plot(s, ph, marker='o', markersize=4, linestyle='-', color=color, alpha=alpha, linewidth=lw, label=label)
            
        # Add legend or maybe too many items?
        # If > 10 items, maybe just legend groups?
        if len(gd) <= 15:
            ax.legend(fontsize='small', loc='best')
        else:
            # Legend for groups only
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=c, lw=2, label=g) for g, c in group_colors.items()]
            ax.legend(handles=handles, fontsize='small', loc='best')
            

            
        ax.set_ylim(-12, 12) # Relative Phase Symmetric
        ax.set_xlim(0, 1)

    def get_export_data(self):
        # Flatten data for export
        import pandas as pd
        
        # Return empty if no data
        # Return empty if no data
        if not hasattr(self, 'gradient_data') or not self.gradient_data:
            return pd.DataFrame(), "phase_gradient_data.csv"

        rows = []
        for entry in self.gradient_data:
            animal = entry['animal']
            group = entry.get('group', 'Unassigned')
            s_vals = entry['s']
            p_vals = entry['phases']
            counts = entry.get('counts', np.full(len(s_vals), np.nan)) 
            slope_h = entry.get('slope_hours', np.nan)
            slope_rad = entry.get('slope_rad', np.nan)
            r_val = entry.get('r_value', np.nan)
            mode = entry.get('mode', '')
            period = entry.get('period', 24.0)
            
            for i in range(len(s_vals)):
                val = p_vals[i]
                rows.append({
                    'Animal_ID': animal,
                    'Group': group,
                    'Gradient_Mode': mode,
                    'Period_Hours_Used': period,
                    'Bin_Index': i,
                    'S_Coordinate': s_vals[i],
                    'S_Coordinate': s_vals[i],
                    'S_Coordinate': s_vals[i],
                    'Mean_Phase_RelHours': float(self._normalize_hours(val, target='rel_pm12')), 
                    'N_Cells_In_Bin': counts[i],
                    'Slope_Hours_Per_Unit': slope_h,
                    'Slope_Rad_Per_Unit': slope_rad,
                    'Slope_R_Value': r_val,
                    'Phase_Domain': 'rel_pm12'
                })
        df = pd.DataFrame(rows)
        return df, "phase_gradient_data.csv"

# --- ADD THIS CLASS TO THE END OF THE FILE ---

class RegionResultViewer(QtWidgets.QWidget):
    """
    Visualizes the results of the Region-Based Analysis.
    Tab 1: Atlas Map colored by Phase Difference.
    Tab 2: Statistical Table.
    Tab 3: Animal Details (Wide Table + Mini Plot).
    """
    def __init__(self, zone_stats, atlas_polys_by_zone, parent=None):
        super().__init__(parent)
        self.zone_stats = zone_stats
        self.atlas_polys = atlas_polys_by_zone
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. Map Tab
        self.map_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.map_tab, "Region Map")
        map_layout = QtWidgets.QVBoxLayout(self.map_tab)
        
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.map_tab)
        map_layout.addWidget(self.toolbar)
        map_layout.addWidget(self.canvas)
        
        # 2. Table Tab
        self.table_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.table_tab, "Stats Table")
        table_layout = QtWidgets.QVBoxLayout(self.table_tab)
        self.stats_table = QtWidgets.QTableWidget()
        table_layout.addWidget(self.stats_table)
        
        # 3. Animal Details Tab
        self.detail_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.detail_tab, "Animal Details")
        detail_layout = QtWidgets.QVBoxLayout(self.detail_tab)
        
        # Header Label
        self.lbl_detail_header = QtWidgets.QLabel("Select a zone in 'Stats Table' to view details.")
        self.lbl_detail_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        detail_layout.addWidget(self.lbl_detail_header)
        
        # Wide Table
        self.detail_table = QtWidgets.QTableWidget()
        self.detail_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.detail_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        detail_layout.addWidget(self.detail_table, stretch=2)
        
        # Mini Plot
        self.detail_fig = Figure(figsize=(5, 3))
        self.detail_canvas = FigureCanvas(self.detail_fig)
        detail_layout.addWidget(self.detail_canvas, stretch=1)
        
        # Buttons for Details Tab
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_export_detail = QtWidgets.QPushButton("Export Table (CSV)")
        self.btn_export_detail.clicked.connect(self.export_current_detail_table)
        btn_layout.addWidget(self.btn_export_detail)
        
        self.btn_export_all = QtWidgets.QPushButton("Export Full Cell Data")
        self.btn_export_all.clicked.connect(self.export_full_cell_data)
        btn_layout.addWidget(self.btn_export_all)
        
        detail_layout.addLayout(btn_layout)
        
        self._draw_map()
        self._populate_table()
        
        # Trigger selection of first row if rows exist
        if self.stats_table.rowCount() > 0:
             self.stats_table.selectRow(0)

    def _draw_map(self):
        ax = self.fig.add_subplot(111)
        ax.set_title("Region Phase Differences (Exp - Ctrl)\n* = Significant (p < 0.05)")
        ax.set_aspect('equal')
        
        # Colormap setup: Blue (Early) -> White -> Red (Late)
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.patches import Polygon as MplPolygon
        
        norm = mcolors.Normalize(vmin=-4, vmax=4)
        mapper = cm.ScalarMappable(norm=norm, cmap='coolwarm')
        
        all_x, all_y = [], []

        for stat in self.zone_stats:
            zid = stat['id']
            diff = stat.get('diff_mean', 0)
            is_sig = stat.get('p_value', 1.0) < 0.05
            
            color = mapper.to_rgba(diff)
            
            polys = self.atlas_polys.get(zid, [])
            for poly_path in polys:
                verts = poly_path.vertices # Path object vertices
                all_x.extend(verts[:, 0])
                all_y.extend(verts[:, 1])
                
                poly = MplPolygon(verts, closed=True, facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
                ax.add_patch(poly)
                
                # Significance Marker
                cx, cy = np.mean(verts[:, 0]), np.mean(verts[:, 1])
                if is_sig:
                    ax.text(cx, cy, "*", fontsize=24, ha='center', va='center', color='black', weight='bold')
                
                # Label
                ax.text(cx, cy, f"\n{diff:+.1f}h", fontsize=9, ha='center', va='top', color='black', weight='bold')

        if all_x:
            pad = 50
            ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            ax.set_ylim(max(all_y)+pad, min(all_y)-pad) # Invert Y for imaging convention
        
        # Colorbar
        cbar = self.fig.colorbar(mapper, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Phase Difference (Hours)")
        cbar.set_ticks([-4, 0, 4])
        cbar.set_ticklabels(["-4h (Earlier)", "0h", "+4h (Later)"])
        
        self.canvas.draw()

    def _populate_table(self):
        cols = ["Zone ID", "Name", "N(Ctrl)", "N(Exp)", "Mean(Ctrl)", "Mean(Exp)", "Diff", "p-value"]
        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.setRowCount(len(self.zone_stats))
        
        for r, stat in enumerate(self.zone_stats):
            self.stats_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(stat['id'])))
            self.stats_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(stat['name'])))
            self.stats_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(stat.get('n_ctrl', 0))))
            self.stats_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(stat.get('n_exp', 0))))
            self.stats_table.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{stat.get('mean_ctrl', np.nan):.2f}"))
            self.stats_table.setItem(r, 5, QtWidgets.QTableWidgetItem(f"{stat.get('mean_exp', np.nan):.2f}"))
            
            diff_item = QtWidgets.QTableWidgetItem(f"{stat.get('diff_mean', np.nan):+.2f}")
            if stat.get('diff_mean', 0) > 0: diff_item.setForeground(QtGui.QColor('red'))
            else: diff_item.setForeground(QtGui.QColor('blue'))
            self.stats_table.setItem(r, 6, diff_item)
            
            p = stat.get('p_value', np.nan)
            p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
            p_item = QtWidgets.QTableWidgetItem(p_str)
            if p < 0.05:
                p_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                p_item.setBackground(QtGui.QColor("#d4edda")) # Light green
            self.stats_table.setItem(r, 7, p_item)
            
        self.stats_table.resizeColumnsToContents()
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.stats_table.itemSelectionChanged.connect(self.on_zone_selection)

    def on_zone_selection(self):
        indexes = self.stats_table.selectionModel().selectedRows()
        if not indexes:
            return
            
        row = indexes[0].row()
        # Item 0 is ID
        z_id_item = self.stats_table.item(row, 0)
        if not z_id_item: return
        
        try:
            z_id = int(z_id_item.text())
        except ValueError:
            return
            
        # Find record
        record = next((r for r in self.zone_stats if r['id'] == z_id), None)
        if record:
            self._update_detail_view(record)

    def _update_detail_view(self, record):
        """Builds the wide table and plots for the selected zone."""
        self.lbl_detail_header.setText(f"ROI: {record['name']} (ID: {record['id']})")
        self.current_detail_record = record # Store for export
        # Store per-group summaries for mini plot reuse (avoid recomputation drift)
        self._current_detail_group_summaries = {}
        
        # 1. Prepare Data
        groups = {}
        for d in record['data']:
            g = d['group']
            if g not in groups: groups[g] = []
            groups[g].append(d)
            
        group_names = sorted(groups.keys())
        if 'Control' in group_names: 
            group_names.remove('Control')
            group_names = ['Control'] + group_names
            
        # 2. Build Wide Table
        self.detail_table.clear()
        self.detail_table.setColumnCount(0)
        self.detail_table.setRowCount(0)
        
        metrics = ["Mean (h)", "Std (h)", "Count (N)", "SEM (h)", 
                   "Mean (rad)", "Mean (h, mod24)", "R (0-1)"]
        summary_metrics = ["Group_Mean (h)", "Group_Std (h)", "Group_N (Animals)", "Group_SEM (h)"]
        
        current_row = 0
        total_cols = 0
        self.detail_table.setSortingEnabled(False)
        
        for g_name in group_names:
            animals = sorted(groups[g_name], key=lambda x: str(x['animal']))
            if not animals: continue
            
            # Header Row for Group
            self.detail_table.insertRow(current_row)
            self.detail_table.setItem(current_row, 0, QtWidgets.QTableWidgetItem(f"Group: {g_name}"))
            font = QtGui.QFont()
            font.setBold(True)
            self.detail_table.item(current_row, 0).setFont(font)
            self.detail_table.item(current_row, 0).setBackground(QtGui.QColor("#e0e0e0"))
            current_row += 1
            
            # Column Headers
            n_animals = len(animals)
            needed_cols = 1 + n_animals + 1
            if needed_cols > total_cols:
                self.detail_table.setColumnCount(needed_cols)
                total_cols = needed_cols
                
            labels = ["Metric"] + [str(a['animal']) for a in animals] + ["Group Summary"]
            self.detail_table.insertRow(current_row)
            for c, pwm in enumerate(labels):
                it = QtWidgets.QTableWidgetItem(pwm)
                it.setBackground(QtGui.QColor("#f0f0f0"))
                self.detail_table.setItem(current_row, c, it)
            current_row += 1
            
            # Data Rows
            animal_stats_list = []
            for d in animals:
                stats = self._compute_roi_animal_stats(np.array(d.get('raw_phases', []), dtype=float), period_h=24.0)
                animal_stats_list.append(stats)
            
            # Group Summary
            # Collect mean_rad from animals with n_cells > 0
            valid_mean_rads = [s['mean_rad'] for s in animal_stats_list
                               if s.get('n_cells', 0) > 0 and np.isfinite(s.get('mean_rad', np.nan))]
            
            g_n = len(valid_mean_rads)
            if g_n > 0:
                # Group Mean of Means (angular)
                g_mean_rad = circmean(np.array(valid_mean_rads, dtype=float), low=-np.pi, high=np.pi)
                # Convert to hours [0, 24)
                g_mean_h_mod24 = (g_mean_rad / (2*np.pi)) * 24.0
                g_mean_h_mod24 = g_mean_h_mod24 % 24.0
                if g_mean_h_mod24 < 0: g_mean_h_mod24 += 24.0

                # Convert to Display Domain
                g_mean_h_display = self._normalize_hours(np.array([g_mean_h_mod24]), target='rel_pm12')[0]
                
                if g_n >= 2:
                    g_std_rad = circstd(np.array(valid_mean_rads, dtype=float), low=-np.pi, high=np.pi)
                    g_std_h = (g_std_rad / (2*np.pi)) * 24.0
                    g_sem_h = g_std_h / np.sqrt(g_n)
                else:
                    g_std_h = np.nan
                    g_sem_h = np.nan
                
                group_summary = {
                    "Group_Mean (h)": g_mean_h_display,
                    "Group_Std (h)": g_std_h,
                    "Group_N (Animals)": g_n,
                    "Group_SEM (h)": g_sem_h
                }
            else:
                group_summary = {}

            # Persist summary for plotting and export consistency
            self._current_detail_group_summaries[g_name] = group_summary

            # Fill Metrics
            for m in metrics:
                self.detail_table.insertRow(current_row)
                self.detail_table.setItem(current_row, 0, QtWidgets.QTableWidgetItem(m))
                
                for i, stats in enumerate(animal_stats_list):
                    key_map = {
                        "Mean (h)": 'mean_h', "Std (h)": 'std_h', "Count (N)": 'n_cells',
                        "SEM (h)": 'sem_h', "Mean (rad)": 'mean_rad', "Mean (h, mod24)": 'mean_h_mod24', "R (0-1)": 'R'
                    }
                    val = stats.get(key_map.get(m, ''), np.nan)
                    
                    if m == "Count (N)" or m == "Group_N (Animals)":
                        txt = f"{int(val)}"
                    elif m == "R (0-1)":
                        txt = f"{val:.3f}"
                    else:
                        txt = f"{val:.2f}"
                        
                    self.detail_table.setItem(current_row, 1+i, QtWidgets.QTableWidgetItem(txt))
                
                current_row += 1
                
            # Fill Group Summaries
            for m in summary_metrics:
                self.detail_table.insertRow(current_row)
                self.detail_table.setItem(current_row, 0, QtWidgets.QTableWidgetItem(m))
                val = group_summary.get(m, np.nan)
                if "N (" in m: txt = f"{int(val)}" if np.isfinite(val) else "0"
                else: txt = f"{val:.2f}"
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFont(font)
                self.detail_table.setItem(current_row, n_animals + 1, item)
                current_row += 1
                
            # Spacer
            self.detail_table.insertRow(current_row)
            current_row += 1

        self.detail_table.resizeColumnsToContents()
        self._plot_mini_distribution(groups)

    def _normalize_hours(self, phases_h, period_h=24.0, target='rel_pm12'):
        """
        Normalize phase hours to a specific domain.
        target='mod24': [0, period)
        target='rel_pm12': [-period/2, +period/2)
        """
        period_h = float(period_h)
        if not np.isfinite(period_h) or period_h <= 0:
            raise ValueError("period_h must be a positive finite number")
            
        ph = np.asarray(phases_h, dtype=float)
        # Wrap robustly to [0, period)
        ph_mod = np.mod(ph, period_h)
        ph_mod = (ph_mod + period_h) % period_h
        
        if target == 'mod24':
            out = ph_mod
        elif target == 'rel_pm12':
            half = period_h / 2.0
            out = ((ph_mod + half) % period_h) - half
        else:
            raise ValueError(f"Unknown target: {target}")
        
        # scalar-safe return
        if out.shape == ():
            return float(out)
        return out

    def _compute_roi_animal_stats(self, phases_hours, period_h=24.0):
        """Computes circular stats."""
        phases_hours = np.asarray(phases_hours, dtype=float)
        phases_hours = phases_hours[np.isfinite(phases_hours)]
        if phases_hours.size == 0:
            return {'mean_h': np.nan, 'std_h': np.nan, 'n_cells': 0, 'sem_h': np.nan, 
                    'mean_rad': np.nan, 'mean_h_mod24': np.nan, 'R': np.nan}
            
        period = float(period_h)
        
        # Strict convention: normalize to [0, period) for trig
        ph_mod24 = self._normalize_hours(phases_hours, period_h=period, target='mod24')
        theta = (ph_mod24 / period) * 2.0 * np.pi # domain: [0, 2pi)
        
        # Circular mean/std must use matching bounds for the theta domain.
        # Compute in [0, 2pi), then wrap the returned mean into [-pi, pi) for reporting.
        mean_rad_0_2pi = circmean(theta, low=0.0, high=2.0*np.pi)
        mean_rad = ((mean_rad_0_2pi + np.pi) % (2.0*np.pi)) - np.pi
        
        # Mean Hours Mod24 [0, 24)
        mean_h_mod24 = (mean_rad / (2.0*np.pi)) * period
        mean_h_mod24 = mean_h_mod24 % period
        if mean_h_mod24 < 0: mean_h_mod24 += period
        
        # Mean Hours Display (Relative [-12, 12))
        mean_h_display = self._normalize_hours(np.array([mean_h_mod24]), period_h=period, target='rel_pm12')[0]
        
        # Resultant Vector Length R
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        R = np.sqrt(C**2 + S**2)
        
        # Std Dev
        std_rad = circstd(theta, low=0.0, high=2.0*np.pi)
        std_h = (std_rad / (2.0*np.pi)) * period
        
        # SEM
        sem_h = std_h / np.sqrt(len(theta))
        
        return {
            'mean_h': mean_h_display,
            'std_h': std_h,
            'n_cells': len(theta),
            'sem_h': sem_h,
            'mean_rad': mean_rad,
            'mean_h_mod24': mean_h_mod24,
            'R': R
        }

    def _plot_mini_distribution(self, groups):
        self.detail_fig.clear()
        ax = self.detail_fig.add_subplot(111)
        
        # Assign colors robustly, independent of exact group naming
        color_cycle = ax._get_lines.prop_cycler
        group_to_color = {}
        
        import matplotlib.colors as mcolors
        fallback_colors = list(mcolors.TABLEAU_COLORS.values())
        
        for g_name in sorted(groups.keys(), key=lambda x: (x != 'Control', str(x))):
            if g_name not in group_to_color:
                try:
                    group_to_color[g_name] = next(color_cycle)['color']
                except StopIteration:
                    group_to_color[g_name] = fallback_colors[len(group_to_color) % len(fallback_colors)]
            animals = groups[g_name]
            color = group_to_color[g_name]
            
            g_phases = []
            for a in animals:
                g_phases.extend(a.get('raw_phases', []))
            
            if not g_phases: continue
            g_phases = np.asarray(g_phases, dtype=float)
            g_phases = g_phases[np.isfinite(g_phases)]
            if g_phases.size == 0:
                continue

            # Normalized to Display Domain [-12, +12)
            g_display = self._normalize_hours(g_phases, target='rel_pm12')
            
            # Histogram
            ax.hist(
                g_display,
                bins=24,
                range=(-12, 12),
                alpha=0.45,
                label=str(g_name),
                color=color,
                density=True
            )
            
            # Vertical Mean line must reuse the table’s between-animal group summary if available
            gs = getattr(self, '_current_detail_group_summaries', {}).get(g_name, {})
            gm = gs.get("Group_Mean (h)", np.nan)
            if np.isfinite(gm):
                ax.axvline(float(gm), color=color, linestyle='--', linewidth=2)

        ax.set_title("Pooled cell phases (visualization only)")
        ax.set_xlabel("Phase (h, wrapped to [-12, +12))")
        ax.set_xlim(-12, 12)
        ax.legend(fontsize='small')
        self.detail_canvas.draw()

    def get_export_data(self):
        data = []
        for s in self.zone_stats:
            row = s.copy()
            if 'data' in row: del row['data'] 
            data.append(row)
        return pd.DataFrame(data), "region_stats.csv"

    def export_current_detail_table(self):
        """Exports the currently displayed detailed table."""
        if not hasattr(self, 'current_detail_record') or not self.current_detail_record:
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Detail Table", 
                                                        f"roi_{self.current_detail_record['id']}_details.csv", 
                                                        "CSV (*.csv)")
        if not path: return
        
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Metadata Header
            rec = self.current_detail_record
            writer.writerow(["ROI_ID", rec['id'], "ROI_NAME", rec['name'], "PHASE_DOMAIN", "rel_pm12", "PERIOD_H", "24.0"])
            writer.writerow([]) # Blank row
            
            # Table Dump
            for r in range(self.detail_table.rowCount()):
                row_data = []
                for c in range(self.detail_table.columnCount()):
                    item = self.detail_table.item(r, c)
                    text = item.text() if item else ""
                    row_data.append(text)
                writer.writerow(row_data)
                
    def export_full_cell_data(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Full Cell Data", "region_raw_cells.csv", "CSV (*.csv)")
        if not path: return
            
        try:
            all_rows = []
            for s in self.zone_stats:
                z_id = s['id']
                z_name = s['name']
                for d in s['data']:
                    for ph in d.get('raw_phases', []):
                        all_rows.append({
                            'Zone_ID': z_id,
                            'Zone_Name': z_name,
                            'Group': d['group'],
                            'Animal': d['animal'],
                            'Group': d['group'],
                            'Animal': d['animal'],
                            'Cell_Phase_RelHours': float(self._normalize_hours(ph, target='rel_pm12')),
                            'Phase_Domain': 'rel_pm12'
                        })
            
            import pandas as pd
            df = pd.DataFrame(all_rows)
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Success", f"Exported {len(df)} cell records to:\n{path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
