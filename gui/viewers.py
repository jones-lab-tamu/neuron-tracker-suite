import numpy as np
import pandas as pd
import colorcet as cet
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from numpy import pi, arctan2

from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path

import cosinor as csn
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
        # Store the current selection state
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

        # Reserve space at the bottom for sliders and right for colorbar (consistency)
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        self.image_artist = ax.imshow(bg_image, cmap="gray")
        ax.set_title("Center of Mass (Click to Select Trajectory)")

        # Position sliders in the reserved space
        # Standardized positions: Slider at y=0.10, Secondary at y=0.05
        ax_contrast = fig.add_axes([0.25, 0.10, 0.60, 0.03])
        ax_brightness = fig.add_axes([0.25, 0.05, 0.60, 0.03])
        
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

        # Reserve space at the bottom for controls and right for consistency
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        # --- Create Widgets ---
        # Previous/Next buttons for trajectory index
        # Positioned at y=0.05 to align with secondary controls
        ax_prev = fig.add_axes([0.65, 0.05, 0.1, 0.04])
        ax_next = fig.add_axes([0.76, 0.05, 0.1, 0.04])
        from matplotlib.widgets import Button
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(self.prev_trajectory)
        self.btn_next.on_clicked(self.next_trajectory)

        # Slider for frame navigation
        # Positioned at y=0.10 to align with primary sliders
        ax_slider = fig.add_axes([0.15, 0.10, 0.45, 0.03])
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
        
        # State for colormap
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        # Reserve space for controls and right for colorbar
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        self.bg_artist = ax.imshow(bg_image, cmap="gray", vmin=vmin, vmax=vmax)
        
        if not self.rhythmic_df.empty:
            self.scatter = ax.scatter(
                self.rhythmic_df['X_Position'], self.rhythmic_df['Y_Position'],
                c=self.rhythmic_df['Relative_Phase_Hours'],
                cmap=self.cmap_cyclic, s=25, edgecolor="black", linewidth=0.5,
            )
            # Manual colorbar placement to prevent resizing the main plot
            cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
            self.cbar = fig.colorbar(self.scatter, cax=cax)
            self.cbar.set_label("Relative Peak Time (hours)", fontsize=10)
            
            # Position controls in reserved space
            # Slider at y=0.10
            ax_slider = fig.add_axes([0.25, 0.10, 0.60, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim)
            
            # Colormap Toggle Button - Positioned below slider at y=0.05
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
        selected_index = np.argmin(distances) # This is the ROW index of the dataframe
        
        if distances[selected_index] < 20:
            # Convert Row Index -> Global Index for consistency
            # Original_ROI_Index is 1-based, convert to 0-based Global Index
            global_index = int(self.rhythmic_df.iloc[selected_index]['Original_ROI_Index'] - 1)
            
            # Highlight using the Global Index (which internal logic converts back to Row Index)
            self.highlight_point(global_index)
            
            # Pass the ROW index to the callback (as expected by the current wrapper in Main)
            if self.on_select_callback: self.on_select_callback(selected_index)

    def highlight_point(self, original_index):
        """
        Highlights a point based on the Global Original Index.
        Scans the dataframe to find the matching row.
        """
        if self.highlight_artist: self.highlight_artist.remove(); self.highlight_artist = None
        
        if original_index is not None and not self.rhythmic_df.empty:
            # Find the row corresponding to this global index
            # 'Original_ROI_Index' is 1-based in the DF, so we add 1 to the 0-based input
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
    def __init__(self, fig, ax, group_df):
        self.fig = fig
        self.ax = ax
        self.group_df = group_df
        self.period_hours = self.group_df['Period_Hours'].iloc[0] if not self.group_df.empty else 24
        
        # State
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        # Reserve space
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        ax.set_title("Group Phase Distribution")
        if not self.group_df.empty:
            self.scatter = ax.scatter(
                self.group_df['Warped_X'], self.group_df['Warped_Y'],
                c=self.group_df['Relative_Phase_Hours'], cmap=self.cmap_cyclic, s=10, alpha=0.8,
            )
            # Manual colorbar placement
            cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
            self.cbar = fig.colorbar(self.scatter, cax=cax)
            self.cbar.set_label("Mean Relative Peak Time (hours)")
            
            # Position controls
            ax_slider = fig.add_axes([0.25, 0.10, 0.60, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim)
            
            # Colormap Toggle Button
            ax_button = fig.add_axes([0.25, 0.05, 0.15, 0.04])
            self.cmap_btn = Button(ax_button, "Mode: Cyclic")
            self.cmap_btn.on_clicked(self.toggle_cmap)

            self.update_clim(max_range)
            
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.scatter.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
        self.fig.canvas.draw_idle()

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
        
        # State
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        # Reserve space - make axes square (height=0.65, so width should also be 0.65)
        # Height: 0.9 - 0.25 = 0.65, Width: right(0.85) - left should = 0.65, so left = 0.20
        self.fig.subplots_adjust(left=0.20, bottom=0.25, right=0.85, top=0.9)

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

        self.im = ax.imshow(binned_grid, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap=self.cmap_cyclic)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        
        # Manual colorbar placement
        cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
        self.cbar = fig.colorbar(self.im, cax=cax)
        self.cbar.set_label("Mean Relative Peak Time (hours)")
        
        # Position controls
        ax_slider = fig.add_axes([0.25, 0.10, 0.60, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- hrs)", valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim)

        # Colormap Toggle Button
        ax_button = fig.add_axes([0.25, 0.05, 0.15, 0.04])
        self.cmap_btn = Button(ax_button, "Mode: Cyclic")
        self.cmap_btn.on_clicked(self.toggle_cmap)

        self.update_clim(max_range)
        self.fig.canvas.draw_idle()

    def toggle_cmap(self, event):
        self.is_cyclic = not self.is_cyclic
        new_cmap = self.cmap_cyclic if self.is_cyclic else self.cmap_diverging
        label = "Mode: Cyclic" if self.is_cyclic else "Mode: Diverging"
        self.im.set_cmap(new_cmap)
        self.cmap_btn.label.set_text(label)
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
        
        # State
        self.is_cyclic = True 
        self.cmap_cyclic = cet.cm.cyclic_mygbm_30_95_c78
        self.cmap_diverging = 'coolwarm'

        # Reserve space
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

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

        self.im = ax.imshow(
            grid_z.T,
            origin="upper",
            extent=[
                x_min - x_buf,
                x_max + x_buf,
                y_min - y_buf,
                y_max + y_buf,
            ],
            cmap=self.cmap_cyclic,
            interpolation="bilinear",
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Manual colorbar placement
        cax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
        self.cbar = fig.colorbar(self.im, cax=cax)
        self.cbar.set_label("Relative Peak Time (hours)", fontsize=10)

        # Position controls
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
        
        # Colormap Toggle Button
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
        # assumes image exists
        for im in self.ax.images:
            im.set_clim(-val, val)
        self.fig.canvas.draw_idle()
