import numpy as np
import pandas as pd
import colorcet as cet
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from numpy import pi, arctan2
from scipy.stats import circmean, linregress, sem

from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.patches import Polygon, Rectangle
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

        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.85, top=0.9)

        ax_prev = fig.add_axes([0.65, 0.05, 0.1, 0.04])
        ax_next = fig.add_axes([0.76, 0.05, 0.1, 0.04])
        from matplotlib.widgets import Button
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(self.prev_trajectory)
        self.btn_next.on_clicked(self.next_trajectory)

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
        self.update()

    def set_trajectory(self, index):
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
            self.ax.plot(
                traj[:, 1], traj[:, 0], '-', color='cyan', linewidth=1, alpha=0.7
            )
            current_pos = traj[current_frame]
            self.ax.plot(
                current_pos[1], current_pos[0], 'o', 
                markersize=10, 
                markerfacecolor=(1, 1, 0, 0.5), 
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
        ax_radio = fig.add_axes([0.4, 0.05, 0.2, 0.1])
        self.group_radio = RadioButtons(ax_radio, ['All', 'Control', 'Experiment'])
        self.group_radio.on_clicked(self.update_filter)

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

    def update_filter(self, label):
        if self.full_scatter.empty: return
        
        if label == 'All':
            df = self.full_scatter
        else:
            df = self.full_scatter[self.full_scatter['Group'] == label]
            
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
             # [Same smoothing logic as before]
            original_grid = binned_grid.copy()
            rows, cols = original_grid.shape
            for r in range(rows):
                for c in range(cols):
                    if np.isnan(original_grid[r, c]):
                        r_min, r_max = max(0, r-1), min(rows, r+2)
                        c_min, c_max = max(0, c-1), min(cols, c+2)
                        window = original_grid[r_min:r_max, c_min:c_max]
                        valid = window[~np.isnan(window)]
                        if valid.size > 0:
                            rads = (valid / (self.period_hours / 2.0)) * np.pi
                            mean_rad = circmean(rads, low=-np.pi, high=np.pi)
                            binned_grid[r, c] = (mean_rad / np.pi) * (self.period_hours / 2.0)

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
        ax.set_ylabel("Relative Phase (CT Hours)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if not gradient_data:
            ax.text(0.5, 0.5, "No gradient data available.", ha='center')
            self.fig.canvas.draw_idle()
            return

        # Group data
        groups = {'Control': [], 'Experiment': []}
        for entry in gradient_data:
            g = entry.get('group', 'Unassigned')
            if g in groups: groups[g].append(entry)
            else:
                if 'Unassigned' not in groups: groups['Unassigned'] = []
                groups['Unassigned'].append(entry)

        colors = {'Control': 'blue', 'Experiment': 'red', 'Unassigned': 'gray'}
        stats_text = "Group Metrics (Mean ± SEM):\n"
        
        for grp, entries in groups.items():
            if not entries: continue
            
            color = colors.get(grp, 'gray')
            all_phases = []
            slopes = []
            
            # Plot Individual Lines
            for entry in entries:
                s = entry['s']
                p = entry['phases']
                ax.plot(s, p, color=color, alpha=0.15, linewidth=1)
                all_phases.append(p)
                
                # Slope per animal
                mask = ~np.isnan(p)
                if np.sum(mask) > 2:
                    slope, _, _, _, _ = linregress(s[mask], p[mask])
                    slopes.append(slope)
            
            # Mean Profile
            all_phases = np.array(all_phases)
            mean_profile = []
            s_axis = entries[0]['s']
            
            for col in range(all_phases.shape[1]):
                col_data = all_phases[:, col]
                valid = col_data[~np.isnan(col_data)]
                if valid.size > 0:
                    rads = (valid / 24.0) * 2 * np.pi
                    m_rad = circmean(rads, low=-np.pi, high=np.pi)
                    mean_profile.append((m_rad / (2*np.pi)) * 24.0)
                else:
                    mean_profile.append(np.nan)
            
            ax.plot(s_axis, mean_profile, color=color, linewidth=3, label=f"{grp} Mean")
            
            # Stats text
            if slopes:
                m_slope = np.mean(slopes)
                s_slope = sem(slopes)
                stats_text += f"{grp} Slope: {m_slope:.2f} ± {s_slope:.2f}\n"

        ax.legend(loc='upper left')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        ax_slider = fig.add_axes([0.25, 0.05, 0.50, 0.03])
        self.range_slider = Slider(ax=ax_slider, label="Phase Range (+/- h)", valmin=1.0, valmax=12.0, valinit=8.0)
        self.range_slider.on_changed(self.update_ylim)
        
        self.update_ylim(8.0) 

    def get_export_data(self):
        rows = []
        for entry in self.gradient_data:
            animal = entry['animal']
            group = entry.get('group', 'Unassigned') # Ensure group is captured
            s_vals = entry['s']
            p_vals = entry['phases']
            
            for i in range(len(s_vals)):
                val = p_vals[i]
                rows.append({
                    'Animal_ID': animal,
                    'Group': group, # Add Group column
                    'Anatomical_Pos_s': s_vals[i],
                    'Relative_Phase_CT': val,
                    'Bin_Index': i
                })
        df = pd.DataFrame(rows)
        return df, "phase_gradient_data.csv"

    def update_ylim(self, val):
        self.ax.set_ylim(-val, val)
        self.fig.canvas.draw_idle()

# --- ADD THIS CLASS TO THE END OF THE FILE ---

class RegionResultViewer(QtWidgets.QWidget):
    """
    Visualizes the results of the Region-Based Analysis.
    Tab 1: Atlas Map colored by Phase Difference.
    Tab 2: Statistical Table.
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
        
        self.detail_table = QtWidgets.QTableWidget()
        detail_layout.addWidget(self.detail_table)
        
        # Buttons for Details Tab
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_export_all = QtWidgets.QPushButton("Export Full Cell Data")
        self.btn_export_all.clicked.connect(self.export_full_cell_data)
        btn_layout.addWidget(self.btn_export_all)
        detail_layout.addLayout(btn_layout)
        
        self._draw_map()
        self._populate_table()
        self._populate_detail_table()

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

    def _populate_detail_table(self):
        """Populates the detail tab with per-animal data."""
        cols = ["Zone ID", "Zone Name", "Group", "Animal", "Mean Phase (h)", "N Cells"]
        self.detail_table.setColumnCount(len(cols))
        self.detail_table.setHorizontalHeaderLabels(cols)
        
        # Calculate total rows
        total_rows = sum(len(s['data']) for s in self.zone_stats)
        self.detail_table.setRowCount(total_rows)
        
        row_idx = 0
        for s in self.zone_stats:
            z_id = str(s['id'])
            z_name = str(s['name'])
            
            # Sort by group then animal for readability
            sorted_data = sorted(s['data'], key=lambda x: (x['group'], x['animal']))
            
            for d in sorted_data:
                self.detail_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(z_id))
                self.detail_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(z_name))
                self.detail_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(str(d['group'])))
                self.detail_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(d['animal'])))
                self.detail_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(f"{d['mean']:.2f}"))
                self.detail_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(str(d['n_cells'])))
                row_idx += 1
                
        self.detail_table.resizeColumnsToContents()

    def get_export_data(self):
        # Flatten stats to DF
        data = []
        for s in self.zone_stats:
            row = s.copy()
            if 'data' in row: del row['data'] 
            data.append(row)
        return pd.DataFrame(data), "region_stats.csv"

    def export_full_cell_data(self):
        """
        Exports the massive raw cell dump.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Full Cell Data", "region_raw_cells.csv", "CSV (*.csv)")
        if not path:
            return
            
        try:
            # Build huge list
            all_rows = []
            for s in self.zone_stats:
                z_id = s['id']
                z_name = s['name']
                for d in s['data']:
                    # d['raw_phases'] is a list or array
                    for ph in d.get('raw_phases', []):
                        all_rows.append({
                            'Zone_ID': z_id,
                            'Zone_Name': z_name,
                            'Group': d['group'],
                            'Animal': d['animal'],
                            'Cell_Phase_CT': ph
                        })
            
            df = pd.DataFrame(all_rows)
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Success", f"Exported {len(df)} cell records to:\n{path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))