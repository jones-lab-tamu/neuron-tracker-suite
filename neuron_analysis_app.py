# -*- coding: utf-8 -*-
"""
Neuron Tracker - Unified Analysis & Visualization Application

This single application provides a complete, end-to-end graphical workflow for
the analysis and exploration of cellular rhythmicity data. It is designed to be
a robust, user-friendly, and scientifically rigorous tool.
"""

# --- Standard Library Imports ---
import os
import sys
import argparse
import threading
import queue
import time
import json
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, Listbox, END

# --- Scientific Library Imports ---
import numpy
from numpy import arctan2, pi
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.path import Path

from scipy.stats import circmean
from scipy import signal
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from scipy.ndimage import generic_filter

from sklearn.decomposition import PCA
import colorcet as cet
from skimage.transform import ThinPlateSplineTransform

# --- Local Project Imports ---
import neuron_tracker_core as ntc

# --- GUI HELPER CLASSES ---

class Tooltip:
    """Creates a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget; self.text = text; self.tooltip_window = None
        widget.bind("<Enter>", self.enter); widget.bind("<Leave>", self.leave)
    def enter(self, event=None): self.schedule()
    def leave(self, event=None): self.unschedule(); self.hidetip()
    def schedule(self): self.unschedule(); self.id = self.widget.after(500, self.showtip)
    def unschedule(self):
        id = getattr(self, 'id', None)
        if id: self.widget.after_cancel(id)
    def showtip(self):
        if self.tooltip_window: return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25; y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True); self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Message(self.tooltip_window, text=self.text, aspect=2000, background="#ffffe0", relief='solid', borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def hidetip(self):
        if self.tooltip_window: self.tooltip_window.destroy()
        self.tooltip_window = None

class ROIDrawerWindow(tk.Toplevel):
    """A popup window for drawing complex inclusion/exclusion ROIs."""
    def __init__(self, master, bg_image, roi_data, output_basename, callback, vmin=None, vmax=None):
        super().__init__(master)
        self.title("Advanced ROI Definition Tool"); self.geometry("800x800")
        self.callback = callback; self.roi_data = roi_data; self.output_basename = output_basename
        self.rois = []; self.current_vertices = []; self.current_line = None
        self.mode = tk.StringVar(value="Include")
        main_frame = ttk.Frame(self); main_frame.pack(fill=tk.BOTH, expand=True)
        instructions = "Select mode, then click to draw a polygon.\nRight-click to remove last point. Click 'Finish Polygon' to close the loop."
        ttk.Label(main_frame, text=instructions, justify='center').pack(pady=5)
        self.fig = plt.Figure(); self.ax = self.fig.add_subplot(111)
        self.ax.imshow(bg_image, cmap='gray', vmin=vmin, vmax=vmax)
        self.ax.plot(roi_data[:, 0], roi_data[:, 1], '.', color='gray', markersize=2, alpha=0.5)
        self.ax.set_title("Click to Define ROI Polygon")
        canvas_frame = ttk.Frame(main_frame); canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        button_frame = ttk.Frame(main_frame); button_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(button_frame, text="Include", variable=self.mode, value="Include").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(button_frame, text="Exclude", variable=self.mode, value="Exclude").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Finish Polygon", command=self.finish_polygon).pack(side=tk.LEFT, expand=True, padx=10)
        self.confirm_button = ttk.Button(button_frame, text="Confirm All ROIs", command=self.confirm_rois)
        self.confirm_button.pack(side=tk.RIGHT, expand=True, padx=5)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1: self.current_vertices.append((event.xdata, event.ydata)); self.update_plot()
        elif event.button == 3:
            if self.current_vertices: self.current_vertices.pop(); self.update_plot()

    def update_plot(self):
        if self.current_line: self.current_line.pop(0).remove()
        if len(self.current_vertices) > 1:
            x, y = zip(*self.current_vertices)
            color = 'g-' if self.mode.get() == "Include" else 'r-'
            self.current_line = self.ax.plot(x, y, color)
        elif self.current_vertices:
            color = 'g+' if self.mode.get() == "Include" else 'r+'
            self.current_line = self.ax.plot(self.current_vertices[0][0], self.current_vertices[0][1], color)
        self.canvas.draw()

    def finish_polygon(self):
        if len(self.current_vertices) > 2:
            self.current_vertices.append(self.current_vertices[0])
            self.update_plot()
            self.rois.append({'path_vertices': self.current_vertices, 'mode': self.mode.get()})
            self.current_vertices = []; self.current_line = None
            self.ax.set_title(f"{len(self.rois)} ROI(s) defined. Draw another or Confirm.")
            self.canvas.draw()

    def confirm_rois(self):
        # --- Save the anatomical ROI polygons to a file ---
        if self.rois and self.output_basename:
            filepath = f"{self.output_basename}_anatomical_roi.json"
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.rois, f, indent=4)
                if hasattr(self.master, 'log_message'):
                    self.master.log_message(f"Saved anatomical ROI to {os.path.basename(filepath)}")
            except Exception as e:
                if hasattr(self.master, 'log_message'):
                    self.master.log_message(f"Error saving ROI file: {e}")

        if not self.rois:
            self.callback(None, None)
        else:
            final_mask = numpy.zeros(len(self.roi_data), dtype=bool)
            for roi in self.rois:
                roi['path'] = Path(roi['path_vertices'])
            include_rois = [roi for roi in self.rois if roi['mode'] == 'Include']
            if include_rois:
                for roi in include_rois: final_mask |= roi['path'].contains_points(self.roi_data)
            else:
                final_mask.fill(True)
            for roi in self.rois:
                if roi['mode'] == 'Exclude': final_mask &= ~roi['path'].contains_points(self.roi_data)
            
            filtered_indices = numpy.where(final_mask)[0]
            
            # --- NEW: Save the filtered ROI data points to a file ---
            if self.output_basename:
                filtered_filepath = f"{self.output_basename}_roi_filtered.csv"
                try:
                    filtered_data = self.roi_data[filtered_indices]
                    numpy.savetxt(filtered_filepath, filtered_data, delimiter=',')
                    if hasattr(self.master, 'log_message'):
                        self.master.log_message(f"Saved filtered ROI data to {os.path.basename(filtered_filepath)}")
                except Exception as e:
                    if hasattr(self.master, 'log_message'):
                        self.master.log_message(f"Error saving filtered ROI file: {e}")

            self.callback(filtered_indices, self.rois)
        self.destroy()

class RegistrationWindow(tk.Toplevel):
    """A popup window for landmark-based Thin Plate Spline registration."""
    def __init__(self, master, atlas_path, target_paths, log_callback):
        super().__init__(master)
        self.title("Atlas Registration Tool"); self.geometry("1200x700")
        self.atlas_path = atlas_path; self.target_paths = list(target_paths); self.log_callback = log_callback
        self.source_landmarks, self.dest_landmarks = [], []
        self.source_artists, self.dest_artists = [], []
        
        with open(self.atlas_path, 'r') as f:
            self.atlas_rois = json.load(f)
        
        main_frame = ttk.Frame(self); main_frame.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(main_frame); plot_frame.pack(fill=tk.BOTH, expand=True)
        button_frame = ttk.Frame(main_frame); button_frame.pack(fill=tk.X, pady=5)

        self.fig = plt.Figure(); self.fig.subplots_adjust(wspace=0.05)
        self.ax_atlas = self.fig.add_subplot(1, 2, 1)
        self.ax_target = self.fig.add_subplot(1, 2, 2)
        
        canvas_frame = ttk.Frame(plot_frame); canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cid_atlas = self.canvas.mpl_connect('button_press_event', self.on_click_atlas)
        self.cid_target = self.canvas.mpl_connect('button_press_event', self.on_click_target)

        ttk.Button(button_frame, text="Calculate Warp", command=self.calculate_warp).pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(button_frame, text="Save Warp & Next", command=self.save_and_next, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Landmarks", command=self.reset_landmarks).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
        self.load_next_target()

    def load_next_target(self):
        if not self.target_paths:
            self.log_callback("All targets have been registered."); self.destroy(); return
        
        self.current_target_path = self.target_paths.pop(0)
        self.log_callback(f"Registering: {os.path.basename(self.current_target_path)}")
        with open(self.current_target_path, 'r') as f:
            self.target_rois = json.load(f)
        self.reset_landmarks()
        self.title(f"Atlas Registration ({len(self.target_paths)+1} remaining)")

    def reset_landmarks(self):
        self.source_landmarks, self.dest_landmarks = [], []
        self.save_button.config(state='disabled')
        self.update_plots()

    def on_click_atlas(self, event):
        if event.inaxes != self.ax_atlas or event.button != 1: return
        if len(self.source_landmarks) > len(self.dest_landmarks): return
        self.source_landmarks.append((event.xdata, event.ydata))
        self.update_plots()

    def on_click_target(self, event):
        if event.inaxes != self.ax_target or event.button != 1: return
        if len(self.dest_landmarks) >= len(self.source_landmarks): return
        self.dest_landmarks.append((event.xdata, event.ydata))
        self.update_plots()

    def update_plots(self, preview_shapes=None, warp_vectors=None):
        for ax, artists in [(self.ax_atlas, self.source_artists), (self.ax_target, self.dest_artists)]:
            for artist in artists: artist.remove()
            artists.clear()
        
        self.ax_atlas.cla(); self.ax_target.cla()
        self.ax_atlas.set_title("Atlas SCN"); self.ax_target.set_title("Target SCN")
        
        # --- PLOTTING IN ORIGINAL COORDINATES ---
        for roi in self.atlas_rois:
            self.ax_atlas.plot(*zip(*roi['path_vertices']), color='black')
        for roi in self.target_rois:
            self.ax_target.plot(*zip(*roi['path_vertices']), color='black')

        for i, (x, y) in enumerate(self.source_landmarks):
            self.source_artists.append(self.ax_atlas.text(x, y, str(i+1), color='red', weight='bold', ha='center', va='center'))
        for i, (x, y) in enumerate(self.dest_landmarks):
            self.dest_artists.append(self.ax_target.text(x, y, str(i+1), color='red', weight='bold', ha='center', va='center'))
        
        if preview_shapes:
            for shape in preview_shapes:
                self.source_artists.append(self.ax_atlas.plot(*zip(*shape), color='cyan', linestyle='--')[0])
        
        if warp_vectors:
            origin_x, origin_y, dx, dy = warp_vectors
            self.dest_artists.append(self.ax_target.quiver(origin_x, origin_y, dx, dy, color='cyan', angles='xy', scale_units='xy', scale=1))

        self.ax_atlas.invert_yaxis(); self.ax_target.invert_yaxis()
        self.ax_atlas.autoscale_view(); self.ax_atlas.set_aspect('equal', adjustable='box')
        self.ax_target.autoscale_view(); self.ax_target.set_aspect('equal', adjustable='box')
        self.canvas.draw()

    def calculate_warp(self):
        if len(self.source_landmarks) < 3 or len(self.source_landmarks) != len(self.dest_landmarks):
            self.log_callback("Error: At least 3 pairs of corresponding landmarks are required."); return
        
        source_pts = numpy.array(self.source_landmarks)
        dest_pts = numpy.array(self.dest_landmarks)

        # --- THE FIX: Center and scale both sets of landmarks before warping ---
        source_centroid = source_pts.mean(axis=0)
        dest_centroid = dest_pts.mean(axis=0)
        source_centered = source_pts - source_centroid
        dest_centered = dest_pts - dest_centroid

        source_scale = numpy.sqrt(numpy.mean(numpy.sum(source_centered**2, axis=1)))
        dest_scale = numpy.sqrt(numpy.mean(numpy.sum(dest_centered**2, axis=1)))
        
        # Avoid division by zero if scale is zero
        if source_scale == 0 or dest_scale == 0:
            self.log_callback("Error: Cannot calculate scale from landmarks."); return

        source_norm = source_centered / source_scale
        dest_norm = dest_centered / dest_scale
        
        # --- Estimate the PURE SHAPE warp using the normalized landmarks ---
        self.tps = ThinPlateSplineTransform()
        self.tps.estimate(dest_norm, source_norm)
        
        # --- Store the full transformation parameters for saving ---
        self.warp_params = {
            'source_centroid': source_centroid.tolist(),
            'dest_centroid': dest_centroid.tolist(),
            'source_scale': source_scale,
            'dest_scale': dest_scale,
            'source_landmarks_norm': source_norm.tolist(),
            'dest_landmarks_norm': dest_norm.tolist()
        }

        # --- Generate previews ---
        # 1. Preview for warped shapes on Atlas panel (in original coordinates)
        warped_rois_preview = []
        if self.target_rois:
            for roi in self.target_rois:
                target_verts = numpy.array(roi['path_vertices'])
                target_verts_norm = ((target_verts - dest_centroid) / dest_scale)
                warped_verts_norm = self.tps(target_verts_norm)
                final_warped_verts = (warped_verts_norm * source_scale) + source_centroid
                warped_rois_preview.append(final_warped_verts)
        
        # 2. Preview for warp vectors on Target panel (now in normalized space)
        dx = source_norm[:, 0] - dest_norm[:, 0]
        dy = source_norm[:, 1] - dest_norm[:, 1]
        warp_vectors = (dest_norm[:, 0], dest_norm[:, 1], dx, dy)

        # --- Redraw plots in NORMALIZED space for the quiver plot preview ---
        self.update_plots_normalized(preview_shapes=warped_rois_preview, warp_vectors=warp_vectors)
        
        self.save_button.config(state='normal')

    def update_plots_normalized(self, preview_shapes, warp_vectors):
        """A new plotting function to show data in the normalized space."""
        source_centroid = numpy.array(self.warp_params['source_centroid'])
        dest_centroid = numpy.array(self.warp_params['dest_centroid'])
        source_scale = self.warp_params['source_scale']
        dest_scale = self.warp_params['dest_scale']

        for ax, artists in [(self.ax_atlas, self.source_artists), (self.ax_target, self.dest_artists)]:
            for artist in artists: artist.remove()
            artists.clear()
        
        self.ax_atlas.cla(); self.ax_target.cla()
        self.ax_atlas.set_title("Atlas SCN (Normalized)"); self.ax_target.set_title("Target SCN (Normalized)")

        # Draw normalized polygons
        for roi in self.atlas_rois:
            verts = (numpy.array(roi['path_vertices']) - source_centroid) / source_scale
            self.ax_atlas.plot(verts[:,0], verts[:,1], color='black')
        for roi in self.target_rois:
            verts = (numpy.array(roi['path_vertices']) - dest_centroid) / dest_scale
            self.ax_target.plot(verts[:,0], verts[:,1], color='black')

        # Draw normalized landmarks
        source_norm = numpy.array(self.warp_params['source_landmarks_norm'])
        dest_norm = numpy.array(self.warp_params['dest_landmarks_norm'])
        for i, (x, y) in enumerate(source_norm):
            self.source_artists.append(self.ax_atlas.text(x, y, str(i+1), color='red', weight='bold', ha='center', va='center'))
        for i, (x, y) in enumerate(dest_norm):
            self.dest_artists.append(self.ax_target.text(x, y, str(i+1), color='red', weight='bold', ha='center', va='center'))
        
        # Draw normalized preview shapes on Atlas
        if preview_shapes:
            for shape in preview_shapes:
                shape_norm = (shape - source_centroid) / source_scale
                self.source_artists.append(self.ax_atlas.plot(shape_norm[:,0], shape_norm[:,1], color='cyan', linestyle='--')[0])
        
        # Draw normalized warp vectors on Target
        if warp_vectors:
            origin_x, origin_y, dx, dy = warp_vectors
            self.dest_artists.append(self.ax_target.quiver(origin_x, origin_y, dx, dy, color='cyan', angles='xy', scale_units='xy', scale=1))

        # --- THE FIX: Invert the Y-axis for both normalized plots ---
        self.ax_atlas.invert_yaxis()
        self.ax_target.invert_yaxis()

        self.ax_atlas.autoscale_view(); self.ax_atlas.set_aspect('equal', adjustable='box')
        self.ax_target.autoscale_view(); self.ax_target.set_aspect('equal', adjustable='box')
        self.canvas.draw()

    def save_and_next(self):
        if not hasattr(self, 'warp_params'):
            self.log_callback("Error: Must calculate warp before saving."); return

        # --- THE FIX: Add the original, un-normalized landmarks to the save file ---
        self.warp_params['source_landmarks'] = self.source_landmarks
        self.warp_params['destination_landmarks'] = self.dest_landmarks
        
        output_path = self.current_target_path.replace('_anatomical_roi.json', '_warp_parameters.json')
        with open(output_path, 'w') as f:
            json.dump(self.warp_params, f, indent=4)
        
        self.log_callback(f"Saved warp parameters to {os.path.basename(output_path)}")
        self.load_next_target()
        
class WarpInspectorWindow(tk.Toplevel):
    """A popup window to visualize the 'before' and 'after' of applying a warp."""
    def __init__(self, master, atlas_roi_path, target_roi_path, original_points, warped_points, 
                 original_points_norm, warped_points_norm, warp_params, title):
        super().__init__(master)
        self.title("Warp Inspector")
        self.geometry("1200x700")

        # Store all data, both original and normalized
        self.original_points = original_points
        self.warped_points = warped_points
        self.original_points_norm = original_points_norm
        self.warped_points_norm = warped_points_norm
        self.atlas_roi_path = atlas_roi_path
        self.target_roi_path = target_roi_path
        self.warp_params = warp_params
        self.overlay_visible = tk.BooleanVar(value=False)
        self.overlay_artists = []

        main_frame = ttk.Frame(self); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = plt.Figure(figsize=(12, 6))
        self.fig.subplots_adjust(wspace=0.1, bottom=0.15)
        self.ax_before = self.fig.add_subplot(1, 2, 1)
        self.ax_after = self.fig.add_subplot(1, 2, 2)
        
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        overlay_check = ttk.Checkbutton(control_frame, text="Show Normalized Original Data as Overlay", variable=self.overlay_visible, command=self.toggle_overlay)
        overlay_check.pack(side=tk.LEFT, padx=10)
        Tooltip(overlay_check, "Toggle a transparent overlay of the normalized 'before' data on the 'After Warp' plot.")
        
        ttk.Button(control_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=10)

        self.draw_plots()
        self.fig.suptitle(f"Warp Inspection for: {title}", fontsize=16)

    def draw_plots(self):
        """Draws the main content of the 'before' and 'after' panels."""
        # --- Left Panel: "Before" (Original, Absolute Coordinates) ---
        self.ax_before.cla()
        self.ax_before.set_title("Before Warp: Original Data")
        with open(self.target_roi_path, 'r') as f:
            target_rois = json.load(f)
        for roi in target_rois:
            self.ax_before.plot(*zip(*roi['path_vertices']), color='black', linewidth=2)
        self.ax_before.scatter(self.original_points[:, 0], self.original_points[:, 1], s=10, alpha=0.7, c='blue')
        self.ax_before.set_aspect('equal', adjustable='box')
        self.ax_before.invert_yaxis()

        # --- Right Panel: "After" (Normalized, Centered Coordinates) ---
        self.ax_after.cla()
        self.ax_after.set_title("After Warp: Data in Normalized Atlas Space")
        
        # Load and draw the NORMALIZED atlas shape
        source_centroid = numpy.array(self.warp_params['source_centroid'])
        source_scale = self.warp_params['source_scale']
        with open(self.atlas_roi_path, 'r') as f:
            atlas_rois = json.load(f)
        for roi in atlas_rois:
            verts_norm = (numpy.array(roi['path_vertices']) - source_centroid) / source_scale
            self.ax_after.plot(verts_norm[:, 0], verts_norm[:, 1], color='black', linewidth=2)
            
        # Plot the NORMALIZED warped data
        self.ax_after.scatter(self.warped_points_norm[:, 0], self.warped_points_norm[:, 1], s=10, alpha=0.7, c='red')
        self.ax_after.set_aspect('equal', adjustable='box')
        self.ax_after.invert_yaxis()
        
        self.toggle_overlay() # Redraw overlay if needed
        self.canvas.draw()

    def toggle_overlay(self):
        """Callback for the overlay checkbox."""
        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists.clear()

        if self.overlay_visible.get():
            # --- Plot the NORMALIZED "Before" data on the "After" panel ---
            dest_centroid = numpy.array(self.warp_params['dest_centroid'])
            dest_scale = self.warp_params['dest_scale']
            
            # Plot the NORMALIZED original anatomical outline
            with open(self.target_roi_path, 'r') as f:
                target_rois = json.load(f)
            for roi in target_rois:
                verts_norm = (numpy.array(roi['path_vertices']) - dest_centroid) / dest_scale
                line = self.ax_after.plot(verts_norm[:, 0], verts_norm[:, 1], color='blue', linestyle='--', linewidth=2, alpha=0.5)[0]
                self.overlay_artists.append(line)
            
            # Plot the NORMALIZED original cell data points
            scatter = self.ax_after.scatter(self.original_points_norm[:, 0], self.original_points_norm[:, 1], s=10, alpha=0.3, c='blue', marker='x')
            self.overlay_artists.append(scatter)
        
        self.ax_after.autoscale_view()
        self.canvas.draw_idle()

# --- MAIN APPLICATION ---
class App(tk.Tk):
    """The main application class, orchestrating all modes and user interactions."""
    def __init__(self):
        super().__init__()
        self.title("Neuron Analysis Workspace"); self.geometry("1200x800")
        # --- State Management ---
        self.params, self.phase_params, self.loaded_data, self.visualization_widgets = {}, {}, {}, {}
        self.unfiltered_data = {}; self.filtered_indices = None; self.rois = None
        self.vmin, self.vmax = None, None # Central store for contrast settings
        self.progress_queue = queue.Queue()
        
        # --- Build UI ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_panel = self.create_control_panel(main_paned_window)
        main_paned_window.add(control_panel, weight=1)
        
        visualization_panel = self.create_visualization_panel(main_paned_window)
        main_paned_window.add(visualization_panel, weight=3)
        
        # Start listening for messages from the background thread
        self.check_progress_queue()
        
    # --- UI Construction Methods ---
    
    def embed_plot(self, tab_frame, fig):
        """Helper method to embed a Matplotlib figure into a Tkinter frame."""
        canvas = FigureCanvasTkAgg(fig, master=tab_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, tab_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return fig, canvas
    
    def set_busy_cursor(self, is_busy):
        """Changes the cursor to/from a busy-state icon."""
        if is_busy:
            self.config(cursor="watch")
            self.update_idletasks()
        else:
            self.config(cursor="")
            self.update_idletasks()

    def create_control_panel(self, parent):
        # This outer container holds the scrollbar and the canvas
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the main components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        panel = self.scrollable_frame # All subsequent widgets go here
        
        # Mode Switcher
        mode_frame = ttk.LabelFrame(panel, text="Mode", padding="10"); mode_frame.pack(fill=tk.X, pady=5, padx=5)
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Animal Analysis", variable=self.mode_var, value="single", command=self.switch_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Atlas Registration", variable=self.mode_var, value="register", command=self.switch_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Apply Warp to Data", variable=self.mode_var, value="apply_warp", command=self.switch_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Group Data Viewer", variable=self.mode_var, value="group_view", command=self.switch_mode).pack(anchor='w')
        
        # Frames for each mode's controls
        self.single_animal_frame = ttk.Frame(panel)
        self.registration_frame = ttk.Frame(panel)
        self.apply_warp_frame = ttk.Frame(panel)
        self.group_view_frame = ttk.Frame(panel)
        
        # Build the widgets for each mode's frame
        self.create_single_animal_widgets(self.single_animal_frame)
        self.create_registration_widgets(self.registration_frame)
        self.create_apply_warp_widgets(self.apply_warp_frame)
        self.create_group_view_widgets(self.group_view_frame)
        
        self.switch_mode() # Set initial visible frame
        
        # --- NEW: Create Log and Progress Bar in the main container, outside the scrollable area ---
        exec_frame = ttk.LabelFrame(container, text="Execution Log", padding="10")
        exec_frame.pack(fill=tk.X, side='bottom', pady=5)
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.rowconfigure(0, weight=1)
        
        self.log = scrolledtext.ScrolledText(exec_frame, height=8, state='disabled')
        self.log.grid(row=0, column=0, sticky="nsew", pady=5)
        Tooltip(self.log, "Displays progress and status messages during analysis.")
        self.progress = ttk.Progressbar(exec_frame, orient='horizontal', mode='determinate')
        self.progress.grid(row=1, column=0, sticky="ew", pady=2)
        
        return container

    def create_single_animal_widgets(self, parent):
        io_frame = ttk.LabelFrame(parent, text="File I/O", padding="10"); io_frame.pack(fill=tk.X, pady=5)
        self.create_io_widgets(io_frame)
        roi_frame = ttk.LabelFrame(parent, text="Region of Interest (ROI)", padding="10"); roi_frame.pack(fill=tk.X, pady=5)
        self.create_roi_widgets(roi_frame)
        analysis_frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding="10"); analysis_frame.pack(fill=tk.X, pady=5)
        self.create_analysis_param_widgets(analysis_frame)
        phase_frame = ttk.LabelFrame(parent, text="Phase Map Parameters", padding="10"); phase_frame.pack(fill=tk.X, pady=5)
        self.create_phase_param_widgets(phase_frame)
        exec_frame = ttk.LabelFrame(parent, text="Execution", padding="10"); exec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.create_exec_widgets(exec_frame)

    def create_registration_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Atlas Registration Setup", padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        atlas_frame = ttk.Frame(frame); atlas_frame.pack(fill=tk.X, pady=5); atlas_frame.columnconfigure(1, weight=1)
        self.atlas_path_var = tk.StringVar()
        ttk.Button(atlas_frame, text="Select Atlas...", command=self.select_atlas).grid(row=0, column=0, padx=5)
        Tooltip(atlas_frame.winfo_children()[0], "Select the '_anatomical_roi.json' file to use as the template.")
        ttk.Entry(atlas_frame, textvariable=self.atlas_path_var, state='readonly').grid(row=0, column=1, sticky='ew')
        target_frame = ttk.Frame(frame); target_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        target_frame.rowconfigure(0, weight=1); target_frame.columnconfigure(0, weight=1)
        self.target_listbox = Listbox(target_frame); self.target_listbox.grid(row=0, column=0, columnspan=2, sticky='nsew')
        btn_add = ttk.Button(target_frame, text="Add Target(s)...", command=self.add_targets); btn_add.grid(row=1, column=0, sticky='ew', padx=2, pady=5)
        Tooltip(btn_add, "Add one or more '_anatomical_roi.json' files to be warped.")
        btn_remove = ttk.Button(target_frame, text="Remove Selected", command=self.remove_target); btn_remove.grid(row=1, column=1, sticky='ew', padx=2, pady=5)
        Tooltip(btn_remove, "Remove the selected file from the list.")
        self.begin_reg_button = ttk.Button(frame, text="Begin Registration...", command=self.begin_registration, state='disabled')
        self.begin_reg_button.pack(fill=tk.X, pady=10)
        Tooltip(self.begin_reg_button, "Start the landmark-based warping process for the selected files.")

    def create_apply_warp_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Apply Warp Setup", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        list_frame = ttk.Frame(frame); list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.warp_files_listbox = Listbox(list_frame)
        self.warp_files_listbox.bind('<<ListboxSelect>>', lambda e: self.check_apply_warp_buttons_state())
        self.warp_files_listbox.grid(row=0, column=0, columnspan=2, sticky='nsew')
        
        btn_add = ttk.Button(list_frame, text="Add Warp Parameter File(s)...", command=self.add_warp_files)
        btn_add.grid(row=1, column=0, sticky='ew', padx=2, pady=5)
        Tooltip(btn_add, "Add one or more '_warp_parameters.json' files.")
        
        btn_remove = ttk.Button(list_frame, text="Remove Selected", command=self.remove_warp_file)
        btn_remove.grid(row=1, column=1, sticky='ew', padx=2, pady=5)
        Tooltip(btn_remove, "Remove the selected file from the list.")
        
        # --- NEW BUTTONS for execution ---
        exec_frame = ttk.Frame(frame); exec_frame.pack(fill=tk.X, pady=10)
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.columnconfigure(1, weight=1)

        self.inspect_warp_button = ttk.Button(exec_frame, text="Inspect Selected Warp...", command=self.inspect_warp, state='disabled')
        self.inspect_warp_button.grid(row=0, column=0, sticky='ew', padx=2)
        Tooltip(self.inspect_warp_button, "Visualize the result of applying the selected warp to its corresponding cell data.")

        self.apply_warp_button = ttk.Button(exec_frame, text="Apply All Warp(s)", command=self.apply_warps, state='disabled')
        self.apply_warp_button.grid(row=0, column=1, sticky='ew', padx=2)
        Tooltip(self.apply_warp_button, "Apply all listed warp transformations to their corresponding ROI data and save the results.")

    def create_group_view_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Group Data Setup", padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        list_frame = ttk.Frame(frame); list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        list_frame.rowconfigure(0, weight=1); list_frame.columnconfigure(0, weight=1)
        self.group_files_listbox = Listbox(list_frame); self.group_files_listbox.grid(row=0, column=0, columnspan=2, sticky='nsew')
        btn_add = ttk.Button(list_frame, text="Add Warped ROI File(s)...", command=self.add_group_files); btn_add.grid(row=1, column=0, sticky='ew', padx=2, pady=5)
        Tooltip(btn_add, "Add one or more '_roi_warped.csv' files for a group.")
        btn_remove = ttk.Button(list_frame, text="Remove Selected", command=self.remove_group_file); btn_remove.grid(row=1, column=1, sticky='ew', padx=2, pady=5)
        Tooltip(btn_remove, "Remove the selected file from the list.")
        param_frame = ttk.Frame(frame); param_frame.pack(fill=tk.X, pady=5); param_frame.columnconfigure(1, weight=1)
        label_grid = ttk.Label(param_frame, text="Grid Resolution:"); label_grid.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.group_grid_res_var = tk.StringVar(value="50"); entry_grid = ttk.Entry(param_frame, textvariable=self.group_grid_res_var)
        entry_grid.grid(row=0, column=1, sticky="ew", padx=5)
        Tooltip(entry_grid, "The resolution of the grid for the group average map (e.g., 50x50).")
        self.group_smooth_var = tk.BooleanVar(); smooth_check = ttk.Checkbutton(param_frame, text="Smooth to fill empty bins", variable=self.group_smooth_var)
        smooth_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        Tooltip(smooth_check, "If checked, interpolates to fill empty grid squares for a continuous map.")
        self.view_group_button = ttk.Button(frame, text="Generate Group Visualizations", command=self.generate_group_visualizations, state='disabled')
        self.view_group_button.pack(fill=tk.X, pady=10)

    def create_io_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        load_movie_btn = ttk.Button(parent, text="Load Movie...", command=self.load_movie)
        load_movie_btn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        Tooltip(load_movie_btn, "Select the raw movie file (.tif) to analyze or view.")
        ttk.Label(parent, text="Input File:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.input_file_var = tk.StringVar()
        in_entry = ttk.Entry(parent, textvariable=self.input_file_var, state='readonly')
        in_entry.grid(row=1, column=1, sticky="ew")
        Tooltip(in_entry, "The path to the selected movie file.")
        ttk.Label(parent, text="Output Basename:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.output_base_var = tk.StringVar()
        out_entry = ttk.Entry(parent, textvariable=self.output_base_var)
        out_entry.grid(row=2, column=1, sticky="ew")
        Tooltip(out_entry, "The base name for all generated output files (.csv, .npy).")

    def create_roi_widgets(self, parent):
        self.roi_button = ttk.Button(parent, text="Define Anatomical ROI...", command=self.open_roi_tool, state='disabled')
        self.roi_button.pack(side=tk.LEFT, expand=True, padx=5)
        Tooltip(self.roi_button, "Draw inclusion/exclusion polygons to select cells from a specific anatomical region.")
        self.clear_roi_button = ttk.Button(parent, text="Clear ROI Filter", command=self.clear_roi_filter, state='disabled')
        self.clear_roi_button.pack(side=tk.LEFT, expand=True, padx=5)
        Tooltip(self.clear_roi_button, "Revert all visualizations to show the full, unfiltered dataset.")

    def create_analysis_param_widgets(self, parent):
        frame = ttk.Frame(parent); frame.pack(fill=tk.X)
        save_btn = ttk.Button(frame, text="Save Params...", command=self.save_parameters)
        save_btn.pack(side=tk.LEFT, expand=True, padx=5, pady=2)
        Tooltip(save_btn, "Save the current analysis parameters to a configuration file.")
        load_btn = ttk.Button(frame, text="Load Params...", command=self.load_parameters)
        load_btn.pack(side=tk.LEFT, expand=True, padx=5, pady=2)
        Tooltip(load_btn, "Load analysis parameters from a configuration file.")
        
        notebook = ttk.Notebook(parent); notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        tooltips = {
            "sigma1": "WHAT IT IS: The size (in pixels) of the smaller Gaussian blur in the filter used to find cells. Conceptually, this should be close to the radius of the cells in your image.\nHOW TO TUNE IT: Decrease for smaller cells, increase for larger cells.\nTHE TRADE-OFF: Too small may detect noisy speckles; too large may miss small cells or merge adjacent ones.",
            "sigma2": "WHAT IT IS: The size (in pixels) of the larger Gaussian blur used for background subtraction. This should be significantly larger than the size of your cells.\nHOW TO TUNE IT: Increase if you have large, uneven variations in background brightness.\nTHE TRADE-OFF: Too small may accidentally subtract the cells themselves; too large may be ineffective at removing background.",
            "blur_sigma": "WHAT IT IS: The blur applied just before measuring cell brightness to rank features. This helps make the brightness measurement more robust to single-pixel noise.\nHOW TO TUNE IT: Should generally be a small value, close to sigma1.\nTHE TRADE-OFF: A value of 0 is acceptable, but small blurring (e.g., 1-2 pixels) is often more stable.",
            "max_features": "WHAT IT IS: The maximum number of brightest features to identify in any single frame.\nHOW TO TUNE IT: Set this higher than the maximum number of cells you expect to see in one frame.\nTHE TRADE-OFF: Too low will miss real cells; too high may include dim, noisy features and slightly slow down the analysis.",
            "search_range": "WHAT IT IS: The maximum number of frames the tracker will look backward in time to find a match for a cell.\nHOW TO TUNE IT: Increase if cells can disappear (e.g., due to dimming) for long periods and then reappear.\nTHE TRADE-OFF: Too high can increase the chance of incorrect matches and will slow down tracking; too low may result in fragmented tracks.",
            "cone_radius_base": "WHAT IT IS: The initial search radius (in pixels) for linking a cell to its position in the very next frame.\nHOW TO TUNE IT: Increase if cells move very rapidly between frames.\nTHE TRADE-OFF: Too small will fail to track fast-moving cells; too large increases the risk of a cell being incorrectly matched to a close neighbor.",
            "cone_radius_multiplier": "WHAT IT IS: A factor that allows the search radius to grow larger when linking across larger gaps in time.\nHOW TO TUNE IT: This is an advanced parameter. Leave at default unless you have very erratic, fast cell movement.\nTHE TRADE-OFF: Increasing this helps track very fast cells over gaps, but significantly raises the risk of incorrect linking.",
            "min_trajectory_length": "WHAT IT IS: The minimum required length of a track to be considered a valid cell, as a fraction of the total movie length (e.g., 0.08 = 8%).\nHOW TO TUNE IT: Increase to be more strict and only keep highly persistent cells. Decrease to include cells that are only clearly visible for part of the recording.\nTHE TRADE-OFF: Too high may discard valid transient cells; too low may include short, noisy, false-positive tracks.",
            "sampling_box_size": "WHAT IT IS: The side length (in pixels) of the square box used to measure a cell's brightness at each frame.\nHOW TO TUNE IT: Should be large enough to encompass the entire cell, typically about 2-3 times the cell's diameter.\nTHE TRADE-OFF: Too small will not capture the full signal; too large will include background noise from surrounding areas, reducing signal-to-noise.",
            "sampling_sigma": "WHAT IT IS: The blur applied within the sampling box to create a weighted measurement, giving more importance to the center pixels.\nHOW TO TUNE IT: Should be close to the cell's radius.\nTHE TRADE-OFF: A value of 0 gives a simple average. A good value (e.g., 2.0) makes the measurement more robust to slight tracking inaccuracies.",
            "max_interpolation_distance": "WHAT IT IS: A safety check. Any track that contains a frame-to-frame jump larger than this many pixels is discarded as a likely tracking error.\nHOW TO TUNE IT: Set this to a value slightly larger than the maximum plausible distance a cell could move in one frame.\nTHE TRADE-OFF: Too low may discard valid fast-moving cells; too high may fail to catch and remove erroneous tracks."
        }
        param_defs = {
            "Detection": [("sigma1", 3.0), ("sigma2", 20.0), ("blur_sigma", 2.0), ("max_features", 200)],
            "Tracking": [("search_range", 50), ("cone_radius_base", 1.5), ("cone_radius_multiplier", 0.125)],
            "Filtering": [("min_trajectory_length", 0.08), ("sampling_box_size", 15), ("sampling_sigma", 2.0), ("max_interpolation_distance", 5.0)]
        }
        for tab_name, params in param_defs.items():
            frame = ttk.Frame(notebook, padding="10"); notebook.add(frame, text=tab_name)
            frame.columnconfigure(1, weight=1)
            for i, (p_name, p_default) in enumerate(params):
                label = ttk.Label(frame, text=f"{p_name}:"); label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
                var = tk.StringVar(value=str(p_default)); entry = ttk.Entry(frame, textvariable=var)
                entry.grid(row=i, column=1, sticky="ew", padx=5)
                self.params[p_name] = (var, type(p_default)); Tooltip(label, tooltips.get(p_name, "No description available."))

    def create_phase_param_widgets(self, parent):
        parent.columnconfigure(1, weight=1)
        label_mpf = ttk.Label(parent, text="Minutes per Frame:"); label_mpf.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        var_mpf = tk.StringVar(value="15.0"); entry_mpf = ttk.Entry(parent, textvariable=var_mpf)
        entry_mpf.grid(row=0, column=1, sticky="ew", padx=5)
        self.phase_params['minutes_per_frame'] = (var_mpf, float); Tooltip(label_mpf, "REQUIRED: The sampling interval of the recording in minutes.")
        label_pmin = ttk.Label(parent, text="Period Min (hrs):"); label_pmin.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        var_pmin = tk.StringVar(value="22.0"); entry_pmin = ttk.Entry(parent, textvariable=var_pmin)
        entry_pmin.grid(row=1, column=1, sticky="ew", padx=5)
        self.phase_params['period_min'] = (var_pmin, float); Tooltip(label_pmin, "Optional: Constrain phase search to periods longer than this. Leave blank for unconstrained search.")
        label_pmax = ttk.Label(parent, text="Period Max (hrs):"); label_pmax.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        var_pmax = tk.StringVar(value="28.0"); entry_pmax = ttk.Entry(parent, textvariable=var_pmax)
        entry_pmax.grid(row=2, column=1, sticky="ew", padx=5)
        self.phase_params['period_max'] = (var_pmax, float); Tooltip(label_pmax, "Optional: Constrain phase search to periods shorter than this. Leave blank for unconstrained search.")
        label_grid = ttk.Label(parent, text="Grid Resolution:"); label_grid.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        var_grid = tk.StringVar(value="100"); entry_grid = ttk.Entry(parent, textvariable=var_grid)
        entry_grid.grid(row=3, column=1, sticky="ew", padx=5)
        self.phase_params['grid_resolution'] = (var_grid, int); Tooltip(label_grid, "The resolution (e.g., 100x100) of the grid for the Interpolated Phase Map. Higher values are more detailed but slower.")
        label_rhythm = ttk.Label(parent, text="Rhythmicity Threshold:"); label_rhythm.grid(row=4, column=0, sticky="w", padx=5, pady=2)
        var_rhythm = tk.StringVar(value="0.0"); entry_rhythm = ttk.Entry(parent, textvariable=var_rhythm)
        entry_rhythm.grid(row=4, column=1, sticky="ew", padx=5)
        self.phase_params['rhythm_threshold'] = (var_rhythm, float); Tooltip(label_rhythm, "The minimum FFT power required for a cell to be included in phase maps. Increase to filter out non-rhythmic cells.")
        
        self.regen_button = ttk.Button(parent, text="Regenerate Phase Maps", command=self.regenerate_phase_maps, state='disabled')
        self.regen_button.grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)
        Tooltip(self.regen_button, "Update the Phase Map and Interpolated Map tabs using the current parameters, without re-running the entire analysis.")

    def create_exec_widgets(self, parent):
        parent.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(parent, text="Run Full Analysis", command=self.start_analysis, state='disabled')
        self.run_button.grid(row=0, column=0, sticky="ew", pady=5)
        Tooltip(self.run_button, "Run the complete neuron tracking and analysis pipeline. This may take several minutes.")
        self.load_button = ttk.Button(parent, text="Load Existing Results", command=self.load_results, state='disabled')
        self.load_button.grid(row=1, column=0, sticky="ew", pady=5)
        Tooltip(self.load_button, "Load previously generated results for the selected movie to view visualizations.")
        self.export_button = ttk.Button(parent, text="Export Current Plot...", command=self.export_current_plot, state='disabled')
        self.export_button.grid(row=2, column=0, sticky="ew", pady=5)
        Tooltip(self.export_button, "Save the currently visible plot as a high-quality image file (PNG, PDF, SVG).")

    def create_visualization_panel(self, parent):
        panel = ttk.Frame(parent, padding="10")
        self.vis_notebook = ttk.Notebook(panel); self.vis_notebook.pack(fill=tk.BOTH, expand=True)
        for name in ["Heatmap", "Center of Mass", "Trajectory Inspector", "Phase Map", "Interpolated Map"]:
            frame = ttk.Frame(self.vis_notebook)
            self.vis_notebook.add(frame, text=name, state='disabled')
            ttk.Label(frame, text=f"{name} will appear here after analysis.").pack(padx=20, pady=20)
        return panel

    # --- Mode Switching ---
    def switch_mode(self):
        self.single_animal_frame.pack_forget(); self.registration_frame.pack_forget()
        self.apply_warp_frame.pack_forget(); self.group_view_frame.pack_forget()
        mode = self.mode_var.get()
        if mode == "single": self.single_animal_frame.pack(fill=tk.BOTH, expand=True)
        elif mode == "register": self.registration_frame.pack(fill=tk.BOTH, expand=True)
        elif mode == "apply_warp": self.apply_warp_frame.pack(fill=tk.BOTH, expand=True)
        elif mode == "group_view": self.group_view_frame.pack(fill=tk.BOTH, expand=True)

    # --- Single Animal Mode Methods ---
    def load_movie(self):
        self.set_busy_cursor(True)
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if filepath:
            self.input_file_var.set(filepath)
            basename = os.path.splitext(filepath)[0]
            self.output_base_var.set(basename)
            self.log_message(f"Loaded movie: {os.path.basename(filepath)}"); self.check_for_existing_results()
        self.set_busy_cursor(False)

    def check_for_existing_results(self):
        basename = self.output_base_var.get()
        if not basename: return
        files_exist = all(os.path.exists(f"{basename}{ext}") for ext in ["_traces.csv", "_roi.csv", "_trajectories.npy"])
        self.load_button.config(state='normal' if files_exist else 'disabled'); self.run_button.config(state='normal')

    def start_analysis(self):
        self.run_button.config(state="disabled"); self.load_button.config(state="disabled")
        self.log.config(state='normal'); self.log.delete('1.0', tk.END); self.log.config(state='disabled')
        self.progress['value'] = 0
        try:
            input_file = self.input_file_var.get(); output_basename = self.output_base_var.get()
            if not input_file: raise ValueError("Input file must be specified.")
            if not output_basename: raise ValueError("Output basename must be specified.")
            analysis_args = {name: p_type(var.get()) for name, (var, p_type) in self.params.items()}
        except ValueError as e:
            self.log_message(f"Error: Invalid analysis parameter - {e}"); self.check_for_existing_results(); return
        self.analysis_thread = threading.Thread(target=self.run_analysis_task, args=(input_file, output_basename, analysis_args))
        self.analysis_thread.daemon = True; self.analysis_thread.start()

    def load_results(self):
        self.set_busy_cursor(True)
        self.log_message("Loading existing results...")
        basename = self.output_base_var.get()
        try:
            self.unfiltered_data['traces'] = numpy.loadtxt(f"{basename}_traces.csv", delimiter=',')
            self.unfiltered_data['roi'] = numpy.loadtxt(f"{basename}_roi.csv", delimiter=',')
            self.unfiltered_data['trajectories'] = numpy.load(f"{basename}_trajectories.npy")
            movie = skimage.io.imread(self.input_file_var.get())
            self.unfiltered_data['background'] = movie[len(movie) // 2]
            self.log_message("Results loaded successfully.")
            self.apply_roi_filter(None, None)
            self.roi_button.config(state='normal'); self.regen_button.config(state='normal'); self.export_button.config(state='normal')
        except Exception as e:
            self.log_message(f"Error loading results: {e}")
        self.set_busy_cursor(False)

    def open_roi_tool(self):
        if 'background' not in self.unfiltered_data: self.log_message("Error: Load data before defining an ROI."); return
        output_basename = self.output_base_var.get()
        if not output_basename: self.log_message("Error: Output Basename must be set to save ROI."); return
        ROIDrawerWindow(self, self.unfiltered_data['background'], self.unfiltered_data['roi'], output_basename, self.apply_roi_filter, vmin=self.vmin, vmax=self.vmax)

    def apply_roi_filter(self, indices, rois=None):
        self.set_busy_cursor(True)
        self.filtered_indices = indices; self.rois = rois
        if self.filtered_indices is None:
            self.loaded_data = self.unfiltered_data.copy()
            self.log_message("ROI filter cleared. Showing all data.")
            self.clear_roi_button.config(state='disabled')
        else:
            self.loaded_data['roi'] = self.unfiltered_data['roi'][self.filtered_indices]
            self.loaded_data['trajectories'] = self.unfiltered_data['trajectories'][self.filtered_indices]
            trace_indices = numpy.concatenate(([0], self.filtered_indices + 1))
            self.loaded_data['traces'] = self.unfiltered_data['traces'][:, trace_indices]
            self.log_message(f"ROI filter applied. Showing {len(self.filtered_indices)} selected cells.")
            self.clear_roi_button.config(state='normal')
        self.populate_visualizations()
        self.set_busy_cursor(False)

    def clear_roi_filter(self):
        self.apply_roi_filter(None, None)

    def save_parameters(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filepath: return
        params_to_save = {name: var.get() for name, (var, p_type) in self.params.items()}
        with open(filepath, 'w') as f: json.dump(params_to_save, f, indent=4)
        self.log_message(f"Analysis parameters saved to {os.path.basename(filepath)}")

    def load_parameters(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not filepath: return
        with open(filepath, 'r') as f: loaded_params = json.load(f)
        for name, value in loaded_params.items():
            if name in self.params: self.params[name][0].set(value)
        self.log_message(f"Analysis parameters loaded from {os.path.basename(filepath)}")

    def export_current_plot(self):
        current_tab_index = self.vis_notebook.index(self.vis_notebook.select())
        vis_widget_keys = list(self.visualization_widgets.keys())
        if current_tab_index >= len(vis_widget_keys): return
        widget_key = vis_widget_keys[current_tab_index]
        fig = self.visualization_widgets[widget_key].fig
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")])
        if not filepath: return
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.log_message(f"Plot saved to {os.path.basename(filepath)}")
        except Exception as e:
            self.log_message(f"Error saving plot: {e}")
            
    # --- Registration Mode Methods ---
    def select_atlas(self):
        filepath = filedialog.askopenfilename(title="Select Atlas ROI File", filetypes=[("Anatomical ROI files", "*_anatomical_roi.json")])
        if filepath: self.atlas_path_var.set(filepath); self.check_reg_button_state()

    def add_targets(self):
        filepaths = filedialog.askopenfilenames(title="Select Target ROI Files", filetypes=[("Anatomical ROI files", "*_anatomical_roi.json")])
        for fp in filepaths: self.target_listbox.insert(END, fp)
        self.check_reg_button_state()

    def remove_target(self):
        selected_indices = self.target_listbox.curselection()
        for i in reversed(selected_indices): self.target_listbox.delete(i)
        self.check_reg_button_state()

    def check_reg_button_state(self):
        if self.atlas_path_var.get() and self.target_listbox.size() > 0:
            self.begin_reg_button.config(state='normal')
        else:
            self.begin_reg_button.config(state='disabled')

    def begin_registration(self):
        atlas_path = self.atlas_path_var.get()
        target_paths = self.target_listbox.get(0, END)
        RegistrationWindow(self, atlas_path, target_paths, self.log_message)

    # --- Apply Warp Mode Methods ---
    def add_warp_files(self):
        filepaths = filedialog.askopenfilenames(title="Select Warp Parameter Files", filetypes=[("Warp Parameters", "*_warp_parameters.json")])
        for fp in filepaths: self.warp_files_listbox.insert(END, fp)
        self.check_apply_warp_button_state()

    def remove_warp_file(self):
        selected_indices = self.warp_files_listbox.curselection()
        for i in reversed(selected_indices): self.warp_files_listbox.delete(i)
        self.check_apply_warp_button_state()

    def check_apply_warp_button_state(self):
        if self.warp_files_listbox.size() > 0:
            self.apply_warp_button.config(state='normal')
        else:
            self.apply_warp_button.config(state='disabled')

    def apply_warps(self):
        self.set_busy_cursor(True)
        warp_files = self.warp_files_listbox.get(0, END)
        if not warp_files:
            self.log_message("No warp files selected.")
        else:
            self.log_message(f"Starting to apply {len(warp_files)} warp(s)...")
            for warp_file in warp_files:
                try:
                    roi_file = warp_file.replace('_warp_parameters.json', '_roi_filtered.csv')
                    if not os.path.exists(roi_file):
                        self.log_message(f"Error: Cannot find corresponding '_roi_filtered.csv' for {os.path.basename(warp_file)}")
                        continue

                    with open(warp_file, 'r') as f:
                        warp_data = json.load(f)
                    
                    roi_points = numpy.loadtxt(roi_file, delimiter=',')
                    
                    # --- THE FIX: Use the correct '_norm' keys ---
                    tps = ThinPlateSplineTransform()
                    tps.estimate(numpy.array(warp_data['dest_landmarks_norm']), numpy.array(warp_data['source_landmarks_norm']))
                    
                    dest_centroid = numpy.array(warp_data['dest_centroid'])
                    dest_scale = warp_data['dest_scale']
                    source_centroid = numpy.array(warp_data['source_centroid'])
                    source_scale = warp_data['source_scale']

                    points_norm = (roi_points - dest_centroid) / dest_scale
                    warped_points_norm = tps(points_norm)
                    warped_points = (warped_points_norm * source_scale) + source_centroid
                    
                    output_path = roi_file.replace('_roi_filtered.csv', '_roi_warped.csv')
                    numpy.savetxt(output_path, warped_points, delimiter=',')
                    self.log_message(f"Successfully created {os.path.basename(output_path)}")

                except Exception as e:
                    self.log_message(f"Failed to process {os.path.basename(warp_file)}: {e}")
            self.log_message("Warp application complete.")
        self.set_busy_cursor(False)

    def add_warp_files(self):
        filepaths = filedialog.askopenfilenames(title="Select Warp Parameter Files", filetypes=[("Warp Parameters", "*_warp_parameters.json")])
        for fp in filepaths: self.warp_files_listbox.insert(END, fp)
        self.check_apply_warp_buttons_state()

    def remove_warp_file(self):
        selected_indices = self.warp_files_listbox.curselection()
        for i in reversed(selected_indices): self.warp_files_listbox.delete(i)
        self.check_apply_warp_buttons_state()

    def check_apply_warp_buttons_state(self):
        num_total = self.warp_files_listbox.size()
        num_selected = len(self.warp_files_listbox.curselection())
        
        if num_total > 0:
            self.apply_warp_button.config(state='normal')
        else:
            self.apply_warp_button.config(state='disabled')
            
        if num_selected == 1:
            self.inspect_warp_button.config(state='normal')
        else:
            self.inspect_warp_button.config(state='disabled')

    def inspect_warp(self):
        selected_indices = self.warp_files_listbox.curselection()
        if not selected_indices:
            self.log_message("Please select a single warp file to inspect.")
            return
        
        warp_file = self.warp_files_listbox.get(selected_indices[0])
        
        self.set_busy_cursor(True)
        try:
            atlas_path = self.atlas_path_var.get()
            if not atlas_path or not os.path.exists(atlas_path):
                raise ValueError("Please select a valid Atlas file in the 'Atlas Registration' tab first.")

            target_roi_filtered_file = warp_file.replace('_warp_parameters.json', '_roi_filtered.csv')
            target_anatomical_file = warp_file.replace('_warp_parameters.json', '_anatomical_roi.json')

            if not os.path.exists(target_roi_filtered_file):
                raise FileNotFoundError(f"Cannot find corresponding '_roi_filtered.csv' for {os.path.basename(warp_file)}")
            if not os.path.exists(target_anatomical_file):
                raise FileNotFoundError(f"Cannot find corresponding '_anatomical_roi.json' for {os.path.basename(warp_file)}")

            with open(warp_file, 'r') as f:
                warp_data = json.load(f)
            
            original_filtered_points = numpy.loadtxt(target_roi_filtered_file, delimiter=',')
            
            # Re-create the full transformation chain
            tps = ThinPlateSplineTransform()
            tps.estimate(numpy.array(warp_data['dest_landmarks_norm']), numpy.array(warp_data['source_landmarks_norm']))
            dest_centroid = numpy.array(warp_data['dest_centroid'])
            dest_scale = warp_data['dest_scale']
            source_centroid = numpy.array(warp_data['source_centroid'])
            source_scale = warp_data['source_scale']

            # Calculate both the normalized "before" and "after" points
            points_norm = (original_filtered_points - dest_centroid) / dest_scale
            warped_points_norm = tps(points_norm)
            
            # Also calculate the final, un-normalized warped points for the Group Viewer later
            warped_points = (warped_points_norm * source_scale) + source_centroid
            
            # Pass all necessary data, both normalized and original, to the inspector
            WarpInspectorWindow(self, 
                                atlas_path, 
                                target_anatomical_file,
                                original_filtered_points,
                                warped_points,
                                points_norm, # Pass normalized original points
                                warped_points_norm, # Pass normalized warped points
                                warp_data, # Pass the full dictionary for centroids/scales
                                title=os.path.basename(target_roi_filtered_file))

        except Exception as e:
            self.log_message(f"Error during warp inspection: {e}")
        finally:
            self.set_busy_cursor(False)

    # --- Group View Mode Methods ---
    def add_group_files(self):
        filepaths = filedialog.askopenfilenames(title="Select Warped ROI Files", filetypes=[("Warped ROI files", "*_roi_warped.csv")])
        for fp in filepaths: self.group_files_listbox.insert(END, fp)
        self.check_group_view_button_state()

    def remove_group_file(self):
        selected_indices = self.group_files_listbox.curselection()
        for i in reversed(selected_indices): self.group_files_listbox.delete(i)
        self.check_group_view_button_state()

    def check_group_view_button_state(self):
        if self.group_files_listbox.size() > 0:
            self.view_group_button.config(state='normal')
        else:
            self.view_group_button.config(state='disabled')

    def generate_group_visualizations(self):
        self.set_busy_cursor(True)
        self.log_message("Loading and processing group data...")
        warped_roi_files = self.group_files_listbox.get(0, END)
        
        all_rois = []
        all_phases = []
        
        try:
            # Get phase parameters from the single-animal panel
            phase_args = {name: p_type(var.get()) for name, (var, p_type) in self.phase_params.items() if var.get() and name not in ['grid_resolution', 'rhythm_threshold']}
            if 'minutes_per_frame' not in phase_args:
                raise ValueError("Minutes per Frame must be set.")

            for i, roi_file in enumerate(warped_roi_files):
                self.log_message(f"  Processing file {i+1}/{len(warped_roi_files)}: {os.path.basename(roi_file)}...")
                self.update_idletasks() # Force GUI to update log

                traces_file = roi_file.replace('_roi_warped.csv', '_traces.csv')
                if not os.path.exists(traces_file):
                    self.log_message(f"    Warning: Cannot find corresponding traces file. Skipping.")
                    continue
                
                warped_rois = numpy.loadtxt(roi_file, delimiter=',')
                traces_data = numpy.loadtxt(traces_file, delimiter=',')
                
                phases, period, _ = calculate_phases_fft(traces_data, **phase_args)
                
                phases_rad = (phases % period) * (2 * numpy.pi / period)
                mean_phase_rad = circmean(phases_rad)
                mean_phase_hours = mean_phase_rad * (period / (2 * numpy.pi))
                relative_phases = (phases - mean_phase_hours + period/2) % period - period/2
                
                all_rois.append(warped_rois)
                all_phases.append(relative_phases)

            if not all_rois:
                raise ValueError("No valid data could be loaded for the group.")

            pooled_rois = numpy.vstack(all_rois)
            pooled_phases = numpy.concatenate(all_phases)
            
            # Clear and prepare visualization tabs
            self.log_message("Generating group plots...")
            for i in range(self.vis_notebook.index("end")):
                tab_frame = self.vis_notebook.winfo_children()[i]
                for widget in tab_frame.winfo_children(): widget.destroy()
            
            # Enable only the first two tabs for group data
            self.vis_notebook.tab(0, state='normal'); self.vis_notebook.tab(1, state='normal')
            for i in [2, 3, 4]: self.vis_notebook.tab(i, state='disabled')

            # Populate Group Scatter Plot
            fig_scatter, _ = self.embed_plot(self.vis_notebook.winfo_children()[0], plt.Figure())
            self.visualization_widgets['group_scatter'] = GroupScatterViewer(fig_scatter, fig_scatter.add_subplot(111), pooled_rois, pooled_phases, period)

            # Populate Group Average Map
            grid_res = int(self.group_grid_res_var.get())
            do_smooth = self.group_smooth_var.get()
            fig_avg, _ = self.embed_plot(self.vis_notebook.winfo_children()[1], plt.Figure())
            self.visualization_widgets['group_average'] = GroupAverageMapViewer(fig_avg, fig_avg.add_subplot(111), pooled_rois, pooled_phases, period, grid_res, do_smooth)
            
            self.log_message("Group visualizations generated successfully.")

        except Exception as e:
            self.log_message(f"Error generating group visualizations: {e}")
        finally:
            # This ensures the cursor always returns to normal
            self.set_busy_cursor(False)

    # --- Visualization and Threading ---
    def on_contrast_change(self, vmin, vmax):
        self.vmin = vmin; self.vmax = vmax
        for key in ['traj', 'phase']:
            if key in self.visualization_widgets:
                self.visualization_widgets[key].update_contrast(vmin, vmax)

    def populate_visualizations(self):
        self.set_busy_cursor(True)
        self.log_message("Generating interactive plots...")
        self.vmin = numpy.min(self.unfiltered_data['background']); self.vmax = numpy.max(self.unfiltered_data['background'])
        for i in range(self.vis_notebook.index("end")):
            tab_frame = self.vis_notebook.winfo_children()[i]
            for widget in tab_frame.winfo_children(): widget.destroy()
            self.vis_notebook.tab(i, state='normal')
        
        fig_heatmap, _ = self.embed_plot(self.vis_notebook.winfo_children()[0], plt.Figure())
        self.visualization_widgets['heatmap'] = HeatmapViewer(fig_heatmap, fig_heatmap.add_subplot(111), self.loaded_data['traces'], self.loaded_data['roi'])
        fig_com, _ = self.embed_plot(self.vis_notebook.winfo_children()[1], plt.Figure())
        self.visualization_widgets['com'] = ContrastViewer(fig_com, fig_com.add_subplot(111), self.unfiltered_data['background'], self.loaded_data['roi'], self.on_contrast_change)
        fig_traj, _ = self.embed_plot(self.vis_notebook.winfo_children()[2], plt.Figure())
        self.visualization_widgets['traj'] = TrajectoryInspector(fig_traj, fig_traj.add_subplot(111), self.loaded_data['trajectories'], self.unfiltered_data['background'])
        
        self.regenerate_phase_maps()
        self.log_message("Visualizations ready.")
        self.set_busy_cursor(False)

    def regenerate_phase_maps(self):
        self.set_busy_cursor(True)
        self.log_message("Regenerating phase maps with current parameters...")
        try:
            for i in [3, 4]:
                tab_frame = self.vis_notebook.winfo_children()[i]
                for widget in tab_frame.winfo_children(): widget.destroy()
            phase_args = {name: p_type(var.get()) for name, (var, p_type) in self.phase_params.items() if var.get() and name not in ['grid_resolution', 'rhythm_threshold']}
            phases, period, rhythm_scores = calculate_phases_fft(self.loaded_data['traces'], **phase_args)
            rhythm_thresh_var, rhythm_thresh_type = self.phase_params['rhythm_threshold']
            rhythm_threshold = rhythm_thresh_type(rhythm_thresh_var.get())
            rhythmic_indices = numpy.where(rhythm_scores >= rhythm_threshold)[0]
            self.log_message(f"Found {len(rhythmic_indices)} cells passing rhythmicity threshold of {rhythm_threshold}.")
            final_roi = self.loaded_data['roi'][rhythmic_indices]; final_phases = phases[rhythmic_indices]
            if len(final_phases) == 0: raise ValueError("No cells passed the rhythmicity filter.")
            phases_rad = (final_phases % period) * (2 * numpy.pi / period)
            mean_phase_rad = circmean(phases_rad)
            mean_phase_hours = mean_phase_rad * (period / (2 * numpy.pi))
            relative_phases = (final_phases - mean_phase_hours + period/2) % period - period/2
            
            fig_phase, _ = self.embed_plot(self.vis_notebook.winfo_children()[3], plt.Figure())
            self.visualization_widgets['phase'] = PhaseMapViewer(fig_phase, fig_phase.add_subplot(111), self.unfiltered_data['background'], final_roi, relative_phases, period, vmin=self.vmin, vmax=self.vmax)
            
            grid_res_var, grid_res_type = self.phase_params['grid_resolution']
            grid_resolution = grid_res_type(grid_res_var.get())
            fig_interp, _ = self.embed_plot(self.vis_notebook.winfo_children()[4], plt.Figure())
            self.visualization_widgets['interp'] = InterpolatedMapViewer(fig_interp, fig_interp.add_subplot(111), final_roi, relative_phases, period, grid_resolution, rois=self.rois)
            
            self.log_message("Phase maps updated.")
        except Exception as e:
            self.log_message(f"Could not generate phase maps: {e}"); self.vis_notebook.tab(3, state='disabled'); self.vis_notebook.tab(4, state='disabled')
        self.set_busy_cursor(False)

    def log_message(self, message):
        self.log.config(state='normal'); self.log.insert(tk.END, message + '\n'); self.log.see(tk.END); self.log.config(state='disabled')

    def check_progress_queue(self):
        try:
            while True:
                message = self.progress_queue.get_nowait()
                if message == "ANALYSIS_COMPLETE": self.log_message("Analysis finished. Loading results for visualization."); self.load_results(); self.check_for_existing_results()
                elif message == "ANALYSIS_FAILED": self.check_for_existing_results()
                elif isinstance(message, float): self.progress['value'] = message * 100
                elif isinstance(message, str): self.log_message(message)
        except queue.Empty: pass
        finally: self.after(100, self.check_progress_queue)

    def progress_callback(self, message):
        progress_map = {"Stage 1/4": 0.0, "Stage 2/4": 0.4, "Stage 3/4": 0.7, "Stage 4/4": 0.8}
        for key, val in progress_map.items():
            if key in message: self.progress_queue.put(val)
        self.progress_queue.put(message)

    def run_analysis_task(self, input_file, output_basename, args):
        try:
            self.progress_callback(f"Loading data from {input_file}...")
            data = skimage.io.imread(input_file); data = ntc.rescale(data, 0.0, 1.0)
            ims, ids, trees, blob_lists = ntc.process_frames(data, sigma1=args['sigma1'], sigma2=args['sigma2'], blur_sigma=args['blur_sigma'], max_features=args['max_features'], progress_callback=self.progress_callback)
            graph, subgraphs = ntc.build_trajectories(blob_lists, trees, ids, search_range=args['search_range'], cone_radius_base=args['cone_radius_base'], cone_radius_multiplier=args['cone_radius_multiplier'], progress_callback=self.progress_callback)
            pruned_subgraphs, reverse_ids = ntc.prune_trajectories(graph, subgraphs, ids, progress_callback=self.progress_callback)
            com, traj, lines = ntc.extract_and_interpolate_data(ims, pruned_subgraphs, reverse_ids, min_trajectory_length=args['min_trajectory_length'], sampling_box_size=args['sampling_box_size'], sampling_sigma=args['sampling_sigma'], max_interpolation_distance=args['max_interpolation_distance'], progress_callback=self.progress_callback)
            if len(lines) == 0: self.progress_callback("\nProcessing complete, but no valid trajectories were found.")
            else:
                self.progress_callback(f"\nProcessing complete. Found {len(lines)} valid trajectories.")
                numpy.savetxt(f"{output_basename}_roi.csv", numpy.column_stack((com[:, 1], com[:, 0])), delimiter=',')
                self.progress_callback(f"Saved center-of-mass data.")
                numpy.savetxt(f"{output_basename}_traces.csv", numpy.column_stack((lines[0, :, 0], lines[:, :, 1].T)), delimiter=",")
                self.progress_callback(f"Saved intensity traces.")
                numpy.save(f"{output_basename}_trajectories.npy", traj)
                self.progress_callback(f"Saved full trajectory data.")
            self.progress_queue.put(1.0); time.sleep(0.1)
            self.progress_queue.put("ANALYSIS_COMPLETE")
        except Exception as e:
            self.progress_queue.put(f"\n--- ANALYSIS FAILED ---\nError: {e}"); import traceback; self.progress_queue.put(traceback.format_exc()); self.progress_queue.put("ANALYSIS_FAILED")

# --- Visualization & Phase Calculation Classes/Functions ---
class HeatmapViewer:
    def __init__(self, fig, ax, traces_data, roi_data):
        self.fig, self.ax, self.traces_data, self.roi_data = fig, ax, traces_data, roi_data
        self.prepare_data()
        self.image_artist = self.ax.imshow(self.normalized_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        self.ax.set_title("Spatially Sorted Intensity Heatmap"); self.ax.set_xlabel("Time (frames)")
        self.cbar = fig.colorbar(self.image_artist, ax=self.ax, label="Normalized Intensity")
        ax_radio = self.fig.add_axes([0.01, 0.7, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.radio_buttons = RadioButtons(ax_radio, ('Y-coord', 'Raster', 'PCA'))
        self.radio_buttons.on_clicked(self.on_sort_change); self.on_sort_change('Y-coord')
    def prepare_data(self):
        intensities = self.traces_data[:, 1:]
        if intensities.shape[1] == 0: self.normalized_data = numpy.array([[]]); return
        self.y_sort_indices = numpy.argsort(self.roi_data[:, 1])
        self.raster_sort_indices = numpy.lexsort((self.roi_data[:, 0], self.roi_data[:, 1]))
        if len(self.roi_data) > 1:
            pca = PCA(n_components=1); projected_coords = pca.fit_transform(self.roi_data)
            self.pca_sort_indices = numpy.argsort(projected_coords.flatten())
        else: self.pca_sort_indices = numpy.array([0])
        mins, maxs = numpy.min(intensities, axis=0), numpy.max(intensities, axis=0)
        range_denom = maxs - mins; range_denom[range_denom == 0] = 1
        self.normalized_data = (intensities - mins) / range_denom
    def on_sort_change(self, label):
        if self.normalized_data.shape[1] == 0: return
        sort_map = {'Y-coord': self.y_sort_indices, 'Raster': self.raster_sort_indices, 'PCA': self.pca_sort_indices}
        self.image_artist.set_data(self.normalized_data[:, sort_map.get(label)].T)
        self.ax.set_ylabel(f"Sorted Cells (by {label})"); self.fig.canvas.draw_idle()

class ContrastViewer:
    def __init__(self, fig, ax, bg_image, com_points, on_change_callback):
        self.fig, self.ax = fig, ax
        self.on_change_callback = on_change_callback
        self.image_artist = ax.imshow(bg_image, cmap='gray')
        if len(com_points) > 0:
            ax.plot(com_points[:, 0], com_points[:, 1], '.', color='red', markersize=3, alpha=0.8)
        ax.set_title("Center of Mass of All Trajectories")
        ax_contrast = fig.add_axes([0.25, 0.06, 0.65, 0.03]); ax_brightness = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        min_val, max_val = numpy.min(bg_image), numpy.max(bg_image)
        self.contrast_slider = Slider(ax=ax_contrast, label='Contrast', valmin=min_val, valmax=max_val, valinit=max_val)
        self.brightness_slider = Slider(ax=ax_brightness, label='Brightness', valmin=min_val, valmax=max_val, valinit=min_val)
        self.contrast_slider.on_changed(self.update)
        self.brightness_slider.on_changed(self.update)
        self.update(None)
    def update(self, val):
        vmin = self.brightness_slider.val; vmax = self.contrast_slider.val
        self.image_artist.set_clim(vmin, vmax); self.fig.canvas.draw_idle()
        if self.on_change_callback: self.on_change_callback(vmin, vmax)

class TrajectoryInspector:
    def __init__(self, fig, ax, trajectories, bg_image):
        self.fig, self.ax, self.trajectories, self.bg_image = fig, ax, trajectories, bg_image
        self.num_trajectories, self.index = len(trajectories), 0
        self.vmin, self.vmax = None, None # Initialize contrast state
        ax_prev = fig.add_axes([0.7, 0.025, 0.1, 0.04]); ax_next = fig.add_axes([0.81, 0.025, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous'); self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_trajectory); self.btn_next.on_clicked(self.next_trajectory)
        self.bg_artist = None; self.update()
    def update_contrast(self, vmin, vmax):
        self.vmin = vmin; self.vmax = vmax
        if self.bg_artist: self.bg_artist.set_clim(vmin, vmax); self.fig.canvas.draw_idle()
    def next_trajectory(self, event):
        if self.num_trajectories > 0: self.index = (self.index + 1) % self.num_trajectories; self.update()
    def prev_trajectory(self, event):
        if self.num_trajectories > 0: self.index = (self.index - 1 + self.num_trajectories) % self.num_trajectories; self.update()
    def update(self):
        self.ax.cla()
        self.bg_artist = self.ax.imshow(self.bg_image, cmap='gray', alpha=0.7, vmin=self.vmin, vmax=self.vmax)
        if self.num_trajectories > 0:
            traj = self.trajectories[self.index]
            self.ax.scatter(traj[:, 1], traj[:, 0], c=numpy.arange(traj.shape[0]), cmap='viridis', s=10)
            self.ax.set_title(f"Trajectory {self.index + 1} / {self.num_trajectories}")
        else: self.ax.set_title("No Trajectories to Display")
        self.ax.set_xlim(0, self.bg_image.shape[1]); self.ax.set_ylim(self.bg_image.shape[0], 0); self.fig.canvas.draw_idle()

class PhaseMapViewer:
    def __init__(self, fig, ax, bg_image, roi_data, relative_phases, period_hours, vmin=None, vmax=None):
        self.fig, self.ax, self.period_hours = fig, ax, period_hours
        self.bg_artist = ax.imshow(bg_image, cmap='gray', vmin=vmin, vmax=vmax)
        if len(roi_data) > 0:
            self.scatter = ax.scatter(roi_data[:, 0], roi_data[:, 1], c=relative_phases, cmap=cet.cm.cyclic_mygbm_30_95_c78, s=25, edgecolor='black', linewidth=0.5)
            self.cbar = fig.colorbar(self.scatter, ax=ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("Relative Peak Time (hours)", fontsize=12)
            ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
            max_range = self.period_hours / 2.0
            self.range_slider = Slider(ax=ax_slider, label='Phase Range (+/- hrs)', valmin=1.0, valmax=max_range, valinit=max_range)
            self.range_slider.on_changed(self.update_clim); self.update_clim(max_range)
        ax.set_title("Spatiotemporal Phase Map", fontsize=16); ax.set_xticks([]); ax.set_yticks([])
    def update_contrast(self, vmin, vmax):
        if self.bg_artist: self.bg_artist.set_clim(vmin, vmax); self.fig.canvas.draw_idle()
    def update_clim(self, val): 
        if hasattr(self, 'scatter'): self.scatter.set_clim(-val, val); self.fig.canvas.draw_idle()

class GroupScatterViewer:
    def __init__(self, fig, ax, roi_data, relative_phases, period_hours):
        self.fig, self.ax = fig, ax
        ax.set_title("Group Phase Distribution", fontsize=16)
        if len(roi_data) > 0:
            ax.scatter(roi_data[:, 0], roi_data[:, 1], c=relative_phases, cmap=cet.cm.cyclic_mygbm_30_95_c78, s=10, alpha=0.8)
        ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

class GroupAverageMapViewer:
    def __init__(self, fig, ax, roi_data, relative_phases, period_hours, grid_resolution, do_smooth):
        self.fig, self.ax = fig, ax
        ax.set_title("Group Average Phase Map", fontsize=16)
        
        x_coords, y_coords = roi_data[:, 0], roi_data[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Create grid bins
        grid_x = numpy.linspace(x_min, x_max, grid_resolution)
        grid_y = numpy.linspace(y_min, y_max, grid_resolution)
        
        binned_phases = numpy.full((grid_resolution, grid_resolution), numpy.nan)
        
        # Assign each point to a bin and calculate circular mean
        for i in range(grid_resolution - 1):
            for j in range(grid_resolution - 1):
                mask = (x_coords >= grid_x[i]) & (x_coords < grid_x[i+1]) & \
                       (y_coords >= grid_y[j]) & (y_coords < grid_y[j+1])
                
                if numpy.any(mask):
                    phases_in_bin = relative_phases[mask]
                    
                    # --- THE FIX: Correctly convert relative phases to radians ---
                    # The previous modulo operation incorrectly handled negative numbers.
                    # We shift the [-period/2, +period/2] range to [0, period] before converting.
                    phases_shifted = phases_in_bin + (period_hours / 2)
                    phases_rad = (phases_shifted / period_hours) * (2 * pi)
                    
                    mean_phase_rad = circmean(phases_rad)
                    
                    # Convert mean angle back to [0, period] range
                    mean_phase_shifted = (mean_phase_rad / (2 * pi)) * period_hours
                    # Shift back to the original [-period/2, +period/2] range
                    binned_phases[j, i] = mean_phase_shifted - (period_hours / 2)

        # Optional smoothing to fill empty bins
        if do_smooth:
            # The nan_circmean helper now needs to perform the same correct conversion
            def nan_circmean(data_in_hours):
                valid_data = data_in_hours[~numpy.isnan(data_in_hours)]
                if valid_data.size == 0:
                    return numpy.nan
                
                # Correctly convert to radians for circmean
                shifted_data = valid_data + (period_hours / 2)
                rad_data = (shifted_data / period_hours) * (2 * pi)
                
                mean_rad = circmean(rad_data)
                
                # Convert back to hours
                mean_shifted = (mean_rad / (2 * pi)) * period_hours
                return mean_shifted - (period_hours / 2)
            
            padded_grid = numpy.pad(binned_phases, pad_width=1, mode='constant', constant_values=numpy.nan)
            smoothed_grid = generic_filter(padded_grid, nan_circmean, footprint=numpy.ones((3,3)), mode='constant', cval=numpy.nan)
            smoothed_grid = smoothed_grid[1:-1, 1:-1]
            binned_phases[numpy.isnan(binned_phases)] = smoothed_grid[numpy.isnan(binned_phases)]

        im = ax.imshow(binned_phases, origin='lower', extent=[x_min, x_max, y_min, y_max],
                       cmap=cet.cm.cyclic_mygbm_30_95_c78)
        
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Relative Peak Time (hours)")
        im.set_clim(-period_hours/2.0, period_hours/2.0)

class InterpolatedMapViewer:
    def __init__(self, fig, ax, roi_data, relative_phases, period_hours, grid_resolution, rois=None):
        self.fig, self.ax, self.period_hours = fig, ax, period_hours
        ax.set_title("Interpolated Spatiotemporal Phase Map", fontsize=16)
        if len(roi_data) < 4:
            ax.text(0.5, 0.5, "Not enough data points (<4) to create an interpolated map.", ha='center', va='center'); return
        phase_angles_rad = (relative_phases / (period_hours / 2.0)) * pi
        x_components, y_components = numpy.cos(phase_angles_rad), numpy.sin(phase_angles_rad)
        x_coords, y_coords = roi_data[:, 0], roi_data[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_buf, y_buf = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
        grid_x, grid_y = numpy.mgrid[x_min-x_buf:x_max+x_buf:complex(grid_resolution), y_min-y_buf:y_max+y_buf:complex(grid_resolution)]
        grid_points = numpy.vstack([grid_x.ravel(), grid_y.ravel()]).T
        rbf_x = RBFInterpolator(roi_data, x_components, kernel='linear', smoothing=1.0)
        rbf_y = RBFInterpolator(roi_data, y_components, kernel='linear', smoothing=1.0)
        grid_x_interp, grid_y_interp = rbf_x(grid_points), rbf_y(grid_points)
        reconstructed_angles_rad = arctan2(grid_y_interp, grid_x_interp)
        grid_z = (reconstructed_angles_rad / pi) * (period_hours / 2.0)
        grid_z = grid_z.reshape(grid_x.shape)
        
        if rois:
            final_mask = numpy.zeros(grid_x.shape, dtype=bool)
            include_rois = [roi['path'] for roi in rois if roi['mode'] == 'Include']
            if include_rois:
                for path in include_rois:
                    final_mask |= path.contains_points(grid_points).reshape(grid_x.shape)
            else:
                if len(roi_data) > 2:
                    hull = ConvexHull(roi_data)
                    hull_path = Path(roi_data[hull.vertices])
                    final_mask = hull_path.contains_points(grid_points).reshape(grid_x.shape)
            for roi in rois:
                if roi['mode'] == 'Exclude':
                    final_mask &= ~roi['path'].contains_points(grid_points).reshape(grid_x.shape)
            grid_z[~final_mask] = numpy.nan
        elif len(roi_data) > 2:
            hull = ConvexHull(roi_data)
            hull_path = Path(roi_data[hull.vertices])
            mask = hull_path.contains_points(grid_points).reshape(grid_x.shape)
            grid_z[~mask] = numpy.nan

        self.image_artist = ax.imshow(grid_z.T, origin='lower', extent=[x_min-x_buf, x_max+x_buf, y_min-y_buf, y_max+y_buf], cmap=cet.cm.cyclic_mygbm_30_95_c78, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        self.cbar = fig.colorbar(self.image_artist, ax=ax, fraction=0.046, pad=0.04)
        # --- THE FIX: Corrected variable name from self.c_bar to self.cbar ---
        self.cbar.set_label("Relative Peak Time (hours)", fontsize=12)
        ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label='Phase Range (+/- hrs)', valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim); self.update_clim(max_range)
        
    def update_clim(self, val): self.image_artist.set_clim(-val, val); self.fig.canvas.draw_idle()

def calculate_phases_fft(traces_data, minutes_per_frame, period_min=None, period_max=None):
    intensities = traces_data[:, 1:]
    if intensities.shape[1] == 0: return numpy.array([]), 0, numpy.array([])
    hours_per_frame = minutes_per_frame / 60.0; sampling_rate_hz = 1.0 / hours_per_frame
    nyquist_freq = 0.5 * sampling_rate_hz; cutoff_freq = 1.0 / 40.0
    b, a = signal.butter(3, cutoff_freq / nyquist_freq, btype='high', analog=False)
    filtered_intensities = signal.filtfilt(b, a, intensities, axis=0)
    mean_signal = numpy.mean(filtered_intensities, axis=1)
    fft_pop = numpy.fft.fft(mean_signal)
    freqs = numpy.fft.fftfreq(intensities.shape[0], d=hours_per_frame)
    positive_freq_indices = numpy.where(freqs > 0)
    if period_min and period_max:
        freq_max, freq_min = 1.0 / period_min, 1.0 / period_max
        search_mask = (freqs[positive_freq_indices] >= freq_min) & (freqs[positive_freq_indices] <= freq_max)
        if not numpy.any(search_mask): raise ValueError("No FFT frequencies found in specified range.")
        peak_freq_index = numpy.where(search_mask)[0][numpy.argmax(numpy.abs(fft_pop[positive_freq_indices][search_mask]))]
    else: peak_freq_index = numpy.argmax(numpy.abs(fft_pop[positive_freq_indices]))
    dominant_freq = freqs[positive_freq_indices][peak_freq_index]
    if dominant_freq == 0: raise ValueError("Could not find a dominant frequency.")
    period_hours = 1.0 / dominant_freq
    
    phases, rhythm_scores = [], []
    for i in range(filtered_intensities.shape[1]):
        fft_cell = numpy.fft.fft(filtered_intensities[:, i])
        phase_angle_rad = numpy.angle(fft_cell[positive_freq_indices][peak_freq_index])
        phase_in_hours = -phase_angle_rad / (2 * numpy.pi * dominant_freq)
        phases.append(phase_in_hours)
        rhythm_scores.append(numpy.abs(fft_cell[positive_freq_indices][peak_freq_index]))
        
    return numpy.array(phases), period_hours, numpy.array(rhythm_scores)

if __name__ == "__main__":
    app = App()
    app.mainloop()