"""
Neuron Tracker - Unified Analysis & Visualization Application

"""

import os
import sys
import argparse
import threading
import queue
import time # Import the time module for the safeguard
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import numpy
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Slider, Button, RadioButtons

from scipy.stats import circmean
from scipy import signal
from sklearn.decomposition import PCA
import colorcet as cet

# Import the core processing functions
import neuron_tracker_core as ntc

# --- Tooltip Class for GUI Help ---
class Tooltip:
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

# --- Main Application Class ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuron Analysis Workspace"); self.geometry("1200x800")
        self.params, self.phase_params, self.loaded_data, self.visualization_widgets = {}, {}, {}, {}
        self.progress_queue = queue.Queue()
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_panel = self.create_control_panel(main_paned_window)
        main_paned_window.add(control_panel, weight=1)
        visualization_panel = self.create_visualization_panel(main_paned_window)
        main_paned_window.add(visualization_panel, weight=3)
        self.check_progress_queue()

    def create_control_panel(self, parent):
        panel = ttk.Frame(parent, padding="10")
        io_frame = ttk.LabelFrame(panel, text="File I/O", padding="10"); io_frame.pack(fill=tk.X, pady=5)
        self.create_io_widgets(io_frame)
        analysis_frame = ttk.LabelFrame(panel, text="Analysis Parameters", padding="10"); analysis_frame.pack(fill=tk.X, pady=5)
        self.create_analysis_param_widgets(analysis_frame)
        phase_frame = ttk.LabelFrame(panel, text="Phase Map Parameters", padding="10"); phase_frame.pack(fill=tk.X, pady=5)
        self.create_phase_param_widgets(phase_frame)
        exec_frame = ttk.LabelFrame(panel, text="Execution", padding="10"); exec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.create_exec_widgets(exec_frame)
        return panel

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

    def create_analysis_param_widgets(self, parent):
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

    def create_exec_widgets(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        self.run_button = ttk.Button(parent, text="Run Full Analysis", command=self.start_analysis, state='disabled')
        self.run_button.grid(row=0, column=0, sticky="ew", pady=5)
        Tooltip(self.run_button, "Run the complete neuron tracking and analysis pipeline. This may take several minutes.")
        self.load_button = ttk.Button(parent, text="Load Existing Results", command=self.load_results, state='disabled')
        self.load_button.grid(row=1, column=0, sticky="ew", pady=5)
        Tooltip(self.load_button, "Load previously generated results for the selected movie to view visualizations.")
        self.log = scrolledtext.ScrolledText(parent, height=10, state='disabled')
        self.log.grid(row=2, column=0, sticky="nsew", pady=5)
        Tooltip(self.log, "Displays progress and status messages during analysis.")
        self.progress = ttk.Progressbar(parent, orient='horizontal', mode='determinate')
        self.progress.grid(row=3, column=0, sticky="ew", pady=2)

    def create_visualization_panel(self, parent):
        panel = ttk.Frame(parent, padding="10")
        self.vis_notebook = ttk.Notebook(panel); self.vis_notebook.pack(fill=tk.BOTH, expand=True)
        for name in ["Heatmap", "Center of Mass", "Trajectory Inspector", "Phase Map"]:
            frame = ttk.Frame(self.vis_notebook)
            self.vis_notebook.add(frame, text=name, state='disabled')
            ttk.Label(frame, text=f"{name} will appear here after analysis.").pack(padx=20, pady=20)
        return panel

    def load_movie(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if not filepath: return
        self.input_file_var.set(filepath)
        basename = os.path.splitext(filepath)[0]
        self.output_base_var.set(basename)
        self.log_message(f"Loaded movie: {os.path.basename(filepath)}"); self.check_for_existing_results()

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
        self.log_message("Loading existing results...")
        basename = self.output_base_var.get()
        try:
            self.loaded_data['traces'] = numpy.loadtxt(f"{basename}_traces.csv", delimiter=',')
            self.loaded_data['roi'] = numpy.loadtxt(f"{basename}_roi.csv", delimiter=',')
            self.loaded_data['trajectories'] = numpy.load(f"{basename}_trajectories.npy")
            movie = skimage.io.imread(self.input_file_var.get())
            self.loaded_data['background'] = movie[len(movie) // 2]
            self.log_message("Results loaded successfully."); self.populate_visualizations()
        except Exception as e:
            self.log_message(f"Error loading results: {e}")

    def populate_visualizations(self):
        self.log_message("Generating interactive plots...")
        for i in range(self.vis_notebook.index("end")):
            tab_frame = self.vis_notebook.winfo_children()[i]
            for widget in tab_frame.winfo_children(): widget.destroy()
            self.vis_notebook.tab(i, state='normal')
        def embed_plot(tab_frame, fig):
            canvas = FigureCanvasTkAgg(fig, master=tab_frame); canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(canvas, tab_frame); toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        fig_heatmap = plt.Figure(); ax_heatmap = fig_heatmap.add_subplot(111)
        embed_plot(self.vis_notebook.winfo_children()[0], fig_heatmap)
        self.visualization_widgets['heatmap'] = HeatmapViewer(fig_heatmap, ax_heatmap, self.loaded_data['traces'], self.loaded_data['roi'])
        fig_com = plt.Figure(); ax_com = fig_com.add_subplot(111)
        embed_plot(self.vis_notebook.winfo_children()[1], fig_com)
        self.visualization_widgets['com'] = ContrastViewer(fig_com, ax_com, self.loaded_data['background'], self.loaded_data['roi'])
        fig_traj = plt.Figure(); ax_traj = fig_traj.add_subplot(111)
        embed_plot(self.vis_notebook.winfo_children()[2], fig_traj)
        self.visualization_widgets['traj'] = TrajectoryInspector(fig_traj, ax_traj, self.loaded_data['trajectories'], self.loaded_data['background'])
        try:
            phase_args = {name: p_type(var.get()) for name, (var, p_type) in self.phase_params.items() if var.get()}
            phases, period = calculate_phases_fft(self.loaded_data['traces'], **phase_args)
            phases_rad = (phases % period) * (2 * numpy.pi / period)
            mean_phase_rad = circmean(phases_rad)
            mean_phase_hours = mean_phase_rad * (period / (2 * numpy.pi))
            relative_phases = (phases - mean_phase_hours + period/2) % period - period/2
            fig_phase = plt.Figure(); ax_phase = fig_phase.add_subplot(111)
            embed_plot(self.vis_notebook.winfo_children()[3], fig_phase)
            self.visualization_widgets['phase'] = PhaseMapViewer(fig_phase, ax_phase, self.loaded_data['background'], self.loaded_data['roi'], relative_phases, period)
            self.log_message("Phase map generated successfully.")
        except Exception as e:
            self.log_message(f"Could not generate phase map: {e}"); self.vis_notebook.tab(3, state='disabled')
        self.log_message("Visualizations ready.")

    def log_message(self, message):
        self.log.config(state='normal'); self.log.insert(tk.END, message + '\n'); self.log.see(tk.END); self.log.config(state='disabled')

    def check_progress_queue(self):
        try:
            while True:
                message = self.progress_queue.get_nowait()
                # Check for special command strings FIRST
                if message == "ANALYSIS_COMPLETE":
                    self.log_message("Analysis finished. Loading results for visualization.")
                    self.load_results()
                    self.check_for_existing_results()
                elif message == "ANALYSIS_FAILED":
                    self.check_for_existing_results()
                # Then handle generic message types
                elif isinstance(message, float):
                    self.progress['value'] = message * 100
                elif isinstance(message, str):
                    self.log_message(message)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_progress_queue)

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
            
            self.progress_queue.put(1.0)
            time.sleep(0.1) # Give the OS a moment to finish writing files
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
        self.y_sort_indices = numpy.argsort(self.roi_data[:, 1])
        self.raster_sort_indices = numpy.lexsort((self.roi_data[:, 0], self.roi_data[:, 1]))
        pca = PCA(n_components=1); projected_coords = pca.fit_transform(self.roi_data)
        self.pca_sort_indices = numpy.argsort(projected_coords.flatten())
        mins, maxs = numpy.min(intensities, axis=0), numpy.max(intensities, axis=0)
        range_denom = maxs - mins; range_denom[range_denom == 0] = 1
        self.normalized_data = (intensities - mins) / range_denom
    def on_sort_change(self, label):
        sort_map = {'Y-coord': self.y_sort_indices, 'Raster': self.raster_sort_indices, 'PCA': self.pca_sort_indices}
        self.image_artist.set_data(self.normalized_data[:, sort_map.get(label)].T)
        self.ax.set_ylabel(f"Sorted Cells (by {label})"); self.fig.canvas.draw_idle()

class ContrastViewer:
    def __init__(self, fig, ax, bg_image, com_points):
        self.fig, self.ax = fig, ax
        self.image_artist = ax.imshow(bg_image, cmap='gray')
        ax.plot(com_points[:, 0], com_points[:, 1], '.', color='red', markersize=3, alpha=0.8)
        ax.set_title("Center of Mass of All Trajectories")
        ax_contrast = fig.add_axes([0.25, 0.06, 0.65, 0.03]); ax_brightness = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        min_val, max_val = numpy.min(bg_image), numpy.max(bg_image)
        self.contrast_slider = Slider(ax=ax_contrast, label='Contrast', valmin=min_val, valmax=max_val, valinit=max_val)
        self.brightness_slider = Slider(ax=ax_brightness, label='Brightness', valmin=min_val, valmax=max_val, valinit=min_val)
        self.contrast_slider.on_changed(self.update); self.brightness_slider.on_changed(self.update)
    def update(self, val): self.image_artist.set_clim(self.brightness_slider.val, self.contrast_slider.val); self.fig.canvas.draw_idle()

class TrajectoryInspector:
    def __init__(self, fig, ax, trajectories, bg_image):
        self.fig, self.ax, self.trajectories, self.bg_image = fig, ax, trajectories, bg_image
        self.num_trajectories, self.index = len(trajectories), 0
        ax_prev = fig.add_axes([0.7, 0.025, 0.1, 0.04]); ax_next = fig.add_axes([0.81, 0.025, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous'); self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_trajectory); self.btn_next.on_clicked(self.next_trajectory)
        self.update()
    def next_trajectory(self, event): self.index = (self.index + 1) % self.num_trajectories; self.update()
    def prev_trajectory(self, event): self.index = (self.index - 1 + self.num_trajectories) % self.num_trajectories; self.update()
    def update(self):
        self.ax.cla(); traj = self.trajectories[self.index]
        self.ax.imshow(self.bg_image, cmap='gray', alpha=0.7)
        self.ax.scatter(traj[:, 1], traj[:, 0], c=numpy.arange(traj.shape[0]), cmap='viridis', s=10)
        self.ax.set_title(f"Trajectory {self.index + 1} / {self.num_trajectories}")
        self.ax.set_xlim(0, self.bg_image.shape[1]); self.ax.set_ylim(self.bg_image.shape[0], 0); self.fig.canvas.draw_idle()

class PhaseMapViewer:
    def __init__(self, fig, ax, bg_image, roi_data, relative_phases, period_hours):
        self.fig, self.ax, self.period_hours = fig, ax, period_hours
        ax.imshow(bg_image, cmap='gray')
        self.scatter = ax.scatter(roi_data[:, 0], roi_data[:, 1], c=relative_phases, cmap=cet.cm.cyclic_mygbm_30_95_c78, s=25, edgecolor='black', linewidth=0.5)
        ax.set_title("Spatiotemporal Phase Map", fontsize=16); ax.set_xticks([]); ax.set_yticks([])
        self.cbar = fig.colorbar(self.scatter, ax=ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("Relative Peak Time (hours)", fontsize=12)
        ax_slider = fig.add_axes([0.25, 0.02, 0.65, 0.03])
        max_range = self.period_hours / 2.0
        self.range_slider = Slider(ax=ax_slider, label='Phase Range (+/- hrs)', valmin=1.0, valmax=max_range, valinit=max_range)
        self.range_slider.on_changed(self.update_clim); self.update_clim(max_range)
    def update_clim(self, val): self.scatter.set_clim(-val, val); self.fig.canvas.draw_idle()

def calculate_phases_fft(traces_data, minutes_per_frame, period_min=None, period_max=None):
    intensities = traces_data[:, 1:]
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
    phases = [-numpy.angle(numpy.fft.fft(filtered_intensities[:, i])[positive_freq_indices][peak_freq_index]) / (2 * numpy.pi * dominant_freq) for i in range(filtered_intensities.shape[1])]
    return numpy.array(phases), period_hours

if __name__ == "__main__":
    app = App()
    app.mainloop()