import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.path import Path
from scipy.stats import circmean

from gui.utils import Tooltip, clear_layout, add_mpl_to_tab
from gui.theme import get_icon
from gui.workers import AnalysisWorker, MovieLoaderWorker
from gui.dialogs.roi_drawer import ROIDrawerDialog
from gui.viewers import (
    HeatmapViewer,
    ContrastViewer,
    TrajectoryInspector,
    PhaseMapViewer,
    InterpolatedMapViewer,
)
from gui.analysis import (
    calculate_phases_fft,
    compute_median_window_frames,
    preprocess_for_rhythmicity,
    strict_cycle_mask,
    RHYTHM_TREND_WINDOW_HOURS,
)
import cosinor as csn

class SingleAnimalPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        
        # Local state for this panel
        self.params = {}
        self.phase_params = {}
        self.filtered_indices = None
        self.rois = None
        self.phase_reference_rois = None
        self.vmin = None
        self.vmax = None
        
        # Thread handles
        self._analysis_worker = None
        self._analysis_thread = None
        self._movie_loader_worker = None
        self._movie_loader_thread = None

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # File I/O
        io_box = QtWidgets.QGroupBox("File I/O")
        io_layout = QtWidgets.QGridLayout(io_box)
        self.btn_load_movie = QtWidgets.QPushButton(get_icon('fa5s.file-video'), "Load Movie...")
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

        # ROI
        roi_box = QtWidgets.QGroupBox("Region of Interest (ROI)")
        roi_layout = QtWidgets.QHBoxLayout(roi_box)
        self.btn_define_roi = QtWidgets.QPushButton(get_icon('fa5s.pen'), "Define Anatomical ROI...")
        self.btn_define_roi.setEnabled(False)
        Tooltip.install(self.btn_define_roi, "Launches the ROI drawing tool. This is required to create the `_anatomical_roi.json` file, which is a prerequisite for the Atlas Registration workflow.")
        self.btn_clear_roi = QtWidgets.QPushButton(get_icon('fa5s.times'), "Clear ROI Filter")
        self.btn_clear_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_define_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addWidget(roi_box)

        # Parameters
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
        self.btn_save_params = QtWidgets.QPushButton(get_icon('fa5s.download'), "Save Params...")
        self.btn_load_params = QtWidgets.QPushButton(get_icon('fa5s.upload'), "Load Params...")
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

        # Phase Map Parameters
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
        self.use_subregion_ref_check.setEnabled(False)
        phase_layout.addRow(self.use_subregion_ref_check)        

        self._add_phase_field(phase_layout, "minutes_per_frame", 15.0, tooltips=param_tooltips)
        self.discovered_period_edit = QtWidgets.QLineEdit("N/A")
        self.discovered_period_edit.setReadOnly(True)
        phase_layout.addRow("Discovered Period (hrs):", self.discovered_period_edit)
        self._add_phase_field(phase_layout, "period_min", 22.0, tooltips=param_tooltips)
        self._add_phase_field(phase_layout, "period_max", 28.0, tooltips=param_tooltips)
        self._add_phase_field(phase_layout, "trend_window_hours", 36.0, tooltips=param_tooltips)
        self._add_phase_field(phase_layout, "grid_resolution", 100, int, param_tooltips)
        
        _, self.rhythm_threshold_label = self._add_phase_field(
            phase_layout, "rhythm_threshold", 2.0, tooltips=param_tooltips,
        )
        rsquared_le, rsquared_label = self._add_phase_field(
            phase_layout, "r_squared_threshold", 0.3, tooltips=param_tooltips,
        )
        self.rsquared_widgets = (rsquared_label, rsquared_le)

        self.emphasize_rhythm_check = QtWidgets.QCheckBox("Emphasize rhythmic cells in all plots")
        Tooltip.install(
            self.emphasize_rhythm_check,
            "<b>What it does:</b> de-emphasizes non-rhythmic cells in all visualizations "
            "(Heatmap, Center of Mass, Phase Maps). Rhythmic cells remain fully visible.",
        )
        phase_layout.addRow(self.emphasize_rhythm_check)

        self.strict_cycle_check = QtWidgets.QCheckBox("Require >= 2 cycles (strict filter)")
        Tooltip.install(
            self.strict_cycle_check,
            "<b>What it does:</b> after FFT/Cosinor thresholds, only keeps cells that show at "
            "least two cycles near the target period, based on peak counting.",
        )
        phase_layout.addRow(self.strict_cycle_check)

        self.btn_regen_phase = QtWidgets.QPushButton(get_icon('fa5s.sync'), "Update Plots")
        Tooltip.install(self.btn_regen_phase, "Re-calculates all relevant plots (Heatmap, Center of Mass, Phase Maps).")
        self.btn_regen_phase.setEnabled(False)
        
        self.btn_save_rhythm = QtWidgets.QPushButton(get_icon('fa5s.save'), "Save Rhythm Results")
        Tooltip.install(self.btn_save_rhythm, "Saves the current rhythmicity status (Approved/Rejected) and phase data for every cell to a CSV file. This locks in these results for Group Analysis.")
        self.btn_save_rhythm.setEnabled(False)
        
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_regen_phase)
        btn_row.addWidget(self.btn_save_rhythm)
        phase_layout.addRow(btn_row)

        layout.addWidget(phase_box)
        layout.addStretch(1)
        
        self._on_analysis_method_changed(0)

    def connect_signals(self):
        self.btn_load_movie.clicked.connect(self.load_movie)
        self.output_base_edit.textChanged.connect(self.update_output_basename)
        self.btn_define_roi.clicked.connect(self.open_roi_tool)
        self.btn_clear_roi.clicked.connect(self.clear_roi_filter)
        self.btn_save_params.clicked.connect(self.save_parameters)
        self.btn_load_params.clicked.connect(self.load_parameters)
        
        self.mw.btn_run_analysis.clicked.connect(self.start_analysis)
        self.mw.btn_load_results.clicked.connect(self.load_results)
        self.mw.btn_export_plot.clicked.connect(self.export_current_plot)
        self.mw.btn_export_data.clicked.connect(self.export_current_data)
        
        self.btn_regen_phase.clicked.connect(self.regenerate_phase_maps)
        self.btn_save_rhythm.clicked.connect(self.save_rhythm_results)
        self.emphasize_rhythm_check.stateChanged.connect(self.regenerate_phase_maps)
        self.use_subregion_ref_check.stateChanged.connect(self.regenerate_phase_maps)
        self.analysis_method_combo.currentIndexChanged.connect(self._on_analysis_method_changed)

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

    def reset_state(self):
        self.filtered_indices = None
        self.rois = None
        self.vmin = None
        self.vmax = None
        self.btn_define_roi.setEnabled(False)
        self.btn_clear_roi.setEnabled(False)
        self.btn_regen_phase.setEnabled(False)
        self.status_traces_label.setText("Traces: —")
        self.status_roi_label.setText("ROI: —")
        self.status_traj_label.setText("Trajectories: —")

    def load_movie(self):
        self.mw._reset_state()
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Movie", start_dir, "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if not path:
            self.input_file_edit.clear()
            self.output_base_edit.clear()
            return
        
        self.mw._set_last_dir(path)
        self.state.input_movie_path = path
        self.input_file_edit.setText(path)
        base, _ = os.path.splitext(path)
        self.state.output_basename = base
        self.output_base_edit.setText(base)
        self.mw.workflow_state["has_input"] = True
        self.mw.log_message(f"Loaded movie: {os.path.basename(path)}")
        self.mw.update_workflow_from_files()

    def update_output_basename(self, text):
        self.state.output_basename = text
        self.mw.update_workflow_from_files()

    def start_analysis(self):
        if not self.state.input_movie_path or not self.state.output_basename:
            self.mw.log_message("Error: input file and output basename required.")
            return
        try:
            args = {name: t(le.text()) for name, (le, t) in self.params.items()}
        except ValueError as e:
            self.mw.log_message(f"Error in analysis parameters: {e}")
            return
        self.mw.log_text.clear()
        self.mw.progress_bar.setValue(0)
        self.mw.btn_run_analysis.setEnabled(False)
        self.mw.btn_load_results.setEnabled(False)
        self._analysis_worker = AnalysisWorker(self.state.input_movie_path, self.state.output_basename, args)
        self._analysis_thread = QtCore.QThread(self)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.message.connect(self.mw.log_message)
        self._analysis_worker.progress.connect(lambda v: self.mw.progress_bar.setValue(int(v * 100)))
        def done(success, msg):
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._analysis_worker = None
            self._analysis_thread = None
            if success:
                self.mw.log_message("Analysis finished. Loading results...")
                self.load_results()
            else:
                self.mw.log_message("Analysis failed.")
            self.mw.update_workflow_from_files()
        self._analysis_worker.finished.connect(done)
        self._analysis_thread.start()

    def load_results(self):
        basename = self.state.output_basename
        if not basename:
            self.mw.log_message("Error: Output basename not set.")
            return

        if self._movie_loader_thread is not None and self._movie_loader_thread.isRunning():
            self.mw.log_message("Movie is already being loaded.")
            return

        try:
            traces = np.loadtxt(f"{basename}_traces.csv", delimiter=",")
            roi = np.loadtxt(f"{basename}_roi.csv", delimiter=",")
            traj = np.load(f"{basename}_trajectories.npy")

            if isinstance(roi, np.ndarray) and roi.ndim == 1:
                roi = roi.reshape(1, -1)
            if isinstance(traces, np.ndarray) and traces.ndim == 1:
                traces = traces.reshape(1, -1)

            self.state.unfiltered_data["traces"] = traces
            self.state.unfiltered_data["roi"] = roi
            self.state.unfiltered_data["trajectories"] = traj

            self.mw.log_message("Loaded ROI and trace data.")
        except Exception as e:
            self.mw.log_message(f"Error loading result files: {e}")
            return

        self.btn_load_movie.setEnabled(False)
        self.mw.btn_load_results.setEnabled(False)
        self.mw.log_message("Loading movie in background... The UI will remain responsive.")

        self._movie_loader_worker = MovieLoaderWorker(self.state.input_movie_path)
        self._movie_loader_thread = QtCore.QThread(self)
        self._movie_loader_worker.moveToThread(self._movie_loader_thread)

        self._movie_loader_thread.started.connect(self._movie_loader_worker.run)
        self._movie_loader_worker.finished.connect(self._on_movie_loaded)
        self._movie_loader_worker.error.connect(self._on_movie_load_error)

        self._movie_loader_worker.finished.connect(self._movie_loader_thread.quit)
        self._movie_loader_thread.finished.connect(self._movie_loader_thread.deleteLater)
        self._movie_loader_worker.finished.connect(self._movie_loader_worker.deleteLater)

        self._movie_loader_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_movie_loaded(self, movie_data):
        self.mw.log_message("Movie loaded successfully.")
        self.state.unfiltered_data["movie"] = movie_data
        self.state.unfiltered_data["background"] = movie_data[len(movie_data) // 2]
        
        self.btn_load_movie.setEnabled(True)
        self.mw.btn_load_results.setEnabled(True)
        self.apply_roi_filter(None, None, None)
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    @QtCore.pyqtSlot(str)
    def _on_movie_load_error(self, error_message):
        self.mw.log_message(f"--- MOVIE LOAD FAILED ---\n{error_message}")
        self.btn_load_movie.setEnabled(True)
        self.mw.btn_load_results.setEnabled(True)
        if self._movie_loader_thread:
            self._movie_loader_thread.quit()
            self._movie_loader_thread.wait()
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    def open_roi_tool(self):
        if "background" not in self.state.unfiltered_data:
            self.mw.log_message("Error: Load data before defining an ROI.")
            return
        if not self.state.output_basename:
            self.mw.log_message("Error: Output Basename required.")
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
        self.mw.update_workflow_from_files()

    def apply_roi_filter(self, indices, rois, phase_ref_rois):
        self.filtered_indices = indices
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
        
        self.use_subregion_ref_check.setEnabled(bool(phase_ref_rois))
        if not phase_ref_rois:
            self.use_subregion_ref_check.setChecked(False)

        if indices is None:
            self.state.loaded_data = dict(self.state.unfiltered_data)
            self.mw.log_message("ROI filter cleared. Showing all data.")
            self.btn_clear_roi.setEnabled(False)
        else:
            self.state.loaded_data["roi"] = self.state.unfiltered_data["roi"][indices]
            self.state.loaded_data["trajectories"] = self.state.unfiltered_data["trajectories"][indices]
            trace_indices = np.concatenate(([0], indices + 1))
            self.state.loaded_data["traces"] = self.state.unfiltered_data["traces"][:, trace_indices]
            self.mw.log_message(f"ROI filter applied. {len(indices)} cells selected.")
            self.btn_clear_roi.setEnabled(True)
        
        self.populate_visualizations()

    def clear_roi_filter(self):
        self.apply_roi_filter(None, None, None)

    def save_parameters(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        data = {name: le.text() for name, (le, _) in self.params.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        self.mw.log_message(f"Parameters saved to {os.path.basename(path)}")

    def load_parameters(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Parameters", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        with open(path, "r") as f:
            data = json.load(f)
        for name, value in data.items():
            if name in self.params:
                self.params[name][0].setText(str(value))
        self.mw.log_message(f"Parameters loaded from {os.path.basename(path)}")

    def export_current_plot(self):
        widget = self.mw.vis_tabs.currentWidget()
        viewer = self.mw.visualization_widgets.get(widget)
        if not viewer or not hasattr(viewer, "fig"):
            self.mw.log_message("No active figure to export.")
            return
        fig = viewer.fig
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot", start_dir,
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            self.mw.log_message(f"Plot saved to {os.path.basename(path)}")
        except Exception as e:
            self.mw.log_message(f"Error saving plot: {e}")

    def export_current_data(self):
        # This was missing in the original snippet but referenced in signals.
        # Implementing a basic version or placeholder.
        self.mw.log_message("Export data not fully implemented in this refactor step.")

    def on_roi_selected(self, original_index):
        self.mw.log_message(f"ROI {original_index + 1} selected.")
        local_index = original_index
        if self.filtered_indices is not None:
            matches = np.where(self.filtered_indices == original_index)[0]
            if len(matches) > 0:
                local_index = matches[0]
            else:
                local_index = -1

        traj_viewer = self.mw.visualization_widgets.get(self.mw.traj_tab)
        if traj_viewer and local_index != -1:
            traj_viewer.set_trajectory(local_index)
            self.mw.vis_tabs.setCurrentWidget(self.mw.traj_tab)

        com_viewer = self.mw.visualization_widgets.get(self.mw.com_tab)
        if com_viewer:
            com_viewer.highlight_point(original_index)
        
        heatmap_viewer = self.mw.visualization_widgets.get(self.mw.heatmap_tab)
        if heatmap_viewer:
            heatmap_viewer.update_selected_trace(original_index)

        phase_viewer = self.mw.visualization_widgets.get(self.mw.phase_tab)
        if phase_viewer:
            phase_viewer.highlight_point(original_index)

    def on_contrast_change(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        traj_viewer = self.mw.visualization_widgets.get(self.mw.traj_tab)
        if traj_viewer:
            traj_viewer.update_contrast(vmin, vmax)
        phase_viewer = self.mw.visualization_widgets.get(self.mw.phase_tab)
        if phase_viewer:
            phase_viewer.update_contrast(vmin, vmax)

    def populate_visualizations(self):
        if not self.state.loaded_data or "background" not in self.state.unfiltered_data:
            return
        self.mw.log_message("Generating interactive plots...")
        bg = self.state.unfiltered_data["background"]
        movie = self.state.unfiltered_data.get("movie")
        if movie is None:
            self.mw.log_message("Error: Full movie stack not found in state.")
            return

        self.vmin, self.vmax = float(bg.min()), float(bg.max())
        single_animal_tabs = [self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab]
        group_tabs = [self.mw.group_scatter_tab, self.mw.group_avg_tab]
        for tab in single_animal_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), True)
        for tab in group_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
        
        try:
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = self._calculate_rhythms()
        except Exception as e:
            self.mw.log_message(f"Could not calculate rhythms: {e}")
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = None, None, None, None, True

        is_emphasized = self.emphasize_rhythm_check.isChecked()
        
        fig_h, _ = add_mpl_to_tab(self.mw.heatmap_tab)
        viewer_h = HeatmapViewer(fig_h, self.state.loaded_data, self.filtered_indices, phases, sort_scores, is_emphasized, rhythm_sort_desc,
                                 period=period, minutes_per_frame=None, reference_phase=None)
        self.mw.visualization_widgets[self.mw.heatmap_tab] = viewer_h
        
        fig_c, _ = add_mpl_to_tab(self.mw.com_tab)
        current_rois = []
        if self.rois: current_rois.extend(self.rois)
        if self.phase_reference_rois: current_rois.extend(self.phase_reference_rois)

        viewer_c = ContrastViewer(
            fig_c, fig_c.add_subplot(111), bg, self.state.loaded_data["roi"], 
            self.on_contrast_change, self.on_roi_selected, 
            filtered_indices=self.filtered_indices, rois=current_rois
        )
        self.mw.visualization_widgets[self.mw.com_tab] = viewer_c
        
        fig_t, _ = add_mpl_to_tab(self.mw.traj_tab)
        viewer_t = TrajectoryInspector(fig_t, fig_t.add_subplot(111), self.state.loaded_data["trajectories"], movie)
        self.mw.visualization_widgets[self.mw.traj_tab] = viewer_t
        
        self.regenerate_phase_maps()
        self.mw.btn_export_plot.setEnabled(True)
        self.mw.btn_export_data.setEnabled(True)

    def _calculate_rhythms(self):
        method = self.analysis_method_combo.currentText()
        self.mw.log_message(f"Calculating rhythms using {method} method...")

        phase_args = {
            name: t(w.text())
            for name, (w, t) in self.phase_params.items()
            if w.text() and name not in ("grid_resolution", "rhythm_threshold", "r_squared_threshold")
        }
        if not phase_args.get("minutes_per_frame"):
            raise ValueError("Minutes per frame is required.")

        if "trend_window_hours" in phase_args:
            phase_args["detrend_window_hours"] = phase_args.pop("trend_window_hours")

        _, discovered_period, _ = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
        self.discovered_period_edit.setText(f"{discovered_period:.2f}")

        if "FFT" in method:
            phases, period, snr_scores = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
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
            trend_window_hours = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
            median_window_frames = compute_median_window_frames(minutes_per_frame, trend_window_hours, T=T)

            for i in range(1, traces.shape[1]):
                raw_intensity = traces[:, i]
                intensity = preprocess_for_rhythmicity(raw_intensity, method=detrend_method, median_window_frames=median_window_frames)
                result = csn.cosinor_analysis(intensity, time_points_hours, period=discovered_period)
                phases.append(result["acrophase"])
                p_values.append(result["p_value"])
                r_squareds.append(result["r_squared"])

            return np.array(phases), discovered_period, np.array(r_squareds), np.array(p_values), True
        return None, None, None, None, True

    @QtCore.pyqtSlot(int)
    def _on_analysis_method_changed(self, index):
        method = self.analysis_method_combo.currentText()
        thresh_edit = self.phase_params["rhythm_threshold"][0]
        if "FFT" in method:
            self.rhythm_threshold_label.setText("Rhythm SNR Threshold (>=):")
            thresh_edit.setText("2.0")
            if hasattr(self, 'rsquared_widgets'):
                for widget in self.rsquared_widgets: widget.hide()
        elif "Cosinor" in method:
            self.rhythm_threshold_label.setText("Rhythm p-value (<=):")
            thresh_edit.setText("0.05")
            if hasattr(self, 'rsquared_widgets'):
                for widget in self.rsquared_widgets: widget.show()

    def regenerate_phase_maps(self):
        if not self.state.loaded_data or "traces" not in self.state.loaded_data:
            if self.emphasize_rhythm_check.isChecked(): self.mw.log_message("Load data before enabling rhythm emphasis.")
            return
        self.mw.log_message("Updating plots based on phase parameters...")
        for tab in (self.mw.phase_tab, self.mw.interp_tab): clear_layout(tab.layout())
        
        # Disable save button until calculation is done
        self.btn_save_rhythm.setEnabled(False)
        self.latest_rhythm_df = None # Reset storage        
        
        try:
            phases, period, sort_scores, filter_scores, sort_desc = self._calculate_rhythms()
            if phases is None: raise ValueError("Rhythm calculation failed.")
            method = self.analysis_method_combo.currentText()
            thresh = float(self.phase_params["rhythm_threshold"][0].text())
            mpf = float(self.phase_params["minutes_per_frame"][0].text())
            try:
                trend_win = float(self.phase_params["trend_window_hours"][0].text())
            except:
                trend_win = 36.0

            if "Cosinor" in method:
                r_thresh = float(self.phase_params["r_squared_threshold"][0].text())
                rhythm_mask = (filter_scores <= thresh) & (sort_scores >= r_thresh)
                self.mw.log_message(f"Applying Cosinor filter: p <= {thresh} AND R² >= {r_thresh}")
                phases_hours = phases
            else:
                rhythm_mask = filter_scores >= thresh
                self.mw.log_message(f"Applying FFT filter: SNR >= {thresh}")
                phases_hours = ((phases / (2 * np.pi)) * period) % period

            if self.strict_cycle_check.isChecked():
                rhythm_mask = strict_cycle_mask(
                    self.state.loaded_data["traces"], minutes_per_frame=mpf, period_hours=period,
                    base_mask=rhythm_mask, min_cycles=2, trend_window_hours=trend_win
                )
                self.mw.log_message("Strict cycle filter applied: requiring >= 2 cycles.")

            is_emphasized = self.emphasize_rhythm_check.isChecked()
            com_viewer = self.mw.visualization_widgets.get(self.mw.com_tab)
            if com_viewer:
                com_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)
            
            rhythmic_indices_relative = np.where(rhythm_mask)[0]
            self.mw.log_message(f"{len(rhythmic_indices_relative)} cells pass rhythmicity threshold(s).")
            
            if len(rhythmic_indices_relative) == 0:
                for t in [self.mw.phase_tab, self.mw.interp_tab]:
                    fig, canvas = add_mpl_to_tab(t)
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, "No cells passed the rhythmicity filter.", ha='center', va='center')
                    canvas.draw()
                self.mw.log_message("No cells passed rhythmicity filter.")
                empty_df = pd.DataFrame()
                fig_p, _ = add_mpl_to_tab(self.mw.phase_tab) 
                viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], empty_df, None, vmin=self.vmin, vmax=self.vmax)
                self.mw.visualization_widgets[self.mw.phase_tab] = viewer_p
                heatmap_viewer = self.mw.visualization_widgets.get(self.mw.heatmap_tab)
                if heatmap_viewer:
                     heatmap_viewer.update_phase_data(phases_hours, sort_scores, rhythm_mask, sort_desc,
                                                 period=period, minutes_per_frame=mpf, 
                                                 reference_phase=None, trend_window_hours=trend_win)
                return

            mean_h = 0.0
            def calc_circ_mean_hours(h_vals, p):
                rads = (h_vals % p) * (2 * np.pi / p)
                m_rad = circmean(rads)
                m_h = m_rad * (p / (2 * np.pi))
                return m_h

            if self.use_subregion_ref_check.isChecked() and self.phase_reference_rois:
                self.mw.log_message("Using drawn sub-region as phase reference.")
                rhythmic_coords = self.state.loaded_data['roi'][rhythmic_indices_relative]
                ref_mask = np.zeros(len(rhythmic_coords), dtype=bool)
                for roi in self.phase_reference_rois:
                    if "path" in roi: path = roi["path"]
                    else: path = Path(roi['path_vertices'])
                    ref_mask |= path.contains_points(rhythmic_coords)
                ref_indices_in_rhythmic_array = np.where(ref_mask)[0]
                self.mw.log_message(f"  -> Found {len(ref_indices_in_rhythmic_array)} rhythmic cells inside reference ROI.")

                if len(ref_indices_in_rhythmic_array) > 0:
                    ref_phases = phases_hours[rhythmic_indices_relative][ref_indices_in_rhythmic_array]
                    mean_h = calc_circ_mean_hours(ref_phases, period)
                    self.mw.log_message(f"  -> Reference Mean Phase: {mean_h:.2f} hours")
                else:
                    self.mw.log_message("  -> Warning: Reference ROI is empty of rhythmic cells. Falling back to global mean.")
                    mean_h = calc_circ_mean_hours(phases_hours[rhythmic_indices_relative], period)
                    self.mw.log_message(f"  -> Global Mean Phase: {mean_h:.2f} hours")
            else:
                if self.use_subregion_ref_check.isChecked():
                    self.mw.log_message("Sub-region reference selected, but no sub-region is defined. Using global mean.")
                mean_h = calc_circ_mean_hours(phases_hours[rhythmic_indices_relative], period)
                self.mw.log_message(f"Global Mean Phase: {mean_h:.2f} hours")
            
            
            if "Cosinor" in method:
                 final_phases_h = phases
            else:
                 final_phases_h = ((phases / (2 * np.pi)) * period) % period
            
            save_data = {
                'Original_ROI_Index': np.arange(len(phases)) + 1,
                'Phase_Hours': final_phases_h,
                'Period_Hours': np.full(len(phases), period),
                'Is_Rhythmic': rhythm_mask
            }
            
            # Add scores for completeness
            if "Cosinor" in method:
                save_data['P_Value'] = filter_scores
                save_data['R_Squared'] = sort_scores
            else:
                save_data['SNR'] = sort_scores
                
            self.latest_rhythm_df = pd.DataFrame(save_data)
            self.btn_save_rhythm.setEnabled(True) # Enable the save button!
            
            heatmap_viewer = self.mw.visualization_widgets.get(self.mw.heatmap_tab)
            if heatmap_viewer:
                 heatmap_viewer.update_phase_data(phases_hours, sort_scores, rhythm_mask, sort_desc,
                                             period=period, minutes_per_frame=mpf, 
                                             reference_phase=mean_h, trend_window_hours=trend_win)
                 heatmap_viewer.update_rhythm_emphasis(rhythm_mask, is_emphasized)

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
            else:
                df_data['SNR'] = sort_scores[rhythmic_indices_relative]
            
            rhythmic_df = pd.DataFrame(df_data)
            
            def phase_map_callback(selected_phase_index):
                original_index = rhythmic_df['Original_ROI_Index'].iloc[selected_phase_index] - 1
                self.on_roi_selected(original_index)
            
            fig_p, _ = add_mpl_to_tab(self.mw.phase_tab)
            viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], rhythmic_df, phase_map_callback, vmin=self.vmin, vmax=self.vmax)
            self.mw.visualization_widgets[self.mw.phase_tab] = viewer_p
            
            grid_res = int(self.phase_params["grid_resolution"][0].text())
            fig_i, _ = add_mpl_to_tab(self.mw.interp_tab)
            viewer_i = InterpolatedMapViewer(fig_i, fig_i.add_subplot(111), rhythmic_df[['X_Position', 'Y_Position']].values, rhythmic_df['Relative_Phase_Hours'].values, period, grid_res, rois=self.rois)
            self.mw.visualization_widgets[self.mw.interp_tab] = viewer_i
            
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.phase_tab), True)
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.interp_tab), True)
            self.btn_regen_phase.setEnabled(True)
            self.mw.log_message("Phase-based plots updated. Review and click 'Save Rhythm Results' to commit.")
            
        except Exception as e:
            self.mw.log_message(f"Could not update plots: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.phase_tab), False)
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.interp_tab), False)

    def save_rhythm_results(self):
        if self.latest_rhythm_df is None:
            return
        
        basename = self.state.output_basename
        if not basename:
            self.mw.log_message("Error: No output basename defined.")
            return
            
        filename = f"{basename}_rhythm_results.csv"
        
        try:
            self.latest_rhythm_df.to_csv(filename, index=False)
            self.mw.log_message(f"SUCCESS: Rhythm results saved to {os.path.basename(filename)}")
            self.mw.log_message("These approved cells will now be used for Group Analysis.")
        except Exception as e:
            self.mw.log_message(f"Error saving rhythm results: {e}")