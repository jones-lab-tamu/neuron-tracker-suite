import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
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

"""
SingleAnimalPanel invariants (do not change without intent and verification):

INDEXING / SELECTION
- "original_index" means the global candidate index in unfiltered space (0-based).
- "filtered_indices" maps local indices (in loaded_data) back to original_index.
- Viewers that display spatial points may show filtered data, but selection callbacks must
  ultimately resolve to original_index, then map to local_index via filtered_indices.

FILTER INTERSECTION
- The definitive filter is the intersection:
    final_mask = roi_mask & metric_mask
- roi_mask and metric_mask are always length == num_total_candidates (unfiltered space).
- The order of applying ROI vs Quality Gate must not change results, only the final intersection matters.

ROI SCHEMA
- ROI dictionaries may contain:
    {"mode": "Include"/"Exclude", "path_vertices": [...], optional "path": matplotlib.path.Path}
- Viewers must not assume "path" exists. Prefer path_vertices, build Path if needed.

PHASE REFERENCE
- phase_reference_rois are in the same coordinate space as state.unfiltered_data["roi"].
- Reference masking uses coordinates from loaded_data (post ROI/gate intersection), so any
  future coordinate transform must be applied consistently to both.
"""

class SingleAnimalPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        
        # Local state
        self.params = {}
        self.phase_params = {}
        self.filtered_indices = None
        self.rois = None
        self.phase_reference_rois = None
        self.vmin = None
        self.vmax = None
        self.metrics_df = None
        self.latest_rhythm_df = None
        
        # Filtering State
        self.roi_mask = None
        self.metric_mask = None
        self.num_total_candidates = 0
        self.quality_presets = {}  # Computed on load
        
        # Thread handles
        self._analysis_worker = None
        self._analysis_thread = None
        self._movie_loader_worker = None
        self._movie_loader_thread = None

        # Visibility Tracking
        self.advanced_widgets = [] # List of widgets to hide/show

        self.init_ui()
        
        # --- Legacy label aliases (MainWindow expects these attribute names) ---
        # Keep these as direct references to the new workflow strip labels so
        # MainWindow.update_workflow_from_files() continues to work unchanged.
        self.status_input_label = self.lbl_status_input
        self.status_traces_label = self.lbl_status_results
        self.status_mode_label = self.lbl_status_mode
        self.status_gate_label = self.lbl_status_gate
        self.status_roi_label = self.lbl_status_roi
                
        # Some older MainWindow code also expects these:
        # We do not have separate strip slots for each file type in the rewrite,
        # so map them to "Results"
        self.status_traj_label = self.lbl_status_results
        self.status_metrics_label = self.lbl_status_mode  # metrics implies scored/strict mode
        self.status_roi_file_label = self.lbl_status_roi  # ROI file presence, treated as ROI status
          
        self.connect_signals()
        
        # Initial State
        self._set_ui_mode(False) # Default to Basic
        self._update_workflow_status()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Top Control Bar ---
        top_row = QtWidgets.QHBoxLayout()
        self.chk_advanced_mode = QtWidgets.QCheckBox("Show Advanced Controls")
        top_row.addWidget(self.chk_advanced_mode)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        # --- 1. File I/O Box ---
        self.io_box = QtWidgets.QGroupBox("File I/O")
        io_layout = QtWidgets.QGridLayout(self.io_box)
        
        self.btn_load_movie = QtWidgets.QPushButton(get_icon('fa5s.file-video'), "Load Movie...")
        io_layout.addWidget(self.btn_load_movie, 0, 0, 1, 2)
        
        io_layout.addWidget(QtWidgets.QLabel("Input File:"), 1, 0)
        self.input_file_edit = QtWidgets.QLineEdit()
        self.input_file_edit.setReadOnly(True)
        io_layout.addWidget(self.input_file_edit, 1, 1)
        
        io_layout.addWidget(QtWidgets.QLabel("Output Basename:"), 2, 0)
        self.output_base_edit = QtWidgets.QLineEdit()
        io_layout.addWidget(self.output_base_edit, 2, 1)
        
        layout.addWidget(self.io_box)

        # --- 2. Workflow Status Strip ---
        self.workflow_box = QtWidgets.QGroupBox("Workflow Status")
        wf_layout = QtWidgets.QHBoxLayout(self.workflow_box)
        
        self.lbl_status_input = QtWidgets.QLabel("Input: none")
        self.lbl_status_results = QtWidgets.QLabel("Results: none")
        self.lbl_status_mode = QtWidgets.QLabel("Mode: unknown")
        self.lbl_status_gate = QtWidgets.QLabel("Gate: off")
        self.lbl_status_roi = QtWidgets.QLabel("ROI: off")
        
        for lbl in [self.lbl_status_input, self.lbl_status_results, self.lbl_status_mode, self.lbl_status_gate, self.lbl_status_roi]:
            lbl.setStyleSheet("font-size: 11px; color: #333; padding: 2px; border: 1px solid #ccc; border-radius: 3px; background: #f9f9f9;")
            wf_layout.addWidget(lbl)
            
        layout.addWidget(self.workflow_box)

        # --- 3. ROI Box ---
        self.roi_box = QtWidgets.QGroupBox("Region of Interest (ROI)")
        roi_layout = QtWidgets.QHBoxLayout(self.roi_box)
        self.btn_define_roi = QtWidgets.QPushButton(get_icon('fa5s.pen'), "Define Anatomical ROI...")
        self.btn_define_roi.setEnabled(False)
        Tooltip.install(self.btn_define_roi, "Draw include/exclude polygons to filter cells spatially.")
        
        self.btn_clear_roi = QtWidgets.QPushButton(get_icon('fa5s.times'), "Clear ROI Filter")
        self.btn_clear_roi.setEnabled(False)
        
        roi_layout.addWidget(self.btn_define_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addWidget(self.roi_box)
        
        # --- 4. Quality Gate Box (Replaces Post-Hoc Filtering) ---
        self.quality_gate_box = QtWidgets.QGroupBox("Quality Gate")
        gate_layout = QtWidgets.QGridLayout(self.quality_gate_box)
        
        # Row 0: Presets & Override
        gate_layout.addWidget(QtWidgets.QLabel("Preset:"), 0, 0)
        self.quality_preset_combo = QtWidgets.QComboBox()
        self.quality_preset_combo.addItems(["Recommended", "Lenient", "Strict", "Manual"])
        gate_layout.addWidget(self.quality_preset_combo, 0, 1)
        
        self.chk_quality_manual_override = QtWidgets.QCheckBox("Manual Override")
        gate_layout.addWidget(self.chk_quality_manual_override, 0, 2)
        
        # Row 1: Manual Controls (Hidden by default unless override)
        self.manual_gate_widget = QtWidgets.QWidget()
        manual_layout = QtWidgets.QHBoxLayout(self.manual_gate_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin_coverage = QtWidgets.QDoubleSpinBox()
        self.spin_coverage.setRange(0.0, 1.0); self.spin_coverage.setSingleStep(0.05)
        self.spin_coverage.setPrefix("Cov >= ")
        manual_layout.addWidget(self.spin_coverage)
        
        self.spin_jitter = QtWidgets.QDoubleSpinBox()
        self.spin_jitter.setRange(0.0, 50.0); self.spin_jitter.setSingleStep(0.5)
        self.spin_jitter.setPrefix("Jit <= ")
        manual_layout.addWidget(self.spin_jitter)
        
        self.spin_snr = QtWidgets.QDoubleSpinBox()
        self.spin_snr.setRange(0.0, 50.0); self.spin_snr.setSingleStep(0.5)
        self.spin_snr.setPrefix("SNR >= ")
        manual_layout.addWidget(self.spin_snr)
        
        gate_layout.addWidget(self.manual_gate_widget, 1, 0, 1, 3)
        self.manual_gate_widget.setVisible(False) # Initial state
        
        # Row 2: Apply & Status
        self.btn_apply_quality_gate = QtWidgets.QPushButton(get_icon('fa5s.filter'), "Apply Quality Gate")
        Tooltip.install(
            self.btn_apply_quality_gate,
            "INPUT: Current ROI candidate set loaded in this panel.\n"
            "CRITERIA: Coverage, Jitter, SNR thresholds.\n"
            "OUTPUT: Quality-Gate pass set (subset of candidates)."
        )
        gate_layout.addWidget(self.btn_apply_quality_gate, 2, 0, 1, 1)
        
        self.lbl_quality_counts = QtWidgets.QLabel("Passing: - / -")
        self.lbl_quality_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_quality_counts.setStyleSheet("font-weight: bold;")
        gate_layout.addWidget(self.lbl_quality_counts, 2, 1, 1, 2)
        
        self.lbl_quality_breakdown = QtWidgets.QLabel("")
        self.lbl_quality_breakdown.setStyleSheet("color: gray; font-size: 10px;")
        gate_layout.addWidget(self.lbl_quality_breakdown, 3, 0, 1, 3)
        
        layout.addWidget(self.quality_gate_box)

        # --- 5. Analysis Parameters (Advanced Only) ---
        self.param_box = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QVBoxLayout(self.param_box)
        
        # Save/Load
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_params = QtWidgets.QPushButton(get_icon('fa5s.download'), "Save Params...")
        self.btn_load_params = QtWidgets.QPushButton(get_icon('fa5s.upload'), "Load Params...")
        btn_row.addWidget(self.btn_save_params)
        btn_row.addWidget(self.btn_load_params)
        param_layout.addLayout(btn_row)
        
        # Tabs
        self.param_tabs = QtWidgets.QTabWidget()
        param_layout.addWidget(self.param_tabs)
        
        # Define Tooltips
        param_tooltips = {
            "sigma1": "Smaller Gaussian blur radius (pixels).",
            "sigma2": "Larger Gaussian blur radius (pixels) for background subtraction.",
            "blur_sigma": "Blur applied before ranking features.",
            "max_features": "Max brightest features per frame.",
            "search_range": "Max frames to look backward for linking.",
            "cone_radius_base": "Initial search radius (pixels).",
            "cone_radius_multiplier": "Radius growth factor per frame gap.",
            "min_trajectory_length": "Min track length (fraction of movie).",
            "sampling_box_size": "Intensity sampling box size (pixels).",
            "sampling_sigma": "Gaussian weights sigma for sampling.",
            "max_interpolation_distance": "Max allowed jump (pixels) between frames."
        }

        # Populate Tabs
        det_tab = QtWidgets.QWidget()
        det_layout = QtWidgets.QFormLayout(det_tab)
        self._add_param_field(det_layout, "sigma1", 3.0, param_tooltips)
        self._add_param_field(det_layout, "sigma2", 20.0, param_tooltips)
        self._add_param_field(det_layout, "blur_sigma", 2.0, param_tooltips)
        self._add_param_field(det_layout, "max_features", 200, param_tooltips)
        self.param_tabs.addTab(det_tab, "Detection")
        
        tr_tab = QtWidgets.QWidget()
        tr_layout = QtWidgets.QFormLayout(tr_tab)
        self._add_param_field(tr_layout, "search_range", 50, param_tooltips)
        self._add_param_field(tr_layout, "cone_radius_base", 1.5, param_tooltips)
        self._add_param_field(tr_layout, "cone_radius_multiplier", 0.125, param_tooltips)
        self.param_tabs.addTab(tr_tab, "Tracking")
        
        fl_tab = QtWidgets.QWidget()
        fl_layout = QtWidgets.QFormLayout(fl_tab)
        self._add_param_field(fl_layout, "min_trajectory_length", 0.08, param_tooltips)
        self._add_param_field(fl_layout, "sampling_box_size", 15, param_tooltips)
        self._add_param_field(fl_layout, "sampling_sigma", 2.0, param_tooltips)
        self._add_param_field(fl_layout, "max_interpolation_distance", 5.0, param_tooltips)
        
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Legacy pipeline (no metrics)", "strict")
        self.mode_combo.addItem("Metrics pipeline (recommended)", "scored")
        Tooltip.install(
            self.mode_combo,
            "This controls the analysis pipeline run by AnalysisWorker. "
            "Separately, the Quality Gate is available only if a valid metrics CSV is loaded."
        )
        
        # Default to scored pipeline so users get full candidate set, then gate post hoc.
        idx = self.mode_combo.findData("scored")
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        
        fl_layout.addRow("Analysis Pipeline:", self.mode_combo)
        
        self.param_tabs.addTab(fl_tab, "Filtering")
        
        layout.addWidget(self.param_box)
        self.advanced_widgets.append(self.param_box) # Whole box is advanced

        # --- 6. Phase Map Parameters ---
        self.phase_box = QtWidgets.QGroupBox("Phase Map Parameters")
        phase_layout = QtWidgets.QFormLayout(self.phase_box)
        
        # Basic Phase Controls
        self.analysis_method_combo = QtWidgets.QComboBox()
        self.analysis_method_combo.addItems(["FFT (SNR)", "Cosinor (p-value)"])
        phase_layout.addRow("Analysis Method:", self.analysis_method_combo)
        
        self._add_phase_field(phase_layout, "minutes_per_frame", 15.0)
        
        self.discovered_period_edit = QtWidgets.QLineEdit("N/A")
        self.discovered_period_edit.setReadOnly(True)
        phase_layout.addRow("Discovered Period (hrs):", self.discovered_period_edit)
        
        _, self.rhythm_threshold_label = self._add_phase_field(phase_layout, "rhythm_threshold", 2.0)
        
        # Cosinor specific (Conditional)
        self.rsquared_le, self.rsquared_label = self._add_phase_field(phase_layout, "r_squared_threshold", 0.3)
        self.rsquared_widgets = [self.rsquared_le, self.rsquared_label]
        
        self.strict_cycle_check = QtWidgets.QCheckBox("Require >= 2 cycles")
        phase_layout.addRow(self.strict_cycle_check)
        
        self.emphasize_rhythm_check = QtWidgets.QCheckBox("Emphasize rhythmic cells")
        phase_layout.addRow(self.emphasize_rhythm_check)
        
        # Advanced Phase Controls (Collected for toggling)
        self.adv_phase_rows = []
        
        self.adv_phase_rows.extend(self._add_phase_field(phase_layout, "period_min", 22.0, is_advanced=True))
        self.adv_phase_rows.extend(self._add_phase_field(phase_layout, "period_max", 28.0, is_advanced=True))
        self.adv_phase_rows.extend(self._add_phase_field(phase_layout, "trend_window_hours", 36.0, is_advanced=True))
        self.adv_phase_rows.extend(self._add_phase_field(phase_layout, "grid_resolution", 100, int, is_advanced=True))
        
        self.use_subregion_ref_check = QtWidgets.QCheckBox("Set phase zero using reference polygon")
        self.use_subregion_ref_check.setEnabled(False)
        phase_layout.addRow(self.use_subregion_ref_check)
        self.adv_phase_rows.append(self.use_subregion_ref_check)
        
        # Actions (Always visible)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_regen_phase = QtWidgets.QPushButton(get_icon('fa5s.sync'), "Update Plots")
        Tooltip.install(
            self.btn_regen_phase,
            "INPUT: Quality-Gate pass set.\n"
            "CRITERIA: Rhythmicity threshold (e.g. SNR/p-value), Strict Cycle check.\n"
            "OUTPUT: Rhythmic subset (visualized in maps).\n"
            "NOTE: This does not save files. Use 'Save Rhythm Results' to write the currently included set to disk."
        )
        self.btn_save_rhythm = QtWidgets.QPushButton(get_icon('fa5s.save'), "Save Rhythm Results")
        Tooltip.install(
            self.btn_save_rhythm,
            "Save Rhythm Results writes ONLY the currently included cells after Quality Gate and Phase Map filtering.\n"
            "These saved rows are what Group View uses."
        )
        self.btn_save_rhythm.setEnabled(False)
        btn_row.addWidget(self.btn_regen_phase)
        btn_row.addWidget(self.btn_save_rhythm)
        phase_layout.addRow(btn_row)
        
        layout.addWidget(self.phase_box)
        layout.addStretch(1)
        
        # Init dynamic UI state
        self._on_analysis_method_changed(0)

    def _add_param_field(self, layout, name, default, tooltips=None):
        label = QtWidgets.QLabel(f"{name}:")
        if tooltips and name in tooltips:
            Tooltip.install(label, tooltips[name])
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(label, le)
        self.params[name] = (le, type(default))
        return le, label

    def _add_phase_field(self, layout, name, default, typ=float, is_advanced=False):
        label = QtWidgets.QLabel(f"{name}:")
        le = QtWidgets.QLineEdit(str(default))
        layout.addRow(label, le)
        self.phase_params[name] = (le, typ)
        if is_advanced:
            self.advanced_widgets.append(label)
            self.advanced_widgets.append(le)
            return [label, le]
        return [le, label]

    def connect_signals(self):
        # Top Level
        self.chk_advanced_mode.toggled.connect(self._set_ui_mode)
        
        # IO
        self.btn_load_movie.clicked.connect(self.load_movie)
        self.output_base_edit.textChanged.connect(self.update_output_basename)
        
        # ROI
        self.btn_define_roi.clicked.connect(self.open_roi_tool)
        self.btn_clear_roi.clicked.connect(self.clear_roi_filter)
        
        # Quality Gate
        self.quality_preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.chk_quality_manual_override.toggled.connect(self._on_manual_override_toggled)
        self.btn_apply_quality_gate.clicked.connect(self.apply_quality_gate)
        
        # Params
        self.btn_save_params.clicked.connect(self.save_parameters)
        self.btn_load_params.clicked.connect(self.load_parameters)
        
        # Main Window Hooks
        self.mw.btn_run_analysis.clicked.connect(self.start_analysis)
        self.mw.btn_load_results.clicked.connect(self.load_results)
        self.mw.btn_export_plot.clicked.connect(self.export_current_plot)
        self.mw.btn_export_data.clicked.connect(self.export_current_data)
        
        # Phase
        self.btn_regen_phase.clicked.connect(self.regenerate_phase_maps)
        self.btn_save_rhythm.clicked.connect(self.save_rhythm_results)
        self.emphasize_rhythm_check.stateChanged.connect(self.regenerate_phase_maps)
        self.use_subregion_ref_check.stateChanged.connect(self.regenerate_phase_maps)
        self.analysis_method_combo.currentIndexChanged.connect(self._on_analysis_method_changed)

    # --- UI Controller Logic ---
    
    def _set_ui_mode(self, is_advanced):
        for widget in self.advanced_widgets:
            widget.setVisible(is_advanced)
        # Handle specialized advanced rows list
        for widget in self.adv_phase_rows:
            widget.setVisible(is_advanced)
            
    def _update_workflow_status(self):
        # 1. Input
        if self.state.input_movie_path:
            self.lbl_status_input.setText(f"Input: {os.path.basename(self.state.input_movie_path)}")
            self.lbl_status_input.setStyleSheet("background: #d4edda; border: 1px solid #c3e6cb;")
        else:
            self.lbl_status_input.setText("Input: none")
            self.lbl_status_input.setStyleSheet("background: #f9f9f9; border: 1px solid #ccc;")

        # 2. Results
        has_traces = self.state.unfiltered_data.get("traces") is not None
        if has_traces:
            self.lbl_status_results.setText("Results: loaded")
            self.lbl_status_results.setStyleSheet("background: #d4edda; border: 1px solid #c3e6cb;")
        else:
            self.lbl_status_results.setText("Results: none")
            self.lbl_status_results.setStyleSheet("background: #f9f9f9; border: 1px solid #ccc;")
            
        # 3. Mode (Determined by metrics availability)
        has_metrics = self.metrics_df is not None
        if has_traces:
            if has_metrics:
                self.lbl_status_mode.setText("Metrics: available")
                self.lbl_status_mode.setStyleSheet("background: #cce5ff; border: 1px solid #b8daff;")
            else:
                self.lbl_status_mode.setText("Metrics: missing")
                self.lbl_status_mode.setStyleSheet("background: #fff3cd; border: 1px solid #ffeeba;")
        else:
            self.lbl_status_mode.setText("Mode: —")
            self.lbl_status_mode.setStyleSheet("background: #f9f9f9;")

        # 4. ROI
        N = self.num_total_candidates
        roi_n = self.roi_mask.sum() if self.roi_mask is not None else 0
        if self.roi_mask is not None and roi_n < N:
             self.lbl_status_roi.setText(f"ROI: Active ({roi_n})")
             self.lbl_status_roi.setStyleSheet("background: #d4edda; border: 1px solid #c3e6cb;")
        else:
             self.lbl_status_roi.setText("ROI: Off")
             self.lbl_status_roi.setStyleSheet("background: #f9f9f9; border: 1px solid #ccc;")

        # 5. Gate
        gate_n = self.metric_mask.sum() if self.metric_mask is not None else 0
        if self.metric_mask is not None and gate_n < N:
            method = "Manual" if self.chk_quality_manual_override.isChecked() else self.quality_preset_combo.currentText()
            self.lbl_status_gate.setText(f"Gate: {method} ({gate_n})")
            self.lbl_status_gate.setStyleSheet("background: #d4edda; border: 1px solid #c3e6cb;")
        else:
            self.lbl_status_gate.setText("Gate: Off")
            self.lbl_status_gate.setStyleSheet("background: #f9f9f9; border: 1px solid #ccc;")
            
        # 6. Overall Counts
        final_n = (self.roi_mask & self.metric_mask).sum() if (self.roi_mask is not None and self.metric_mask is not None) else 0
        if N > 0:
            pct = (final_n / N) * 100.0
            self.lbl_quality_counts.setText(f"Passing: {final_n} / {N} ({pct:.1f}%)")
        else:
            self.lbl_quality_counts.setText("Passing: - / -")

    def _on_preset_changed(self):
        # Do not apply immediately. Just update UI spinboxes if we are in a mode that allows it.
        # This gives visual feedback of what "Strict" actually means in numbers.
        preset_name = self.quality_preset_combo.currentText()
        if preset_name == "Manual":
            self.chk_quality_manual_override.setChecked(True)
            return

        vals = self.quality_presets.get(preset_name)
        if vals:
            self.spin_coverage.setValue(vals.get('cov', 0.0))
            self.spin_jitter.setValue(vals.get('jit', 50.0))
            self.spin_snr.setValue(vals.get('snr', 0.0))

    def _on_manual_override_toggled(self, checked):
        self.manual_gate_widget.setVisible(checked)
        if checked:
            self.quality_preset_combo.setCurrentText("Manual")

    def _compute_quality_presets(self):
        """Derive preset thresholds dynamically from loaded metrics percentiles."""
        if self.metrics_df is None: 
            self.quality_presets = {}
            return

        df = self.metrics_df
        presets = {}
        
        # Helper to get percentile or default
        def get_p(col, p, default, reverse=False):
            if col not in df.columns: return default
            vals = pd.to_numeric(df[col], errors='coerce').values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0: return default
            return np.percentile(vals, p)

        # 1. Lenient (Bottom 50% cov, Top 95% jitter, Top 90% SNR)
        presets["Lenient"] = {
            'cov': get_p('detected_fraction', 50, 0.0),
            'jit': get_p('spatial_jitter_detrended', 95, 50.0),
            'snr': get_p('trace_snr_proxy', 10, 0.0)
        }
        
        # 2. Recommended (Top 75% cov, Bottom 75% jitter, Top 25% SNR)
        presets["Recommended"] = {
            'cov': get_p('detected_fraction', 75, 0.5),
            'jit': get_p('spatial_jitter_detrended', 75, 5.0),
            'snr': get_p('trace_snr_proxy', 25, 2.0)
        }
        
        # 3. Strict (Top 90% cov, Bottom 50% jitter, Top 50% SNR)
        presets["Strict"] = {
            'cov': get_p('detected_fraction', 90, 0.8),
            'jit': get_p('spatial_jitter_detrended', 50, 2.0),
            'snr': get_p('trace_snr_proxy', 50, 5.0)
        }
        
        self.quality_presets = presets
        # Update spinboxes to Recommended by default without applying
        if "Recommended" in presets:
            self.spin_coverage.setValue(presets["Recommended"]['cov'])
            self.spin_jitter.setValue(presets["Recommended"]['jit'])
            self.spin_snr.setValue(presets["Recommended"]['snr'])

    # --- Data Loading & Analysis ---

    def reset_state(self):
        """
        Called by MainWindow._reset_state().

        This must exist even if we do most state clearing in MainWindow,
        because older workflow assumes each panel can reset itself safely.
        """
        # Local state (panel)
        self.filtered_indices = None
        self.rois = None
        self.phase_reference_rois = None
        self.vmin = None
        self.vmax = None
        self.metrics_df = None
        self.latest_rhythm_df = None

        # Masks and counts
        self.roi_mask = None
        self.metric_mask = None
        self.num_total_candidates = 0
        self.quality_presets = {}

        # UI: disable things that require loaded data
        if hasattr(self, "input_file_edit"):
            self.input_file_edit.setText("")
        if hasattr(self, "output_base_edit"):
            self.output_base_edit.setText("")

        if hasattr(self, "btn_define_roi"):
            self.btn_define_roi.setEnabled(False)
        if hasattr(self, "btn_clear_roi"):
            self.btn_clear_roi.setEnabled(False)

        # Quality gate: default to disabled until metrics load
        if hasattr(self, "quality_gate_box"):
            self.quality_gate_box.setEnabled(False)
            self.quality_gate_box.setTitle("Quality Gate (Unavailable - No Metrics)")
        if hasattr(self, "lbl_quality_counts"):
            self.lbl_quality_counts.setText("Passing: - / -")
        if hasattr(self, "lbl_quality_breakdown"):
            self.lbl_quality_breakdown.setText("")

        # Phase controls that depend on ROI refs
        if hasattr(self, "use_subregion_ref_check"):
            self.use_subregion_ref_check.setChecked(False)
            self.use_subregion_ref_check.setEnabled(False)

        if hasattr(self, "btn_save_rhythm"):
            self.btn_save_rhythm.setEnabled(False)

        # Clear visualization tabs safely if they exist
        try:
            for tab in (self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab):
                if tab.layout() is not None:
                    clear_layout(tab.layout())
        except Exception:
            pass

        self._update_workflow_status()

    def load_movie(self):
        self.mw._reset_state()
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Movie", start_dir, "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if not path: return
        
        self.mw._set_last_dir(path)
        self.state.input_movie_path = path
        self.input_file_edit.setText(path)
        base, _ = os.path.splitext(path)
        self.state.output_basename = base
        self.output_base_edit.setText(base)
        self.mw.workflow_state["has_input"] = True
        self.mw.log_message(f"Loaded movie: {os.path.basename(path)}")
        self.mw.update_workflow_from_files()
        self._update_workflow_status()

    def update_output_basename(self, text):
        self.state.output_basename = text
        self.mw.update_workflow_from_files()

    def start_analysis(self):
        if not self.state.input_movie_path or not self.state.output_basename:
            self.mw.log_message("Error: input file and output basename required.")
            return
        try:
            args = {name: t(le.text()) for name, (le, t) in self.params.items()}
            mode = self.mode_combo.currentData()
            args['mode'] = mode if mode else 'strict'
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
            import gc; gc.collect()

            if success:
                self.mw.log_message("Analysis finished. Loading results...")
                self.load_results()
            else:
                self.mw.log_message("Analysis failed.")
            self.mw.update_workflow_from_files()
            self._update_workflow_status()
            
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
            # Load basic data
            traces = np.loadtxt(f"{basename}_traces.csv", delimiter=",")
            roi = np.loadtxt(f"{basename}_roi.csv", delimiter=",")
            traj = np.load(f"{basename}_trajectories.npy")

            if isinstance(roi, np.ndarray) and roi.ndim == 1: roi = roi.reshape(1, -1)
            if isinstance(traces, np.ndarray) and traces.ndim == 1: traces = traces.reshape(1, -1)

            self.state.unfiltered_data["traces"] = traces
            self.state.unfiltered_data["roi"] = roi
            self.state.unfiltered_data["trajectories"] = traj
            
            # Reset masks
            self.num_total_candidates = len(roi)
            self.roi_mask = np.ones(self.num_total_candidates, dtype=bool)
            self.metric_mask = np.ones(self.num_total_candidates, dtype=bool)

            self.mw.log_message(f"[{basename}] ROI load: candidates={self.num_total_candidates} (file={basename}_roi.csv)")
            
            # Load Metrics
            metrics_path = f"{basename}_metrics.csv"
            has_metrics = False
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                if "candidate_id" in df.columns:
                    df = df.sort_values("candidate_id").reset_index(drop=True)
                    if len(df) == self.num_total_candidates:
                        self.metrics_df = df
                        has_metrics = True
                        self._compute_quality_presets()
                        self.mw.log_message(f"Metrics loaded: {len(df)} rows. Scored mode active.")
            
            if not has_metrics:
                self.metrics_df = None
                
                # Metrics missing, force strict mode to match what the pipeline can support.
                idx = self.mode_combo.findData("strict")
                if idx >= 0:
                    self.mode_combo.setCurrentIndex(idx)
                
                self.quality_gate_box.setEnabled(False)
                self.quality_gate_box.setTitle("Quality Gate (Unavailable - No Metrics)")
                self.mw.log_message("No valid metrics found. Strict mode active (manual ROI only).")
            else:
                self.quality_gate_box.setEnabled(True)
                self.quality_gate_box.setTitle("Quality Gate")
                self.quality_preset_combo.setCurrentText("Recommended")
                # Metrics present, scored mode is valid and should be the default.
                idx = self.mode_combo.findData("scored")
                if idx >= 0:
                    self.mode_combo.setCurrentIndex(idx)

            self.mw.log_message("Loaded ROI and trace data.")
            self._update_workflow_status()
            
        except Exception as e:
            self.mw.log_message(f"Error loading result files: {e}")
            return

        self.btn_load_movie.setEnabled(False)
        self.mw.btn_load_results.setEnabled(False)
        self.mw.log_message("Loading movie in background...")

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
        self.mw.log_message("Movie loaded.")
        self.state.unfiltered_data["movie"] = movie_data
        self.state.unfiltered_data["background"] = movie_data[len(movie_data) // 2]
        
        self.btn_load_movie.setEnabled(True)
        self.mw.btn_load_results.setEnabled(True)
        
        # Enable ROI tool if results are present
        self.btn_define_roi.setEnabled(True)
        
        self._resolve_filters()
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    @QtCore.pyqtSlot(str)
    def _on_movie_load_error(self, error_message):
        self.mw.log_message(f"Movie load failed: {error_message}")
        self.btn_load_movie.setEnabled(True)
        self.mw.btn_load_results.setEnabled(True)
        if self._movie_loader_thread:
            self._movie_loader_thread.quit()
            self._movie_loader_thread.wait()
        self._movie_loader_thread = None
        self._movie_loader_worker = None

    # --- Filtering Logic ---

    def apply_post_hoc_filters(self):
        """Compatibility wrapper for apply_quality_gate."""
        self.apply_quality_gate()

    def apply_quality_gate(self):
        if self.metrics_df is None: return
        
        use_manual = self.chk_quality_manual_override.isChecked()
        preset_name = self.quality_preset_combo.currentText()
        
        # Determine thresholds
        if use_manual or preset_name == "Manual":
            cov_min = self.spin_coverage.value()
            jit_max = self.spin_jitter.value()
            snr_min = self.spin_snr.value()
        else:
            vals = self.quality_presets.get(preset_name, {})
            cov_min = vals.get('cov', 0.0)
            jit_max = vals.get('jit', 50.0)
            snr_min = vals.get('snr', 0.0)
            
        # Get data columns (safe fallback)
        df = self.metrics_df
        def get_col(name):
             if name in df.columns: return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)
             return np.full(len(df), np.nan) # Fail closed if column missing? No, logic below handles NaN.

        cov = get_col('detected_fraction')
        jit = get_col('spatial_jitter_detrended')
        snr = get_col('trace_snr_proxy')

        # Compute Masks (handling NaNs as Fail)
        cov_ok = np.isfinite(cov) & (cov >= cov_min)
        jit_ok = np.isfinite(jit) & (jit <= jit_max)
        snr_ok = np.isfinite(snr) & (snr >= snr_min)
        
        self.metric_mask = cov_ok & jit_ok & snr_ok
        
        # Update breakdown label
        n_fail_cov = (~cov_ok).sum()
        n_fail_jit = (~jit_ok).sum()
        n_fail_snr = (~snr_ok).sum()
        self.lbl_quality_breakdown.setText(f"Failures: Cov={n_fail_cov}, Jit={n_fail_jit}, SNR={n_fail_snr}")
        
        self.lbl_quality_breakdown.setText(f"Failures: Cov={n_fail_cov}, Jit={n_fail_jit}, SNR={n_fail_snr}")
        
        n_in = len(self.metric_mask)
        n_pass = int(self.metric_mask.sum())
        base = self.state.output_basename if self.state.output_basename else "Session"
        
        try:
             thresh_msg = f"Thresholds: Cov>={cov_min}, Jit<={jit_max}, SNR>={snr_min}"
        except NameError:
             thresh_msg = "Thresholds: unavailable"

        self.mw.log_message(f"[{os.path.basename(base)}] QualityGate: in={n_in}, pass={n_pass}, drop={n_in - n_pass} | {thresh_msg}")

        self._resolve_filters()

    def open_roi_tool(self):
        if "background" not in self.state.unfiltered_data:
            self.mw.log_message("Error: Load data before defining an ROI.")
            return
        if not self.state.output_basename:
            self.mw.log_message("Error: Output Basename required.")
            return


        # Robust movie data lookup
        movie_data = None
        found_key = None
        
        # Check unfiltered_data
        for key in ["movie", "movie_frames"]:
            val = self.state.unfiltered_data.get(key)
            if val is not None:
                movie_data = val
                found_key = f"unfiltered_data['{key}']"
                break
        
        # Check loaded_data if still not found
        if movie_data is None and hasattr(self.state, "loaded_data") and self.state.loaded_data:
            for key in ["movie", "movie_frames"]:
                val = self.state.loaded_data.get(key)
                if val is not None:
                    movie_data = val
                    found_key = f"loaded_data['{key}']"
                    break

        if movie_data is not None:
            dtype_str = str(type(movie_data))
            shape_str = getattr(movie_data, 'shape', 'N/A')
            self.mw.log_message(f"ROI Drawer: Found movie data in {found_key}. Type={dtype_str}, Shape={shape_str}")
        else:
            self.mw.log_message("ROI Drawer: movie frames unavailable in current pipeline, using static background only.")
        
        dlg = ROIDrawerDialog(
            self,
            self.state.unfiltered_data["background"],
            self.state.unfiltered_data["roi"],
            self.state.output_basename,
            self.apply_roi_filter,
            vmin=self.vmin,
            vmax=self.vmax,
            is_region_mode=False,
            movie_data=movie_data
        )
        dlg.exec_()
        self.mw.update_workflow_from_files()

    def apply_roi_filter(self, indices, rois, phase_ref_rois, extra_mask=None):
        self.rois = rois
        self.phase_reference_rois = phase_ref_rois
        self.use_subregion_ref_check.setEnabled(bool(phase_ref_rois))
        if not phase_ref_rois: 
            self.use_subregion_ref_check.setChecked(False)

        if not rois:
            self.roi_mask = np.ones(self.num_total_candidates, dtype=bool)
        else:
            full_roi_data = self.state.unfiltered_data["roi"]
            mask = np.zeros(self.num_total_candidates, dtype=bool)
            
            # Robust Include
            has_includes = False
            for r in rois:
                if r.get("mode", "").lower() == "include":
                    if "path_vertices" in r:
                        path = Path(r["path_vertices"])
                        mask |= path.contains_points(full_roi_data)
                        has_includes = True
            
            # If no includes defined but excludes exist, start with ALL TRUE
            if not has_includes:
                mask[:] = True
                
            # Exclude
            for r in rois:
                if r.get("mode", "").lower() == "exclude":
                    if "path_vertices" in r:
                        path = Path(r["path_vertices"])
                        mask &= ~path.contains_points(full_roi_data)
                        
            self.roi_mask = mask
            
        self._resolve_filters()

    def clear_roi_filter(self):
        self.rois = None
        self.phase_reference_rois = None
        self.use_subregion_ref_check.setChecked(False)
        self.use_subregion_ref_check.setEnabled(False)
        self.roi_mask = np.ones(self.num_total_candidates, dtype=bool)
        self._resolve_filters()

    def _resolve_filters(self):
        if self.roi_mask is None: self.roi_mask = np.ones(self.num_total_candidates, dtype=bool)
        if self.metric_mask is None: self.metric_mask = np.ones(self.num_total_candidates, dtype=bool)
        
        final_mask = self.roi_mask & self.metric_mask
        final_indices = np.where(final_mask)[0]
        
        self.filtered_indices = final_indices
        self._update_workflow_status()
        
        # Check State Consistency
        is_filtered = len(final_indices) < self.num_total_candidates
        
        if not is_filtered:
             self.state.loaded_data = dict(self.state.unfiltered_data)
             self.btn_clear_roi.setEnabled(False)
        else:
             self.state.loaded_data = {} # Clear/Init
             self.state.loaded_data["roi"] = self.state.unfiltered_data["roi"][final_indices]
             self.state.loaded_data["trajectories"] = self.state.unfiltered_data["trajectories"][final_indices]
             trace_indices = np.concatenate(([0], final_indices + 1))
             self.state.loaded_data["traces"] = self.state.unfiltered_data["traces"][:, trace_indices]
             self.btn_clear_roi.setEnabled(True)

        self.populate_visualizations()

    # --- Parameter IO ---

    def save_parameters(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", start_dir, "JSON files (*.json)"
        )
        if not path: return
        self.mw._set_last_dir(path)
        data = {name: le.text() for name, (le, _) in self.params.items()}
        mode = self.mode_combo.currentData()
        if mode: data['mode'] = mode
        with open(path, "w") as f: json.dump(data, f, indent=4)
        self.mw.log_message(f"Parameters saved to {os.path.basename(path)}")

    def load_parameters(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Parameters", start_dir, "JSON files (*.json)"
        )
        if not path: return
        self.mw._set_last_dir(path)
        with open(path, "r") as f: data = json.load(f)
        for name, value in data.items():
            if name in self.params: self.params[name][0].setText(str(value))
        if 'mode' in data:
            index = self.mode_combo.findData(data['mode'])
            if index >= 0: self.mode_combo.setCurrentIndex(index)
        self.mw.log_message(f"Parameters loaded from {os.path.basename(path)}")

    # --- Visualization & Interaction ---

    def on_roi_selected(self, original_index):
        """Handle selection from a viewer (e.g. click on map)."""
        self.mw.log_message(f"ROI {original_index + 1} selected.")
        
        local_index = -1
        if self.filtered_indices is not None:
            matches = np.where(self.filtered_indices == original_index)[0]
            if len(matches) > 0: 
                local_index = matches[0]
        else:
            local_index = original_index

        if local_index != -1:
            traj_viewer = self.mw.visualization_widgets.get(self.mw.traj_tab)
            if traj_viewer:
                traj_viewer.set_trajectory(local_index)
                self.mw.vis_tabs.setCurrentWidget(self.mw.traj_tab)
            
            heatmap_viewer = self.mw.visualization_widgets.get(self.mw.heatmap_tab)
            if heatmap_viewer: 
                heatmap_viewer.update_selected_trace(original_index)

            com_viewer = self.mw.visualization_widgets.get(self.mw.com_tab)
            if com_viewer: 
                com_viewer.highlight_point(original_index)
                
            phase_viewer = self.mw.visualization_widgets.get(self.mw.phase_tab)
            if phase_viewer: 
                phase_viewer.highlight_point(original_index)

    def on_contrast_change(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        traj_viewer = self.mw.visualization_widgets.get(self.mw.traj_tab)
        if traj_viewer: traj_viewer.update_contrast(vmin, vmax)
        phase_viewer = self.mw.visualization_widgets.get(self.mw.phase_tab)
        if phase_viewer: phase_viewer.update_contrast(vmin, vmax)

    def _normalize_rois_for_viewers(self, rois):
        """
        Normalize ROI dicts for viewers.

        Important:
        - Some ROI dicts may already contain a matplotlib Path under key "path".
          We preserve it if present.
        - Some viewers only need path_vertices (and will rebuild Path internally).
        - Never assume "path" exists, never require it.
        """
        if not rois:
            return []

        clean = []
        for r in rois:
            verts = r.get("path_vertices", None)
            if verts is None:
                continue

            item = {
                "path_vertices": verts,
                "mode": r.get("mode", "Include"),
            }

            # Preserve prebuilt Path if present (optional, do not require).
            p = r.get("path", None)
            if isinstance(p, Path):
                item["path"] = p

            clean.append(item)

        return clean

    def populate_visualizations(self):
        if not self.state.loaded_data or "background" not in self.state.unfiltered_data: return
        
        bg = self.state.unfiltered_data["background"]
        movie = self.state.unfiltered_data.get("movie")
        if movie is None: return
        
        self.vmin, self.vmax = float(bg.min()), float(bg.max())
        
        # Enable Tabs
        for tab in [self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab]:
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), True)

        # 1. Heatmap
        try:
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = self._calculate_rhythms()
        except:
            phases, period, sort_scores, filter_scores, rhythm_sort_desc = None, None, None, None, True
            
        fig_h, _ = add_mpl_to_tab(self.mw.heatmap_tab)
        viewer_h = HeatmapViewer(fig_h, self.state.loaded_data, self.filtered_indices, phases, sort_scores, 
                                 self.emphasize_rhythm_check.isChecked(), rhythm_sort_desc, 
                                 period=period)
        self.mw.visualization_widgets[self.mw.heatmap_tab] = viewer_h

        # 2. CoM
        fig_c, _ = add_mpl_to_tab(self.mw.com_tab)
        
        current_rois = self._normalize_rois_for_viewers((self.rois or []) + (self.phase_reference_rois or []))
        
        viewer_c = ContrastViewer(fig_c, fig_c.add_subplot(111), bg, self.state.loaded_data["roi"], 
                                  self.on_contrast_change, self.on_roi_selected, 
                                  filtered_indices=self.filtered_indices, rois=current_rois)
        self.mw.visualization_widgets[self.mw.com_tab] = viewer_c

        # 3. Trajectory
        fig_t, _ = add_mpl_to_tab(self.mw.traj_tab)
        viewer_t = TrajectoryInspector(fig_t, fig_t.add_subplot(111), self.state.loaded_data["trajectories"], movie)
        self.mw.visualization_widgets[self.mw.traj_tab] = viewer_t

        self.regenerate_phase_maps()

    def _calculate_rhythms(self):
        method = self.analysis_method_combo.currentText()
        phase_args = {name: t(w.text()) for name, (w, t) in self.phase_params.items() 
                      if w.text() and name not in ("grid_resolution", "rhythm_threshold", "r_squared_threshold")}
        
        if not phase_args.get("minutes_per_frame"): raise ValueError("Minutes per frame is required.")
        if "trend_window_hours" in phase_args: phase_args["detrend_window_hours"] = phase_args.pop("trend_window_hours")
        
        # NOTE (tech debt): FFT is currently called twice (once for discovered_period, once for phases/scores).
        # This is acceptable for now but should be refactored later to avoid redundant work.
        
        # 1. Discover Period
        _, discovered_period, _ = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
        self.discovered_period_edit.setText(f"{discovered_period:.2f}")
        
        # 2. Compute Metrics
        if "FFT" in method:
            phases, period, snr_scores = calculate_phases_fft(self.state.loaded_data["traces"], **phase_args)
            return phases, period, snr_scores, snr_scores, True # FFT sort descending (SNR)
        elif "Cosinor" in method:
            traces = self.state.loaded_data["traces"]
            time_points_hours = traces[:, 0] * (phase_args["minutes_per_frame"] / 60.0)
            
            phases, p_values, r_squareds = [], [], []
            
            # Use same detrend logic as FFT
            trend_window_hours = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
            med_win = compute_median_window_frames(phase_args["minutes_per_frame"], trend_window_hours, T=traces.shape[0])
            
            for i in range(1, traces.shape[1]):
                res = csn.cosinor_analysis(
                    preprocess_for_rhythmicity(traces[:, i], method="running_median", median_window_frames=med_win),
                    time_points_hours, 
                    period=discovered_period
                )
                phases.append(res["acrophase"])
                p_values.append(res["p_value"])
                r_squareds.append(res["r_squared"])
                
            return np.array(phases), discovered_period, np.array(r_squareds), np.array(p_values), True

        return None, None, None, None, True

    @QtCore.pyqtSlot(int)
    def _on_analysis_method_changed(self, index):
        method = self.analysis_method_combo.currentText()
        thresh_edit = self.phase_params["rhythm_threshold"][0]
        
        if "FFT" in method:
            self.rhythm_threshold_label.setText("SNR Threshold (>=):")
            thresh_edit.setText("2.0")
            for w in self.rsquared_widgets: w.setVisible(False)
        elif "Cosinor" in method:
            self.rhythm_threshold_label.setText("p-value (<=):")
            thresh_edit.setText("0.05")
            # Only show R2 if in Advanced Mode, or force visibility?
            # Basic Mode rules say R2 is visible for Cosinor.
            # So we respect the current mode logic or just show it if Basic?
            # User requirement: "Show r_squared_threshold widgets" if Cosinor selected.
            # But check Advanced/Basic toggle.
            is_adv = self.chk_advanced_mode.isChecked()
            # If basic, we still show R2 if Cosinor is picked.
            for w in self.rsquared_widgets: w.setVisible(True)

    def regenerate_phase_maps(self):
        if not self.state.loaded_data or "traces" not in self.state.loaded_data: return
        
        self.btn_save_rhythm.setEnabled(False)
        for tab in (self.mw.phase_tab, self.mw.interp_tab): clear_layout(tab.layout())
        
        try:
            phases, period, sort_scores, filter_scores, sort_desc = self._calculate_rhythms()
            if phases is None: return
            
            method = self.analysis_method_combo.currentText()
            thresh = float(self.phase_params["rhythm_threshold"][0].text())
            mpf = float(self.phase_params["minutes_per_frame"][0].text())
            try: trend_win = float(self.phase_params["trend_window_hours"][0].text())
            except: trend_win = 36.0
            
            # --- Rhythm Gating ---
            if "Cosinor" in method:
                r_thresh = float(self.phase_params["r_squared_threshold"][0].text())
                rhythm_mask = (filter_scores <= thresh) & (sort_scores >= r_thresh)
                phases_hours = phases
            else:
                rhythm_mask = filter_scores >= thresh
                phases_hours = ((phases / (2 * np.pi)) * period) % period
                
            if self.strict_cycle_check.isChecked():
                rhythm_mask = strict_cycle_mask(self.state.loaded_data["traces"], mpf, period, rhythm_mask, min_cycles=2, trend_window_hours=trend_win)
            
            if self.strict_cycle_check.isChecked():
                rhythm_mask = strict_cycle_mask(self.state.loaded_data["traces"], mpf, period, rhythm_mask, min_cycles=2, trend_window_hours=trend_win)
            
            if self.strict_cycle_check.isChecked():
                rhythm_mask = strict_cycle_mask(self.state.loaded_data["traces"], mpf, period, rhythm_mask, min_cycles=2, trend_window_hours=trend_win)
            
            n_in_phase = len(rhythm_mask)
            n_rhythm = int(rhythm_mask.sum())
            strict_status = "ON" if self.strict_cycle_check.isChecked() else "OFF"
            base = self.state.output_basename if self.state.output_basename else "Session"
            
            try:
                meth_msg = f"Method={method}, Thresh={thresh}"
            except NameError:
                meth_msg = "Method=unknown, Thresh=unknown"

            self.mw.log_message(f"[{os.path.basename(base)}] PhaseFilter: in={n_in_phase}, pass={n_rhythm}, drop={n_in_phase - n_rhythm} | {meth_msg}, strict_cycle={strict_status}")
            
            # --- Emphasis & Viewers ---
            com_viewer = self.mw.visualization_widgets.get(self.mw.com_tab)
            if com_viewer: com_viewer.update_rhythm_emphasis(rhythm_mask, self.emphasize_rhythm_check.isChecked())
            
            heatmap_viewer = self.mw.visualization_widgets.get(self.mw.heatmap_tab)
            if heatmap_viewer: 
                heatmap_viewer.update_phase_data(phases_hours, sort_scores, rhythm_mask, sort_desc, period=period, minutes_per_frame=mpf)

            rhythmic_indices = np.where(rhythm_mask)[0]
            if len(rhythmic_indices) == 0:
                self.mw.log_message("No rhythmic cells found.")
                return

            # --- Reference Phase ---
            mean_h = 0.0
            
            if self.use_subregion_ref_check.isChecked() and self.phase_reference_rois:
                # Calc subset mean
                r_coords = self.state.loaded_data['roi'][rhythmic_indices]
                sub_mask = np.zeros(len(r_coords), dtype=bool)
                for r in self.phase_reference_rois:
                    p = Path(r["path_vertices"])
                    sub_mask |= p.contains_points(r_coords)
                
                if sub_mask.sum() > 0:
                    sub_phases = phases_hours[rhythmic_indices][sub_mask]
                    rads = (sub_phases % period) * (2 * np.pi / period)
                    mean_h = (circmean(rads) / (2 * np.pi)) * period
                    self.mw.log_message(f"Ref Phase (Subregion): {mean_h:.2f}h")
                else:
                    self.mw.log_message("Ref Region empty of rhythmic cells. Using global mean.")
                    rads = (phases_hours[rhythmic_indices] % period) * (2 * np.pi / period)
                    mean_h = (circmean(rads) / (2 * np.pi)) * period
            else:
                rads = (phases_hours[rhythmic_indices] % period) * (2 * np.pi / period)
                mean_h = (circmean(rads) / (2 * np.pi)) * period

            # --- Dataframes ---
            rel_phases = (phases_hours[rhythmic_indices] - mean_h + period / 2) % period - period / 2
            
            orig_indices = self.filtered_indices[rhythmic_indices] if self.filtered_indices is not None else rhythmic_indices
            
            df_map = pd.DataFrame({
                'Original_ROI_Index': orig_indices + 1,
                'X_Position': self.state.loaded_data['roi'][rhythmic_indices, 0],
                'Y_Position': self.state.loaded_data['roi'][rhythmic_indices, 1],
                'Relative_Phase_Hours': rel_phases,
                'Period_Hours': period
            })
            
            # Save DF (Full set for group)
            if "Cosinor" in method:
                 raw_phases = phases
            else:
                 raw_phases = ((phases / (2 * np.pi)) * period) % period
                 
            if self.filtered_indices is not None:
                # Ensure Original_ROI_Index aligns 1:1 with phase outputs.
                if len(self.filtered_indices) != len(phases):
                     raise ValueError(f"CRITICAL: Phase result count {len(phases)} != Filtered ROI count {len(self.filtered_indices)}. ID mapping corrupted.")
                orig_ids = self.filtered_indices + 1
            else:
                orig_ids = np.arange(len(phases)) + 1
            
            if not (len(orig_ids) == len(phases) == len(rhythm_mask)):
                  raise ValueError(f"CRITICAL: length mismatch orig_ids={len(orig_ids)} phases={len(phases)} rhythm_mask={len(rhythm_mask)}")

            self.latest_rhythm_df = pd.DataFrame({
                'Original_ROI_Index': orig_ids,
                'Phase_Hours': raw_phases,
                'Period_Hours': period,
                'Is_Rhythmic': rhythm_mask,
                'Metric_Score': sort_scores, # R2 or SNR
                'Filter_Score': filter_scores # p-value or SNR
            })
            self.btn_save_rhythm.setEnabled(True)

            # --- Map Viewers ---
            def on_map_click(idx):
                if 0 <= idx < len(df_map):
                    self.on_roi_selected(int(df_map.iloc[idx]['Original_ROI_Index']) - 1)

            fig_p, _ = add_mpl_to_tab(self.mw.phase_tab)
            viewer_p = PhaseMapViewer(fig_p, fig_p.add_subplot(111), self.state.unfiltered_data["background"], df_map, on_map_click, vmin=self.vmin, vmax=self.vmax)
            self.mw.visualization_widgets[self.mw.phase_tab] = viewer_p
            
            grid_res = int(self.phase_params["grid_resolution"][0].text())
            fig_i, _ = add_mpl_to_tab(self.mw.interp_tab)
            
            # Fix: Ensure ROIs are clean dicts for interpolator
            clean_rois = self._normalize_rois_for_viewers(self.rois)
            viewer_i = InterpolatedMapViewer(fig_i, fig_i.add_subplot(111), df_map[['X_Position', 'Y_Position']].values, df_map['Relative_Phase_Hours'].values, period, grid_res, rois=clean_rois)
            self.mw.visualization_widgets[self.mw.interp_tab] = viewer_i
            
        except Exception as e:
            self.mw.log_message(f"Plot Error: {e}")

    def save_rhythm_results(self):
        """
        Save the most recent rhythm results table to CSV.

        Uses:
        - self.latest_rhythm_df (set in regenerate_phase_maps)
        - self.state.output_basename (for default filename)
        """
        if self.latest_rhythm_df is None or len(self.latest_rhythm_df) == 0:
            self.mw.log_message("No rhythm results available to save. Click 'Update Plots' first.")
            return

        start_dir = self.mw._get_last_dir()
        default_name = "rhythm_results.csv"
        if getattr(self.state, "output_basename", None):
            default_name = os.path.basename(self.state.output_basename) + "_rhythm_results.csv"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Rhythm Results",
            os.path.join(start_dir, default_name),
            "CSV files (*.csv)"
        )
        if not path:
            return

        self.mw._set_last_dir(path)
        self.mw._set_last_dir(path)
        try:
            self.latest_rhythm_df.to_csv(path, index=False)
            base = self.state.output_basename if self.state.output_basename else "Session"
            self.mw.log_message(f"[{os.path.basename(base)}] SaveRhythmResults: rows={len(self.latest_rhythm_df)}, file={os.path.basename(path)}")
            self.mw.log_message(f"Saved set definition: (QualityGate PASS) ∩ (Phase filters PASS). These rows will drive Group View.")
        except Exception as e:
            self.mw.log_message(f"Failed to save rhythm results: {e}")

    def export_current_plot(self):
        widget = self.mw.vis_tabs.currentWidget()
        viewer = self.mw.visualization_widgets.get(widget)
        if not viewer or not hasattr(viewer, "fig"): return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", self.mw._get_last_dir(), "PNG (*.png)")
        if path: viewer.fig.savefig(path, dpi=300, bbox_inches="tight")

    def export_current_data(self):
        widget = self.mw.vis_tabs.currentWidget()
        viewer = self.mw.visualization_widgets.get(widget)
        if viewer and hasattr(viewer, "get_export_data"):
            df, name = viewer.get_export_data()
            if df is not None:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", os.path.join(self.mw._get_last_dir(), name), "CSV (*.csv)")
                if path: df.to_csv(path, index=False)