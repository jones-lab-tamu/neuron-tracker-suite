import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.path import Path
from scipy.stats import circmean

from gui.utils import Tooltip, add_mpl_to_tab, clear_layout
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer
from gui.analysis import (
    calculate_phases_fft, 
    preprocess_for_rhythmicity, 
    compute_median_window_frames,
    strict_cycle_mask,
    RHYTHM_TREND_WINDOW_HOURS
)
import cosinor as csn
from gui.theme import get_icon

class GroupViewPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Group Data Setup")
        b = QtWidgets.QVBoxLayout(box)
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        b.addWidget(self.group_list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton(get_icon('fa5s.plus'), "Add Warped ROI File(s)...")
        self.btn_remove_group = QtWidgets.QPushButton(get_icon('fa5s.minus'), "Remove Selected")
        row.addWidget(self.btn_add_group)
        row.addWidget(self.btn_remove_group)
        b.addLayout(row)
        
        param_box = QtWidgets.QGroupBox("Group Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_box)

        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth to fill empty bins")
        Tooltip.install(self.group_smooth_check, "<b>What it is:</b> A 3x3 circular mean smoothing filter.<br><b>How it works:</b> For each empty bin in the Group Average Map, it calculates the circular mean of its 8 neighbors. If any neighbors have data, the empty bin is filled with that mean value.<br><b>Trade-off:</b> Creates a visually smoother map but interpolates data. The unsmoothed map is a more direct representation of the raw data.")
        param_layout.addRow("Grid Resolution:", self.group_grid_res_edit)
        param_layout.addRow(self.group_smooth_check)

        norm_label = QtWidgets.QLabel("Phase Normalization Method:")
        self.norm_global_radio = QtWidgets.QRadioButton("Global Mean (per animal)")
        self.norm_anatomical_radio = QtWidgets.QRadioButton("Individual Phase Reference ROI")
        self.norm_method_group = QtWidgets.QButtonGroup(self)
        self.norm_method_group.addButton(self.norm_global_radio)
        self.norm_method_group.addButton(self.norm_anatomical_radio)
        self.norm_global_radio.setChecked(True)
        
        param_layout.addRow(norm_label)
        param_layout.addRow(self.norm_global_radio)
        param_layout.addRow(self.norm_anatomical_radio)

        self.ref_roi_widget = QtWidgets.QWidget()
        ref_roi_layout = QtWidgets.QHBoxLayout(self.ref_roi_widget)
        ref_roi_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select_ref_roi = QtWidgets.QPushButton(get_icon('fa5s.folder-open'), "Load Reference ROI...")
        self.ref_roi_path_edit = QtWidgets.QLineEdit()
        self.ref_roi_path_edit.setReadOnly(True)
        ref_roi_layout.addWidget(self.btn_select_ref_roi)
        ref_roi_layout.addWidget(self.ref_roi_path_edit)
        param_layout.addRow(self.ref_roi_widget)
        self.ref_roi_widget.hide()

        b.addWidget(param_box)        
        
        self.btn_view_group = QtWidgets.QPushButton(get_icon('fa5s.chart-pie'), "Generate Group Visualizations")
        Tooltip.install(self.btn_view_group, "Loads all specified warped ROI and trace files, calculates phases, and generates the Group Scatter and Group Average Map plots.")
        self.btn_view_group.setEnabled(False)
        b.addWidget(self.btn_view_group)
        layout.addWidget(box)
        layout.addStretch(1)

    def connect_signals(self):
        self.btn_add_group.clicked.connect(self.add_group_files)
        self.btn_remove_group.clicked.connect(self.remove_group_file)
        self.btn_view_group.clicked.connect(self.generate_group_visualizations)
        self.norm_anatomical_radio.toggled.connect(self._on_norm_method_changed)
        self.btn_select_ref_roi.clicked.connect(self._select_reference_roi)

    def add_group_files(self):
        start_dir = self.mw._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Warped ROI Files",
            start_dir,
            "Warped ROI files (*_roi_warped.csv)",
        )
        if not files:
            return
        self.mw._set_last_dir(files[0])
        for f in files:
            if f not in self.state.group_data_paths:
                self.state.group_data_paths.append(f)
                self.group_list.addItem(f)
        self._update_group_view_button()

    def remove_group_file(self):
        for item in self.group_list.selectedItems():
            row = self.group_list.row(item)
            path = self.group_list.item(row).text()
            if path in self.state.group_data_paths:
                self.state.group_data_paths.remove(path)
            self.group_list.takeItem(row)
        self._update_group_view_button()

    def _update_group_view_button(self):
        self.btn_view_group.setEnabled(len(self.state.group_data_paths) > 0)

    @QtCore.pyqtSlot(bool)
    def _on_norm_method_changed(self, checked):
        # We no longer show the master reference loader, as we rely on individual files.
        self.ref_roi_widget.hide()
            
    def _select_reference_roi(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Phase Reference ROI File",
            start_dir,
            "JSON files (*.json)",
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        self.state.reference_roi_path = path
        self.ref_roi_path_edit.setText(path)
        self.mw.log_message(f"Loaded phase reference ROI: {os.path.basename(path)}")

    def generate_group_visualizations(self):
        if not self.state.group_data_paths:
            self.mw.log_message("No group files selected.")
            return
        
        self.mw.log_message("Loading and processing group data...")
        single_animal_tabs = [self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab]
        group_tabs = [self.mw.group_scatter_tab, self.mw.group_avg_tab]
        
        try:
            valid_fft_keys = ["minutes_per_frame", "period_min", "period_max"]
            phase_args = {}
            
            # Grab global settings for fallback usage
            if hasattr(self.mw, 'single_panel'):
                for key in valid_fft_keys:
                    if key in self.mw.single_panel.phase_params:
                        widget, type_caster = self.mw.single_panel.phase_params[key]
                        value_str = widget.text().strip()
                        if value_str: phase_args[key] = type_caster(value_str)
            
            if "minutes_per_frame" not in phase_args: 
                raise ValueError("Minutes per Frame must be set for group analysis.")
            
            # Global fallback settings
            method = self.mw.single_panel.analysis_method_combo.currentText()
            thresh = float(self.mw.single_panel.phase_params["rhythm_threshold"][0].text())
            r_thresh = float(self.mw.single_panel.phase_params["r_squared_threshold"][0].text())
            
            all_dfs = []
            
            for i, roi_file in enumerate(self.state.group_data_paths):
                self.mw.log_message(f"  [{i + 1}/{len(self.state.group_data_paths)}] {os.path.basename(roi_file)}")
                
                traces_file = roi_file.replace("_roi_warped.csv", "_traces.csv")
                unfiltered_roi_file = roi_file.replace("_roi_warped.csv", "_roi.csv")
                filtered_roi_file = roi_file.replace("_roi_warped.csv", "_roi_filtered.csv")
                rhythm_file = roi_file.replace("_roi_warped.csv", "_rhythm_results.csv")
                
                # Load Spatial Data
                if not os.path.exists(roi_file) or not os.path.exists(filtered_roi_file):
                     self.mw.log_message("    Error: Missing spatial files. Skipping.")
                     continue
                
                warped_rois = np.atleast_2d(np.loadtxt(roi_file, delimiter=","))
                native_filtered_rois = np.atleast_2d(np.loadtxt(filtered_roi_file, delimiter=","))
                
                phases = None
                period = None
                mask = None

                # --- STRATEGY 1: LOAD LOCKED-IN RESULTS (Preferred) ---
                if os.path.exists(rhythm_file):
                    self.mw.log_message(f"    -> Found saved rhythm results. Loading...")
                    try:
                        rhythm_df = pd.read_csv(rhythm_file)
                        
                        # Validation: Ensure row counts match
                        if len(rhythm_df) != len(warped_rois):
                            self.mw.log_message("    WARNING: Rhythm file length mismatch vs Spatial file. Falling back to re-calculation.")
                        else:
                            # Load Data from File
                            phases = rhythm_df['Phase_Hours'].values # Already in Hours
                            period = rhythm_df['Period_Hours'].iloc[0]
                            mask = rhythm_df['Is_Rhythmic'].astype(bool).values
                    except Exception as e:
                        self.mw.log_message(f"    ERROR reading results file: {e}. Falling back.")

                # --- STRATEGY 2: RE-CALCULATE (Fallback) ---
                if phases is None:
                    self.mw.log_message("    -> Re-calculating using current global settings...")
                    
                    if not os.path.exists(traces_file) or not os.path.exists(unfiltered_roi_file):
                         self.mw.log_message("    Error: Missing trace data for re-calculation. Skipping.")
                         continue

                    traces = np.loadtxt(traces_file, delimiter=",")
                    unfiltered_rois = np.loadtxt(unfiltered_roi_file, delimiter=",")
                    
                    # Match indices
                    indices = []
                    for point in native_filtered_rois:
                        diff = np.sum((unfiltered_rois - point)**2, axis=1)
                        idx = np.argmin(diff)
                        if diff[idx] < 1e-9: indices.append(idx)
                    
                    if len(indices) != len(native_filtered_rois):
                        self.mw.log_message("    Error: Could not match filtered ROIs to traces. Skipping.")
                        continue
                        
                    trace_indices_to_keep = np.array(indices) + 1
                    filtered_traces_data = traces[:, np.concatenate(([0], trace_indices_to_keep))]
                    
                    if "Cosinor" in method:
                        mpf = phase_args["minutes_per_frame"]
                        time_points_hours = filtered_traces_data[:, 0] * (mpf / 60.0)
                        _, period, _ = calculate_phases_fft(filtered_traces_data, **phase_args)
                        
                        phases_list, p_values, r_squareds = [], [], []
                        trend_window_hours = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
                        median_window_frames = compute_median_window_frames(mpf, trend_window_hours, T=filtered_traces_data.shape[0])
                        
                        for col in range(1, filtered_traces_data.shape[1]):
                            raw_trace = filtered_traces_data[:, col]
                            detrended = preprocess_for_rhythmicity(raw_trace, method="running_median", median_window_frames=median_window_frames)
                            res = csn.cosinor_analysis(detrended, time_points_hours, period)
                            phases_list.append(res["acrophase"])
                            p_values.append(res["p_value"])
                            r_squareds.append(res["r_squared"])
                        
                        phases = np.array(phases_list)
                        mask = (np.array(p_values) <= thresh) & (np.array(r_squareds) >= r_thresh)
                    else:
                        phases_rad, period, scores = calculate_phases_fft(filtered_traces_data, **phase_args)
                        mask = scores >= thresh
                        phases = (phases_rad / (2 * np.pi)) * period
                        phases = phases % period

                    # Apply Strict Cycle if Recalculating
                    if self.mw.single_panel.strict_cycle_check.isChecked():
                        mpf = phase_args["minutes_per_frame"]
                        trend_win = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
                        mask = strict_cycle_mask(filtered_traces_data, minutes_per_frame=mpf, period_hours=period,
                                                 base_mask=mask, min_cycles=2, trend_window_hours=trend_win)
                        self.mw.log_message(f"    -> Applied Strict Cycle Filter")

                # --- APPLY MASK ---
                phases = phases[mask]
                warped_rois = warped_rois[mask]
                native_filtered_rois = native_filtered_rois[mask]
                
                if len(phases) == 0:
                    self.mw.log_message("    Warning: No valid cells found. Skipping.")
                    continue
                
                self.mw.log_message(f"    -> Kept {len(phases)} cells")

                # --- NORMALIZATION ---
                mean_h = 0.0
                
                if self.norm_global_radio.isChecked():
                     ph_rad = (phases / period) * (2 * np.pi)
                     mean_rad = circmean(ph_rad)
                     mean_h = mean_rad * (period / (2 * np.pi))
                     mean_h = mean_h % period
                     self.mw.log_message(f"    -> Norm: Global Mean ({mean_h:.2f}h)")

                elif self.norm_anatomical_radio.isChecked():
                    individual_roi_path = roi_file.replace("_roi_warped.csv", "_anatomical_roi.json")
                    individual_ref_poly = []
                    
                    if os.path.exists(individual_roi_path):
                        with open(individual_roi_path, 'r') as f:
                            ind_rois = json.load(f)
                            individual_ref_poly = [r for r in ind_rois if r.get("mode") == "Phase Reference"]

                    ref_indices = []
                    if individual_ref_poly:
                        self.mw.log_message("    -> Norm: Individual 'Phase Reference' ROI (Native Space)")
                        ref_mask = np.zeros(len(native_filtered_rois), dtype=bool)
                        for roi in individual_ref_poly:
                            path = Path(roi['path_vertices'])
                            ref_mask |= path.contains_points(native_filtered_rois)
                        ref_indices = np.where(ref_mask)[0]
                    else:
                         self.mw.log_message("    -> Warning: No 'Phase Reference' ROI. Falling back to Global Mean.")

                    if len(ref_indices) > 0:
                        ref_phases = phases[ref_indices]
                        ref_rad = (ref_phases / period) * (2 * np.pi)
                        ref_mean_rad = circmean(ref_rad)
                        mean_h = ref_mean_rad * (period / (2 * np.pi))
                        self.mw.log_message(f"       Ref Mean: {mean_h:.2f}h (n={len(ref_indices)})")
                    elif individual_ref_poly:
                        self.mw.log_message("    Warning: Ref ROI empty. Skipping.")
                        continue
                    else:
                        # Fallback calc already done
                        ph_rad = (phases / period) * (2 * np.pi)
                        mean_rad = circmean(ph_rad)
                        mean_h = mean_rad * (period / (2 * np.pi))

                rel_phases = (phases - mean_h + period / 2) % period - period / 2

                animal_df = pd.DataFrame({
                    'Source_Animal': os.path.basename(roi_file).replace('_roi_warped.csv', ''),
                    'Warped_X': warped_rois[:, 0],
                    'Warped_Y': warped_rois[:, 1],
                    'Relative_Phase_Hours': rel_phases,
                    'Period_Hours': period
                })
                all_dfs.append(animal_df)

            if not all_dfs: 
                self.mw.log_message("No valid group data generated.")
                return
            
            group_scatter_df = pd.concat(all_dfs, ignore_index=True)
            period = group_scatter_df['Period_Hours'].iloc[0]
            grid_res = int(self.group_grid_res_edit.text())
            do_smooth = self.group_smooth_check.isChecked()
            
            x_min, x_max = group_scatter_df['Warped_X'].min(), group_scatter_df['Warped_X'].max()
            y_min, y_max = group_scatter_df['Warped_Y'].min(), group_scatter_df['Warped_Y'].max()
            
            # --- SQUARE BINS CALCULATION ---
            width = x_max - x_min
            height = y_max - y_min
            
            if width >= height:
                n_bins_x = grid_res
                bin_size = width / n_bins_x
                n_bins_y = int(round(height / bin_size))
            else:
                n_bins_y = grid_res
                bin_size = height / n_bins_y
                n_bins_x = int(round(width / bin_size))
                
            n_bins_x = max(1, n_bins_x)
            n_bins_y = max(1, n_bins_y)
            # -------------------------------

            # --- EXTENDED GRID FOR VISUALIZATION ---
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            max_range = max(width, height)
            half_span = (max_range * 1.15) / 2 
            
            view_x_min = cx - half_span
            view_x_max = cx + half_span
            view_y_min = cy - half_span
            view_y_max = cy + half_span
            
            start_x = x_min - np.ceil((x_min - view_x_min) / bin_size) * bin_size
            end_x = x_max + np.ceil((view_x_max - x_max) / bin_size) * bin_size
            start_y = y_min - np.ceil((y_min - view_y_min) / bin_size) * bin_size
            end_y = y_max + np.ceil((view_y_max - y_max) / bin_size) * bin_size
            
            grid_x_bins = np.arange(start_x, end_x + bin_size/1000, bin_size)
            grid_y_bins = np.arange(start_y, end_y + bin_size/1000, bin_size)
            # ---------------------------------------
            
            # --- DATA BINNING ---
            calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
            calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)

            group_scatter_df['Grid_X_Index'] = pd.cut(group_scatter_df['Warped_X'], bins=calc_x_bins, labels=False, include_lowest=True).fillna(-1).astype(int)
            group_scatter_df['Grid_Y_Index'] = pd.cut(group_scatter_df['Warped_Y'], bins=calc_y_bins, labels=False, include_lowest=True).fillna(-1).astype(int)

            def circmean_phase(series):
                rad = (series / (period / 2.0)) * np.pi
                # Force circmean to return values centered at 0 (-pi to +pi) 
                mean_rad = circmean(rad, low=-np.pi, high=np.pi)
                return (mean_rad / np.pi) * (period / 2.0)
            
            group_binned_df = group_scatter_df[group_scatter_df['Grid_X_Index'] >= 0].groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
            
            fig_s, _ = add_mpl_to_tab(self.mw.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), group_scatter_df, grid_bins=(grid_x_bins, grid_y_bins))
            self.mw.visualization_widgets[self.mw.group_scatter_tab] = viewer_s
            
            fig_g, _ = add_mpl_to_tab(self.mw.group_avg_tab)
            viewer_g = GroupAverageMapViewer(fig_g, fig_g.add_subplot(111), group_binned_df, group_scatter_df, (n_bins_x, n_bins_y), do_smooth)
            self.mw.visualization_widgets[self.mw.group_avg_tab] = viewer_g
            
            for tab in group_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), True)
            for tab in single_animal_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
            
            self.mw.vis_tabs.setCurrentWidget(self.mw.group_scatter_tab)
            self.mw._mark_step_ready("group_view")
            self.mw.log_message("Group visualizations generated.")
            self.mw.btn_export_data.setEnabled(True)
            
        except Exception as e:
            import traceback
            self.mw.log_message(f"Error generating group visualizations: {e}")
            self.mw.log_message(traceback.format_exc())
            for tab in group_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
