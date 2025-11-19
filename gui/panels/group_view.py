import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.path import Path
from scipy.stats import circmean

from gui.utils import Tooltip, add_mpl_to_tab
from gui.analysis import calculate_phases_fft
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer

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
        self.norm_anatomical_radio = QtWidgets.QRadioButton("Anatomical Reference ROI")
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
        if self.norm_anatomical_radio.isChecked():
            self.ref_roi_widget.show()
        else:
            self.ref_roi_widget.hide()
            
    def _select_reference_roi(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Anatomical Reference ROI File",
            start_dir,
            "JSON files (*.json)",
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        self.state.reference_roi_path = path
        self.ref_roi_path_edit.setText(path)
        self.mw.log_message(f"Loaded anatomical reference ROI: {os.path.basename(path)}")

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
            # Access phase params from the SingleAnimalPanel
            # Assuming main_window has 'single_panel' attribute
            if not hasattr(self.mw, 'single_panel'):
                raise RuntimeError("SingleAnimalPanel not found in MainWindow")
                
            for key in valid_fft_keys:
                if key in self.mw.single_panel.phase_params:
                    widget, type_caster = self.mw.single_panel.phase_params[key]
                    value_str = widget.text().strip()
                    if value_str: phase_args[key] = type_caster(value_str)
            if "minutes_per_frame" not in phase_args: raise ValueError("Minutes per Frame must be set for group analysis.")
            
            master_ref_rois = []
            if self.norm_anatomical_radio.isChecked():
                if not self.state.reference_roi_path or not os.path.exists(self.state.reference_roi_path):
                    QtWidgets.QMessageBox.warning(self, "Error", "Anatomical Reference method selected, but no valid reference ROI file has been loaded.")
                    return
                with open(self.state.reference_roi_path, 'r') as f:
                    master_ref_rois = json.load(f)

            all_dfs = []
            
            for i, roi_file in enumerate(self.state.group_data_paths):
                self.mw.log_message(f"  [{i + 1}/{len(self.state.group_data_paths)}] {os.path.basename(roi_file)}")
                
                traces_file = roi_file.replace("_roi_warped.csv", "_traces.csv")
                unfiltered_roi_file = roi_file.replace("_roi_warped.csv", "_roi.csv")
                filtered_roi_file = roi_file.replace("_roi_warped.csv", "_roi_filtered.csv")
                
                required_files = [traces_file, unfiltered_roi_file, filtered_roi_file, roi_file]
                if not all(os.path.exists(f) for f in required_files):
                    self.mw.log_message("    Warning: missing one or more required files; skipping.")
                    continue
                
                warped_rois = np.atleast_2d(np.loadtxt(roi_file, delimiter=","))
                native_filtered_rois = np.atleast_2d(np.loadtxt(filtered_roi_file, delimiter=",")) 
                
                if warped_rois.shape[0] != native_filtered_rois.shape[0]:
                    self.mw.log_message(f"    ERROR: Data inconsistency (warped vs filtered count). Skipping.")
                    continue
                
                traces = np.loadtxt(traces_file, delimiter=",")
                unfiltered_rois = np.loadtxt(unfiltered_roi_file, delimiter=",")
                
                indices = []
                for point in native_filtered_rois:
                    diff = np.sum((unfiltered_rois - point)**2, axis=1)
                    idx = np.argmin(diff)
                    if diff[idx] < 1e-9: indices.append(idx)
                
                if len(indices) != native_filtered_rois.shape[0]:
                    self.mw.log_message("    Warning: Could not match filtered to unfiltered ROIs; skipping.")
                    continue
                
                trace_indices_to_keep = np.array(indices) + 1
                filtered_traces_data = traces[:, np.concatenate(([0], trace_indices_to_keep))]
                
                phases, period, _ = calculate_phases_fft(filtered_traces_data, **phase_args)

                mean_h = 0.0
                
                if self.norm_global_radio.isChecked():
                     ph_rad = (phases % period) * (2 * np.pi / period)
                     mean_rad = circmean(ph_rad)
                     mean_h = mean_rad * (period / (2 * np.pi))
                     self.mw.log_message(f"    -> Norm: Global Mean ({mean_h:.2f}h)")

                else:
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
                        self.mw.log_message("    -> Norm: Master Atlas ROI (Warped Space)")
                        master_ref_poly = [r for r in master_ref_rois if r.get("mode") == "Phase Reference"]
                        
                        if not master_ref_poly:
                            self.mw.log_message("    ERROR: No 'Phase Reference' polygon found in Master Atlas file. Cannot normalize.")
                            continue
                            
                        ref_mask = np.zeros(len(warped_rois), dtype=bool)
                        for roi in master_ref_poly:
                            path = Path(roi['path_vertices'])
                            ref_mask |= path.contains_points(warped_rois)
                        ref_indices = np.where(ref_mask)[0]

                    if len(ref_indices) == 0:
                        self.mw.log_message("    Warning: No rhythmic cells found in reference region. Skipping.")
                        continue

                    ref_phases = phases[ref_indices]
                    ref_rad = (ref_phases % period) * (2 * np.pi / period)
                    ref_mean_rad = circmean(ref_rad)
                    mean_h = ref_mean_rad * (period / (2 * np.pi))
                    self.mw.log_message(f"       Ref Mean: {mean_h:.2f}h (n={len(ref_indices)})")

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
            
            grid_x_bins = np.linspace(x_min, x_max, grid_res + 1)
            grid_y_bins = np.linspace(y_min, y_max, grid_res + 1)
            
            group_scatter_df['Grid_X_Index'] = pd.cut(group_scatter_df['Warped_X'], bins=grid_x_bins, labels=False, include_lowest=True).fillna(-1).astype(int)
            group_scatter_df['Grid_Y_Index'] = pd.cut(group_scatter_df['Warped_Y'], bins=grid_y_bins, labels=False, include_lowest=True).fillna(-1).astype(int)

            def circmean_phase(series):
                rad = (series / (period / 2.0)) * np.pi
                mean_rad = circmean(rad)
                return (mean_rad / np.pi) * (period / 2.0)
            
            group_binned_df = group_scatter_df[group_scatter_df['Grid_X_Index'] >= 0].groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
            
            fig_s, _ = add_mpl_to_tab(self.mw.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), group_scatter_df)
            self.mw.visualization_widgets[self.mw.group_scatter_tab] = viewer_s
            
            fig_g, _ = add_mpl_to_tab(self.mw.group_avg_tab)
            viewer_g = GroupAverageMapViewer(fig_g, fig_g.add_subplot(111), group_binned_df, group_scatter_df, grid_res, do_smooth)
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
