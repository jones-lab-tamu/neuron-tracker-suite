import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from scipy.stats import circmean, linregress, f as f_dist
from skimage.draw import polygon as draw_polygon
from typing import Tuple

from gui.utils import Tooltip, add_mpl_to_tab, clear_layout, project_points_to_polyline
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer, PhaseGradientViewer, RegionResultViewer
from gui.statistics import watson_williams_f
from gui.dialogs.roi_drawer import ROIDrawerDialog
from gui.theme import get_icon

def load_and_join_rhythm_and_coords(rhythm_csv_path: str, warped_ids_csv_path: str) -> "Tuple[pd.DataFrame, dict]":
    """
    Loads rhythm results and warped coordinates-with-IDs and returns:
      - merged DataFrame (inner join on Original_ROI_Index)
      - stats dict with keys: n_rhythm, n_warped, n_merged, drop_rhythm, drop_warped, dup_rhythm, dup_warped
    Must not touch GUI state.
    Must enforce the no-row-order rule.
    """
    rhythm_df = pd.read_csv(rhythm_csv_path)
    warped_df = pd.read_csv(warped_ids_csv_path)
    
    for col in ['Original_ROI_Index', 'Is_Rhythmic', 'Phase_Hours', 'Period_Hours']:
        if col not in rhythm_df.columns: raise ValueError(f"Missing '{col}' in {rhythm_csv_path}")
    for col in ['Original_ROI_Index', 'X_Warped', 'Y_Warped']:
        if col not in warped_df.columns: raise ValueError(f"Missing '{col}' in {warped_ids_csv_path}")

    dup_rhythm = rhythm_df.duplicated(subset='Original_ROI_Index').sum()
    dup_warped = warped_df.duplicated(subset='Original_ROI_Index').sum()
    n_rhythm_raw, n_warped_raw = len(rhythm_df), len(warped_df)

    rhythm_df = rhythm_df.drop_duplicates(subset='Original_ROI_Index', keep='first')
    warped_df = warped_df.drop_duplicates(subset='Original_ROI_Index', keep='first')

    merged = rhythm_df.merge(warped_df, on="Original_ROI_Index", how="inner", validate="one_to_one")
    
    stats = {
        'n_rhythm': n_rhythm_raw, 'n_warped': n_warped_raw, 'n_merged': len(merged),
        'drop_rhythm': n_rhythm_raw - len(merged), 'drop_warped': n_warped_raw - len(merged),
        'dup_rhythm': dup_rhythm, 'dup_warped': dup_warped
    }
    return merged, stats

class GroupViewPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        self.zones_polygons_map = {} 
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Group Data Setup")
        b = QtWidgets.QVBoxLayout(box)
        
        self.group_list = QtWidgets.QTreeWidget()
        self.group_list.setHeaderLabels(["File Path", "Group"])
        self.group_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        b.addWidget(self.group_list)
        
        row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton(get_icon('fa5s.plus'), "Add Warped ROI File(s)...")
        self.btn_remove_group = QtWidgets.QPushButton(get_icon('fa5s.minus'), "Remove Selected")
        row.addWidget(self.btn_add_group)
        row.addWidget(self.btn_remove_group)
        b.addLayout(row)
        
        assign_row = QtWidgets.QHBoxLayout()
        self.btn_assign_control = QtWidgets.QPushButton("Set Selected as Control")
        self.btn_assign_exp = QtWidgets.QPushButton("Set Selected as Experiment")
        assign_row.addWidget(self.btn_assign_control)
        assign_row.addWidget(self.btn_assign_exp)
        b.addLayout(assign_row)
        
        param_box = QtWidgets.QGroupBox("Continuous Map Parameters")
        param_layout = QtWidgets.QFormLayout(param_box)
        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth to fill empty bins")
        param_layout.addRow("Grid Resolution:", self.group_grid_res_edit)
        param_layout.addRow(self.group_smooth_check)
        
        self.norm_mode_combo = QtWidgets.QComboBox()
        self.norm_mode_combo.addItems(["Normalize: Control mean", "Normalize: Per-animal mean"])
        param_layout.addRow(self.norm_mode_combo)
        self.norm_mode_combo.addItems(["Normalize: Control mean", "Normalize: Per-animal mean"])
        param_layout.addRow(self.norm_mode_combo)
        
        # Gradient Parameters
        self.grad_mode_combo = QtWidgets.QComboBox()
        self.grad_mode_combo.addItems(["DV (Y-normalized)", "Phase Axis (polyline)"])
        param_layout.addRow("Gradient Coordinate:", self.grad_mode_combo)
        
        self.collapse_gradient_cb = QtWidgets.QCheckBox("Collapse bilateral axes (Left+Right)")
        self.collapse_gradient_cb.setToolTip("If checked, bilateral gradients are also pooled into a single 'Collapsed' gradient (requires 2 Phase Axes).")
        param_layout.addRow(self.collapse_gradient_cb)
        
        b.addWidget(param_box)        
        
        region_box = QtWidgets.QGroupBox("Regional Analysis Setup")
        region_layout = QtWidgets.QFormLayout(region_box)
        
        atlas_box = QtWidgets.QWidget()
        atlas_layout = QtWidgets.QHBoxLayout(atlas_box)
        atlas_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_load_atlas = QtWidgets.QPushButton(get_icon('fa5s.map'), "Load Atlas Template...")
        self.atlas_path_label = QtWidgets.QLineEdit()
        self.atlas_path_label.setPlaceholderText("No Atlas Loaded")
        self.atlas_path_label.setReadOnly(True)
        atlas_layout.addWidget(self.btn_load_atlas)
        atlas_layout.addWidget(self.atlas_path_label)
        region_layout.addRow("Atlas Template:", atlas_box)
        
        self.btn_define_regions = QtWidgets.QPushButton(get_icon('fa5s.draw-polygon'), "Define Analysis Regions...")
        self.region_status_label = QtWidgets.QLabel("No regions defined")
        self.region_status_label.setStyleSheet("color: gray; font-style: italic;")
        region_layout.addRow(self.btn_define_regions, self.region_status_label)

        self.min_region_cells_spin = QtWidgets.QSpinBox()
        self.min_region_cells_spin.setRange(1, 100)
        self.min_region_cells_spin.setValue(10)
        region_layout.addRow("Min Cells / Region / Animal:", self.min_region_cells_spin)

        self.min_animals_spin = QtWidgets.QSpinBox()
        self.min_animals_spin.setRange(2, 50)
        self.min_animals_spin.setValue(3)
        region_layout.addRow("Min Animals / Group (Stats):", self.min_animals_spin)

        b.addWidget(region_box)
        
        self.btn_view_group = QtWidgets.QPushButton(get_icon('fa5s.chart-pie'), "Generate Group Visualizations")
        Tooltip.install(self.btn_view_group, "Runs Continuous Grid analysis, Gradient analysis (if axes present), and Regional Statistical analysis.")
        self.btn_view_group.setEnabled(False)
        b.addWidget(self.btn_view_group)
        
        layout.addWidget(box)
        layout.addStretch(1)

    def connect_signals(self):
        self.btn_add_group.clicked.connect(self.add_group_files)
        self.btn_remove_group.clicked.connect(self.remove_group_file)
        self.btn_view_group.clicked.connect(self.generate_group_visualizations)
        self.btn_load_atlas.clicked.connect(self.load_atlas_template)
        self.btn_assign_control.clicked.connect(lambda: self.assign_group("Control"))
        self.btn_assign_exp.clicked.connect(lambda: self.assign_group("Experiment"))
        self.btn_define_regions.clicked.connect(self.define_regions)
        self.mw.btn_export_data.clicked.connect(self.export_current_data)
        self.norm_mode_combo.currentIndexChanged.connect(self._on_norm_mode_changed)
    
    
    def _clear_tab_contents(self, tab_widget: QtWidgets.QWidget) -> None:
        """
        Remove all matplotlib canvases/figures associated with this tab to prevent duplicates.
        Must be safe if tab has no layout, nested widgets, or scroll areas.
        """
        if tab_widget is None:
            return

        # (1) Drop stored viewer for this tab (if present) so it can be GC'd.
        # IMPORTANT: viewer may not have `.fig`, so do not rely only on viewer.fig.
        if hasattr(self.mw, "visualization_widgets") and isinstance(self.mw.visualization_widgets, dict):
            self.mw.visualization_widgets.pop(tab_widget, None)

        # (2) Close + delete ALL FigureCanvasQTAgg under this tab.
        # This is the REQUIRED hardening: close canvas.figure even if viewer has no `.fig`.
        for canvas in tab_widget.findChildren(FigureCanvasQTAgg):
            try:
                plt.close(canvas.figure)
            except Exception:
                pass
            canvas.setParent(None)
            canvas.deleteLater()

        # (3) Clear ONLY the DIRECT layout on the tab (do NOT recursively clear every child layout).
        # Reason: recursive clearing can delete non-plot UI elements if present.
        lay = tab_widget.layout()
        if lay is not None:
            clear_layout(lay)

    def _on_norm_mode_changed(self):
        if not hasattr(self, "_last_master_df") or self._last_master_df is None:
            return

        df = self._last_master_df.copy()

        mode = self.norm_mode_combo.currentText()
        use_animal_norm = (mode == "Normalize: Per-animal mean")

        if use_animal_norm:
            if "Rel_Phase_Animal" not in df.columns:
                self.mw.log_message("Error: Per-animal normalization data missing.")
                return
            df["Rel_Phase"] = df["Rel_Phase_Animal"]
            self.mw.log_message("Updated to Per-Animal Normalization.")
        else:
            # Require valid control normalization to switch
            if "Rel_Phase_Control" not in df.columns:
                self.mw.log_message("Cannot switch to Control Mean: No control data available from last run.")
                return
            arr = df["Rel_Phase_Control"].to_numpy(dtype=float, copy=False)
            if (~np.isfinite(arr)).all():
                self.mw.log_message("Cannot switch to Control Mean: No valid control data available from last run.")
                return
            df["Rel_Phase"] = df["Rel_Phase_Control"]
            self.mw.log_message("Updated to Control Mean Normalization.")

        # Clear ALL tabs that are plotted into by THIS FILE.
        # CRITICAL: use the real tab attribute names used in add_mpl_to_tab calls.
        # Find all occurrences of: add_mpl_to_tab(self.mw.<TABNAME>)
        # and clear exactly those tabs.
        self._clear_tab_contents(self.mw.group_scatter_tab)
        self._clear_tab_contents(self.mw.group_avg_tab)
        self._clear_tab_contents(getattr(self.mw, 'grad_tab', None))

        # If there is a regional plot tab that uses add_mpl_to_tab, include it too.
        # DO NOT guess attribute names. Add only if found via the search above.

        # Now regenerate visualizations using df (with df["Rel_Phase"] already selected)
        self._generate_continuous_maps(df)
        self._generate_gradient_analysis(df)
        self._generate_regional_stats(df)
    def assign_group(self, group_name: str):
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select one or more files to assign.")
            return
        for item in selected_items:
            item.setText(1, group_name)
            item.setForeground(1, QtGui.QColor('blue') if group_name == "Control" else QtGui.QColor('red'))
    
    def define_regions(self):
        atlas_path = self.state.atlas_roi_path
        if not atlas_path or not os.path.exists(atlas_path):
            QtWidgets.QMessageBox.warning(self, "No Atlas", "Please load an Atlas Template first.")
            return

        try:
            with open(atlas_path, 'r') as f:
                atlas_rois = json.load(f)
            
            all_verts = []
            for r in atlas_rois:
                if 'path_vertices' in r: all_verts.extend(r['path_vertices'])
            
            if not all_verts: return
            all_verts = np.array(all_verts)
            max_x, max_y = np.max(all_verts[:, 0]), np.max(all_verts[:, 1])
            bg_image = np.zeros((int(max_y) + 50, int(max_x) + 50), dtype=float)
            
            for r in atlas_rois:
                if r.get('mode') == 'Include':
                    poly = np.array(r['path_vertices'])
                    rr, cc = draw_polygon(poly[:, 1], poly[:, 0], shape=bg_image.shape)
                    bg_image[rr, cc] = 0.3 
            
            region_file = atlas_path.replace('_anatomical_roi.json', '_anatomical_regions.json')
            existing_regions = []
            if os.path.exists(region_file):
                with open(region_file, 'r') as f: existing_regions = json.load(f)

            def save_callback(indices, rois_list, refs_list):
                final_list = (rois_list or []) + (refs_list or [])
                if not final_list: return
                serializable = []
                for r in final_list:
                    verts = r["path_vertices"]
                    if hasattr(verts, 'tolist'): verts = verts.tolist()
                    elif isinstance(verts, np.ndarray): verts = verts.tolist()
                    item = {"path_vertices": verts, "mode": r["mode"]}
                    for k in ["zone_id", "lobe", "name"]:
                        if k in r: item[k] = r[k]
                    serializable.append(item)
                with open(region_file, 'w') as f: json.dump(serializable, f, indent=4)
                self.mw.log_message(f"Saved {len(serializable)} regions.")
                self.region_status_label.setText(f"{len(serializable)} regions loaded.")
                self.region_status_label.setStyleSheet("color: green; font-weight: bold;")

            dlg = ROIDrawerDialog(self.mw, bg_image, None, None, save_callback, vmin=0, vmax=1, is_region_mode=True)
            if existing_regions:
                dlg.rois = existing_regions
                dlg.redraw_finished_rois()
            dlg.exec_()

        except Exception as e:
            self.mw.log_message(f"Error launching region definition: {e}")

    def load_atlas_template(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Master Atlas ROI", start_dir, "JSON files (*.json)"
        )
        if not path: return
        self.mw._set_last_dir(path)
        self.state.atlas_roi_path = path
        self.atlas_path_label.setText(os.path.basename(path))
        
        region_file = path.replace('_anatomical_roi.json', '_anatomical_regions.json')
        if os.path.exists(region_file):
            with open(region_file, 'r') as f: regions = json.load(f)
            self.region_status_label.setText(f"{len(regions)} regions found.")
            self.region_status_label.setStyleSheet("color: green;")

    def add_group_files(self):
        start_dir = self.mw._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Warped ROI Files", start_dir, "Warped ROI files (*_roi_warped_with_ids.csv)"
        )
        if not files: return
        self.mw._set_last_dir(files[0])
        for f in files:
            if f not in self.state.group_data_paths:
                self.state.group_data_paths.append(f)
                item = QtWidgets.QTreeWidgetItem(self.group_list)
                item.setText(0, f)
                item.setText(1, "Unassigned")
        self._update_group_view_button()

    def remove_group_file(self):
        selected_items = self.group_list.selectedItems()
        for item in selected_items:
            path = item.text(0)
            if path in self.state.group_data_paths:
                self.state.group_data_paths.remove(path)
            (item.parent() or self.group_list.invisibleRootItem()).removeChild(item)
        self._update_group_view_button()

    def _update_group_view_button(self):
        self.btn_view_group.setEnabled(len(self.state.group_data_paths) > 0)

    def generate_group_visualizations(self):
        self.mw.log_message("--- Starting Group Analysis ---")
        
        if not self.state.group_data_paths: return
        atlas_path = self.state.atlas_roi_path
        if not atlas_path:
            self.mw.log_message("Error: No Atlas Template loaded.")
            return

        region_file = atlas_path.replace('_anatomical_roi.json', '_anatomical_regions.json')
        if not os.path.exists(region_file):
            self.mw.log_message(f"Error: Regions file not found. Please Define Regions first.")
            return

        try:
            with open(region_file, 'r') as f: raw_regions = json.load(f)
            self.zones = {} 
            self.zones_polygons_map = {} 
            
            for r in raw_regions:
                if 'zone_id' not in r: continue
                zid = r['zone_id']
                if zid not in self.zones:
                    self.zones[zid] = {'name': r.get('name', f"Zone {zid}"), 'polygons': []}
                    self.zones_polygons_map[zid] = []
                
                path_obj = Path(np.array(r['path_vertices']))
                self.zones[zid]['polygons'].append(path_obj)
                self.zones_polygons_map[zid].append(path_obj)
            
            self.mw.log_message(f"Loaded {len(self.zones)} anatomical zones.")
        except Exception as e:
            self.mw.log_message(f"Error parsing regions: {e}")
            return

        all_dfs = []
        try:
            group_map = {}
            root = self.group_list.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                group_map[item.text(0)] = item.text(1)

            for roi_path in self.state.group_data_paths:
                base = os.path.basename(roi_path).replace('_roi_warped_with_ids.csv', '')
                group = group_map.get(roi_path, "Unassigned")
                if group not in ["Control", "Experiment"]: continue
                
                rhythm_path = roi_path.replace("_roi_warped_with_ids.csv", "_rhythm_results.csv")
                if not os.path.exists(rhythm_path): 
                     self.mw.log_message(f"Skipping {base}: No rhythm results.")
                     continue
                if not os.path.exists(roi_path):
                     self.mw.log_message(f"Skipping {base}: missing _roi_warped_with_ids.csv. Re-run Apply Warp after updating pipeline.")
                     continue

                merged, stats = load_and_join_rhythm_and_coords(rhythm_path, roi_path)
                self.mw.log_message(f"Join {base}: rhythm={stats['n_rhythm']}, warped={stats['n_warped']}, merged={stats['n_merged']}, drop_rhythm={stats['drop_rhythm']}, drop_warped={stats['drop_warped']}, dup_rhythm={stats['dup_rhythm']}, dup_warped={stats['dup_warped']}")

                rhythmic = merged[merged['Is_Rhythmic'].astype(bool)]
                self.mw.log_message(f"GroupView[{base}] uses only rows where Is_Rhythmic==True after join. Final Count={len(rhythmic)}")
                if rhythmic.empty: continue

                phases = rhythmic['Phase_Hours'].values
                periods = rhythmic['Period_Hours'].values
                phases_ct = (phases / periods) * 24.0
                
                df = pd.DataFrame({
                    'Animal': base, 'Group': group,
                    'X': rhythmic['X_Warped'].values, 'Y': rhythmic['Y_Warped'].values,
                    'Phase_CT': phases_ct,
                    'Period_Hours': 24.0
                })
                all_dfs.append(df)

            if not all_dfs:
                self.mw.log_message("No valid animal data found.")
                return
            master_df = pd.concat(all_dfs, ignore_index=True)

            # Clear tabs BEFORE plotting new data (entrypoint hardening)
            self._clear_tab_contents(self.mw.group_scatter_tab)
            self._clear_tab_contents(self.mw.group_avg_tab)
            self._clear_tab_contents(getattr(self.mw, 'grad_tab', None))

            # Helpers for normalization
            def wrap_pm12(x):
                return (x + 12.0) % 24.0 - 12.0

            def circmean_hours_from_hours(phases_hours):
                phases_hours = phases_hours[np.isfinite(phases_hours)]
                if len(phases_hours) == 0:
                    return np.nan
                phases_hours = np.mod(phases_hours, 24.0)
                r = (phases_hours / 24.0) * (2 * np.pi)
                m = circmean(r, low=0.0, high=2*np.pi)
                return (m / (2 * np.pi)) * 24.0

            # Determine mode EARLY
            mode = self.norm_mode_combo.currentText()
            use_animal_norm = (mode == "Normalize: Per-animal mean")

            # 1) Control Normalization (Conditional)
            # Only compute if needed (default mode) or if we want to support switching TO it later?
            # User req: "Control-mean normalization should only run when needed... If use_animal_norm is True... DO NOT require controls"
            # BUT for switching, we might want it if controls exist. 
            # Compromise: Try to compute it if controls exist, but don't error if they don't AND we are in animal mode.
            
            ctrl_phases = master_df[master_df['Group'] == 'Control']['Phase_CT'].dropna().values
            
            if len(ctrl_phases) > 0:
                rads = (ctrl_phases / 24.0) * (2*np.pi)
                ref_rad = circmean(rads, low=0.0, high=2*np.pi)
                ref_phase = (ref_rad / (2*np.pi)) * 24.0
                master_df['Rel_Phase_Control'] = wrap_pm12(master_df['Phase_CT'] - ref_phase)
                if not use_animal_norm:
                    self.mw.log_message(f"Using Control Mean Normalization: {ref_phase:.2f}h")
            else:
                # No controls
                if not use_animal_norm:
                    self.mw.log_message("Error: No control cells found for normalization.")
                    return
                # If using animal norm, we assume Rel_Phase_Control can be missing
            
            # 2) Per-Animal Normalization (Always compute for robustness or only if needed?)
            # User Requirement D implies computing it. 
            # It relies on each animal's own data, so always safe to compute if data exists.
            
            animal_means = master_df.groupby('Animal')['Phase_CT'].apply(
                lambda s: circmean_hours_from_hours(s.to_numpy(dtype=float, copy=False))
            )
            master_df['Animal_Mean_Phase'] = master_df['Animal'].map(animal_means)
            master_df['Rel_Phase_Animal'] = wrap_pm12(master_df['Phase_CT'] - master_df['Animal_Mean_Phase'])
            
            if use_animal_norm:
                self.mw.log_message("Using Per-Animal Normalization.")
                master_df['Rel_Phase'] = master_df['Rel_Phase_Animal']
            else:
                # We already checked for controls above
                if 'Rel_Phase_Control' in master_df.columns:
                    master_df['Rel_Phase'] = master_df['Rel_Phase_Control']
                else:
                    # Should have returned earlier
                    return
            
            # Cache COPY to prevent mutation of stored state during view generation
            self._last_master_df = master_df.copy()


            self._generate_continuous_maps(master_df)
            self._generate_gradient_analysis(master_df)
            self._generate_regional_stats(master_df)
            self.mw.update_export_buttons_state()

        except Exception as e:
            self.mw.log_message(f"Analysis Error: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())

    def _generate_continuous_maps(self, df):
        single_animal_tabs = [self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab]
        for tab in single_animal_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
        
        try:
            grid_res = int(self.group_grid_res_edit.text())
            do_smooth = self.group_smooth_check.isChecked()
            
            x_min, x_max = df['X'].min(), df['X'].max()
            y_min, y_max = df['Y'].min(), df['Y'].max()
            width, height = x_max - x_min, y_max - y_min
            
            if width >= height:
                n_bins_x = grid_res
                bin_size = width / n_bins_x
                n_bins_y = max(1, int(round(height / bin_size)))
            else:
                n_bins_y = grid_res
                bin_size = height / n_bins_y
                n_bins_x = max(1, int(round(width / bin_size)))

            start_x = x_min - bin_size; end_x = x_max + bin_size
            start_y = y_min - bin_size; end_y = y_max + bin_size
            
            grid_x_bins = np.arange(start_x, end_x, bin_size)
            grid_y_bins = np.arange(start_y, end_y, bin_size)
            calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
            calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)
            

            
            # Select column based on combo box; logic moved to generate_group_visualizations but
            # ensuring we use 'Rel_Phase' which is populated there.
            # The prompt requested: "When generating scatter_df, set 'Relative_Phase_Hours' to..."
            # Since 'Rel_Phase' in master_df is now context-sensitive (see below), we just map it.
            
            scatter_df = df.rename(columns={'Animal': 'Source_Animal', 'X': 'Warped_X', 'Y': 'Warped_Y', 'Rel_Phase': 'Relative_Phase_Hours'})
            fig_s, _ = add_mpl_to_tab(self.mw.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), scatter_df, grid_bins=(grid_x_bins, grid_y_bins))
            self.mw.visualization_widgets[self.mw.group_scatter_tab] = viewer_s
            
            scatter_df['Grid_X_Index'] = pd.cut(scatter_df['Warped_X'], bins=calc_x_bins, labels=False, include_lowest=True)
            scatter_df['Grid_Y_Index'] = pd.cut(scatter_df['Warped_Y'], bins=calc_y_bins, labels=False, include_lowest=True)
            
            def circmean_phase(series):
                rad = (series / 12.0) * np.pi 
                mean_rad = circmean(rad, low=-np.pi, high=np.pi)
                return (mean_rad / np.pi) * 12.0
            
            group_binned = scatter_df.groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
            
            fig_g, _ = add_mpl_to_tab(self.mw.group_avg_tab)
            viewer_g = GroupAverageMapViewer(fig_g, fig_g.add_subplot(111), group_binned, scatter_df, (n_bins_x, n_bins_y), do_smooth)
            self.mw.visualization_widgets[self.mw.group_avg_tab] = viewer_g
            
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.group_scatter_tab), True)
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.group_avg_tab), True)
            
        except Exception as e:
            self.mw.log_message(f"Grid Visualization Error: {e}")

    def _generate_regional_stats(self, df):
        self.final_zone_stats = []
        min_cells = self.min_region_cells_spin.value()
        min_animals = self.min_animals_spin.value()
        unique_animals = df['Animal'].unique()
        
        self.mw.log_message("Calculating per-zone statistics...")

        for zid, z_info in self.zones.items():
            polys = z_info['polygons']
            zone_record = {'id': zid, 'name': z_info['name'], 'data': []}
            
            for animal in unique_animals:
                subset = df[df['Animal'] == animal]
                if subset.empty: continue
                
                points = subset[['X', 'Y']].values
                mask_in_zone = np.zeros(len(points), dtype=bool)
                for poly in polys:
                    mask_in_zone |= poly.contains_points(points)
                
                valid_phases = subset.loc[mask_in_zone, 'Rel_Phase'].values
                
                if len(valid_phases) >= min_cells:
                    rads = (valid_phases / 24.0) * (2 * np.pi)
                    m_rad = circmean(rads, low=-np.pi, high=np.pi)
                    m_val = (m_rad / (2 * np.pi)) * 24.0
                    
                    # Compute Resultant Vector Length R
                    C = np.mean(np.cos(rads))
                    S = np.mean(np.sin(rads))
                    R = np.sqrt(C**2 + S**2)
                    
                    zone_record['data'].append({
                        'animal': animal,
                        'group': subset['Group'].iloc[0],
                        'mean': m_val,
                        'mean_h': m_val,          # Viewers expect this
                        'mean_rad': m_rad,
                        'mean_h_mod24': m_val % 24.0,
                        'R': R,
                        'n_cells': len(valid_phases),
                        'raw_phases': valid_phases.tolist()
                    })

            ctrl_means = [d['mean'] for d in zone_record['data'] if d['group'] == 'Control']
            exp_means = [d['mean'] for d in zone_record['data'] if d['group'] == 'Experiment']
            
            zone_record['n_ctrl'] = len(ctrl_means)
            zone_record['n_exp'] = len(exp_means)
            
            if len(ctrl_means) >= min_animals and len(exp_means) >= min_animals:
                def group_circ(vals):
                    r = (np.array(vals) / 24.0) * (2*np.pi)
                    return (circmean(r) / (2*np.pi)) * 24.0

                zone_record['mean_ctrl'] = group_circ(ctrl_means)
                zone_record['mean_exp'] = group_circ(exp_means)
                
                f_val = watson_williams_f(ctrl_means, exp_means)
                df2 = len(ctrl_means) + len(exp_means) - 2
                p_val = 1.0 - f_dist.cdf(f_val, 1, df2)
                
                zone_record['p_value'] = p_val
                
                c_mean = zone_record['mean_ctrl']
                e_mean = zone_record['mean_exp']
                diff = (e_mean - c_mean + 12.0) % 24.0 - 12.0
                zone_record['diff_mean'] = diff
            else:
                zone_record['p_value'] = 1.0
                zone_record['diff_mean'] = 0.0

            self.final_zone_stats.append(zone_record)

        if not hasattr(self.mw, 'region_tab'):
            self.mw.region_tab = QtWidgets.QWidget()
            self.mw.vis_tabs.addTab(self.mw.region_tab, "Region Stats")
        
        # Handle layout safely to prevent console warnings
        layout = self.mw.region_tab.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.mw.region_tab)
        else:
            clear_layout(layout)
        
        viewer = RegionResultViewer(self.final_zone_stats, self.zones_polygons_map)
        self.mw.visualization_widgets[self.mw.region_tab] = viewer
        
        layout.addWidget(viewer)
        
        self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.region_tab), True)
        self.mw.vis_tabs.setCurrentWidget(self.mw.region_tab)
        self.mw.log_message("Region Analysis Complete.")

    def _get_scn_boundary_y_extents(self, atlas_data):
        """
        Determines the Y-extents of the SCN boundary for DV normalization.
        Rule: Use the 'Include' mode polygon with the largest area to define the anatomical boundary.
        Filters out degenerate (small area or invalid) polygons.
        Returns: (y_min, y_max) or None if no valid boundary found.
        """
        candidate_polys = []
        considered_count = 0
        rejected_count = 0
        MIN_AREA_PX2 = 500.0  # Threshold to ignore noise/strokes
        
        for r in atlas_data:
            if r.get('mode') == 'Include' and 'path_vertices' in r:
                considered_count += 1
                try:
                    pts = np.array(r['path_vertices'])
                    
                    # Validation A: Shape Structure
                    if pts.ndim != 2 or pts.shape[1] != 2:
                        rejected_count += 1
                        continue

                    # Validation B: Finiteness and Length
                    if len(pts) < 3 or not np.all(np.isfinite(pts)):
                         rejected_count += 1
                         continue
                         
                    # Shoelace formula for area
                    x, y = pts[:,0], pts[:,1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    
                    # Validation C: Area Threshold
                    if area < MIN_AREA_PX2:
                        rejected_count += 1
                        continue
                        
                    candidate_polys.append((area, pts))
                except Exception:
                    rejected_count += 1
                    continue
        
        if not candidate_polys:
            if rejected_count > 0:
                self.mw.log_message(f"SCN Boundary Search: Considered {considered_count}, Rejected {rejected_count} (Degenerate/Small). None valid found.")
            return None

        # Select largest
        candidate_polys.sort(key=lambda x: x[0], reverse=True)
        best_area, best_pts = candidate_polys[0]
        accepted_count = len(candidate_polys)
        
        self.mw.log_message(f"SCN Boundary Search: Considered {considered_count}, Accepted {accepted_count}, Rejected {rejected_count}. Selected Largest (Area={best_area:.0f} px²).")
        
        y_min = np.min(best_pts[:, 1])
        y_max = np.max(best_pts[:, 1])
        return y_min, y_max

    def _generate_gradient_analysis(self, df):
        atlas_path = self.state.atlas_roi_path
        if not atlas_path: return
        try:
            with open(atlas_path, 'r') as f: atlas_data = json.load(f)
            
            # Determine Coordinate Mode
            mode = self.grad_mode_combo.currentText()
            self.mw.log_message(f"Running Gradient Analysis (Mode: {mode})...")
            
            s_func = None
            
            # Helper to setup DV function
            def setup_dv_func():
                extents = self._get_scn_boundary_y_extents(atlas_data)
                if not extents:
                    self.mw.log_message("Error: No valid Include boundary polygon found for DV normalization.")
                    return None, None
                
                y_min, y_max = extents
                dy = y_max - y_min
                if dy == 0: dy = 1
                self.mw.log_message(f"DV Mode: Range Y=[{y_min:.1f}, {y_max:.1f}]. s=0@Min, s=1@Max.")
                
                def f(points):
                    s = (points[:, 1] - y_min) / dy
                    return np.clip(s, 0.0, 1.0)
                return f, (y_min, y_max)

            # Helpers
            def get_polyline_dists(points, poly):
                """Compute min distance from each point to the polyline."""
                dists = np.full(len(points), np.inf)
                if len(poly) < 2: return dists
                
                for i in range(len(poly) - 1):
                    A = poly[i]
                    B = poly[i+1]
                    AB = B - A
                    sq_len = np.dot(AB, AB)
                    if sq_len == 0:
                        d = np.linalg.norm(points - A, axis=1)
                        dists = np.minimum(dists, d)
                        continue
                    
                    AP = points - A
                    t = np.sum(AP * AB, axis=1) / sq_len
                    t = np.clip(t, 0.0, 1.0)
                    C = A + np.outer(t, AB)
                    d = np.linalg.norm(points - C, axis=1)
                    dists = np.minimum(dists, d)
                return dists

            def compute_gradient_stats(animal, points, phases, s_func, axis_label, group_name, mode, min_cells_bin, period_for_slope):
                """Computes binned stats and slope for a subset of points/phases. Pure function."""
                s_vals = s_func(points)
                s_vals = np.clip(s_vals, 0.0, 1.0) # Ensure s is valid
                
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                binned_phases = []
                binned_counts = []
                valid_centers_for_slope = []
                valid_phases_for_slope = [] # in hours

                for k in range(len(bins)-1):
                    low, high = bins[k], bins[k+1]
                    if k == len(bins)-2:
                        mask = (s_vals >= low) & (s_vals <= high)
                    else:
                        mask = (s_vals >= low) & (s_vals < high)
                    
                    n_cells = np.sum(mask)
                    binned_counts.append(n_cells)
                    
                    if n_cells >= min_cells_bin:
                        p_bin = phases[mask]
                        rads = (p_bin / period_for_slope) * 2 * np.pi
                        m_rad = circmean(rads, low=-np.pi, high=np.pi)
                        m_h = (m_rad / (2 * np.pi)) * period_for_slope
                        binned_phases.append(m_h)
                        
                        valid_centers_for_slope.append(bin_centers[k])
                        valid_phases_for_slope.append(m_h)
                    else:
                        binned_phases.append(np.nan)
                
                slope_h = np.nan
                slope_rad = np.nan
                r_val = np.nan
                
                if len(valid_centers_for_slope) >= 3:
                     # Unwrap for slope
                     v_phases = np.array(valid_phases_for_slope)
                     rads = (v_phases / period_for_slope) * 2 * np.pi
                     unwrapped_rads = np.unwrap(rads)
                     x_s = np.array(valid_centers_for_slope)
                     
                     res = linregress(x_s, unwrapped_rads)
                     slope_rad = res.slope
                     r_val = res.rvalue
                     slope_h = slope_rad * (period_for_slope / (2 * np.pi))
                
                return {
                    'animal': animal,
                    'group': group_name,
                    'axis_label': axis_label,
                    's': bin_centers,
                    'phases': np.array(binned_phases),
                    'counts': np.array(binned_counts),
                    'slope_hours': slope_h,
                    'slope_rad': slope_rad,
                    'r_value': r_val,
                    'mode': mode,
                    'period': period_for_slope
                }

            axes_config = [] # List of filtering/config dicts

            if mode == "DV (Y-normalized)":
                self.collapse_gradient_cb.setChecked(False)
                self.collapse_gradient_cb.setEnabled(False)
                
                func, _ = setup_dv_func()
                if not func: return
                axes_config.append({
                    'id': 'DV',
                    'func': func,
                    'mode': 'DV'
                })
                
            else:
                # Phase Axis Mode
                axis_rois = [r for r in atlas_data if r.get('mode') == 'Phase Axis']
                n_phase_axis_rois_total = len(axis_rois)
                
                # Robust parsing
                parsed_axes = []
                for roi in axis_rois:
                    verts = roi.get('path_vertices', None)
                    if verts is None: continue
                    
                    try:
                        poly = np.asarray(verts, dtype=float)
                    except (ValueError, TypeError):
                        continue
                        
                    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 2:
                        continue
                        
                    # Truncate to first 2 columns (X, Y) if needed
                    if poly.shape[1] > 2:
                        poly = poly[:, :2]
                    
                    if not np.all(np.isfinite(poly)):
                        continue
                        
                    centroid_x = np.mean(poly[:, 0])
                    parsed_axes.append({'poly': poly, 'cx': centroid_x})

                n_valid_axes = len(parsed_axes)
                n_invalid_axes = n_phase_axis_rois_total - n_valid_axes
                
                if n_invalid_axes > 0:
                    self.mw.log_message(f"Gradient: ignored {n_invalid_axes}/{n_phase_axis_rois_total} Phase Axis ROIs (invalid path_vertices).")

                if not parsed_axes:
                     self.mw.log_message("Warning: No valid Phase Axis found. Switching to DV mode.")
                     self.collapse_gradient_cb.setChecked(False)
                     self.collapse_gradient_cb.setEnabled(False)
                     
                     func, _ = setup_dv_func()
                     if not func: return
                     axes_config.append({
                         'id': 'DV',
                         'func': func,
                         'mode': 'DV'
                     })
                     mode = "DV (Y-normalized)"
                
                elif len(parsed_axes) == 1:
                     # Single Axis
                     self.collapse_gradient_cb.setChecked(False)
                     self.collapse_gradient_cb.setEnabled(False)
                     
                     poly = parsed_axes[0]['poly']
                     axes_config.append({
                         'id': 'Primary',
                         'func': lambda pts, p=poly: project_points_to_polyline(pts, p),
                         'mode': 'Single',
                         'poly': poly
                     })

                elif len(parsed_axes) == 2:
                    # Bilateral
                    self.collapse_gradient_cb.setEnabled(True)
                    
                    # Sort axes by centroid X
                    parsed_axes.sort(key=lambda a: a['cx'])
                    left_axis = parsed_axes[0]
                    right_axis = parsed_axes[1]
                    
                    self.mw.log_message(f"Bilateral Axis Mode: Left Axis (cx={left_axis['cx']:.1f}), Right Axis (cx={right_axis['cx']:.1f})")
                    
                    # Pre-compute s-flips (Dorsal-to-Ventral normalization)
                    # s=0 should be Dorsal (smaller Y). If start.y > end.y, start is Ventral -> flip.
                    # Use tolerance to avoid unstable flipping on horizontal lines.
                    flip_eps = 1e-6
                    l_poly = left_axis['poly']
                    l_flip = (l_poly[0, 1] - l_poly[-1, 1]) > flip_eps
                    
                    r_poly = right_axis['poly']
                    r_flip = (r_poly[0, 1] - r_poly[-1, 1]) > flip_eps
                    
                    if self.collapse_gradient_cb.isChecked():
                        self.mw.log_message(f"Collapsed Mode Active: S-Flip Flags -> Left: {l_flip}, Right: {r_flip} (Criteria: Start.Y > End.Y)")

                    axes_config.append({
                        'mode': 'Bilateral',
                        'left_poly': l_poly,
                        'right_poly': r_poly,
                        'left_func': lambda pts, p=l_poly: project_points_to_polyline(pts, p),
                        'right_func': lambda pts, p=r_poly: project_points_to_polyline(pts, p),
                        'left_flip': l_flip,
                        'right_flip': r_flip
                    })

                else:
                     # > 2 Axes
                     msg = f"Found {n_valid_axes} valid Phase Axes (of {n_phase_axis_rois_total} Phase Axis ROIs).\nPlease use exactly 1 (unilateral) or 2 (bilateral) Phase Axes."
                     QtWidgets.QMessageBox.critical(self, "Gradient Analysis Error", msg)
                     self.mw.log_message(msg)
                     
                     self.collapse_gradient_cb.setChecked(False)
                     self.collapse_gradient_cb.setEnabled(False)
                     return

            gradient_data = []
            animals = df['Animal'].unique()
            min_cells_bin = 5
            period_for_slope = 24.0
            
            ambiguous_total = 0
            collapse_enabled = self.collapse_gradient_cb.isChecked()
            
            for animal in animals:
                subset = df[df['Animal'] == animal]
                points_all = subset[['X', 'Y']].values
                phases_all = subset['Rel_Phase'].values
                group_name = subset['Group'].iloc[0] 
                
                # Execute Config
                for cfg in axes_config:
                    if cfg['mode'] == 'Bilateral':
                        # Robust distance-based assignment
                        d_left = get_polyline_dists(points_all, cfg['left_poly'])
                        d_right = get_polyline_dists(points_all, cfg['right_poly'])
                        
                        eps = 1e-9
                        # Assign to closer axis
                        left_mask = (d_left + eps < d_right)
                        right_mask = (d_right + eps < d_left)
                        ambiguous_mask = ~(left_mask | right_mask)
                        
                        n_amb = int(np.sum(ambiguous_mask))
                        if n_amb > 0:
                            ambiguous_total += n_amb
                            self.mw.log_message(f"{animal}: dropped {n_amb} ambiguous points (equal distance to both axes).")
                        
                        # Compute Left
                        # Note: store points/phases/s for potential collapse
                        pts_left = points_all[left_mask]
                        phs_left = phases_all[left_mask]
                        
                        if np.sum(left_mask) > 0:
                            res = compute_gradient_stats(
                                animal, pts_left, phs_left,
                                cfg['left_func'], 'Left', group_name, mode, min_cells_bin, period_for_slope
                            )
                            gradient_data.append(res)
                            
                        # Compute Right
                        pts_right = points_all[right_mask]
                        phs_right = phases_all[right_mask]

                        if np.sum(right_mask) > 0:
                            res = compute_gradient_stats(
                                animal, pts_right, phs_right,
                                cfg['right_func'], 'Right', group_name, mode, min_cells_bin, period_for_slope
                            )
                            gradient_data.append(res)
                            
                        # Compute Collapsed (Optional)
                        if collapse_enabled and (len(pts_left) > 0 or len(pts_right) > 0):
                            # 1. Compute Raw S
                            s_left = cfg['left_func'](pts_left) if len(pts_left) > 0 else np.array([])
                            s_right = cfg['right_func'](pts_right) if len(pts_right) > 0 else np.array([])
                            s_left = np.clip(s_left, 0.0, 1.0)
                            s_right = np.clip(s_right, 0.0, 1.0)
                            
                            # 2. Apply S-Flips (pre-calculated)
                            if cfg['left_flip']:
                                s_left = 1.0 - s_left
                            if cfg['right_flip']:
                                s_right = 1.0 - s_right
                            
                            # 3. Pool
                            s_pooled = np.concatenate([s_left, s_right])
                            pts_pooled = np.concatenate([pts_left, pts_right]) # Just for len/shape, func ignored
                            phs_pooled = np.concatenate([phs_left, phs_right])
                            
                            # Defensive Check
                            if len(s_pooled) != len(phs_pooled):
                                raise ValueError("Collapsed gradient internal error: len(s_pooled) != len(phs_pooled)")
                            
                            # 4. Compute Stats from Pooled Data
                            # We pass a dummy identity func for s because we already computed s_pooled
                            res_coll = compute_gradient_stats(
                                animal, pts_pooled, phs_pooled,
                                lambda p: s_pooled, 'Collapsed', group_name, mode, min_cells_bin, period_for_slope
                            )
                            gradient_data.append(res_coll)
                            
                    else:
                        # Single or DV
                        # Use all points
                        res = compute_gradient_stats(
                            animal, points_all, phases_all, cfg['func'], cfg['id'], group_name, mode, min_cells_bin, period_for_slope
                        )
                        gradient_data.append(res)
            
            if ambiguous_total > 0:
                self.mw.log_message(f"Gradient analysis: dropped {ambiguous_total} ambiguous points total (equal distance to both axes).")

                
            if not hasattr(self.mw, 'grad_tab'):
                self.mw.grad_tab = QtWidgets.QWidget()
                self.mw.vis_tabs.addTab(self.mw.grad_tab, "Gradient")
            
            fig, _ = add_mpl_to_tab(self.mw.grad_tab)
            viewer_g = PhaseGradientViewer(fig, fig.add_subplot(111), gradient_data)
            self.mw.visualization_widgets[self.mw.grad_tab] = viewer_g
            
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.grad_tab), True)

        except Exception as e:
            self.mw.log_message(f"Gradient Analysis Warning: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())

    def export_current_data(self):
        current_tab = self.mw.vis_tabs.currentWidget()
        if current_tab == getattr(self.mw, 'region_tab', None):
             viewer = current_tab.findChild(RegionResultViewer)
             if viewer:
                 df, fname = viewer.get_export_data()
                 path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
                 if path:
                     df.to_csv(path, index=False)
                     self.mw.log_message(f"Saved to {path}")
                 return

        viewer = self.mw.visualization_widgets.get(current_tab)
        if viewer and hasattr(viewer, 'get_export_data'):
             df, fname = viewer.get_export_data()
             path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
             if path:
                 df.to_csv(path, index=False)
                 self.mw.log_message(f"Saved to {path}")