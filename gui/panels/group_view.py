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
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer, PhaseGradientViewer, RegionResultViewer, ClusterResultViewerWidget
from gui.statistics import watson_williams_f
from gui.dialogs.roi_drawer import ROIDrawerDialog
from gui.dialogs.cluster_config_dialog import ClusterConfigDialog
import analysis.cluster_stats as cluster_stats
from gui.theme import get_icon

def load_and_join_rhythm_and_coords(rhythm_csv_path: str, warped_ids_csv_path: str) -> "Tuple[pd.DataFrame, dict]":
    # ... (function body unchanged) ...
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
    ROLE_PATH = QtCore.Qt.UserRole
    ROLE_GROUP = QtCore.Qt.UserRole + 1

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        self.zones_polygons_map = {} 
        self.init_ui()
        self.connect_signals()
        
        # Defensive check
        for w_name in ['group_grid_res_edit', 'group_smooth_check', 'btn_cluster_stats']:
            if not hasattr(self, w_name):
                 raise RuntimeError(f"UI Initialization Failed: Missing {w_name}")

    def init_ui(self):
        def safe_icon(k):
            try: return get_icon(k)
            except Exception: return QtGui.QIcon()

        layout = QtWidgets.QHBoxLayout(self)
        
        # Left Control Panel
        box = QtWidgets.QWidget() 
        b = QtWidgets.QVBoxLayout(box)
        
        # 1. Group Files
        group_box = QtWidgets.QGroupBox("Group Input Files")
        gb_layout = QtWidgets.QVBoxLayout(group_box)
        
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        gb_layout.addWidget(self.file_list)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton(safe_icon('fa5s.plus'), "Add Files")
        self.btn_remove_group = QtWidgets.QPushButton(safe_icon('fa5s.minus'), "Remove")
        btn_row.addWidget(self.btn_add_group)
        btn_row.addWidget(self.btn_remove_group)
        gb_layout.addLayout(btn_row)
        
        assign_row = QtWidgets.QHBoxLayout()
        self.btn_assign_control = QtWidgets.QPushButton("Set Control")
        self.btn_assign_exp = QtWidgets.QPushButton("Set Experiment")
        assign_row.addWidget(self.btn_assign_control)
        assign_row.addWidget(self.btn_assign_exp)
        gb_layout.addLayout(assign_row)
        
        b.addWidget(group_box)
        
        # 2. Parameters
        param_box = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_box)
        
        self.norm_mode_combo = QtWidgets.QComboBox()
        self.norm_mode_combo.addItems(["Normalize: Global Mean", "Normalize: Per-animal mean"])
        param_layout.addRow("Normalization:", self.norm_mode_combo)
        
        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        param_layout.addRow("Grid Res (bins):", self.group_grid_res_edit)
        
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth Maps (Gaussian)")
        param_layout.addRow(self.group_smooth_check)
        
        self.grad_mode_combo = QtWidgets.QComboBox()
        self.grad_mode_combo.addItems(["DV (Y-normalized)", "Phase Axis (polyline)"])
        param_layout.addRow("Gradient Coordinate:", self.grad_mode_combo)
        
        self.collapse_gradient_cb = QtWidgets.QCheckBox("Collapse bilateral axes (Left+Right)")
        self.collapse_gradient_cb.setToolTip("If checked, bilateral gradients are also pooled into a single 'Collapsed' gradient (requires 2 Phase Axes).")
        param_layout.addRow(self.collapse_gradient_cb)
        
        b.addWidget(param_box)        
        
        # 3. Regional Analysis
        region_box = QtWidgets.QGroupBox("Regional Analysis Setup")
        region_layout = QtWidgets.QFormLayout(region_box)
        
        atlas_box = QtWidgets.QWidget()
        atlas_layout = QtWidgets.QHBoxLayout(atlas_box)
        atlas_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_load_atlas = QtWidgets.QPushButton(safe_icon('fa5s.map'), "Load Atlas Template...")
        self.atlas_path_label = QtWidgets.QLineEdit()
        self.atlas_path_label.setPlaceholderText("No Atlas Loaded")
        self.atlas_path_label.setReadOnly(True)
        atlas_layout.addWidget(self.btn_load_atlas)
        atlas_layout.addWidget(self.atlas_path_label)
        region_layout.addRow("Atlas Template:", atlas_box)
        
        self.btn_define_regions = QtWidgets.QPushButton(safe_icon('fa5s.draw-polygon'), "Define Analysis Regions...")
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
        
        # 4. Actions
        self.btn_view_group = QtWidgets.QPushButton(safe_icon('fa5s.chart-pie'), "Generate Group Visualizations")
        Tooltip.install(self.btn_view_group, "Runs Continuous Grid analysis, Gradient analysis (if axes present), and Regional Statistical analysis.")
        self.btn_view_group.setEnabled(False)
        b.addWidget(self.btn_view_group)
        
        self.btn_cluster_stats = QtWidgets.QPushButton(safe_icon('fa5s.microscope'), "Run Cluster Statistics...")
        self.btn_cluster_stats.setEnabled(False) 
        self.btn_cluster_stats.setToolTip("Run bin-level permutation cluster analysis (Control vs Experiment).")
        b.addWidget(self.btn_cluster_stats)
        
        b.addStretch()
        
        layout.addWidget(box)
        layout.addStretch(1)

    def _compute_grid_assignment(self, df, grid_res_str=None, smooth_check=False):
        """
        Shared helper to compute grid bins and assign cells to them.
        Returns:
            df (pd.DataFrame): Copy of input with Grid_X_Index, Grid_Y_Index columns.
            info (dict): dict containing grid geometry.
        """
        df = df.copy()
        try:
            val = grid_res_str if grid_res_str is not None else self.group_grid_res_edit.text()
            grid_res = int(val)
        except:
            grid_res = 50
            
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
        
        # Proper edges for histogramming/cutting
        # Note: pd.cut uses intervals. We define edges to align with bin size.
        calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
        calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)
        
        # Assign
        df['Grid_X_Index'] = pd.cut(df['X'], bins=calc_x_bins, labels=False, include_lowest=True)
        df['Grid_Y_Index'] = pd.cut(df['Y'], bins=calc_y_bins, labels=False, include_lowest=True)
        
        info = {
            'n_bins_x': n_bins_x, 'n_bins_y': n_bins_y,
            'grid_shape': (n_bins_y, n_bins_x), # Consistently (H, W)
            'bin_size': bin_size,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'calc_x_bins': calc_x_bins, 'calc_y_bins': calc_y_bins,
            'do_smooth': smooth_check,
             # Kept for viewers if needed, but not used for logic
            'grid_x_bins': np.arange(start_x, end_x, bin_size), 
            'grid_y_bins': np.arange(start_y, end_y, bin_size),
        }
        return df, info

    def run_cluster_analysis(self):
        if not hasattr(self, "_last_master_df") or self._last_master_df is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please generate group visualizations first to prepare data.")
            return

        df = self._last_master_df.copy()
        
        # D3. Group Consistency
        group_counts = df.groupby('Animal')['Group'].nunique()
        inconsistent = group_counts[group_counts > 1]
        if not inconsistent.empty:
            bad_animals = inconsistent.index.tolist()
            QtWidgets.QMessageBox.critical(self, "Data Error", f"Found animals with multiple group labels: {bad_animals}. Fix input CSVs.")
            return

        # D4. Units Check
        valid_phases = df['Rel_Phase'].dropna()
        if not valid_phases.empty:
             abs_max = valid_phases.abs().max()
             if abs_max > 48.0:
                  QtWidgets.QMessageBox.critical(self, "Units Error", f"Rel_Phase values > 48 hours found (max {abs_max:.1f}). Check normalization.")
                  return
        
        # Check Groups
        unique_groups = df['Group'].unique()
        if 'Control' not in unique_groups or 'Experiment' not in unique_groups:
             QtWidgets.QMessageBox.warning(self, "Missing Groups", "Both 'Control' and 'Experiment' groups are required.")
             return
             
        # B. Use Shared Grid Logic
        try:
             df_grid, info = self._compute_grid_assignment(df)
        except Exception as e:
             QtWidgets.QMessageBox.critical(self, "Grid Error", f"Failed to compute grid: {e}")
             return
             
        n_bins_x, n_bins_y = info['n_bins_x'], info['n_bins_y']
        
        # 3. Create Lobe Mask
        lobe_mask = np.zeros((n_bins_y, n_bins_x), dtype=int)
        
        # Center coordinates - using same calc_x_bins as cut
        # Centers are (n_bins,) array
        cx_centers = 0.5 * (info['calc_x_bins'][:-1] + info['calc_x_bins'][1:])
        cy_centers = 0.5 * (info['calc_y_bins'][:-1] + info['calc_y_bins'][1:])
        
        # Meshgrid: gx corresponds to cols (x), gy to rows (y)
        gx, gy = np.meshgrid(cx_centers, cy_centers) 
        # Ravel order: varies by impl, but we need consistency with reshape((H,W))
        # By default meshgrid 'xy' indexing: gx is (H,W), gy is (H,W)
        centers = np.column_stack((gx.ravel(), gy.ravel()))
        
        if not hasattr(self, 'zones') or not self.zones:
             QtWidgets.QMessageBox.warning(self, "No Regions", "No regions/lobes defined. Cannot run cluster analysis.")
             return
             
        # Rasterize zones
        found_any_lobe = False
        for zid, zone_info in self.zones.items():
            lobe_id = zone_info.get('lobe', 0)
            if lobe_id == 0: continue
            
            polys = zone_info.get('polygons', [])
            if not polys: continue
            
            mask_in_zone = np.zeros(len(centers), dtype=bool)
            for poly in polys:
                # poly.contains_points expects (x, y)
                if len(poly.vertices) > 2:
                    mask_in_zone |= poly.contains_points(centers)
            
            mask_2d = mask_in_zone.reshape((n_bins_y, n_bins_x))
            lobe_mask[mask_2d] = int(lobe_id)
            found_any_lobe = True
            
        if not found_any_lobe:
            QtWidgets.QMessageBox.warning(self, "No Lobes", "No zones with Lobe ID > 0 found. Please define Left (1) / Right (2) lobes in Region setup.")
            return

        # 4. Aggregate Phases & Prepare Stats
        grouped_phases_grid = {}
        bin_stats = [] # List of (n_c, n_e) for dialog
        
        df_valid = df_grid.dropna(subset=['Grid_X_Index', 'Grid_Y_Index', 'Rel_Phase'])
        
        # Iterate all bins in grid
        for (by, bx), bin_df in df_valid.groupby(['Grid_Y_Index', 'Grid_X_Index']): # Group by Row then Col
            bx, by = int(bx), int(by)
            if not (0 <= by < n_bins_y and 0 <= bx < n_bins_x): continue
            
            # Check Lobe: Only gather stats for lobe bins (Candidate Bins)
            if lobe_mask[by, bx] == 0: continue
            
            c_dict = {}
            e_dict = {}
            
            for animal_name, anim_df in bin_df.groupby('Animal'):
                phases = anim_df['Rel_Phase'].values
                if len(phases) == 0: continue
                # Circular Mean
                rads = (phases / 24.0) * (2*np.pi)
                m_rad = circmean(rads, low=-np.pi, high=np.pi)
                
                group = anim_df['Group'].iloc[0]
                if group == 'Control':
                    c_dict[animal_name] = m_rad
                elif group == 'Experiment':
                    e_dict[animal_name] = m_rad
            
            n_c = len(c_dict)
            n_e = len(e_dict)
            
            # Record stat for dialog: Include if EITHER group is present (Candidate)
            # The dialog will filter strictly by MinN
            if n_c > 0 or n_e > 0:
                bin_stats.append((n_c, n_e))
            
            # Store in grid if candidate (has data)
            # Logic: Cluster stats will filter strict Min N, but we pass data if it exists.
            if n_c > 0 or n_e > 0:
                 grouped_phases_grid[(by, bx)] = {'Control': c_dict, 'Experiment': e_dict} 
        
        # 5. Dialog with Dynamic updates
        dlg = ClusterConfigDialog(self, (n_bins_y, n_bins_x), bin_stats)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        # 6. Run Worker
        self.mw.log_message(f"Starting Cluster Analysis: MinN={dlg.min_n}, Perms={dlg.n_perm}, Alpha={dlg.alpha}, Seed={dlg.seed}")
        self.btn_cluster_stats.setEnabled(False)
        self.mw.set_status("Running Cluster Analysis...")
        
        # Ensure session dir is valid
        s_dir = getattr(self.mw, 'session_dir', None)
        if not s_dir or not os.path.isdir(s_dir):
            s_dir = os.path.join(os.getcwd(), 'analysis_results')
            os.makedirs(s_dir, exist_ok=True)
            self.mw.log_message(f"Warning: Session dir not found. Saving to {s_dir}")
        
        self.cluster_worker = ClusterWorker(
            grouped_phases_grid, (n_bins_y, n_bins_x), lobe_mask,
            dlg.min_n, dlg.n_perm, dlg.seed, dlg.alpha, dlg.save_plot,
            s_dir
        )
        self.cluster_worker.finished.connect(self.on_cluster_finished)
        self.cluster_worker.error.connect(self.on_cluster_error)
        self.cluster_worker.start()
        
    def on_cluster_finished(self, results):
        self.btn_cluster_stats.setEnabled(True)
        self.mw.set_status("Cluster Analysis Complete.")
        self.mw.log_message(f"Cluster Analysis Done. T0={results.get('T0', 0):.4f}, Clusters={len(results.get('clusters', []))}")
        
        # Layout management: Reuse or Create Tab
        if not hasattr(self.mw, 'cluster_tab'):
             self.mw.cluster_tab = QtWidgets.QWidget()
             self.mw.vis_tabs.addTab(self.mw.cluster_tab, "Cluster Stats")
        
        lay = self.mw.cluster_tab.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(self.mw.cluster_tab)
            self.mw.cluster_tab.setLayout(lay)
        else:
            clear_layout(lay)
            
        viewer = ClusterResultViewerWidget(results, self.mw.cluster_tab)
        lay.addWidget(viewer)
        
        self.mw.visualization_widgets[self.mw.cluster_tab] = viewer
        self.mw.vis_tabs.setCurrentWidget(self.mw.cluster_tab)
        self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.cluster_tab), True)

    def on_cluster_error(self, err_msg):
        self.btn_cluster_stats.setEnabled(True)
        self.mw.set_status("Cluster Analysis Failed.")
        self.mw.log_message(f"Cluster Analysis Error: {err_msg}")
        QtWidgets.QMessageBox.critical(self, "Analysis Failed", err_msg)


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
        self.btn_cluster_stats.clicked.connect(self.run_cluster_analysis)
    
    
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
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select one or more files to assign.")
            return
        for item in selected_items:
            path = item.data(self.ROLE_PATH)
            if not path: continue
            
            item.setData(self.ROLE_GROUP, group_name)
            bn = os.path.basename(path)
            item.setText(f"[{group_name}] {bn}")
            color = QtGui.QColor('blue') if group_name == "Control" else QtGui.QColor('red')
            item.setForeground(QtGui.QBrush(color))
        
        self._sync_state_group_paths_from_ui()
        self._update_group_view_button()
    
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
        
        # Get existing paths to prevent duplicates
        existing_paths = set()
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            p = item.data(self.ROLE_PATH)
            if p: existing_paths.add(p)
        
        for f in files:
            if f not in existing_paths:
                item = QtWidgets.QListWidgetItem()
                bn = os.path.basename(f)
                item.setText(bn)
                item.setData(self.ROLE_PATH, f)
                item.setData(self.ROLE_GROUP, None)
                item.setToolTip(f)
                self.file_list.addItem(item)
                existing_paths.add(f)
        
        self._sync_state_group_paths_from_ui()
        self._update_group_view_button()

    def remove_group_file(self):
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            self.file_list.takeItem(self.file_list.row(item))
            
        self._sync_state_group_paths_from_ui()
        self._update_group_view_button()

    def _sync_state_group_paths_from_ui(self):
        self.state.group_data_paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            p = item.data(self.ROLE_PATH)
            if p:
                self.state.group_data_paths.append(p)

    def _get_assigned_group_entries(self):
        entries = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            path = item.data(self.ROLE_PATH)
            grp = item.data(self.ROLE_GROUP)
            if not path or not grp: continue
            if grp not in ("Control", "Experiment"): continue
            entries.append((path, grp))
        return entries

    def _update_group_view_button(self):
        entries = self._get_assigned_group_entries()
        has_control = any(g == 'Control' for _, g in entries)
        has_exp = any(g == 'Experiment' for _, g in entries)
        can_run = has_control and has_exp
        
        self.btn_view_group.setEnabled(can_run)
        self.btn_cluster_stats.setEnabled(can_run)

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
                    self.zones[zid] = {
                        'name': r.get('name', f"Zone {zid}"), 
                        'polygons': [],
                        'lobe': r.get('lobe', 0) # Capture lobe if present
                    }
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
            # Iterate assigned entries from UI helper (Source of Truth)
            for roi_path, group in self._get_assigned_group_entries():
                base = os.path.basename(roi_path).replace('_roi_warped_with_ids.csv', '')
                
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

    def _compute_grid_assignment(self, df, grid_res_str=None, smooth_check=False):
        """
        Shared helper to compute grid bins and assign cells to them.
        Returns:
            df (pd.DataFrame): Copy of input with Grid_X_Index, Grid_Y_Index columns.
            info (dict): dict containing grid geometry (n_bins_x, n_bins_y, bin_size, etc.)
        """
        df = df.copy()
        try:
            val = grid_res_str if grid_res_str is not None else self.group_grid_res_edit.text()
            grid_res = int(val)
        except:
            grid_res = 50
            
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
        
        # Proper edges for histogramming/cutting
        calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
        calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)
        
        # Assign
        df['Grid_X_Index'] = pd.cut(df['X'], bins=calc_x_bins, labels=False, include_lowest=True)
        df['Grid_Y_Index'] = pd.cut(df['Y'], bins=calc_y_bins, labels=False, include_lowest=True)
        
        info = {
            'n_bins_x': n_bins_x, 'n_bins_y': n_bins_y,
            'bin_size': bin_size,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'calc_x_bins': calc_x_bins, 'calc_y_bins': calc_y_bins,
            'grid_x_bins': calc_x_bins, # Use exact edges for viewer consistency
            'grid_y_bins': calc_y_bins,
            'do_smooth': smooth_check
        }
        return df, info


    def _generate_continuous_maps(self, df):
        single_animal_tabs = [self.mw.heatmap_tab, self.mw.com_tab, self.mw.traj_tab, self.mw.phase_tab, self.mw.interp_tab]
        for tab in single_animal_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
        
        try:
            # Use shared helper
            df_grid, info = self._compute_grid_assignment(df, self.group_grid_res_edit.text(), self.group_smooth_check.isChecked())
            
            # Map Rel_Phase for scatter viewer
            scatter_df = df_grid.rename(columns={'Animal': 'Source_Animal', 'X': 'Warped_X', 'Y': 'Warped_Y', 'Rel_Phase': 'Relative_Phase_Hours'})
            
            fig_s, _ = add_mpl_to_tab(self.mw.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), scatter_df, grid_bins=(info['grid_x_bins'], info['grid_y_bins']))
            self.mw.visualization_widgets[self.mw.group_scatter_tab] = viewer_s
            
            # Group Average Map
            def circmean_phase(series):
                rad = (series / 12.0) * np.pi 
                mean_rad = circmean(rad, low=-np.pi, high=np.pi)
                return (mean_rad / np.pi) * 12.0
            
            group_binned = scatter_df.groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
            
            fig_g, _ = add_mpl_to_tab(self.mw.group_avg_tab)
            viewer_g = GroupAverageMapViewer(fig_g, fig_g.add_subplot(111), group_binned, scatter_df, (info['n_bins_x'], info['n_bins_y']), info['do_smooth'])
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
        
        # Branch 1: Region Stats Export
        if current_tab == getattr(self.mw, 'region_tab', None):
             viewer = current_tab.findChild(RegionResultViewer)
             if viewer:
                 df, fname = viewer.get_export_data()
                 path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
                 if path:
                     df.to_csv(path, index=False)
                     self.mw.log_message(f"Saved to {path}")
             else:
                 self.mw.log_message("Region Stats export unavailable: RegionResultViewer not found.")
             return

        # Branch 2: General Visualization Export (e.g. Group Map, Grid Analysis)
        viewer = self.mw.visualization_widgets.get(current_tab)
        if viewer and hasattr(viewer, 'get_export_data'):
             df, fname = viewer.get_export_data()
             path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
             if path:
                 df.to_csv(path, index=False)
                 self.mw.log_message(f"Saved to {path}")

                 # New Feature: Grid Bin Summary Export
                 # Only trigger if the columns look like grid analysis
                 # Silent detection first
                 
                 def _find_col(candidates):
                     for c in candidates:
                         if c in df.columns: return c
                     return None

                 col_x = 'Grid_X_Index'
                 col_y = 'Grid_Y_Index'
                 col_grp = 'Group'
                 
                 anim_cands = ['Source_Animal', 'Animal', 'Animal_ID', 'SourceAnimal']
                 col_anim = _find_col(anim_cands)
                 
                 phase_cands = ['Relative_Phase_Hours', 'Rel_Phase_Animal', 'Rel_Phase_Control', 'Rel_Phase', 'Phase_CT']
                 col_phase = _find_col(phase_cands)
                 
                 # Strict requirements for grid summary
                 missing_cols = []
                 if col_x not in df.columns: missing_cols.append(col_x)
                 if col_y not in df.columns: missing_cols.append(col_y)
                 if col_grp not in df.columns: missing_cols.append(col_grp)
                 if not col_anim: missing_cols.append("Animal")
                 if not col_phase: missing_cols.append("Phase")
                 
                 # If valid grid data, proceed to summary
                 if not missing_cols:
                     self.mw.log_message(f"Grid export detected, generating summary...")
                     try:
                         # Create normalized DF for summary
                         norm_df = pd.DataFrame()
                         norm_df['Grid_X_Index'] = df[col_x]
                         norm_df['Grid_Y_Index'] = df[col_y]
                         norm_df['Group'] = df[col_grp]
                         norm_df['Source_Animal'] = df[col_anim]
                         norm_df['Relative_Phase_Hours'] = df[col_phase]
                         
                         if 'Phase_Domain' in df.columns:
                             norm_df['Phase_Domain'] = df['Phase_Domain']

                         summary_df = self._compute_grid_bin_summary(norm_df)
                         
                         base_root, ext = os.path.splitext(path)
                         summary_path = f"{base_root}_grid_bin_summary{ext}"
                         summary_df.to_csv(summary_path, index=False)
                         
                         self.mw.log_message(f"Saved Grid Bin Summary to {summary_path} (shape: {summary_df.shape})")
                     except Exception as e:
                          self.mw.log_message(f"Error generating Grid Bin Summary: {e}")
                          import traceback
                          self.mw.log_message(traceback.format_exc())
             return

    def _compute_grid_bin_summary(self, df):
        """
        Aggregates per-cell grid data into per-bin statistics (cell-pooled and animal-level).
        Returns: pd.DataFrame with one row per (Group, Grid_X, Grid_Y).
        """
        # Ensure numeric type
        df = df.copy()
        df['Relative_Phase_Hours'] = pd.to_numeric(df['Relative_Phase_Hours'], errors='coerce')
        
        # Group extraction
        groups = df.groupby(['Group', 'Grid_X_Index', 'Grid_Y_Index'], sort=True)
        rows = []
        
        # Period is standard 24.0 for this app's "Hours" exports
        period = 24.0
        min_cells_per_animal = 5
        
        # Helper for domain inference
        def _infer_and_convert(mean_h, domain):
            """
            Converts/wraps the raw circular mean (approx [-12, 12] from atan2)
            into the target domain format.
            """
            if not np.isfinite(mean_h): return mean_h
            if domain == 'abs_0_24':
                return (mean_h + period) % period
            if domain == 'rel_pm12':
                # Wrap to [-12, +12)
                return ((mean_h + period/2) % period) - period/2
            return mean_h # 'unknown' -> leave as-is

        for (grp_name, gx, gy), sub in groups:
            # 1. Basic Counts (Metadata)
            n_cells = len(sub)
            source_counts = sub['Source_Animal'].value_counts()
            n_animals = len(source_counts)
            
            # Form pipe-separated list of animals (sorted)
            uniq_animals = sorted(source_counts.index.astype(str))
            animals_str = "|".join(uniq_animals)
            if len(animals_str) > 2000: animals_str = animals_str[:2000] + "..."
            
            # Cells Per Animal stats (Unfiltered, metadata)
            med_cells = source_counts.median()
            min_cells = source_counts.min() if len(source_counts) > 0 else 0
            max_cells = source_counts.max() if len(source_counts) > 0 else 0
            
            # 2. Domain Inference per bin
            # Identify valid phases (FINITE only) for the whole bin
            # Robust extraction: coerce -> numpy -> filter finite
            raw_vals = pd.to_numeric(sub['Relative_Phase_Hours'], errors='coerce').values
            valid_phases_all = raw_vals[np.isfinite(raw_vals)]
            
            phase_domain = 'unknown'
            
            # (A) Explicit Phase_Domain column priority
            if 'Phase_Domain' in sub.columns:
                # Extract non-null, strip, lower
                d_vals = sub['Phase_Domain'].dropna().astype(str).str.strip().str.lower()
                # Filter to ONLY allowed domains
                allowed = {'rel_pm12', 'abs_0_24'}
                d_vals = d_vals[d_vals.isin(allowed)]
                
                if len(d_vals) > 0:
                    # Pick most frequent allowed value
                    phase_domain = d_vals.value_counts().idxmax()
            
            # (B) Heuristic fallback if still unknown
            if phase_domain == 'unknown' and len(valid_phases_all) > 0:
                # If we see ANY negative value (with tolerance), assume [-12, 12). Else assume [0, 24).
                if np.min(valid_phases_all) < -1e-6:
                    phase_domain = 'rel_pm12'
                else:
                    phase_domain = 'abs_0_24'
            
            # 3. Cell-Pooled Circular Stats
            if len(valid_phases_all) > 0:
                rads = (valid_phases_all / period) * (2 * np.pi)
                S = np.mean(np.sin(rads))
                C = np.mean(np.cos(rads))
                pooled_R = np.sqrt(S**2 + C**2)
                mean_rad = np.arctan2(S, C) # [-pi, pi]
                # Convert back to hours, initially in [-12, 12] range approx
                raw_mean_h = (mean_rad / (2 * np.pi)) * period
                pooled_mean_h = _infer_and_convert(raw_mean_h, phase_domain)
            else:
                pooled_R = np.nan
                pooled_mean_h = np.nan
            
            # 4. Animal-Level Circular Stats
            # CRITICAL: Filter animals based on VALID FINITE PHASES, not raw counts.
            # Collect means from valid animals.
            animal_means_rad = []
            
            # Pass over all unique animals in this bin
            for anim in source_counts.index:
                # Extract phases for this animal, strictly finite
                a_raw = pd.to_numeric(sub.loc[sub['Source_Animal'] == anim, 'Relative_Phase_Hours'], errors='coerce').values
                a_phases = a_raw[np.isfinite(a_raw)]
                
                if len(a_phases) >= min_cells_per_animal:
                    # Valid animal
                    a_rads = (a_phases / period) * (2 * np.pi)
                    aS = np.mean(np.sin(a_rads))
                    aC = np.mean(np.cos(a_rads))
                    a_mean_rad = np.arctan2(aS, aC)
                    animal_means_rad.append(a_mean_rad)
            
            animal_level_n_used = len(animal_means_rad)
            
            animal_level_R = np.nan
            animal_level_mean_h = np.nan
            
            if animal_level_n_used > 0:
                am_arr = np.array(animal_means_rad)
                S_am = np.mean(np.sin(am_arr))
                C_am = np.mean(np.cos(am_arr))
                animal_level_R = np.sqrt(S_am**2 + C_am**2)
                mean_am_rad = np.arctan2(S_am, C_am)
                raw_am_h = (mean_am_rad / (2 * np.pi)) * period
                animal_level_mean_h = _infer_and_convert(raw_am_h, phase_domain)

            rows.append({
                'Group': grp_name,
                'Grid_X_Index': gx,
                'Grid_Y_Index': gy,
                'Phase_Domain': phase_domain,
                'N_Cells': n_cells,
                'N_Animals': n_animals, # Unique source animals (unfiltered)
                'Animals_List': animals_str,
                'Cells_Per_Animal_Median': med_cells,
                'Cells_Per_Animal_Min': min_cells,
                'Cells_Per_Animal_Max': max_cells,
                'CellPooled_CircMean_RelPhaseHours': pooled_mean_h,
                'CellPooled_R': pooled_R,
                'AnimalLevel_CircMean_RelPhaseHours': animal_level_mean_h,
                'AnimalLevel_R': animal_level_R,
                'AnimalLevel_N': animal_level_n_used, # Contributing animals
                'AnimalLevel_N_Filtered': animal_level_n_used,
                'Meets_MinAnimals_ForInference': (animal_level_n_used >= 2),
                'Meets_MinCells_ForDisplay': (n_cells >= 10)
            })
            
        res_df = pd.DataFrame(rows)
        
        # Diagnostic Safeguard (Self-Check)
        # Random sample of 5 bins to verify aggregation counts
        if len(rows) >= 5:
            try:
                indices = np.random.choice(len(rows), 5, replace=False)
                for idx in indices:
                    r = rows[idx]
                    # query original df
                    mask = (df['Group'] == r['Group']) & \
                           (df['Grid_X_Index'] == r['Grid_X_Index']) & \
                           (df['Grid_Y_Index'] == r['Grid_Y_Index'])
                    sub_check = df[mask]
                    
                    real_n_cells = len(sub_check)
                    real_n_animals = sub_check['Source_Animal'].nunique()
                    
                    if real_n_cells != r['N_Cells']:
                        self.mw.log_message(f"Grid Export Diagnostic Error: Bin({r['Grid_X_Index']},{r['Grid_Y_Index']}) N_Cells mismatch: {real_n_cells} vs {r['N_Cells']}")
                    if real_n_animals != r['N_Animals']:
                        self.mw.log_message(f"Grid Export Diagnostic Error: Bin({r['Grid_X_Index']},{r['Grid_Y_Index']}) N_Animals mismatch: {real_n_animals} vs {r['N_Animals']}")
            except Exception as e:
                self.mw.log_message(f"Grid Export Diagnostic skipped due to error: {e}")
                
        return res_df

    def _normalize_lobe_id(self, lobe_id):
        """
        Robustly convert lobe_id to integer (or None if invalid/0).
        Supports: int, numeric string, 'Left'/'L'->1, 'Right'/'R'->2.
        Returns None for 0/unassigned to enforce explicit valid assignment.
        """
        if isinstance(lobe_id, (int, np.integer)):
            val = int(lobe_id)
            return val if val != 0 else None
        
        if isinstance(lobe_id, str):
            s = lobe_id.strip().lower()
            if not s: 
                return None
            if s.isdigit():
                val = int(s)
                return val if val != 0 else None
            if s in ("left", "l"):
                return 1
            if s in ("right", "r"):
                return 2
                
        return None

    def run_cluster_analysis(self):
        if not hasattr(self, "_last_master_df") or self._last_master_df is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please generate group visualizations first to prepare data.")
            return

        df = self._last_master_df.copy()
        
        # D3. Group Consistency
        group_counts = df.groupby('Animal')['Group'].nunique()
        inconsistent = group_counts[group_counts > 1]
        if not inconsistent.empty:
            bad_animals = inconsistent.index.tolist()
            QtWidgets.QMessageBox.critical(self, "Data Error", f"Found animals with multiple group labels: {bad_animals}. Fix input CSVs.")
            return

        # D4. Units Check
        valid_phases = df['Rel_Phase'].dropna()
        if not valid_phases.empty:
             abs_max = valid_phases.abs().max()
             if abs_max > 48.0:
                  QtWidgets.QMessageBox.critical(self, "Units Error", f"Rel_Phase values > 48 hours found (max {abs_max:.1f}). Check normalization.")
                  return
        
        # Check Groups
        unique_groups = df['Group'].unique()
        if 'Control' not in unique_groups or 'Experiment' not in unique_groups:
             QtWidgets.QMessageBox.warning(self, "Missing Groups", "Both 'Control' and 'Experiment' groups are required.")
             return
             
        # B. Use Shared Grid Logic
        try:
             df_grid, info = self._compute_grid_assignment(df)
        except Exception as e:
             QtWidgets.QMessageBox.critical(self, "Grid Error", f"Failed to compute grid: {e}")
             return
             
        n_bins_x, n_bins_y = info['n_bins_x'], info['n_bins_y']
        
        # 3. Create Lobe Mask
        lobe_mask = np.zeros((n_bins_y, n_bins_x), dtype=int)
        
        # Center coordinates
        # Reconstruct centers from keys or bins
        # info['calc_x_bins'] are edges.
        cx_centers = 0.5 * (info['calc_x_bins'][:-1] + info['calc_x_bins'][1:])
        cy_centers = 0.5 * (info['calc_y_bins'][:-1] + info['calc_y_bins'][1:])
        gx, gy = np.meshgrid(cx_centers, cy_centers)
        centers = np.column_stack((gx.ravel(), gy.ravel()))
        
        if not hasattr(self, 'zones') or not self.zones:
             QtWidgets.QMessageBox.warning(self, "No Regions", "No regions/lobes defined. Cannot run cluster analysis.")
             return
             
        # Rasterize zones
        invalid_lobes = []
        
        for zid, zone_info in self.zones.items():
            lobe_raw = zone_info.get('lobe', 0)
            
            lobe_int = self._normalize_lobe_id(lobe_raw)
            if lobe_int is None:
                s = str(lobe_raw)
                trunc_s = s[:20] + ('...' if len(s) > 20 else '')
                invalid_lobes.append(f"{zid}: {trunc_s}")
                continue
                
            polys = zone_info['polygons']
            mask_in_zone = np.zeros(len(centers), dtype=bool)
            for poly in polys:
                mask_in_zone |= poly.contains_points(centers)
            
            mask_2d = mask_in_zone.reshape((n_bins_y, n_bins_x))
            lobe_mask[mask_2d] = lobe_int
            
        if invalid_lobes:
            msg = "Cluster analysis cannot run because some regions have invalid lobe values:\n" + "\n".join(invalid_lobes[:10])
            if len(invalid_lobes) > 10: msg += "\n..."
            QtWidgets.QMessageBox.critical(self, "Invalid Lobes", msg)
            return

        if np.max(lobe_mask) == 0:
            QtWidgets.QMessageBox.warning(self, "No Lobes", "No zones with Lobe ID > 0 found. Please define Left (1) / Right (2) lobes in Region setup.")
            return

        # 4. Aggregate Phases & Prepare Stats
        grouped_phases_grid = {}
        bin_stats = [] # List of (n_c, n_e) for dialog
        
        df_valid = df_grid.dropna(subset=['Grid_X_Index', 'Grid_Y_Index', 'Rel_Phase'])
        
        # Iterate all bins in grid
        for (bx, by), bin_df in df_valid.groupby(['Grid_X_Index', 'Grid_Y_Index']):
            bx, by = int(bx), int(by)
            if not (0 <= by < n_bins_y and 0 <= bx < n_bins_x): continue
            
            # Check Lobe: Only gather stats for lobe bins (Candidate Bins)
            if lobe_mask[by, bx] == 0: continue
            
            c_dict = {}
            e_dict = {}
            
            for animal_name, anim_df in bin_df.groupby('Animal'):
                phases = anim_df['Rel_Phase'].values
                if len(phases) == 0: continue
                # Circular Mean
                rads = (phases / 24.0) * (2*np.pi)
                m_rad = circmean(rads, low=-np.pi, high=np.pi)
                
                group = anim_df['Group'].iloc[0]
                if group == 'Control':
                    c_dict[animal_name] = m_rad
                elif group == 'Experiment':
                    e_dict[animal_name] = m_rad
            
            n_c = len(c_dict)
            n_e = len(e_dict)
            
            # Record stat for dialog
            if n_c > 0 or n_e > 0:
                bin_stats.append((n_c, n_e))
            
            if n_c > 0 and n_e > 0:
                 grouped_phases_grid[(by, bx)] = {'Control': c_dict, 'Experiment': e_dict} 
        
        # 5. Dialog with Dynamic updates
        dlg = ClusterConfigDialog(self, (n_bins_y, n_bins_x), bin_stats)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        # 6. Run Worker
        self.mw.log_message(f"Starting Cluster Analysis: MinN={dlg.min_n}, Perms={dlg.n_perm}, Alpha={dlg.alpha}, Seed={dlg.seed}")
        self.btn_cluster_stats.setEnabled(False)
        self.mw.set_status("Running Cluster Analysis...")
        
        # Robustly get session dir
        s_dir = getattr(self.mw, "session_dir", None)
        if (not s_dir) or (not os.path.isdir(s_dir)):
            s_dir = os.path.join(os.getcwd(), "analysis_results")
            os.makedirs(s_dir, exist_ok=True)
            self.mw.log_message(f"Warning: Session dir not found in MW. Saving to {s_dir}")

        self.cluster_worker = ClusterWorker(
            grouped_phases_grid, (n_bins_y, n_bins_x), lobe_mask,
            dlg.min_n, dlg.n_perm, dlg.seed, dlg.alpha, dlg.save_plot,
            s_dir
        )
        self.cluster_worker.finished.connect(self.on_cluster_finished)
        self.cluster_worker.error.connect(self.on_cluster_error)
        self.cluster_worker.start()
        
    def on_cluster_finished(self, results):
        self.btn_cluster_stats.setEnabled(True)
        self.mw.set_status("Cluster Analysis Complete.")
        self.mw.log_message(f"Cluster Analysis Done. T0={results.get('T0', 0):.4f}, Clusters={len(results.get('clusters', []))}")
        
        if not hasattr(self.mw, 'cluster_tab'):
             self.mw.cluster_tab = QtWidgets.QWidget()
             self.mw.vis_tabs.addTab(self.mw.cluster_tab, "Cluster Stats")
        
        # Layout management
        lay = self.mw.cluster_tab.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(self.mw.cluster_tab)
            self.mw.cluster_tab.setLayout(lay)
        else:
            clear_layout(lay)
            
        viewer = ClusterResultViewerWidget(results, self.mw.cluster_tab)
        lay.addWidget(viewer)
        
        self.mw.visualization_widgets[self.mw.cluster_tab] = viewer
        self.mw.vis_tabs.setCurrentWidget(self.mw.cluster_tab)
        self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.cluster_tab), True)

    def on_cluster_error(self, err_msg):
        self.btn_cluster_stats.setEnabled(True)
        self.mw.set_status("Cluster Analysis Failed.")
        self.mw.log_message(f"Cluster Analysis Error: {err_msg}")
        QtWidgets.QMessageBox.critical(self, "Analysis Failed", err_msg)

class ClusterWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, grouped_phases, grid_shape, mask, min_n, n_perm, seed, alpha, save_plot, session_dir):
        super().__init__()
        self.grouped_phases = grouped_phases
        self.grid_shape = grid_shape
        self.mask = mask
        self.min_n = min_n
        self.n_perm = n_perm
        self.seed = seed
        self.alpha = alpha
        self.save_plot = save_plot
        self.session_dir = session_dir
        
    def run(self):
        try:
            results = cluster_stats.run_bin_cluster_analysis(
                self.grouped_phases, self.grid_shape, self.mask,
                min_n=self.min_n, n_perm=self.n_perm, seed=self.seed, alpha=self.alpha
            )
            # Add alpha
            results['alpha'] = self.alpha
            
            # Save Results
            out_dir = os.path.join(self.session_dir, "cluster_stats")
            cluster_stats.save_cluster_results(results, out_dir)
            
            if self.save_plot:
                png_path = os.path.join(out_dir, "diagnostic_plot.png")
                cluster_stats.plot_cluster_results(results, png_path)
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

