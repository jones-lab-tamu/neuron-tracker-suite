import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from scipy.stats import circmean, f as f_dist
from skimage.draw import polygon as draw_polygon

from gui.utils import Tooltip, add_mpl_to_tab, clear_layout, project_points_to_polyline
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer, PhaseGradientViewer
from gui.statistics import watson_williams_f
from gui.dialogs.roi_drawer import ROIDrawerDialog
from gui.theme import get_icon

# -----------------------------------------------------------------------------
# Visualization Classes
# -----------------------------------------------------------------------------

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
        
        self._draw_map()
        self._populate_table()

    def _draw_map(self):
        ax = self.fig.add_subplot(111)
        ax.set_title("Region Phase Differences (Exp - Ctrl)\n* = Significant (p < 0.05)")
        ax.set_aspect('equal')
        
        # Colormap setup: Blue (Early) -> White -> Red (Late)
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
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

    def get_export_data(self):
        # Flatten stats to DF
        data = []
        for s in self.zone_stats:
            row = s.copy()
            if 'data' in row: del row['data'] 
            data.append(row)
        return pd.DataFrame(data), "region_stats.csv"

# -----------------------------------------------------------------------------
# Main Panel
# -----------------------------------------------------------------------------

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
        
        # File List
        self.group_list = QtWidgets.QTreeWidget()
        self.group_list.setHeaderLabels(["File Path", "Group"])
        self.group_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        b.addWidget(self.group_list)
        
        # Add/Remove Buttons
        row = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton(get_icon('fa5s.plus'), "Add Warped ROI File(s)...")
        self.btn_remove_group = QtWidgets.QPushButton(get_icon('fa5s.minus'), "Remove Selected")
        row.addWidget(self.btn_add_group)
        row.addWidget(self.btn_remove_group)
        b.addLayout(row)
        
        # Group Assignment Buttons
        assign_row = QtWidgets.QHBoxLayout()
        self.btn_assign_control = QtWidgets.QPushButton("Set Selected as Control")
        self.btn_assign_exp = QtWidgets.QPushButton("Set Selected as Experiment")
        assign_row.addWidget(self.btn_assign_control)
        assign_row.addWidget(self.btn_assign_exp)
        b.addLayout(assign_row)
        
        # Continuous Map Parameters
        param_box = QtWidgets.QGroupBox("Continuous Map Parameters")
        param_layout = QtWidgets.QFormLayout(param_box)
        self.group_grid_res_edit = QtWidgets.QLineEdit("50")
        self.group_smooth_check = QtWidgets.QCheckBox("Smooth to fill empty bins")
        param_layout.addRow("Grid Resolution:", self.group_grid_res_edit)
        param_layout.addRow(self.group_smooth_check)
        b.addWidget(param_box)        
        
        # Regional Analysis Setup
        region_box = QtWidgets.QGroupBox("Regional Analysis Setup")
        region_layout = QtWidgets.QFormLayout(region_box)
        
        # Atlas Template Loader
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
        
        # Define Regions Button
        self.btn_define_regions = QtWidgets.QPushButton(get_icon('fa5s.draw-polygon'), "Define Analysis Regions...")
        self.region_status_label = QtWidgets.QLabel("No regions defined")
        self.region_status_label.setStyleSheet("color: gray; font-style: italic;")
        region_layout.addRow(self.btn_define_regions, self.region_status_label)

        # Stats Parameters
        self.min_region_cells_spin = QtWidgets.QSpinBox()
        self.min_region_cells_spin.setRange(1, 100)
        self.min_region_cells_spin.setValue(10)
        region_layout.addRow("Min Cells / Region / Animal:", self.min_region_cells_spin)

        self.min_animals_spin = QtWidgets.QSpinBox()
        self.min_animals_spin.setRange(2, 50)
        self.min_animals_spin.setValue(3)
        region_layout.addRow("Min Animals / Group (Stats):", self.min_animals_spin)

        b.addWidget(region_box)
        
        # Generate Button
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
    
    def assign_group(self, group_name: str):
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select one or more files to assign.")
            return
        for item in selected_items:
            item.setText(1, group_name)
            item.setForeground(1, QtGui.QColor('blue') if group_name == "Control" else QtGui.QColor('red'))
    
    def define_regions(self):
        """Draw/Edit regions using the ROI Drawer on top of the Atlas."""
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
            
            # Draw atlas Include polygons in Gray
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
        
        # Check for existing regions file
        region_file = path.replace('_anatomical_roi.json', '_anatomical_regions.json')
        if os.path.exists(region_file):
            with open(region_file, 'r') as f: regions = json.load(f)
            self.region_status_label.setText(f"{len(regions)} regions found.")
            self.region_status_label.setStyleSheet("color: green;")

    def add_group_files(self):
        start_dir = self.mw._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Warped ROI Files", start_dir, "Warped ROI files (*_roi_warped.csv)"
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

    # -------------------------------------------------------------------------
    # Analysis Logic
    # -------------------------------------------------------------------------

    def generate_group_visualizations(self):
        self.mw.log_message("--- Starting Group Analysis ---")
        
        # 1. Validation
        if not self.state.group_data_paths: return
        atlas_path = self.state.atlas_roi_path
        if not atlas_path:
            self.mw.log_message("Error: No Atlas Template loaded.")
            return

        # 2. Load Regions
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

        # 3. Load Animal Data
        all_dfs = []
        try:
            group_map = {}
            root = self.group_list.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                group_map[item.text(0)] = item.text(1)

            for roi_path in self.state.group_data_paths:
                base = os.path.basename(roi_path).replace('_roi_warped.csv', '')
                group = group_map.get(roi_path, "Unassigned")
                if group not in ["Control", "Experiment"]: continue
                
                rhythm_path = roi_path.replace("_roi_warped.csv", "_rhythm_results.csv")
                if not os.path.exists(rhythm_path): 
                     self.mw.log_message(f"Skipping {base}: No rhythm results.")
                     continue
                
                coords = np.loadtxt(roi_path, delimiter=",")
                rhythm_df = pd.read_csv(rhythm_path)
                
                # Check consistency
                if len(coords) != len(rhythm_df):
                    self.mw.log_message(f"Warning: Length mismatch in {base}. Skipping.")
                    continue
                    
                mask = rhythm_df['Is_Rhythmic'].astype(bool).values
                if not np.any(mask): continue

                # Standardize to CT
                phases = rhythm_df['Phase_Hours'][mask].values
                periods = rhythm_df['Period_Hours'][mask].values
                phases_ct = (phases / periods) * 24.0
                
                df = pd.DataFrame({
                    'Animal': base, 'Group': group,
                    'X': coords[mask, 0], 'Y': coords[mask, 1],
                    'Phase_CT': phases_ct
                })
                all_dfs.append(df)

            if not all_dfs:
                self.mw.log_message("No valid animal data found.")
                return
            master_df = pd.concat(all_dfs, ignore_index=True)

            # 4. Global Normalization
            ctrl_phases = master_df[master_df['Group'] == 'Control']['Phase_CT'].values
            if len(ctrl_phases) == 0:
                self.mw.log_message("Error: No control cells found for normalization.")
                return
            
            # Grand Control Mean
            rads = (ctrl_phases / 24.0) * (2 * np.pi)
            ref_rad = circmean(rads)
            ref_phase = (ref_rad / (2 * np.pi)) * 24.0
            
            master_df['Rel_Phase'] = (master_df['Phase_CT'] - ref_phase + 12.0) % 24.0 - 12.0
            self.mw.log_message(f"Normalized to Control Mean Phase: {ref_phase:.2f}h")

            # --- PART A: CONTINUOUS GRID MAPS ---
            self._generate_continuous_maps(master_df)

            # --- PART B: GRADIENT ANALYSIS (Legacy) ---
            self._generate_gradient_analysis(master_df)

            # --- PART C: REGIONAL STATS ---
            self._generate_regional_stats(master_df)

        except Exception as e:
            self.mw.log_message(f"Analysis Error: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())

    def _generate_continuous_maps(self, df):
        # Create tabs
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
            
            # Scatter Plot
            scatter_df = df.rename(columns={'Animal': 'Source_Animal', 'X': 'Warped_X', 'Y': 'Warped_Y', 'Rel_Phase': 'Relative_Phase_Hours'})
            fig_s, _ = add_mpl_to_tab(self.mw.group_scatter_tab)
            viewer_s = GroupScatterViewer(fig_s, fig_s.add_subplot(111), scatter_df, grid_bins=(grid_x_bins, grid_y_bins))
            self.mw.visualization_widgets[self.mw.group_scatter_tab] = viewer_s
            
            # Average Map
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
            
            # 1. Aggregate per Animal
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
                    
                    zone_record['data'].append({
                        'animal': animal,
                        'group': subset['Group'].iloc[0],
                        'mean': m_val
                    })

            # 2. Run Statistics (Watson-Williams)
            ctrl_means = [d['mean'] for d in zone_record['data'] if d['group'] == 'Control']
            exp_means = [d['mean'] for d in zone_record['data'] if d['group'] == 'Experiment']
            
            zone_record['n_ctrl'] = len(ctrl_means)
            zone_record['n_exp'] = len(exp_means)
            
            if len(ctrl_means) >= min_animals and len(exp_means) >= min_animals:
                # Use Circmean for group summary too
                def group_circ(vals):
                    r = (np.array(vals) / 24.0) * (2*np.pi)
                    return (circmean(r) / (2*np.pi)) * 24.0

                zone_record['mean_ctrl'] = group_circ(ctrl_means)
                zone_record['mean_exp'] = group_circ(exp_means)
                
                # Watson-Williams
                f_val = watson_williams_f(ctrl_means, exp_means)
                df2 = len(ctrl_means) + len(exp_means) - 2
                p_val = 1.0 - f_dist.cdf(f_val, 1, df2)
                
                zone_record['p_value'] = p_val
                
                # Circular Diff (Exp - Ctrl)
                c_mean = zone_record['mean_ctrl']
                e_mean = zone_record['mean_exp']
                diff = (e_mean - c_mean + 12.0) % 24.0 - 12.0
                zone_record['diff_mean'] = diff
            else:
                zone_record['p_value'] = 1.0
                zone_record['diff_mean'] = 0.0

            self.final_zone_stats.append(zone_record)

        # 3. Create Viewer
        if not hasattr(self.mw, 'region_tab'):
            self.mw.region_tab = QtWidgets.QWidget()
            self.mw.vis_tabs.addTab(self.mw.region_tab, "Region Stats")
        
        if self.mw.region_tab.layout(): clear_layout(self.mw.region_tab.layout())
        
        viewer = RegionResultViewer(self.final_zone_stats, self.zones_polygons_map)
        self.mw.visualization_widgets[self.mw.region_tab] = viewer
        
        layout = QtWidgets.QVBoxLayout(self.mw.region_tab)
        layout.addWidget(viewer)
        
        self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.region_tab), True)
        self.mw.vis_tabs.setCurrentWidget(self.mw.region_tab)
        self.mw.log_message("Region Analysis Complete.")

    def _generate_gradient_analysis(self, df):
        """Calculates phase gradients if 'Phase Axis' exists in Atlas."""
        atlas_path = self.state.atlas_roi_path
        if not atlas_path: return
        try:
            with open(atlas_path, 'r') as f: atlas_data = json.load(f)
            axis_rois = [r for r in atlas_data if r.get('mode') == 'Phase Axis']
            if not axis_rois: return
            
            self.mw.log_message(f"Running Gradient Analysis ({len(axis_rois)} Axes)...")
            
            # Simple Single-Axis Logic for Robustness
            axis_poly = np.array(axis_rois[0]['path_vertices'])
            
            gradient_data = []
            animals = df['Animal'].unique()
            
            for animal in animals:
                subset = df[df['Animal'] == animal]
                points = subset[['X', 'Y']].values
                phases = subset['Rel_Phase'].values
                
                s_vals = project_points_to_polyline(points, axis_poly)
                
                # Binning
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                binned_phases = []
                
                for k in range(len(bins)-1):
                    mask = (s_vals >= bins[k]) & (s_vals < bins[k+1])
                    if np.sum(mask) > 0:
                        p_bin = phases[mask]
                        rads = (p_bin / 24.0) * 2 * np.pi
                        m_rad = circmean(rads, low=-np.pi, high=np.pi)
                        m_h = (m_rad / (2 * np.pi)) * 24.0
                        binned_phases.append(m_h)
                    else:
                        binned_phases.append(np.nan)
                
                gradient_data.append({
                    'animal': animal,
                    's': bin_centers,
                    'phases': np.array(binned_phases)
                })
                
            if not hasattr(self.mw, 'grad_tab'):
                self.mw.grad_tab = QtWidgets.QWidget()
                self.mw.vis_tabs.addTab(self.mw.grad_tab, "Gradient")
            
            if self.mw.grad_tab.layout(): clear_layout(self.mw.grad_tab.layout())
            
            fig, _ = add_mpl_to_tab(self.mw.grad_tab)
            viewer_g = PhaseGradientViewer(fig, fig.add_subplot(111), gradient_data)
            self.mw.visualization_widgets[self.mw.grad_tab] = viewer_g
            
            layout = QtWidgets.QVBoxLayout(self.mw.grad_tab)
            layout.addWidget(viewer_g)
            
            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.grad_tab), True)

        except Exception as e:
            self.mw.log_message(f"Gradient Analysis Warning: {e}")

    def export_current_data(self):
        current_tab = self.mw.vis_tabs.currentWidget()
        
        # Region Viewer Nested Case
        if current_tab == getattr(self.mw, 'region_tab', None):
             viewer = current_tab.findChild(RegionResultViewer)
             if viewer:
                 df, fname = viewer.get_export_data()
                 path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
                 if path:
                     df.to_csv(path, index=False)
                     self.mw.log_message(f"Saved to {path}")
                 return

        # Standard Case
        viewer = self.mw.visualization_widgets.get(current_tab)
        if viewer and hasattr(viewer, 'get_export_data'):
             df, fname = viewer.get_export_data()
             path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", fname, "CSV (*.csv)")
             if path:
                 df.to_csv(path, index=False)
                 self.mw.log_message(f"Saved to {path}")