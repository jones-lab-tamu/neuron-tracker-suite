import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.path import Path
from scipy.stats import circmean
from skimage.draw import polygon as draw_polygon


from gui.utils import Tooltip, add_mpl_to_tab, clear_layout, project_points_to_polyline
from gui.viewers import GroupScatterViewer, GroupAverageMapViewer, GroupDifferenceViewer, PhaseGradientViewer, GraphDifferenceViewer
from gui.analysis import (
    calculate_phases_fft, 
    preprocess_for_rhythmicity, 
    compute_median_window_frames,
    strict_cycle_mask,
    RHYTHM_TREND_WINDOW_HOURS
)
import cosinor as csn
from gui.theme import get_icon
from gui.statistics import build_animal_phase_matrix, cluster_based_permutation_test_by_animal, run_graph_cbpt
from gui.scaffold import AnatomicalNodeScaffold
from gui.dialogs.roi_drawer import ROIDrawerDialog


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
        
        # File List
        self.group_list = QtWidgets.QTreeWidget()
        self.group_list.setHeaderLabels(["File Path", "Group"])
        self.group_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        self.group_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
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
        
        # Parameters Box
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
        
        # Comparison Controls (Added as sibling to param_box)
        comp_box = QtWidgets.QGroupBox("Group Comparison")
        # Use a Form Layout for clarity
        comp_layout = QtWidgets.QFormLayout(comp_box)
        
        # Row 1: Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_set_reference = QtWidgets.QPushButton("Set Current as Control (Ref)")
        Tooltip.install(self.btn_set_reference, "...")
        self.btn_set_reference.setEnabled(False)
        
        self.btn_compare = QtWidgets.QPushButton("Run CBPT Analysis")
        Tooltip.install(self.btn_compare, "...")
        self.btn_compare.setEnabled(False)
        
        btn_row.addWidget(self.btn_set_reference)
        btn_row.addWidget(self.btn_compare)
        comp_layout.addRow(btn_row)
        
        # Dedicated Checkbox for Bridging
        self.bridge_gaps_check = QtWidgets.QCheckBox("Bridge Gaps in Cluster Test")
        Tooltip.install(self.bridge_gaps_check, "Connects significant pixels separated by small gaps, allowing larger clusters to form. Use this if your data is sparse.")
        self.bridge_gaps_check.setChecked(True) # Default to ON
        comp_layout.addRow(self.bridge_gaps_check)
        
        # graph_box = QtWidgets.QGroupBox("Graph-Based Spatial Analysis")
        # graph_layout = QtWidgets.QFormLayout(graph_box)

        # # 1. Load Scaffold Button
        # scaffold_widget = QtWidgets.QWidget()
        # scaffold_layout = QtWidgets.QHBoxLayout(scaffold_widget)
        # scaffold_layout.setContentsMargins(0, 0, 0, 0)
        # self.btn_load_scaffold = QtWidgets.QPushButton(get_icon('fa5s.cogs'), "Load Node Scaffold...")
        # self.scaffold_path_edit = QtWidgets.QLineEdit()
        # self.scaffold_path_edit.setReadOnly(True)
        # self.scaffold_path_edit.setPlaceholderText("No scaffold loaded.")
        # scaffold_layout.addWidget(self.btn_load_scaffold)
        # scaffold_layout.addWidget(self.scaffold_path_edit)
        # graph_layout.addRow("Scaffold File:", scaffold_widget)

        # # 2. Parameters
        # self.knn_spinbox = QtWidgets.QSpinBox()
        # self.knn_spinbox.setRange(1, 20)
        # self.knn_spinbox.setValue(6)
        # Tooltip.install(self.knn_spinbox, "<b>k-NN Neighbors (k):</b> Defines connectivity. How many nearest nodes are considered neighbors for forming clusters. (Default: 6)")
        # graph_layout.addRow("k-NN Neighbors:", self.knn_spinbox)

        # self.search_radius_spinbox = QtWidgets.QDoubleSpinBox()
        # self.search_radius_spinbox.setRange(1.0, 500.0)
        # self.search_radius_spinbox.setValue(40.0)
        # Tooltip.install(self.search_radius_spinbox, "<b>Cell Search Radius (μm):</b> Max distance to assign a cell to a node.")
        # graph_layout.addRow("Cell Search Radius:", self.search_radius_spinbox)

        # self.min_cells_spinbox = QtWidgets.QSpinBox()
        # self.min_cells_spinbox.setRange(1, 20)
        # self.min_cells_spinbox.setValue(2)
        # Tooltip.install(self.min_cells_spinbox, "<b>Min Cells / Animal / Node:</b> An animal's data is only used at a node if it has at least this many cells nearby.")
        # graph_layout.addRow("Min Cells / Animal / Node:", self.min_cells_spinbox)

        # self.min_animals_spinbox = QtWidgets.QSpinBox()
        # self.min_animals_spinbox.setRange(1, 20)
        # self.min_animals_spinbox.setValue(3)
        # Tooltip.install(self.min_animals_spinbox, "<b>Min Animals / Group / Node:</b> A node is only tested if BOTH groups have at least this many animals with valid data there.")
        # graph_layout.addRow("Min Animals / Group / Node:", self.min_animals_spinbox)

        # self.cluster_alpha_spinbox = QtWidgets.QDoubleSpinBox()
        # self.cluster_alpha_spinbox.setRange(0.01, 1.0)
        # self.cluster_alpha_spinbox.setSingleStep(0.05)
        # self.cluster_alpha_spinbox.setValue(0.2)
        # Tooltip.install(self.cluster_alpha_spinbox, "<b>Cluster Alpha:</b> The initial p-value threshold for a node to be considered part of a potential cluster.")
        # graph_layout.addRow("Cluster Alpha:", self.cluster_alpha_spinbox)

        # self.permutations_spinbox = QtWidgets.QSpinBox()
        # self.permutations_spinbox.setRange(100, 100000)
        # self.permutations_spinbox.setSingleStep(100)
        # self.permutations_spinbox.setValue(5000)
        # Tooltip.install(self.permutations_spinbox, "<b>Permutations:</b> Number of random shuffles to build the null distribution. More is better but slower. (5000+ recommended)")
        # graph_layout.addRow("Permutations:", self.permutations_spinbox)

        # b.addWidget(graph_box)
        
        # Regional Analysis Setup
        region_box = QtWidgets.QGroupBox("Regional Analysis Setup")
        region_layout = QtWidgets.QFormLayout(region_box)
        
        # Atlas Template Loader (for Phase Axis)
        atlas_box = QtWidgets.QWidget()
        atlas_layout = QtWidgets.QHBoxLayout(atlas_box)
        atlas_layout.setContentsMargins(0, 0, 0, 0)
        
        # Check if self.btn_load_atlas already exists otherwise re-declare it here exactly as it was.
        self.btn_load_atlas = QtWidgets.QPushButton(get_icon('fa5s.map'), "Load Atlas Template...")
        Tooltip.install(self.btn_load_atlas, "Load the Master Atlas ROI file. Required for defining regions and running Regional Analysis.")
        
        self.atlas_path_label = QtWidgets.QLineEdit()
        self.atlas_path_label.setPlaceholderText("No Atlas Loaded")
        self.atlas_path_label.setReadOnly(True)
        
        atlas_layout.addWidget(self.btn_load_atlas)
        atlas_layout.addWidget(self.atlas_path_label)
        
        param_layout.addRow("Atlas Template:", atlas_box)
        
        # 2. Define Regions Button
        self.btn_define_regions = QtWidgets.QPushButton(get_icon('fa5s.draw-polygon'), "Define Analysis Regions...")
        self.btn_define_regions.setToolTip("Draw sub-regions (e.g. Dorsal, Ventral) on the Atlas for group statistics.")

        # Status label for the region file
        self.region_status_label = QtWidgets.QLabel("No regions defined")
        self.region_status_label.setStyleSheet("color: gray; font-style: italic;")

        region_layout.addRow(self.btn_define_regions, self.region_status_label)

        # 3. Parameters
        self.min_region_cells_spin = QtWidgets.QSpinBox()
        self.min_region_cells_spin.setRange(1, 100)
        self.min_region_cells_spin.setValue(10)
        self.min_region_cells_spin.setToolTip("Minimum number of cells an animal must have in a region to be included.")
        region_layout.addRow("Min Cells / Region / Animal:", self.min_region_cells_spin)

        b.addWidget(region_box)
        
        b.addWidget(comp_box)
        
        # Generate Button
        self.btn_view_group = QtWidgets.QPushButton(get_icon('fa5s.chart-pie'), "Generate Group Visualizations")
        Tooltip.install(self.btn_view_group, "Loads all specified warped ROI and trace files, calculates phases, and generates the Group Scatter and Group Average Map plots.")
        self.btn_view_group.setEnabled(False)
        b.addWidget(self.btn_view_group)
        
        layout.addWidget(box)
        layout.addStretch(1)
    
    def assign_group(self, group_name: str):
        """Assigns the selected files in the tree to a specified group."""
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select one or more files to assign.")
            return
        for item in selected_items:
            item.setText(1, group_name)
            # Optional: Add color coding for clarity
            if group_name == "Control":
                item.setForeground(1, QtGui.QColor('blue'))
            else:
                item.setForeground(1, QtGui.QColor('red'))
    
    def define_regions(self):
        """
        Opens the ROI Drawer to define Dorsal/Middle/Ventral regions on the Atlas.
        """
        # 1. Check if Atlas is loaded
        atlas_path = self.state.atlas_roi_path
        if not atlas_path or not os.path.exists(atlas_path):
            QtWidgets.QMessageBox.warning(self, "No Atlas", "Please load an Atlas Template first.")
            return

        try:
            # 2. Load Atlas Polygons
            with open(atlas_path, 'r') as f:
                atlas_rois = json.load(f)
            
            # 3. Create a Visualization Mask (The "Image" to draw on)
            # We need to find the bounding box of the atlas to size the image
            all_verts = []
            for r in atlas_rois:
                if 'path_vertices' in r:
                    all_verts.extend(r['path_vertices'])
            
            if not all_verts:
                self.mw.log_message("Error: Atlas file contains no vertices.")
                return
                
            all_verts = np.array(all_verts)
            max_x = np.max(all_verts[:, 0])
            max_y = np.max(all_verts[:, 1])
            
            # Create a blank image (plus some padding)
            H, W = int(max_y) + 50, int(max_x) + 50
            bg_image = np.zeros((H, W), dtype=float)
            
            # "Paint" the Atlas Include polygons onto this image so the user can see them
            for r in atlas_rois:
                if r.get('mode') == 'Include':
                    poly = np.array(r['path_vertices'])
                    rr, cc = draw_polygon(poly[:, 1], poly[:, 0], shape=(H, W))
                    bg_image[rr, cc] = 0.3 # Dark gray for the whole SCN
            
            # 4. Check if we already have regions defined
            region_file = atlas_path.replace('_anatomical_roi.json', '_anatomical_regions.json')
            existing_regions = []
            if os.path.exists(region_file):
                try:
                    with open(region_file, 'r') as f:
                        existing_regions = json.load(f)
                except:
                    pass

            # 5. Define Callback to Save Results
            def save_callback(indices, rois_list, refs_list):
                # We only care about the 'rois_list' (the polygons drawn)
                final_list = (rois_list or []) + (refs_list or [])
                
                if not final_list:
                    return

                # Convert to serializable format
                serializable = []
                for r in final_list:
                    verts = r["path_vertices"]
                    if hasattr(verts, 'tolist'): verts = verts.tolist()
                    elif isinstance(verts, np.ndarray): verts = verts.tolist()
                    
                    # Base item
                    item = {
                        "path_vertices": verts,
                        "mode": r["mode"]
                    }
                    
                    # Preserve the Region Tags
                    if "zone_id" in r:
                        item["zone_id"] = r["zone_id"]
                    if "lobe" in r:
                        item["lobe"] = r["lobe"]
                    if "name" in r:
                        item["name"] = r["name"]
                    
                    serializable.append(item)
                
                with open(region_file, 'w') as f:
                    json.dump(serializable, f, indent=4)
                
                self.mw.log_message(f"Saved {len(serializable)} regions to {os.path.basename(region_file)}")
                self.region_status_label.setText(f"{len(serializable)} regions loaded.")
                self.region_status_label.setStyleSheet("color: green; font-weight: bold;")

            # 6. Launch Dialog
            # We pass None for roi_data because we don't have cells here, just the atlas shape
            dlg = ROIDrawerDialog(
                self.mw,
                bg_image,
                roi_data=None, 
                output_basename=None, # We handle save manually
                callback=save_callback,
                vmin=0, vmax=1,
                is_region_mode=True # Enable the tagging popup
            )
            
            # Pre-load existing regions if any
            if existing_regions:
                dlg.rois = existing_regions
                dlg.redraw_finished_rois()
                
            dlg.exec_()

        except Exception as e:
            self.mw.log_message(f"Error launching region definition: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())
    
    def set_reference_data(self):
        if hasattr(self, 'current_grid_def') and hasattr(self, 'current_raw_data'):
            self.state.reference_grid_def = self.current_grid_def
            self.state.reference_raw_data = self.current_raw_data
            
            
            self.btn_set_reference.setText("Reference Set (Active)")
            self.btn_set_reference.setEnabled(False) 
        else:
            self.mw.log_message("Error: No valid data. Run analysis first.")

    def load_atlas_template(self):
            start_dir = self.mw._get_last_dir()
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Master Atlas ROI", start_dir, "JSON files (*.json)"
            )
            if not path:
                return
            self.mw._set_last_dir(path)
            self.state.atlas_roi_path = path
            self.atlas_path_label.setText(os.path.basename(path))
            
            # Quick validation
            with open(path, 'r') as f:
                data = json.load(f)
            
            has_axis = any(r.get('mode') == 'Phase Axis' for r in data)
            if has_axis:
                self.mw.log_message("Atlas loaded. 'Phase Axis' found -> Gradient Analysis enabled.")
            else:
                self.mw.log_message("Atlas loaded. Warning: No 'Phase Axis' found. Gradient Analysis will be skipped.")

    def load_scaffold(self):
        """
        Opens a file dialog to select a scaffold CSV file and initializes
        the AnatomicalNodeScaffold object.
        """
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Anatomical Node Scaffold",
            start_dir,
            "Scaffold files (*.csv)",
        )
        if not path:
            return

        self.mw._set_last_dir(path)
        
        try:
            k_neighbors = self.knn_spinbox.value()
            scaffold = AnatomicalNodeScaffold(path, k_neighbors)
            
            # Store the successfully loaded scaffold in the global state
            self.state.node_scaffold = scaffold
            
            self.scaffold_path_edit.setText(os.path.basename(path))
            self.mw.log_message(
                f"Successfully loaded scaffold with {len(scaffold.nodes)} nodes "
                f"from {os.path.basename(path)}."
            )
            # You could potentially disable the old grid-based controls here
            self.group_grid_res_edit.setEnabled(False)
            self.group_smooth_check.setEnabled(False)

        except Exception as e:
            self.state.node_scaffold = None
            self.scaffold_path_edit.clear()
            self.mw.log_message(f"--- ERROR: Failed to load scaffold ---")
            self.mw.log_message(str(e))
            # Re-enable grid controls on failure
            self.group_grid_res_edit.setEnabled(True)
            self.group_smooth_check.setEnabled(True)

    def run_comparison_analysis(self):
        if self.state.reference_raw_data is None or not hasattr(self, 'current_raw_data'):
            self.mw.log_message("Error: Missing Reference or Experimental data.")
            return

        self.mw.log_message("Running Animal-Level Cluster Permutation Test...")
        self.mw.log_message("Please wait, this may take 20-40 seconds.")
        
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        try:
            # 1. Get Grid Shape
            # grid_def is (calc_x, calc_y, grid_x, grid_y, nx, ny)
            nx = self.state.reference_grid_def[4]
            ny = self.state.reference_grid_def[5]
            grid_shape = (ny, nx)
            period = 24.0

            # 2. Build Per-Animal Matrices
            self.mw.log_message("  -> Collapsing cell data to per-animal means...")
            ref_ids, ref_matrix = build_animal_phase_matrix(self.state.reference_raw_data, grid_shape, period)
            exp_ids, exp_matrix = build_animal_phase_matrix(self.current_raw_data, grid_shape, period)
            self.mw.log_message(f"  -> Found {len(ref_ids)} Control animals and {len(exp_ids)} Experimental animals.")

            # 3. Run the new CBPT
            do_bridge = self.bridge_gaps_check.isChecked()
            if do_bridge:
                self.mw.log_message("  -> Bridging gaps for cluster detection...")

            results = cluster_based_permutation_test_by_animal(
                ref_matrix=ref_matrix,
                exp_matrix=exp_matrix,
                grid_shape=grid_shape,
                period=period,
                n_permutations=10000,
                min_n=2,
                cluster_alpha=0.4,
                bridge_gaps=do_bridge
            )
            
            if results is None:
                self.mw.log_message("Error: No valid overlapping pixels found (Min N=3).")
                QtWidgets.QApplication.restoreOverrideCursor()
                return
            
            # 4. Visualization
            if not hasattr(self.mw, 'diff_tab'):
                self.mw.diff_tab = QtWidgets.QWidget()
                self.mw.vis_tabs.addTab(self.mw.diff_tab, "Diff Map")
            
            fig, _ = add_mpl_to_tab(self.mw.diff_tab)
            
            viewer = GroupDifferenceViewer(
                fig, fig.add_subplot(111),
                results['difference_map'],
                results['significance_mask'],
                results['p_values'],
                self.state.reference_grid_def
            )
            
            self.mw.visualization_widgets[self.mw.diff_tab] = viewer
            self.mw.vis_tabs.setCurrentWidget(self.mw.diff_tab)
            self.mw.log_message("Analysis Complete. Significant clusters displayed.")
            
        except Exception as e:
            self.mw.log_message(f"Statistical Analysis Failed: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def connect_signals(self):
        self.btn_add_group.clicked.connect(self.add_group_files)
        self.btn_remove_group.clicked.connect(self.remove_group_file)
        self.btn_view_group.clicked.connect(self.generate_group_visualizations)
        self.norm_anatomical_radio.toggled.connect(self._on_norm_method_changed)
        self.btn_select_ref_roi.clicked.connect(self._select_reference_roi)
        self.btn_set_reference.clicked.connect(self.set_reference_data)
        self.btn_compare.clicked.connect(self.run_comparison_analysis)
        self.btn_load_atlas.clicked.connect(self.load_atlas_template)
        # self.btn_load_scaffold.clicked.connect(self.load_scaffold)
        
        self.btn_assign_control.clicked.connect(lambda: self.assign_group("Control"))
        self.btn_assign_exp.clicked.connect(lambda: self.assign_group("Experiment"))
        
        self.btn_define_regions.clicked.connect(self.define_regions)
        
        try: self.mw.btn_export_data.clicked.disconnect() 
        except: pass
        self.mw.btn_export_data.clicked.connect(self.export_current_data)
            
    def export_current_data(self):
        """
        Exports data from the currently visible tab/viewer.
        """
        current_tab = self.mw.vis_tabs.currentWidget()
        viewer = self.mw.visualization_widgets.get(current_tab)
        
        if viewer and hasattr(viewer, 'get_export_data'):
            try:
                df, filename = viewer.get_export_data()
                if df is None or df.empty:
                    self.mw.log_message("No data available to export.")
                    return
                
                start_dir = self.mw._get_last_dir()
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export Data", os.path.join(start_dir, filename), "CSV files (*.csv)"
                )
                
                if path:
                    df.to_csv(path, index=False)
                    self.mw.log_message(f"Data exported to {os.path.basename(path)}")
                    self.mw._set_last_dir(path)
            except Exception as e:
                self.mw.log_message(f"Export failed: {e}")
        else:
            self.mw.log_message("The current view does not support data export.")
    
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
                item = QtWidgets.QTreeWidgetItem(self.group_list)
                item.setText(0, f)
                item.setText(1, "Unassigned") # Default group
        self._update_group_view_button()

    def remove_group_file(self):
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            path = item.text(0)
            if path in self.state.group_data_paths:
                self.state.group_data_paths.remove(path)
            
            # Remove from TreeWidget
            (item.parent() or self.group_list.invisibleRootItem()).removeChild(item)
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
        self.mw.log_message("--- Starting Region-Based Analysis ---")
        
        # 1. Validation & Setup
        if not self.state.group_data_paths:
            self.mw.log_message("Error: No group files selected.")
            return

        atlas_path = self.state.atlas_roi_path
        if not atlas_path:
            self.mw.log_message("Error: No Atlas Template loaded.")
            return
            
        region_file = atlas_path.replace('_anatomical_roi.json', '_anatomical_regions.json')
        if not os.path.exists(region_file):
            self.mw.log_message(f"Error: Regions file not found: {os.path.basename(region_file)}")
            return

        # 2. Load and Organize Regions (Zone grouping)
        # We need to group Left/Right polygons into single "Zones"
        try:
            with open(region_file, 'r') as f:
                raw_regions = json.load(f)
            
            # Structure: zones[1] = {'name': 'Dorsal', 'polygons': [Path1, Path2]}
            self.zones = {} 
            
            for r in raw_regions:
                # Skip legacy regions without tags
                if 'zone_id' not in r: 
                    self.mw.log_message("Warning: Found region without Zone ID. Skipping.")
                    continue
                    
                zid = r['zone_id']
                if zid not in self.zones:
                    self.zones[zid] = {'name': r.get('name', f"Zone {zid}"), 'polygons': []}
                
                # Convert to Path object for fast checking
                verts = np.array(r['path_vertices'])
                self.zones[zid]['polygons'].append(Path(verts))
                
            sorted_zone_ids = sorted(self.zones.keys())
            self.mw.log_message(f"Loaded {len(sorted_zone_ids)} anatomical zones.")
            
        except Exception as e:
            self.mw.log_message(f"Error parsing regions: {e}")
            return

        # 3. Load Animal Data (Standard Robust Loader)
        self.mw.log_message("Loading animal data...")
        all_dfs = []
        
        try:
            # Get Group Assignments
            group_map = {}
            root = self.group_list.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                group_map[item.text(0)] = item.text(1) # Key is full path here

            # Load Files
            for roi_path in self.state.group_data_paths:
                base = os.path.basename(roi_path).replace('_roi_warped.csv', '')
                group = group_map.get(roi_path, "Unassigned")
                
                # Robust checks
                if group not in ["Control", "Experiment"]: continue
                
                rhythm_path = roi_path.replace("_roi_warped.csv", "_rhythm_results.csv")
                if not os.path.exists(rhythm_path): continue
                
                # Load coordinates and phases
                coords = np.loadtxt(roi_path, delimiter=",")
                rhythm_df = pd.read_csv(rhythm_path)
                
                # Filter Rhythmic
                mask = rhythm_df['Is_Rhythmic'].astype(bool).values
                if not np.any(mask): continue
                
                # CT Standardization
                phases = rhythm_df['Phase_Hours'][mask].values
                periods = rhythm_df['Period_Hours'][mask].values
                phases_ct = (phases / periods) * 24.0
                
                df = pd.DataFrame({
                    'Animal': base,
                    'Group': group,
                    'X': coords[mask, 0],
                    'Y': coords[mask, 1],
                    'Phase_CT': phases_ct
                })
                all_dfs.append(df)

            if not all_dfs:
                self.mw.log_message("No valid animal data found.")
                return
                
            master_df = pd.concat(all_dfs, ignore_index=True)

            # 4. Global Normalization (Control Mean Reference)
            self.mw.log_message("Normalizing to Control Group Mean...")
            ctrl_phases = master_df[master_df['Group'] == 'Control']['Phase_CT'].values
            
            if len(ctrl_phases) == 0:
                self.mw.log_message("Error: No control cells found.")
                return
                
            # Circmean of controls
            rads = (ctrl_phases / 24.0) * (2 * np.pi)
            ref_rad = circmean(rads)
            ref_phase = (ref_rad / (2 * np.pi)) * 24.0
            
            self.mw.log_message(f"Reference Phase (CT): {ref_phase:.2f}h")
            
            # Apply to all
            master_df['Rel_Phase'] = (master_df['Phase_CT'] - ref_phase + 12.0) % 24.0 - 12.0

        except Exception as e:
            self.mw.log_message(f"Data Loading Error: {e}")
            import traceback
            self.mw.log_message(traceback.format_exc())
            return

        # 5. Map Cells to Zones and Aggregate
        self.mw.log_message("Mapping cells to anatomical zones...")
        
        # Storage: [ {zone_id, name, animals: [ {id, group, mean, n}, ... ]}, ... ]
        self.final_zone_stats = []
        min_cells = self.min_region_cells_spin.value()
        
        unique_animals = master_df['Animal'].unique()
        
        for zid in sorted_zone_ids:
            z_info = self.zones[zid]
            polys = z_info['polygons']
            
            zone_record = {'id': zid, 'name': z_info['name'], 'data': []}
            
            for animal in unique_animals:
                # Get animal cells
                subset = master_df[master_df['Animal'] == animal]
                if subset.empty: continue
                
                # Check point-in-polygon for ALL polygons in this zone (L + R)
                points = subset[['X', 'Y']].values
                mask_in_zone = np.zeros(len(points), dtype=bool)
                
                for poly in polys:
                    mask_in_zone |= poly.contains_points(points)
                
                # Aggregate
                valid_phases = subset.loc[mask_in_zone, 'Rel_Phase'].values
                
                if len(valid_phases) >= min_cells:
                    # Circular Mean of Relative Phases
                    # Since Rel_Phase is [-12, 12], we treat it as circular [-pi, pi]
                    rads = (valid_phases / 24.0) * (2 * np.pi)
                    m_rad = circmean(rads, low=-np.pi, high=np.pi)
                    m_val = (m_rad / (2 * np.pi)) * 24.0
                    
                    zone_record['data'].append({
                        'animal': animal,
                        'group': subset['Group'].iloc[0],
                        'mean': m_val,
                        'n': len(valid_phases)
                    })
            
            self.final_zone_stats.append(zone_record)
            self.mw.log_message(f"  Zone {zid} ({z_info['name']}): {len(zone_record['data'])} animals.")

        self.mw.log_message("Aggregation Complete. Ready for Statistics.")

    def _legacy_generate_graph_analysis(self):
        # Check if we are running in Graph mode or Grid mode
        is_graph_mode = self.state.node_scaffold is not None
        if is_graph_mode:
            self.mw.log_message("--- Running in Graph-Based Analysis Mode ---")
            scaffold = self.state.node_scaffold
            # Update scaffold k-neighbors value from UI in case it changed
            if scaffold.k_neighbors != self.knn_spinbox.value():
                scaffold.k_neighbors = self.knn_spinbox.value()
                scaffold._build_adjacency_graph()
                self.mw.log_message(f"Updated scaffold k-NN to {scaffold.k_neighbors}")
        else:
            self.mw.log_message("--- Running in legacy Grid-Based Analysis Mode ---")
        
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
                
                if not os.path.exists(roi_file) or not os.path.exists(filtered_roi_file):
                     self.mw.log_message("    Error: Missing spatial files. Skipping.")
                     continue
                
                warped_rois = np.atleast_2d(np.loadtxt(roi_file, delimiter=","))
                native_filtered_rois = np.atleast_2d(np.loadtxt(filtered_roi_file, delimiter=","))
                
                phases = None
                period = None
                mask = None

                # --- STRATEGY 1: LOAD LOCKED-IN RESULTS ---
                if os.path.exists(rhythm_file):
                    self.mw.log_message(f"    -> Found saved rhythm results. Loading...")
                    try:
                        rhythm_df = pd.read_csv(rhythm_file)
                        if len(rhythm_df) != len(warped_rois):
                            self.mw.log_message("    WARNING: Rhythm file length mismatch. Falling back.")
                        else:
                            phases = rhythm_df['Phase_Hours'].values
                            period = rhythm_df['Period_Hours'].iloc[0]
                            mask = rhythm_df['Is_Rhythmic'].astype(bool).values
                    except Exception as e:
                        self.mw.log_message(f"    ERROR reading results file: {e}. Falling back.")

                # --- STRATEGY 2: RE-CALCULATE ---
                if phases is None:
                    self.mw.log_message("    -> Re-calculating using current global settings...")
                    
                    if not os.path.exists(traces_file) or not os.path.exists(unfiltered_roi_file):
                         self.mw.log_message("    Error: Missing trace data. Skipping.")
                         continue

                    traces = np.loadtxt(traces_file, delimiter=",")
                    unfiltered_rois = np.loadtxt(unfiltered_roi_file, delimiter=",")
                    
                    indices = []
                    for point in native_filtered_rois:
                        diff = np.sum((unfiltered_rois - point)**2, axis=1)
                        idx = np.argmin(diff)
                        if diff[idx] < 1e-9: indices.append(idx)
                    
                    if len(indices) != len(native_filtered_rois):
                        self.mw.log_message("    Error: Could not match filtered ROIs. Skipping.")
                        continue
                        
                    filtered_traces_data = traces[:, np.concatenate(([0], np.array(indices) + 1))]
                    
                    if "Cosinor" in method:
                        mpf = phase_args["minutes_per_frame"]
                        time_points_hours = filtered_traces_data[:, 0] * (mpf / 60.0)
                        _, period, _ = calculate_phases_fft(filtered_traces_data, **phase_args)
                        
                        phases_list, p_values, r_squareds = [], [], []
                        trend_win = phase_args.get("detrend_window_hours", RHYTHM_TREND_WINDOW_HOURS)
                        med_win = compute_median_window_frames(mpf, trend_win, T=filtered_traces_data.shape[0])
                        
                        for col in range(1, filtered_traces_data.shape[1]):
                            raw_trace = filtered_traces_data[:, col]
                            detr = preprocess_for_rhythmicity(raw_trace, method="running_median", median_window_frames=med_win)
                            res = csn.cosinor_analysis(detr, time_points_hours, period)
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

                # Standardize to Circadian Time (CT) immediately
                # Convert raw phases to a 24-hour scale
                phases_ct = (phases / period) * 24.0

                animal_df = pd.DataFrame({
                    'Source_Animal': os.path.basename(roi_file).replace('_roi_warped.csv', ''),
                    'Warped_X': warped_rois[:, 0],
                    'Warped_Y': warped_rois[:, 1],
                    'Phase_Hours_CT': phases_ct, # Store the CT-scaled phase
                    'Period_Hours': np.full(len(phases), 24.0) # Period is now fixed at 24.0
                })
                all_dfs.append(animal_df)

            if not all_dfs: 
                self.mw.log_message("No valid group data generated.")
                return
            
            group_scatter_df = pd.concat(all_dfs, ignore_index=True)
            
            # Global Normalization Step
            self.mw.log_message("\nNormalizing all animals to a common reference...")

            # 1. Get group assignments for all animals
            group_map = {}
            root = self.group_list.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                path = item.text(0)
                group = item.text(1)
                animal_id = os.path.basename(path).replace('_roi_warped.csv', '')
                group_map[animal_id] = group

            # 2. Calculate the single reference phase from the Control group's CT data
            control_animal_ids = [aid for aid, grp in group_map.items() if grp == "Control"]
            if not control_animal_ids:
                raise ValueError("No animals assigned to 'Control' group for normalization.")
                
            control_phases_ct = group_scatter_df[group_scatter_df['Source_Animal'].isin(control_animal_ids)]['Phase_Hours_CT'].values

            # All calculations are now consistently on a 24h clock
            control_rads = (control_phases_ct / 24.0) * (2 * np.pi)
            control_mean_rad = circmean(control_rads)
            grand_control_mean_h = (control_mean_rad / (2 * np.pi)) * 24.0

            self.mw.log_message(f"Grand Control Mean Phase (Reference CT): {grand_control_mean_h:.2f}h")

            # 3. Apply this single reference to ALL cells from ALL animals
            raw_phases_ct = group_scatter_df['Phase_Hours_CT'].values

            # The math is now simple and consistent because period is always 24.0
            relative_phases = (raw_phases_ct - grand_control_mean_h + 12.0) % 24.0 - 12.0

            group_scatter_df['Relative_Phase_Hours'] = relative_phases
            
            if group_scatter_df.empty:
                self.mw.log_message("No rhythmic cells found in any group file.")
                return

            period = group_scatter_df['Period_Hours'].mean()
            
            self.current_period = period
            
            grid_res = int(self.group_grid_res_edit.text())
            do_smooth = self.group_smooth_check.isChecked()
            
            x_min, x_max = group_scatter_df['Warped_X'].min(), group_scatter_df['Warped_X'].max()
            y_min, y_max = group_scatter_df['Warped_Y'].min(), group_scatter_df['Warped_Y'].max()
            
            # CONDITIONAL ANALYSIS FORK
            if is_graph_mode:
                # Graph CBPT logic
                self.mw.log_message("\nAssigning cells to anatomical nodes...")
                try:
                    search_radius = self.search_radius_spinbox.value()
                    
                    # Use the scaffold's pre-built KDTree to find the nearest node for every cell
                    cell_coords = group_scatter_df[['Warped_X', 'Warped_Y']].values
                    distances, node_indices = scaffold.kdtree.query(cell_coords)
                    
                    # Filter out cells that are too far from any node
                    valid_assignment_mask = distances <= search_radius
                    
                    group_scatter_df['Node_ID'] = node_indices
                    assigned_cells_df = group_scatter_df[valid_assignment_mask].copy()
                    
                    num_assigned = len(assigned_cells_df)
                    num_total = len(valid_assignment_mask)
                    self.mw.log_message(
                        f"Assigned {num_assigned} / {num_total} rhythmic cells to nodes "
                        f"(within {search_radius}μm radius)."
                    )

                except Exception as e:
                    self.mw.log_message(f"Error during cell-to-node assignment: {e}")
                    return

                # Per-Animal, Per-Node Aggregation
                self.mw.log_message("\nAggregating data to per-animal, per-node means...")
                try:
                    min_cells_per_node = self.min_cells_spinbox.value()
                    
                    # Group by animal and node to count cells
                    cell_counts = assigned_cells_df.groupby(['Source_Animal', 'Node_ID']).size()
                    
                    # Define a function for circular mean that handles the cell count threshold
                    # Note: We use the period from the loaded data for accuracy
                    period = assigned_cells_df['Period_Hours'].mean() # Should be 24.0 for CT
                    def circular_mean_with_qc(phases):
                        if len(phases) >= min_cells_per_node:
                            radians = (phases / period) * (2 * np.pi)
                            mean_rad = circmean(radians)
                            return (mean_rad / (2 * np.pi)) * period
                        else:
                            return np.nan

                    # Apply this function to get our per-animal, per-node mean phases
                    aggregated_phases = assigned_cells_df.groupby(['Source_Animal', 'Node_ID'])['Relative_Phase_Hours'].apply(circular_mean_with_qc)
                    
                    # Convert the result into a clean DataFrame
                    aggregated_df = aggregated_phases.reset_index()
                    aggregated_df = aggregated_df.rename(columns={'Relative_Phase_Hours': 'Mean_Phase_Hours'})
                    
                    # Pivot the table to get the desired [animal, node] shape
                    phase_matrix_df = aggregated_df.pivot_table(
                        index='Source_Animal', columns='Node_ID', values='Mean_Phase_Hours'
                    )
                    
                    # Ensure all nodes from the scaffold are present as columns
                    all_node_ids = np.arange(len(scaffold.nodes))
                    phase_matrix_df = phase_matrix_df.reindex(columns=all_node_ids, fill_value=np.nan)

                    # Get the final data in the required format
                    self.animal_ids = phase_matrix_df.index.to_list()
                    self.phase_matrix = phase_matrix_df.to_numpy() # Shape: (n_animals, n_nodes)
                    
                    self.mw.log_message("Aggregation complete.")
                    self.mw.log_message(f"Created phase matrix of shape: {self.phase_matrix.shape}")
                    
                    # Store an optional n_cells matrix for diagnostics
                    n_cells_df = assigned_cells_df.pivot_table(
                        index='Source_Animal', columns='Node_ID', aggfunc='size', fill_value=0
                    )
                    n_cells_df = n_cells_df.reindex(columns=all_node_ids, fill_value=0)
                    self.n_cells_matrix = n_cells_df.to_numpy()
                    
                    self.mw.log_message("\nData is now ready for the statistical engine.")
                    self.mw.log_message("Next step: Implement gui/statistics.py/run_graph_cbpt and visualization.")

                    # --- Final Step: Run Statistics and Visualize ---
                    self.mw.log_message("\nRunning statistical analysis...")
                    QtWidgets.QApplication.processEvents()
                    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

                    try:
                        # Build group_labels from UI assignments
                        group_map = {}
                        root = self.group_list.invisibleRootItem()
                        for i in range(root.childCount()):
                            item = root.child(i)
                            path = item.text(0)
                            group = item.text(1)
                            animal_id = os.path.basename(path).replace('_roi_warped.csv', '')
                            group_map[animal_id] = group

                        if "Control" not in group_map.values() or "Experiment" not in group_map.values():
                            raise ValueError("Both 'Control' and 'Experiment' groups must be assigned.")

                        # Create the group_labels array in the same order as self.animal_ids
                        group_labels = np.array([
                            0 if group_map.get(animal_id) == "Control" else 1
                            for animal_id in self.animal_ids
                        ])

                        if len(np.unique(group_labels)) != 2:
                            raise ValueError("Error in group assignment. Check that both groups have valid data.")
                        
                        # Get parameters from the UI
                        min_animals = self.min_animals_spinbox.value()
                        cluster_alpha = self.cluster_alpha_spinbox.value()
                        n_perms = self.permutations_spinbox.value()
                        
                        self.mw.log_message(f"  Control animals: {np.sum(group_labels == 0)}")
                        self.mw.log_message(f"  Experiment animals: {np.sum(group_labels == 1)}")
                        
                        results = run_graph_cbpt(
                            phase_matrix=self.phase_matrix,
                            group_labels=group_labels,
                            scaffold=scaffold,
                            min_animals_per_group=min_animals,
                            cluster_alpha=cluster_alpha,
                            n_permutations=n_perms
                        )
                        
                        self.mw.log_message("Statistical analysis complete.")
                        
                        # --- Visualization ---
                        if not hasattr(self.mw, 'diff_tab'):
                            self.mw.diff_tab = QtWidgets.QWidget()
                            self.mw.vis_tabs.addTab(self.mw.diff_tab, "Graph Diff Map")

                        fig, canvas = add_mpl_to_tab(self.mw.diff_tab)
                        viewer = GraphDifferenceViewer(fig, fig.add_subplot(111), scaffold, results)
                        
                        # Store the viewer for exporting, etc.
                        self.mw.visualization_widgets[self.mw.diff_tab] = viewer
                        
                        # Show the new tab
                        for tab in single_animal_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
                        self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.diff_tab), True)
                        self.mw.vis_tabs.setCurrentWidget(self.mw.diff_tab)
                        self.mw.btn_export_data.setEnabled(True)
                        
                    except Exception as e:
                        import traceback
                        self.mw.log_message(f"--- STATISTICAL ANALYSIS FAILED ---")
                        self.mw.log_message(f"{e}")
                        self.mw.log_message(traceback.format_exc())
                    finally:
                        QtWidgets.QApplication.restoreOverrideCursor()

                except Exception as e:
                    import traceback
                    self.mw.log_message(f"Error during data aggregation: {e}")
                    self.mw.log_message(traceback.format_exc())
                    return

            else:
            
                # --- GRID LOGIC ---
                if self.state.reference_grid_def is not None:
                    self.mw.log_message("    -> Using Locked Reference Grid.")
                    calc_x_bins, calc_y_bins, grid_x_bins, grid_y_bins, n_bins_x, n_bins_y = self.state.reference_grid_def
                else:
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
                    
                    calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
                    calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)
                    
                    self.current_grid_def = (calc_x_bins, calc_y_bins, grid_x_bins, grid_y_bins, n_bins_x, n_bins_y)

                group_scatter_df['Grid_X_Index'] = pd.cut(group_scatter_df['Warped_X'], bins=calc_x_bins, labels=False, include_lowest=True)
                group_scatter_df['Grid_Y_Index'] = pd.cut(group_scatter_df['Warped_Y'], bins=calc_y_bins, labels=False, include_lowest=True)
                
                # Drop points outside the grid (only relevant if using Reference Grid and data is outliers)
                group_scatter_df.dropna(subset=['Grid_X_Index', 'Grid_Y_Index'], inplace=True)
                group_scatter_df['Grid_X_Index'] = group_scatter_df['Grid_X_Index'].astype(int)
                group_scatter_df['Grid_Y_Index'] = group_scatter_df['Grid_Y_Index'].astype(int)

                # --- Capture Raw Data for Stats ---
                self.current_raw_data = {}
                grouped = group_scatter_df.groupby(['Grid_Y_Index', 'Grid_X_Index'])
                for (gy, gx), group in grouped:
                    self.current_raw_data[(gy, gx)] = {
                        'phases': group['Relative_Phase_Hours'].values,
                        'animals': group['Source_Animal'].values
                    }

                def circmean_phase(series):
                    rad = (series / (period / 2.0)) * np.pi
                    mean_rad = circmean(rad, low=-np.pi, high=np.pi)
                    return (mean_rad / np.pi) * (period / 2.0)
                
                group_binned_df = group_scatter_df.groupby(['Grid_X_Index', 'Grid_Y_Index'])['Relative_Phase_Hours'].apply(circmean_phase).reset_index()
                
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
                
                # Enable Reference Setting
                self.btn_set_reference.setEnabled(True)
                
                if self.state.reference_raw_data is not None:
                    self.btn_compare.setEnabled(True)
            
                # Gradient Analysis (Anatomical Lobe Assignment, Phase vs. Axis)
                if self.state.atlas_roi_path and os.path.exists(self.state.atlas_roi_path):
                    try:
                        with open(self.state.atlas_roi_path, 'r') as f:
                            atlas_data = json.load(f)
                        
                        # 1. Find Phase Axes
                        axis_rois = [r for r in atlas_data if r.get('mode') == 'Phase Axis']
                        
                        if not axis_rois:
                            self.mw.log_message("Atlas loaded, but no 'Phase Axis' found. Gradient skipped.")
                        else:
                            self.mw.log_message(f"Running Gradient Analysis ({len(axis_rois)} Axes)...")
                            
                            # Sort Axes Left-to-Right
                            axes_polys = [] # List of {'poly': array, 'cx': float}
                            for r in axis_rois:
                                p = np.array(r['path_vertices'])
                                axes_polys.append({'poly': p, 'cx': np.mean(p[:, 0])})
                            axes_polys.sort(key=lambda x: x['cx'])
                            
                            # 2. Find Anatomical Regions (Green Outlines)
                            # We need these to assign cells to the correct axis
                            anat_rois = [r for r in atlas_data if r.get('mode') == 'Include']
                            anat_polys = []
                            for r in anat_rois:
                                p = np.array(r['path_vertices'])
                                anat_polys.append({'path': Path(p), 'cx': np.mean(p[:, 0]), 'poly': p})
                            anat_polys.sort(key=lambda x: x['cx']) # Sort Left-to-Right
                            
                            # 3. Pair Axes with Regions
                            # Scenario A: 2 Axes, >=2 Regions. Pair Left-Left, Right-Right.
                            # Scenario B: 1 Axis. Use it for everything.
                            
                            gradient_data = []
                            animals = group_scatter_df['Source_Animal'].unique()
                            
                            for animal in animals:
                                subset = group_scatter_df[group_scatter_df['Source_Animal'] == animal]
                                points = subset[['Warped_X', 'Warped_Y']].values
                                phases = subset['Relative_Phase_Hours'].values
                                
                                s_final = []
                                p_final = []
                                
                                # --- DUAL LOBE LOGIC ---
                                if len(axes_polys) >= 2 and len(anat_polys) >= 2:
                                    # Assume index 0 is Left, index 1 is Right for both lists
                                    axis_L, axis_R = axes_polys[0]['poly'], axes_polys[1]['poly']
                                    path_L, path_R = anat_polys[0]['path'], anat_polys[1]['path']
                                    
                                    # Masking
                                    is_in_L = path_L.contains_points(points)
                                    is_in_R = path_R.contains_points(points)
                                    
                                    # Handle Edge Cases (Outside both or Inside both?)
                                    # Assign to whichever centroid is closer
                                    cx_L, cx_R = anat_polys[0]['cx'], anat_polys[1]['cx']
                                    dist_L = np.abs(points[:, 0] - cx_L)
                                    dist_R = np.abs(points[:, 0] - cx_R)
                                    
                                    # Final Decision Mask
                                    # If strictly in one, use it. If in neither/both, use distance.
                                    use_L = (is_in_L & ~is_in_R) | ((~is_in_L & ~is_in_R) & (dist_L < dist_R))
                                    use_R = ~use_L # The rest go Right
                                    
                                    if np.any(use_L):
                                        s_L = project_points_to_polyline(points[use_L], axis_L)
                                        s_final.extend(s_L)
                                        p_final.extend(phases[use_L])
                                        
                                    if np.any(use_R):
                                        s_R = project_points_to_polyline(points[use_R], axis_R)
                                        s_final.extend(s_R)
                                        p_final.extend(phases[use_R])
                                        
                                # --- SINGLE AXIS LOGIC ---
                                elif len(axes_polys) == 1:
                                    axis = axes_polys[0]['poly']
                                    s_vals = project_points_to_polyline(points, axis)
                                    s_final.extend(s_vals)
                                    p_final.extend(phases)
                                    
                                else:
                                    self.mw.log_message("Warning: Mismatch between Axes and Anatomical Regions. Cannot assign lobes.")
                                    continue

                                # Convert to numpy
                                s_final = np.array(s_final)
                                p_final = np.array(p_final)
                                
                                if len(s_final) == 0: continue
                                
                                # Binning Logic (Same as before)
                                bins = np.linspace(0, 1, 11)
                                bin_centers = (bins[:-1] + bins[1:]) / 2
                                
                                binned_phases = []
                                for k in range(len(bins)-1):
                                    mask = (s_final >= bins[k]) & (s_final < bins[k+1])
                                    if np.sum(mask) > 0:
                                        p_bin = p_final[mask]
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
                            
                            # Create Tab
                            if not hasattr(self.mw, 'grad_tab'):
                                self.mw.grad_tab = QtWidgets.QWidget()
                                self.mw.vis_tabs.addTab(self.mw.grad_tab, "Gradient")
                                
                            from gui.viewers import PhaseGradientViewer
                            fig, _ = add_mpl_to_tab(self.mw.grad_tab)
                            viewer_g = PhaseGradientViewer(fig, fig.add_subplot(111), gradient_data)
                            self.mw.visualization_widgets[self.mw.grad_tab] = viewer_g
                            self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(self.mw.grad_tab), True)
                            self.mw.log_message("Gradient Analysis Complete.")

                    except Exception as e:
                        self.mw.log_message(f"Gradient Analysis Failed: {e}")
                        import traceback
                        self.mw.log_message(traceback.format_exc())
        
        except Exception as e:
            import traceback
            self.mw.log_message(f"Error generating group visualizations: {e}")
            self.mw.log_message(traceback.format_exc())
            for tab in group_tabs: self.mw.vis_tabs.setTabEnabled(self.mw.vis_tabs.indexOf(tab), False)
