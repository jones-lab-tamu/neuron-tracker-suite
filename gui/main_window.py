import os
import sys
import json
import numpy as np
from PyQt5 import QtWidgets, QtCore

from gui.state import AnalysisState
from gui.utils import Tooltip, clear_layout
from gui.theme import get_icon
from gui.panels.single_animal import SingleAnimalPanel
from gui.panels.atlas_registration import AtlasRegistrationPanel
from gui.panels.apply_warp import ApplyWarpPanel
from gui.panels.group_view import GroupViewPanel
from gui.viewers import RegionResultViewer

class MainWindow(QtWidgets.QMainWindow):
    """
    PyQt main window implementing the full original workflow.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuron Analysis Workspace")
        self.resize(1400, 900)

        # Centralized state object
        self.state = AnalysisState()

        self.visualization_widgets = {}

        self.workflow_state = {
            "has_input": False,
            "has_results": False,
            "has_anatomical_roi": False,
            "has_warp": False,
            "has_group_data": False,
        }

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        
        # Initialize panels
        self.single_panel = SingleAnimalPanel(self)
        self.register_panel = AtlasRegistrationPanel(self)
        self.apply_panel = ApplyWarpPanel(self)
        self.group_panel = GroupViewPanel(self)
        
        self.mode_stack.addWidget(self.single_panel)
        self.mode_stack.addWidget(self.register_panel)
        self.mode_stack.addWidget(self.apply_panel)
        self.mode_stack.addWidget(self.group_panel)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Main Splitter (Left: Controls, Right: Vis)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter, 1)
        
        # --- Left Side: Navigation & Controls ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(left_widget)
        
        # 1. Navigation List (Replaces Workflow & Active Panel)
        self._build_navigation_section(left_layout)
        
        # 2. Scrollable Control Area (For specific panel content)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_layout.addWidget(scroll, 1) # Takes remaining space
        
        self.ctrl_container = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl_container)
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.ctrl_container)
        
        # Stacked Widget for Panels
        self.mode_stack = QtWidgets.QStackedWidget()
        self.ctrl_layout.addWidget(self.mode_stack)
        
        # Execution Box (Shared controls) - Kept in scroll area or pinned to bottom of left?
        # Let's pin it to the bottom of the scroll content for now.
        exec_box = QtWidgets.QGroupBox("Global Actions")
        exec_layout = QtWidgets.QVBoxLayout(exec_box)
        
        hbox_run = QtWidgets.QHBoxLayout()
        self.btn_run_analysis = QtWidgets.QPushButton(get_icon('fa5s.play'), "Run Analysis")
        Tooltip.install(self.btn_run_analysis, "Run full pipeline on loaded movie.")
        self.btn_run_analysis.setEnabled(False)
        
        self.btn_load_results = QtWidgets.QPushButton(get_icon('fa5s.folder-open'), "Load Results")
        self.btn_load_results.setEnabled(False)
        hbox_run.addWidget(self.btn_run_analysis)
        hbox_run.addWidget(self.btn_load_results)
        
        hbox_export = QtWidgets.QHBoxLayout()
        self.btn_export_data = QtWidgets.QPushButton(get_icon('fa5s.file-csv'), "Export Data")
        self.btn_export_data.setEnabled(False)
        
        self.btn_export_plot = QtWidgets.QPushButton(get_icon('fa5s.image'), "Export Plot")
        self.btn_export_plot.setEnabled(False)
        hbox_export.addWidget(self.btn_export_data)
        hbox_export.addWidget(self.btn_export_plot)
        
        exec_layout.addLayout(hbox_run)
        exec_layout.addLayout(hbox_export)
        self.ctrl_layout.addWidget(exec_box)
        self.ctrl_layout.addStretch(1)

        # --- Right Side: Visualizations ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(right_widget)
        
        self.vis_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.vis_tabs)
        
        # --- Bottom: Log and Progress ---
        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5) # Slim progress bar
        bottom_layout.addWidget(self.progress_bar)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100) # Limit log height
        self.log_text.setPlaceholderText("Execution log...")
        bottom_layout.addWidget(self.log_text)
        
        main_layout.addWidget(bottom_widget, 0)
        
        # Splitter Ratios
        self.splitter.setStretchFactor(0, 1) # Left
        self.splitter.setStretchFactor(1, 3) # Right
        
        self._build_vis_tabs()

    def _build_toolbar(self):
        toolbar = QtWidgets.QToolBar("Main Toolbar")
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        open_act = QtWidgets.QAction(get_icon('fa5s.folder-open'), "Open Project", self)
        open_act.triggered.connect(self.load_project)
        toolbar.addAction(open_act)
        
        save_act = QtWidgets.QAction(get_icon('fa5s.save'), "Save Project", self)
        save_act.triggered.connect(self.save_project)
        toolbar.addAction(save_act)

    def _build_vis_tabs(self):
        self.heatmap_tab = QtWidgets.QWidget()
        self.com_tab = QtWidgets.QWidget()
        self.traj_tab = QtWidgets.QWidget()
        self.phase_tab = QtWidgets.QWidget()
        self.interp_tab = QtWidgets.QWidget()
        self.group_scatter_tab = QtWidgets.QWidget()
        self.group_avg_tab = QtWidgets.QWidget()
        
        tabs_to_add = [
            (self.heatmap_tab, "Heatmap"),
            (self.com_tab, "CoM"),
            (self.traj_tab, "Trajectories"),
            (self.phase_tab, "Phase Map"),
            (self.interp_tab, "Interp Map"),
            (self.group_scatter_tab, "Grp Scatter"),
            (self.group_avg_tab, "Grp Avg Map"),
        ]
        
        for tab, name in tabs_to_add:
            self.vis_tabs.addTab(tab, name)
            layout = QtWidgets.QVBoxLayout(tab)
            label = QtWidgets.QLabel(
                f"{name} will appear here after analysis."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
        
        for i in range(self.vis_tabs.count()):
            self.vis_tabs.setTabEnabled(i, False)

        self.vis_tabs.currentChanged.connect(lambda index: self.update_export_buttons_state())

    def _build_navigation_section(self, parent_layout):
        """Builds the navigation list acting as both workflow guide and panel switcher."""
        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setFixedHeight(120) # Compact fixed height
        self.nav_list.setIconSize(QtCore.QSize(24, 24))
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #ffffff;
                color: #007acc;
                border-left: 4px solid #007acc;
            }
        """)
        
        items = [
            ("single", "1. Single Animal Analysis", "fa5s.paw"),
            ("register", "2. Atlas Registration", "fa5s.map"),
            ("apply_warp", "3. Apply Warp", "fa5s.dna"),
            ("group_view", "4. Group Analysis", "fa5s.users"),
        ]
        
        for key, text, icon_name in items:
            item = QtWidgets.QListWidgetItem(get_icon(icon_name), text)
            item.setData(QtCore.Qt.UserRole, key)
            self.nav_list.addItem(item)
            
        self.nav_list.currentItemChanged.connect(self._on_nav_item_changed)
        parent_layout.addWidget(self.nav_list)

    def update_export_buttons_state(self):
        """
        Updates the enabled state of the Export Data button based on the current context.
        Checks if the active tab has a viewer with exportable data.
        """
        item = self.nav_list.currentItem()
        if not item:
            self.btn_export_data.setEnabled(False)
            return

        mode_key = item.data(QtCore.Qt.UserRole)
        
        # Only Group Analysis uses generic viewer export for now
        if mode_key == "group_view":
            current_tab = self.vis_tabs.currentWidget()
            exportable = False
            
            try:
                # 1. Special Case: Region Stats Tab
                # The viewer is a child widget, not registered in visualization_widgets in the same way?
                # Actually group_view registers it, but checking findChild is more robust for composite widgets.
                if current_tab == getattr(self, 'region_tab', None):
                    viewer = current_tab.findChild(RegionResultViewer)
                    if viewer:
                        data, _ = viewer.get_export_data()
                        if data is not None and not data.empty:
                            exportable = True
                else:
                    # 2. Standard Case: Registered Viewers
                    viewer = self.visualization_widgets.get(current_tab)
                    if viewer and hasattr(viewer, 'get_export_data'):
                        res = viewer.get_export_data()
                        # Handle (df, name) tuple or just df
                        if isinstance(res, tuple):
                            data = res[0]
                        else:
                            data = res
                            
                        if data is not None and not data.empty:
                            exportable = True
            except Exception as e:
                self.log_message(f"Export Data check failed for tab {current_tab}: {e}")
                exportable = False
            
            self.btn_export_data.setEnabled(exportable)
        else:
            self.btn_export_data.setEnabled(False)

    def _on_nav_item_changed(self, current, previous):
        if not current:
            return
        mode_key = current.data(QtCore.Qt.UserRole)
        self._switch_mode(mode_key)
        self.update_export_buttons_state()

    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QtWidgets.QAction("&Open Project...", self)
        open_action.triggered.connect(self.load_project)
        file_menu.addAction(open_action)

        save_action = QtWidgets.QAction("&Save Project", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        save_as_action = QtWidgets.QAction("Save Project &As...", self)
        save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        file_menu.addAction(save_as_action)

    def _switch_mode(self, mode_name: str):
        mode_map = {
            "single": 0, "register": 1, "apply_warp": 2, "group_view": 3,
        }
        idx = mode_map.get(mode_name)
        if idx is not None:
            self.mode_stack.setCurrentIndex(idx)
            # Ensure list selection matches (in case called programmatically)
            for i in range(self.nav_list.count()):
                item = self.nav_list.item(i)
                if item.data(QtCore.Qt.UserRole) == mode_name:
                    self.nav_list.setCurrentItem(item)
                    break
            self.log_message(f"Switched to '{mode_name}' panel.")
        else:
            self.log_message(f"Error: Unknown mode '{mode_name}'")

    def _set_mode_enabled(self, mode_key, enabled: bool):
        # In the list widget, we can disable items or just visually dim them.
        # For now, let's just keep them enabled but maybe change icon color if we wanted.
        # Standard QListWidgetItem doesn't have setEnabled in a way that blocks selection easily without flags.
        for i in range(self.nav_list.count()):
            item = self.nav_list.item(i)
            if item.data(QtCore.Qt.UserRole) == mode_key:
                if enabled:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                break

    def _mark_step_ready(self, mode_key):
        for i in range(self.nav_list.count()):
            item = self.nav_list.item(i)
            if item.data(QtCore.Qt.UserRole) == mode_key:
                # Add a checkmark or change color to indicate readiness
                item.setIcon(get_icon('fa5s.check-circle', color='#2e7d32')) # Green check
                break

    def log_message(self, text: str):
        self.log_text.appendPlainText(text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _get_last_dir(self):
        settings = QtCore.QSettings()
        return settings.value("last_dir", "")

    def _set_last_dir(self, path):
        if not path:
            return
        settings = QtCore.QSettings()
        directory = os.path.dirname(path) if os.path.isfile(path) else path
        settings.setValue("last_dir", directory)

    def _reset_state(self):
        self.log_message("Resetting workspace for new analysis...")
        self.state.reset()
        
        # Reset panel-specific states
        if hasattr(self, 'single_panel'):
            self.single_panel.reset_state()
            
        self.visualization_widgets.clear()
        for i in range(self.vis_tabs.count()):
            tab = self.vis_tabs.widget(i)
            layout = tab.layout()
            if layout is not None:
                clear_layout(layout)
            else:
                layout = QtWidgets.QVBoxLayout(tab)
            label = QtWidgets.QLabel(
                f"{self.vis_tabs.tabText(i)} will appear here after analysis."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
            self.vis_tabs.setTabEnabled(i, False)
            
        self.btn_run_analysis.setEnabled(False)
        self.btn_load_results.setEnabled(False)
        self.btn_export_plot.setEnabled(False)
        self.btn_export_data.setEnabled(False)
        self.progress_bar.setValue(0)

    def update_workflow_from_files(self):
        basename = self.state.output_basename
        if not basename:
            return
        input_present = bool(self.state.input_movie_path)
        if input_present:
            self.workflow_state["has_input"] = True
            self._mark_step_ready("single")
            self.btn_run_analysis.setEnabled(True)
        
        traces_path = f"{basename}_traces.csv"
        roi_path = f"{basename}_roi.csv"
        traj_path = f"{basename}_trajectories.npy"
        has_traces = os.path.exists(traces_path)
        has_roi = os.path.exists(roi_path)
        has_traj = os.path.exists(traj_path)
        
        if hasattr(self, 'single_panel'):
            self.single_panel.status_traces_label.setText(f"Traces: {'found' if has_traces else 'missing'}")
            self.single_panel.status_roi_label.setText(f"ROI: {'found' if has_roi else 'missing'}")
            self.single_panel.status_traj_label.setText(f"Trajectories: {'found' if has_traj else 'missing'}")
        
        has_results = has_traces and has_roi and has_traj
        self.workflow_state["has_results"] = has_results
        self.btn_load_results.setEnabled(has_results)
        
        if has_results:
            self._mark_step_ready("single")
            self._set_mode_enabled("register", True)
            if hasattr(self, 'single_panel'):
                self.single_panel.btn_define_roi.setEnabled(True)
                self.single_panel.btn_regen_phase.setEnabled(True)
                self.btn_export_plot.setEnabled(True)
                
        anatomical_roi_path = f"{basename}_anatomical_roi.json"
        has_anatomical_roi = os.path.exists(anatomical_roi_path)
        self.workflow_state["has_anatomical_roi"] = has_anatomical_roi
        if has_anatomical_roi:
            self._mark_step_ready("register")
            self._set_mode_enabled("register", True)
            self._set_mode_enabled("apply_warp", True)
            
        dirname = os.path.dirname(basename) or "."
        warp_files = [f for f in os.listdir(dirname) if f.endswith("_warp_parameters.json")]
        warped_roi_files = [f for f in os.listdir(dirname) if f.endswith("_roi_warped.csv")]
        has_warp = len(warp_files) > 0
        has_group = len(warped_roi_files) > 0
        self.workflow_state["has_warp"] = has_warp
        self.workflow_state["has_group_data"] = has_group
        
        if has_warp:
            self._mark_step_ready("apply_warp")
            self._set_mode_enabled("apply_warp", True)
        if has_group:
            self._mark_step_ready("group_view")
            self._set_mode_enabled("group_view", True)
            if hasattr(self, 'group_panel'):
                self.group_panel.btn_view_group.setEnabled(True)

    def save_project(self, save_as=False):
        project_path = self.state.project_path
        if not project_path or save_as:
            start_dir = self._get_last_dir()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Project", start_dir, "Neuron Tracker Project (*.ntp)"
            )
            if not path:
                return
            project_path = path
        
        try:
            self._sync_state_from_ui()
            state_to_save = {
                'atlas_roi_path': self.state.atlas_roi_path,
                'target_roi_paths': self.state.target_roi_paths,
                'warp_param_paths': self.state.warp_param_paths,
                'group_data_paths': self.state.group_data_paths,
            }
            with open(project_path, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            
            self._set_last_dir(project_path)
            self.state.project_path = project_path
            self.log_message(f"Project saved to {os.path.basename(self.state.project_path)}")
            self.setWindowTitle(f"{os.path.basename(self.state.project_path)} - Neuron Analysis Workspace")

        except Exception as e:
            self.log_message(f"Error saving project: {e}")

    def load_project(self):
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", start_dir, "Neuron Tracker Project (*.ntp)"
        )
        if not path:
            return
        
        self._set_last_dir(path)
        self._reset_state()
        
        try:
            with open(path, 'r') as f:
                loaded_state = json.load(f)
            
            self.state.project_path = path
            self.state.atlas_roi_path = loaded_state.get('atlas_roi_path', "")
            self.state.target_roi_paths = loaded_state.get('target_roi_paths', [])
            self.state.warp_param_paths = loaded_state.get('warp_param_paths', [])
            self.state.group_data_paths = loaded_state.get('group_data_paths', [])
            
            self.log_message(f"Loaded project: {os.path.basename(path)}")
            self._update_ui_from_state()
            self.setWindowTitle(f"{os.path.basename(path)} - Neuron Analysis Workspace")

        except Exception as e:
            self.log_message(f"Error loading project: {e}")

    def _update_ui_from_state(self):
        if hasattr(self, 'register_panel'):
            self.register_panel.atlas_path_edit.setText(self.state.atlas_roi_path)
            self.register_panel.target_list.clear()
            self.register_panel.target_list.addItems(self.state.target_roi_paths)
            self.register_panel._update_reg_button_state()
            
        if hasattr(self, 'apply_panel'):
            self.apply_panel.warp_list.clear()
            self.apply_panel.warp_list.addItems(self.state.warp_param_paths)
            self.apply_panel.check_apply_warp_buttons_state()
            
        if hasattr(self, 'group_panel'):
            self.group_panel.group_list.clear()
            self.group_panel.group_list.addItems(self.state.group_data_paths)
            self.group_panel._update_group_view_button()
            
        self.log_message("UI updated from loaded project.")

    def _sync_state_from_ui(self):
        if hasattr(self, 'register_panel'):
            self.state.atlas_roi_path = self.register_panel.atlas_path_edit.text()
            self.state.target_roi_paths = [self.register_panel.target_list.item(i).text() for i in range(self.register_panel.target_list.count())]
            
        if hasattr(self, 'apply_panel'):
            self.state.warp_param_paths = [self.apply_panel.warp_list.item(i).text() for i in range(self.apply_panel.warp_list.count())]
            
        if hasattr(self, 'group_panel'):
            self.state.group_data_paths = [self.group_panel.group_list.item(i).text() for i in range(self.group_panel.group_list.count())]
            
        self.log_message("Internal state synchronized from UI.")
