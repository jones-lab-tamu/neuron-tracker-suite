import os
import json
import numpy as np

from PyQt5 import QtWidgets, QtCore
from gui.dialogs.registration import RegistrationWindow
from gui.dialogs.roi_drawer import ROIDrawerDialog
from matplotlib.path import Path
from gui.theme import get_icon

class AtlasRegistrationPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Atlas Registration Setup")
        b = QtWidgets.QVBoxLayout(box)
        
        row = QtWidgets.QHBoxLayout()
        self.btn_select_atlas = QtWidgets.QPushButton(get_icon('fa5s.map'), "Select Atlas...")
        self.atlas_path_edit = QtWidgets.QLineEdit()
        self.atlas_path_edit.setReadOnly(True)
        
        # Edit Button
        self.btn_edit_atlas = QtWidgets.QPushButton(get_icon('fa5s.pen'), "Edit Atlas / Add Axis")
        self.btn_edit_atlas.setEnabled(False) # Disabled until atlas is selected
        
        row.addWidget(self.btn_select_atlas)
        row.addWidget(self.btn_edit_atlas)
        row.addWidget(self.atlas_path_edit)
        b.addLayout(row)
        
        self.target_list = QtWidgets.QListWidget()
        b.addWidget(self.target_list)
        row_btn = QtWidgets.QHBoxLayout()
        self.btn_add_targets = QtWidgets.QPushButton(get_icon('fa5s.plus'), "Add Target(s)...")
        self.btn_remove_target = QtWidgets.QPushButton(get_icon('fa5s.minus'), "Remove Selected")
        row_btn.addWidget(self.btn_add_targets)
        row_btn.addWidget(self.btn_remove_target)
        b.addLayout(row_btn)
        self.btn_begin_reg = QtWidgets.QPushButton(get_icon('fa5s.play-circle'), "Begin Registration...")
        self.btn_begin_reg.setEnabled(False)
        b.addWidget(self.btn_begin_reg)
        layout.addWidget(box)
        layout.addStretch(1)

    def connect_signals(self):
        self.btn_select_atlas.clicked.connect(self.select_atlas)
        self.btn_edit_atlas.clicked.connect(self.edit_atlas)
        self.btn_add_targets.clicked.connect(self.add_targets)
        self.btn_remove_target.clicked.connect(self.remove_target)
        self.btn_begin_reg.clicked.connect(self.begin_registration)

    def select_atlas(self):
        start_dir = self.mw._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Atlas ROI File",
            start_dir,
            "Anatomical ROI files (*_anatomical_roi.json)",
        )
        if not path:
            return
        self.mw._set_last_dir(path)
        self.state.atlas_roi_path = path
        self.atlas_path_edit.setText(path)
        self.btn_edit_atlas.setEnabled(True)
        self._update_reg_button_state()

    def edit_atlas(self):
        path = self.state.atlas_roi_path
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, 'r') as f:
                rois = json.load(f)
        except Exception as e:
            self.mw.log_message(f"Error loading atlas: {e}")
            return

        # 1. Calculate Bounding Box
        all_x = []
        all_y = []
        for r in rois:
            pts = np.array(r['path_vertices'])
            all_x.extend(pts[:, 0])
            all_y.extend(pts[:, 1])
            
        if not all_x:
            width, height = 1000, 1000
        else:
            pad = 50
            max_x = max(all_x) + pad
            max_y = max(all_y) + pad
            width = int(max_x)
            height = int(max_y)
        
        # 2. Create Blank Image
        bg_image = np.zeros((height, width)) 
        
        # 3. Define callback to save changes
        def save_callback(indices, updated_rois, updated_refs):
            final_list = []
            if updated_rois: final_list.extend(updated_rois)
            if updated_refs: final_list.extend(updated_refs)
            
            serializable = []
            for r in final_list:
                verts = r["path_vertices"]
                if hasattr(verts, 'tolist'): verts = verts.tolist()
                elif isinstance(verts, np.ndarray): verts = verts.tolist()
                
                item = {
                    "path_vertices": verts,
                    "mode": r["mode"]
                }
                serializable.append(item)
                
            try:
                with open(path, 'w') as f:
                    json.dump(serializable, f, indent=4)
                self.mw.log_message(f"Atlas updated: {os.path.basename(path)}")
            except Exception as e:
                self.mw.log_message(f"Failed to save atlas: {e}")

        # 4. Launch Dialog
        dlg = ROIDrawerDialog(
            self.mw,
            bg_image,
            None, # No cell dots
            None, # No basename needed, we handle save manually
            save_callback, 
            vmin=0, vmax=1
        )
        
        dlg.rois = rois 
        dlg.redraw_finished_rois()
        
        dlg.exec_()

    def add_targets(self):
        start_dir = self.mw._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Target ROI Files",
            start_dir,
            "Anatomical ROI files (*_anatomical_roi.json)",
        )
        if not files:
            return
        self.mw._set_last_dir(files[0])
        for f in files:
            if f not in self.state.target_roi_paths:
                self.state.target_roi_paths.append(f)
                self.target_list.addItem(f)
        self._update_reg_button_state()

    def remove_target(self):
        for item in self.target_list.selectedItems():
            row = self.target_list.row(item)
            path = self.target_list.item(row).text()
            if path in self.state.target_roi_paths:
                self.state.target_roi_paths.remove(path)
            self.target_list.takeItem(row)
        self._update_reg_button_state()

    def _update_reg_button_state(self):
        has_atlas = bool(self.state.atlas_roi_path)
        has_targets = len(self.state.target_roi_paths) > 0
        self.btn_begin_reg.setEnabled(has_atlas and has_targets)

    def begin_registration(self):
        if not self.state.atlas_roi_path or not self.state.target_roi_paths: return
        dlg = RegistrationWindow(self.mw, self.state, self.mw.log_message)
        dlg.exec_()
        self.mw.update_workflow_from_files()
