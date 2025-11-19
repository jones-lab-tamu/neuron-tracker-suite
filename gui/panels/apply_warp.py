import os
import json
import numpy as np
from PyQt5 import QtWidgets, QtCore
from skimage.transform import ThinPlateSplineTransform
from gui.dialogs.warp_inspector import WarpInspectorWindow

from gui.theme import get_icon

class ApplyWarpPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = main_window.state
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Apply Warp Setup")
        b = QtWidgets.QVBoxLayout(box)
        self.warp_list = QtWidgets.QListWidget()
        self.warp_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        b.addWidget(self.warp_list)
        row = QtWidgets.QHBoxLayout()
        self.btn_add_warp = QtWidgets.QPushButton(get_icon('fa5s.plus'), "Add Warp Parameter File(s)...")
        self.btn_remove_warp = QtWidgets.QPushButton(get_icon('fa5s.minus'), "Remove Selected")
        row.addWidget(self.btn_add_warp)
        row.addWidget(self.btn_remove_warp)
        b.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        self.btn_inspect_warp = QtWidgets.QPushButton(get_icon('fa5s.search'), "Inspect Selected Warp...")
        self.btn_inspect_warp.setEnabled(False)
        self.btn_apply_warp = QtWidgets.QPushButton(get_icon('fa5s.check-double'), "Apply All Warp(s)")
        self.btn_apply_warp.setEnabled(False)
        row2.addWidget(self.btn_inspect_warp)
        row2.addWidget(self.btn_apply_warp)
        b.addLayout(row2)
        layout.addWidget(box)
        layout.addStretch(1)

    def connect_signals(self):
        self.btn_add_warp.clicked.connect(self.add_warp_files)
        self.btn_remove_warp.clicked.connect(self.remove_warp_file)
        self.warp_list.itemSelectionChanged.connect(self.check_apply_warp_buttons_state)
        self.btn_apply_warp.clicked.connect(self.apply_warps)
        self.btn_inspect_warp.clicked.connect(self.inspect_warp)

    def add_warp_files(self):
        start_dir = self.mw._get_last_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Warp Parameter Files",
            start_dir,
            "Warp Parameters (*_warp_parameters.json)",
        )
        if not files:
            return
        self.mw._set_last_dir(files[0])
        for f in files:
            if f not in self.state.warp_param_paths:
                self.state.warp_param_paths.append(f)
                self.warp_list.addItem(f)
        self.check_apply_warp_buttons_state()

    def remove_warp_file(self):
        for item in self.warp_list.selectedItems():
            row = self.warp_list.row(item)
            path = self.warp_list.item(row).text()
            if path in self.state.warp_param_paths:
                self.state.warp_param_paths.remove(path)
            self.warp_list.takeItem(row)
        self.check_apply_warp_buttons_state()

    def check_apply_warp_buttons_state(self):
        total = len(self.state.warp_param_paths)
        selected = len(self.warp_list.selectedItems())
        self.btn_apply_warp.setEnabled(total > 0)
        self.btn_inspect_warp.setEnabled(selected == 1)

    def apply_warps(self):
        if not self.state.warp_param_paths:
            self.mw.log_message("No warp files selected.")
            return
        self.mw.log_message(f"Applying {len(self.state.warp_param_paths)} warp(s)...")
        for warp_file in self.state.warp_param_paths:
            try:
                with open(warp_file, "r") as f: warp = json.load(f)
                roi_file = warp_file.replace("_warp_parameters.json", "_roi_filtered.csv")
                if not os.path.exists(roi_file):
                    self.mw.log_message(f"Missing _roi_filtered.csv for {os.path.basename(warp_file)}")
                    continue
                roi_pts = np.loadtxt(roi_file, delimiter=",")
                tps = ThinPlateSplineTransform()
                tps.estimate(np.array(warp["dest_landmarks_norm"]), np.array(warp["source_landmarks_norm"]))
                dc = np.array(warp["dest_centroid"])
                ds = warp["dest_scale"]
                sc = np.array(warp["source_centroid"])
                ss = warp["source_scale"]
                pts_norm = (roi_pts - dc) / ds
                warped_norm = tps(pts_norm)
                warped_pts = warped_norm * ss + sc
                out = roi_file.replace("_roi_filtered.csv", "_roi_warped.csv")
                np.savetxt(out, warped_pts, delimiter=",")
                self.mw.log_message(f"Created {os.path.basename(out)}")
            except Exception as e:
                self.mw.log_message(f"Failed to process {os.path.basename(warp_file)}: {e}")
        self.mw.log_message("Warp application complete.")
        self.mw.update_workflow_from_files()

    def inspect_warp(self):
        items = self.warp_list.selectedItems()
        if not items:
            self.mw.log_message("Select a warp file to inspect.")
            return
        warp_file = items[0].text()
        try:
            if not self.state.atlas_roi_path or not os.path.exists(self.state.atlas_roi_path):
                base_dir = os.path.dirname(warp_file)
                potential_atlas = os.path.join(base_dir, "atlas_anatomical_roi.json")
                if os.path.exists(potential_atlas):
                    self.state.atlas_roi_path = potential_atlas
                    self.mw.register_panel.atlas_path_edit.setText(potential_atlas)
                    self.mw.log_message(f"Auto-detected atlas: {os.path.basename(potential_atlas)}")
                else:
                    raise ValueError("Atlas file not set. Please select one in the 'Atlas Registration' panel.")
            
            target_roi_filtered = warp_file.replace("_warp_parameters.json", "_roi_filtered.csv")
            target_anatomical = warp_file.replace("_warp_parameters.json", "_anatomical_roi.json")
            if not os.path.exists(target_roi_filtered): raise FileNotFoundError(f"Missing _roi_filtered.csv for {os.path.basename(warp_file)}")
            if not os.path.exists(target_anatomical): raise FileNotFoundError(f"Missing _anatomical_roi.json for {os.path.basename(warp_file)}")
            
            with open(warp_file, "r") as f: warp = json.load(f)
            orig_pts = np.loadtxt(target_roi_filtered, delimiter=",")
            tps = ThinPlateSplineTransform()
            tps.estimate(np.array(warp["dest_landmarks_norm"]), np.array(warp["source_landmarks_norm"]))
            dc = np.array(warp["dest_centroid"])
            ds = warp["dest_scale"]
            sc = np.array(warp["source_centroid"])
            ss = warp["source_scale"]
            pts_norm = (orig_pts - dc) / ds
            warped_norm = tps(pts_norm)
            warped_pts = warped_norm * ss + sc
            
            dlg = WarpInspectorWindow(
                self.mw, self.state.atlas_roi_path, target_anatomical,
                orig_pts, warped_pts, pts_norm, warped_norm, warp,
                title=os.path.basename(target_roi_filtered)
            )
            dlg.exec_()
        except Exception as e:
            self.mw.log_message(f"Error during warp inspection: {e}")
