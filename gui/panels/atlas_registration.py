import os
from PyQt5 import QtWidgets, QtCore
from gui.dialogs.registration import RegistrationWindow

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
        row.addWidget(self.btn_select_atlas)
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
        self._update_reg_button_state()

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
