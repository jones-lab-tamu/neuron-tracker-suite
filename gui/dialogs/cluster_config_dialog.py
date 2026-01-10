from PyQt5 import QtWidgets, QtCore

class ClusterConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent, grid_shape, bin_stats):
        """
        bin_stats: list of tuples (n_control, n_experiment) for all CANDIDATE bins (lobe > 0).
        """
        super().__init__(parent)
        self.setWindowTitle("Cluster Analysis Configuration")
        self.resize(400, 300)
        
        self.bin_stats = bin_stats
        self.n_candidates = len(bin_stats)
        
        self.min_n = 4
        self.n_perm = 5000
        self.alpha_forming = 0.20
        self.alpha_sig = 0.05
        # compat
        self.alpha = 0.05
        self.seed = 42
        self.save_plot = False
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info Group
        info_box = QtWidgets.QGroupBox("Data Summary")
        form = QtWidgets.QFormLayout(info_box)
        form.addRow("Grid Shape (HxW):", QtWidgets.QLabel(f"{grid_shape[0]} x {grid_shape[1]}"))
        form.addRow("Candidate Bins (Lobe > 0):", QtWidgets.QLabel(str(self.n_candidates)))
        
        self.lbl_excluded = QtWidgets.QLabel("0")
        self.lbl_excluded.setStyleSheet("color: red")
        form.addRow("Excluded (Min N):", self.lbl_excluded)
        
        self.lbl_included = QtWidgets.QLabel("0")
        self.lbl_included.setStyleSheet("color: green; font-weight: bold")
        form.addRow("Included for Analysis:", self.lbl_included)
        
        layout.addWidget(info_box)
        
        # Params Group
        param_box = QtWidgets.QGroupBox("Analysis Parameters")
        p_layout = QtWidgets.QFormLayout(param_box)
        
        self.spin_min_n = QtWidgets.QSpinBox()
        self.spin_min_n.setRange(2, 50)
        self.spin_min_n.setValue(self.min_n)
        self.spin_min_n.valueChanged.connect(self.update_counts)
        p_layout.addRow("Min Animals per Bin (validity):", self.spin_min_n)
        
        self.spin_n_perm = QtWidgets.QSpinBox()
        self.spin_n_perm.setRange(100, 100000)
        self.spin_n_perm.setSingleStep(100)
        self.spin_n_perm.setValue(self.n_perm)
        p_layout.addRow("Permutations:", self.spin_n_perm)
        
        self.spin_alpha_forming = QtWidgets.QDoubleSpinBox()
        self.spin_alpha_forming.setRange(0.001, 0.5)
        self.spin_alpha_forming.setSingleStep(0.01)
        self.spin_alpha_forming.setValue(self.alpha_forming)
        p_layout.addRow("Alpha (Cluster forming, T0):", self.spin_alpha_forming)

        self.spin_alpha_sig = QtWidgets.QDoubleSpinBox()
        self.spin_alpha_sig.setRange(0.001, 0.5)
        self.spin_alpha_sig.setSingleStep(0.01)
        self.spin_alpha_sig.setValue(self.alpha_sig)
        p_layout.addRow("Alpha (Significance, p_corr):", self.spin_alpha_sig)
        
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(self.seed)
        p_layout.addRow("Random Seed:", self.spin_seed)
        
        # Connectivity
        self.combo_connectivity = QtWidgets.QComboBox()
        self.combo_connectivity.addItem("4-neighbor (edges only) [Default]", 4)
        self.combo_connectivity.addItem("8-neighbor (edges + diagonals)", 8)
        self.combo_connectivity.setCurrentIndex(0) # Default to 4
        p_layout.addRow("Connectivity:", self.combo_connectivity)
        
        self.chk_cross_lobe = QtWidgets.QCheckBox("Allow clusters across lobes")
        self.chk_cross_lobe.setChecked(False) # Off by default
        p_layout.addRow(self.chk_cross_lobe)
        
        self.check_plot = QtWidgets.QCheckBox("Save diagnostic plot (PNG)")
        p_layout.addRow(self.check_plot)
        
        layout.addWidget(param_box)
        
        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        # Initial update
        self.update_counts()
        
    def update_counts(self):
        mn = self.spin_min_n.value()
        excluded = 0
        for nc, ne in self.bin_stats:
            # Strictly excluded if ANY group has < min_n
            if nc < mn or ne < mn:
                excluded += 1
        
        included = self.n_candidates - excluded
        
        self.lbl_excluded.setText(str(excluded))
        self.lbl_included.setText(str(included))
        
    def accept(self):
        af = self.spin_alpha_forming.value()
        asig = self.spin_alpha_sig.value()
        
        if af < asig:
            QtWidgets.QMessageBox.warning(self, "Invalid Alpha", 
                "Cluster-forming alpha must be >= significance alpha.\nAdjusting cluster-forming alpha to match significance alpha.")
            af = asig
            self.spin_alpha_forming.setValue(af)
            # DO NOT return; continue to save corrected value
 
        self.min_n = self.spin_min_n.value()
        self.n_perm = self.spin_n_perm.value()
        self.alpha_forming = af
        self.alpha_sig = asig
        self.seed = self.spin_seed.value()
        self.save_plot = self.check_plot.isChecked()
        self.connectivity = self.combo_connectivity.currentData()
        self.allow_cross_lobe = self.chk_cross_lobe.isChecked()
        super().accept()
