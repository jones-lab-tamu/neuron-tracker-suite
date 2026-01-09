
class ClusterResultViewerWidget(QtWidgets.QWidget):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.clusters = results.get('clusters', [])
        
        self.layout = QtWidgets.QHBoxLayout(self)
        
        # 1. Plot Area
        self.plot_widget = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.plot_widget)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        
        # 2. Table Area
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Lobe", "N Bins", "Mass", "p_corr"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemClicked.connect(self.on_table_click)
        self.table.setFixedWidth(300)
        
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.table)
        self.layout.addWidget(self.splitter)
        
        self.plot()
        self.populate_table()
        
    def plot(self):
        self.ax.clear()
        dm = self.results['delta_mu_map']
        cluster_map = self.results['cluster_map']
        
        # Background: Delta Mu
        im = self.ax.imshow(dm, cmap='coolwarm', vmin=-np.pi, vmax=np.pi, origin='lower')
        self.figure.colorbar(im, ax=self.ax, label='Delta Mu (rad)')
        
        # Outlines for significant clusters (p < 0.05)
        # Create a mask for sig clusters
        # We can just iterate and draw contours
        
        import matplotlib.colors as mcolors
        
        # Plot all clusters with thin lines, significant with thick/colored
        ids = np.unique(cluster_map)
        ids = ids[ids > 0]
        
        for cid in ids:
            mask = (cluster_map == cid).astype(int)
            # Find props
            # Check p_corr
            match = next((c for c in self.clusters if c['id'] == cid), None)
            if not match: continue
            
            p = match['p_corr']
            is_sig = p < 0.05
            
            color = 'black' if not is_sig else 'lime'
            linewidth = 1 if not is_sig else 2
            alpha = 0.5 if not is_sig else 1.0
            
            self.ax.contour(mask, levels=[0.5], colors=[color], linewidths=[linewidth], alpha=alpha, origin='lower')
            
            # Label
            # standard center
            cy, cx = np.mean(np.where(mask), axis=1)
            if is_sig:
               self.ax.text(cx, cy, f"#{cid}\np={p:.3f}", color='black', fontsize=8, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, ec='none'))
            
        self.ax.set_title(f"Cluster Analysis (T0={self.results.get('T0', 0):.3f})")
        self.canvas.draw()
        
    def populate_table(self):
        self.table.setRowCount(len(self.clusters))
        for i, c in enumerate(self.clusters):
            # ID
            item = QtWidgets.QTableWidgetItem(str(c['id']))
            self.table.setItem(i, 0, item)
            
            # Lobe
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(c.get('lobe', '?'))))
            
            # N Bins
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(c['n_bins'])))
            
            # Mass
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{c['mass']:.2f}"))
            
            # p_corr
            p = c['p_corr']
            p_item = QtWidgets.QTableWidgetItem(f"{p:.4f}")
            if p < 0.05:
                p_item.setBackground(QtGui.QColor(200, 255, 200)) # Light Green
            self.table.setItem(i, 4, p_item)
            
            # Store ID in user data for click lookup
            item.setData(QtCore.Qt.UserRole, c['id'])

    def on_table_click(self, item):
        row = item.row()
        id_item = self.table.item(row, 0)
        cid = int(id_item.text())
        pass
