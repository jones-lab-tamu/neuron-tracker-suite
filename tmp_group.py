
    def run_cluster_analysis(self):
        if not hasattr(self, "_last_master_df") or self._last_master_df is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please generate group visualizations first to prepare data.")
            return

        df = self._last_master_df.copy()
        
        # Check Groups
        unique_groups = df['Group'].unique()
        if 'Control' not in unique_groups or 'Experiment' not in unique_groups:
             QtWidgets.QMessageBox.warning(self, "Missing Groups", "Both 'Control' and 'Experiment' groups are required.")
             return
             
        # 1. Reconstruct Grid (Same logic as _generate_continuous_maps)
        # We need the bins to match the visualization.
        try:
            grid_res = int(self.group_grid_res_edit.text())
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
            
        calc_x_bins = np.linspace(x_min, x_max, n_bins_x + 1)
        calc_y_bins = np.linspace(y_min, y_max, n_bins_y + 1)
        
        # 2. Assign cells to bins
        df['Grid_X_Index'] = pd.cut(df['X'], bins=calc_x_bins, labels=False, include_lowest=True)
        df['Grid_Y_Index'] = pd.cut(df['Y'], bins=calc_y_bins, labels=False, include_lowest=True)
        
        # 3. Create Lobe Mask
        lobe_mask = np.zeros((n_bins_y, n_bins_x), dtype=int)
        
        # Bin centers
        bin_w = (x_max - x_min) / n_bins_x
        bin_h = (y_max - y_min) / n_bins_y
        
        cy_indices, cx_indices = np.mgrid[0:n_bins_y, 0:n_bins_x]
        cy_centers = y_min + (cy_indices + 0.5) * bin_h
        cx_centers = x_min + (cx_indices + 0.5) * bin_w
        centers = np.column_stack((cx_centers.ravel(), cy_centers.ravel()))
        
        if not hasattr(self, 'zones') or not self.zones:
             QtWidgets.QMessageBox.warning(self, "No Regions", "No regions/lobes defined. Cannot run cluster analysis.")
             return
             
        # Rasterize zones
        for zid, info in self.zones.items():
            lobe_id = info.get('lobe', 0)
            if lobe_id == 0: continue
            
            polys = info['polygons']
            mask_in_zone = np.zeros(len(centers), dtype=bool)
            for poly in polys:
                mask_in_zone |= poly.contains_points(centers)
            
            mask_2d = mask_in_zone.reshape((n_bins_y, n_bins_x))
            lobe_mask[mask_2d] = int(lobe_id)
            
        if np.max(lobe_mask) == 0:
            QtWidgets.QMessageBox.warning(self, "No Lobes", "No zones with Lobe ID > 0 found. Please define Left (1) / Right (2) lobes in Region setup.")
            return

        # 4. Aggregate Phases
        grouped_phases_grid = {}
        
        df_valid = df.dropna(subset=['Grid_X_Index', 'Grid_Y_Index', 'Rel_Phase'])
        
        for (bx, by), bin_df in df_valid.groupby(['Grid_X_Index', 'Grid_Y_Index']):
            bx, by = int(bx), int(by)
            if not (0 <= by < n_bins_y and 0 <= bx < n_bins_x): continue
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
            
            if c_dict or e_dict:
                grouped_phases_grid[(by, bx)] = {'Control': c_dict, 'Experiment': e_dict} # (row, col) = (y, x)
        
        # 5. Dialog
        n_obs = len(grouped_phases_grid)
        dlg = ClusterConfigDialog(self, (n_bins_y, n_bins_x), n_obs, 0)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        # 6. Run Worker
        self.mw.log_message(f"Starting Cluster Analysis: MinN={dlg.min_n}, Perms={dlg.n_perm}, Alpha={dlg.alpha}, Seed={dlg.seed}")
        self.btn_cluster_stats.setEnabled(False)
        self.mw.set_status("Running Cluster Analysis...")
        
        self.cluster_worker = ClusterWorker(
            grouped_phases_grid, (n_bins_y, n_bins_x), lobe_mask,
            dlg.min_n, dlg.n_perm, dlg.seed, dlg.alpha, dlg.save_plot,
            self.mw.session_dir
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
