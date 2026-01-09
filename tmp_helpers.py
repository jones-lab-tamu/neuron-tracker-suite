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
            'grid_x_bins': np.arange(start_x, end_x, bin_size), # For viewers usually
            'grid_y_bins': np.arange(start_y, end_y, bin_size),
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
            
            # Reset scatter tab
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
