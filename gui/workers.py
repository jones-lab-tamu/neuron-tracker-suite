import os
import numpy as np
import pandas as pd
import skimage.io
from PyQt5 import QtCore
import neuron_tracker_core as ntc

class AnalysisWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(float)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)  # success, error_msg

    def __init__(self, input_file, output_basename, args):
        super().__init__()
        self.input_file = input_file
        self.output_basename = output_basename
        self.args = args

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.message.emit(f"Loading data from {self.input_file}...")
            data = skimage.io.imread(self.input_file)
            data = ntc.rescale(data, 0.0, 1.0)
            T = data.shape[0]

            # --- Robust callback wrappers (core may send str, float, tuple, etc.) ---
            stage_min = 0.0
            stage_max = 0.0

            def _set_stage(a, b):
                nonlocal stage_min, stage_max
                stage_min, stage_max = float(a), float(b)

            def _log_progress(x):
                # If core sends numeric progress, map it into the current stage range.
                try:
                    if isinstance(x, (int, float, np.floating)):
                        v = float(x)
                        if 0.0 <= v <= 1.0:
                            self.progress.emit(stage_min + v * (stage_max - stage_min))
                            return
                except Exception:
                    pass

                # Otherwise treat as message-like update.
                try:
                    self.message.emit(str(x))
                except Exception:
                    self.message.emit("Progress update (unprintable)")

            self.message.emit("Stage 1/4: Detecting features...")
            _set_stage(0.00, 0.25)
            ims, ids, trees, blob_lists = ntc.process_frames(
                data,
                sigma1=self.args["sigma1"],
                sigma2=self.args["sigma2"],
                blur_sigma=self.args["blur_sigma"],
                max_features=self.args["max_features"],
                progress_callback=_log_progress,
            )

            self.message.emit("Stage 2/4: Building trajectories...")
            _set_stage(0.25, 0.50)
            graph, subgraphs = ntc.build_trajectories(
                blob_lists,
                trees,
                ids,
                search_range=self.args["search_range"],
                cone_radius_base=self.args["cone_radius_base"],
                cone_radius_multiplier=self.args["cone_radius_multiplier"],
                progress_callback=_log_progress,
            )

            self.message.emit("Stage 3/4: Pruning trajectories...")
            _set_stage(0.50, 0.70)
            pruned_subgraphs, reverse_ids = ntc.prune_trajectories(
                graph,
                subgraphs,
                ids,
                progress_callback=_log_progress,
            )

            self.message.emit("Stage 4/4: Extracting traces & Computing Metrics...")
            
            mode = self.args.get('mode', 'strict')

            # Use return_candidates=True to get rich object data
            _set_stage(0.70, 1.00)
            candidates = ntc.extract_and_interpolate_data(
                ims,
                pruned_subgraphs,
                reverse_ids,
                min_trajectory_length=self.args["min_trajectory_length"],
                sampling_box_size=self.args["sampling_box_size"],
                sampling_sigma=self.args["sampling_sigma"],
                max_interpolation_distance=self.args["max_interpolation_distance"],
                progress_callback=_log_progress,
                mode=mode,
                return_candidates=True
            )

            # --- EXPLICIT GATING LOGIC ---
            accepted_candidates = []
            
            for c in candidates:
                # 1. Hard invariant: Must have data
                if c.trace is None:
                    continue

                # 2. Mode-specific Selection
                if mode == 'strict':
                    # Legacy contract: must pass all strict checks in core
                    if getattr(c, "accepted", False):
                        accepted_candidates.append(c)
                else:
                    # Scored contract: keep candidates with a usable trace AND spatially plausible track.
                    # This preserves post-hoc flexibility without admitting clearly broken tracks.
                    if getattr(c, "trace_extracted", False) and getattr(c, "is_valid_spatial", True):
                        accepted_candidates.append(c)

            if not accepted_candidates:
                raise RuntimeError("Processing complete, but no valid trajectories were found.")
            else:
                self.message.emit(
                    f"Processing complete. Saving {len(accepted_candidates)} trajectories."
                )

                coms_list = []
                trajectories_list = []
                metrics_data = []

                for i, cand in enumerate(accepted_candidates):
                    # --- HARD INVARIANTS ---
                    if cand.positions is None:
                        raise ValueError(
                            f"Candidate {i} has a trace but no positions, cannot save trajectories."
                        )
                    if cand.trace is None:
                        raise ValueError(
                            f"Candidate {i} has positions but no trace, cannot save trajectories."
                        )

                    # --- SAFETY ASSERTIONS ---
                    # Ensure alignment and shape integrity before saving
                    if cand.positions.shape != (T, 2):
                        raise ValueError(
                            f"Candidate {i} position shape mismatch: {cand.positions.shape} vs ({T}, 2)"
                        )
                    if cand.trace.shape != (T,):
                        raise ValueError(
                            f"Candidate {i} trace shape mismatch: {cand.trace.shape} vs ({T},)"
                        )

                    # --- Arrays ---
                    
                    coms_list.append(np.mean(cand.positions, axis=0))
                    trajectories_list.append(cand.positions)

                    # --- Metrics ---
                    m = getattr(cand, "metrics", None)
                    metrics_data.append({
                        'candidate_id': i,  # Stable index 0..N-1 matching arrays
                        'original_graph_id': getattr(cand, "id", np.nan),

                        'n_detected': getattr(m, "n_detected", np.nan),
                        'detected_fraction': getattr(m, "detected_fraction", np.nan),
                        'path_node_fraction': getattr(m, "path_node_fraction", np.nan),
                        'max_gap': getattr(m, "max_gap", np.nan),
                        'max_step': getattr(m, "max_step", np.nan),
                        'boundary_fail': getattr(m, "boundary_fail", np.nan),
                        'spatial_jitter_raw': getattr(m, "spatial_jitter_raw", np.nan),
                        'spatial_jitter_detrended': getattr(m, "spatial_jitter_detrended", np.nan),
                        'trace_snr_proxy': getattr(m, "trace_snr_proxy", np.nan),

                        'cell_fraction': getattr(m, 'cell_fraction', np.nan),
                        'cell_logz_median': getattr(m, 'cell_logz_median', np.nan),
                        'cell_area_median': getattr(m, 'cell_area_median', np.nan),
                        'cell_ecc_median': getattr(m, 'cell_ecc_median', np.nan),
                        'cell_center_annulus_ratio_median': getattr(m, 'cell_center_annulus_ratio_median', np.nan),

                        'reject_reason': getattr(cand, "reject_reason", ""),
                        'is_valid_spatial': getattr(cand, "is_valid_spatial", np.nan),
                        'trace_extracted': getattr(cand, "trace_extracted", np.nan),
                    })

                # Convert lists to numpy arrays for saving
                com_arr = np.stack(coms_list, axis=0).astype(np.float32)
                
                H, W = data.shape[1], data.shape[2]
                
                if np.any((com_arr[:, 0] < 0) | (com_arr[:, 0] >= H) | (com_arr[:, 1] < 0) | (com_arr[:, 1] >= W)):
                    self.message.emit("Warning: COM coordinates fall outside image bounds, check (y,x) vs (x,y) convention in core.")
                
                traj_arr = np.stack(trajectories_list, axis=0).astype(np.float32)
                
                # Reconstruct traces.csv matrix
                intensity_matrix = np.column_stack([cand.trace for cand in accepted_candidates]).astype(np.float32)
                frame_index = np.arange(T, dtype=np.int32)
                traces_out = np.column_stack((frame_index, intensity_matrix))

                # ROI is saved as (x, y) for GUI consumption.
                # com_arr is computed from image coordinates: (y, x).
                y = com_arr[:, 0]
                x = com_arr[:, 1]
                roi_out = np.column_stack((x, y))
                
                # --- SAVE FILES ---
                np.savetxt(f"{self.output_basename}_roi.csv", roi_out, delimiter=",")
                self.message.emit("Saved center-of-mass data.")

                np.savetxt(f"{self.output_basename}_traces.csv", traces_out, delimiter=",")
                self.message.emit("Saved intensity traces.")

                np.save(f"{self.output_basename}_trajectories.npy", traj_arr)
                self.message.emit("Saved full trajectory data.")
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_csv(f"{self.output_basename}_metrics.csv", index=False)
                self.message.emit("Saved quality metrics.")

            self.progress.emit(1.0)
            self.finished.emit(True, "")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.message.emit(f"--- ANALYSIS FAILED ---\nError: {e}")
            self.message.emit(tb)
            self.finished.emit(False, str(e))

class MovieLoaderWorker(QtCore.QObject):
    """Worker to load a movie file in a background thread."""
    finished = QtCore.pyqtSignal(object)  # Emits the loaded numpy array
    error = QtCore.pyqtSignal(str)

    def __init__(self, movie_path):
        super().__init__()
        self.movie_path = movie_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if not self.movie_path or not os.path.exists(self.movie_path):
                raise FileNotFoundError("Movie file not found.")
            data = skimage.io.imread(self.movie_path)
            self.finished.emit(data)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Failed to load movie: {e}\n{tb}")