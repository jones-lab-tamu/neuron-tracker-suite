import os
import numpy as np
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

            self.message.emit("Stage 1/4: Detecting features...")
            ims, ids, trees, blob_lists = ntc.process_frames(
                data,
                sigma1=self.args["sigma1"],
                sigma2=self.args["sigma2"],
                blur_sigma=self.args["blur_sigma"],
                max_features=self.args["max_features"],
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.25)

            self.message.emit("Stage 2/4: Building trajectories...")
            graph, subgraphs = ntc.build_trajectories(
                blob_lists,
                trees,
                ids,
                search_range=self.args["search_range"],
                cone_radius_base=self.args["cone_radius_base"],
                cone_radius_multiplier=self.args["cone_radius_multiplier"],
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.5)

            self.message.emit("Stage 3/4: Pruning trajectories...")
            pruned_subgraphs, reverse_ids = ntc.prune_trajectories(
                graph,
                subgraphs,
                ids,
                progress_callback=self.message.emit,
            )
            self.progress.emit(0.7)

            self.message.emit("Stage 4/4: Extracting traces...")
            com, traj, lines = ntc.extract_and_interpolate_data(
                ims,
                pruned_subgraphs,
                reverse_ids,
                min_trajectory_length=self.args["min_trajectory_length"],
                sampling_box_size=self.args["sampling_box_size"],
                sampling_sigma=self.args["sampling_sigma"],
                max_interpolation_distance=self.args["max_interpolation_distance"],
                progress_callback=self.message.emit,
            )

            if len(lines) == 0:
                self.message.emit(
                    "Processing complete, but no valid trajectories were found."
                )
            else:
                self.message.emit(
                    f"Processing complete. Found {len(lines)} valid trajectories."
                )

                # ROI centers (x, y)
                np.savetxt(
                    f"{self.output_basename}_roi.csv",
                    np.column_stack((com[:, 1], com[:, 0])),
                    delimiter=",",
                )
                self.message.emit("Saved center-of-mass data.")

                # Traces: time + intensities
                np.savetxt(
                    f"{self.output_basename}_traces.csv",
                    np.column_stack((lines[0, :, 0], lines[:, :, 1].T)),
                    delimiter=",",
                )
                self.message.emit("Saved intensity traces.")

                # Full trajectories
                np.save(f"{self.output_basename}_trajectories.npy", traj)
                self.message.emit("Saved full trajectory data.")

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
