"""
Neuron Tracker - Command-Line Interface (CLI)

This script provides a command-line interface to the neuron tracking tool.
It is designed for users who want to automate the analysis or run it in a
scripting environment.

It imports its core logic from 'neuron_tracker_core.py'.
"""

import os
import sys
import argparse
import skimage.io
import numpy
import matplotlib.pyplot as plt

# Import the core processing functions from the separate library file.
import neuron_tracker_core as ntc

def main():
    """
    Main function to parse command-line arguments and run the full pipeline.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Neuron Tracking and Data Extraction Tool (CLI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input image file (e.g., a multi-frame TIFF).")
    parser.add_argument("--output_basename", help="Basename for the output CSV files. If not provided, it is derived from the input file name.")
    parser.add_argument("--visualize", action="store_true", help="If set, display summary plots after processing.")
    
    g1 = parser.add_argument_group('Feature Detection Parameters')
    g1.add_argument('--sigma1', type=float, default=3.0, help="Sigma for the smaller Gaussian in DoG filter.")
    g1.add_argument('--sigma2', type=float, default=20.0, help="Sigma for the larger Gaussian in DoG filter.")
    g1.add_argument('--blur_sigma', type=float, default=2.0, help="Sigma for blurring before measuring feature magnitude.")
    g1.add_argument('--max_features', type=int, default=800, help="Max number of features to detect per frame.")

    g2 = parser.add_argument_group('Trajectory Building Parameters')
    g2.add_argument('--search_range', type=int, default=50, help="Number of previous frames to search for a connection.")
    g2.add_argument('--cone_radius_base', type=float, default=1.5, help="Base radius for the backward search cone.")
    g2.add_argument('--cone_radius_multiplier', type=float, default=0.125, help="Growth factor for the search cone radius per frame.")

    g3 = parser.add_argument_group('Filtering and Sampling Parameters')
    g3.add_argument('--min_trajectory_length', type=float, default=0.08, help="Minimum trajectory length as a fraction of total frames.")
    g3.add_argument('--sampling_box_size', type=int, default=15, help="Side length of the intensity sampling box (must be odd).")
    g3.add_argument('--sampling_sigma', type=float, default=2.0, help="Sigma of the Gaussian for weighted intensity sampling.")
    g3.add_argument('--max_interpolation_distance', type=float, default=5.0, help="Safety check: max allowed pixel distance between frames in a final trajectory.")

    args = parser.parse_args()

    # --- 1. Load and Prepare Data ---
    print(f"Loading data from {args.input_file}...")
    try:
        data = skimage.io.imread(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
        
    data = ntc.rescale(data, 0.0, 1.0)

    # --- 2. Run Core Processing Pipeline ---
    progress_callback = lambda msg: print(msg)
    ims, ids, trees, blob_lists = ntc.process_frames(data, args.sigma1, args.sigma2, args.blur_sigma, args.max_features, progress_callback)
    graph, subgraphs = ntc.build_trajectories(blob_lists, trees, ids, args.search_range, args.cone_radius_base, args.cone_radius_multiplier, progress_callback)
    pruned_subgraphs, reverse_ids = ntc.prune_trajectories(graph, subgraphs, ids, progress_callback)
    com, traj, lines = ntc.extract_and_interpolate_data(ims, pruned_subgraphs, reverse_ids, args.min_trajectory_length, args.sampling_box_size, args.sampling_sigma, args.max_interpolation_distance, progress_callback)

    # --- 3. Handle Results and Save Files ---
    if len(lines) == 0:
        print("\nProcessing complete, but no valid trajectories were found.", file=sys.stderr)
        print("Consider adjusting the parameters.", file=sys.stderr)
        sys.exit(0)

    print(f"\nProcessing complete. Found {len(lines)} valid trajectories.")
    basename = args.output_basename if args.output_basename else os.path.splitext(os.path.basename(args.input_file))[0]
    
    roi_filename = f"{basename}_roi.csv"
    roi_data = numpy.column_stack((com[:, 1], com[:, 0]))
    numpy.savetxt(roi_filename, roi_data, delimiter=',')
    print(f"Saved center-of-mass data to {roi_filename}")

    traces_filename = f"{basename}_traces.csv"
    time_column = lines[0, :, 0]
    intensity_data = lines[:, :, 1].T
    traces_data = numpy.column_stack((time_column, intensity_data))
    numpy.savetxt(traces_filename, traces_data, delimiter=",")
    print(f"Saved intensity traces to {traces_filename}")

    # --- 4. Save the full trajectory data ---
    traj_filename = f"{basename}_trajectories.npy"
    numpy.save(traj_filename, traj)
    print(f"Saved full trajectory data to {traj_filename}")


    # --- 5. Optional Visualization ---
    if args.visualize:
        print("Generating visualizations...")
        plt.figure(figsize=(12, 8))
        for line in lines: plt.plot(line[:, 0], line[:, 1])
        plt.title(f"All {len(lines)} Intensity Traces"); plt.xlabel("Time (frames)"); plt.ylabel("Normalized Intensity")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(ims[len(ims) // 2], cmap='gray')
        plt.plot(com[:, 1], com[:, 0], '.', color='red', markersize=5, alpha=0.7)
        plt.title("Center of Mass of All Trajectories")
        plt.show()

        if len(traj) > 0:
            plt.figure(figsize=(10, 10))
            single_traj = traj[0, :, :]
            time_colors = numpy.arange(single_traj.shape[0])
            plt.scatter(single_traj[:, 1], single_traj[:, 0], c=time_colors, cmap='viridis', s=10)
            cbar = plt.colorbar(); cbar.set_label('Time (frames)', rotation=270, labelpad=15)
            plt.title("Movement of a Single Trajectory (Trajectory 0)")
            plt.xlim(0, ims[0].shape[1]); plt.ylim(ims[0].shape[0], 0)
            plt.show()

if __name__ == "__main__":
    main()