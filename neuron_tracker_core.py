# -*- coding: utf-8 -*-
"""
Neuron Tracker Core Logic Library

This file contains all the core functions for the neuron tracking and data
extraction pipeline. It is designed to be a self-contained library.
"""

import os
import atexit
import numpy
import scipy.ndimage
import scipy.spatial
import networkx
from multiprocessing import Pool, shared_memory

# --- Worker Global State (initialized per process) ---
_shm_handle = None
_shm_arr = None

def _cleanup_worker():
    """Closes the shared memory handle on worker exit."""
    global _shm_handle
    if _shm_handle is not None:
        try:
            _shm_handle.close()
        except:
            pass
        _shm_handle = None

def _init_worker(shm_name, shape, dtype):
    """
    Initializes the worker process by attaching to the existing SharedMemory block.
    """
    global _shm_handle, _shm_arr
    try:
        _shm_handle = shared_memory.SharedMemory(name=shm_name)
        # Create a numpy view into the shared memory
        _shm_arr = numpy.ndarray(shape, dtype=dtype, buffer=_shm_handle.buf)
        # Register cleanup to run when this worker process eventually exits
        atexit.register(_cleanup_worker)
    except Exception as e:
        # On Windows, this print might get lost, but it's better than silent failure
        print(f"Worker initialization failed: {e}")
        raise

# --- Core Detection Logic (Factored out to ensure Serial/Parallel equivalence) ---
def _detect_features_in_image(im, sigma1, sigma2, blur_sigma, max_features):
    """
    Pure function to detect features in a single image array.
    """
    # Difference of Gaussians on the negative image
    # Note: We negate 'im' inside the call to match original logic
    fltrd = (
        scipy.ndimage.gaussian_filter(-im, sigma1)
        - scipy.ndimage.gaussian_filter(-im, sigma2)
    )
    fltrd = rescale(fltrd, 0.0, 1.0)

    # Local minima in the filtered image
    locations = list(zip(*detect_local_minima(fltrd)))

    if not locations:
        return []

    # Rank by intensity on a blurred version of the original image
    blurred_im = scipy.ndimage.gaussian_filter(im, blur_sigma)
    mags = [blurred_im[i, j] for i, j in locations]

    indices = numpy.argsort(mags)
    # Take the top N, but maintain the sort order (lowest to highest magnitude)
    indices = indices[-max_features:]
    
    xys = [locations[i] for i in indices]
    return xys

def _process_frame_task_shared(t, sigma1, sigma2, blur_sigma, max_features):
    """
    Worker task: reads from shared memory, calculates features.
    """
    # Read frame t from the shared array (zero-copy view)
    im = _shm_arr[t]
    xys = _detect_features_in_image(im, sigma1, sigma2, blur_sigma, max_features)
    return t, xys

def _process_frame_star(args):
    """Shim to unpack arguments for starmap/imap."""
    return _process_frame_task_shared(*args)


# --- Public API ---

def detect_local_minima(arr):
    """Takes a 2D array and detects local minima (troughs)."""
    neighborhood = scipy.ndimage.generate_binary_structure(len(arr.shape), 3)
    local_min = (scipy.ndimage.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = scipy.ndimage.binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    detected_minima = local_min ^ eroded_background
    return numpy.where(detected_minima)


def rescale(signal, minimum, maximum):
    """Rescales a numpy array to a new specified minimum and maximum range."""
    mins = numpy.min(signal)
    maxs = numpy.max(signal)
    if maxs == mins:
        return numpy.full(signal.shape, minimum)
    output = numpy.array(signal, dtype=float)
    output -= mins
    output *= (maximum - minimum) / (maxs - mins)
    output += minimum
    return output


def process_frames(data, sigma1, sigma2, blur_sigma, max_features, progress_callback=None, n_processes=None):
    """
    Detects features (neurons) in each frame.
    
    Args:
        data: 3D numpy array (T, Y, X).
        n_processes: Number of worker processes. 
                     If None, defaults to min(cpu_count, 8).
                     If 1, runs serially in main process (no shared memory).
    """
    if progress_callback is None:
        progress_callback = lambda msg: None

    T = data.shape[0]
    
    # Determine Process Count
    if n_processes is None:
        # Cap at 8 to prevent UI starvation/memory contention
        n_processes = min(os.cpu_count() or 1, 8)

    results = []
    step = max(1, T // 20)

    # --- SERIAL PATHWAY (n=1) ---
    if n_processes < 2:
        progress_callback("Stage 1/4: Detecting features (Serial)...")
        for t in range(T):
            xys = _detect_features_in_image(data[t], sigma1, sigma2, blur_sigma, max_features)
            results.append((t, xys))
            
            if (t + 1) % step == 0 or t == T - 1:
                progress_callback(f"  Processed frame {t+1}/{T} ({100.0 * (t+1) / float(T):.1f}%)")

    # --- PARALLEL PATHWAY (n > 1) ---
    else:
        # Allocate shared memory
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        
        try:
            # Copy data to shared memory
            shm_arr = numpy.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shm_arr[:] = data[:] 
            
            progress_callback(f"Stage 1/4: Detecting features (Parallel on {n_processes} cores)...")
            
            # Generator expression for O(1) memory overhead during task creation
            tasks = (
                (t, sigma1, sigma2, blur_sigma, max_features) 
                for t in range(T)
            )

            init_args = (shm.name, data.shape, data.dtype)

            with Pool(processes=n_processes, initializer=_init_worker, initargs=init_args) as pool:
                # imap_unordered for better throughput
                for i, res in enumerate(pool.imap_unordered(_process_frame_star, tasks)):
                    results.append(res)
                    if (i + 1) % step == 0 or i == T - 1:
                        progress_callback(
                            f"  Processed frame {i+1}/{T} ({100.0 * (i+1) / float(T):.1f}%)"
                        )
        
        finally:
            shm.close()
            shm.unlink()

    # --- Deterministic Assembly ---
    # Sort by frame index t to ensure strict ordering before ID assignment
    results.sort(key=lambda x: x[0])
    
    blob_lists, trees, ims, ids = [], [], [], {}
    
    for t, xys in results:
        # Store reference to original data
        ims.append(data[t]) 
        
        if not xys:
            blob_lists.append([])
            trees.append(None)
            continue
            
        # Serial ID assignment ensures identical IDs across runs
        for x, y in xys:
            ids[(t, (x, y))] = len(ids)
            
        trees.append(scipy.spatial.KDTree(xys))
        blob_lists.append(xys)

    return ims, ids, trees, blob_lists


def build_trajectories(
    blob_lists, trees, ids, search_range, cone_radius_base, cone_radius_multiplier, progress_callback=None,
):
    """Connects features across frames to build trajectories."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    T = len(blob_lists)
    graph = networkx.Graph()

    progress_callback("Stage 2/4: Building trajectories by connecting features...")
    step = max(1, T // 20) if T > 0 else 1

    for t in range(1, T):
        if (t + 1) % step == 0 or t == T - 1:
            progress_callback(f"  Connecting features up to frame {t+1}/{T} ({100.0 * (t+1) / float(T):.1f}%)")

        if not blob_lists[t]:
            continue

        for blob in blob_lists[t]:
            start = max(0, t - search_range)
            for bt in range(t - 1, start - 1, -1):
                if trees[bt] is None or not blob_lists[bt]:
                    continue

                distance, neighbor_idx = trees[bt].query(blob, 1)
                cone_radius = cone_radius_base + cone_radius_multiplier * (t - bt)

                if distance < cone_radius:
                    neighbor_coords = tuple(trees[bt].data[neighbor_idx])
                    graph.add_edge(ids[(bt, neighbor_coords)], ids[(t, blob)])
                    break

    subgraphs = list(networkx.connected_components(graph))
    return graph, subgraphs


def prune_trajectories(graph, subgraphs, ids, progress_callback=None):
    """Prunes branched trajectories to ensure a single path."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    progress_callback("Stage 3/4: Pruning branched trajectories...")

    reverse_ids = {v: k for k, v in ids.items()}
    pruned_subgraphs = []

    for nodes in subgraphs:
        subgraph = graph.subgraph(nodes)
        nodes_by_time = {}

        for node in nodes:
            t, _ = reverse_ids[node]
            nodes_by_time.setdefault(t, []).append(node)

        times = sorted(nodes_by_time.keys())
        if len(times) < 2:
            continue

        start_node = nodes_by_time[times[0]][0]
        end_node = nodes_by_time[times[-1]][0]

        try:
            path = networkx.shortest_path(subgraph, start_node, end_node)
            if path:
                pruned_subgraphs.append(path)
        except networkx.NetworkXNoPath:
            continue

    progress_callback(f"  Found {len(pruned_subgraphs)} potential trajectories.")
    return pruned_subgraphs, reverse_ids


def extract_and_interpolate_data(
    ims, pruned_subgraphs, reverse_ids, min_trajectory_length, sampling_box_size, sampling_sigma, max_interpolation_distance, progress_callback=None,
):
    """Filters, interpolates, and samples data from the final, pruned trajectories."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    progress_callback("Stage 4/4: Filtering, interpolating, and sampling final trajectories...")

    T = len(ims)
    lines, coms, trajectories = [], [], []

    if sampling_box_size % 2 == 0:
        sampling_box_size += 1

    blur = numpy.zeros((sampling_box_size, sampling_box_size))
    center = sampling_box_size // 2
    blur[center, center] = 1.0
    blur = scipy.ndimage.gaussian_filter(blur, sampling_sigma)
    blur /= numpy.sum(blur)

    for i, path in enumerate(pruned_subgraphs):
        if i > 0 and len(pruned_subgraphs) > 10 and i % (len(pruned_subgraphs) // 10) == 0:
            progress_callback(f"  Sampling trajectory {i+1}/{len(pruned_subgraphs)}")

        if len(path) < min_trajectory_length * T:
            continue

        interpolated_pos = {}
        interpolated_val = {}

        for node in path:
            t, c = reverse_ids[node]
            interpolated_pos[t] = c

        detected_times = sorted(interpolated_pos.keys())
        if not detected_times:
            continue

        start_pos = interpolated_pos[detected_times[0]]
        for t in range(0, detected_times[0]):
            interpolated_pos[t] = start_pos

        end_pos = interpolated_pos[detected_times[-1]]
        for t in range(detected_times[-1] + 1, T):
            interpolated_pos[t] = end_pos

        for j in range(len(detected_times) - 1):
            lt, rt = detected_times[j], detected_times[j + 1]
            if rt == lt + 1:
                continue
            lc = numpy.array(interpolated_pos[lt])
            rc = numpy.array(interpolated_pos[rt])
            for t in range(lt + 1, rt):
                alpha = float(t - lt) / (rt - lt)
                interpolated_pos[t] = tuple(lc * (1.0 - alpha) + rc * alpha)

        full_trajectory = []
        cancel = False
        b_half = sampling_box_size // 2

        for t in range(T):
            c = numpy.round(numpy.array(interpolated_pos[t])).astype(int)
            full_trajectory.append(c)

            h, w = ims[t].shape
            y, x = c

            if not (b_half <= y < h - b_half and b_half <= x < w - b_half):
                cancel = True
                break

            patch = ims[t][
                y - b_half : y + b_half + 1,
                x - b_half : x + b_half + 1,
            ]
            interpolated_val[t] = numpy.sum(patch * blur)

        if cancel:
            continue

        full_trajectory = numpy.array(full_trajectory)

        distances = numpy.sqrt(numpy.sum(numpy.diff(full_trajectory, axis=0) ** 2, axis=1))
        if numpy.any(distances > max_interpolation_distance):
            continue

        coms.append(numpy.mean(full_trajectory, axis=0))
        trajectories.append(full_trajectory)
        lines.append(numpy.array(sorted(interpolated_val.items())))

    return (
        numpy.array(coms),
        numpy.array(trajectories),
        numpy.array(lines),
    )