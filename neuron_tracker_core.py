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
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrajectoryMetrics:
    """Quality metrics for a single trajectory candidate."""
    # Spatial Counts
    n_detected: int = 0
    
    # Coverage Metrics
    path_node_fraction: float = 0.0  # Legacy metric (len(path) / T)
    detected_fraction: float = 0.0   # Scientific metric (n_detected / T)
    
    # Spatial Quality
    max_gap: int = 0
    max_step: float = 0.0
    boundary_fail: bool = False
    
    # Jitter (Stability)
    spatial_jitter_raw: float = 0.0      
    spatial_jitter_detrended: float = 0.0 
    
    # Trace Metrics (Only computed if trace extracted)
    trace_snr_proxy: float = 0.0         # Median / MAD

@dataclass
class TrajectoryCandidate:
    """A single cell trajectory candidate with its metadata and data."""
    id: int
    pruned_path_nodes: list
    
    # Spatial Data
    positions: numpy.ndarray             # Shape (T, 2) [y, x]
    detected_mask: numpy.ndarray         # Shape (T,) boolean
    
    # Intensity Data (Optional, populated in Pass 2)
    trace: Optional[numpy.ndarray] = None # Shape (T,)
    
    # Metrics
    metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)
    
    # Pipeline State
    is_valid_spatial: bool = False  # Passed loose spatial gate?
    trace_extracted: bool = False   # Trace successfully sampled?
    accepted: bool = False          # Passed final policy?
    
    reject_reason: str = ""

# =============================================================================
# PARALLEL WORKER STATE & INIT
# =============================================================================

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
        _shm_arr = numpy.ndarray(shape, dtype=dtype, buffer=_shm_handle.buf)
        atexit.register(_cleanup_worker)
    except Exception as e:
        print(f"Worker initialization failed: {e}")
        raise

# =============================================================================
# CORE DETECTION LOGIC
# =============================================================================

def _detect_features_in_image(im, sigma1, sigma2, blur_sigma, max_features):
    """Pure function to detect features in a single image array."""
    fltrd = (
        scipy.ndimage.gaussian_filter(-im, sigma1)
        - scipy.ndimage.gaussian_filter(-im, sigma2)
    )
    fltrd = rescale(fltrd, 0.0, 1.0)

    locations = list(zip(*detect_local_minima(fltrd)))

    if not locations:
        return []

    blurred_im = scipy.ndimage.gaussian_filter(im, blur_sigma)
    mags = [blurred_im[i, j] for i, j in locations]

    indices = numpy.argsort(mags)
    indices = indices[-max_features:]
    
    xys = [locations[i] for i in indices]
    return xys

def _process_frame_task_shared(t, sigma1, sigma2, blur_sigma, max_features):
    """Worker task: reads from shared memory, calculates features."""
    im = _shm_arr[t]
    xys = _detect_features_in_image(im, sigma1, sigma2, blur_sigma, max_features)
    return t, xys

def _process_frame_star(args):
    return _process_frame_task_shared(*args)


# =============================================================================
# PUBLIC API: HELPERS
# =============================================================================

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

# =============================================================================
# PUBLIC API: PIPELINE STEPS
# =============================================================================

def process_frames(data, sigma1, sigma2, blur_sigma, max_features, progress_callback=None, n_processes=None):
    """Step 1: Detect features in each frame."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    T = data.shape[0]
    if n_processes is None:
        cpu = os.cpu_count() or 1
        n_processes = min(max(1, cpu - 2), 16)

    results = []
    step = max(1, T // 20)

    # --- SERIAL PATHWAY ---
    if n_processes < 2:
        progress_callback("Stage 1/4: Detecting features (Serial)...")
        for t in range(T):
            xys = _detect_features_in_image(data[t], sigma1, sigma2, blur_sigma, max_features)
            results.append((t, xys))
            if (t + 1) % step == 0 or t == T - 1:
                progress_callback(f"  Processed frame {t+1}/{T} ({100.0 * (t+1) / float(T):.1f}%)")

    # --- PARALLEL PATHWAY ---
    else:
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        try:
            shm_arr = numpy.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shm_arr[:] = data[:] 
            
            progress_callback(f"Stage 1/4: Detecting features (Parallel on {n_processes} cores)...")
            
            tasks = ((t, sigma1, sigma2, blur_sigma, max_features) for t in range(T))
            init_args = (shm.name, data.shape, data.dtype)

            with Pool(processes=n_processes, initializer=_init_worker, initargs=init_args) as pool:
                for i, res in enumerate(pool.imap_unordered(_process_frame_star, tasks)):
                    results.append(res)
                    if (i + 1) % step == 0 or i == T - 1:
                        progress_callback(f"  Processed frame {i+1}/{T} ({100.0 * (i+1) / float(T):.1f}%)")
        finally:
            shm.close()
            shm.unlink()

    # Deterministic Assembly
    results.sort(key=lambda x: x[0])
    
    blob_lists, trees, ims, ids = [], [], [], {}
    
    for t, xys in results:
        ims.append(data[t]) 
        if not xys:
            blob_lists.append([])
            trees.append(None)
            continue
        for x, y in xys:
            ids[(t, (x, y))] = len(ids)
        trees.append(scipy.spatial.KDTree(xys))
        blob_lists.append(xys)

    return ims, ids, trees, blob_lists


def build_trajectories(
    blob_lists, trees, ids, search_range, cone_radius_base, cone_radius_multiplier, progress_callback=None,
):
    """Step 2: Connect features across frames."""
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
    """Step 3: Prune branched trajectories."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    progress_callback("Stage 3/4: Pruning branched trajectories...")

    reverse_ids = {v: k for k, v in ids.items()}
    pruned_subgraphs = []
    
    # Ensure deterministic processing order for subgraphs
    sorted_subgraphs = sorted(subgraphs, key=lambda nodes: min(nodes))

    for nodes in sorted_subgraphs:
        # Create a FRESH graph component and populate it in sorted order.
        # This guarantees that networkx's internal adjacency dicts have a fixed insertion order,
        # ensuring that BFS (shortest_path) tie-breaking is deterministic.
        sorted_nodes = sorted(list(nodes))
        subgraph_view = graph.subgraph(nodes)
        
        det_subgraph = networkx.Graph()
        det_subgraph.add_nodes_from(sorted_nodes)
        # Sort edges to ensure deterministic edge insertion order
        sorted_edges = sorted([tuple(sorted(e)) for e in subgraph_view.edges()])
        det_subgraph.add_edges_from(sorted_edges)

        nodes_by_time = {}
        for node in sorted_nodes:
            t, _ = reverse_ids[node]
            nodes_by_time.setdefault(t, []).append(node)

        times = sorted(nodes_by_time.keys())
        if len(times) < 2:
            continue

        # Deterministic start/end
        nodes_by_time[times[0]].sort()
        nodes_by_time[times[-1]].sort()
        
        start_node = nodes_by_time[times[0]][0]
        end_node = nodes_by_time[times[-1]][0]

        try:
            path = networkx.shortest_path(det_subgraph, start_node, end_node)
            if path:
                pruned_subgraphs.append(path)
        except networkx.NetworkXNoPath:
            continue

    progress_callback(f"  Found {len(pruned_subgraphs)} potential trajectories.")
    return pruned_subgraphs, reverse_ids


def extract_and_interpolate_data(
    ims, 
    pruned_subgraphs, 
    reverse_ids, 
    min_trajectory_length, 
    sampling_box_size, 
    sampling_sigma, 
    max_interpolation_distance, 
    progress_callback=None,
    mode='strict', # 'strict' or 'scored'
    return_candidates=False
):
    """Step 4: Two-Pass Pipeline (Spatial -> Trace)."""
    if progress_callback is None:
        progress_callback = lambda msg: None

    progress_callback("Stage 4/4: Filtering and sampling (Two-Pass Pipeline)...")

    T = len(ims)
    
    if sampling_box_size % 2 == 0:
        sampling_box_size += 1
    
    blur_kernel = numpy.zeros((sampling_box_size, sampling_box_size))
    center = sampling_box_size // 2
    blur_kernel[center, center] = 1.0
    blur_kernel = scipy.ndimage.gaussian_filter(blur_kernel, sampling_sigma)
    blur_kernel /= numpy.sum(blur_kernel)
    b_half = sampling_box_size // 2
    
    # Enforce shape invariant
    if T > 0:
        h0, w0 = ims[0].shape
        for t in range(1, T):
            if ims[t].shape != (h0, w0):
                raise ValueError(f"Frame dimensions mismatch at index {t}: expected {(h0, w0)}, got {ims[t].shape}")
        h, w = h0, w0
    else:
        h, w = 0, 0 # Fallback for empty (though T=0 is unlikely here)

    candidates: List[TrajectoryCandidate] = []

    # --- PASS 1: SPATIAL ANALYSIS & GATING ---
    for i, path in enumerate(pruned_subgraphs):
        if i > 0 and len(pruned_subgraphs) > 10 and i % (len(pruned_subgraphs) // 10) == 0:
            progress_callback(f"  Processing trajectory {i+1}/{len(pruned_subgraphs)}")

        # 1.1: Basic Interpolation
        known_pos = {}
        for node in path:
            t, c = reverse_ids[node]
            known_pos[t] = c
            
        detected_times = sorted(known_pos.keys())
        if not detected_times: continue
        
        # 1.2: Metrics Prep
        n_detected = len(detected_times)
        detected_mask = numpy.zeros(T, dtype=bool)
        detected_mask[detected_times] = True
        
        interpolated_pos = numpy.zeros((T, 2))
        for t in detected_times: interpolated_pos[t] = known_pos[t]
        interpolated_pos[:detected_times[0]] = known_pos[detected_times[0]]
        interpolated_pos[detected_times[-1] + 1:] = known_pos[detected_times[-1]]
        
        max_gap = 0
        for j in range(len(detected_times) - 1):
            t1, t2 = detected_times[j], detected_times[j+1]
            gap = t2 - t1 - 1
            if gap > max_gap: max_gap = gap
            if gap > 0:
                p1 = numpy.array(known_pos[t1])
                p2 = numpy.array(known_pos[t2])
                for t in range(t1 + 1, t2):
                    alpha = (t - t1) / (t2 - t1)
                    interpolated_pos[t] = p1 * (1.0 - alpha) + p2 * alpha

        # 1.3: Compute Spatial Metrics
        int_pos = numpy.round(interpolated_pos).astype(int)
        jumps = numpy.sqrt(numpy.sum(numpy.diff(int_pos, axis=0) ** 2, axis=1))
        max_step = numpy.max(jumps) if len(jumps) > 0 else 0.0

        y_coords = int_pos[:, 0]
        x_coords = int_pos[:, 1]
        in_bounds_y = (y_coords >= b_half) & (y_coords < h - b_half)
        in_bounds_x = (x_coords >= b_half) & (x_coords < w - b_half)
        boundary_fail = not numpy.all(in_bounds_y & in_bounds_x)

        if n_detected > 1:
            det_points = numpy.array([known_pos[t] for t in detected_times])
            std_devs = numpy.std(det_points, axis=0)
            spatial_jitter_raw = numpy.sqrt(numpy.sum(std_devs**2))
            
            smooth_path = scipy.ndimage.uniform_filter1d(det_points, size=5, axis=0, mode='nearest')
            residuals = det_points - smooth_path
            res_std = numpy.std(residuals, axis=0)
            spatial_jitter_detrended = numpy.sqrt(numpy.sum(res_std**2))
        else:
            spatial_jitter_raw = 0.0
            spatial_jitter_detrended = 0.0

        cand = TrajectoryCandidate(
            id=i,
            pruned_path_nodes=path,
            positions=int_pos,
            detected_mask=detected_mask,
        )
        cand.metrics.n_detected = n_detected
        cand.metrics.path_node_fraction = len(path) / T # Legacy metric
        cand.metrics.detected_fraction = n_detected / T # Scientific metric
        cand.metrics.max_gap = max_gap
        cand.metrics.max_step = max_step
        cand.metrics.boundary_fail = boundary_fail
        cand.metrics.spatial_jitter_raw = spatial_jitter_raw
        cand.metrics.spatial_jitter_detrended = spatial_jitter_detrended
        
        # 1.4: GATE LOGIC
        
        is_valid = False

        if mode == 'strict':
            # Strict Policy: Mimic legacy checks
            if cand.metrics.path_node_fraction < min_trajectory_length:
                cand.reject_reason = "min_length_strict"
                is_valid = False
            elif cand.metrics.boundary_fail:
                cand.reject_reason = "boundary_strict"
                is_valid = False
            elif cand.metrics.max_step > max_interpolation_distance:
                cand.reject_reason = "max_step_strict"
                is_valid = False
            else:
                is_valid = True
        else:
            # Scored Policy: Loose Gate
            if boundary_fail:
                cand.reject_reason = "boundary_physical"
                is_valid = False
            elif n_detected < 3:
                cand.reject_reason = "insufficient_points"
                is_valid = False
            elif cand.metrics.path_node_fraction < 0.01:
                 cand.reject_reason = "too_short_loose"
                 is_valid = False
            else:
                 is_valid = True

        cand.is_valid_spatial = is_valid
        candidates.append(cand)

    # --- PASS 2: TRACE EXTRACTION ---
    for cand in candidates:
        if not cand.is_valid_spatial:
            continue
            
        trace = numpy.zeros(T)
        for t in range(T):
            y, x = cand.positions[t]
            patch = ims[t][
                y - b_half : y + b_half + 1,
                x - b_half : x + b_half + 1,
            ]
            trace[t] = numpy.sum(patch * blur_kernel)
            
        cand.trace = trace
        cand.trace_extracted = True
        
        med = numpy.median(trace)
        mad = numpy.median(numpy.abs(trace - med))
        if mad > 0:
            cand.metrics.trace_snr_proxy = med / mad
        else:
            cand.metrics.trace_snr_proxy = 0.0

    # --- PASS 3: FINAL ACCEPTANCE ---
    for cand in candidates:
        # In strict mode, acceptance was already decided by the gate.
        # In scored mode, we accept everything that survived extraction (for now).
        if cand.trace_extracted:
            cand.accepted = True
        else:
            cand.accepted = False
            
    # --- OUTPUT GENERATION ---
    if return_candidates:
        return candidates

    final_coms = []
    final_trajectories = []
    final_lines = []
    
    for cand in candidates:
        # Safety: trace must be present to write output
        if cand.accepted and cand.trace is not None:
            line_data = numpy.column_stack((numpy.arange(T), cand.trace))
            final_coms.append(numpy.mean(cand.positions, axis=0))
            final_trajectories.append(cand.positions)
            final_lines.append(line_data)
            
    return (
        numpy.array(final_coms),
        numpy.array(final_trajectories),
        numpy.array(final_lines),
    )