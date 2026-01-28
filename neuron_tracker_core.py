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
import time
from multiprocessing import Pool, shared_memory
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

DEFAULT_PERIOD_HOURS = 24.0

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrajectoryMetrics:
    """Quality metrics for a single trajectory candidate."""
    # Spatial Counts
    n_detected: int = 0
    
    # Coverage Metrics
    path_node_fraction: float = 0.0  # (len(path) / T)
    detected_fraction: float = 0.0   # (n_detected / T)
    
    # Spatial Quality
    max_gap: int = 0
    max_step: float = 0.0
    boundary_fail: bool = False
    
    # Jitter (Stability)
    spatial_jitter_raw: float = 0.0      
    spatial_jitter_detrended: float = 0.0 
    
    # Trace Metrics (Only computed if trace extracted)
    trace_snr_proxy: float = 0.0         # Median / MAD

    # Cellness Metrics
    cell_fraction: float = 0.0
    cell_logz_median: float = 0.0
    cell_area_median: float = 0.0
    cell_ecc_median: float = 0.0
    cell_center_annulus_ratio_median: float = 0.0

    # Robust Cellness (Identity Rescue)
    cell_fraction_robust: float = 0.0
    id_template_n: int = 0
    id_sim_median: float = 0.0
    id_sim_p10: float = 0.0
    id_sim_fraction_ge_thr: float = 0.0

@dataclass
class TrajectoryCandidate:
    """A single cell trajectory candidate with its metadata and data."""
    id: int
    pruned_path_nodes: list
    
    # Spatial Data
    positions: numpy.ndarray             # Shape (T, 2) [y, x]
    detected_mask: numpy.ndarray         # Shape (T,) boolean
    
    # Intensity Data (Optional)
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

def _robust_center_vs_dist_z(center_val, dist_arr, eps=1e-12):
    """Compute robust z-score of a scalar relative to a distribution."""
    med = numpy.median(dist_arr)
    mad = numpy.median(numpy.abs(dist_arr - med))
    return (center_val - med) / (1.4826 * mad + eps)

def _extract_center_patch(im: numpy.ndarray, y: int, x: int, patch_half: int) -> Optional[numpy.ndarray]:
    """Extracts a square patch centered at (y, x). Returns None if out of bounds."""
    H, W = im.shape
    y_start = y - patch_half
    y_end = y + patch_half + 1
    x_start = x - patch_half
    x_end = x + patch_half + 1
    
    if y_start < 0 or x_start < 0 or y_end > H or x_end > W:
        return None
        
    return im[y_start:y_end, x_start:x_end]

def _normalize_patch(patch: numpy.ndarray, eps: float = 1e-12) -> numpy.ndarray:
    """Normalize patch to mean 0, std 1."""
    p = patch.astype(numpy.float32)
    mean = numpy.mean(p)
    std = numpy.std(p)
    return (p - mean) / (std + eps)

def _build_template(patches: List[numpy.ndarray], method: str = "median") -> numpy.ndarray:
    """Computes a template patch from a list of patches."""
    if not patches:
        raise ValueError("Cannot build template from empty patch list")
        
    stack = numpy.stack(patches, axis=0) # (N, S, S)
    if method == "median":
        return numpy.median(stack, axis=0).astype(numpy.float32)
    else:
        raise ValueError("Only method='median' is supported for identity templates.")

def _template_similarity(template_norm: numpy.ndarray, patch_norm: numpy.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two normalized patches."""
    # Cosine sim = (A . B) / (|A| |B|)
    # num = sum(template_norm * patch_norm)
    # den = sqrt(sum(template_norm^2)) * sqrt(sum(patch_norm^2)) + eps
    
    num = numpy.sum(template_norm * patch_norm)
    denom = numpy.sqrt(numpy.sum(template_norm**2)) * numpy.sqrt(numpy.sum(patch_norm**2)) + eps
    return float(num / denom)

def _compute_cellness_features_for_patch(patch, center_rc, log_sigma=1.5,
                                         thr_k=1.0, inner_r=3, annulus_r0=6, annulus_r1=10):
    """Compute per-patch cellness features."""
    features = {
        "logz": 0.0,
        "area": 0.0,
        "ecc": 1.0,
        "center_annulus_ratio": 0.0,
        "blob_ok": False
    }
    
    # i) Compute LoG image
    patch_float = patch.astype(float)
    log_im = scipy.ndimage.gaussian_laplace(patch_float, sigma=log_sigma)
    
    # Peak is negative Laplacian at center. 
    # Use -log_im for both peak extraction and distribution stats to be consistent.
    peak_im = -log_im
    center_val = peak_im[center_rc]
    
    # ii) Robust z-score of peak vs all LoG pixels
    features["logz"] = _robust_center_vs_dist_z(center_val, peak_im)
    
    # iii) Threshold in INTENSITY space
    mu = numpy.mean(patch_float)
    sig = numpy.std(patch_float)
    thr = mu + thr_k * sig
    bin_im = patch_float > thr
    
    labeled, n_lbl = scipy.ndimage.label(bin_im)
    lbl_at_center = labeled[center_rc]
    
    if lbl_at_center == 0:
        features["area"] = numpy.nan
        features["ecc"] = numpy.nan
        features["center_annulus_ratio"] = numpy.nan  # Consistent with NaN-awareness
        features["blob_ok"] = False
    else:
        # Properties of the center component
        coords = numpy.argwhere(labeled == lbl_at_center)
        features["area"] = float(len(coords))
        
        if len(coords) > 2:
            cov = numpy.cov(coords.T)
            eigvals = numpy.linalg.eigvalsh(cov) # returns [min, max] typically
            l1, l2 = eigvals[-1], eigvals[0] # l1 >= l2
            features["ecc"] = numpy.sqrt(1.0 - (l2 / (l1 + 1e-12)))
    
    # iv) Center dominance ratio
    H, W = patch.shape
    y, x = numpy.ogrid[:H, :W]
    r_grid = numpy.sqrt((y - center_rc[0])**2 + (x - center_rc[1])**2)
    
    inner_mask = r_grid <= inner_r
    ann_mask = (r_grid >= annulus_r0) & (r_grid <= annulus_r1)
    
    if numpy.any(ann_mask):
        ratio = numpy.mean(patch_float[inner_mask]) / (numpy.mean(patch_float[ann_mask]) + 1e-12)
    else:
        ratio = 0.0
    features["center_annulus_ratio"] = ratio
    
    # v) blob_ok definition
    features["blob_ok"] = (
        (features["area"] >= 6) and 
        (features["area"] <= 500) and 
        (features["ecc"] <= 0.85) and 
        (features["center_annulus_ratio"] >= 1.10)
    )
    
    return features


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
    return_candidates=False,
    # Identity Rescue (Opt-in)
    enable_identity_rescue: bool = False,
    identity_patch_half: int = 15,
    identity_min_template_frames: int = 8,
    identity_stride_frames: Optional[int] = None,
    identity_sim_threshold: float = 0.35,
    identity_confidence_rule: str = "blob_ok"
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
    
    t0 = time.perf_counter()
    def log_step(msg: str) -> None:
         dt = time.perf_counter() - t0
         progress_callback(f"[Core +{dt:0.2f}s] {msg}")

    # --- SPATIAL ANALYSIS & GATING ---
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
        cand.metrics.path_node_fraction = len(path) / T # Legacy
        cand.metrics.detected_fraction = n_detected / T
        cand.metrics.max_gap = max_gap
        cand.metrics.max_step = max_step
        cand.metrics.boundary_fail = boundary_fail
        cand.metrics.spatial_jitter_raw = spatial_jitter_raw
        cand.metrics.spatial_jitter_detrended = spatial_jitter_detrended
        
        # GATE LOGIC
        
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

    log_step(f"Trajectory loop complete ({len(pruned_subgraphs)}/{len(pruned_subgraphs)}). Starting post-loop finalize work.")

    log_step("Finalize: Trace Extraction")
    # --- TRACE EXTRACTION ---
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

    log_step("Finalize: Trace Extraction done")
    
    log_step("Finalize: Cellness Metrics")
    # --- CELLNESS METRICS ---
    # Run only for spatially valid + extracted candidates
    stride_step = max(1, T // 50)
    patch_half = 15 # for size 31
    
    # Helper to safely get scalar from potentially all-NaN array
    def safe_nanmedian(values):
        m = numpy.nanmedian(values)
        return float(m) if numpy.isfinite(m) else numpy.nan

    for cand in candidates:
        if cand.is_valid_spatial and cand.trace_extracted:
            valid_patches = []
            
            # Sampling loop
            for t in range(0, T, stride_step):
                y, x = cand.positions[t]
                
                # Extract 31x31 patch
                # Coordinates are stored as [y, x] in int form
                y_start = int(y) - patch_half
                y_end = int(y) + patch_half + 1
                x_start = int(x) - patch_half
                x_end = int(x) + patch_half + 1
                
                # Bounds check strict
                if y_start < 0 or x_start < 0 or y_end > h or x_end > w:
                    continue
                    
                patch = ims[t][y_start:y_end, x_start:x_end]
                if patch.shape != (31, 31):
                    continue
                
                feats = _compute_cellness_features_for_patch(patch, (patch_half, patch_half))
                valid_patches.append(feats)
            
            if valid_patches:
                n_ok = sum(1 for f in valid_patches if f["blob_ok"])
                cand.metrics.cell_fraction = float(n_ok) / len(valid_patches)
                
                # Median aggregation (NaN-aware)
                cand.metrics.cell_logz_median = safe_nanmedian([f["logz"] for f in valid_patches])
                cand.metrics.cell_area_median = safe_nanmedian([f["area"] for f in valid_patches])
                cand.metrics.cell_ecc_median = safe_nanmedian([f["ecc"] for f in valid_patches])
                cand.metrics.cell_center_annulus_ratio_median = safe_nanmedian([f["center_annulus_ratio"] for f in valid_patches])
            else:
                cand.metrics.cell_fraction = 0.0
                cand.metrics.cell_logz_median = 0.0
                cand.metrics.cell_area_median = 0.0
                cand.metrics.cell_ecc_median = 0.0
                cand.metrics.cell_center_annulus_ratio_median = 0.0

    log_step("Finalize: Cellness Metrics done")

    # --- IDENTITY RESCUE (OPTIONAL) ---
    if enable_identity_rescue:
        log_step("Finalize: Identity Rescue")
        # Determine stride for identity pass
        if identity_stride_frames is not None:
            id_stride = max(1, int(identity_stride_frames))
        else:
            id_stride = max(1, T // 50)
             
        for cand in candidates:
            if cand.is_valid_spatial and cand.trace_extracted:
                feats_list = []
                patch_list = []
                template_patches_norm = []
                
                # Collect info
                for t in range(0, T, id_stride):
                    y, x = cand.positions[t]
                    # Extract patch
                    # Note: cand.positions is [y, x] (int)
                    patch = _extract_center_patch(ims[t], int(y), int(x), identity_patch_half)
                    if patch is None:
                        continue
                        
                    feats = _compute_cellness_features_for_patch(patch, (identity_patch_half, identity_patch_half))
                    
                    feats_list.append(feats)
                    patch_list.append(patch)
                    
                    # Confident frame check
                    confident = False
                    if identity_confidence_rule == "blob_ok":
                        confident = feats["blob_ok"]
                        
                    if confident:
                        template_patches_norm.append(_normalize_patch(patch))
                
                # Build template
                template_norm = None
                if len(template_patches_norm) >= identity_min_template_frames:
                    template = _build_template(template_patches_norm, method="median")
                    template_norm = _normalize_patch(template)
                    
                # Compute similarity
                sim_values = []
                ok_robust_count = 0
                valid_count = 0
                
                for i in range(len(patch_list)):
                    patch = patch_list[i]
                    feats = feats_list[i]
                    valid_count += 1
                    
                    sim = 0.0
                    if template_norm is not None:
                        sim = _template_similarity(template_norm, _normalize_patch(patch))
                    sim_values.append(sim)
                    
                    rescue_ok = (template_norm is not None) and (sim >= identity_sim_threshold)
                    ok_robust = feats["blob_ok"] or rescue_ok
                    if ok_robust:
                        ok_robust_count += 1
                        
                # Populate metrics
                cand.metrics.id_template_n = int(len(template_patches_norm))
                
                if valid_count == 0:
                    cand.metrics.cell_fraction_robust = 0.0
                    cand.metrics.id_sim_median = 0.0
                    cand.metrics.id_sim_p10 = 0.0
                    cand.metrics.id_sim_fraction_ge_thr = 0.0
                else:
                    cand.metrics.cell_fraction_robust = float(ok_robust_count) / float(valid_count)
                    cand.metrics.id_sim_median = float(numpy.median(sim_values)) if sim_values else 0.0
                    cand.metrics.id_sim_p10 = float(numpy.percentile(sim_values, 10)) if sim_values else 0.0
                    cand.metrics.id_sim_fraction_ge_thr = float(sum(1 for s in sim_values if s >= identity_sim_threshold)) / float(len(sim_values)) if sim_values else 0.0

        log_step("Finalize: Identity Rescue done")


    # --- FINAL ACCEPTANCE ---
    for cand in candidates:
        # In strict mode, acceptance was already decided by the gate.
        # In scored mode, we accept everything that survived extraction (for now).
        if cand.trace_extracted:
            cand.accepted = True
        else:
            cand.accepted = False
            
    # --- OUTPUT GENERATION ---
    log_step("Analysis complete")
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