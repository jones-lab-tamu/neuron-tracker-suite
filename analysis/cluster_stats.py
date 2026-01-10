"""
Bin-Level Cluster Analysis Module
Performs spatial cluster-based correction on 2D grids of circular phase data.

Logic:
1. Bin-level statistics: Circular mean difference (wrapped) between two groups.
2. Permutation test: GLOBAL Shuffle labels at the animal level (preserving group sizes).
3. Clustering: 8-neighbor connectivity within defined lobes (strictly separated).
4. Correction: Max cluster mass permutation test.
5. Validity: Strict handling of bins failing min_n during permutation (NaN masking).
6. Thresholding: Global cluster-forming threshold T0 (95th percentile of pooled valid nulls).
"""

import numpy as np
from scipy import stats, ndimage
import pandas as pd
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

def _structure_from_connectivity(connectivity: int):
    """Generates NDImage structure based on neighbor count."""
    if connectivity == 4:
        # Edge adjacent only (cross)
        return ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        # Edges + Diagonals (square)
        return ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError(f"Invalid connectivity: {connectivity} (must be 4 or 8)")

def _wrap_to_pi(vals):
    """Wraps angles to [-pi, pi]."""
    return (vals + np.pi) % (2 * np.pi) - np.pi

def _circ_mean(phases):
    """Computes circular mean of a list of phases."""
    if len(phases) == 0:
        return np.nan
    return stats.circmean(phases, high=np.pi, low=-np.pi)

def _find_clusters(sig_mask: np.ndarray, abs_delta_mu: np.ndarray, lobe_mask: np.ndarray, connectivity: int = 8, strict_mode: bool = False, allow_cross_lobe: bool = False):
    """
    Finds clusters of True values in sig_mask within each lobe.
    Returns:
       cluster_map: (H, W) int array of cluster IDs
       cluster_props: list of dicts (id, mass, lobe)
    """
    # Create connectivity structure
    if isinstance(connectivity, (int, np.integer)):
        connectivity = int(connectivity)
        structure = _structure_from_connectivity(connectivity)
    else:
        # Fallback if structure passed directly (internal use case, defensive)
        structure = connectivity
    
    # Process each lobe separately to enforce strict separation
    unique_lobes = np.unique(lobe_mask)
    if not isinstance(unique_lobes, np.ndarray): # Handle scalar case if single lobe
         unique_lobes = np.array([unique_lobes])
    unique_lobes = unique_lobes[unique_lobes != 0] # 0 might be background
    
    full_label_map = np.zeros_like(sig_mask, dtype=int)
    current_max_id = 0
    cluster_props = []
    
    
    if allow_cross_lobe:
        # MODE 2: Global clustering across lobes
        # Mask: Significant AND in any valid lobe (non-zero)
        valid_mask = sig_mask & (lobe_mask > 0)
        
        labeled_array, num_features = ndimage.label(valid_mask, structure=structure)
        full_label_map = labeled_array # In this mode, no need to offset per lobe
        
        if num_features > 0:
            for global_id in range(1, num_features + 1):
                # Get indices
                cluster_bool = (labeled_array == global_id)
                n_bins = np.sum(cluster_bool)
                
                # Mass
                mass_vals = abs_delta_mu[cluster_bool]
                total_mass = np.sum(mass_vals[np.isfinite(mass_vals)])
                
                # Lobes present in this cluster
                cluster_lobes = np.unique(lobe_mask[cluster_bool])
                cluster_lobes = sorted([int(l) for l in cluster_lobes if l > 0])
                
                # Primary lobe assignment for backward compat:
                # If spans multiple, use 0. If single, use that ID.
                if len(cluster_lobes) == 1:
                    primary_lobe = cluster_lobes[0]
                else:
                    primary_lobe = 0
                
                # Coords
                rows, cols = np.where(cluster_bool)
                members = [[int(r), int(c)] for r, c in zip(rows, cols)]
                
                cluster_props.append({
                    'id': int(global_id),
                    'lobe': int(primary_lobe),
                    'lobes': cluster_lobes,
                    'n_bins': int(n_bins),
                    'mass': float(total_mass),
                    'members': json.dumps(members)
                })
    else:
        # MODE 1: Per-lobe separation (Original)
        for lobe_id in unique_lobes:
            # Mask for this lobe AND significant bins
            valid_mask = (sig_mask) & (lobe_mask == lobe_id)
            
            # Label connected components
            labeled_array, num_features = ndimage.label(valid_mask, structure=structure)
            
            if num_features > 0:
                # Shift IDs to be unique across lobes
                labeled_array[labeled_array > 0] += current_max_id
                 
                # Calculate properties
                for local_id in range(1, num_features + 1):
                    global_id = local_id + current_max_id
                    
                    # Get indices for this cluster
                    cluster_bool = (labeled_array == global_id)
                    n_bins = np.sum(cluster_bool)
                    
                    # Calculate mass
                    # Check for NaNs in abs_delta_mu just in case, though they shouldn't be in sig_mask
                    mass_vals = abs_delta_mu[cluster_bool]
                    total_mass = np.sum(mass_vals[np.isfinite(mass_vals)])
                    
                    # Get coords
                    rows, cols = np.where(cluster_bool)
                    # Store as list of [r, c] for JSON serialization
                    members = [[int(r), int(c)] for r, c in zip(rows, cols)]
                    
                    cluster_props.append({
                        'id': int(global_id),
                        'lobe': int(lobe_id),
                        'lobes': [int(lobe_id)],
                        'n_bins': int(n_bins),
                        'mass': float(total_mass),
                        'members': json.dumps(members)
                    })
                
                # Add to full map with overlap check
                mask = labeled_array > 0
                
                # Defensive check for overlap
                overlap_mask = (full_label_map > 0) & mask
                if np.any(overlap_mask):
                    overlap_count = np.sum(overlap_mask)
                    
                    # Sample overlap coordinates for logging
                    rows, cols = np.where(overlap_mask)
                    coords = list(zip(rows, cols))[:10]
                    coord_str = ", ".join([f"({r},{c})" for r, c in coords])
                    if len(rows) > 10: coord_str += "..."
                    
                    msg = (f"Cluster overlap detected at {overlap_count} bins during Lobe {lobe_id} processing. "
                           f"Overlap Coords: {coord_str}. Previous Max ID: {current_max_id}.")
                    
                    if strict_mode:
                        raise ValueError(f"Strict Mode Violation: {msg}")
                    else:
                        logging.warning(msg)
                    
                # Merge with assignment, but DO NOT overwrite existing labels
                write_mask = mask & (~overlap_mask)
                full_label_map[write_mask] = labeled_array[write_mask]
                current_max_id += num_features
            
    return full_label_map, cluster_props

def run_bin_cluster_analysis(
    grouped_phases_grid: Dict[Tuple[int, int], Dict[str, Dict[str, float]]],
    grid_shape: Tuple[int, int],
    lobe_mask: np.ndarray,
    min_n: int = 4,
    n_perm: int = 5000,
    seed: int = 42,
    alpha_forming: float = 0.05,
    alpha_sig: float = 0.05,
    connectivity: int = 4,
    allow_cross_lobe: bool = False,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    alpha: float = None # Legacy compat
):
    """
    Main entry point for bin-level cluster analysis.
    connectivity: 4 (edges) or 8 (edges+diagonals).
    """
    if connectivity not in (4, 8):
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    # Legacy compatibility: if alpha passed but alpha_forming not explicitly set to new value,
    # assume user meant alpha_forming.
    if alpha is not None:
        alpha_val = float(alpha)
        alpha_forming = alpha_val
        # Map to alpha_sig only if caller didn't explicitly override default (0.05)
        # Logic: if alpha_sig is 0.05 (default) and alpha is different,
        # assume legacy single-alpha mode where alpha controls both.
        if alpha_sig == 0.05 and alpha_val != 0.05:
            alpha_sig = alpha_val
         
    rng = np.random.default_rng(seed)
    H, W = grid_shape
    
    t0_start = time.perf_counter()
    if progress_cb: progress_cb("Init", 0, 1, "setup")
    
    # 2. Materialize Bin Data (First Pass - Identify Valid Animals)
    temp_bin_data = [] # List of (r, c, c_dict, e_dict)
    used_animals_control = set()
    used_animals_experiment = set()
    inclusion_mask = np.zeros((H, W), dtype=bool) # Observed inclusion (pre-perm check)
    
    for (r, c), groups in grouped_phases_grid.items():
        if not (0 <= r < H and 0 <= c < W): continue
        if lobe_mask[r, c] == 0: continue
        
        c_dict = groups.get('Control', {})
        e_dict = groups.get('Experiment', {})
        
        # Check observed Min N
        if len(c_dict) < min_n or len(e_dict) < min_n:
            continue
            
        inclusion_mask[r, c] = True
        temp_bin_data.append((r, c, c_dict, e_dict))
        
        # Track animals that ACTUALLY contribute to analysis
        used_animals_control.update(c_dict.keys())
        used_animals_experiment.update(e_dict.keys())
        
    n_valid = len(temp_bin_data)

    # Empty Return Path
    if n_valid == 0:
        logging.warning("No bins met strict Min-N criteria (Control>=min_n AND Experiment>=min_n). Returning empty result.")
        return {
            'delta_mu_map': np.full((H, W), np.nan),
            'p_unc_map': np.ones((H, W)),
            'cluster_map': np.zeros((H, W), dtype=int),
            'inclusion_mask': np.zeros((H, W), dtype=bool),
            'clusters': [],
            'max_mass_distributions': {},
            'T0': np.inf,
            'n_valid_perm_per_bin': np.zeros((H, W), dtype=int),
            'stability_mask': np.zeros((H, W), dtype=bool),
            'alpha_forming': float(alpha_forming),
            'alpha_sig': float(alpha_sig),
            'alpha': float(alpha_sig),
            'seed': int(seed),
            'min_n': int(min_n),
            'n_perm': int(n_perm),
            'connectivity': int(connectivity),
            'allow_cross_lobe': bool(allow_cross_lobe)
        }

    # 3. Construct Permutation Universe (Restricted)
    # Only include animals that appeared in at least one valid bin.
    # This ensures "silent" animals don't dilute the permutation space.
    
    # Check for overlap within the USED set
    if not used_animals_control.isdisjoint(used_animals_experiment):
        overlap = used_animals_control.intersection(used_animals_experiment)
        raise ValueError(f"Animal ID overlap detected in valid bins: {list(overlap)[:5]}...")
        
    all_animals_list = sorted(list(used_animals_control.union(used_animals_experiment)))
    n_total_animals = len(all_animals_list)
    animal_to_idx = {anim: i for i, anim in enumerate(all_animals_list)}
    
    # Original Group Labels (0=Control, 1=Experiment)
    original_labels = np.full(n_total_animals, -1, dtype=int)
    
    for anim in used_animals_control:
        original_labels[animal_to_idx[anim]] = 0
    for anim in used_animals_experiment:
        original_labels[animal_to_idx[anim]] = 1
        
    if np.any(original_labels == -1):
        raise ValueError("Logic Error: Some used animals were not assigned a group.")

    # LOGGING: Permutation Universe Stats
    n_ctrl = len(used_animals_control)
    n_exp = len(used_animals_experiment)
    logging.info(f"Permutation Universe: n_total={n_total_animals}, n_ctrl={n_ctrl}, n_exp={n_exp}, "
                 f"n_valid_bins={n_valid}, min_n={min_n}, n_perm={n_perm}, alpha_forming={alpha_forming}, alpha_sig={alpha_sig}, seed={seed}")
    logging.info(f"ClusterStats: allow_cross_lobe={allow_cross_lobe}, connectivity={connectivity}, alpha_forming={alpha_forming}, alpha_sig={alpha_sig}, n_perm={n_perm}")

    # 4. Finalize Bin Data with Indices
    t1_obs = time.perf_counter()
    if progress_cb: progress_cb("Compute observed maps", 0, 1, "processing bins")
    valid_bins_data = [] # List of dicts
    valid_bin_ctrl_dicts = [] # Helper for matrix fill
    valid_bin_exp_dicts = [] # Helper for matrix fill
    obs_delta_mu_linear = []
    
    for (r, c, c_dict, e_dict) in temp_bin_data:
        # Compute Observed Stat
        phases_c = list(c_dict.values())
        phases_e = list(e_dict.values())
        mu_c = _circ_mean(phases_c)
        mu_e = _circ_mean(phases_e)
        dm = _wrap_to_pi(mu_e - mu_c)
        obs_delta_mu_linear.append(dm)
        
        # Prepare for permutation
        bin_indices = []
        bin_phases = []
        
        for anim, ph in c_dict.items():
            bin_indices.append(animal_to_idx[anim])
            bin_phases.append(ph)
        for anim, ph in e_dict.items():
            bin_indices.append(animal_to_idx[anim])
            bin_phases.append(ph)
            
        valid_bins_data.append({
            'c': c,
            # Store flattened index into matrices for reference if needed
            'flat_idx': len(valid_bins_data)
        })
        
        valid_bin_ctrl_dicts.append(c_dict)
        valid_bin_exp_dicts.append(e_dict)
        
    # Vectorization Setup: Build (N_animals_total, n_valid) matrices
    # Initialize with 0.0
    matrices_sin = np.zeros((n_total_animals, n_valid), dtype=np.float32)
    matrices_cos = np.zeros((n_total_animals, n_valid), dtype=np.float32)
    # VALID matrix: 1 if animal has data for bin, 0 otherwise. Used for Min-N counts.
    matrices_valid = np.zeros((n_total_animals, n_valid), dtype=np.uint8)
    
    # Defensive checks (Runtime integrity)
    if len(valid_bin_ctrl_dicts) != n_valid or len(valid_bin_exp_dicts) != n_valid:
        raise ValueError(f"Logic Error: Valid bin dicts count mismatch. n_valid={n_valid}, ctrl={len(valid_bin_ctrl_dicts)}, exp={len(valid_bin_exp_dicts)}, bins={len(valid_bins_data)}")
        
    # Check shape integrity strictly
    if matrices_sin.shape[1] != n_valid or matrices_cos.shape[1] != n_valid or matrices_valid.shape[1] != n_valid:
        raise ValueError(f"Logic Error: Matrix shape mismatch. Expected width {n_valid}, got sin={matrices_sin.shape}, cos={matrices_cos.shape}, valid={matrices_valid.shape}")
    
    # Fill matrices using ONLY valid-bin index j
    for j in range(n_valid):
        c_dict = valid_bin_ctrl_dicts[j]
        e_dict = valid_bin_exp_dicts[j]
        
        # Control animals in this bin
        for anim, ph in c_dict.items():
            idx = animal_to_idx[anim]
            matrices_sin[idx, j] = np.sin(ph)
            matrices_cos[idx, j] = np.cos(ph)
            matrices_valid[idx, j] = 1
            
        # Experiment animals in this bin
        for anim, ph in e_dict.items():
            idx = animal_to_idx[anim]
            matrices_sin[idx, j] = np.sin(ph)
            matrices_cos[idx, j] = np.cos(ph)
            matrices_valid[idx, j] = 1
    
    obs_delta_mu_linear = np.array(obs_delta_mu_linear)
    obs_abs_delta_mu_linear = np.abs(obs_delta_mu_linear)
    
    t2_perm = time.perf_counter()
    
    # 3. Permutation Loop (Global)
    # Initialize with NaNs to mask invalid perms
    perms_linear = np.full((n_perm, n_valid), np.nan, dtype=np.float32)
    
    unique_lobes = sorted(set(int(x) for x in np.unique(lobe_mask) if int(x) != 0))
    # Stats storage:
    # If cross-lobe OFF: key=lobe_id, val=array of max masses
    # If cross-lobe ON: key='global', val=array of max masses (single null distribution)
    max_mass_dist = {}
    if allow_cross_lobe:
        max_mass_dist['global'] = np.zeros(n_perm)
    else:
        for lobe in unique_lobes:
            max_mass_dist[int(lobe)] = np.zeros(n_perm)
    
    REPORT_EVERY = max(10, n_perm // 100)
    if progress_cb: progress_cb("Permutations", 0, n_perm, "starting")

    for k in range(n_perm):
        if progress_cb and (k % REPORT_EVERY == 0):
            elapsed = time.perf_counter() - t2_perm
            rate = (k + 1) / max(elapsed, 1e-9)
            eta = (n_perm - (k + 1)) / max(rate, 1e-9)
            progress_cb("Permutations", k + 1, n_perm, f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s")
            
        # Global Shuffle of labels (0=Control, 1=Experiment)
        shuffled_labels = rng.permutation(original_labels)
        
        # Create masks for groups (n_total_animals,)
        mask_c = (shuffled_labels == 0)
        mask_e = (shuffled_labels == 1)
        
        # Vectorized Summation: (n_valid_bins,)
        # Sum Sin/Cos for Control
        sin_sum_c = np.sum(matrices_sin[mask_c, :], axis=0)
        cos_sum_c = np.sum(matrices_cos[mask_c, :], axis=0)
        n_c = np.sum(matrices_valid[mask_c, :], axis=0)
        
        # Sum Sin/Cos for Experiment
        sin_sum_e = np.sum(matrices_sin[mask_e, :], axis=0)
        cos_sum_e = np.sum(matrices_cos[mask_e, :], axis=0)
        n_e = np.sum(matrices_valid[mask_e, :], axis=0)
        
        # Validity Mask for this Permutation (Vectorized Min-N)
        valid_perm_mask = (n_c >= min_n) & (n_e >= min_n)
        
        # Compute Circular Means (where valid)
        # arctan2 handles (y, x) -> (sin, cos)
        mu_c = np.arctan2(sin_sum_c, cos_sum_c)
        mu_e = np.arctan2(sin_sum_e, cos_sum_e)
        
        # Delta & Wrap
        dm = _wrap_to_pi(mu_e - mu_c)
        
        # Store absolute delta mu, or NaN where invalid
        # Note: We use np.where to safely handle potential invalid arithmetics in invalid bins
        # though arctan2 is generally robust (0,0 gives 0).
        perms_linear[k, :] = np.where(valid_perm_mask, np.abs(dm), np.nan)
            
    if progress_cb: progress_cb("Permutations", n_perm, n_perm, "done")
    t3_stats = time.perf_counter()
            
    # 4. Statistics & Clustering
    
    # Stability Check
    # How many valid permutations per bin?
    valid_mask = np.isfinite(perms_linear)
    n_valid_per_bin = np.sum(valid_mask, axis=0)
    stability_threshold = max(100, 0.2 * n_perm)
    
    # Initial stability mask based on threshold
    is_stable_bin_initial = n_valid_per_bin >= stability_threshold
    
    # Strict Guard: valid_per_bin MUST be > 0 (redundant if threshold >=1, but explicit safety)
    has_any_finite = n_valid_per_bin > 0
    is_stable_bin = is_stable_bin_initial & has_any_finite
    
    # Log warning if we dropped bins due to the strict >0 check (unlikely but possible if threshold=0)
    # REPLACE WITH NEW LOGGING
    
    # A) ZERO-FINITE WARNING (reachable, directly computed)
    zero_finite_idx = np.where(n_valid_per_bin == 0)[0]
    if zero_finite_idx.size > 0:
        sample_coords = []
        for idx in zero_finite_idx[:5]:
            d = valid_bins_data[idx]
            sample_coords.append(f"({d['r']},{d['c']})")
        logging.warning(
            f"Found {zero_finite_idx.size} bins with 0 finite permutation stats (always unstable). "
            f"Sample: {', '.join(sample_coords)}"
        )

    # B) UNSTABLE-BUT-OBSERVED-INCLUDED WARNING (actionable)
    # Bins that fail stability (~is_stable_bin) but have SOME valid data (n_valid_per_bin > 0).
    # This prevents overlap with Warning A (0 valid).
    unstable_idx = np.where((~is_stable_bin) & (n_valid_per_bin > 0))[0]
    
    if unstable_idx.size > 0:
        sample_coords = []
        for idx in unstable_idx[:5]:
            d = valid_bins_data[idx]
            sample_coords.append(f"({d['r']},{d['c']})")
        logging.warning(
            f"Excluded {unstable_idx.size} bins that were observed-included but failed stability threshold ({stability_threshold:.1f}). "
            f"Sample: {', '.join(sample_coords)}"
        )
    
    # A. Uncorrected P-values (Bin-wise)
    p_unc_linear = np.ones(n_valid) # Default 1.0
    
    for i in range(n_valid):
        if is_stable_bin[i]:
            # Get valid permutation stats for this bin
            valid_stats = perms_linear[valid_mask[:, i], i]
            obs = obs_abs_delta_mu_linear[i]
            # (1 + count(perm >= obs)) / (1 + n_valid)
            n_greater = np.sum(valid_stats >= obs)
            p = (1.0 + n_greater) / (1.0 + len(valid_stats))
            p_unc_linear[i] = p
            
    # B. Global Cluster-Forming Threshold (T0) (Option A)
    # Pool ALL valid permutation statistics from STABLE bins
    pooled_null = perms_linear[:, is_stable_bin] # Shape (n_perm, n_stable)
    pooled_null = pooled_null[np.isfinite(pooled_null)] # Flatten and remove NaNs
    
    if pooled_null.size == 0:
        # Failsafe if everything collapsed
        T0 = np.inf
    else:
        # Use alpha_forming for T0
        T0 = np.percentile(pooled_null, 100 * (1.0 - alpha_forming))
                 
    # C. Max Cluster Mass Distribution (per lobe, per permutation)
    for k in range(n_perm):
        # Linear mask for this perm: valid AND > T0
        row_vals = perms_linear[k, :]
        is_finite = np.isfinite(row_vals)
        
        # Combined mask: Stable AND Finite
        sig_indices = (is_stable_bin & is_finite)
        
        # Threshold: stat > T0
        with np.errstate(invalid='ignore'):
            is_sig = (row_vals > T0) & sig_indices
        
        # Build maps
        sig_perm_map = np.zeros((H, W), dtype=bool)
        mass_perm_map = np.zeros((H, W))
        
        for i, bdata in enumerate(valid_bins_data):
            if is_sig[i]:
                r, c = bdata['r'], bdata['c']
                sig_perm_map[r, c] = True
                mass_perm_map[r, c] = row_vals[i] # Mass is the permuted stat
        
        
        # Clustering
        # Use consistent connectivity for null distribution
        _, props = _find_clusters(sig_perm_map, mass_perm_map, lobe_mask, connectivity=connectivity, allow_cross_lobe=allow_cross_lobe)
        
        # Max mass logic
        if allow_cross_lobe:
            # Global Null
            cur_max = 0.0
            for p in props:
                if p['mass'] > cur_max:
                    cur_max = p['mass']
            max_mass_dist['global'][k] = cur_max
        else:
            # Per-Lobe Null
            current_maxes = {int(l): 0.0 for l in unique_lobes}
            for p in props:
                l = int(p['lobe'])
                m = p['mass']
                if m > current_maxes.get(l, 0):
                    current_maxes[l] = m
            for l in unique_lobes:
                max_mass_dist[int(l)][k] = current_maxes[int(l)]
            
    # 5. Observed Clusters
    # Reconstruct Maps
    p_unc_map = np.ones((H, W))
    delta_mu_map = np.full((H, W), np.nan)
    n_valid_perm_map = np.zeros((H, W), dtype=int)
    stability_mask = np.zeros((H, W), dtype=bool)
    
    for i, bdata in enumerate(valid_bins_data):
        r, c = bdata['r'], bdata['c']
        p_unc_map[r, c] = p_unc_linear[i]
        delta_mu_map[r, c] = obs_delta_mu_linear[i]
        n_valid_perm_map[r, c] = n_valid_per_bin[i]
        stability_mask[r, c] = is_stable_bin[i]
        
    # Observed Forming Mask: Stable AND > T0
    # OPTION 1: Apply all masks BEFORE labeling
    # We construct 'final_inclusion' and 'valid_obs_mask' and 'T0' check combined.
    # This forms the 'sig_mask_obs' which is the SINGLE source of truth for labeling.
    final_inclusion = (inclusion_mask & stability_mask)
    valid_obs_mask = np.isfinite(delta_mu_map)
    
    # Threshold check
    with np.errstate(invalid='ignore'):
         sig_mask_obs = (final_inclusion & valid_obs_mask) & (np.abs(delta_mu_map) > T0)
    
    # Debug Logging (Part C)
    unique_lobes_log = sorted(set(int(x) for x in np.unique(lobe_mask) if int(x) != 0))
    for l_id in unique_lobes_log:
        # Check eligibility for this lobe
        eligible = sig_mask_obs & (lobe_mask == l_id)
        # Check components
        structure_check = _structure_from_connectivity(connectivity)
        _, n_cc_log = ndimage.label(eligible, structure=structure_check)
        logging.info(f"ClusterLabeling: lobe={l_id}, eligible_bins={int(eligible.sum())}, n_components={n_cc_log}, connectivity={connectivity}")

    # Find Clusters (Mass using abs delta mu)
    # The _find_clusters function applies (sig_mask_obs & lobe_mask) then labels.
    # This aligns with Option 1: Mask then Label. 
    # No further masking is applied to the map.
    cluster_map_obs, cluster_props_obs = _find_clusters(sig_mask_obs, np.abs(delta_mu_map), lobe_mask, connectivity=connectivity, strict_mode=True, allow_cross_lobe=allow_cross_lobe)
    
    final_clusters = []
    for p in cluster_props_obs:
        obs_mass = p['mass']
        
        if allow_cross_lobe:
            # Correct against Global Null
            null_dist = max_mass_dist['global']
            lobe = p['lobe'] # might be 0
        else:
            # Correct against Specific Lobe Null
            lobe = int(p['lobe'])
            if lobe not in max_mass_dist:
                raise KeyError(f"Lobe ID {lobe} not found in max_mass_dist keys: {list(max_mass_dist.keys())}")
            null_dist = max_mass_dist[lobe]
            
        if len(null_dist) == 0:
            key_str = "global" if allow_cross_lobe else f"lobe={lobe}"
            raise ValueError(
                f"Empty null distribution for {key_str}. "
                f"n_perm={n_perm}, allow_cross_lobe={allow_cross_lobe}, connectivity={connectivity}, "
                f"alpha_forming={alpha_forming}, alpha_sig={alpha_sig}."
            )
        else:
            # P-value = (1 + sum(null >= obs)) / (1 + K)
            n_greater = np.sum(null_dist >= obs_mass)
            p_corr = (1.0 + n_greater) / (1.0 + len(null_dist))
            
        p['p_corr'] = float(p_corr)
        final_clusters.append(p)
        
    # Part A: Hard Diagnostic Assertion
    # Check that each cluster ID in result corresponds to exactly 1 connected component
    structure_check = _structure_from_connectivity(connectivity)
    for clust in final_clusters:
        cid = clust["id"]
        lobe = clust["lobe"]
        # Extract binary mask for this cluster
        mask_c = (cluster_map_obs == cid)
        
        # Count connected components
        _, n_cc_check = ndimage.label(mask_c, structure=structure_check)
        
        if n_cc_check != 1:
            msg = f"Cluster integrity failure: cid={cid}, lobe={lobe}, n_cc={n_cc_check}, n_bins={int(mask_c.sum())} (conn={connectivity})"
            # Log it first just in case
            logging.error(msg)
            raise ValueError(msg)
            
    t4_end = time.perf_counter()
    
    dt_init = t1_obs - t0_start
    dt_obs = t2_perm - t1_obs
    dt_perm = t3_stats - t2_perm
    dt_stats = t4_end - t3_stats
    
    logging.info(f"ClusterStats Timing (sec): Init={dt_init:.3f}, Obs={dt_obs:.3f}, Perms={dt_perm:.3f}, Stats={dt_stats:.3f}")
    
    if progress_cb: progress_cb("Finalize observed clusters", n_perm, n_perm, "complete")

    return {
        'delta_mu_map': delta_mu_map, 
        'p_unc_map': p_unc_map,
        'cluster_map': cluster_map_obs,
        'inclusion_mask': final_inclusion,
        'clusters': final_clusters,
        'max_mass_distributions': max_mass_dist,
        'T0': float(T0),
        'n_valid_perm_per_bin': n_valid_perm_map,
        'stability_mask': stability_mask,
        'alpha_forming': float(alpha_forming),
        'alpha_sig': float(alpha_sig),
        'alpha': float(alpha_sig),
        'seed': int(seed),
        'min_n': int(min_n),
        'n_perm': int(n_perm),
        'connectivity': int(connectivity),
        'allow_cross_lobe': bool(allow_cross_lobe)
    }

def save_cluster_results(results: Dict[str, Any], output_dir: str):
    """Saves analysis artifacts to disk."""
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.savez_compressed(os.path.join(output_dir, 'maps.npz'),
        delta_mu=results['delta_mu_map'],
        p_unc=results['p_unc_map'],
        cluster_map=results['cluster_map'],
        inclusion=results['inclusion_mask'],
        n_valid_perm=results['n_valid_perm_per_bin'],
        T0=np.array([results['T0']])
    )
    
    # Save Clusters CSV
    clusters = results['clusters']
    if clusters:
        pd.DataFrame(clusters).to_csv(os.path.join(output_dir, 'cluster_results.csv'), index=False)
    else:
        # Empty file to indicate no clusters
        pd.DataFrame(columns=['id', 'lobe', 'n_bins', 'mass', 'p_corr']).to_csv(os.path.join(output_dir, 'cluster_results.csv'), index=False)
        
    # Save Meta
    meta = {
        'T0': results['T0'],
        'n_clusters': len(clusters),
        'alpha': results.get('alpha', 0.05),
        'seed': results.get('seed', 'unknown'),
        'min_n': results.get('min_n', 'unknown'),
        'n_perm': results.get('n_perm', 'unknown')
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Saved cluster results to {output_dir}")

def plot_cluster_results(results: Dict[str, Any], output_path: str):
    """
    Creates a sanity heatmap of Delta Mu with cluster outlines.
    """
    import matplotlib.pyplot as plt
    try:
        from matplotlib.patches import Patch
    except ImportError:
        pass
        
    delta_mu = results['delta_mu_map']
    p_unc = results['p_unc_map']
    cluster_map = results['cluster_map']
    # significant clusters only
    alpha = results.get('alpha', 0.05)
    sig_clusters = [c['id'] for c in results['clusters'] if c['p_corr'] < alpha]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Delta Mu
    im1 = axes[0].imshow(delta_mu, cmap='coolwarm', vmin=-np.pi, vmax=np.pi, origin='lower')
    axes[0].set_title('Delta Mu (Exp - Ctrl)')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. P Uncorrected
    im2 = axes[1].imshow(p_unc, cmap='viridis_r', vmin=0, vmax=0.1, origin='lower')
    axes[1].set_title('Uncorrected P-values')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Clusters (Significant Only)
    # create custom cmap: 0=bg, others=random or distinct
    # Mask non-sig clusters
    final_cluster_map = np.zeros_like(cluster_map)
    for cid in sig_clusters:
        final_cluster_map[cluster_map == cid] = cid
        
    im3 = axes[2].imshow(final_cluster_map, cmap='tab20', interpolation='nearest', origin='lower')
    axes[2].set_title(f'T0={results["T0"]:.3f} | Sig Clusters (a={alpha})')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
