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
from typing import Dict, List, Tuple, Optional, Any
import logging
import json

def _wrap_to_pi(vals):
    """Wraps angles to [-pi, pi]."""
    return (vals + np.pi) % (2 * np.pi) - np.pi

def _circ_mean(phases):
    """Computes circular mean of a list of phases."""
    if len(phases) == 0:
        return np.nan
    return stats.circmean(phases, high=np.pi, low=-np.pi)

def _find_clusters(sig_mask: np.ndarray, abs_delta_mu: np.ndarray, lobe_mask: np.ndarray, connectivity=8, strict_mode=False):
    """
    Finds clusters of True values in sig_mask within each lobe.
    Returns:
       cluster_map: (H, W) int array of cluster IDs
       cluster_props: list of dicts (id, mass, lobe)
    """
    # Create structure for 8-connectivity
    structure = ndimage.generate_binary_structure(2, 2) # 8-neighbor
    
    # Process each lobe separately to enforce strict separation
    unique_lobes = np.unique(lobe_mask)
    if not isinstance(unique_lobes, np.ndarray): # Handle scalar case if single lobe
         unique_lobes = np.array([unique_lobes])
    unique_lobes = unique_lobes[unique_lobes != 0] # 0 might be background
    
    full_label_map = np.zeros_like(sig_mask, dtype=int)
    current_max_id = 0
    cluster_props = []
    
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
    alpha: float = 0.05
):
    """
    Main entry point for bin-level cluster analysis.
    """
    rng = np.random.default_rng(seed)
    H, W = grid_shape
    

    
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
            'alpha': float(alpha),
            'seed': int(seed),
            'min_n': int(min_n),
            'n_perm': int(n_perm)
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
                 f"n_valid_bins={n_valid}, min_n={min_n}, n_perm={n_perm}, alpha={alpha}, seed={seed}")

    # 4. Finalize Bin Data with Indices
    valid_bins_data = [] # List of dicts
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
            'r': r,
            'c': c,
            'indices': np.array(bin_indices, dtype=int),
            'phases': np.array(bin_phases, dtype=float)
        })
    
    obs_delta_mu_linear = np.array(obs_delta_mu_linear)
    obs_abs_delta_mu_linear = np.abs(obs_delta_mu_linear)
    
    # 3. Permutation Loop (Global)
    # Initialize with NaNs to mask invalid perms
    perms_linear = np.full((n_perm, n_valid), np.nan, dtype=np.float32)
    
    unique_lobes = np.unique(lobe_mask[lobe_mask > 0])
    max_mass_dist = {lobe: np.zeros(n_perm) for lobe in unique_lobes}
    
    for k in range(n_perm):
        # Global Shuffle
        shuffled_labels = rng.permutation(original_labels)
        
        # Compute for all valid bins using THESE labels
        for i, bdata in enumerate(valid_bins_data):
            # Extract labels for animals in this bin
            bin_anim_indices = bdata['indices']
            bin_labs = shuffled_labels[bin_anim_indices]
            bin_phs = bdata['phases']
            
            # Split
            mask_c = (bin_labs == 0)
            mask_e = (bin_labs == 1)
            
            # Strict Min N check for VALIDITY
            if np.sum(mask_c) < min_n or np.sum(mask_e) < min_n:
                # Leave as NaN
                continue
                
            pc = bin_phs[mask_c]
            pe = bin_phs[mask_e]
            
            dm = _wrap_to_pi(_circ_mean(pe) - _circ_mean(pc))
            perms_linear[k, i] = abs(dm)
            
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
        T0 = np.percentile(pooled_null, 100 * (1.0 - alpha))
                 
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
        _, props = _find_clusters(sig_perm_map, mass_perm_map, lobe_mask)
        
        # Max mass logic
        current_maxes = {l: 0.0 for l in unique_lobes}
        for p in props:
            l = p['lobe']
            m = p['mass']
            if m > current_maxes.get(l, 0):
                current_maxes[l] = m
        for l in unique_lobes:
            max_mass_dist[l][k] = current_maxes[l]
            
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
    final_inclusion = (inclusion_mask & stability_mask)
    valid_obs_mask = np.isfinite(delta_mu_map)
    
    # Threshold check
    with np.errstate(invalid='ignore'):
         sig_mask_obs = (final_inclusion & valid_obs_mask) & (np.abs(delta_mu_map) > T0)
    
    # Find Clusters (Mass using abs delta mu)
    cluster_map_obs, cluster_props_obs = _find_clusters(sig_mask_obs, np.abs(delta_mu_map), lobe_mask)
    
    final_clusters = []
    for p in cluster_props_obs:
        obs_mass = p['mass']
        lobe = p['lobe']
        null_dist = max_mass_dist[lobe]
        n_beats = np.sum(null_dist >= obs_mass)
        p_corr = (1.0 + n_beats) / (1.0 + n_perm)
        p.update({'p_corr': p_corr})
        final_clusters.append(p)
        
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
        'alpha': float(alpha),
        'seed': int(seed),
        'min_n': int(min_n),
        'n_perm': int(n_perm)
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
