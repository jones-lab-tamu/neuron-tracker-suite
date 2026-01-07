"""
Circular Statistics Analysis Module
Performs genotype x region interaction tests on circular phase data using
animal-level summaries.

Primary Method: Permutation test for interaction.
Hypothesis: Is the pattern of phase differences between groups constant across zones?
Null: Interaction is zero (differences are constant).

Usage:
    python -m analysis.circular_stats --in zone_phase_animal_summary.csv --out results.json --nperm 10000
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
from collections.abc import Callable
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Circular Stats Analysis")
    # Support backward compatible flags but prefer --in and --out
    parser.add_argument("--in", "--in_file", dest="in_file", type=str, required=True, help="Input CSV (animal x zone summary)")
    parser.add_argument("--out", "--out_file", dest="out_file", type=str, required=True, help="Output JSON results")
    parser.add_argument("--nperm", type=int, default=10000, help="Number of permutations")
    parser.add_argument("--min_cells", type=int, default=5, help="Min cells per animal-zone to include")
    parser.add_argument("--min_animals_zone", type=int, default=2, help="Min animals per group per zone to keep zone")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()

def zone_group_counts_ok(df, zones, groups, min_animals_zone) -> bool:
    """
    Checks if all specified zones have enough unique animals per group.
    """
    g1, g2 = groups[0], groups[1]
    
    # Optimization: Restrict to relevant zones first
    df_sub = df[df['Zone_Name'].isin(zones)]
    counts = df_sub.groupby(['Zone_Name', 'Group'])['Animal_ID'].nunique()
    
    for z in zones:
        # Check Group 1
        if counts.get((z, g1), 0) < min_animals_zone:
            return False
        # Check Group 2
        if counts.get((z, g2), 0) < min_animals_zone:
            return False
            
    return True

def calculate_interaction_stat(df, groups, zones):
    """
    Computes interaction statistic T based on dispersion of group differences across zones.
    df: DataFrame with 'Animal_ID', 'Group', 'Zone_Name', 'cos', 'sin'
    groups: list of 2 group names [Control, Exp]
    zones: list of zone names
    
    Statistic:
    For each zone z:
      u_g1,z = mean unit vector group 1
      u_g2,z = mean unit vector group 2
      delta_z = u_g2,z - u_g1,z  (vector in R2)
    
    T = sum_z || delta_z - mean_z(delta_z) ||^2
    """
    g1, g2 = groups[0], groups[1]
    
    delta_vecs = []
    
    for z in zones:
        df_z = df[df['Zone_Name'] == z]
        
        # Group 1
        df_g1 = df_z[df_z['Group'] == g1]
        
        # Group 2
        df_g2 = df_z[df_z['Group'] == g2]
        
        # Strict: Zones passed in MUST be valid (pre-filtered).
        # Double check locally just in case
        if len(df_g1) == 0 or len(df_g2) == 0:
            raise RuntimeError(f"Zone '{z}' missing group data after validation. This invalidates the permutation test.")
            
        # Mean vectors
        v1 = np.array([df_g1['cos'].mean(), df_g1['sin'].mean()])
        norm1 = np.linalg.norm(v1)
        u1 = v1 / norm1 if norm1 > 1e-9 else v1
        
        v2 = np.array([df_g2['cos'].mean(), df_g2['sin'].mean()])
        norm2 = np.linalg.norm(v2)
        u2 = v2 / norm2 if norm2 > 1e-9 else v2
        
        delta = u2 - u1
        delta_vecs.append(delta)
        
    delta_vecs = np.array(delta_vecs) # Shape (n_valid_zones, 2)
    
    if len(delta_vecs) < 1:
        return 0.0 # Should be caught earlier
        
    # Mean delta vector across zones
    mean_delta = np.mean(delta_vecs, axis=0) # Shape (2,)
    
    # T = sum squared errors from mean delta
    # sum of squared euclidean norms of (vec - mean)
    diffs = delta_vecs - mean_delta
    sq_norms = np.sum(diffs**2, axis=1)
    T = np.sum(sq_norms)
    
    return T

from collections.abc import Callable

def run_interaction_test_from_df(
    df: pd.DataFrame,
    nperm: int = 10000,
    min_cells: int = 5,
    min_animals_zone: int = 2,
    seed: int | None = None,
    max_attempts_factor: int = 50,
    should_cancel: Callable[[], bool] | None = None
) -> dict:
    """
    Runs the full interaction test analysis from a DataFrame.
    """
    # Use local RNG for reproducibility and thread safety
    rng = np.random.default_rng(seed)
        
    # Validation
    req_cols = ['Animal_ID', 'Group', 'Zone_Name', 'Mean_Phase_Radians', 'N_Cells']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in input data")
            
    # Filter Min Cells
    n_input_rows = len(df)
    # Work on a copy internally
    df_params = df[df['N_Cells'] >= min_cells].copy()
    n_rows_after_min_cells = len(df_params)
    
    # Integrity: 1 Group per Animal
    animal_integrity = df_params.groupby('Animal_ID')['Group'].nunique()
    if (animal_integrity > 1).any():
        bad_ids = animal_integrity[animal_integrity > 1].index.tolist()
        raise ValueError(f"Inconsistent Group labels for animals: {bad_ids}")
        
    # Precompute cos/sin
    df_params['cos'] = np.cos(df_params['Mean_Phase_Radians'])
    df_params['sin'] = np.sin(df_params['Mean_Phase_Radians'])
    
    # Identify Valid Zones
    all_groups = sorted(df_params['Group'].unique())
    if len(all_groups) != 2:
        raise ValueError(f"Interaction test requires exactly 2 groups, found {len(all_groups)}: {all_groups}")
        
    groups = all_groups # Exactly 2
    
    valid_zones = []
    dropped_zones = [] # List of dicts with reasons
    unique_zones = sorted(df_params['Zone_Name'].unique())
    
    for z in unique_zones:
        df_z = df_params[df_params['Zone_Name'] == z]
        g1_count = df_z[df_z['Group'] == groups[0]]['Animal_ID'].nunique()
        g2_count = df_z[df_z['Group'] == groups[1]]['Animal_ID'].nunique()
        
        if g1_count >= min_animals_zone and g2_count >= min_animals_zone:
            valid_zones.append(z)
        else:
            dropped_zones.append({
                'Zone_Name': z,
                'counts': {groups[0]: g1_count, groups[1]: g2_count},
                'reason': f"below min_animals_zone ({min_animals_zone})"
            })
            
    if len(valid_zones) < 2:
        msg = [f"Need at least 2 valid zones (>= {min_animals_zone} animals/group) for interaction test. Found {len(valid_zones)}: {valid_zones}"]
        if dropped_zones:
            msg.append("Dropped Zones:")
            for d in dropped_zones[:10]: # Limit to first 10 to avoid spam
                msg.append(f"  - {d['Zone_Name']}: {d['reason']} (Counts: {d['counts']})")
            if len(dropped_zones) > 10:
                msg.append(f"  ... and {len(dropped_zones)-10} more.")
        raise ValueError("\n".join(msg))
        
    # Filter DF to valid zones only
    df_params = df_params[df_params['Zone_Name'].isin(valid_zones)].copy()
    
    # Check total animals per group
    g_counts = df_params.groupby('Group')['Animal_ID'].nunique()
    if (g_counts < 2).any():
        raise ValueError(f"Need at least 2 animals per group overall. Found: {g_counts.to_dict()}")
        
    # Observed Statistic
    T_obs = calculate_interaction_stat(df_params, groups, valid_zones)
    
    # Permutation Test
    animal_group_map = df_params.groupby('Animal_ID')['Group'].first()
    unique_animals = animal_group_map.index.values
    original_labels = animal_group_map.values
    
    perm_stats = []
    
    # Create a clean DF for calculation to avoid constant copying
    df_calc = df_params.copy() 
    
    attempts = 0
    max_attempts = nperm * max_attempts_factor
    accepted_count = 0
    
    while accepted_count < nperm:
        # Cancellation check
        if should_cancel and should_cancel():
            raise RuntimeError("Operation cancelled by user.")
            
        attempts += 1
        if attempts > max_attempts:
            rejected = attempts - accepted_count
            rej_rate = (float(rejected) / attempts) * 100.0
            
            msg = [
                f"Error: Exceeded max attempts ({max_attempts}) to find valid permutations.",
                f"Accepted {accepted_count}/{nperm} permutations.",
                f"Attempts: {attempts}",
                f"Rejected: {rejected}",
                f"Rejection rate: {rej_rate:.2f}%",
                f"Total Animals: {len(unique_animals)}",
                f"Groups: {groups}",
                f"Min Animals per Zone: {min_animals_zone}",
                f"Valid Zones: {valid_zones}",
                "",
                "Original Counts per Zone (Diagnostic):"
            ]
            
            # Diagnostic from original df_params
            diag_counts = df_params.groupby(['Zone_Name', 'Group'])['Animal_ID'].nunique()
            
            msg.append(f"  {'Zone_Name':<20} {groups[0]:<10} {groups[1]:<10}")
            msg.append("  " + "-"*42)
            
            for z in valid_zones:
                c1 = diag_counts.get((z, groups[0]), 0)
                c2 = diag_counts.get((z, groups[1]), 0)
                msg.append(f"  {z:<20} {c1:<10} {c2:<10}")
                
            msg.append("")
            msg.append("Suggested fixes:")
            msg.append("  - Lower --min_animals_zone")
            msg.append("  - Increase sample size per group")
            msg.append("  - Check for sparse zones or missing data")
            
            raise RuntimeError("\n".join(msg))
            
        # Shuffle labels using local RNG
        # rng.permutation returns a copy, safe to use
        shuffled_labels = rng.permutation(original_labels)
        shuffled_map = dict(zip(unique_animals, shuffled_labels))
        
        # Apply strict mapping
        df_calc['Group'] = df_calc['Animal_ID'].map(shuffled_map)
        
        # Strict Validity Check
        if not zone_group_counts_ok(df_calc, valid_zones, groups, min_animals_zone):
            continue
            
        t_perm = calculate_interaction_stat(df_calc, groups, valid_zones)
        perm_stats.append(t_perm)
        accepted_count += 1
        
    perm_stats = np.array(perm_stats)
    p_val = (1.0 + np.sum(perm_stats >= T_obs)) / (1.0 + nperm)
    
    # Descriptive Stats
    desc = df_params.groupby(['Group', 'Zone_Name']).agg({
        'R_Resultant': 'mean',
        'N_Cells': 'sum',
        'Animal_ID': 'nunique'
    }).rename(columns={'Animal_ID': 'N_Animals', 'N_Cells': 'Total_Cells'})
    
    circ_means = df_params.groupby(['Group', 'Zone_Name']).apply(
        lambda x: np.arctan2(x['sin'].mean(), x['cos'].mean())
    )
    circ_means_h = (circ_means / (2*np.pi)) * 24.0
    
    results_list = []
    
    for (g, z), row in desc.iterrows():
        cm_rad = circ_means.loc[(g, z)]
        cm_rad = (cm_rad + np.pi) % (2*np.pi) - np.pi
        
        cm_h = circ_means_h.loc[(g, z)]
        cm_h = (cm_h + 12) % 24 - 12
        
        entry = {
            'Group': g,
            'Zone_Name': z,
            'N_Animals': int(row['N_Animals']),
            'Total_Cells': int(row['Total_Cells']),
            'Mean_Phase_Rad': float(cm_rad),
            'Mean_Phase_Hours': float(cm_h),
            'Mean_R': float(row['R_Resultant'])
        }
        results_list.append(entry)
        
    results = {
        'model_type_used': 'Permutation Test (Animal-Level Shuffle)',
        'params': {
            'nperm': nperm,
            'min_cells': min_cells,
            'min_animals_zone': min_animals_zone,
            'seed': seed,
            'max_attempts_factor': max_attempts_factor
        },
        'n_animals': int(len(unique_animals)),
        'n_zones': int(len(valid_zones)),
        'valid_zones': valid_zones,
        'dropped_zones': dropped_zones,
        'n_rows_input': n_input_rows,
        'n_rows_after_min_cells': n_rows_after_min_cells,
        'n_rows_used': int(len(df_params)),
        'permutation_p_interaction': float(p_val),
        'observed_T': float(T_obs),
        'permutations_accepted': accepted_count,
        'permutations_rejected': attempts - accepted_count,
        'descriptive_stats': results_list
    }
    return results

def run_interaction_test_from_csv(in_path: str, nperm=10000, min_cells=5, min_animals_zone=2, seed=None) -> dict:
    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {in_path} not found.")
    return run_interaction_test_from_df(df, nperm=nperm, min_cells=min_cells, min_animals_zone=min_animals_zone, seed=seed)

def write_results_bundle(results: dict, out_json_path: str) -> tuple[str, str]:
    with open(out_json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    csv_path = os.path.splitext(out_json_path)[0] + ".csv"
    pd.DataFrame(results['descriptive_stats']).to_csv(csv_path, index=False)
    return out_json_path, csv_path

def main():
    args = parse_args()
    
    print(f"Loading {args.in_file}...")
    try:
        results = run_interaction_test_from_csv(
            args.in_file,
            nperm=args.nperm,
            min_cells=args.min_cells,
            min_animals_zone=args.min_animals_zone,
            seed=args.seed
        )
        
        print(f"Permutations done. Rejected: {results['permutations_rejected']}")
        print(f"Permutation p-value: {results['permutation_p_interaction']:.4f}")
        
        json_p, csv_p = write_results_bundle(results, args.out_file)
        print(f"Descriptive CSV written to: {csv_p}")
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
