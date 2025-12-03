import numpy as np
from scipy.stats import f
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_closing, label

def circmean_hours(phases, period):
    """
    Circular mean of phases given in hours.
    Returns mean in hours on the same scale.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        return np.nan
    
    # Assuming phase is [-period/2, period/2], convert to radians [-pi, pi]
    angles = phases * (np.pi / (period / 2.0))
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    
    # Convert back to hours
    return mean_angle * (period / (2.0 * np.pi))


def build_animal_phase_matrix(data_map, grid_shape, period):
    """
    Build an array of per-animal circular mean phase per bin.
    """
    ny, nx = grid_shape
    animal_set = set()
    for key, entry in data_map.items():
        animals = np.asarray(entry["animals"])
        for a in np.unique(animals):
            animal_set.add(a)

    animal_ids = sorted(list(animal_set))
    n_animals = len(animal_ids)
    id_to_index = {a: i for i, a in enumerate(animal_ids)}

    phase_matrix = np.full((n_animals, ny, nx), np.nan, dtype=float)

    for (y, x), entry in data_map.items():
        animals = np.asarray(entry["animals"])
        phases  = np.asarray(entry["phases"], dtype=float)

        for a in np.unique(animals):
            mask = (animals == a)
            mean_phase = circmean_hours(phases[mask], period)
            ai = id_to_index[a]
            phase_matrix[ai, y, x] = mean_phase

    return animal_ids, phase_matrix

def watson_williams_f(group_a, group_b):
    """
    Watson-Williams F-test for two circular samples.
    Note: Input is now a single array per group (per-animal means).
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    
    n1 = len(a)
    n2 = len(b)
    N = n1 + n2
    
    if n1 == 0 or n2 == 0:
        return 0.0

    # Convert hours to radians
    period = 24.0 # Assumes CT
    rad_a = a * (np.pi / (period / 2.0))
    rad_b = b * (np.pi / (period / 2.0))
    
    R1 = np.abs(np.sum(np.exp(1j * rad_a)))
    R2 = np.abs(np.sum(np.exp(1j * rad_b)))
    
    combined = np.concatenate([rad_a, rad_b])
    R = np.abs(np.sum(np.exp(1j * combined)))
    
    numerator = (N - 2) * (R1 + R2 - R)
    denominator = (N - (R1 + R2))
    
    if denominator < 1e-9:
        return 0.0
    
    f_val = numerator / denominator
    return max(0, f_val)

def cluster_based_permutation_test_by_animal(
    ref_matrix, exp_matrix, grid_shape, period,
    n_permutations=1000, min_n=3, cluster_alpha=0.05, bridge_gaps=True
):
    """
    Cluster-Based Permutation Test using animals as the unit of analysis.
    """
    ny, nx = grid_shape
    n_ref = ref_matrix.shape[0]
    n_exp = exp_matrix.shape[0]
    n_total = n_ref + n_exp

    all_data = np.concatenate([ref_matrix, exp_matrix], axis=0)
    group_labels = np.concatenate([np.zeros(n_ref, dtype=int), np.ones(n_exp, dtype=int)])

    stat_indices = []
    for y in range(ny):
        for x in range(nx):
            values = all_data[:, y, x]
            valid = ~np.isnan(values)
            g = group_labels[valid]
            n_ref_bin = np.sum(g == 0)
            n_exp_bin = np.sum(g == 1)
            if n_ref_bin >= min_n and n_exp_bin >= min_n:
                stat_indices.append((y, x))

    if not stat_indices:
        return None

    def compute_f_map(current_group_labels):
        f_map = np.zeros(grid_shape, dtype=float)
        for key in stat_indices:
            y, x = key
            vals = all_data[:, y, x]
            valid_mask = ~np.isnan(vals)
            vals = vals[valid_mask]
            gl = current_group_labels[valid_mask]
            group_a = vals[gl == 0]
            group_b = vals[gl == 1]
            f_map[y, x] = watson_williams_f(group_a, group_b)
        return f_map

    def find_clusters(binary_map):
        if bridge_gaps:
            structure = np.ones((3, 3), dtype=bool)
            processed = binary_closing(binary_map, structure=structure)
            labels_arr, num = label(processed, structure=structure)
        else:
            labels_arr, num = label(binary_map)
        return labels_arr, num

    real_f_map = compute_f_map(group_labels)
    p_map = np.ones(grid_shape, dtype=float)

    for key in stat_indices:
        y, x = key
        vals = all_data[:, y, x]
        N_bin = np.sum(~np.isnan(vals))
        if N_bin > 2:
            f_val = real_f_map[y, x]
            p_map[y, x] = 1.0 - f.cdf(f_val, 1, N_bin - 2)

    binary_map = p_map < cluster_alpha
    labeled_array, num_features = find_clusters(binary_map)
    real_cluster_masses = [np.sum(real_f_map[labeled_array == c]) for c in range(1, num_features + 1)]

    max_cluster_masses = []
    f_crit = f.ppf(1.0 - cluster_alpha, 1, n_total - 2) # Approximation

    for _ in range(n_permutations):
        perm_labels = np.random.permutation(group_labels)
        perm_f_map = compute_f_map(perm_labels)
        perm_binary = perm_f_map > f_crit
        perm_labels_arr, perm_num = find_clusters(perm_binary)
        if perm_num > 0:
            masses = [np.sum(perm_f_map[perm_labels_arr == k]) for k in range(1, perm_num + 1)]
            max_cluster_masses.append(max(masses))
        else:
            max_cluster_masses.append(0.0)

    max_cluster_masses = np.asarray(max_cluster_masses, dtype=float)
    final_mask = np.zeros(grid_shape, dtype=bool)

    for i, mass in enumerate(real_cluster_masses, start=1):
        p_cluster = np.mean(max_cluster_masses >= mass)
        # if p_cluster < 0.05:
        if p_cluster < 0.1:
            final_mask[labeled_array == i] = True

    diff_map = np.full(grid_shape, np.nan, dtype=float)
    for y in range(ny):
        for x in range(nx):
            ref_vals = ref_matrix[:, y, x]
            exp_vals = exp_matrix[:, y, x]
            ref_vals = ref_vals[~np.isnan(ref_vals)]
            exp_vals = exp_vals[~np.isnan(exp_vals)]
            if ref_vals.size > 0 and exp_vals.size > 0:
                mean_ref = circmean_hours(ref_vals, period)
                mean_exp = circmean_hours(exp_vals, period)
                diff = mean_exp - mean_ref
                diff_map[y, x] = (diff + period / 2) % period - period / 2

    return {
        "difference_map": diff_map,
        "significance_mask": final_mask,
        "p_values": p_map,
    }
    
def run_graph_cbpt(phase_matrix, group_labels, scaffold,
                   min_animals_per_group=3, cluster_alpha=0.2, n_permutations=5000):
    """
    Performs a Graph-Based Cluster Permutation Test on per-animal, per-node data.

    Args:
        phase_matrix (np.ndarray): Shape (n_animals, n_nodes). Contains mean phases or NaNs.
        group_labels (np.ndarray): Shape (n_animals,). Labels (0 or 1) for each animal.
        scaffold (AnatomicalNodeScaffold): The scaffold object containing node and edge info.
        min_animals_per_group (int): Minimum N per group to test a node.
        cluster_alpha (float): Initial p-value threshold for cluster formation.
        n_permutations (int): Number of permutations for the null distribution.

    Returns:
        dict: A dictionary containing the results of the test.
    """
    n_animals, n_nodes = phase_matrix.shape
    unique_labels = np.unique(group_labels)
    if len(unique_labels) != 2:
        raise ValueError("group_labels must contain exactly two unique groups.")

    # --- Helper function to compute node-wise stats ---
    def compute_node_stats(current_phase_matrix, current_group_labels):
        f_vals = np.zeros(n_nodes)
        p_vals = np.ones(n_nodes)
        
        for i in range(n_nodes):
            node_phases = current_phase_matrix[:, i]
            valid_mask = ~np.isnan(node_phases)
            
            if not np.any(valid_mask):
                continue

            valid_phases = node_phases[valid_mask]
            valid_labels = current_group_labels[valid_mask]
            
            group_a = valid_phases[valid_labels == 0]
            group_b = valid_phases[valid_labels == 1]
            
            if len(group_a) < min_animals_per_group or len(group_b) < min_animals_per_group:
                continue
            
            f_stat = watson_williams_f(group_a, group_b)
            f_vals[i] = f_stat
            
            # Degrees of freedom for the F-test
            df1 = 1
            df2 = len(group_a) + len(group_b) - 2
            if df2 > 0:
                p_vals[i] = 1.0 - f.cdf(f_stat, df1, df2)
        
        return f_vals, p_vals

    # --- Helper function to find cluster masses ---
    def find_cluster_masses(stat_map, p_map):
        # 1. Identify suprathreshold nodes
        suprathreshold_nodes = np.where(p_map < cluster_alpha)[0]
        
        if len(suprathreshold_nodes) == 0:
            return [0.0]

        # 2. Build the subgraph of only significant nodes
        # Cosecant-Sagittal-Graph (CSGraph) operates on sparse matrices
        adjacency = scaffold.adjacency_matrix
        
        # Select the part of the adjacency matrix corresponding to significant nodes
        subgraph_adj = adjacency[suprathreshold_nodes, :][:, suprathreshold_nodes]
        
        # 3. Find connected components on this subgraph
        n_clusters, labels = connected_components(
            csgraph=subgraph_adj, directed=False, return_labels=True
        )
        
        if n_clusters == 0:
            return [0.0]

        # 4. Compute cluster masses
        cluster_masses = []
        for i in range(n_clusters):
            # Find which original nodes belong to this cluster
            cluster_node_indices = suprathreshold_nodes[labels == i]
            mass = np.sum(stat_map[cluster_node_indices])
            cluster_masses.append(mass)

        return cluster_masses

    # --- 1. Compute real statistics on original data ---
    print("Computing stats on original data...")
    real_f_map, real_p_map = compute_node_stats(phase_matrix, group_labels)
    real_cluster_masses = find_cluster_masses(real_f_map, real_p_map)
    
    # --- 2. Build null distribution via permutation ---
    print(f"Running {n_permutations} permutations...")
    max_cluster_mass_null = []
    for i in range(n_permutations):
        if (i + 1) % (n_permutations // 10 or 1) == 0:
            print(f"  Permutation {i+1}/{n_permutations}...")
        
        permuted_labels = np.random.permutation(group_labels)
        perm_f_map, perm_p_map = compute_node_stats(phase_matrix, permuted_labels)
        perm_cluster_masses = find_cluster_masses(perm_f_map, perm_p_map)
        max_cluster_mass_null.append(np.max(perm_cluster_masses))

    max_cluster_mass_null = np.array(max_cluster_mass_null)

    # --- 3. Determine significance of real clusters ---
    print("Finalizing results...")
    significant_cluster_indices = []
    cluster_p_values = []
    
    for mass in real_cluster_masses:
        if mass == 0:
            p_val = 1.0
        else:
            # Calculate p-value as the proportion of the null distribution >= real mass
            p_val = np.mean(max_cluster_mass_null >= mass)
        cluster_p_values.append(p_val)
        
        if p_val < 0.05: # Standard significance level
            significant_cluster_indices.append(len(significant_cluster_indices))
    
    # --- 4. Create final output mask and difference map ---
    # Re-run cluster finding on real data to get labels
    suprathreshold = np.where(real_p_map < cluster_alpha)[0]
    subgraph_adj = scaffold.adjacency_matrix[suprathreshold, :][:, suprathreshold]
    _, labels = connected_components(csgraph=subgraph_adj, directed=False, return_labels=True)
    
    final_sig_mask = np.zeros(n_nodes, dtype=bool)
    for i, p_val in enumerate(cluster_p_values):
        if p_val < 0.05:
            cluster_node_indices = suprathreshold[labels == i]
            final_sig_mask[cluster_node_indices] = True
    
    # Compute circular difference map for visualization
    diff_map = np.full(n_nodes, np.nan)
    for i in range(n_nodes):
        node_phases = phase_matrix[:, i]
        valid_mask = ~np.isnan(node_phases)
        if not np.any(valid_mask): continue
        
        group_a = node_phases[valid_mask & (group_labels == 0)]
        group_b = node_phases[valid_mask & (group_labels == 1)]
        
        if len(group_a) > 0 and len(group_b) > 0:
            mean_a = circmean_hours(group_a, 24.0)
            mean_b = circmean_hours(group_b, 24.0)
            # Circular difference wrapped to [-12, 12]
            diff = mean_b - mean_a
            diff_map[i] = (diff + 12) % 24 - 12

    return {
        "node_f_values": real_f_map,
        "node_p_values": real_p_map,
        "cluster_p_values": np.array(cluster_p_values),
        "significance_mask": final_sig_mask,
        "difference_map": diff_map,
        "real_cluster_masses": np.array(real_cluster_masses)
    }