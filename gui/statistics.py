import numpy as np
from scipy.stats import f
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