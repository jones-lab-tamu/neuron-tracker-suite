import numpy as np
from scipy.stats import circmean, f
from scipy.ndimage import label, binary_closing

def watson_williams_f(phases_a, phases_b):
    """
    Vectorized Watson-Williams F-test calculation for two circular samples.
    """
    f_stats = []
    
    for i in range(len(phases_a)):
        a = phases_a[i]
        b = phases_b[i]
        
        n1 = len(a)
        n2 = len(b)
        N = n1 + n2
        
        # Resultant Vector Lengths
        R1 = np.abs(np.sum(np.exp(1j * a)))
        R2 = np.abs(np.sum(np.exp(1j * b)))
        
        # Combined R
        combined = np.concatenate([a, b])
        R = np.abs(np.sum(np.exp(1j * combined)))
        
        numerator = (N - 2) * (R1 + R2 - R)
        denominator = (N - (R1 + R2))
        
        if denominator < 1e-9:
            f_val = 0.0 
        else:
            f_val = numerator / denominator
            
        f_stats.append(max(0, f_val))
        
    return np.array(f_stats)

def compute_difference_map_from_lists(phases_a_list, phases_b_list, grid_shape, valid_indices, period):
    """
    Helper to reconstruct the 2D Difference Map (in Hours).
    """
    diff_map = np.full(grid_shape, np.nan)
    
    for i, key in enumerate(valid_indices):
        a = phases_a_list[i]
        b = phases_b_list[i]
        
        # Convert to radians for circular mean calculation
        rad_a = (a / period) * 2 * np.pi
        rad_b = (b / period) * 2 * np.pi
        
        mean_a = circmean(rad_a, low=-np.pi, high=np.pi)
        mean_b = circmean(rad_b, low=-np.pi, high=np.pi)
        
        # Calculate Shortest Path difference
        diff_rad = (mean_b - mean_a + np.pi) % (2 * np.pi) - np.pi
        
        # Convert back to Hours
        diff_h = (diff_rad / (2 * np.pi)) * period
        diff_map[key] = diff_h
        
    return diff_map

def cluster_based_permutation_test(
    ref_data_map, exp_data_map, grid_shape, period,
    n_permutations=10000, min_n=2, cluster_alpha=0.3, bridge_gaps=True
):
    """
    Performs Cluster-Based Permutation Testing (CBPT) on 2D Grid Data.
    
    Args:
        ...
        bridge_gaps (bool): If True, applies morphological closing to connect
                            nearby clusters separated by small gaps.
    """
    ny, nx = grid_shape
    
    # 1. Flatten and Filter Data (Apply Min N Mask)
    valid_indices = []
    pooled_data = []
    
    for y in range(ny):
        for x in range(nx):
            key = (y, x)
            if key in ref_data_map and key in exp_data_map:
                ref_entry = ref_data_map[key]
                exp_entry = exp_data_map[key]
                
                n_ref = len(np.unique(ref_entry['animals']))
                n_exp = len(np.unique(exp_entry['animals']))
                
                if n_ref >= min_n and n_exp >= min_n:
                    valid_indices.append(key)
                    p_ref = ref_entry['phases']
                    p_exp = exp_entry['phases']
                    
                    combined = np.concatenate([p_ref, p_exp])
                    labels = np.concatenate([np.zeros(len(p_ref)), np.ones(len(p_exp))])
                    pooled_data.append((combined, labels))

    if not valid_indices:
        return None

    # Internal Helper: Calculate F-map 
    def get_f_map(current_pooled_data):
        group_a_list = []
        group_b_list = []
        for phases, labels in current_pooled_data:
            group_a_list.append(phases[labels == 0])
            group_b_list.append(phases[labels == 1])
        
        f_vals = watson_williams_f(group_a_list, group_b_list)
        
        f_map = np.zeros(grid_shape)
        for idx, val in zip(valid_indices, f_vals):
            f_map[idx] = val
        return f_map

    # Internal Helper: Find Clusters (with optional Bridging)
    def find_clusters(binary_map):
        if bridge_gaps:
            # Use a 3x3 solid structure to bridge 1-pixel gaps/diagonals
            structure = np.ones((3,3))
            # Close gaps (Dilate then Erode)
            processed_map = binary_closing(binary_map, structure=structure)
            # Label the bridged regions
            labels, num = label(processed_map, structure=structure)
        else:
            # Standard clustering
            labels, num = label(binary_map)
        return labels, num

    # 2. Calculate Real Statistics
    real_f_map = get_f_map(pooled_data)
    
    p_map = np.ones(grid_shape)
    pixel_f_thresholds = {} 
    
    for i, key in enumerate(valid_indices):
        phases, _ = pooled_data[i]
        N = len(phases)
        f_val = real_f_map[key]
        
        if N > 2:
            p_val = 1 - f.cdf(f_val, 1, N-2)
            p_map[key] = p_val
            pixel_f_thresholds[key] = f.ppf(1 - cluster_alpha, 1, N-2)
        else:
            pixel_f_thresholds[key] = 9999.9

    # 3. Form Real Clusters
    binary_map = p_map < cluster_alpha
    labeled_array, num_features = find_clusters(binary_map)
    
    real_cluster_masses = []
    for i in range(1, num_features + 1):
        # Mass = Sum of F-statistics within the cluster
        # Note: Even if 'bridge_gaps' added pixels to the cluster mask, 
        # real_f_map is 0.0 at those locations, so they contribute 0 mass.
        # This ensures we don't fabricate statistical weight.
        mass = np.sum(real_f_map[labeled_array == i])
        real_cluster_masses.append(mass)
        
    # 4. Permutation Loop
    max_cluster_masses = []
    
    for _ in range(n_permutations):
        perm_pooled = []
        for phases, labels in pooled_data:
            shuffled = np.random.permutation(labels)
            perm_pooled.append((phases, shuffled))
            
        perm_f_map = get_f_map(perm_pooled)
        
        perm_binary = np.zeros(grid_shape, dtype=bool)
        for key in valid_indices:
            if perm_f_map[key] > pixel_f_thresholds[key]:
                perm_binary[key] = True
                
        perm_labels, perm_num = find_clusters(perm_binary)
        
        if perm_num > 0:
            masses = [np.sum(perm_f_map[perm_labels == k]) for k in range(1, perm_num+1)]
            max_cluster_masses.append(max(masses))
        else:
            max_cluster_masses.append(0)
            
    # 5. Calculate Significance
    max_cluster_masses = np.array(max_cluster_masses)
    final_mask = np.zeros(grid_shape, dtype=bool)
    
    for i in range(1, num_features + 1):
        mass = real_cluster_masses[i-1]
        p_val = np.mean(max_cluster_masses >= mass)
        
        if p_val < 0.05:
            final_mask[labeled_array == i] = True
            
    # 6. Compute Difference Map
    list_a = [d[0][d[1] == 0] for d in pooled_data]
    list_b = [d[0][d[1] == 1] for d in pooled_data]
    
    diff_map = compute_difference_map_from_lists(list_a, list_b, grid_shape, valid_indices, period)
    
    return {
        'difference_map': diff_map,
        'significance_mask': final_mask,
        'p_values': p_map
    }