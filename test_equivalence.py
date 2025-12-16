import unittest
import numpy as np
import scipy.ndimage
import scipy.spatial
import networkx
import random
import neuron_tracker_core as ntc

# =============================================================================
# LEGACY BASELINE IMPLEMENTATION (FROZEN)
# =============================================================================

def _legacy_detect_local_minima(arr):
    neighborhood = scipy.ndimage.generate_binary_structure(len(arr.shape), 3)
    local_min = (scipy.ndimage.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = scipy.ndimage.binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)

def _legacy_rescale(signal, minimum, maximum):
    mins = np.min(signal)
    maxs = np.max(signal)
    if maxs == mins:
        return np.full(signal.shape, minimum)
    output = np.array(signal, dtype=float)
    output -= mins
    output *= (maximum - minimum) / (maxs - mins)
    output += minimum
    return output

def legacy_extract_and_interpolate_data(
    ims, pruned_subgraphs, reverse_ids, min_trajectory_length, sampling_box_size, sampling_sigma, max_interpolation_distance
):
    """
    Original extraction logic, preserved for regression testing.
    """
    T = len(ims)
    lines, coms, trajectories = [], [], []

    if sampling_box_size % 2 == 0:
        sampling_box_size += 1

    blur = np.zeros((sampling_box_size, sampling_box_size))
    center = sampling_box_size // 2
    blur[center, center] = 1.0
    blur = scipy.ndimage.gaussian_filter(blur, sampling_sigma)
    blur /= np.sum(blur)

    for i, path in enumerate(pruned_subgraphs):
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
            lc = np.array(interpolated_pos[lt])
            rc = np.array(interpolated_pos[rt])
            for t in range(lt + 1, rt):
                alpha = float(t - lt) / (rt - lt)
                interpolated_pos[t] = tuple(lc * (1.0 - alpha) + rc * alpha)

        full_trajectory = []
        cancel = False
        b_half = sampling_box_size // 2

        for t in range(T):
            c = np.round(np.array(interpolated_pos[t])).astype(int)
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
            interpolated_val[t] = np.sum(patch * blur)

        if cancel:
            continue

        full_trajectory = np.array(full_trajectory)

        distances = np.sqrt(np.sum(np.diff(full_trajectory, axis=0) ** 2, axis=1))
        if np.any(distances > max_interpolation_distance):
            continue

        coms.append(np.mean(full_trajectory, axis=0))
        trajectories.append(full_trajectory)
        lines.append(np.array(sorted(interpolated_val.items())))

    return (
        np.array(coms),
        np.array(trajectories),
        np.array(lines),
    )

def legacy_reference_process_frames(data, sigma1, sigma2, blur_sigma, max_features):
    blob_lists, trees, ims, ids = [], [], [], {}
    T = data.shape[0]

    for t in range(T):
        im = data[t]
        ims.append(im)
        fltrd = (
            scipy.ndimage.gaussian_filter(-im, sigma1)
            - scipy.ndimage.gaussian_filter(-im, sigma2)
        )
        fltrd = _legacy_rescale(fltrd, 0.0, 1.0)
        locations = list(zip(*_legacy_detect_local_minima(fltrd)))

        if not locations:
            blob_lists.append([])
            trees.append(None)
            continue

        blurred_im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        mags = [blurred_im[i, j] for i, j in locations]

        indices = np.argsort(mags)
        indices = indices[-max_features:]
        xys = [locations[i] for i in indices]

        for x, y in xys:
            ids[(t, (x, y))] = len(ids)

        trees.append(scipy.spatial.KDTree(xys))
        blob_lists.append(xys)

    return ims, ids, trees, blob_lists


# =============================================================================
# EQUIVALENCE TEST SUITE
# =============================================================================

class TestNeuronTrackerEquivalence(unittest.TestCase):
    def setUp(self):
        # Synthetic Movie
        np.random.seed(12345) 
        self.frames = np.random.rand(10, 64, 64).astype(np.float32)
        for t in range(10):
            y, x = 20 + t, 20 + t
            self.frames[t, y-2:y+3, x-2:x+3] += 5.0 
        for t in range(10):
            self.frames[t, 40:45, 40:45] += 5.0

        self.params = {
            'sigma1': 1.0, 'sigma2': 5.0, 'blur_sigma': 1.0, 'max_features': 10
        }
        self.extract_params = {
            'min_trajectory_length': 0.1, # 1 frame out of 10
            'sampling_box_size': 5,
            'sampling_sigma': 1.0,
            'max_interpolation_distance': 10.0
        }

    def canonicalize_edges(self, graph):
        return sorted([tuple(sorted(e)) for e in graph.edges()])

    def test_tiebreaker_determinism(self):
        """
        Verify that prune_trajectories breaks ties deterministically even if
        graph construction order is randomized.
        """
        print("\nRunning Randomized Tie-Breaker Stress Test (100 iterations)...")
        nodes = [0, 1, 2, 3]
        base_edges = [(0,1), (0,2), (1,3), (2,3)] # Two equal paths: 0-1-3 and 0-2-3
        
        ids = {
            (0, (0,0)): 0, 
            (1, (10,10)): 1, 
            (1, (20,20)): 2, 
            (2, (30,30)): 3
        }
        subgraphs = [{0, 1, 2, 3}]
        
        reference_path = None
        
        for i in range(100):
            # Adversarial randomization: Shuffle edge insertion order
            edges = list(base_edges)
            random.shuffle(edges)
            
            G = networkx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            pruned, _ = ntc.prune_trajectories(G, subgraphs, ids, progress_callback=lambda m:None)
            
            self.assertEqual(len(pruned), 1, "Should pick exactly one path")
            path = pruned[0]
            
            if reference_path is None:
                reference_path = path
            else:
                self.assertEqual(path, reference_path, f"Tie-break failed at iter {i}: {path} vs {reference_path}")
                
        print(f"  [OK] Tie-breaking is deterministic (Path: {reference_path}).")

    def test_full_pipeline_equivalence_loop(self):
        print("\nRunning Full Pipeline Loop (50 iterations)...")
        
        # 1. Legacy Baseline (Computed Once)
        ims_L, ids_L, trees_L, blobs_L = legacy_reference_process_frames(self.frames, **self.params)
        
        # Note: We must use ntc.build_trajectories because we didn't freeze that one.
        # This is safe because build_trajectories logic wasn't refactored significantly.
        graph_L, subs_L = ntc.build_trajectories(blobs_L, trees_L, ids_L, 5, 5.0, 0.1)
        
        # We must use ntc.prune_trajectories, which NOW HAS SORTING.
        # This guarantees that the legacy run output is also deterministic.
        pruned_L, rev_ids_L = ntc.prune_trajectories(graph_L, subs_L, ids_L)
        
        com_L, traj_L, lines_L = legacy_extract_and_interpolate_data(
            ims_L, pruned_L, rev_ids_L, **self.extract_params
        )
        
        for i in range(50):
            # 2. New Core (Parallel)
            ims_N, ids_N, trees_N, blobs_N = ntc.process_frames(self.frames, n_processes=2, progress_callback=None, **self.params)
            graph_N, subs_N = ntc.build_trajectories(blobs_N, trees_N, ids_N, 5, 5.0, 0.1)
            pruned_N, rev_ids_N = ntc.prune_trajectories(graph_N, subs_N, ids_N, progress_callback=None)
            com_N, traj_N, lines_N = ntc.extract_and_interpolate_data(
                ims_N, pruned_N, rev_ids_N, **self.extract_params, mode='strict', progress_callback=None
            )

            # 3. Assertions
            self.assertEqual(len(blobs_L), len(blobs_N))
            np.testing.assert_array_equal(com_L, com_N)
            
            # Trajectories (list of arrays)
            for tl, tn in zip(traj_L, traj_N):
                np.testing.assert_array_equal(tl, tn)
                
            # Lines
            for ll, ln in zip(lines_L, lines_N):
                np.testing.assert_array_equal(ll, ln)
                
        print("  [OK] 50/50 runs matched legacy bit-exact.")
        print("SUCCESS.")

if __name__ == '__main__':
    unittest.main()