import unittest
import numpy as np
import scipy.ndimage
import scipy.spatial
import neuron_tracker_core as ntc

# =============================================================================
# LEGACY BASELINE IMPLEMENTATION
# =============================================================================
# These functions are copied VERBATIM from the original source code.
# They are inlined here to ensure the test compares against the HISTORICAL logic,
# protecting against accidental changes to the helper functions in the main module.

def _legacy_detect_local_minima(arr):
    """Legacy implementation of local minima detection."""
    neighborhood = scipy.ndimage.generate_binary_structure(len(arr.shape), 3)
    local_min = (scipy.ndimage.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = scipy.ndimage.binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)

def _legacy_rescale(signal, minimum, maximum):
    """Legacy implementation of array rescaling."""
    mins = np.min(signal)
    maxs = np.max(signal)
    if maxs == mins:
        return np.full(signal.shape, minimum)
    output = np.array(signal, dtype=float)
    output -= mins
    output *= (maximum - minimum) / (maxs - mins)
    output += minimum
    return output

def legacy_reference_process_frames(data, sigma1, sigma2, blur_sigma, max_features):
    """
    The original serial loop, strictly using the local legacy helpers.
    """
    blob_lists, trees, ims, ids = [], [], [], {}
    T = data.shape[0]

    for t in range(T):
        im = data[t]
        ims.append(im)

        # Difference of Gaussians on the negative image
        fltrd = (
            scipy.ndimage.gaussian_filter(-im, sigma1)
            - scipy.ndimage.gaussian_filter(-im, sigma2)
        )
        # Call LOCAL legacy version
        fltrd = _legacy_rescale(fltrd, 0.0, 1.0)

        # Call LOCAL legacy version
        locations = list(zip(*_legacy_detect_local_minima(fltrd)))

        if not locations:
            blob_lists.append([])
            trees.append(None)
            continue

        # Rank by intensity
        blurred_im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        mags = [blurred_im[i, j] for i, j in locations]

        indices = np.argsort(mags)
        indices = indices[-max_features:]
        xys = [locations[i] for i in indices]

        # Assign unique IDs
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
        # --- OPTION 1: REAL DATA ---
        import skimage.io
        print("Loading real movie...")
        # Load the full TIFF
        full_movie = skimage.io.imread("D:/SCN_GCaMP/Mus/120525-121025_Mus-Rhabdomys_series4.tif")
        
        # OPTIONAL: Slice it for speed (e.g., first 200 frames). 
        # Comparison on the full movie might take a long time.
        self.frames = full_movie[:200] 
        
        # --- OPTION 2: SYNTHETIC DATA (Commented Out) ---
        # np.random.seed(12345) 
        # self.frames = np.random.rand(10, 64, 64).astype(np.float32)
        # for t in range(10):
        #     y, x = 20 + t, 20 + t
        #     self.frames[t, y-2:y+3, x-2:x+3] += 5.0 
        # for t in range(10):
        #     self.frames[t, 40:45, 40:45] += 5.0

        self.params = {
            'sigma1': 1.0,
            'sigma2': 5.0,
            'blur_sigma': 1.0,
            # 'max_features': 10 # For artificial data
            'max_features': 200 # Increased for real data
        }

    def canonicalize_edges(self, graph):
        """Returns a sorted list of sorted edge tuples for strict comparison."""
        # Sort node pair (u, v) -> (min, max) then sort list of pairs
        return sorted([tuple(sorted(e)) for e in graph.edges()])

    def test_legacy_vs_parallel_equivalence(self):
        """
        Verifies that the NEW Parallel implementation produces bit-exact
        identical outputs to the FROZEN Legacy implementation.
        """
        print("\nRunning Equivalence Test: Frozen Legacy vs Parallel (New, n=2)...")
        
        # --- 1. Run Legacy (Ground Truth from local frozen functions) ---
        _, ids_legacy, trees_legacy, blobs_legacy = legacy_reference_process_frames(
            self.frames, **self.params
        )
        
        # Run build_trajectories using the legacy output
        # (Note: build_trajectories was not refactored significantly, so we use ntc version)
        graph_legacy, subgraphs_legacy = ntc.build_trajectories(
            blobs_legacy, trees_legacy, ids_legacy, 
            search_range=5, cone_radius_base=5.0, cone_radius_multiplier=0.1
        )
        
        # --- 2. Run New Parallel (Candidate from module) ---
        _, ids_parallel, trees_parallel, blobs_parallel = ntc.process_frames(
            self.frames, n_processes=2, **self.params
        )
        
        graph_parallel, subgraphs_parallel = ntc.build_trajectories(
            blobs_parallel, trees_parallel, ids_parallel, 
            search_range=5, cone_radius_base=5.0, cone_radius_multiplier=0.1
        )

        # --- 3. Assertions ---
        
        # A. Detection Equality (Strict Order)
        self.assertEqual(len(blobs_legacy), len(blobs_parallel), "Frame count mismatch")
        for t, (ref, par) in enumerate(zip(blobs_legacy, blobs_parallel)):
            # Strictly check the LIST. Order matters for IDs.
            ref_l = [tuple(r) for r in ref]
            par_l = [tuple(p) for p in par]
            self.assertEqual(ref_l, par_l, f"Detection mismatch at frame {t}")
            
        print("  [OK] Per-frame detections match Legacy exactly.")

        # B. ID Assignment Equality
        self.assertEqual(ids_legacy, ids_parallel, "ID dictionary mismatch vs Legacy")
        print("  [OK] Cell ID assignments match Legacy exactly.")

        # C. Strict Graph Topology (Edge Sets)
        edges_legacy = self.canonicalize_edges(graph_legacy)
        edges_parallel = self.canonicalize_edges(graph_parallel)
        
        self.assertEqual(edges_legacy, edges_parallel, "Graph Edge Set mismatch vs Legacy")
        print("  [OK] Trajectory graph structure matches Legacy exactly.")
        
        print("SUCCESS: New Parallel implementation preserves scientific integrity.")

if __name__ == '__main__':
    unittest.main()