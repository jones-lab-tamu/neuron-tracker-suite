import time
import os
import skimage.io
import neuron_tracker_core as ntc

def benchmark():
    # 1. Setup
    # Update this path to your real file
    path = "C:/Folder/Subfolder/filename.tif"
    
    print(f"Loading movie from {path}...")
    full_data = skimage.io.imread(path)
    
    # Slice first 200 frames for a quick test (remove slicing for full test)
    data = full_data[:200] 
    
    # Standard params
    params = {
        'sigma1': 3.0, 'sigma2': 20.0, 'blur_sigma': 2.0, 'max_features': 200
    }

    print(f"\n--- Benchmarking on {len(data)} frames ---")

    # 2. Serial Run
    print("Running Serial Mode (n_processes=1)...")
    t0 = time.time()
    ntc.process_frames(data, n_processes=1, **params)
    t_serial = time.time() - t0
    print(f"Serial Time: {t_serial:.2f} seconds")

    # 3. Parallel Run
    print("Running Parallel Mode (n_processes=Default)...")
    t0 = time.time()
    # Default uses min(cpu_count, 8)
    ntc.process_frames(data, n_processes=None, **params)
    t_parallel = time.time() - t0
    print(f"Parallel Time: {t_parallel:.2f} seconds")

    # 4. Results
    speedup = t_serial / t_parallel
    print(f"\nSummary:")
    print(f"Speedup Factor: {speedup:.2f}x")
    if speedup > 1.5:
        print("SUCCESS: Significant performance gain detected.")
    else:
        print("WARNING: Speedup is negligible. Check CPU usage or shared memory overhead.")

if __name__ == "__main__":
    benchmark()