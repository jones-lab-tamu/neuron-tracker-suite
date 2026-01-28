# benchmark_tuning.py
import time
import os
import skimage.io
import neuron_tracker_core as ntc

def benchmark_tuning():
    # --- CONFIG ---
    # Update this path to your real file
    path = "C:/Folder/Subfolder/filename.tif"
    frames_to_test = 50  # 50 frames is enough to gauge speed
    
    # Params matching your typical workload
    params = {
        'sigma1': 3.0, 
        'sigma2': 20.0, 
        'blur_sigma': 2.0, 
        'max_features': 200
    }
    # --------------

    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"Loading first {frames_to_test} frames from {os.path.basename(path)}...")
    full_data = skimage.io.imread(path)
    data = full_data[:frames_to_test]
    
    cpu_count = os.cpu_count() or 4
    # Test candidates: 1 (Serial), 4, 8, 12, 16, 24... up to actual CPU count
    candidates = [1, 4, 8, 12, 16, 24, 32]
    candidates = [n for n in candidates if n <= cpu_count]
    
    # Ensure at least one parallel run if possible
    if cpu_count > 1 and cpu_count not in candidates:
        candidates.append(cpu_count)
        candidates.sort()

    print(f"\n--- Tuning Benchmark (Frames: {frames_to_test}, Image Shape: {data.shape[1:]}) ---")
    print(f"{'Cores':<10} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 35)

    baseline_time = None
    
    for n in candidates:
        start_time = time.time()
        
        # Run detection
        ntc.process_frames(
            data, 
            n_processes=n, 
            progress_callback=None, # Silence output for clean benchmarking
            **params
        )
        
        elapsed = time.time() - start_time
        
        if n == 1:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed if baseline_time else 0.0
            
        print(f"{n:<10} | {elapsed:<10.2f} | {speedup:<10.2f}x")

    print("\nDone. Pick the N where speedup plateaus.")

if __name__ == "__main__":
    benchmark_tuning()