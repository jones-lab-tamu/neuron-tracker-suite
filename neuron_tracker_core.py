# -*- coding: utf-8 -*-
"""
Neuron Tracker Core Logic Library

This file contains all the core functions for the neuron tracking and data
extraction pipeline. It is designed to be a self-contained library, completely
decoupled from any user interface. This separation of logic allows it to be
imported and used by various front-ends, such as a command-line script (CLI)
or a graphical user interface (GUI).

The main workflow is orchestrated by calling these functions in sequence.
"""

import numpy
import scipy.ndimage
import scipy.spatial
import networkx

def detect_local_minima(arr):
    """
    Takes a 2D array and detects local minima (troughs).
    """
    neighborhood = scipy.ndimage.generate_binary_structure(len(arr.shape), 3)
    local_min = (scipy.ndimage.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = scipy.ndimage.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    return numpy.where(detected_minima)

def rescale(signal, minimum, maximum):
    """
    Rescales a numpy array to a new specified minimum and maximum range.
    """
    mins = numpy.min(signal)
    maxs = numpy.max(signal)
    if maxs == mins:
        return numpy.full(signal.shape, minimum)
    output = numpy.array(signal, dtype=float)
    output -= mins
    output *= (maximum - minimum) / (maxs - mins)
    output += minimum
    return output

def process_frames(data, sigma1, sigma2, blur_sigma, max_features, progress_callback=None):
    """
    Detects features (neurons) in each frame of the image sequence.
    """
    if progress_callback is None: progress_callback = lambda msg: None
    blob_lists, trees, ims, ids = [], [], [], {}
    T = data.shape[0]
    progress_callback("Stage 1/4: Detecting features in each frame...")
    for t in range(T):
        im = data[t]
        ims.append(im)
        fltrd = scipy.ndimage.gaussian_filter(-im, sigma1) - scipy.ndimage.gaussian_filter(-im, sigma2)
        fltrd = rescale(fltrd, 0.0, 1.0)
        locations = list(zip(*detect_local_minima(fltrd)))
        if not locations: locations.append((-500, -500))
        mags = []
        blurred_im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        for i, j in locations: mags.append(blurred_im[i, j])
        indices = numpy.argsort(mags)
        xys = [locations[i] for i in indices[-max_features:]]
        for x, y in xys: ids[(t, (x, y))] = len(ids)
        trees.append(scipy.spatial.KDTree(xys))
        blob_lists.append(xys)
        if t % (T // 20) == 0 or t == T - 1:
            progress_callback(f"  Processed frame {t+1}/{T} ({100.0 * (t+1) / float(T):.1f}%)")
    return ims, ids, trees, blob_lists

def build_trajectories(blob_lists, trees, ids, search_range, cone_radius_base, cone_radius_multiplier, progress_callback=None):
    """
    Connects features across frames to build trajectories.
    """
    if progress_callback is None: progress_callback = lambda msg: None
    T = len(blob_lists)
    graph = networkx.Graph()
    progress_callback("Stage 2/4: Building trajectories by connecting features...")
    for t in range(1, T):
        if t % (T // 20) == 0 or t == T - 1:
            progress_callback(f"  Connecting features up to frame {t+1}/{T} ({100.0 * (t+1) / float(T):.1f}%)")
        for blob in blob_lists[t]:
            for bt in range(t - 1, max(-1, t - search_range), -1):
                distance, neighbor_idx = trees[bt].query(blob, 1)
                cone_radius = cone_radius_base + cone_radius_multiplier * (t - bt)
                if distance < cone_radius:
                    neighbor_coords = tuple(trees[bt].data[neighbor_idx])
                    graph.add_edge(ids[(bt, neighbor_coords)], ids[(t, blob)])
                    break
    subgraphs = list(networkx.connected_components(graph))
    return graph, subgraphs

def prune_trajectories(graph, subgraphs, ids, progress_callback=None):
    """
    Prunes branched trajectories to ensure a single path.
    """
    if progress_callback is None: progress_callback = lambda msg: None
    progress_callback("Stage 3/4: Pruning branched trajectories...")
    reverse_ids = {v: k for k, v in ids.items()}
    pruned_subgraphs = []
    for i, nodes in enumerate(subgraphs):
        subgraph = graph.subgraph(nodes)
        nodes_by_time = {}
        for node in nodes:
            t, _ = reverse_ids[node]
            if t not in nodes_by_time: nodes_by_time[t] = []
            nodes_by_time[t].append(node)
        times = sorted(nodes_by_time.keys())
        if not times or len(times) < 2: continue
        start_node, end_node = nodes_by_time[times[0]][0], nodes_by_time[times[-1]][0]
        try:
            path = networkx.shortest_path(subgraph, start_node, end_node)
            if path: pruned_subgraphs.append(path)
        except networkx.NetworkXNoPath: continue
    progress_callback(f"  Found {len(pruned_subgraphs)} potential trajectories.")
    return pruned_subgraphs, reverse_ids

# --- THIS IS THE CORRECTED FUNCTION ---
def extract_and_interpolate_data(ims, pruned_subgraphs, reverse_ids, 
                                 min_trajectory_length, sampling_box_size, 
                                 sampling_sigma, max_interpolation_distance, 
                                 progress_callback=None):
    """
    Filters, interpolates, and samples data from the final, pruned trajectories.
    (Parameter names have been updated to match the GUI/CLI for consistency).
    """
    if progress_callback is None: progress_callback = lambda msg: None
    progress_callback("Stage 4/4: Filtering, interpolating, and sampling final trajectories...")

    T = len(ims)
    lines, coms, trajectories = [], [], []

    if sampling_box_size % 2 == 0:
        sampling_box_size += 1
    
    blur = numpy.zeros((sampling_box_size, sampling_box_size))
    blur[sampling_box_size // 2, sampling_box_size // 2] = 1
    blur = scipy.ndimage.gaussian_filter(blur, sampling_sigma)
    blur /= numpy.sum(blur)

    for i, path in enumerate(pruned_subgraphs):
        if i > 0 and len(pruned_subgraphs) > 10 and i % (len(pruned_subgraphs) // 10) == 0:
             progress_callback(f"  Sampling trajectory {i+1}/{len(pruned_subgraphs)}")

        if len(path) < min_trajectory_length * T:
            continue

        interpolated_pos, interpolated_val = {}, {}
        for node in path:
            t, c = reverse_ids[node]
            interpolated_pos[t] = c
        detected_times = sorted(interpolated_pos.keys())
        
        start_pos = interpolated_pos[detected_times[0]]
        for t in range(0, detected_times[0]): interpolated_pos[t] = start_pos
        end_pos = interpolated_pos[detected_times[-1]]
        for t in range(detected_times[-1] + 1, T): interpolated_pos[t] = end_pos

        for j in range(len(detected_times) - 1):
            lt, rt = detected_times[j], detected_times[j+1]
            if rt == lt + 1: continue
            lc, rc = numpy.array(interpolated_pos[lt]), numpy.array(interpolated_pos[rt])
            for t in range(lt + 1, rt):
                alpha = float(t - lt) / (rt - lt)
                interpolated_pos[t] = tuple(lc * (1.0 - alpha) + rc * alpha)

        full_trajectory, cancel = [], False
        for t in range(T):
            c = numpy.round(numpy.array(interpolated_pos[t])).astype(int)
            full_trajectory.append(c)
            h, w = ims[t].shape
            y, x = c
            b_half = sampling_box_size // 2
            if not (b_half <= y < h - b_half and b_half <= x < w - b_half):
                cancel = True; break
            patch = ims[t][y - b_half : y + b_half + 1, x - b_half : x + b_half + 1]
            interpolated_val[t] = numpy.sum(patch * blur)

        if cancel: continue

        full_trajectory = numpy.array(full_trajectory)
        distances = numpy.sqrt(numpy.sum(numpy.diff(full_trajectory, axis=0)**2, axis=1))
        if numpy.any(distances > max_interpolation_distance):
            continue

        coms.append(numpy.mean(full_trajectory, axis=0))
        trajectories.append(full_trajectory)
        lines.append(numpy.array(sorted(interpolated_val.items())))

    return numpy.array(coms), numpy.array(trajectories), numpy.array(lines)