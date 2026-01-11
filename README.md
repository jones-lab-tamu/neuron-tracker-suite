# Neuron Analysis Suite

A Python-based GUI application (PyQt5 + Matplotlib) for analyzing circadian rhythms in neuronal populations. The suite provides a pipeline for:

1.  Movie processing and feature tracking
2.  Anatomical ROI definition and atlas registration
3.  Single-cell rhythmicity analysis (FFT & Cosinor)
4.  Group-level statistical analysis
5.  Interactive visualizations (Phase maps, Actograms, Warped Heatmaps)

## Installation

Requirements:
- Python 3.8+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
python neuron_analysis_app.py
```

## Data Inputs and Naming Conventions

The GUI and group-analysis logic infer related files by suffix substitution. Do not rename only one file in a set.

### Canonical per-preparation files
- `*_roi.csv`: Unfiltered ROI centroid coordinates (X,Y), output from analysis.
- `*_traces.csv`: Extracted intensity traces, first column is a time or frame-like axis, subsequent columns are per-cell traces.
- `*_trajectories.npy`: Per-cell trajectories over frames.
- `*_anatomical_roi.json`: User-drawn polygons (include, exclude, and phase reference objects).
- `*_roi_filtered.csv`: ROI centroids after applying include/exclude polygons.
- `*_roi_filtered_ids.csv`: One-column ID list keyed to `*_roi_filtered.csv`, header is `Original_ROI_Index`.
- `*_rhythm_results.csv`: Per-cell rhythm and phase table, keyed by `Original_ROI_Index`.
- `*_warp_parameters.json`: Landmark-derived transform parameters for atlas registration.
- `*_roi_warped.csv`: Warped ROI coordinates (no IDs).
- `*_roi_warped_with_ids.csv`: Warped ROI coordinates with IDs, header is `Original_ROI_Index,X_Warped,Y_Warped`.

### What group analysis will try to load for each preparation
Given an input `XYZ_roi_warped_with_ids.csv`, group analysis looks for:
- `XYZ_rhythm_results.csv` (same base name)
- `XYZ_traces.csv` (required for Warped Heatmap)
- An atlas regions file derived from the atlas ROI filename:
  - `atlas_anatomical_roi.json` -> `atlas_anatomical_regions.json`

---

## Workflow 1: Single Animal Analysis

This is the entry point for raw data. Load a movie (TIFF/AVI) and its corresponding `*_roi.csv` and `*_traces.csv` (automatically discovered if naming matches).

The Single Animal panel exposes two analysis modes:
1.  **Coordinate Map**: Inspection of raw ROI locations.
2.  **Phase Map**: Computation of circadian phase and rhythmicity.

### ROI Tools
**Define Anatomical ROI...** lets you draw polygon objects with multiple modes:
- **Include** and **Exclude** polygons: produce a filtered cell subset.
- **Phase Reference** polygon: used to define phase zero if you enable that option.
- **Phase Axis** object: stored for downstream reference (used by Warped Heatmap to determine Dorsal-Ventral positioning).

When you confirm ROIs, the tool writes:
- `*_anatomical_roi.json` containing all ROI objects.
- `*_roi_filtered.csv` containing the filtered coordinate subset.
- `*_roi_filtered_ids.csv` containing the `Original_ROI_Index` for each filtered row.

### Quality Gate (post-hoc filtering)
The GUI includes a Quality Gate block with presets (Recommended, Lenient, Strict, Manual). The gate operates on a metrics-bearing result set. Conceptually:
- The analysis step generates candidates and metrics.
- The Quality Gate filters the candidate set after the fact.
- Rhythm analysis and "Save Rhythm Results" reflect the gated set.

### Rhythm and phase estimation
In **Phase Map Parameters**, you select **FFT (SNR)** or **Cosinor (p-value)**. Key controls:
- `minutes_per_frame` (required).
- `period_min`, `period_max`, `trend_window_hours`.
- `Require >= 2 cycles`, `Emphasize rhythmic cells`.

### Saved rhythm results table (`*_rhythm_results.csv`)
When you click **Save Rhythm Results**, the tool writes a table containing:
- `Original_ROI_Index` (stable identifier linking back to traces)
- `Phase_Hours` (phase expressed in hours, modulo the discovered or fallback period)
- `Period_Hours`
- `Is_Rhythmic` (boolean mask based on the selected method and thresholds)
- `Metric_Score` and `Filter_Score` (SNR, R^2, or p-value depending on method)

This table is the authoritative input to group analysis. If you change gating criteria or ROI filtering, **re-save the rhythm results** so the group layer is consistent.

---

## Workflow 2: Atlas Registration

The atlas is represented by an `*_anatomical_roi.json` file that defines the template boundaries in a shared coordinate space. You register each target preparation to this atlas by placing corresponding landmarks.

### Landmark registration
1. Go to **2. Atlas Registration**.
2. Load the atlas ROI file.
3. Add one or more target ROI files (`*_roi.csv` or similar).
4. Begin registration and place at least 3 corresponding landmarks per target.
5. Export warp parameters for each target as `*_warp_parameters.json`.

---

## Workflow 3: Apply Warp

### Creating group-ready warped ROI files
Go to **3. Apply Warp**. This panel takes:
- A warp parameter file `*_warp_parameters.json`
- The target preparation's `*_roi_filtered.csv` and `*_roi_filtered_ids.csv` (inferred)

It produces:
- `*_roi_warped.csv` (warped coordinates only)
- `*_roi_warped_with_ids.csv` (warped coordinates plus `Original_ROI_Index`)

**Important**: Group analysis expects the `*_roi_warped_with_ids.csv` form to join coordinates with rhythm results. If you re-run ROI filtering, you **must** re-run Apply Warp so the file outputs remain aligned.

---

## Workflow 4: Group Analysis

### What you select
Use **Add Files** to select one or more `*_roi_warped_with_ids.csv` files.

Assign each file to a group label (**Control** or **Experiment**). Only assigned items are included in analysis.

### Standard Visualizations
- Group phase scatter and distribution views.
- Group average maps (requires defined atlas regions).
- Phase Gradient analysis.
- Circular Statistics (Watson-Williams F-test).

### Warped Heatmap (axis-projected atlas view)
A dedicated group visualization located in the **Warped Heatmap** box.
- **Modes**:
  - `Individual`: View one animal at a time.
  - `Control pooled`: Aggregate all Control animals.
  - `Experiment pooled`: Aggregate all Experiment animals.
- **Requirements**:
  - **Atlas**: Must utilize an atlas ROI file containing at least one **"Phase Axis"** object.
  - **Data**: Requires `*_roi_warped_with_ids.csv`, `*_rhythm_results.csv`, AND `*_traces.csv` for every animal.
- **Logic**:
  - Projects warped ROI coordinates onto the atlas Phase Axis.
  - If two axes are present (Bilateral), points are assigned to the closest axis.
  - Automatically flips the axis so Dorsal (top) is 0 and Ventral (bottom) is 1.
  - Filters to include only rhythmic cells (`Is_Rhythmic == True`).
  - Stacks traces as rows sorted by their dorsal-to-ventral position.

---

## CLI usage (batch processing)

A command-line entry point is provided for non-interactive analysis:

```bash
python run_tracker_cli.py /path/to/movie.tif --visualize
```

It writes core analysis products (`*_roi.csv`, `*_traces.csv`, `*_trajectories.npy`) to the same directory. Note that the CLI does not currently perform the full atlas/warping workflow; it is intended for the initial feature extraction step.

---

## Common failure modes and how to avoid them

### 1) Group analysis skips an animal with "No rhythm results"
**Cause**: `*_rhythm_results.csv` does not exist for that base name.
**Fix**: In Single Animal Analysis, regenerate phase maps with your chosen gating, then click **Save Rhythm Results**.

### 2) Group analysis complains about missing atlas regions
**Cause**: `*_anatomical_regions.json` is missing for the atlas template.
**Fix**: Define atlas regions (zones) in the Group View panel so the expected regions file exists.

### 3) Row-count mismatch between IDs and coordinates during warp
**Cause**: `*_roi_filtered.csv` and `*_roi_filtered_ids.csv` are out of sync, typically due to partial reruns or manual file edits.
**Fix**: Re-run **Define Anatomical ROI...** and confirm both filtered files are regenerated together, then re-run **Apply Warp**.

### 4) "My group results look wrong after I changed the quality gate"
**Cause**: The saved `*_rhythm_results.csv` is stale relative to your new gating choices.
**Fix**: After changing gate thresholds or ROI filtering, **Save Rhythm Results** again before running group analysis.

### 5) Phase averages look nonsensical in external analysis
**Cause**: Linear averaging of a circular variable (phase wraparound).
**Fix**: Use circular statistics (convert to radians, compute circular mean, convert back) rather than linear averaging.

---

## Developer notes (for lab-level reproducibility)

- The repository includes `benchmark.py`, `benchmark_tuning.py`, and `test_equivalence.py` for performance and regression checking.
- FFT-based period discovery is currently performed in a way that can be redundant in the GUI (documented as known technical debt), but scientific correctness is maintained.
- The analysis is designed to be auditable through explicit saved artifacts (ROI coordinates, traces, rhythm tables, warp parameters, and ID-bearing warped coordinates).

---

## Suggested lab practice (so results stay interpretable)

- Treat `Original_ROI_Index` as sacrosanct: do not edit it, do not reindex it, do not delete rows without propagating the same change to the paired tables.
- Any time you modify ROI filtering, quality gating, or phase thresholds, re-save `*_rhythm_results.csv` and re-run Apply Warp if the filtered ROI set changed.
- When sharing datasets across lab members, share the full file set for a preparation (ROIs, traces, trajectories, anatomical ROI JSON, filtered ROI, filtered IDs, rhythm results, warp parameters, warped ROI with IDs).

If you break the ID mapping between ROI coordinates and rhythm tables, every group-level plot becomes untrustworthy.
## Acknowledgements

This project is a significant refactoring, modernization, and extension of an original analysis workflow developed in the Herzog Lab. The core scientific algorithms are based on the excellent foundational work of others, and we gratefully acknowledge their contributions:

*   **Ben Bales (Petzold Lab, UCSB):** For writing the original `extract_data.py` Python script, which forms the core of the neuron tracking algorithm.
*   **Matt Tso (Herzog Lab, WashU):** For developing the initial workflow, creating the supplementary analysis scripts, and for the presentation that inspired this project.
