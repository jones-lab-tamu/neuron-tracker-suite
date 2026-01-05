# Neuron Tracker and Analysis Suite

End-to-end analysis of cellular bioluminescence or fluorescence imaging movies for circadian ensemble rhythm mapping (built around SCN-like preparations). The suite combines (1) single-preparation tracking and rhythm extraction, (2) landmark-based atlas registration, (3) warp application to ROI coordinates, and (4) group-level visualization in a common atlas space.

## What this tool does (conceptual overview)

### Inputs
- A time-lapse imaging movie as a TIFF stack (`.tif`), typically a single channel, frames ordered over time.
- Optional: user-drawn anatomical include/exclude polygons (ROIs) and phase reference polygons.
- Optional: landmark sets for atlas registration across preparations.

### Core outputs
- Per-cell centroid coordinates (ROI coordinates).
- Per-cell intensity traces across time.
- Per-cell rhythm metrics and phase estimates (FFT SNR or Cosinor).
- Warp parameters for atlas registration.
- Warped ROI coordinate tables with stable IDs, suitable for group aggregation.

### Why the “stable ID” system exists
Many pipelines fail when a spatial filtering step changes the set or ordering of cells, but downstream tables still assume the original ordering. This suite explicitly writes and propagates an `Original_ROI_Index` so that:
- Rhythm-result tables are keyed to the cell identity in the original detection set.
- ROI coordinate files can be filtered, warped, and reloaded without silently losing identity.

If you keep the ID chain intact, group analysis becomes a deterministic join between coordinates and rhythm outputs, not an index-guessing exercise.

---

## Installation

### Requirements
Install dependencies using the repository `requirements.txt`:

```bash
pip install -r requirements.txt
```

The dependency set includes a PyQt5 GUI stack plus scientific Python libraries (NumPy, SciPy, Pandas, Matplotlib, scikit-image, and related GUI helpers used by the app).

### Launch the GUI
From the project root:

```bash
python neuron_analysis_app.py
```

---

## Quickstart (minimal reproducible workflow)

### Single preparation
1. Launch the GUI.
2. Go to **1. Single Animal Analysis**.
3. **Load Movie…** and select a `.tif`.
4. **Run Analysis** (detect, track, extract).
5. **Load Results** to populate the visualization tabs.
6. (Optional but recommended) **Define Anatomical ROI…** to generate:
   - `*_anatomical_roi.json`
   - `*_roi_filtered.csv`
   - `*_roi_filtered_ids.csv` (contains `Original_ROI_Index`)
7. Set **Phase Map Parameters** (at least `minutes_per_frame`), then **Update Plots**.
8. **Save Rhythm Results** to create `*_rhythm_results.csv`.

### Atlas + group
1. Choose one preparation to serve as the atlas (its `*_anatomical_roi.json` is the atlas ROI outline).
2. Go to **2. Atlas Registration**, load the atlas ROI file, add target ROI files, place landmarks, export warp parameters (`*_warp_parameters.json`).
3. Go to **3. Apply Warp**, load one or more warp parameter files, apply warp to generate:
   - `*_roi_warped.csv`
   - `*_roi_warped_with_ids.csv` (this is what group analysis expects)
4. Go to **4. Group Analysis**, add `*_roi_warped_with_ids.csv` files, assign them to **Control** or **Experiment**, confirm you have defined atlas regions, then generate group visualizations.

---

## File naming conventions (critical for automation)

The GUI and group-analysis logic infer related files by suffix substitution. Do not rename only one file in a set.

### Canonical per-preparation files
- `*_roi.csv`  
  Unfiltered ROI centroid coordinates (X,Y), output from analysis.
- `*_traces.csv`  
  Extracted intensity traces, first column is a time or frame-like axis, subsequent columns are per-cell traces.
- `*_trajectories.npy`  
  Per-cell trajectories over frames.
- `*_anatomical_roi.json`  
  User-drawn polygons (include, exclude, and phase reference objects).
- `*_roi_filtered.csv`  
  ROI centroids after applying include/exclude polygons.
- `*_roi_filtered_ids.csv`  
  One-column ID list keyed to `*_roi_filtered.csv`, header is `Original_ROI_Index`.
- `*_rhythm_results.csv`  
  Per-cell rhythm and phase table, keyed by `Original_ROI_Index`.
- `*_warp_parameters.json`  
  Landmark-derived transform parameters for atlas registration.
- `*_roi_warped.csv`  
  Warped ROI coordinates (no IDs).
- `*_roi_warped_with_ids.csv`  
  Warped ROI coordinates with IDs, header is `Original_ROI_Index,X_Warped,Y_Warped`.

### What group analysis will try to load for each preparation
Given an input `XYZ_roi_warped_with_ids.csv`, group analysis looks for:
- `XYZ_rhythm_results.csv` (same base name)
- An atlas regions file derived from the atlas ROI filename:
  - `atlas_anatomical_roi.json` → `atlas_anatomical_regions.json`

If those inferred files do not exist, the tool will skip or halt with an explicit error.

---

## Workflow 1: Single Animal Analysis (deep usage)

### Pipeline modes
The Single Animal panel exposes two analysis modes:
- **Metrics pipeline (recommended)**: produces candidate-set metrics suitable for post-hoc gating.
- **Legacy pipeline (no metrics)**: maintained for compatibility, fewer diagnostics.

### ROI tool (anatomical include/exclude, plus phase references)
**Define Anatomical ROI…** lets you draw polygon objects with multiple modes:
- **Include** and **Exclude** polygons: produce a filtered cell subset.
- **Phase Reference** polygon: used to define phase zero if you enable that option.
- **Phase Axis** object: stored for downstream reference (for example, along-axis visualizations).

When you confirm ROIs, the tool writes:
- `*_anatomical_roi.json` containing all ROI objects (include, exclude, and phase reference objects).
- `*_roi_filtered.csv` containing the filtered ROI coordinate subset.
- `*_roi_filtered_ids.csv` containing the `Original_ROI_Index` for each filtered row.

### Quality Gate (post-hoc filtering)
The GUI includes a Quality Gate block with presets:
- Recommended
- Lenient
- Strict
- Manual (with optional override controls)

The gate is designed to operate on a metrics-bearing result set. Conceptually:
- The analysis step generates candidates and metrics.
- The Quality Gate filters the candidate set after the fact.
- Rhythm analysis and “Save Rhythm Results” should reflect the gated set.

### Rhythm and phase estimation
In **Phase Map Parameters**, you select:
- **FFT (SNR)**: rhythm threshold is an SNR cutoff (higher is “more rhythmic”).
- **Cosinor (p-value)**: rhythm threshold is a p-value cutoff, and there is an additional R² threshold control.

Key controls you will use in practice:
- `minutes_per_frame` (required for phase in hours and for cosinor timepoints).
- `period_min`, `period_max` (advanced).
- `trend_window_hours` (advanced, used for detrending window logic).
- `Require >= 2 cycles` (strict cycle check).
- `Emphasize rhythmic cells` (visual emphasis, not a substitute for filtering).

### Saved rhythm results table (`*_rhythm_results.csv`)
When you click **Save Rhythm Results**, the tool writes a table containing:
- `Original_ROI_Index` (stable join key)
- `Phase_Hours` (phase expressed in hours, modulo the discovered or fallback period)
- `Period_Hours`
- `Is_Rhythmic` (boolean mask based on the selected method and thresholds)
- `Metric_Score` and `Filter_Score` (interpretation depends on method, for FFT they are SNR-like, for Cosinor they include R² and p-value)

This table is the authoritative input to group analysis. If you change gating criteria or ROI filtering, re-save the rhythm results so the group layer is consistent.

---

## Workflow 2: Atlas Registration

### What “atlas” means here
The atlas is represented by an `*_anatomical_roi.json` file that defines the template boundaries in a shared coordinate space. You register each target preparation to this atlas by placing corresponding landmarks.

### Landmark registration
1. Load the atlas ROI file.
2. Add one or more target ROI files.
3. Begin registration and place at least 3 corresponding landmarks per target (more is better if distortion is non-affine).
4. Export warp parameters for each target as `*_warp_parameters.json`.

The tool uses a thin-plate spline style transform (with normalization steps) to model non-linear warps typical of slice-to-slice deformation.

### Defining atlas regions (required for group region statistics)
Group analysis requires an atlas regions file:
- `*_anatomical_regions.json`

This file is expected to contain region definitions with at least:
- `zone_id`
- `name` (optional)
- `path_vertices` for each polygon

If group analysis reports “Define Regions first”, you have not created the atlas region file corresponding to your atlas ROI template.

---

## Workflow 3: Apply Warp (creating group-ready warped ROI files)

Apply Warp takes:
- A warp parameter file `*_warp_parameters.json`
- The target preparation’s `*_roi_filtered.csv` and `*_roi_filtered_ids.csv`

It produces:
- `*_roi_warped.csv` (warped coordinates only)
- `*_roi_warped_with_ids.csv` (warped coordinates plus `Original_ROI_Index`)

Group analysis expects the `*_roi_warped_with_ids.csv` form, because it must join warped coordinates to rhythm results using `Original_ROI_Index`.

Practical note: if you re-run ROI filtering, you must re-run Apply Warp so the warped coordinate file and its ID list remain aligned.

---

## Workflow 4: Group Analysis

### What you select
Use **Add Warped ROI File(s)…** to select one or more `*_roi_warped_with_ids.csv` files.

You then assign each file to a group label. Current group logic is explicit:
- Only items assigned to **Control** or **Experiment** are included in the downstream group analysis.

### What group analysis loads per file
For each warped ROI file:
- It infers the corresponding rhythm file by replacing the suffix:
  - `_roi_warped_with_ids.csv` → `_rhythm_results.csv`
- It loads atlas regions from:
  - `atlas_anatomical_roi.json` → `atlas_anatomical_regions.json`

### What group analysis produces
The specific visualizations depend on the tabs enabled in your build, but the design intent is:
- Group phase scatter and distribution views (by preparation and region)
- Group average maps in atlas space
- Region-level summary statistics, keyed to atlas-defined zone polygons

Interpretation note: phase is a circular quantity (modulo the period). If you export data and analyze outside the GUI, use circular statistics (for example, convert to radians, compute circular mean, then convert back) rather than linear averaging.

---

## CLI usage (batch processing)

A command-line entry point is provided for non-interactive processing:

```bash
python run_tracker_cli.py --tif /path/to/movie.tif --out /path/to/output_dir --name SessionName
```

The CLI writes the same core analysis products to the output directory:
- `SessionName_roi.csv`
- `SessionName_traces.csv`
- `SessionName_trajectories.npy`

Use the CLI when you want to run many movies headlessly, then inspect and curate results in the GUI.

---

## Common failure modes and how to avoid them

### 1) Group analysis skips an animal with “No rhythm results”
Cause: `*_rhythm_results.csv` does not exist for that base name.  
Fix: in Single Animal Analysis, regenerate phase maps with your chosen gating, then **Save Rhythm Results**.

### 2) Group analysis complains about missing atlas regions
Cause: `*_anatomical_regions.json` is missing for the atlas template.  
Fix: define atlas regions (zones) for the atlas ROI template so the expected regions file exists.

### 3) Row-count mismatch between IDs and coordinates during warp
Cause: `*_roi_filtered.csv` and `*_roi_filtered_ids.csv` are out of sync, typically due to partial reruns or manual file edits.  
Fix: re-run **Define Anatomical ROI…** and confirm both filtered files are regenerated together, then re-run **Apply Warp**.

### 4) “My group results look wrong after I changed the quality gate”
Cause: the saved `*_rhythm_results.csv` is now stale relative to your gating choices.  
Fix: after changing gate thresholds or ROI filtering, regenerate plots and re-save rhythm results before running group analysis.

### 5) Phase averages look nonsensical in external analysis
Cause: linear averaging of a circular variable (phase wraparound).  
Fix: use circular methods outside the GUI, phase is modulo-period by definition.

---

## Developer notes (for lab-level reproducibility)

- The repository includes `benchmark.py`, `benchmark_tuning.py`, and `test_equivalence.py` for performance and regression checking.
- FFT-based period discovery is currently performed in a way that can be redundant in the GUI (documented as known technical debt in code comments), this is a performance concern, not a scientific one.
- The analysis is designed to be auditable through explicit saved artifacts (ROI coordinates, traces, rhythm tables, warp parameters, and ID-bearing warped coordinates).

---

## Suggested lab practice (so results stay interpretable)

- Treat `Original_ROI_Index` as sacrosanct, do not edit it, do not reindex it, do not delete rows without propagating the same change to the paired tables.
- Any time you modify ROI filtering, quality gating, or phase thresholds, re-save `*_rhythm_results.csv` and re-run Apply Warp if the filtered ROI set changed.
- When sharing datasets across lab members, share the full file set for a preparation (ROI, traces, trajectories, anatomical ROI JSON, filtered ROI, filtered IDs, rhythm results, warp parameters, warped ROI with IDs).

If you break the ID mapping between ROI coordinates and rhythm tables, every group-level plot becomes untrustworthy.

## Acknowledgements

This project is a significant refactoring, modernization, and extension of an original analysis workflow developed in the Herzog Lab. The core scientific algorithms are based on the excellent foundational work of others, and we gratefully acknowledge their contributions:

*   **Ben Bales (Petzold Lab, UCSB):** For writing the original `extract_data.py` Python script, which forms the core of the neuron tracking algorithm.
*   **Matt Tso (Herzog Lab, WashU):** For developing the initial workflow, creating the supplementary analysis scripts, and for the presentation that inspired this project.