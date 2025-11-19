# Neuron Tracker and Analysis Suite

## Project Overview

This suite of tools provides a complete, end-to-end pipeline for the analysis of cellular bioluminescence or fluorescence imaging data, specifically tailored for studying circadian rhythms in neuronal ensembles like the Suprachiasmatic Nucleus (SCN).

The application supports a comprehensive workflow within a single, user-friendly graphical interface:
1.  **Single Animal Analysis:** Process a raw movie (`.tif`) to detect and track individual cells, extract their intensity traces, and generate interactive visualizations (heatmaps, phase maps, etc.).
2.  **Atlas Registration:** A landmark-based warping tool to register multiple individual animal datasets into a common, standardized atlas space.
3.  **Apply Warp:** Transform ROI coordinates from individual datasets into common atlas space using calculated warp parameters.
4.  **Group Analysis:** Pool data from multiple warped datasets to generate group-level visualizations, such as a group phase distribution scatter plot and a group average phase map.

The application features a powerful, interactive visualization environment, allowing users to click on any cell in a spatial plot to instantly view its detailed trajectory and intensity trace over time.

---

## Installation

This project requires Python 3. The necessary external libraries can be installed in a single step using the provided `requirements.txt` file.

1.  Open a terminal or command prompt.
2.  Navigate to the directory containing the project files.
3.  Run the following command:
    ```    
	pip install -r requirements.txt
    ```
---

## Core Workflows

The application is organized around four primary scientific workflows, accessible via the numbered navigation list on the left side of the main window.

### Workflow 1: Single Animal Analysis

This is the foundational workflow for processing a single movie and exploring its results.

1.  **Launch the Application:**
    ```
    python neuron_analysis_app.py
    ```
2.  **Load a Movie:** In the "1. Single Animal Analysis" panel, click **"Load Movie..."** and select your raw `.tif` file.
3.  **Run Analysis:** Click **"Run Analysis"** in the Global Actions section. This will detect, track, and extract data for all cells, which may take several minutes.
4.  **Load Results:** Once the analysis is complete, click **"Load Results"** in the Global Actions section. This will populate all the interactive visualization tabs on the right.

### Workflow 2: Atlas Registration

This workflow is for aligning multiple datasets to a common atlas. This requires that you have already run the single-animal analysis on each animal and used the **"Define Anatomical ROI..."** tool to create an `_anatomical_roi.json` file for each.

1.  Select **"2. Atlas Registration"** from the navigation list.
2.  Select your master atlas file (which is itself an `_anatomical_roi.json` file).
3.  Add one or more target `_anatomical_roi.json` files from your individual animals.
4.  Click **"Begin Registration..."** to launch the landmark-placement tool. For each target, place 3+ corresponding landmarks on the atlas and the target to calculate the warp.

### Workflow 3: Apply Warp

After registering datasets to an atlas, use this workflow to apply the calculated warps to your analysis results.

1.  Select **"3. Apply Warp"** from the navigation list.
2.  Load the warp parameter files (`.json`) generated during registration.
3.  Click **"Apply Warp to ROI Data"** to transform your ROI coordinates into atlas space.

### Workflow 4: Group Analysis

This workflow is for visualizing data from multiple animals that have already been warped to the atlas space.

1.  Select **"4. Group Analysis"** from the navigation list.
2.  Click **"Add Warped ROI File(s)..."** and select all the `_roi_warped.csv` files you want to include in the group.
3.  Click **"Generate Group Visualizations"**. This will populate the "Grp Scatter" and "Grp Avg Map" tabs.

---

## Guide to Interactive Visualizations

The power of this tool lies in its interconnected visualization tabs.

### Visualization Tabs

The application includes seven visualization tabs on the right side:
- **Heatmap:** Displays all cell intensity traces as rows, with sortable ordering
- **CoM (Center of Mass):** Spatial plot of all detected cells with adjustable contrast
- **Trajectories:** Frame-by-frame visualization of individual cell trajectories
- **Phase Map:** Spatial visualization of rhythmic cells colored by their peak timing
- **Interp Map:** Interpolated spatial representation of phase patterns
- **Grp Scatter:** Combined scatter plot from multiple warped datasets (group analysis)
- **Grp Avg Map:** Averaged phase map across multiple animals (group analysis)

### Interactive Selection

*   **Click-to-Select:** In the **"CoM"** or **"Phase Map"** tabs, you can click directly on any cell's dot.
*   **View Trajectory:** The application will automatically switch to the **"Trajectories"** tab and display the full trajectory of the selected cell.
*   **View Intensity Trace:** The application will also update the line plot at the bottom of the **"Heatmap"** tab to show the raw intensity trace for the selected cell.
*   **Cross-Plot Highlighting:** The selected cell will be highlighted with a circle on all relevant plots so you never lose context.

### Trajectory Inspector with Frame Scrubbing

The **"Trajectories"** tab is a powerful validation tool.
*   Use the **"Previous" / "Next"** buttons to step through different cells.
*   Use the **"Frame"** slider at the bottom to scrub through the movie frame by frame. The background image will update, and a prominent marker will show the cell's exact position at that moment in time, allowing you to visually confirm the tracking accuracy.

### Advanced Heatmap Features

The **"Heatmap"** tab is a central analysis tool.
*   **Sorting:** The radio buttons allow you to sort the heatmap by scientifically relevant criteria:
    *   **Y-coordinate:** Sorts cells spatially from top to bottom.
    *   **Phase:** Sorts cells by their calculated peak time, making temporal patterns (like waves) easy to see.
    *   **Rhythmicity:** Sorts cells by the quality of their rhythm (either SNR or R-squared). This brings the "best" cells to the top.
*   **Rhythm Emphasis:** In the "Phase Map Parameters" panel, check **"Emphasize rhythmic cells"**. This will visually de-emphasize non-rhythmic cells in both the "CoM" plot (making them gray) and the "Heatmap" (covering them with a semi-transparent gray mask), allowing you to focus on the data of interest without losing context.

### Choice of Analysis Method

In the "Phase Map Parameters" panel, you can choose your rhythm analysis engine from the **"Analysis Method"** dropdown.

*   **FFT (SNR):** This model-free method is excellent for discovering rhythms and analyzing non-sinusoidal data. The "rhythmicity score" is a signal-to-noise ratio, where a value > 2.0 is a good starting point for a confident rhythm.
*   **Cosinor (p-value):** This model-based method is statistically rigorous and provides standard circadian parameters. It uses a two-factor threshold:
    *   **p-value:** The statistical significance of the rhythm's existence (e.g., `<= 0.05`).
    *   **R-squared:** The "goodness of fit," or how much of the data's variance is explained by the cosine model (e.g., `>= 0.3`).

---

## Acknowledgements

This project is a significant refactoring, modernization, and extension of an original analysis workflow developed in the Herzog Lab. The core scientific algorithms are based on the excellent foundational work of others, and we gratefully acknowledge their contributions:

*   **Ben Bales (Petzold Lab, UCSB):** For writing the original `extract_data.py` Python script, which forms the core of the neuron tracking algorithm.
*   **Matt Tso (Herzog Lab, WashU):** For developing the initial workflow, creating the supplementary analysis scripts, and for the presentation that inspired this project.