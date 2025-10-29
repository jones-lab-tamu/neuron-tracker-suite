# Neuron Tracker and Analysis Suite

## Project Overview

This suite of tools provides a complete, end-to-end pipeline for the analysis of cellular bioluminescence or fluorescence imaging data, specifically tailored for studying circadian rhythms in neuronal ensembles like the Suprachiasmatic Nucleus (SCN).

The entire workflow, from loading a raw movie file to exploring the final, interactive phase map, is handled within a single, user-friendly graphical application. A command-line version is also provided for power users and for automating batch processing.

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

## Workflow & Usage

The primary workflow is a simple two-step process: first, you run the analysis to process a movie file, and second, you explore the results.

### Step 1: Run Analysis

You must first process your raw movie file (`.tif`) to generate the data files. You can use either the GUI for ease of use or the CLI for automation.

#### Option A: Using the Graphical Application (Recommended)

This is the main, all-in-one tool for analysis and visualization.

1.  **Launch the Application:**
    ```
    python neuron_analysis_app.py
    ```
    This will open the main workspace window.

2.  **Load a Movie:**
    *   Click the **"Load Movie..."** button in the top-left "File I/O" panel.
    *   Select your raw movie file (e.g., `.tif`).
    *   The application will automatically suggest an output file basename.

3.  **Run the Analysis:**
    *   (Optional) Adjust any analysis or phase map parameters in the "Control Panel". Hover your mouse over any parameter's label to see a detailed tooltip explaining its function.
    *   Click the large **"Run Full Analysis"** button.
    *   Monitor the progress in the log at the bottom. The analysis may take several minutes.

#### Option B: Using the Command-Line Interface (CLI)

This is ideal for batch processing multiple files.

1.  **Run the analysis with default parameters:**
    ```
    python run_tracker_cli.py path/to/your_movie.tif
    ```
2.  **To see all tunable parameters and their descriptions, run:**
    ```
    python run_tracker_cli.py --help
    ```

**Output Files from Analysis:**
Running the analysis will produce three files, which are the inputs for the visualization step:
*   `your_movie_traces.csv`: A CSV file where the first column is time and subsequent columns are the intensity traces for each detected cell.
*   `your_movie_roi.csv`: A CSV file with the average X,Y coordinates (center of mass) for each cell.
*   `your_movie_trajectories.npy`: A NumPy binary file containing the full X,Y coordinates for every cell at every frame.

---

### Step 2: Explore and Visualize Results

After the analysis is complete, you can explore the results at any time using the same graphical application.

1.  **Launch the Application:**
    ```
    python neuron_analysis_app.py
    ```
2.  **Load the Movie:**
    *   Click the **"Load Movie..."** button and select the *original movie file* that you analyzed.
    *   The application will detect that results for this movie already exist.

3.  **Load the Results:**
    *   The **"Load Existing Results"** button will become active. Click it.
    *   The saved data files will be loaded instantly.

4.  **Explore the Interactive Plots:**
    *   The **"Visualization"** panel on the right will become active, populated with four tabs:
        1.  **Heatmap:** A raster plot of all cell traces, with radio buttons to sort the cells spatially.
        2.  **Center of Mass:** The location of each cell overlaid on the background image, with sliders to adjust image contrast.
        3.  **Trajectory Inspector:** A viewer that lets you step through the movement path of every single cell.
        4.  **Phase Map:** The final spatiotemporal phase map, with a slider to adjust the color range of the relative peak times.

---

## Acknowledgements

This project is a significant refactoring, modernization, and extension of an original analysis workflow developed in the Herzog Lab. The core scientific algorithms are based on the excellent foundational work of others, and we gratefully acknowledge their contributions:

*   **Ben Bales (Petzold Lab, UCSB):** For writing the original `extract_data.py` Python script, which forms the core of the neuron tracking algorithm.
*   **Matt Tso (Herzog Lab, WashU):** For developing the initial workflow, creating the supplementary analysis scripts, and for the presentation that inspired this project.