# gui/state.py
class AnalysisState:
    """A simple class to hold all application state."""
    def __init__(self):
        self.reset()

    def reset(self):
        # File paths
        self.project_path = "" # Path to the .ntp project file
        self.input_movie_path = ""
        self.output_basename = ""
        self.atlas_roi_path = ""
        self.reference_roi_path = ""
        
        # Lists for workflow panels
        self.target_roi_paths = []
        self.warp_param_paths = []
        self.group_data_paths = []
        
        # In-memory data
        self.unfiltered_data = {}
        self.loaded_data = {}

        # Group Comparison State 
        self.reference_grid_def = None  
        self.reference_raw_data = None