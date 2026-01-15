from PyQt5 import QtGui, QtWidgets, QtCore
import matplotlib.pyplot as plt

def apply_app_style(app):
    """
    Applies a modern, clean Light Theme to the QApplication using the Fusion style.
    This provides a softer, more standard desktop application appearance.
    """
    app.setStyle("Fusion")

    # Define the Light Palette
    window_bg = QtGui.QColor(250, 250, 252)   # Very light gray/white
    base_color = QtGui.QColor(255, 255, 255)  # White for inputs/lists
    text_color = QtGui.QColor(33, 33, 33)     # Dark gray for text
    btn_color = QtGui.QColor(240, 240, 240)   # Light gray for buttons
    btn_text = QtGui.QColor(33, 33, 33)
    accent_color = QtGui.QColor(0, 110, 200)  # Professional Blue
    
    palette = QtGui.QPalette()
    
    # Window & Backgrounds
    palette.setColor(QtGui.QPalette.Window, window_bg)
    palette.setColor(QtGui.QPalette.WindowText, text_color)
    palette.setColor(QtGui.QPalette.Base, base_color)
    palette.setColor(QtGui.QPalette.AlternateBase, window_bg)
    palette.setColor(QtGui.QPalette.ToolTipBase, base_color)
    palette.setColor(QtGui.QPalette.ToolTipText, text_color)
    palette.setColor(QtGui.QPalette.Text, text_color)
    
    # Buttons
    palette.setColor(QtGui.QPalette.Button, btn_color)
    palette.setColor(QtGui.QPalette.ButtonText, btn_text)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    
    # Links & Highlights
    palette.setColor(QtGui.QPalette.Link, accent_color)
    palette.setColor(QtGui.QPalette.Highlight, accent_color)
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)

    # Disabled states
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(160, 160, 160))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(160, 160, 160))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Button, QtGui.QColor(235, 235, 235))

    app.setPalette(palette)

    # Set fonts
    font_db = QtGui.QFontDatabase()
    system_font = font_db.systemFont(QtGui.QFontDatabase.GeneralFont)
    # Prefer Segoe UI/Roboto/San Francisco if available, else system default
    preferred_families = ["Segoe UI", "Roboto", "Helvetica Neue", "Arial"]
    found_family = system_font.family()
    for fam in preferred_families:
        if fam in font_db.families():
            found_family = fam
            break
            
    font = QtGui.QFont(found_family, 9) # 9pt is standard for desktop apps
    app.setFont(font)
    
    # Global Stylesheet for spacing and modernization
    app.setStyleSheet("""
        QMainWindow {
            background-color: #fafbfc;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            margin-top: 20px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #444;
        }
        QListWidget {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            padding: 4px;
        }
        QListWidget::item {
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 2px;
        }
        QListWidget::item:selected {
            background-color: #e3f2fd;
            color: #0d47a1;
            border: 1px solid #bbdefb;
        }
        QListWidget::item:hover:!selected {
            background-color: #f5f5f5;
        }
        QPushButton {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #fff;
            padding: 5px 12px;
            min-height: 18px;
        }
        QPushButton:hover {
            background-color: #f0f8ff;
            border-color: #0078d7;
        }
        QPushButton:pressed {
            background-color: #e0e0e0;
        }
        QPushButton:disabled {
            background-color: #f0f0f0;
            color: #999;
            border-color: #dcdcdc;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
            min-height: 18px;
            background-color: white;
        }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #0078d7;
        }
        QTabWidget::pane {
            border: 1px solid #ccc;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-bottom: none;
            padding: 6px 12px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 1px solid white; 
            font-weight: bold;
        }
        QStatusBar {
            background-color: #f0f0f0;
            color: #333;
        }
        QToolBar {
            border: none;
            background-color: #fafbfc;
            spacing: 5px;
        }
    """)

def apply_theme(app):
    """Compatibility wrapper for applying style."""
    apply_app_style(app)

def setup_matplotlib_theme():
    """Configures Matplotlib to look good with the light theme."""
    plt.rcdefaults()
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc={
            "axes.facecolor": "white",
            "figure.facecolor": "#fafbfc",
            "grid.color": "#e0e0e0",
        })
    except ImportError:
        # Fallback
        plt.style.use('fast')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "figure.facecolor": "#fafbfc",
            "grid.alpha": 0.5,
        })
    
    # consistent font size
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
    })

def get_icon(name, color='#444444'):
    """
    Returns a standard Qt icon based on the mapped name using QApplication style.
    The mapping covers all fa5s.* keys used in the app. Unknown keys fall back safely.
    
    Args:
        name (str): Key identifying the icon (e.g. 'fa5s.save').
        color (str): Ignored, kept for backward compatibility.
    
    Returns:
        QtGui.QIcon: A standard Qt icon.
    """
    # Coerce input safely
    name = str(name) if name is not None else ""

    style = QtWidgets.QApplication.style()
    
    # Helper to fetch by attribute name safely
    def standard(key):
        if hasattr(QtWidgets.QStyle, key):
            px = getattr(QtWidgets.QStyle, key)
            return style.standardIcon(px)
        return style.standardIcon(QtWidgets.QStyle.SP_FileIcon) # Fallback

    mapping = {
        'fa5s.save': 'SP_DialogSaveButton',
        'fa5s.folder-open': 'SP_DirOpenIcon',
        'fa5s.play': 'SP_MediaPlay', 
        'fa5s.play-circle': 'SP_MediaPlay',
        'fa5s.times': 'SP_DialogCloseButton',
        'fa5s.plus': 'SP_FileDialogNewFolder', 
        'fa5s.minus': 'SP_TitleBarMinButton',
        'fa5s.check': 'SP_DialogApplyButton',
        'fa5s.check-circle': 'SP_DialogApplyButton',
        'fa5s.check-double': 'SP_DialogApplyButton',
        'fa5s.trash': 'SP_TrashIcon',
        'fa5s.file-video': 'SP_FileIcon',
        'fa5s.file-csv': 'SP_FileIcon',
        'fa5s.image': 'SP_FileIcon',
        'fa5s.download': 'SP_ArrowDown',
        'fa5s.upload': 'SP_ArrowUp',
        'fa5s.sync': 'SP_BrowserReload',
        'fa5s.search': 'SP_FileDialogContentsView',
        'fa5s.map': 'SP_DriveNetIcon',
        'fa5s.pen': 'SP_FileDialogDetailedView',
        'fa5s.calculator': 'SP_ComputerIcon',
        'fa5s.undo': 'SP_ArrowBack',
        'fa5s.paw': 'SP_ComputerIcon',
        'fa5s.dna': 'SP_FileLinkIcon',
        'fa5s.users': 'SP_DirIcon',
        'fa5s.filter': 'SP_FileDialogListView',
    }
    
    mapped_key = mapping.get(name)
    if not mapped_key:
        # Fallbacks for specific tricky ones if not in map
        if 'play' in name: mapped_key = 'SP_MediaPlay'
        elif 'check' in name: mapped_key = 'SP_DialogApplyButton'
        else: mapped_key = 'SP_FileIcon'
    
    # Check availability and return icon
    return standard(mapped_key)
