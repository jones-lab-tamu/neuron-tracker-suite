from PyQt5 import QtGui, QtWidgets, QtCore
import qtawesome as qta
import matplotlib.pyplot as plt

def apply_theme(app):
    """
    Applies a modern, clean Light Theme to the QApplication using the Fusion style.
    This provides a softer, more standard desktop application appearance.
    """
    app.setStyle("Fusion")

    # Define the Light Palette
    window_bg = QtGui.QColor(245, 245, 247)   # Very light gray (macOS-like)
    base_color = QtGui.QColor(255, 255, 255)  # White for inputs/lists
    text_color = QtGui.QColor(33, 33, 33)     # Dark gray for text (softer than pure black)
    btn_color = QtGui.QColor(235, 235, 235)   # Light gray for buttons
    accent_color = QtGui.QColor(0, 122, 204)  # Professional Blue accent
    highlight_color = QtGui.QColor(0, 120, 215) # Standard selection blue

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
    palette.setColor(QtGui.QPalette.ButtonText, text_color)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    
    # Links & Highlights
    palette.setColor(QtGui.QPalette.Link, accent_color)
    palette.setColor(QtGui.QPalette.Highlight, highlight_color)
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)

    # Disabled states
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(170, 170, 170))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(170, 170, 170))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Button, QtGui.QColor(240, 240, 240))

    app.setPalette(palette)

    # Set a modern standard font
    font = QtGui.QFont("Segoe UI", 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)
    
    # Additional Stylesheet for softer UI elements
    app.setStyleSheet(f"""
        QToolTip {{ 
            color: #212121; 
            background-color: #ffffff; 
            border: 1px solid #cccccc; 
        }}
        QGroupBox {{ 
            border: 1px solid #d0d0d0; 
            margin-top: 1.5em; 
            border-radius: 6px;
            background-color: #ffffff;
            padding-top: 10px;
        }}
        QGroupBox::title {{ 
            subcontrol-origin: margin; 
            subcontrol-position: top left; 
            padding: 0 5px; 
            color: #007acc; /* Blue title */
            font-weight: bold;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 4px;
            background-color: #ffffff;
            selection-background-color: #0078d7;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border: 1px solid #0078d7;
        }}
        QPushButton {{
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            border-radius: 5px;
            padding: 6px 12px;
            min-width: 60px;
            color: #333333;
        }}
        QPushButton:hover {{
            background-color: #f0f8ff; /* AliceBlue hover */
            border: 1px solid #0078d7;
        }}
        QPushButton:pressed {{
            background-color: #e0e0e0;
        }}
        QPushButton:disabled {{
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            color: #a0a0a0;
        }}
        QListWidget {{
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
        }}
        QTabWidget::pane {{ 
            border: 1px solid #d0d0d0;
            background: #ffffff;
            border-radius: 4px;
        }}
        QTabBar::tab {{
            background: #e0e0e0;
            border: 1px solid #c0c0c0;
            padding: 4px 8px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            color: #555555;
        }}
        QTabBar::tab:selected {{
            background: #ffffff;
            border-bottom-color: #ffffff; /* Blend with pane */
            color: #000000;
            font-weight: bold;
        }}
        QTabBar::tab:hover {{
            background: #f0f0f0;
        }}
    """)

def setup_matplotlib_theme():
    """Configures Matplotlib to look good with the light theme."""
    # Reset to defaults first to clear dark theme settings
    plt.rcdefaults()
    
    # Use a clean style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback if seaborn style isn't available
        plt.style.use('fast')
        plt.grid(True, alpha=0.3)

    # Custom tweaks for the app
    plt.rcParams.update({
        "figure.facecolor": "#f5f5f7", # Match window bg
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#f5f5f7",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.7,
        "lines.linewidth": 1.5,
        "font.size": 10,
        "figure.autolayout": False,
        # Professional color cycle
        "axes.prop_cycle": plt.cycler(color=["#007acc", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]),
    })

def get_icon(name, color='#444444'):
    """Wrapper to get qtawesome icons. Default color is now dark for light theme."""
    return qta.icon(name, color=color)
