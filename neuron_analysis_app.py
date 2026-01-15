import sys
import os
from PyQt5 import QtWidgets, QtCore
from gui.main_window import MainWindow
from gui.theme import apply_app_style, setup_matplotlib_theme

def main():
    # Enable High DPI display with PyQt5
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    
    # Set application details for QSettings
    QtCore.QCoreApplication.setOrganizationName("JonesLab")
    QtCore.QCoreApplication.setApplicationName("NeuronAnalysis")

    # Apply Theme
    apply_app_style(app)
    setup_matplotlib_theme()

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
