import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


class AspectRatioWidget(QtWidgets.QWidget):
    """
    A widget that maintains a square (1:1) aspect ratio for its content.
    """
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.aspect_ratio = 1.0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        self.widget = widget

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Calculate size that maintains aspect ratio
        w = event.size().width()
        h = event.size().height()
        
        if w / h > self.aspect_ratio:
            # Width is limiting factor
            new_h = h
            new_w = int(h * self.aspect_ratio)
        else:
            # Height is limiting factor
            new_w = w
            new_h = int(w / self.aspect_ratio)
        
        # Center the widget
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        self.widget.setGeometry(x, y, new_w, new_h)


# ------------------------------------------------------------
# Tooltip Helper
# ------------------------------------------------------------
class Tooltip(QtCore.QObject):
    """A custom tooltip that can be styled and supports rich text."""
    _instance = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QtWidgets.QLabel(
            None,
            flags=QtCore.Qt.ToolTip | QtCore.Qt.BypassWindowManagerHint,
        )
        self._label.setWindowOpacity(0.95)
        self._label.setStyleSheet("""
            QLabel {
                border: 1px solid #555;
                padding: 5px;
                background-color: #ffffe0; /* Light yellow */
                color: #000;
                border-radius: 3px;
            }
        """)
        self._label.setWordWrap(True)
        self._label.setAlignment(QtCore.Qt.AlignLeft)
        self._timer = QtCore.QTimer(self, singleShot=True)
        self._timer.setInterval(750)  # ms delay
        self._timer.timeout.connect(self._show)
        self._text = ""

    def _show(self):
        if not self._text:
            return
        self._label.setText(self._text)
        self._label.adjustSize()
        pos = QtGui.QCursor.pos()
        x = pos.x() + 20
        y = pos.y() + 20
        
        screen_geo = QtWidgets.QApplication.desktop().availableGeometry(pos)
        if x + self._label.width() > screen_geo.right():
            x = pos.x() - self._label.width() - 20
        if y + self._label.height() > screen_geo.bottom():
            y = pos.y() - self._label.height() - 20
            
        self._label.move(x, y)
        self._label.show()

    def show_tooltip(self, text):
        self._text = text
        self._timer.start()

    def hide_tooltip(self):
        self._timer.stop()
        self._label.hide()

    @classmethod
    def install(cls, widget, text):
        if cls._instance is None:
            # Ensure the instance has a parent to be properly managed
            parent = widget.window() if widget.window() else QtWidgets.QApplication.instance()
            cls._instance = cls(parent)
        
        widget.setMouseTracking(True)
        
        class Filter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QtCore.QEvent.Enter:
                        cls._instance.show_tooltip(text)
                    elif event.type() == QtCore.QEvent.Leave:
                        cls._instance.hide_tooltip()
                return False

        if not hasattr(widget, '_tooltip_filter'):
            widget._tooltip_filter = Filter(widget)
            widget.installEventFilter(widget._tooltip_filter)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clear_layout(layout: QtWidgets.QLayout):
    """Remove all widgets/items from a layout without replacing the layout object."""
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)


def add_mpl_to_tab(tab: QtWidgets.QWidget):
    """
    Ensure the tab has a QVBoxLayout, clear it, and add a FigureCanvas+Toolbar.
    Returns (fig, canvas).
    """
    layout = tab.layout()
    if layout is None:
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
    clear_layout(layout)

    # Create figure with square aspect ratio for consistency
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, tab)

    layout.addWidget(toolbar)
    layout.addWidget(canvas, 1)

    return fig, canvas

def project_points_to_polyline(points, polyline):
    """
    Projects 2D points onto a multi-segment polyline.
    Returns the normalized arc-length position 's' [0.0, 1.0] for each point.
    
    Args:
        points (Nx2 array): Cell coordinates.
        polyline (Mx2 array): Ordered vertices of the Phase Axis.
        
    Returns:
        s (N array): Position along the curve (0=Start, 1=End).
    """
    points = np.asarray(points)
    polyline = np.asarray(polyline)
    
    if len(polyline) < 2:
        return np.zeros(len(points))
        
    # 1. Calculate segment lengths and total length
    # diffs = P[i+1] - P[i]
    seg_vecs = np.diff(polyline, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = np.sum(seg_lens)
    
    # Cumulative length at each vertex [0, L1, L1+L2, ...]
    cum_len = np.concatenate(([0], np.cumsum(seg_lens)))
    
    s_values = []
    
    for p in points:
        best_dist = np.inf
        best_s = 0.0
        
        # Check each segment
        for i in range(len(seg_vecs)):
            # Segment from A to B
            A = polyline[i]
            B = polyline[i+1]
            vec_AB = seg_vecs[i]
            len_AB = seg_lens[i]
            
            if len_AB == 0: continue
                
            # Vector from A to Point
            vec_AP = p - A
            
            # Project AP onto AB (t = dot(AP, AB) / |AB|^2)
            t = np.dot(vec_AP, vec_AB) / (len_AB**2)
            
            # Clamp t to segment [0, 1]
            t_clamped = np.clip(t, 0.0, 1.0)
            
            # Closest point on segment
            closest = A + t_clamped * vec_AB
            dist = np.linalg.norm(p - closest)
            
            if dist < best_dist:
                best_dist = dist
                # Calculate absolute distance along the whole polyline
                abs_pos = cum_len[i] + (t_clamped * len_AB)
                best_s = abs_pos / total_len
                
        s_values.append(best_s)
        
    return np.array(s_values)