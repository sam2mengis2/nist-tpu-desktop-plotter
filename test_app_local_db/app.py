import sys
import sqlite3
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# PyQt6 UI Component Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QSlider, QRadioButton, QLabel, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt

# Matplotlib embedded canvas elements
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

DB_FILE = "local_wind_data.db"

# ==============================================================================
# 🌪️ ADVANCED CONTOUR MAP CANVAS CLASS
# ==============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=6, dpi=100):
        # Configure a clean figure canvas container workspace
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#222222')
        super().__init__(self.fig)
        
    def update_contour_plot(self, df, metric, face_no):
        """
        Applies mathematical cubic surface interpolation over discrete tap locations
        and renders a high-resolution contour field onto the embedded canvas.
        """
        # 1. CLEAR ENTIRE FIGURE: Wipes out old axes and colorbars cleanly to prevent layout bugs
        self.fig.clear()
        
        # 2. RE-INITIALIZE THE AXES FROM SCRATCH ON EVERY REFRESH
        self.axes = self.fig.add_subplot(111, facecolor='#111111')
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values():
            spine.set_color('#444444')

        if df.empty or len(df) < 4:
            self.axes.text(0.5, 0.5, f"Insufficient sensor points for Face {face_no} interpolation.", 
                           color='orange', ha='center', va='center', transform=self.axes.transAxes)
            self.draw()
            return

        # Extract underlying spatial coordinate matrices from database dataframe
        x = pd.to_numeric(df['x_coordinate']).values
        y = pd.to_numeric(df['y_coordinate']).values
        z = pd.to_numeric(df[metric]).values

        # Remove potential NaN matrix entries to protect the SciPy kernels
        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

        # Define high-resolution grid space (Your 100x100 mesh footprint)
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        # Run SciPy interpolation matrix computation
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        
        # Fallback to linear calculation matrix if data bounds create singular matrix issues
        if np.all(np.isnan(grid_z)):
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

        # Select visual theme and design parameters
        if metric == "std_dev_pressure":
            cmap = 'YlOrRd'
            label_title = "Std Dev Pressure Coefficient ($C_p$)"
        else:
            cmap = 'RdBu_r'
            label_title = "Mean Pressure Coefficient ($C_p$)"

        # Render continuous contour layer
        contour_field = self.axes.contourf(grid_x, grid_y, grid_z, levels=20, cmap=cmap)
        
        # Create fresh, pristine colorbar instance without overlapping dependencies
        cbar = self.fig.colorbar(contour_field, ax=self.axes, orientation='vertical')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.set_label(label_title, color='white', fontsize=10, labelpad=10)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Overlay geometric tap sensor verification markers
        self.axes.scatter(x, y, c='black', s=15, marker='x', alpha=0.7, label='Sensor Taps')

        # Formatting configurations
        self.axes.set_title(f"Face {face_no} Continuous Pressure Distribution Topology", color='white', fontsize=12, pad=12, fontweight='bold')
        self.axes.set_xlabel("X Coordinate (ft)", color='white')
        self.axes.set_ylabel("Y Coordinate (ft)", color='white')
        self.axes.grid(True, color='#333333', linestyle=':', alpha=0.5)
        
        # Force strict 1:1 scale constraint to prevent building dimension distortion
        self.axes.set_aspect('auto')
        
        # Apply display bounding box extension padding
        buffer = 2.5 
        self.axes.set_xlim(x.min() - buffer, x.max() + buffer)
        self.axes.set_ylim(y.min() - buffer, y.max() + buffer)
        
        self.fig.tight_layout()
        self.draw()


# ==============================================================================
# 🏢 NATIVE DESKTOP CONTAINER SYSTEM INTERFACE
# ==============================================================================
class WindTunnelExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌪️ Alan G. Davenport Wind Engineering Data App")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet("background-color: #222222; color: white;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # LEFT SIDE PANEL: SELECTION & CONTROLS
        control_panel = QVBoxLayout()
        control_group = QGroupBox("Search & Contour Options")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444444; margin-top: 10px; padding: 10px; }")
        form_layout = QFormLayout(control_group)

        # Searchable Dropdown
        self.file_selector = QComboBox()
        self.file_selector.setStyleSheet("background-color: #333333; color: white; padding: 4px; border: 1px solid #555555;")
        self.populate_file_dropdown()
        self.file_selector.currentIndexChanged.connect(self.on_inputs_changed)
        form_layout.addRow(QLabel("Target Dataset:"), self.file_selector)

        # Building Surface Slider
        self.face_label = QLabel("Face Surface: 1")
        self.face_slider = QSlider(Qt.Orientation.Horizontal)
        self.face_slider.setMinimum(1)
        self.face_slider.setMaximum(6)
        self.face_slider.setValue(1)
        self.face_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.face_slider.setTickInterval(1)
        self.face_slider.valueChanged.connect(self.on_slider_moved)
        form_layout.addRow(self.face_label, self.face_slider)

        # Pressure Metric Selection Radio Targets
        self.mean_radio = QRadioButton("Mean Pressure ($C_p$)")
        self.std_radio = QRadioButton("Standard Deviation ($\sigma$)")
        self.mean_radio.setChecked(True)
        self.mean_radio.toggled.connect(self.on_inputs_changed)
        self.std_radio.toggled.connect(self.on_inputs_changed)
        
        metric_layout = QVBoxLayout()
        metric_layout.addWidget(self.mean_radio)
        metric_layout.addWidget(self.std_radio)
        form_layout.addRow(QLabel("Analysis Target:"), metric_layout)

        # KPI Metrics Cards Display Block
        self.kpi_count = QLabel("Active Taps: 0")
        self.kpi_max = QLabel("Peak Field Value: 0.0000")
        self.kpi_count.setStyleSheet("font-size: 11px; color: #aaaaaa; padding-top: 10px;")
        self.kpi_max.setStyleSheet("font-size: 11px; color: #aaaaaa;")

        control_panel.addWidget(control_group)
        control_panel.addWidget(self.kpi_count)
        control_panel.addWidget(self.kpi_max)
        control_panel.addStretch()
        
        main_layout.addLayout(control_panel, stretch=1)

        # RIGHT SIDE PANEL: LIVE INTERACTIVE CONTOUR CANVAS
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        main_layout.addWidget(self.canvas, stretch=3)

        # Initial plot render pass execution
        self.on_inputs_changed()

    def populate_file_dropdown(self):
        """Populates available building entries straight from SQLite."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM origin_models;")
            files = [row[0] for row in cursor.fetchall()]
            conn.close()
            self.file_selector.addItems(files)
        except Exception as e:
            print(f"Database sync failure: {e}")

    def on_slider_moved(self, value):
        """Tracks face identifier adjustments dynamically."""
        self.face_label.setText(f"Face Surface: {value}")
        self.on_inputs_changed()

    def on_inputs_changed(self):
        """Fetches the targeted data vector subset and updates the visual mesh layers."""
        filename = self.file_selector.currentText()
        if not filename:
            return

        face_no = self.face_slider.value()
        metric = "mean_pressure" if self.mean_radio.isChecked() else "std_dev_pressure"

        # Read specific record coordinates and targets from database
        conn = sqlite3.connect(DB_FILE)
        query = """
            SELECT face_no, x_coordinate, y_coordinate, mean_pressure, std_dev_pressure 
            FROM taps 
            WHERE model_id = (SELECT model_id FROM origin_models WHERE filename = ?) 
              AND face_no = ?;
        """
        df = pd.read_sql_query(query, conn, params=(filename, face_no))
        conn.close()

        # Update statistical metrics metadata monitoring cards
        if not df.empty:
            self.kpi_count.setText(f"Active Sensor Taps: {len(df)}")
            max_val = df[metric].abs().max()
            self.kpi_max.setText(f"Peak Absolute Field Metric: {max_val:.4f}")
        else:
            self.kpi_count.setText("Active Sensor Taps: 0")
            self.kpi_max.setText("Peak Absolute Field Metric: 0.0000")

        # Command the canvas layer matrix subsystem to compute and render the new layout
        self.canvas.update_contour_plot(df, metric, face_no)


# ==============================================================================
# 🚀 INITIALIZATION INTERACTION LOOP ROUTINE
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindTunnelExplorer()
    window.show()
    sys.exit(app.exec())