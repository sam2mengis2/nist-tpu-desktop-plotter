import os
import sys
from tkinter.tix import ComboBox
# Make sure your analyzer class imports are at the top (pandas, matplotlib, pyhdf, etc.)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                             QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from display_charts import NIST_DATA_ANALYZER

class AeroDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 1. Internal Memory ---
        self.analyzer = None      # Will hold your class instance
        self.dataframes = {}      # Will hold the extracted data dictionary

        # --- 2. Window Configuration ---
        self.setWindowTitle("Aerodynamic Data Viewer V1")
        self.setFixedSize(450, 300)

        # --- 3. Create UI Elements ---
        self.status_label = QLabel("Status: Waiting for NIST .HDF File...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.status_label.font()
        font.setPointSize(12)
        self.status_label.setFont(font)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)

        self.btn_load = QPushButton("1. Load .HDF File")
        self.btn_plot_3d = QPushButton("2. Render 3D Wireframe")
        self.btn_plot_2d = QPushButton("3. Render 2D Surface Map")
        # Dynamic Contour Section (Face Selection)
        self.lbl_face = QLabel("Select Face:")
        self.combo_face = QComboBox()
        self.btn_plot_contour = QPushButton("4. Render Mean Contour Map")

        # Dynamic Timeseries Section (Tap Selection)
        self.lbl_tap = QLabel("Select Tap:")
        self.combo_tap = QComboBox()
        self.btn_plot_timeseries = QPushButton("5. Render Timeseries Chart")

        # Disable plotting until a file is actually loaded
        self.btn_plot_3d.setEnabled(False)
        self.btn_plot_2d.setEnabled(False)
        self.btn_plot_contour.setEnabled(False)
        self.btn_plot_timeseries.setEnabled(False)

        # --- 4. Routing (Signals & Slots) ---
        self.btn_load.clicked.connect(self.load_file)
        self.btn_plot_3d.clicked.connect(self.trigger_3d_plot)
        self.btn_plot_2d.clicked.connect(self.trigger_2d_plot)
        self.btn_plot_contour.clicked.connect(self.trigger_mean_contour)
        self.btn_plot_timeseries.clicked.connect(self.trigger_full_timeseries)

        # --- 5. Layout Management ---
        master_layout = QVBoxLayout()
        master_layout.addWidget(self.status_label)
        master_layout.addWidget(divider)
        master_layout.addWidget(self.btn_load)
        master_layout.addWidget(self.btn_plot_3d)
        master_layout.addWidget(self.btn_plot_2d)

        # Horizontal row for Contour Inputs
        contour_layout = QHBoxLayout()
        contour_layout.addWidget(self.lbl_face)
        contour_layout.addWidget(self.combo_face)
        contour_layout.addWidget(self.btn_plot_contour, stretch=1)
        master_layout.addLayout(contour_layout)

        # Horizontal row for Timeseries Inputs
        timeseries_layout = QHBoxLayout()
        timeseries_layout.addWidget(self.lbl_tap)
        timeseries_layout.addWidget(self.combo_tap)
        timeseries_layout.addWidget(self.btn_plot_timeseries, stretch=1)
        master_layout.addLayout(timeseries_layout)

        container = QWidget()
        container.setLayout(master_layout)
        self.setCentralWidget(container)


    def disable_controls(self):
        """Helper to lock interface on startup."""
        for widget in [self.btn_plot_3d, self.btn_plot_2d, self.combo_face, 
                       self.btn_plot_contour, self.combo_tap, self.btn_plot_timeseries]:
            widget.setEnabled(False)

    # --- 6. Backend Logic ---
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NIST .HDF File", "", "HDF Files (*.hdf *.hdf4 *.h4);;All Files (*)"
        )

        if file_path:
            import os
            clean_path = os.path.normpath(file_path)
            self.status_label.setText("Status: Extracting Data...")
            QApplication.processEvents() 
            
            try:
                self.analyzer = NIST_DATA_ANALYZER(clean_path)
                self.dataframes = self.analyzer.extract_dataframes()
                
                if self.dataframes:
                    self.status_label.setText(f"Status: Loaded Successfully ✅")
                    self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                    
                    # 🟢 DYNAMICALLY POPULATE DROPDOWNS
                    # 1. Populate Faces from the Flat Tap Coordinates file
                    flat_tap_df = self.dataframes['Flat_Tap_Coordinates'].T
                    unique_faces = sorted(flat_tap_df[1].dropna().unique())
                    self.combo_face.clear()
                    self.combo_face.addItems([str(int(f)) for f in unique_faces])

                    # 2. Populate Taps from the Time Series columns length
                    time_series_df = self.dataframes['Time_Series'].T
                    total_taps = time_series_df.shape[1] # Number of columns = number of taps
                    self.combo_tap.clear()
                    self.combo_tap.addItems([str(i) for i in range(total_taps)])

                    # Unlock everything
                    for widget in [self.btn_plot_3d, self.btn_plot_2d, self.combo_face, 
                                   self.btn_plot_contour, self.combo_tap, self.btn_plot_timeseries]:
                        widget.setEnabled(True)
                else:
                    self.status_label.setText("Status: Extraction Failed ❌")
            except Exception as e:
                self.status_label.setText("Status: Error reading file ❌")
                print(f"UI Error: {e}")

    def trigger_3d_plot(self):
        try:
            # Extract the specific datasets your 3D function needs.
            # (Note: .T is used here assuming your matrices need to be transposed based on earlier steps)
            tap_df = self.dataframes['Tap_Coordinates_3D'].T
            frame_df = self.dataframes['Wire_Frame_Lines_3D'].T
            corners_df = self.dataframes['Building_Corners_3D'].T
            
            # Fire your class method! (This will pop up the Matplotlib window)
            self.analyzer.get_wind_frame_plot_3D(tap_df, frame_df, corners_df)
            
        except KeyError as e:
            QMessageBox.critical(self, "Missing Data", f"Could not find dataset: {e}")

    def trigger_2d_plot(self):
        try:
            # Extract the specific datasets your 2D function needs
            flat_frames_df = self.dataframes['Flat_Wire_Frame_Lines'].T
            flat_corners_df = self.dataframes['Building_Corners_Flat'].T
            flat_tap_df = self.dataframes['Flat_Tap_Coordinates'].T
            
            # Fire your class method!
            self.analyzer.get_wind_2d_plot(flat_frames_df, flat_corners_df, flat_tap_df)
            
        except KeyError as e:
            QMessageBox.critical(self, "Missing Data", f"Could not find dataset: {e}")


    def trigger_mean_contour(self):
        try:
            # Grab the selected face number directly from the dropdown box!
            selected_face = float(self.combo_face.currentText())
            
            flat_tap_coords = self.dataframes['Flat_Tap_Coordinates'].T
            pressure_df = self.dataframes['Time_Series'].T
            
            # Fire your class function using the live input
            self.analyzer.get_mean_contour(selected_face, flat_tap_coords, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render contour map: {e}")

    def trigger_full_timeseries(self):
        try:
            # Grab the selected tap channel number from the dropdown box!
            selected_tap = int(self.combo_tap.currentText())
            
            pressure_time_series = self.dataframes['Time_Series'].T
            
            # Fire your class function to show the history graph
            self.analyzer.view_full_series_tap(pressure_time_series, selected_tap)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render timeseries: {e}")

        

# --- Execution Loop ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = AeroDataViewer()
    window.show()
    sys.exit(app.exec())





            


    