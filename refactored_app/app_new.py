import os
import sys
import numpy as np
# Ensure standard PyQt6 dependencies are localized
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                             QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from TPU_plotters import TPU_HIGH_RISE_ANALYZER
from NIST_plotter import NIST_DATA_ANALYZER

class AeroDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.analyzer = None      
        self.file_type = None      
        self.dataframes = {}  # Safe structural storage for tracking data caches

        self.setWindowTitle("Adaptive Wind Data Viewer")
        self.setFixedSize(500, 480)  # Standardized geometry baseline to support all panels cleanly

        # --- 1. Universal Header Elements ---
        self.status_label = QLabel("Status: Waiting for Data File (.HDF / .mat)...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_load = QPushButton("📁 Load Data File")
        self.btn_load.clicked.connect(self.load_file)

        # =====================================================================
        # 2. NIST CONTAINER SETUP (Linked to NIST_plotter.py)
        # =====================================================================
        self.nist_frame = QFrame()
        nist_layout = QVBoxLayout(self.nist_frame)
        
        self.lbl_nist = QLabel("<b>NIST 3D Analysis Options:</b>")
        nist_layout.addWidget(self.lbl_nist)

        # Standalone Structural Layout Hooks
        self.btn_plot_3d_nist = QPushButton("2. Render 3D Wireframe")
        self.btn_plot_2d_nist = QPushButton("3. Render 2D Surface Map")
        nist_layout.addWidget(self.btn_plot_3d_nist)
        nist_layout.addWidget(self.btn_plot_2d_nist)

        # Dynamic Spatial Contour Sub-section
        nist_contour_layout = QHBoxLayout()
        self.lbl_face_nist = QLabel("Select Face:")
        self.combo_face_nist = QComboBox()
        
        nist_btn_box = QVBoxLayout()
        self.btn_plot_contour_nist = QPushButton("4. Render Mean Contour Map")
        self.btn_plot_std_contour_nist = QPushButton("4b. Render Std Dev Contour Map")
        nist_btn_box.addWidget(self.btn_plot_contour_nist)
        nist_btn_box.addWidget(self.btn_plot_std_contour_nist)
        
        nist_contour_layout.addWidget(self.lbl_face_nist)
        nist_contour_layout.addWidget(self.combo_face_nist)
        nist_contour_layout.addLayout(nist_btn_box, stretch=1)
        nist_layout.addLayout(nist_contour_layout)

        # Dynamic Signal History Sub-section
        nist_timeseries_layout = QHBoxLayout()
        self.lbl_tap_nist = QLabel("Select Tap:")
        self.combo_tap_nist = QComboBox()
        self.btn_plot_timeseries_nist = QPushButton("5. Render Timeseries Chart")
        
        nist_timeseries_layout.addWidget(self.lbl_tap_nist)
        nist_timeseries_layout.addWidget(self.combo_tap_nist)
        nist_timeseries_layout.addWidget(self.btn_plot_timeseries_nist, stretch=1)
        nist_layout.addLayout(nist_timeseries_layout)

        # Interface Slots Mapping
        self.btn_plot_3d_nist.clicked.connect(self.trigger_3d_plot)
        self.btn_plot_2d_nist.clicked.connect(self.trigger_2d_plot)
        self.btn_plot_contour_nist.clicked.connect(self.trigger_mean_contour)
        self.btn_plot_std_contour_nist.clicked.connect(self.trigger_nist_std_contour)
        self.btn_plot_timeseries_nist.clicked.connect(self.trigger_full_timeseries)

        # =====================================================================
        # 3. TPU CONTAINER SETUP (Linked to TPU_plotters.py)
        # =====================================================================
        self.tpu_frame = QFrame()
        tpu_layout = QVBoxLayout(self.tpu_frame)
        
        self.lbl_tpu = QLabel("<b>TPU Analysis Options:</b>")
        tpu_layout.addWidget(self.lbl_tpu)

        # Spatial Sensor Matrix Reference Check
        self.btn_plot_grid = QPushButton("1. Render Channel Positions Grid")
        self.btn_plot_grid.clicked.connect(self.trigger_grid_plot)
        tpu_layout.addWidget(self.btn_plot_grid)
        
        # Time-Series Sub-section
        tpu_timeseries_layout = QHBoxLayout()
        self.lbl_tap_tpu = QLabel("Select Tap:")
        self.combo_tap_tpu = QComboBox()
        self.btn_tpu_timeseries = QPushButton("2. Render Timeseries Chart")
        self.btn_tpu_timeseries.clicked.connect(self.trigger_tpu_timeseries)
        
        tpu_timeseries_layout.addWidget(self.lbl_tap_tpu)
        tpu_timeseries_layout.addWidget(self.combo_tap_tpu)
        tpu_timeseries_layout.addWidget(self.btn_tpu_timeseries, stretch=1)
        tpu_layout.addLayout(tpu_timeseries_layout)

        # Localized Face Contour Mapping Section (Updated for multi-face control blocks)
        tpu_contour_layout = QHBoxLayout()
        self.lbl_face_tpu = QLabel("Select Face:")
        self.combo_face_tpu = QComboBox()
        
        tpu_btn_box = QVBoxLayout()
        self.btn_tpu_contour = QPushButton("3. Render Mean Contour Map")
        self.btn_tpu_std_contour = QPushButton("4. Render Std Dev Contour Map")

        self.btn_tpu_contour.clicked.connect(self.trigger_tpu_contour)
        self.btn_tpu_std_contour.clicked.connect(self.trigger_tpu_std_contour)
        tpu_btn_box.addWidget(self.btn_tpu_contour)
        tpu_btn_box.addWidget(self.btn_tpu_std_contour)
        
        tpu_contour_layout.addWidget(self.lbl_face_tpu)
        tpu_contour_layout.addWidget(self.combo_face_tpu)
        tpu_contour_layout.addLayout(tpu_btn_box, stretch=1)
        tpu_layout.addLayout(tpu_contour_layout)



        

        # =====================================================================
        # 4. MASTER LAYOUT ASSEMBLY
        # =====================================================================
        self.master_layout = QVBoxLayout()
        self.master_layout.addWidget(self.status_label)
        self.master_layout.addWidget(self.btn_load)
        
        self.master_layout.addWidget(self.nist_frame)
        self.master_layout.addWidget(self.tpu_frame)
        self.master_layout.addStretch()

        self.nist_frame.hide()
        self.tpu_frame.hide()

        container = QWidget()
        container.setLayout(self.master_layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Wind Database File", 
            "", 
            "Supported Files (*.hdf *.hdf4 *.h4 *.mat);;All Files (*)"
        )

        if file_path:
            clean_path = os.path.normpath(file_path)
            ext = os.path.splitext(clean_path)[1].lower()
            
            self.status_label.setText("Status: Extracting Data Matrix...")
            self.status_label.setStyleSheet("color: black; font-weight: normal;")
            
            self.nist_frame.hide()
            self.tpu_frame.hide()
            QApplication.processEvents() 
            
            try:
                # =============================================================
                # 🟢 PIPELINE A: HANDLING TPU MATLAB FILES
                # =============================================================
                if ext == '.mat':
                    self.file_type = 'TPU'
                    
                    self.analyzer = TPU_HIGH_RISE_ANALYZER(clean_path)
                    loc_df = self.analyzer.get_loc_df()
                    pressure_df = self.analyzer.get_timeseries_df()
                    
                    # Track available discrete channels matching the array structure
                    total_taps = pressure_df.shape[1]
                    self.combo_tap_tpu.clear()
                    self.combo_tap_tpu.addItems([str(i) for i in range(total_taps)])
                    
                    # Dynamically track structural Face indices directly from the location records
                    unique_faces = sorted(loc_df['Face_No'].dropna().unique())
                    self.combo_face_tpu.clear()
                    self.combo_face_tpu.addItems([str(int(f)) for f in unique_faces])
                    
                    self.status_label.setText("Mode: TPU MATLAB Engine Active ✅")
                    self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                    self.tpu_frame.show()

                # =============================================================
                # 🔵 PIPELINE B: HANDLING NIST HDF FILES
                # =============================================================
                elif ext in ['.hdf', '.hdf4', '.h4']:
                    self.file_type = 'NIST'
                    
                    self.analyzer = NIST_DATA_ANALYZER(clean_path)
                    self.dataframes = self.analyzer.extract_dataframes()
                    
                    if self.dataframes:
                        flat_tap_df = self.dataframes['Flat_Tap_Coordinates'].T
                        unique_faces = sorted(flat_tap_df[1].dropna().unique())
                        self.combo_face_nist.clear()
                        self.combo_face_nist.addItems([str(int(f)) for f in unique_faces])

                        time_series_df = self.dataframes['Time_Series'].T
                        total_taps = time_series_df.shape[1]
                        self.combo_tap_nist.clear()
                        self.combo_tap_nist.addItems([str(i) for i in range(total_taps)])
                        
                        self.status_label.setText("Mode: NIST HDF4 Engine Active ✅")
                        self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                        self.nist_frame.show()
                    else:
                        self.status_label.setText("Status: NIST Extraction Failed ❌")
                        return

                self.adjustSize()

            except Exception as e:
                self.status_label.setText("Status: Error reading file ❌")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                print(f"UI Loading Error: {e}")

    # =====================================================================
    # NIST EVENT ACTIONS
    # =====================================================================
    def trigger_3d_plot(self):
        try:
            tap_df = self.dataframes['Tap_Coordinates_3D'].T
            frame_df = self.dataframes['Wire_Frame_Lines_3D'].T
            corners_df = self.dataframes['Building_Corners_3D'].T
            self.analyzer.get_wind_frame_plot_3D(tap_df, frame_df, corners_df)
        except KeyError as e:
            QMessageBox.critical(self, "Missing Data", f"Could not find dataset: {e}")

    def trigger_2d_plot(self):
        try:
            flat_frames_df = self.dataframes['Flat_Wire_Frame_Lines'].T
            flat_corners_df = self.dataframes['Building_Corners_Flat'].T
            flat_tap_df = self.dataframes['Flat_Tap_Coordinates'].T
            self.analyzer.get_wind_2d_plot(flat_frames_df, flat_corners_df, flat_tap_df)
        except KeyError as e:
            QMessageBox.critical(self, "Missing Data", f"Could not find dataset: {e}")

    def trigger_mean_contour(self):
        try:
            selected_face = float(self.combo_face_nist.currentText())
            flat_tap_coords = self.dataframes['Flat_Tap_Coordinates'].T
            pressure_df = self.dataframes['Time_Series'].T
            self.analyzer.get_mean_contour(selected_face, flat_tap_coords, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render contour map: {e}")

    def trigger_nist_std_contour(self):
        try:
            selected_face = float(self.combo_face_nist.currentText())
            flat_tap_coords = self.dataframes['Flat_Tap_Coordinates'].T
            pressure_df = self.dataframes['Time_Series'].T
            self.analyzer.get_std_contour(selected_face, flat_tap_coords, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render standard deviation map: {e}")

    def trigger_full_timeseries(self):
        try:
            selected_tap = int(self.combo_tap_nist.currentText())
            pressure_time_series = self.dataframes['Time_Series'].T
            self.analyzer.view_full_series_tap(pressure_time_series, selected_tap)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render timeseries: {e}")

    # =====================================================================
    # TPU EVENT ACTIONS
    # =====================================================================
    def trigger_grid_plot(self):
        try:
            loc_df = self.analyzer.get_loc_df()
            pressure_df = self.analyzer.get_timeseries_df()
            self.analyzer.get_channel_plot(loc_df, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render plot: {e}")

    def trigger_tpu_timeseries(self):
        try:
            pressure_df = self.analyzer.get_timeseries_df()
            chosen_tap = int(self.combo_tap_tpu.currentText())
            self.analyzer.view_all_steps(pressure_df, chosen_tap)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render plot: {e}")

    def trigger_tpu_contour(self):
            pressure_df = self.analyzer.get_timeseries_df()
            loc_df = self.analyzer.get_loc_df()
            chosen_face = int(self.combo_face_tpu.currentText())
            self.analyzer.mean_cp_contour(pressure_df, loc_df, chosen_face)

    def trigger_tpu_std_contour(self):
            pressure_df = self.analyzer.get_timeseries_df()
            loc_df = self.analyzer.get_loc_df()
            chosen_face = int(self.combo_face_tpu.currentText())
            self.analyzer.std_cp_contour(pressure_df, loc_df, chosen_face)


# --- Execution Loop ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = AeroDataViewer()
    window.show()
    sys.exit(app.exec())