import os
import sys
from tkinter.tix import ComboBox
# Make sure your analyzer class imports are at the top (pandas, matplotlib, pyhdf, etc.)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                             QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from display_charts import NIST_DATA_ANALYZER, TPU_DATA_ANALYZER

class AeroDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.analyzer = None      
        self.file_type = None      

        self.setWindowTitle("Adaptive Wind Data Viewer")
        self.setFixedSize(500, 420)  # Slightly bumped height to fit layouts comfortably

        # --- 1. Universal Header Elements ---
        self.status_label = QLabel("Status: Waiting for Data File (.HDF / .mat)...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_load = QPushButton("📁 Load Data File")
        self.btn_load.clicked.connect(self.load_file)

        # =====================================================================
        # 2. NIST CONTAINER SETUP
        # =====================================================================
        self.nist_frame = QFrame()
        nist_layout = QVBoxLayout(self.nist_frame)
        
        self.lbl_nist = QLabel("<b>NIST 3D Analysis Options:</b>")
        nist_layout.addWidget(self.lbl_nist)

        # Standalone Buttons
        self.btn_plot_3d_nist = QPushButton("2. Render 3D Wireframe")
        self.btn_plot_2d_nist = QPushButton("3. Render 2D Surface Map")
        nist_layout.addWidget(self.btn_plot_3d_nist)
        nist_layout.addWidget(self.btn_plot_2d_nist)

        # Dynamic Contour Section (NIST Face Selection)
        nist_contour_layout = QHBoxLayout()
        self.lbl_face_nist = QLabel("Select Face:")
        self.combo_face_nist = QComboBox()
        self.btn_plot_contour_nist = QPushButton("4. Render Mean Contour Map")
        
        nist_contour_layout.addWidget(self.lbl_face_nist)
        nist_contour_layout.addWidget(self.combo_face_nist)
        nist_contour_layout.addWidget(self.btn_plot_contour_nist, stretch=1)
        nist_layout.addLayout(nist_contour_layout)

        # Dynamic Timeseries Section (NIST Tap Selection)
        nist_timeseries_layout = QHBoxLayout()
        self.lbl_tap_nist = QLabel("Select Tap:")
        self.combo_tap_nist = QComboBox()
        self.btn_plot_timeseries_nist = QPushButton("5. Render Timeseries Chart")
        
        nist_timeseries_layout.addWidget(self.lbl_tap_nist)
        nist_timeseries_layout.addWidget(self.combo_tap_nist)
        nist_timeseries_layout.addWidget(self.btn_plot_timeseries_nist, stretch=1)
        nist_layout.addLayout(nist_timeseries_layout)

        # NIST Signal Connections (Slots)
        self.btn_plot_3d_nist.clicked.connect(self.trigger_3d_plot)
        self.btn_plot_2d_nist.clicked.connect(self.trigger_2d_plot)
        self.btn_plot_contour_nist.clicked.connect(self.trigger_mean_contour)
        self.btn_plot_timeseries_nist.clicked.connect(self.trigger_full_timeseries)

        # =====================================================================
        # 3. TPU CONTAINER SETUP
        # =====================================================================
        self.tpu_frame = QFrame()
        tpu_layout = QVBoxLayout(self.tpu_frame)
        
        self.lbl_tpu = QLabel("<b>TPU Analysis Options:</b>")
        tpu_layout.addWidget(self.lbl_tpu)

        # 1️⃣ Button 1: Channel Positions (Standalone)
        self.btn_plot_grid = QPushButton("1. Render Channel Positions Grid")
        self.btn_plot_grid.clicked.connect(self.trigger_grid_plot)
        tpu_layout.addWidget(self.btn_plot_grid)
        
        # 2️⃣ Button 2: Timeseries at a Tap (With Dropdown Input)
        tpu_timeseries_layout = QHBoxLayout()
        self.lbl_tap_tpu = QLabel("Select Tap:")
        self.combo_tap_tpu = QComboBox()
        self.btn_tpu_timeseries = QPushButton("2. Render Timeseries Chart")
        self.btn_tpu_timeseries.clicked.connect(self.trigger_tpu_timeseries)
        
        tpu_timeseries_layout.addWidget(self.lbl_tap_tpu)
        tpu_timeseries_layout.addWidget(self.combo_tap_tpu)
        tpu_timeseries_layout.addWidget(self.btn_tpu_timeseries, stretch=1)
        tpu_layout.addLayout(tpu_timeseries_layout)

        # 3️⃣ Button 3: Global Mean Pressure Contour (Standalone - No Dropdown)
        self.btn_tpu_contour = QPushButton("3. Render Global Mean Contour Map")
        self.btn_tpu_contour.clicked.connect(self.trigger_tpu_contour)
        tpu_layout.addWidget(self.btn_tpu_contour)

        # =====================================================================
        # 4. MASTER LAYOUT ASSEMBLY
        # =====================================================================
        self.master_layout = QVBoxLayout()
        self.master_layout.addWidget(self.status_label)
        self.master_layout.addWidget(self.btn_load)
        
        # Stack both adaptive frames
        self.master_layout.addWidget(self.nist_frame)
        self.master_layout.addWidget(self.tpu_frame)
        self.master_layout.addStretch()

        # Hide BOTH panels on startup until file extension is evaluated
        self.nist_frame.hide()
        self.tpu_frame.hide()

        container = QWidget()
        container.setLayout(self.master_layout)
        self.setCentralWidget(container)


    def disable_controls(self):
        """Helper to lock interface on startup."""
        for widget in [self.btn_plot_3d, self.btn_plot_2d, self.combo_face, 
                       self.btn_plot_contour, self.combo_tap, self.btn_plot_timeseries]:
            widget.setEnabled(False)

    # --- 6. Backend Logic ---
    def load_file(self):
        # 1. Allow selection of both .mat and .hdf file formats in the dialog browser
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Wind Database File", 
            "", 
            "Supported Files (*.hdf *.hdf4 *.h4 *.mat);;All Files (*)"
        )

        if file_path:
            import os
            clean_path = os.path.normpath(file_path)
            ext = os.path.splitext(clean_path)[1].lower()
            
            self.status_label.setText("Status: Extracting Data Matrix...")
            self.status_label.setStyleSheet("color: black; font-weight: normal;")
            
            # Reset UI: Hide both frames until the file is successfully parsed
            self.nist_frame.hide()
            self.tpu_frame.hide()
            QApplication.processEvents() 
            
            try:
                # =============================================================
                # 🟢 PIPELINE A: HANDLING TPU MATLAB FILES
                # =============================================================
                if ext == '.mat':
                    self.file_type = 'TPU'
                    from display_charts import TPU_DATA_ANALYZER
                    
                    self.analyzer = TPU_DATA_ANALYZER(clean_path)
                    loc_df = self.analyzer.get_loc_df()
                    pressure_df = self.analyzer.get_timeseries_df()
                    
                    # Populate Tap Dropdown for TPU Timeseries (using column length)
                    total_taps = pressure_df.shape[1]
                    self.combo_tap_tpu.clear()
                    self.combo_tap_tpu.addItems([str(i) for i in range(total_taps)])
                    
                    # Update status text and display the clean 3-button TPU layout
                    self.status_label.setText("Mode: TPU MATLAB Engine Active ✅")
                    self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                    self.tpu_frame.show()

                # =============================================================
                # 🔵 PIPELINE B: HANDLING NIST HDF FILES
                # =============================================================
                elif ext in ['.hdf', '.hdf4', '.h4']:
                    self.file_type = 'NIST'
                    from display_charts import NIST_DATA_ANALYZER
                    
                    self.analyzer = NIST_DATA_ANALYZER(clean_path)
                    self.dataframes = self.analyzer.extract_dataframes()
                    
                    if self.dataframes:
                        # 1. Populate Faces from the Flat Tap Coordinates array
                        flat_tap_df = self.dataframes['Flat_Tap_Coordinates'].T
                        unique_faces = sorted(flat_tap_df[1].dropna().unique())
                        self.combo_face_nist.clear()
                        self.combo_face_nist.addItems([str(int(f)) for f in unique_faces])

                        # 2. Populate Taps from the Time Series columns length
                        time_series_df = self.dataframes['Time_Series'].T
                        total_taps = time_series_df.shape[1]
                        self.combo_tap_nist.clear()
                        self.combo_tap_nist.addItems([str(i) for i in range(total_taps)])
                        
                        # Update status text and display the NIST analysis options layout
                        self.status_label.setText("Mode: NIST HDF4 Engine Active ✅")
                        self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                        self.nist_frame.show()
                    else:
                        self.status_label.setText("Status: NIST Extraction Failed ❌")
                        return

                # Cleanly snap window boundaries around whichever panel became visible
                self.adjustSize()

            except Exception as e:
                self.status_label.setText("Status: Error reading file ❌")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                print(f"UI Loading Error: {e}")

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
            selected_face = float(self.combo_face_nist.currentText())
            
            flat_tap_coords = self.dataframes['Flat_Tap_Coordinates'].T
            pressure_df = self.dataframes['Time_Series'].T
            
            # Fire your class function using the live input
            self.analyzer.get_mean_contour(selected_face, flat_tap_coords, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render contour map: {e}")

    def trigger_full_timeseries(self):
        try:
            # Grab the selected tap channel number from the dropdown box!
            selected_tap = int(self.combo_tap_nist.currentText())
            
            pressure_time_series = self.dataframes['Time_Series'].T
            
            # Fire your class function to show the history graph
            self.analyzer.view_full_series_tap(pressure_time_series, selected_tap)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render timeseries: {e}")


    #channel postiions
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
        try:
            pressure_df = self.analyzer.get_timeseries_df()
            loc_df = self.analyzer.get_loc_df()

            self.analyzer.mean_cp_contour(pressure_df, loc_df)

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render plot: {e}")
       

# --- Execution Loop ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = AeroDataViewer()
    window.show()
    sys.exit(app.exec())





            


    