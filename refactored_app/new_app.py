import os
import sys
import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                             QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from TPU_plotters import TPU_HIGH_RISE_ANALYZER
from NIST_plotter import NIST_DATA_ANALYZER

# Cloud connection credentials
COCKROACH_DB_URL = "postgresql://Samuel:S2E8zH83RkfwAG5THekg4w@windeee-cluster-27677.j77.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=require"

class AeroDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.analyzer = None      
        self.file_type = None      
        self.dataframes = {}  
        self.master_df = None 
        self.db_metadata_records = [] 

        self.setWindowTitle("Adaptive Wind Data Viewer")
        self.setFixedSize(750, 600)

        # --- 1. Universal Header Elements ---
        self.status_label = QLabel("Status: Initializing and connecting to Cloud DB...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #005A9C;")
        
        # --- 2. Cloud Database Explorer Panel ---
        self.db_frame = QFrame()
        self.db_frame.setFrameShape(QFrame.Shape.StyledPanel)
        db_layout = QVBoxLayout(self.db_frame)
        
        db_title = QLabel("<b>☁️ CockroachDB Parameter Filter:</b>")
        db_layout.addWidget(db_title)
        
        filter_layout = QHBoxLayout()
        self.combo_slope = QComboBox()
        self.combo_exposure = QComboBox()
        self.combo_scale = QComboBox()
        self.combo_leakage = QComboBox()
        self.combo_eave = QComboBox()
        self.combo_angle = QComboBox()
        
        filter_layout.addWidget(QLabel("Slope:"))
        filter_layout.addWidget(self.combo_slope)
        filter_layout.addWidget(QLabel("Exposure:"))
        filter_layout.addWidget(self.combo_exposure)
        filter_layout.addWidget(QLabel("Scale:"))
        filter_layout.addWidget(self.combo_scale)
        filter_layout.addWidget(QLabel("Leakage:"))
        filter_layout.addWidget(self.combo_leakage)
        filter_layout.addWidget(QLabel("Eave Ht:"))
        filter_layout.addWidget(self.combo_eave)
        filter_layout.addWidget(QLabel("Angle:"))
        filter_layout.addWidget(self.combo_angle)
        db_layout.addLayout(filter_layout)
        
        match_layout = QHBoxLayout()
        self.combo_matched_models = QComboBox()
        self.btn_load_cloud_df = QPushButton("🚀 Compile Master DataFrame")
        self.btn_load_cloud_df.setEnabled(False)
        
        match_layout.addWidget(QLabel("Matched Model ID:"))
        match_layout.addWidget(self.combo_matched_models, stretch=2)
        match_layout.addWidget(self.btn_load_cloud_df, stretch=1)
        db_layout.addLayout(match_layout)

        self.btn_load = QPushButton("📁 Alternative: Load Local Data File (.HDF / .mat)")
        self.btn_load.clicked.connect(self.load_file)

        # --- 3. NIST CONTAINER SETUP ---
        self.nist_frame = QFrame()
        nist_layout = QVBoxLayout(self.nist_frame)
        self.lbl_nist = QLabel("<b>NIST 3D Analysis Options:</b>")
        nist_layout.addWidget(self.lbl_nist)

        self.btn_plot_3d_nist = QPushButton("2. Render 3D Wireframe")
        self.btn_plot_2d_nist = QPushButton("3. Render 2D Surface Map")
        nist_layout.addWidget(self.btn_plot_3d_nist)
        nist_layout.addWidget(self.btn_plot_2d_nist)

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

        nist_timeseries_layout = QHBoxLayout()
        self.lbl_tap_nist = QLabel("Select Tap:")
        self.combo_tap_nist = QComboBox()
        self.btn_plot_timeseries_nist = QPushButton("5. Render Timeseries Chart")
        
        nist_timeseries_layout.addWidget(self.lbl_tap_nist)
        nist_timeseries_layout.addWidget(self.combo_tap_nist)
        nist_timeseries_layout.addWidget(self.btn_plot_timeseries_nist, stretch=1)
        nist_layout.addLayout(nist_timeseries_layout)

        self.btn_plot_3d_nist.clicked.connect(self.trigger_3d_plot)
        self.btn_plot_2d_nist.clicked.connect(self.trigger_2d_plot)
        self.btn_plot_contour_nist.clicked.connect(self.trigger_mean_contour)
        self.btn_plot_std_contour_nist.clicked.connect(self.trigger_nist_std_contour)
        self.btn_plot_timeseries_nist.clicked.connect(self.trigger_full_timeseries)

        # --- 4. TPU CONTAINER SETUP ---
        self.tpu_frame = QFrame()
        tpu_layout = QVBoxLayout(self.tpu_frame)
        self.lbl_tpu = QLabel("<b>TPU Analysis Options:</b>")
        tpu_layout.addWidget(self.lbl_tpu)

        self.btn_plot_grid = QPushButton("1. Render Channel Positions Grid")
        self.btn_plot_grid.clicked.connect(self.trigger_grid_plot)
        tpu_layout.addWidget(self.btn_plot_grid)
        
        tpu_timeseries_layout = QHBoxLayout()
        self.lbl_tap_tpu = QLabel("Select Tap:")
        self.combo_tap_tpu = QComboBox()
        self.btn_tpu_timeseries = QPushButton("2. Render Timeseries Chart")
        self.btn_tpu_timeseries.clicked.connect(self.trigger_tpu_timeseries)
        
        tpu_timeseries_layout.addWidget(self.lbl_tap_tpu)
        tpu_timeseries_layout.addWidget(self.combo_tap_tpu)
        tpu_timeseries_layout.addWidget(self.btn_tpu_timeseries, stretch=1)
        tpu_layout.addLayout(tpu_timeseries_layout)

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

        # --- 5. MASTER LAYOUT ASSEMBLY ---
        self.master_layout = QVBoxLayout()
        self.master_layout.addWidget(self.status_label)
        self.master_layout.addWidget(self.db_frame)
        self.master_layout.addWidget(self.btn_load)
        self.master_layout.addWidget(self.nist_frame)
        self.master_layout.addWidget(self.tpu_frame)
        self.master_layout.addStretch()

        self.nist_frame.hide()
        self.tpu_frame.hide()

        container = QWidget()
        container.setLayout(self.master_layout)
        self.setCentralWidget(container)

        self.combo_slope.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.combo_exposure.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.combo_scale.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.combo_leakage.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.combo_eave.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.combo_angle.currentIndexChanged.connect(self.evaluate_filter_cascade)
        self.btn_load_cloud_df.clicked.connect(self.fetch_selected_master_matrix)

        self.initialize_database_pipeline()

    # =====================================================================
    # NATIVE CONTOUR INTERPOLATION PLOTTER
    # =====================================================================
    def get_mean_contour(self, face_no, flat_tap_coords, pressure_series):
        """Generates a mean pressure coefficient contour map natively from the UI state."""
        is_cloud_data = False
        if isinstance(flat_tap_coords, pd.DataFrame):
            if 'x_coordinate' in flat_tap_coords.columns or 'x_coordinate' in flat_tap_coords.index:
                is_cloud_data = True

        if is_cloud_data:
            # Drop geometry duplicate nodes across intersecting channels
            clean_df = flat_tap_coords.drop_duplicates(subset=['x_coordinate', 'y_coordinate'])
            
            # Filter rows dynamically using the in-memory parsed face_no integer column
            face_df = clean_df[clean_df['face_no'] == int(face_no)]
            
            x = pd.to_numeric(face_df['x_coordinate']).values
            y = pd.to_numeric(face_df['y_coordinate']).values
            z = pd.to_numeric(face_df['mean_cp']).values
        else:
            # Legacy local file structural pathway
            mean_cp_series = pressure_series.mean(axis=0)
            pressure_df = mean_cp_series.reset_index()
            pressure_df.columns = ['Tap no.', 'mean_cp']
            
            all_means = pressure_df['mean_cp'].values
            num_taps = len(flat_tap_coords)
            matched_means = all_means[:num_taps]
            
            flat_tap_coords = flat_tap_coords.copy()
            flat_tap_coords['mean_cp'] = matched_means        
            
            unique_faces = flat_tap_coords[1].dropna().unique()
            face_dfs = {face_num: flat_tap_coords[flat_tap_coords[1] == face_num].copy() for face_num in unique_faces}
            
            face_df = face_dfs.get(face_no, list(face_dfs.values())[0])
            clean_df = face_df.drop_duplicates(subset=[2, 3])
            
            x = clean_df[2].values
            y = clean_df[3].values
            z = clean_df['mean_cp'].values

        # Protect against empty slices
        if len(x) == 0:
            QMessageBox.warning(self, "Plot Alert", f"No valid coordinates found to plot for Face {face_no}.")
            return

        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label="Mean $C_p$")
        plt.scatter(x, y, c='black', s=15, marker='x', label='Taps')

        plt.title(f"Mean Pressure Coefficient ($C_p$) - Face {int(face_no)}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        current_x_min, current_x_max = plt.xlim()
        current_y_min, current_y_max = plt.ylim()
        buffer = 2.5 
        plt.xlim(current_x_min - buffer, current_x_max + buffer)
        plt.ylim(current_y_min - buffer, current_y_max + buffer)
        plt.show()

    # =====================================================================
    # CLOUD DATABASE ROUTINES
    # =====================================================================
    def initialize_database_pipeline(self):
        try:
            connection = psycopg2.connect(COCKROACH_DB_URL)
            query = 'SELECT id::text, roof_slope, exposure_val, model_scale, leakage, eave_height, angle, data_origin FROM "Origin_Table";'
            df = pd.read_sql_query(query, connection)
            connection.close()
            
            if df.empty:
                self.status_label.setText("Status: Cloud Database is completely empty! ⚠️")
                return
                
            self.db_metadata_records = df
            
            slopes = [str(int(s)) for s in sorted(df['roof_slope'].unique())]
            exposures = sorted(df['exposure_val'].unique())
            scales = [str(int(sc)) for sc in sorted(df['model_scale'].dropna().unique())]
            leakages = sorted(df['leakage'].dropna().unique())
            eaves = [str(int(e)) for e in sorted(df['eave_height'].dropna().unique())]
            angles = [str(int(a)) for a in sorted(df['angle'].unique())]
            
            self.combo_slope.addItems(slopes)
            self.combo_exposure.addItems(exposures)
            self.combo_scale.addItems(scales)
            self.combo_leakage.addItems(leakages)
            self.combo_eave.addItems(eaves)
            self.combo_angle.addItems(angles)
            
            self.status_label.setText("Mode: CockroachDB Connection Synchronized Active ✅")
            self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
            self.btn_load_cloud_df.setEnabled(True)
            
            self.evaluate_filter_cascade()
            
        except Exception as e:
            self.status_label.setText("Status: Cloud DB Connection Failed ❌")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Database Handshake Failure: {e}")

    def evaluate_filter_cascade(self):
        if len(self.db_metadata_records) == 0:
            return
            
        target_slope = int(self.combo_slope.currentText() or 0)
        target_exposure = self.combo_exposure.currentText()
        target_scale = int(self.combo_scale.currentText() or 0)
        target_leakage = self.combo_leakage.currentText()
        target_eave = int(self.combo_eave.currentText() or 0)
        target_angle = float(self.combo_angle.currentText() or 0.0)
        
        df = self.db_metadata_records
        mask = (df['roof_slope'] == target_slope) & \
               (df['exposure_val'] == target_exposure) & \
               (df['model_scale'] == target_scale) & \
               (df['leakage'] == target_leakage) & \
               (df['eave_height'] == target_eave) & \
               (df['angle'] == target_angle)
               
        matched_ids = df[mask]['id'].tolist()
        
        self.combo_matched_models.clear()
        if matched_ids:
            self.combo_matched_models.addItems(matched_ids)
            self.btn_load_cloud_df.setEnabled(True)
            self.status_label.setText(f"Status: Found {len(matched_ids)} matching configuration runs.")
            self.status_label.setStyleSheet("color: #2E8B57; font-weight: normal;")
        else:
            self.combo_matched_models.addItem("No match found")
            self.btn_load_cloud_df.setEnabled(False)
            self.status_label.setText("Status: No matching Model ID matches this unique property row combination.")
            self.status_label.setStyleSheet("color: #D32F2F; font-weight: normal;")

    def fetch_selected_master_matrix(self):
        selected_id = self.combo_matched_models.currentText()
        if selected_id == "No match found" or not selected_id:
            return
            
        self.status_label.setText("Status: Compiling Heavy Matrix Arrays from Cloud...")
        self.status_label.setStyleSheet("color: #005A9C; font-weight: bold;")
        QApplication.processEvents()
        
        try:
            connection = psycopg2.connect(COCKROACH_DB_URL)
            master_query = """
                SELECT 
                    t.id AS tap_id, t.tap_no, t.x AS x_coordinate, t.y AS y_coordinate,
                    p.mean_cp, p.stddev_cp, p.min_cp, p.max_cp, p.skew_cp, p.kurtosis_cp, p.fft_magnitude
                FROM taps t
                INNER JOIN pressure_series p ON t.id = p.tap_id
                WHERE t.model_id = %s
                ORDER BY t.tap_no ASC;
            """
            raw_df = pd.read_sql_query(master_query, connection, params=(int(selected_id),))
            connection.close()
            
            if raw_df.empty:
                QMessageBox.warning(self, "Empty Matrix", "This configuration contains meta definitions but pressure_series values are unpopulated.")
                self.status_label.setText("Status: Incomplete Row Intersection ⚠️")
                return
                
            # 💡 SOLUTIONS LAYER: Extract the UWO/NIST face integer index dynamically from the string layout
            # e.g., '3112' -> index position 1 is '1' -> Face 1
            raw_df['face_no'] = raw_df['tap_no'].astype(str).str.strip().str[1].astype(int)
            self.master_df = raw_df

            QMessageBox.information(self, "Success", f"Compiled complete Master DataFrame!\nShape: {self.master_df.shape}\nSuccessfully mapped {len(self.master_df)} distinct spatial features.")
            self.status_label.setText(f"Active Model Run Matrix: ID {selected_id} Loaded ✅")
            self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
            
            df_meta = self.db_metadata_records
            origin_type = df_meta[df_meta['id'] == selected_id]['data_origin'].values[0]
            
            if origin_type == "NIST":
                self.tpu_frame.hide()
                self.file_type = 'NIST'
                
                # Automatically populate dropdown using the in-memory dynamically generated face values
                unique_faces = [str(int(f)) for f in sorted(self.master_df['face_no'].dropna().unique())]
                self.combo_face_nist.clear()
                self.combo_face_nist.addItems(unique_faces)
                
                self.combo_tap_nist.clear()
                self.combo_tap_nist.addItems([str(int(t)) for t in self.master_df['tap_no'].tolist()])
                self.nist_frame.show()
            else:
                self.nist_frame.hide()
                self.file_type = 'TPU'
                self.combo_tap_tpu.clear()
                self.combo_tap_tpu.addItems([str(int(t)) for t in self.master_df['tap_no'].tolist()])
                self.tpu_frame.show()
                
            self.adjustSize()
            
        except Exception as e:
            QMessageBox.critical(self, "Query Crash", f"Failed compilation execution protocol: {e}")
            self.status_label.setText("Status: Execution Pipeline Broken ❌")

    # =====================================================================
    # LOCAL FILE HANDLING (LEGACY FALLBACK)
    # =====================================================================
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Wind Database File", "", 
            "Supported Files (*.hdf *.hdf4 *.h4 *.mat);;All Files (*)"
        )

        if file_path:
            clean_path = os.path.normpath(file_path)
            ext = os.path.splitext(clean_path)[1].lower()
            
            self.status_label.setText("Status: Extracting Data Matrix...")
            self.status_label.setStyleSheet("color: black; font-weight: normal;")
            self.nist_frame.hide()
            self.tpu_frame.hide()
            self.master_df = None 
            QApplication.processEvents() 
            
            try:
                if ext == '.mat':
                    self.file_type = 'TPU'
                    self.analyzer = TPU_HIGH_RISE_ANALYZER(clean_path)
                    loc_df = self.analyzer.get_loc_df()
                    pressure_df = self.analyzer.get_timeseries_df()
                    
                    total_taps = pressure_df.shape[1]
                    self.combo_tap_tpu.clear()
                    self.combo_tap_tpu.addItems([str(i) for i in range(total_taps)])
                    
                    unique_faces = sorted(loc_df['Face_No'].dropna().unique())
                    self.combo_face_tpu.clear()
                    self.combo_face_tpu.addItems([str(int(f)) for f in unique_faces])
                    
                    self.status_label.setText("Mode: TPU MATLAB Engine Active ✅")
                    self.status_label.setStyleSheet("color: #2E8B57; font-weight: bold;")
                    self.tpu_frame.show()

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
    # EVENT TRIGGER MAP COUPLINGS
    # =====================================================================
    def trigger_mean_contour(self):
        try:
            selected_face = float(self.combo_face_nist.currentText())
            if self.master_df is not None:
                # Pass master_df straight to our native class plotter
                self.get_mean_contour(selected_face, self.master_df, self.master_df)
            else:
                # Fallback legacy path
                flat_tap_coords = self.dataframes['Flat_Tap_Coordinates'].T
                pressure_df = self.dataframes['Time_Series'].T
                self.get_mean_contour(selected_face, flat_tap_coords, pressure_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to render contour map: {e}")

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = AeroDataViewer()
    window.show()
    sys.exit(app.exec())