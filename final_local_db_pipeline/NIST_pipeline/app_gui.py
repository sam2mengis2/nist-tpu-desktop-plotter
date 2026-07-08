# -*- coding: utf-8 -*-
"""
NIST_pipeline/app_gui.py - Dedicated NIST Aerodynamic User Interface Frame
Cleanly refactored with background thread pools and full tap time-history plotting.
"""

import sys
import os
import sqlite3
import numpy as np
import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QPushButton, QTextEdit, QGroupBox, QSplitter, QFrame,
    QFileDialog
)
from PyQt6.QtCore import Qt, QRunnable, QObject, pyqtSignal, QThreadPool

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Modular Sibling Inclusions
from NIST_scraper import analyze_nist_architecture, get_cached_links_for_leaf
from NIST_parser import (
    populate_database_from_archive, 
    scan_archive_pure_memory, 
    initialize_local_database, 
    clear_session_data, 
    DB_PATH,
    DROP_FOLDER
)


class WorkerSignals(QObject):
    """Defines communication channels between the background thread and the UI."""
    progress_signal = pyqtSignal(str)
    success_signal = pyqtSignal(dict)
    failure_signal = pyqtSignal(str)


class ArchiveScannerRunnable(QRunnable):
    """BACKGROUND WORKER: Executes file streaming and memory scanning tasks on a separate thread pool."""
    def __init__(self, session, download_url, save_path):
        super().__init__()
        self.session = session
        self.download_url = download_url
        self.save_path = save_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.progress_signal.emit("📥 Streaming master package archive directly from server...")
            with self.session.get(self.download_url, stream=True, timeout=45, verify=False) as r:
                r.raise_for_status()
                with open(self.save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
            
            self.signals.progress_signal.emit("⚡ Running in-memory scan pass over nested zip index layers...")
            extracted_map = scan_archive_pure_memory(self.save_path)
            self.signals.success_signal.emit(extracted_map)
        except Exception as err:
            self.signals.failure_signal.emit(str(err))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class NISTDesktopWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIST Aerodynamic Database Engineering Workbench")
        self.setMinimumSize(1200, 750)
        
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.current_url = "https://www.nist.gov/el/mssd/nist-aerodynamic-database/university-western-ontario-data-sets/data-sets-test-number"
        self.resolved_page_url = ""
        self.file_links_map = {}
        self.active_model_id = None
        self.active_wind_angle = None
        self.extracted_hdf_map = {}
        
        self.threadpool = QThreadPool.globalInstance()
        
        initialize_local_database()
        self.init_ui_layout()
        self.log_message("🚀 NIST Workbench Initialized. Ready to process index matrix.")

    def init_ui_layout(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        ingest_group = QGroupBox("NIST Configurator Layer")
        ingest_layout = QVBoxLayout(ingest_group)
        
        self.btn_connect = QPushButton("Connect & Process Index Matrix")
        self.btn_connect.clicked.connect(self.handle_portal_connection)
        ingest_layout.addWidget(self.btn_connect)
        
        ingest_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        ingest_layout.addWidget(QLabel("Dynamic Matrix Level Selections:"))
        self.options_combo = QComboBox()
        ingest_layout.addWidget(self.options_combo)
        
        self.btn_next_level = QPushButton("Drill Down Next Layout Layer")
        self.btn_next_level.clicked.connect(self.handle_drill_down)
        self.btn_next_level.setEnabled(False)
        ingest_layout.addWidget(self.btn_next_level)
        
        ingest_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        ingest_layout.addWidget(QLabel("Select Target File Code Matrix:"))
        self.angle_combo = QComboBox()
        ingest_layout.addWidget(self.angle_combo)
        
        self.btn_scan_zip = QPushButton("Download & Scan Zip Archive Package")
        self.btn_scan_zip.setStyleSheet("font-weight: bold; background-color: #e67e22; color: white;")
        self.btn_scan_zip.clicked.connect(self.handle_zip_download_and_scan)
        self.btn_scan_zip.setEnabled(False)
        ingest_layout.addWidget(self.btn_scan_zip)
        
        ingest_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        ingest_layout.addWidget(QLabel("Available Wind Angles in Archive:"))
        self.zip_angle_combo = QComboBox()
        ingest_layout.addWidget(self.zip_angle_combo)
        
        self.btn_ingest = QPushButton("Ingest Selected Wind Angle to Cache")
        self.btn_ingest.setStyleSheet("font-weight: bold; background-color: #2a75d3; color: white;")
        self.btn_ingest.clicked.connect(self.handle_selected_hdf_ingestion)
        self.btn_ingest.setEnabled(False)
        ingest_layout.addWidget(self.btn_ingest)
        left_layout.addWidget(ingest_group)

        dashboard_group = QGroupBox("Dashboard Capabilities")
        dashboard_layout = QVBoxLayout(dashboard_group)
        
        self.btn_export_all_time = QPushButton("Export FULL Time-Series Grid to CSV")
        self.btn_export_all_time.clicked.connect(self.export_full_time_series_csv)
        self.btn_export_all_time.setEnabled(False)
        dashboard_layout.addWidget(self.btn_export_all_time)
        
        self.btn_export_summary = QPushButton("Export Spatial Metrics & Faces to CSV")
        self.btn_export_summary.clicked.connect(self.export_spatial_summary_csv)
        self.btn_export_summary.setEnabled(False)
        dashboard_layout.addWidget(self.btn_export_summary)
        
        dashboard_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        dashboard_layout.addWidget(QLabel("Select Target Metric to Visualize:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("Mean Pressure Coefficient (Mean Cp)", "mean_cp")
        self.metric_combo.addItem("Standard Deviation (Std Dev Cp)", "std_cp")
        dashboard_layout.addWidget(self.metric_combo)
        
        dashboard_layout.addWidget(QLabel("Select Target Face to Visualize:"))
        self.face_combo = QComboBox()
        self.face_combo.addItem("All Faces Combined", "all")
        dashboard_layout.addWidget(self.face_combo)
        
        self.btn_plot_contour = QPushButton("Render Spatial Surface Contour Plot")
        self.btn_plot_contour.setStyleSheet("font-weight: bold; background-color: #2aa25b; color: white;")
        self.btn_plot_contour.clicked.connect(self.render_spatial_contour_map)
        self.btn_plot_contour.setEnabled(False)
        dashboard_layout.addWidget(self.btn_plot_contour)
        
        dashboard_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        
        # 🎯 NEW FEATURE: Interactive Tap Selection Dropdown
        dashboard_layout.addWidget(QLabel("Select Target Tap for Time History Plot:"))
        self.tap_combo = QComboBox()
        dashboard_layout.addWidget(self.tap_combo)
        
        # 🎯 NEW FEATURE: Time History Generation Button
        self.btn_plot_history = QPushButton("Render Tap Cp Time History Plot")
        self.btn_plot_history.setStyleSheet("font-weight: bold; background-color: #8e44ad; color: white;")
        self.btn_plot_history.clicked.connect(self.render_tap_time_history_plot)
        self.btn_plot_history.setEnabled(False)
        dashboard_layout.addWidget(self.btn_plot_history)
        
        left_layout.addWidget(dashboard_group)

        log_group = QGroupBox("Activity Diagnostics Console")
        log_layout = QVBoxLayout(log_group)
        self.console_out = QTextEdit()
        self.console_out.setReadOnly(True)
        self.console_out.setStyleSheet("background-color: #1e1e1e; color: #a9dc76; font-family: Consolas;")
        log_layout.addWidget(self.console_out)
        left_layout.addWidget(log_group)
        splitter.addWidget(left_panel)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        plot_group = QGroupBox("Integrated Visualization Display Array")
        plot_box_layout = QVBoxLayout(plot_group)
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_box_layout.addWidget(self.toolbar)
        plot_box_layout.addWidget(self.canvas)
        right_layout.addWidget(plot_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 750])

    def log_message(self, message): self.console_out.append(message)
        
    def handle_portal_connection(self):
        self.log_message(f"\n🌐 Contacting NIST Database Repository...")
        self.populate_dropdown_combobox()

    def populate_dropdown_combobox(self):
        self.options_combo.clear()
        self.btn_next_level.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            options_map, resolved_url, is_final_page = analyze_nist_architecture(self.session, self.current_url, self.headers)
            err_msg = None
        except Exception as e:
            options_map, resolved_url, is_final_page = {}, self.current_url, False
            err_msg = f"❌ Ingestion Exception: {e}"
        QApplication.restoreOverrideCursor()
        
        if err_msg:
            self.log_message(err_msg)
            return
        self.resolved_page_url = resolved_url
        
        if is_final_page:
            self.log_message("📊 Status: Configuration targeted successfully. Parsing downloadable nodes...")
            self.parse_final_links_grid()
            return

        if options_map:
            self.log_message(f"📋 Loaded {len(options_map)} options into layout step.")
            for text, target_link in sorted(options_map.items()): self.options_combo.addItem(text, target_link)
            self.btn_next_level.setEnabled(True)

    def handle_drill_down(self):
        selected_text = self.options_combo.currentText()
        next_target_url = self.options_combo.currentData()
        if next_target_url:
            self.log_message(f"🔍 Drilling into: '{selected_text}'")
            self.current_url = next_target_url
            self.populate_dropdown_combobox()

    def parse_final_links_grid(self):
        self.angle_combo.clear()
        self.btn_scan_zip.setEnabled(False)
        self.file_links_map = {}
        
        target_links = get_cached_links_for_leaf(self.resolved_page_url)
        for idx, (label, link) in enumerate(target_links):
            self.angle_combo.addItem(f"File Code Matrix: {label}", idx)
            self.file_links_map[idx] = link
            
        if target_links:
            self.log_message(f"🎯 Success! Located {len(target_links)} code targets.")
            self.btn_scan_zip.setEnabled(True)

    def handle_zip_download_and_scan(self):
        self.zip_angle_combo.clear()
        self.btn_ingest.setEnabled(False)
        self.extracted_hdf_map = {}
        
        target_key = self.angle_combo.currentData()
        download_url = self.file_links_map[target_key]
        file_name = download_url.split('/')[-1]
        
        master_zip_path = os.path.join(DROP_FOLDER, file_name)
        os.makedirs(DROP_FOLDER, exist_ok=True)
        
        self.btn_scan_zip.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        runnable = ArchiveScannerRunnable(self.session, download_url, master_zip_path)
        runnable.signals.progress_signal.connect(self.log_message)
        runnable.signals.success_signal.connect(self.handle_background_scan_success)
        runnable.signals.failure_signal.connect(self.handle_background_scan_failure)
        
        self.threadpool.start(runnable)

    def handle_background_scan_success(self, extracted_map):
        self.extracted_hdf_map = extracted_map
        QApplication.restoreOverrideCursor()
        self.btn_scan_zip.setEnabled(True)
        
        if self.extracted_hdf_map:
            for encoded_angle in sorted(self.extracted_hdf_map.keys()):
                display_angle = float(encoded_angle / 10.0)
                self.zip_angle_combo.addItem(f"Wind Orientation Angle: {display_angle}°", encoded_angle)
            self.log_message(f"🎉 Scan Complete! Populated Drop-Down menu with {len(self.extracted_hdf_map)} wind orientations.")
            self.btn_ingest.setEnabled(True)
        else:
            self.log_message("❌ Ingestion Error: Parser backend failed to isolate matching entries inside zip layers.")

    def handle_background_scan_failure(self, error_message):
        QApplication.restoreOverrideCursor()
        self.log_message(f"❌ Thread Ingestion Error: {error_message}")
        self.btn_scan_zip.setEnabled(True)

    def handle_selected_hdf_ingestion(self):
        """🎯 STAGE 2: Extracts the single selected HDF target and saves it to the unified database."""
        self.btn_export_all_time.setEnabled(False)
        self.btn_export_summary.setEnabled(False)
        self.btn_plot_contour.setEnabled(False)
        self.btn_plot_history.setEnabled(False)
        
        chosen_encoded_angle = self.zip_angle_combo.currentData()
        target_info = self.extracted_hdf_map.get(chosen_encoded_angle)
        
        if not target_info:
            self.log_message("❌ Ingestion Fault: Selected target data block is missing from temporary workspace cache.")
            return
            
        self.log_message(f"⚡ Unboxing datasets for wind angle: {float(chosen_encoded_angle / 10.0)}°...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # 🎯 FIXED: Direct, clean call to the background unboxer function
            model_id, wind_angle = populate_database_from_archive(
                target_info[0],       # master zip file path
                target_info[1],       # internal target .HDF path name
                chosen_encoded_angle  # raw wind angle code key
            )

            if model_id is not None:
                self.active_model_id, self.active_wind_angle = model_id, wind_angle
                self.log_message("🎉 Ingestion complete! Cache is synchronized.")
                
                # Dynamic Face Dropdown Update
                self.face_combo.clear()
                self.face_combo.addItem("All Faces Combined", "all")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT face FROM tap_measurements WHERE model_id=? AND wind_angle=? AND face IS NOT NULL ORDER BY face", (self.active_model_id, self.active_wind_angle))
                for face_num in [row[0] for row in cursor.fetchall()]: 
                    self.face_combo.addItem(f"Face {face_num} Layout", face_num)
                
                # Query and populate distinct sensor tap identifiers
                cursor.execute("SELECT DISTINCT tap_number FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
                discovered_taps = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                self.tap_combo.clear()
                for tap_num in discovered_taps:
                    self.tap_combo.addItem(f"Physical Tap {tap_num}", tap_num)
                
                self.log_message(f"📋 Loaded {len(discovered_taps)} active spatial tap channels into plot selectors.")
                
                self.btn_export_all_time.setEnabled(True)
                self.btn_export_summary.setEnabled(True)
                self.btn_plot_contour.setEnabled(True)
                self.btn_plot_history.setEnabled(True)
            else:
                self.log_message("❌ Processing Pipeline Ingestion Fault: Parser returned NULL data blocks.")
        except Exception as e:
            self.log_message(f"❌ Pipeline Fault: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def render_tap_time_history_plot(self):
        """🎯 NEW PLOTTER ENGINE: Extracts raw horizontal float streams out of SQLite and renders full history graphs."""
        if not self.active_model_id: 
            return
            
        selected_tap = self.tap_combo.currentData()
        if selected_tap is None: 
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT time_history 
                FROM tap_measurements 
                WHERE model_id=? AND wind_angle=? AND tap_number=?
            """, (self.active_model_id, self.active_wind_angle, selected_tap))
            row = cursor.fetchone()
            conn.close()
            
            if not row or not row[0]:
                self.log_message(f"❌ Plotter fault: No time history payload cached for Tap {selected_tap}.")
                return
                
            # Unbox the flat continuous byte chunk straight back into low-overhead float metrics
            series = np.frombuffer(row[0], dtype=np.float32)
            time_steps = np.arange(len(series))
            
            # Refresh Matplotlib frame
            self.canvas.figure.clear()
            self.canvas.axes = self.canvas.figure.add_subplot(111)
            
            # High-performance crisp fine-line rendering parameter assignment
            self.canvas.axes.plot(time_steps, series, color='#8e44ad', linewidth=0.7, alpha=0.85)
            
            # Graphical annotations and typography
            angle_deg = float(self.active_wind_angle / 10.0 if self.active_wind_angle > 360 else self.active_wind_angle)
            self.canvas.axes.set_title(f"Tap {selected_tap} Fluctuation Signal History (Wind Angle: {angle_deg}°)")
            self.canvas.axes.set_xlabel("Time History Frames (Steps)")
            self.canvas.axes.set_ylabel("Pressure Coefficient Coefficient ($C_p$)")
            self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            self.log_message(f"🎨 UI Visualization frame re-rendered for Tap {selected_tap} Fluctuation Waveform.")
            
        except Exception as e:
            self.log_message(f"❌ Time History Renderer Fault: {e}")

    def export_full_time_series_csv(self):
        if not self.active_model_id: return
        out_csv, _ = QFileDialog.getSaveFileName(self, "Export Full Time Series", f"nist_series_{self.active_wind_angle}.csv", "CSV Files (*.csv)")
        if not out_csv: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT tap_number, time_history FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
            rows = cursor.fetchall()
            conn.close()
            
            csv_data, max_len = {}, 0
            for row in rows:
                series = np.frombuffer(row[1], dtype=np.float32)
                csv_data[f"Tap_{row[0]}_Cp"] = series
                if len(series) > max_len: max_len = len(series)
            
            with open(out_csv, "w") as f:
                f.write(",".join(["TimeStep"] + list(csv_data.keys())) + "\n")
                arrays = [csv_data[k] for k in csv_data.keys()]
                for step in range(max_len):
                    f.write(",".join([str(step)] + [str(arr[step]) if step < len(arr) else "" for arr in arrays]) + "\n")
            self.log_message(f"🎉 CSV Saved: {out_csv}")
        except Exception as e: self.log_message(f"❌ Export Error: {e}")
        finally: QApplication.restoreOverrideCursor()

    def export_spatial_summary_csv(self):
        if not self.active_model_id: return
        out_csv, _ = QFileDialog.getSaveFileName(self, "Export Spatial Summary", f"nist_spatial_{self.active_wind_angle}.csv", "CSV Files (*.csv)")
        if not out_csv: return
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT tap_number, mean_cp, std_cp, face, x_coord, y_coord FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
            rows = cursor.fetchall()
            conn.close()
            with open(out_csv, "w") as f:
                f.write("Tap_Number,Mean_Cp,Std_Dev_Cp,Raw_Face_Code,X_Coordinate,Y_Coordinate\n")
                for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3] if r[3] is not None else ''},{r[4] if r[4] is not None else ''},{r[5] if r[5] is not None else ''}\n")
            self.log_message(f"🎉 Summary Saved: {out_csv}")
        except Exception as e: self.log_message(f"❌ Export Error: {e}")

    def render_spatial_contour_map(self):
        if not self.active_model_id: return
        selected_face = self.face_combo.currentData()
        selected_metric = self.metric_combo.currentData()
        col_target = "mean_cp" if selected_metric == "mean_cp" else "std_cp"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if selected_face == "all":
                cursor.execute(f"SELECT {col_target}, x_coord, y_coord FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
            else:
                cursor.execute(f"SELECT {col_target}, x_coord, y_coord FROM tap_measurements WHERE model_id=? AND wind_angle=? AND face=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle, selected_face))
            records = cursor.fetchall()
            conn.close()
            
            if len(records) < 4: return
            values, x_coords, y_coords = np.array([r[0] for r in records]), np.array([r[1] for r in records]), np.array([r[2] for r in records])

            self.canvas.figure.clear()
            self.canvas.axes = self.canvas.figure.add_subplot(111)
            contour = self.canvas.axes.tricontourf(x_coords, y_coords, values, levels=15, cmap='RdBu_r' if selected_metric == "mean_cp" else 'viridis')
            self.canvas.axes.scatter(x_coords, y_coords, color='black', edgecolors='white', s=40, linewidths=1.2, zorder=3)
            
            self.canvas.axes.set_title(f"NIST Spatial Grid Distribution Map")
            self.canvas.axes.set_xlabel("Absolute Coordinate Tracking Axis (X)")
            self.canvas.axes.set_ylabel("Absolute Coordinate Tracking Axis (Y)")
            self.canvas.figure.colorbar(contour, ax=self.canvas.axes, label="Cp Vector Intensity")
            self.canvas.figure.tight_layout()
            self.canvas.draw()
        except Exception as e: self.log_message(f"❌ Plotter Error: {e}")

    def closeEvent(self, event):
        clear_session_data()
        event.accept()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = app.palette()
    from PyQt6.QtGui import QPalette, QColor
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    workbench = NISTDesktopWorkbench()
    workbench.show()
    sys.exit(app.exec())