# -*- coding: utf-8 -*-
"""
app_gui.py - PyQt6 Unified Desktop Workbench for Wind Engineering Analysis
Tailored explicitly for the TPU_pipeline directory environment.
"""

import sys
import os
import sqlite3
import numpy as np
import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QPushButton, QTextEdit, QGroupBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt

# Embed Matplotlib natively inside Qt layouts
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 🎯 Fixed Imports: Pulling directly from your actual TPU_parser.py file
from TPU_parser import (
    populate_database_from_mat, 
    initialize_local_database, 
    clear_session_data, 
    DB_PATH
)
from tpu_scraper import create_legacy_session, find_dropdown_recursively

# Explicit absolute path to your root file drop folder
DROP_FOLDER = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"


class MplCanvas(FigureCanvas):
    """Integrated graphics rendering canvas component."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class TPUDesktopWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPU Wind Tunnel Data Engineering Workbench")
        self.setMinimumSize(1200, 750)
        
        # Core Session State Tracking Variables
        self.session = create_legacy_session()
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.current_url = ""
        self.resolved_page_url = ""
        self.angle_links_map = {}
        self.active_model_id = None
        self.active_wind_angle = None
        
        # Portal routes matching legacy database schemas
        self.portals = {
            "High-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm",
            "Low-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/Homepage/homepageLDF.htm",
            "Low-Rise Buildings with Eaves": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/eaves/Homepage/homepageEDF.htm",
            "Low-Rise Buildings (Non-Isolated)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise2/Homepage/homepageLDF2.htm",
        }

        # Auto-initialize database storage layer tables on workbench launch
        initialize_local_database()
        
        self.init_ui_layout()
        self.log_message("🚀 Workbench initialized. Ready to map target portal configuration topologies.")

    def init_ui_layout(self):
        """Constructs a responsive, scannable split-pane interface structure."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QHBoxLayout(main_widget)
        
        # Use a splitter layout container so users can dynamically resize the data window grids
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)

        # ==========================================
        # LEFT CONTROL PANEL: Ingestion Controls
        # ==========================================
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Group 1: Portal Navigation Wizard Controls
        ingest_group = QGroupBox("Crawler Configuration Layer")
        ingest_layout = QVBoxLayout(ingest_group)
        
        ingest_layout.addWidget(QLabel("Select Database Portal Category:"))
        self.portal_combo = QComboBox()
        self.portal_combo.addItems(self.portals.keys())
        ingest_layout.addWidget(self.portal_combo)
        
        self.btn_connect = QPushButton("Connect & Fetch Sub-Options")
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
        
        ingest_layout.addWidget(QLabel("Target Evaluation Wind Angle:"))
        self.angle_combo = QComboBox()
        ingest_layout.addWidget(self.angle_combo)
        
        self.btn_ingest = QPushButton("Ingest Target Dataset Matrix into Cache")
        self.btn_ingest.setStyleSheet("font-weight: bold; background-color: #2a75d3; color: white;")
        self.btn_ingest.clicked.connect(self.handle_dataset_ingestion)
        self.btn_ingest.setEnabled(False)
        ingest_layout.addWidget(self.btn_ingest)
        
        left_layout.addWidget(ingest_group)

        # Group 2: Output Operations Control Panel
        dashboard_group = QGroupBox("Live Session Dashboard Capabilities")
        dashboard_layout = QVBoxLayout(dashboard_group)
        
        self.btn_export_all_time = QPushButton("Export FULL Time-Series Grid to CSV")
        self.btn_export_all_time.clicked.connect(self.export_full_time_series_csv)
        self.btn_export_all_time.setEnabled(False)
        dashboard_layout.addWidget(self.btn_export_all_time)
        
        self.btn_export_summary = QPushButton("Export Spatial Metrics & Faces to CSV")
        self.btn_export_summary.clicked.connect(self.export_spatial_summary_csv)
        self.btn_export_summary.setEnabled(False)
        dashboard_layout.addWidget(self.btn_export_summary)
        
        self.btn_plot_contour = QPushButton("Render Spatial Surface Contour Plot")
        self.btn_plot_contour.setStyleSheet("font-weight: bold; background-color: #2aa25b; color: white;")
        self.btn_plot_contour.clicked.connect(self.render_spatial_contour_map)
        self.btn_plot_contour.setEnabled(False)
        dashboard_layout.addWidget(self.btn_plot_contour)
        
        left_layout.addWidget(dashboard_group)

        # Group 3: Console Output Logging Box
        log_group = QGroupBox("System Activity Diagnostics Console")
        log_layout = QVBoxLayout(log_group)
        self.console_out = QTextEdit()
        self.console_out.setReadOnly(True)
        self.console_out.setStyleSheet("background-color: #1e1e1e; color: #a9dc76; font-family: Consolas;")
        log_layout.addWidget(self.console_out)
        
        left_layout.addWidget(log_group)
        splitter.addWidget(left_panel)

        # ==========================================
        # RIGHT PANEL: Embedded Surface Plots Canvas
        # ==========================================
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        plot_group = QGroupBox("Integrated Visualization Display Array")
        plot_box_layout = QVBoxLayout(plot_group)
        
        # Attach the Matplotlib UI widgets natively inside our frame layout
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_box_layout.addWidget(self.toolbar)
        plot_box_layout.addWidget(self.canvas)
        right_layout.addWidget(plot_group)
        
        splitter.addWidget(right_panel)
        
        # Standardize left-to-right spacing weights
        splitter.setSizes([450, 750])

    # ==========================================
    # WORKBENCH INTERACTIVE FUNCTION ROUTINES
    # ==========================================
    def log_message(self, message):
        self.console_out.append(message)
        
    def handle_portal_connection(self):
        selected_key = self.portal_combo.currentText()
        self.current_url = self.portals[selected_key]
        self.log_message(f"\n🌐 Contacting Portal Gateway Matrix: {selected_key}...")
        self.populate_dropdown_combobox()

    def populate_dropdown_combobox(self):
        """Scans current nested layout structure contexts and translates fields to UI elements."""
        self.options_combo.clear()
        self.btn_next_level.setEnabled(False)
        
        select_tag, actual_url = find_dropdown_recursively(self.session, self.current_url, self.headers)
        
        if not select_tag:
            self.log_message("📊 Status: Grid leaf node reached. Final matrix table uncovered.")
            self.resolved_page_url = actual_url or self.current_url
            self.parse_final_angles_grid()
            return

        self.resolved_page_url = actual_url
        options_found = False
        for option in select_tag.find_all("option"):
            val = option.get("value")
            text = option.get_text().strip()
            if val == "def" or not val or "please select" in text.lower():
                continue
            self.options_combo.addItem(text, urljoin(self.resolved_page_url, val))
            options_found = True

        if options_found:
            self.log_message(f"📋 Loaded {self.options_combo.count()} sub-configuration categories into selection matrix.")
            self.btn_next_level.setEnabled(False if self.options_combo.count() == 0 else True)

    def handle_drill_down(self):
        """Advances down another level of the network frame structure."""
        selected_text = self.options_combo.currentText()
        next_target_url = self.options_combo.currentData()
        
        if next_target_url:
            self.log_message(f"🔍 Advancing downstream layer: '{selected_text}'")
            self.current_url = next_target_url
            self.populate_dropdown_combobox()

    def parse_final_angles_grid(self):
        """Parses the data rows to locate wind angle matrix coordinates and links."""
        self.angle_combo.clear()
        self.btn_ingest.setEnabled(False)
        self.angle_links_map = {}
        
        try:
            res = self.session.get(self.resolved_page_url, headers=self.headers, timeout=12, verify=False)
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            self.log_message(f"❌ Scraping Network Fault: {e}")
            return

        data_file_row, direction_row = None, None
        for row in soup.find_all("tr"):
            a_tags = row.find_all("a", href=True)
            if len([a for a in a_tags if ".mat" in a["href"].lower()]) > 1:
                data_file_row = row
                break

        if data_file_row:
            cur = data_file_row.find_previous_sibling("tr")
            while cur:
                if "Wind direction" in cur.get_text():
                    direction_row = cur
                    break
                cur = cur.find_previous_sibling("tr")

        if not direction_row or not data_file_row:
            self.log_message("❌ Framework Parse Error: Could not resolve wind angle table distribution grids.")
            return

        angles = []
        for td in direction_row.find_all(["td", "th"]):
            m = re.search(r"(\d+)", td.get_text().strip())
            if m:
                angles.append(int(m.group(1)))

        links = [urljoin(self.resolved_page_url, td.find("a")["href"]) for td in data_file_row.find_all("td") if td.find("a") and ".mat" in td.find("a")["href"].lower()]
        self.angle_links_map = dict(zip(angles, links[:len(angles)]))
        
        for angle in sorted(self.angle_links_map.keys()):
            self.angle_combo.addItem(f"Wind Angle: {angle}°", angle)
            
        if self.angle_links_map:
            self.log_message(f"🎯 Success! Discovered {len(self.angle_links_map)} evaluation wind coordinate datasets.")
            self.btn_ingest.setEnabled(True)

    def handle_dataset_ingestion(self):
        """Streams the target binary matrix, extracts characteristics, and activates capabilities."""
        self.btn_export_all_time.setEnabled(False)
        self.btn_export_summary.setEnabled(False)
        self.btn_plot_contour.setEnabled(False)
        
        target_angle = self.angle_combo.currentData()
        download_url = self.angle_links_map[target_angle]
        
        file_name = download_url.split('/')[-1]
        if not file_name.endswith('.mat'):
            file_name = f"tpu_angle_{target_angle}.mat"
            
        local_path = os.path.join(DROP_FOLDER, file_name)
        os.makedirs(DROP_FOLDER, exist_ok=True)
        
        self.log_message(f"\n📥 Running download streaming pipe for angle {target_angle}°...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            with self.session.get(download_url, stream=True, timeout=30, verify=False) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            self.log_message("⚡ Processing data into SQL database matrix...")
            model_id, wind_angle = populate_database_from_mat(local_path)
            
            if model_id is not None:
                self.active_model_id = model_id
                self.active_wind_angle = wind_angle
                self.log_message("🎉 Ingestion complete! Transient cache database locked and loaded.")
                
                self.btn_export_all_time.setEnabled(True)
                self.btn_export_summary.setEnabled(True)
                self.btn_plot_contour.setEnabled(True)
                
        except Exception as e:
            self.log_message(f"❌ Processing Pipeline Ingestion Fault: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

    def export_full_time_series_csv(self):
        """Unpacks SQL binary blobs and compiles full time-series matrices to CSV directly from UI."""
        if not self.active_model_id:
            return
            
        self.log_message("⏳ Unpacking compressed binary array timelines from database cache...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tap_number, time_history 
                FROM tap_measurements 
                WHERE model_id=? AND wind_angle=? 
                ORDER BY tap_number
            """, (self.active_model_id, self.active_wind_angle))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                self.log_message("❌ Extraction failed: No transient data located inside cache.")
                return

            csv_data = {}
            max_len = 0
            for row in rows:
                tap_num = row[0]
                series = np.frombuffer(row[1], dtype=np.float32)
                csv_data[f"Tap_{tap_num}_Cp"] = series
                if len(series) > max_len:
                    max_len = len(series)

            out_csv = os.path.join(DROP_FOLDER, f"tpu_tabular_time_history_angle_{self.active_wind_angle}.csv")
            
            with open(out_csv, "w") as f:
                f.write(",".join(["TimeStep"] + list(csv_data.keys())) + "\n")
                arrays = [csv_data[k] for k in csv_data.keys()]
                for step in range(max_len):
                    vals = [str(step)] + [str(arr[step]) if step < len(arr) else "" for arr in arrays]
                    f.write(",".join(vals) + "\n")
                    
            self.log_message(f"🎉 Success! Full Time-Series CSV written to:\n📁 {out_csv}")
        except Exception as e:
            self.log_message(f"❌ CSV Matrix Compilation Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_spatial_summary_csv(self):
        """Exports calculated statistics alongside their raw MATLAB face indices."""
        if not self.active_model_id:
            return
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tap_number, mean_cp, std_cp, face 
                FROM tap_measurements 
                WHERE model_id=? AND wind_angle=? 
                ORDER BY tap_number
            """, (self.active_model_id, self.active_wind_angle))
            rows = cursor.fetchall()
            conn.close()
            
            out_csv = os.path.join(DROP_FOLDER, f"tpu_spatial_summary_grid_angle_{self.active_wind_angle}.csv")
            with open(out_csv, "w") as f:
                f.write("Tap_Number,Mean_Cp,Std_Dev_Cp,Raw_Face_Code\n")
                for r in rows:
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3] if r[3] is not None else ''}\n")
                    
            self.log_message(f"🎉 Success! Spatial Geometry statistics sheet generated:\n📁 {out_csv}")
        except Exception as e:
            self.log_message(f"❌ Statistical mapping compilation failure: {e}")

    def render_spatial_contour_map(self):
        """Queries statistical vectors and updates the embedded Matplotlib element on the UI layout canvas."""
        if not self.active_model_id:
            return
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT mean_cp FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
            means = [r[0] for r in cursor.fetchall()]
            conn.close()
            
            if not means:
                self.log_message("❌ Visualization fault: No records found to map coordinates.")
                return

            self.canvas.axes.cla()
            
            total = len(means)
            cols = int(np.ceil(np.sqrt(total)))
            rows = int(np.ceil(total / cols))
            
            padded = np.zeros(rows * cols)
            padded[:total] = means
            grid_z = padded.reshape((rows, cols))
            
            contour = self.canvas.axes.contourf(grid_z, cmap='RdBu_r', levels=15)
            self.canvas.axes.set_title(f"TPU Grid Coordinate Wind Distribution Mapping (Angle: {self.active_wind_angle}°)")
            self.canvas.axes.set_xlabel("Horizontal Grid Space Axis (X)")
            self.canvas.axes.set_ylabel("Vertical Grid Space Axis (Y)")
            
            if hasattr(self, 'colorbar'):
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
            self.colorbar = self.canvas.figure.colorbar(contour, ax=self.canvas.axes, label="Mean Pressure Coefficient ($C_p$)")
            
            self.canvas.draw_idle()
            self.log_message("🎨 Visualization canvas re-rendered successfully.")
            
        except Exception as e:
            self.log_message(f"❌ Contour Renderer Error: {e}")

    def closeEvent(self, event):
        """Ties database evacuation hooks directly to desktop lifecycle close signals."""
        self.log_message("🧹 Terminating window context... Clearing transient cache...")
        clear_session_data()
        event.accept()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Apply a modern dark theme layout
    palette = app.palette()
    from PyQt6.QtGui import QPalette, QColor
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    workbench = TPUDesktopWorkbench()
    workbench.show()
    sys.exit(app.exec())