# -*- coding: utf-8 -*-
"""
app_gui.py - PyQt6 Unified Desktop Workbench for Wind Engineering Analysis
Equipped with absolute spatial mesh rendering, dynamic face discovery, and custom user export paths.
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
    QLabel, QComboBox, QPushButton, QTextEdit, QGroupBox, QSplitter, QFrame,
    QFileDialog  # 🎯 Added native file system picker framework
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Link with actual updated TPU_parser.py script
from TPU_parser import (
    populate_database_from_mat, 
    initialize_local_database, 
    clear_session_data, 
    DB_PATH,
    DROP_FOLDER
)
from tpu_scraper import create_legacy_session


def analyze_page_architecture(session, url, headers, visited=None):
    """
    Hybrid Navigation Scanner: Identifies final data tables,
    dropdown menus, or link grids to navigate configurations.
    """
    if visited is None:
        visited = set()
    if url in visited:
        return {}, url, False
    visited.add(url)
    
    url = url.strip()
    if url.startswith("about:") or "javascript:" in url.lower():
        return {}, url, False
        
    response = session.get(url, headers=headers, timeout=12, verify=False)
    response.raise_for_status()
    html_text = response.text

    soup = BeautifulSoup(html_text, "html.parser")
    
    # Priority 1: Inspect the current page context for explicit select dropdown menus
    select_tag = (
        soup.find("select", attrs={"name": "mysel"}) or 
        soup.find("select", attrs={"name": "urlsel"}) or 
        soup.find("select", attrs={"name": "building"}) or
        soup.find("select")
    )
    if select_tag:
        options_map = {}
        for option in select_tag.find_all("option"):
            val = option.get("value")
            if val is None:
                val = option.get_text().strip()
            else:
                val = val.strip()
                
            text = option.get_text().strip()
            if val == "def" or not val or "please select" in text.lower():
                continue
            options_map[text] = urljoin(url, val)
        if options_map:
            return options_map, url, False

    # Priority 2: Check sub-frames recursively but aggregate all options found
    frames = soup.find_all(["frame", "iframe"])
    if frames:
        combined_options = {}
        potential_final_url = url
        potential_is_final = False
        
        for frame in frames:
            src = frame.get("src")
            if src:
                sub_url = urljoin(url, src.strip())
                try:
                    opts, final_url, is_final = analyze_page_architecture(session, sub_url, headers, visited)
                    if is_final:
                        potential_final_url = final_url
                        potential_is_final = True
                    if opts:
                        combined_options.update(opts)
                except Exception:
                    pass 
                    
        if combined_options:
            return combined_options, url, False
            
        if potential_is_final:
            return {}, potential_final_url, True

    # Priority 3: Check for the true data table row signature
    is_final_data_page = False
    for row in soup.find_all("tr"):
        row_mat_links = [a for a in row.find_all("a", href=True) if ".mat" in a["href"].lower()]
        if len(row_mat_links) > 1:
            is_final_data_page = True
            break
            
    if is_final_data_page:
        return {}, url, True

    # Priority 4: Fallback text links
    options_map = {}
    a_tags = soup.find_all("a", href=True)
    for a in a_tags:
        href = a["href"].strip()
        text = a.get_text().strip()
        
        if not href or href.startswith("#") or "javascript:" in href.lower():
            continue
            
        if "http" in href.lower() and "arch.t-kougei.ac.jp" not in href.lower():
            continue
            
        clean_href = href.split('?')[0].split('#')[0].lower()
        if clean_href.endswith(('.htm', '.html', '/')) or not '.' in clean_href:
            if text and len(text) > 1 and not text.lower() in ['back', 'home', 'top', 'return']:
                options_map[text] = urljoin(url, href)
                    
    if options_map:
        return options_map, url, False

    return {}, url, False


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class TPUDesktopWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPU Wind Tunnel Data Engineering Workbench")
        self.setMinimumSize(1200, 750)
        
        self.session = create_legacy_session()
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.current_url = ""
        self.resolved_page_url = ""
        self.angle_links_map = {}
        self.active_model_id = None
        self.active_wind_angle = None
        
        self.portals = {
            "High-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm",
            "Low-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/mainpage.html",
            "Low-Rise Buildings with Eaves": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowriseeave/mainpage.html",
            "Low-Rise Buildings (Non-Isolated)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/grouplowrise/mainpage.html",
        }

        initialize_local_database()
        self.init_ui_layout()
        self.log_message("🚀 Workbench initialized. Ready to map portal layouts.")

    def init_ui_layout(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
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

        log_group = QGroupBox("System Activity Diagnostics Console")
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

    def log_message(self, message):
        self.console_out.append(message)
        
    def handle_portal_connection(self):
        selected_key = self.portal_combo.currentText()
        self.current_url = self.portals[selected_key]
        self.log_message(f"\n🌐 Mapping Portal Structure: {selected_key}...")
        self.populate_dropdown_combobox()

    def populate_dropdown_combobox(self):
        self.options_combo.clear()
        self.btn_next_level.setEnabled(False)
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            options_map, resolved_url, is_final_page = analyze_page_architecture(self.session, self.current_url, self.headers)
            err_msg = None
        except Exception as e:
            options_map, resolved_url, is_final_page = {}, self.current_url, False
            err_msg = f"❌ Ingestion Crawler Exception: {e}"
        QApplication.restoreOverrideCursor()
        
        if err_msg:
            self.log_message(err_msg)
            return
            
        self.resolved_page_url = resolved_url
        
        if is_final_page:
            self.log_message("📊 Status: Leaf data page reached successfully. Opening wind directions grid...")
            self.parse_final_angles_grid()
            return

        if options_map:
            self.log_message(f"📋 Loaded {len(options_map)} options into the selection layout step.")
            for text, target_link in sorted(options_map.items()):
                self.options_combo.addItem(text, target_link)
            self.btn_next_level.setEnabled(True)
        else:
            self.log_message("⚠️ Warning: No navigation elements found on this layer link.")

    def handle_drill_down(self):
        selected_text = self.options_combo.currentText()
        next_target_url = self.options_combo.currentData()
        
        if next_target_url:
            self.log_message(f"🔍 Advancing to next configuration layer: '{selected_text}'")
            self.current_url = next_target_url
            self.populate_dropdown_combobox()

    def parse_final_angles_grid(self):
        self.angle_combo.clear()
        self.btn_ingest.setEnabled(False)
        self.angle_links_map = {}
        
        try:
            res = self.session.get(self.resolved_page_url, headers=self.headers, timeout=12, verify=False)
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            self.log_message(f"❌ Scraping Network Fault: {e}")
            return

        data_file_row = None
        for row in soup.find_all("tr"):
            row_mat_links = [a for a in row.find_all("a", href=True) if ".mat" in a["href"].lower()]
            if len(row_mat_links) > 1:
                data_file_row = row
                break

        if not data_file_row:
            self.log_message("❌ Framework Parse Error: Could not locate binary matrix download array table.")
            return

        def cell_contains_angle(td):
            if re.search(r"\d+", td.get_text()):
                return True
            img = td.find("img")
            if img and img.get("src") and re.search(r"\d+", os.path.basename(img["src"])):
                return True
            return False

        direction_row = None
        cur = data_file_row.find_previous_sibling("tr")
        while cur:
            has_mat = any(".mat" in a["href"].lower() for a in cur.find_all("a", href=True))
            if not has_mat and len([td for td in cur.find_all(["td", "th"]) if cell_contains_angle(td)]) > 1:
                direction_row = cur
                break
            cur = cur.find_previous_sibling("tr")

        if not direction_row:
            parent_table = data_file_row.find_parent("table")
            if parent_table:
                for row in parent_table.find_all("tr"):
                    if row is data_file_row:
                        continue
                    has_mat = any(".mat" in a["href"].lower() for a in row.find_all("a", href=True))
                    if not has_mat and len([td for td in row.find_all(["td", "th"]) if cell_contains_angle(td)]) > 1:
                        direction_row = row
                        break

        if not direction_row:
            self.log_message("❌ Framework Parse Error: Could not resolve header wind angle row mapping coordinates.")
            return

        angles = []
        for td in direction_row.find_all(["td", "th"]):
            text_val = td.get_text().strip()
            m_text = re.search(r"(\d+)", text_val)
            if m_text:
                angles.append(int(m_text.group(1)))
                continue
                
            img_tag = td.find("img")
            if img_tag and img_tag.get("src"):
                filename_base = os.path.basename(img_tag["src"])
                m_img = re.search(r"(\d+)", filename_base)
                if m_img:
                    angles.append(int(m_img.group(1)))
                    continue

        links = [urljoin(self.resolved_page_url, td.find("a")["href"]) for td in data_file_row.find_all("td") if td.find("a") and ".mat" in td.find("a")["href"].lower()]
        self.angle_links_map = dict(zip(angles, links[:len(angles)]))
        
        if self.angle_links_map:
            for angle in sorted(self.angle_links_map.keys()):
                self.angle_combo.addItem(f"Wind Angle: {angle}°", angle)
            self.log_message(f"🎯 Success! Located {len(self.angle_links_map)} available wind direction options.")
            self.btn_ingest.setEnabled(True)
        else:
            self.log_message("❌ Framework Parse Error: Identified file rows but parsed 0 wind direction angles from the table header.")

    def handle_dataset_ingestion(self):
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
        
        self.log_message(f"\n📥 Streaming dataset for wind angle {target_angle}°...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            with self.session.get(download_url, stream=True, timeout=30, verify=False) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            self.log_message("⚡ Ingesting data matrices into SQL container layout...")
            QApplication.processEvents()
            
            model_id, wind_angle = populate_database_from_mat(local_path, target_angle=target_angle)
            
            if model_id is not None:
                self.active_model_id = model_id
                self.active_wind_angle = wind_angle
                self.log_message("🎉 Load Complete! Session cache database is active.")
                
                self.face_combo.clear()
                self.face_combo.addItem("All Faces Combined", "all")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT face 
                    FROM tap_measurements 
                    WHERE model_id=? AND wind_angle=? AND face IS NOT NULL 
                    ORDER BY face
                """, (self.active_model_id, self.active_wind_angle))
                discovered_faces = [row[0] for row in cursor.fetchall()]
                # Query and populate distinct sensor tap identifiers
                cursor.execute("SELECT DISTINCT tap_number FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (self.active_model_id, self.active_wind_angle))
                discovered_taps = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                for face_num in discovered_faces:
                    self.face_combo.addItem(f"Face {face_num} Layout", face_num)
                
                self.log_message(f"📋 Discovered {len(discovered_faces)} unique surface faces inside matrix architecture.")
                
                self.tap_combo.clear()
                for tap_num in discovered_taps:
                    self.tap_combo.addItem(f"Physical Tap {tap_num}", tap_num)
                
                self.log_message(f"📋 Loaded {len(discovered_taps)} active spatial tap channels into plot selectors.")

                self.btn_export_all_time.setEnabled(True)
                self.btn_export_summary.setEnabled(True)
                self.btn_plot_contour.setEnabled(True)
                self.btn_plot_history.setEnabled(True)
            else:
                self.log_message("❌ Processing Pipeline Ingestion Fault: Parser failed to unbox valid database rows.")
                
        except Exception as e:
            self.log_message(f"❌ Processing Pipeline Ingestion Fault: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
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
        """🎯 UPGRADE: Let user choose the save location natively and prevent automated wiping."""
        if not self.active_model_id:
            return
            
        # Spawn native system window to request target save location
        default_filename = f"tpu_tabular_time_history_angle_{self.active_wind_angle}.csv"
        out_csv, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Full Time-Series Grid", 
            default_filename, 
            "CSV Files (*.csv)"
        )
        
        # If user closes or cancels window, exit out gracefully
        if not out_csv:
            self.log_message("⚠️ Export cancelled by user.")
            return

        self.log_message("⏳ Generating matrix export spreadsheet... This may take a moment.")
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
                self.log_message("❌ Extraction failed: Data missing from cache.")
                return

            csv_data = {}
            max_len = 0
            for row in rows:
                tap_num = row[0]
                series = np.frombuffer(row[1], dtype=np.float32)
                csv_data[f"Tap_{tap_num}_Cp"] = series
                if len(series) > max_len:
                    max_len = len(series)
            
            with open(out_csv, "w") as f:
                f.write(",".join(["TimeStep"] + list(csv_data.keys())) + "\n")
                arrays = [csv_data[k] for k in csv_data.keys()]
                for step in range(max_len):
                    vals = [str(step)] + [str(arr[step]) if step < len(arr) else "" for arr in arrays]
                    f.write(",".join(vals) + "\n")
                    
            self.log_message(f"🎉 Success! Full Matrix CSV generated:\n📁 {out_csv}")
        except Exception as e:
            self.log_message(f"❌ CSV Compilation Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_spatial_summary_csv(self):
        """🎯 UPGRADE: Let user choose the save location natively and prevent automated wiping."""
        if not self.active_model_id:
            return
            
        # Spawn native system window to request target save location
        default_filename = f"tpu_spatial_summary_grid_angle_{self.active_wind_angle}.csv"
        out_csv, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Spatial Metrics & Faces", 
            default_filename, 
            "CSV Files (*.csv)"
        )
        
        if not out_csv:
            self.log_message("⚠️ Export cancelled by user.")
            return
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tap_number, mean_cp, std_cp, face, x_coord, y_coord 
                FROM tap_measurements 
                WHERE model_id=? AND wind_angle=? 
                ORDER BY tap_number
            """, (self.active_model_id, self.active_wind_angle))
            rows = cursor.fetchall()
            conn.close()
            
            with open(out_csv, "w") as f:
                f.write("Tap_Number,Mean_Cp,Std_Dev_Cp,Raw_Face_Code,X_Coordinate,Y_Coordinate\n")
                for r in rows:
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3] if r[3] is not None else ''},{r[4] if r[4] is not None else ''},{r[5] if r[5] is not None else ''}\n")
                    
            self.log_message(f"🎉 Success! Spatial statistics CSV written:\n📁 {out_csv}")
        except Exception as e:
            self.log_message(f"❌ Statistical mapping compilation failure: {e}")

    def render_spatial_contour_map(self):
        if not self.active_model_id:
            return
            
        selected_face = self.face_combo.currentData()
        selected_metric = self.metric_combo.currentData()
        metric_label = "Mean Cp" if selected_metric == "mean_cp" else "Std Dev Cp"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            col_target = "mean_cp" if selected_metric == "mean_cp" else "std_cp"
            
            if selected_face == "all":
                cursor.execute(f"""
                    SELECT {col_target}, x_coord, y_coord 
                    FROM tap_measurements 
                    WHERE model_id=? AND wind_angle=? 
                    ORDER BY tap_number
                """, (self.active_model_id, self.active_wind_angle))
                face_label_str = "All Faces Combined"
            else:
                cursor.execute(f"""
                    SELECT {col_target}, x_coord, y_coord 
                    FROM tap_measurements 
                    WHERE model_id=? AND wind_angle=? AND face=?
                    ORDER BY tap_number
                """, (self.active_model_id, self.active_wind_angle, selected_face))
                face_label_str = f"Isolated Face {selected_face}"

            records = cursor.fetchall()
            conn.close()
            
            if not records:
                self.log_message(f"❌ Visualization fault: No records found to map for {face_label_str}.")
                return

            if len(records) < 4:
                self.log_message(f"⚠️ Warning: Not enough spatial data nodes on {face_label_str} to compute triangulation gradients (minimum 4 nodes required).")
                return

            values = np.array([r[0] for r in records])
            x_coords = np.array([r[1] for r in records])
            y_coords = np.array([r[2] for r in records])

            if None in x_coords or None in y_coords:
                self.log_message("⚠️ Spatial tracking vectors incomplete. Unable to perform mesh layout rendering.")
                return

            self.canvas.figure.clear()
            self.canvas.axes = self.canvas.figure.add_subplot(111)
            
            chosen_cmap = 'RdBu_r' if selected_metric == "mean_cp" else 'viridis'
            
            contour = self.canvas.axes.tricontourf(x_coords, y_coords, values, levels=15, cmap=chosen_cmap)
            self.canvas.axes.scatter(x_coords, y_coords, color='black', edgecolors='white', s=40, linewidths=1.2, alpha=0.9, zorder=3)
            
            self.canvas.axes.set_title(f"TPU Spatial Grid {metric_label} Map - {face_label_str} (Angle: {self.active_wind_angle}°)")
            self.canvas.axes.set_xlabel("Absolute Coordinate Tracking Axis (X)")
            self.canvas.axes.set_ylabel("Absolute Coordinate Tracking Axis (Y)")
            
            cbar_title = "Mean Pressure Coefficient (C_p)" if selected_metric == "mean_cp" else "Standard Deviation (\u03c3)"
            self.canvas.figure.colorbar(contour, ax=self.canvas.axes, label=cbar_title)
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            self.log_message(f"🎨 UI Visualization frame re-rendered for {metric_label} ({face_label_str}).")
            
        except Exception as e:
            self.log_message(f"❌ Contour Renderer Error: {e}")

    def closeEvent(self, event):
        self.log_message("🧹 Terminating window context... Clearing transient cache...")
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
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    workbench = TPUDesktopWorkbench()
    workbench.show()
    sys.exit(app.exec())