# -*- coding: utf-8 -*-
"""
main_launcher.py - Centralized Dashboard Launcher Hub
"""

import sys
import os
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor

class WorkbenchHub(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wind Engineering Workbench Hub")
        self.setFixedSize(450, 220)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Select Target Analysis Environment Container:")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        layout.addWidget(title)

        btn_tpu = QPushButton("🚀 Launch TPU Multi-Page Scraper Workbench")
        btn_tpu.setStyleSheet("padding: 12px; font-size: 12px; font-weight: bold; background-color: #2a75d3; color: white;")
        btn_tpu.clicked.connect(self.launch_tpu)
        layout.addWidget(btn_tpu)

        btn_nist = QPushButton("📊 Launch NIST Single-Page Rowspan Index Workbench")
        btn_nist.setStyleSheet("padding: 12px; font-size: 12px; font-weight: bold; background-color: #2aa25b; color: white;")
        btn_nist.clicked.connect(self.launch_nist)
        layout.addWidget(btn_nist)

        self.setLayout(layout)

    def launch_tpu(self):
        script_path = os.path.join("TPU_pipeline", "app_gui.py")
        subprocess.Popen([sys.executable, script_path])
        self.close()

    def launch_nist(self):
        script_path = os.path.join("NIST_pipeline", "app_gui.py")
        subprocess.Popen([sys.executable, script_path])
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    hub = WorkbenchHub()
    hub.show()
    sys.exit(app.exec())