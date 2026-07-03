# -*- coding: utf-8 -*-
"""
TPU_parser.py - High-Performance Vectorized Storage Layer with Block Navigation
"""

import os
import re
import sqlite3
import numpy as np
import scipy.io

DB_PATH = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\TPU_pipeline\local_wind_data.db"
DROP_FOLDER = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"


def initialize_local_database():
    """Initializes database schemas, automatically upgrading outdated configurations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(tap_measurements)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        if existing_columns and "x_coord" not in existing_columns:
            cursor.execute("DROP TABLE IF EXISTS tap_measurements")
            cursor.execute("DROP TABLE IF EXISTS origin_models")
    except Exception:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS origin_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            total_taps INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tap_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            wind_angle INTEGER,
            tap_number INTEGER,
            mean_cp REAL,
            std_cp REAL,
            face INTEGER,
            x_coord REAL,
            y_coord REAL,
            time_history BLOB,
            FOREIGN KEY(model_id) REFERENCES origin_models(id),
            UNIQUE(model_id, wind_angle, tap_number)
        )
    """)
    conn.commit()
    conn.close()


def populate_database_from_mat(mat_file_path, target_angle=None):
    """
    Parses a TPU .mat file, extracts spatial coordinates, metrics, 
    and populates SQLite using a resilient fallback taxonomy layer.
    """
    base_name = os.path.basename(mat_file_path)
    clean_name = base_name.replace(".mat", "")
    
    # 🎯 FIX: Default straight to the UI's selected target angle 
    wind_angle = target_angle
    model_name = clean_name

    # Attempt to extract clean taxonomy parts if available
    match = re.search(r"([A-Za-z0-9_]+)_(\d+)(?:_\d+)?$", clean_name)
    if match:
        model_name = match.group(1)
        if wind_angle is None:
            wind_angle = int(match.group(2))
    elif "_" in clean_name:
        parts = clean_name.split("_")
        if parts[-1].isdigit() and wind_angle is None:
            wind_angle = int(parts[-1])
            model_name = "_".join(parts[:-1])
        elif len(parts) >= 2 and parts[-2].isdigit() and wind_angle is None:
            wind_angle = int(parts[-2])
            model_name = "_".join(parts[:-2])

    if wind_angle is None:
        wind_angle = 0  # Absolute fallback anchor

    print(f"📦 Extracting TPU data from: {mat_file_path} (Model: {model_name}, Angle: {wind_angle}°)...")
    mat_contents = scipy.io.loadmat(mat_file_path)

    all_numeric_arrays = []

    def find_all_arrays(item):
        if not isinstance(item, np.ndarray):
            return
        if item.dtype.kind in 'bifc':
            if len(item.shape) == 2:
                all_numeric_arrays.append(item)
            return
        if item.dtype.names:
            for name in item.dtype.names:
                find_all_arrays(item[name])
            return
        if item.dtype.kind == 'O':
            for sub_item in item.ravel():
                if isinstance(sub_item, np.ndarray):
                    find_all_arrays(sub_item)

    for key, value in mat_contents.items():
        if not key.startswith("__"):
            find_all_arrays(value)

    if not all_numeric_arrays:
        print("❌ Error: Could not locate any valid numeric data matrices inside the file.")
        return None, None

    data_matrix = max(all_numeric_arrays, key=lambda x: x.size)
    time_steps, total_taps = data_matrix.shape

    location_matrix = None
    for arr in all_numeric_arrays:
        if arr is data_matrix:
            continue
        if arr.shape[1] == total_taps and arr.shape[0] >= 4:
            location_matrix = arr
            break
        elif arr.shape[0] == total_taps and arr.shape[1] >= 4:
            location_matrix = arr.T
            break

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        
        cursor.execute(
            "INSERT OR IGNORE INTO origin_models (model_name, total_taps) VALUES (?, ?)",
            (model_name, total_taps)
        )
        cursor.execute("SELECT id FROM origin_models WHERE model_name = ?", (model_name,))
        model_id = cursor.fetchone()[0]
        
        for tap_idx in range(total_taps):
            tap_number = tap_idx + 1
            raw_series = data_matrix[:, tap_idx]

            mean_val = float(np.mean(raw_series))
            std_val = float(np.std(raw_series))

            x_coord, y_coord, face_code = None, None, None
            if location_matrix is not None:
                try:
                    x_coord = float(location_matrix[0, tap_idx])   
                    y_coord = float(location_matrix[1, tap_idx])   
                    face_code = int(round(location_matrix[3, tap_idx]))  
                except Exception:
                    pass

            binary_blob = sqlite3.Binary(raw_series.astype(np.float32).tobytes())

            cursor.execute("""
                INSERT OR REPLACE INTO tap_measurements 
                (model_id, wind_angle, tap_number, mean_cp, std_cp, face, x_coord, y_coord, time_history) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_id, wind_angle, tap_number, mean_val, std_val, face_code, x_coord, y_coord, binary_blob))

        conn.commit()
        return model_id, wind_angle

    except Exception as db_err:
        conn.rollback()
        print(f"❌ Database Transaction Failed: {db_err}")
        return None, None
    finally:
        conn.close()


def clear_session_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM tap_measurements")
        cursor.execute("DELETE FROM origin_models")
        conn.commit()
        cursor.execute("VACUUM")
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

    if os.path.exists(DROP_FOLDER):
        for filename in os.listdir(DROP_FOLDER):
            file_path = os.path.join(DROP_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
            except Exception:
                pass