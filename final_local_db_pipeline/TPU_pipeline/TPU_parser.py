# -*- coding: utf-8 -*-
"""
tpu_to_sqlite.py - Storage Layer with Auto-Schema Alignment & Raw Integer Face Tracking
"""

import os
import re
import sqlite3
import numpy as np
import scipy.io

DB_PATH = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\TPU_pipeline\local_wind_data.db"
DROP_FOLDER = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"


def initialize_local_database():
    """Initializes clean database schemas, automatically correcting outdated schemas from previous runs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 🎯 FIX: Inspect current table columns to see if we need to drop an old test schema
    try:
        cursor.execute("PRAGMA table_info(tap_measurements)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # If the table exists but is missing the new 'face' tracking column, force a clean slate
        if existing_columns and "face" not in existing_columns:
            print("🔄 Outdated schema layout detected. Re-aligning session cache tables...")
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
            face INTEGER,         -- Raw MATLAB structural placement codes (1, 2, 3, 4)
            time_history BLOB,
            FOREIGN KEY(model_id) REFERENCES origin_models(id),
            UNIQUE(model_id, wind_angle, tap_number)
        )
    """)

    conn.commit()
    conn.close()


def populate_database_from_mat(mat_file_path):
    """Parses a TPU .mat file, extracts stats/raw face indices, and populates SQLite."""
    base_name = os.path.basename(mat_file_path)
    match = re.search(r"([A-Za-z0-9_]+)_(\d+)_\d+\.mat", base_name)
    
    if not match:
        print(f"❌ Error: Filename '{base_name}' doesn't match standard TPU naming taxonomy.")
        return None, None

    model_name = match.group(1)
    wind_angle = int(match.group(2))

    print(f"📦 Extracting TPU data from: {mat_file_path}...")
    mat_contents = scipy.io.loadmat(mat_file_path)

    all_numeric_arrays = []

    def find_all_arrays(item):
        if isinstance(item, np.ndarray):
            if item.dtype.kind in 'bifc' and len(item.shape) == 2:
                all_numeric_arrays.append(item)
            elif item.dtype.names:
                for name in item.dtype.names:
                    for sub_item in item[name].flat:
                        find_all_arrays(sub_item)
            elif item.dtype.kind == 'O':
                for sub_item in item.flat:
                    find_all_arrays(sub_item)

    for key, value in mat_contents.items():
        if not key.startswith("__"):
            find_all_arrays(value)

    if not all_numeric_arrays:
        print("❌ Error: Could not locate any valid numeric data matrices inside the file.")
        return None, None

    # Isolate the true time-series pressure matrix
    data_matrix = max(all_numeric_arrays, key=lambda x: x.size)
    time_steps, total_taps = data_matrix.shape
    print(f"📊 Matrix signature unboxed: {time_steps} time steps across {total_taps} channels.")

    # Isolate the tap properties/location matrix containing face numbers
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

    if location_matrix is not None:
        print("🔍 Successfully located tap metadata properties array.")
    else:
        print("⚠️ Warning: Tap metadata properties table not detected. Face codes set to NULL.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT OR IGNORE INTO origin_models (model_name, total_taps) VALUES (?, ?)",
            (model_name, total_taps)
        )
        cursor.execute("SELECT id FROM origin_models WHERE model_name = ?", (model_name,))
        model_id = cursor.fetchone()[0]

        print(f"🔄 Processing records and raw face rows for {total_taps} pressure taps...")
        
        for tap_idx in range(total_taps):
            tap_number = tap_idx + 1
            raw_series = data_matrix[:, tap_idx]

            mean_val = float(np.mean(raw_series))
            std_val = float(np.std(raw_series))

            # Grab the raw face code integer straight from row 4 (index 3) of the matrix
            face_code = None
            if location_matrix is not None:
                try:
                    face_code = int(round(location_matrix[3, tap_idx]))
                except Exception:
                    pass

            binary_blob = sqlite3.Binary(raw_series.astype(np.float32).tobytes())

            cursor.execute("""
                INSERT OR REPLACE INTO tap_measurements 
                (model_id, wind_angle, tap_number, mean_cp, std_cp, face, time_history) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_id, wind_angle, tap_number, mean_val, std_val, face_code, binary_blob))

        conn.commit()
        print(f"🎉 Database session cache populated from '{base_name}'!")
        return model_id, wind_angle

    except Exception as db_err:
        conn.rollback()
        print(f"❌ Database Transaction Failed: {db_err}")
        return None, None
    finally:
        conn.close()


def clear_session_data():
    """Wipes rows from the database, runs VACUUM, and clears out the file_drop folder completely."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        print("\n实时 Data Cache Wipe: Initiating session teardown protocol...")
        cursor.execute("DELETE FROM tap_measurements")
        cursor.execute("DELETE FROM origin_models")
        conn.commit()
        
        print("⚡ Reclaiming database disk blocks via VACUUM...")
        cursor.execute("VACUUM")
        conn.commit()
        print("✨ Database successfully reset back to a zero state.")
    except Exception as e:
        print(f"❌ Database clear failed: {e}")
    finally:
        conn.close()

    # Completely clear out the file_drop folder contents
    if os.path.exists(DROP_FOLDER):
        print(f"🧹 Sweeping clean the file_drop workspace folder...")
        for filename in os.listdir(DROP_FOLDER):
            file_path = os.path.join(DROP_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not delete file {filename}: {e}")
        print("✨ file_drop folder completely emptied back to a clean state.")