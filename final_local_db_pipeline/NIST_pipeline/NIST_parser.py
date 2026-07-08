# -*- coding: utf-8 -*-
"""
NIST_pipeline/NIST_parser.py - High-Performance File Extraction & Storage Backend
Implements In-Memory RAM Archive Sifting, Scale Calibration, and Explicit 2D Coordinates Mapping.
"""

import os
import re
import sqlite3
import zipfile
import shutil
import io
import numpy as np
from pyhdf.SD import SD, SDC  # Requires: pip install pyhdf

DB_PATH = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\NIST_pipeline\local_wind_data.db"
DROP_FOLDER = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"
ARCHIVE_TEMP_DIR = os.path.join(DROP_FOLDER, "unpacked_temp")

 
def initialize_local_database():
    """Initializes unified database schemas at the project root level."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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


def scan_archive_pure_memory(master_zip_path):
    """
    🎯 IN-MEMORY SCANNER: Maps out wind orientation angles purely inside RAM bytes buffers
    without extracting anything to your hard drive. Extremely fast and prevents UI freeze.
    """
    extracted_hdf_map = {}

    def walk_zip_stream(zip_file_context, source_path_or_buffer):
        for entry in zip_file_context.infolist():
            if entry.filename.lower().endswith('.zip'):
                try:
                    nested_zip_bytes = zip_file_context.read(entry.filename)
                    nested_stream = io.BytesIO(nested_zip_bytes)
                    with zipfile.ZipFile(nested_stream, 'r') as sub_z:
                        walk_zip_stream(sub_z, source_path_or_buffer)
                except Exception as e:
                    print(f"⚠️ Failed to parse inner nested zip memory stream: {e}")
            elif entry.filename.lower().endswith('.hdf'):
                filename_base = os.path.basename(entry.filename)
                clean_name = filename_base.replace(".hdf", "").replace(".HDF", "")
                
                if len(clean_name) >= 19 and clean_name[15:19].isdigit():
                    angle_key = int(clean_name[15:19])
                    extracted_hdf_map[angle_key] = (source_path_or_buffer, entry.filename)

    if os.path.exists(master_zip_path):
        try:
            with zipfile.ZipFile(master_zip_path, 'r') as z:
                walk_zip_stream(z, master_zip_path)
        except Exception as e:
            print(f"❌ Failed to parse master archive stream: {e}")
            
    return extracted_hdf_map


def populate_database_from_archive(zip_path, internal_hdf_name, target_angle):
    """
    🎯 TARGETED PARSER: Extracts ONLY the chosen HDF file, applies calibration scale factor, 
    parses row-wise histories, maps coordinates from Flat_Tap_Coordinates, and populates SQLite.
    """
    target_hdf_dest = os.path.join(DROP_FOLDER, os.path.basename(internal_hdf_name))
    os.makedirs(DROP_FOLDER, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            try:
                with open(target_hdf_dest, 'wb') as f:
                    f.write(z.read(internal_hdf_name))
            except KeyError:
                def locate_and_extract(parent_zip):
                    for entry in parent_zip.infolist():
                        if entry.filename.lower().endswith('.zip'):
                            nested_stream = io.BytesIO(parent_zip.read(entry.filename))
                            with zipfile.ZipFile(nested_stream, 'r') as sub_z:
                                if locate_and_extract(sub_z): return True
                        elif entry.filename == internal_hdf_name:
                            with open(target_hdf_dest, 'wb') as f:
                                f.write(parent_zip.read(entry.filename))
                            return True
                    return False
                locate_and_extract(z)
    except Exception as e:
        print(f"❌ Direct Target Extraction Failure: {e}")
        return None, None

    if not os.path.exists(target_hdf_dest):
        return None, None

    base_name = os.path.basename(target_hdf_dest)
    name_part = base_name.split('.')[0]
    model_name = name_part[:15] if len(name_part) >= 15 else name_part
    wind_angle = int(int(name_part[15:19]) / 10) if (len(name_part) >= 19 and name_part[15:19].isdigit()) else int(target_angle / 10)

    try:
        hdf_obj = SD(target_hdf_dest, SDC.READ)
        datasets_dict = hdf_obj.datasets()
    except Exception as e:
        print(f"❌ HDF4 Open Error: {e}")
        if os.path.exists(target_hdf_dest): os.remove(target_hdf_dest)
        return None, None

    main_ds_name = None
    max_size = 0
    all_numeric_datasets = {}
    
    for ds_name in datasets_dict.keys():
        try:
            ds_select = hdf_obj.select(ds_name)
            all_numeric_datasets[ds_name] = ds_select.get()
            _, rank, dims, _, _ = ds_select.info()
            ds_select.endaccess()
            
            size = 1
            for d in dims: size *= d
            if size > max_size and rank == 2:
                max_size = size
                main_ds_name = ds_name
        except Exception:
            pass

    if not main_ds_name:
        hdf_obj.end()
        if os.path.exists(target_hdf_dest): os.remove(target_hdf_dest)
        return None, None

    data_matrix = all_numeric_datasets[main_ds_name]
    
    # Extract Calibration Scaling Factor
    try:
        main_ds_obj = hdf_obj.select(main_ds_name)
        scale_factor = float(main_ds_obj.attr('scale_factor').get())
        main_ds_obj.endaccess()
    except Exception:
        scale_factor = 0.00296273  # High-precision verification fallback

    hdf_obj.end()
    if os.path.exists(target_hdf_dest):
        try: os.remove(target_hdf_dest)
        except Exception: pass

    # Resolve Matrix Orientation bounds safely
    if data_matrix.shape[0] < data_matrix.shape[1] and data_matrix.shape[0] > 1:
        total_channels, time_steps = data_matrix.shape
        is_row_wise = True
    else:
        time_steps, total_channels = data_matrix.shape
        is_row_wise = False

    # Isolate Pre-Calculated Statistics Table
    stats_matrix = None
    for name, arr in all_numeric_datasets.items():
        if name.lower() == "tap_max_min_mean_rms":
            stats_matrix = arr
            break

    mean_vector, rms_vector = None, None
    if stats_matrix is not None:
        if stats_matrix.shape[0] == 4:
            mean_vector = stats_matrix[2, :] * scale_factor
            rms_vector = stats_matrix[3, :] * scale_factor
        elif stats_matrix.shape[1] == 4:
            mean_vector = stats_matrix[:, 2] * scale_factor
            rms_vector = stats_matrix[:, 3] * scale_factor

    # -----------------------------------------------------------------
    # 🎯 TARGETED FIX: Explicitly extract the "Flat_Tap_Coordinates" table
    # -----------------------------------------------------------------
    location_matrix = None
    for name, arr in all_numeric_datasets.items():
        if name.lower() == "flat_tap_coordinates":
            location_matrix = arr
            break

    if location_matrix is not None:
        # Standardize matrix orientation to 4 x N layout for explicit row lookups
        # Row 0: Tap ID, Row 1: Face Number, Row 2: X-Coord, Row 3: Y-Coord
        if location_matrix.shape[1] == 4 and location_matrix.shape[0] != 4:
            location_matrix = location_matrix.T
            
        total_active_taps = location_matrix.shape[1]
        print(f"📐 Found 'Flat_Tap_Coordinates' mapping table. Total Active Taps constrained to: {total_active_taps}")
    else:
        total_active_taps = total_channels
        print(f"⚠️ Warning: 'Flat_Tap_Coordinates' dataset table missing. Defaulting to all {total_channels} channels.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        
        cursor.execute("INSERT OR IGNORE INTO origin_models (model_name, total_taps) VALUES (?, ?)", (model_name, total_active_taps))
        cursor.execute("SELECT id FROM origin_models WHERE model_name = ?", (model_name,))
        model_id = cursor.fetchone()[0]
        
        for tap_idx in range(total_active_taps):
            if is_row_wise:
                raw_series = data_matrix[tap_idx, :].astype(np.float32) * scale_factor
            else:
                raw_series = data_matrix[:, tap_idx].astype(np.float32) * scale_factor

            if mean_vector is not None and rms_vector is not None:
                mean_val = float(mean_vector[tap_idx])
                std_val = float(rms_vector[tap_idx])
            else:
                mean_val = float(np.mean(raw_series))
                std_val = float(np.std(raw_series))

            # 🎯 HARVEST ROUTINE: Map metrics out row-by-row based on your exact layout rules
            if location_matrix is not None:
                tap_number = int(round(location_matrix[0, tap_idx])) # Row 1 (Index 0): Tap ID
                face_code  = int(round(location_matrix[1, tap_idx])) # Row 2 (Index 1): Face ID
                x_coord    = float(location_matrix[2, tap_idx])      # Row 3 (Index 2): X Coordinate
                y_coord    = float(location_matrix[3, tap_idx])      # Row 4 (Index 3): Y Coordinate
            else:
                tap_number = tap_idx + 1
                face_code = 1
                x_coord = None
                y_coord = None

            binary_blob = sqlite3.Binary(raw_series.tobytes())
            cursor.execute("""
                INSERT OR REPLACE INTO tap_measurements 
                (model_id, wind_angle, tap_number, mean_cp, std_cp, face, x_coord, y_coord, time_history) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_id, wind_angle, tap_number, mean_val, std_val, face_code, x_coord, y_coord, binary_blob))

        conn.commit()
        return model_id, wind_angle
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
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
    except Exception: pass
    finally: conn.close()

    if os.path.exists(ARCHIVE_TEMP_DIR):
        try: shutil.rmtree(ARCHIVE_TEMP_DIR)
        except Exception: pass
    if os.path.exists(DROP_FOLDER):
        for filename in os.listdir(DROP_FOLDER):
            file_path = os.path.join(DROP_FOLDER, filename)
            try:
                if os.path.isfile(file_path): os.remove(file_path)
            except Exception: pass