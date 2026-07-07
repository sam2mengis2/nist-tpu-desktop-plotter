# -*- coding: utf-8 -*-
"""
NIST_pipeline/NIST_parser.py - High-Performance File Extraction & Storage Backend
Implements In-Memory RAM Archive Sifting and Targeted HDF4 Processing.
"""

import os
import re
import sqlite3
import zipfile
import io
import numpy as np
from pyhdf.SD import SD, SDC  # Requires: pip install pyhdf

DB_PATH = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\local_wind_data.db"
DROP_FOLDER = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"


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
                    # Unbox the nested zip file directly into a RAM byte stream buffer
                    nested_zip_bytes = zip_file_context.read(entry.filename)
                    nested_stream = io.BytesIO(nested_zip_bytes)
                    with zipfile.ZipFile(nested_stream, 'r') as sub_z:
                        walk_zip_stream(sub_z, source_path_or_buffer)
                except Exception as e:
                    print(f"⚠️ Failed to parse inner nested zip memory stream: {e}")
            elif entry.filename.lower().endswith('.hdf'):
                filename_base = os.path.basename(entry.filename)
                clean_name = filename_base.replace(".hdf", "").replace(".HDF", "")
                
                # README Position Slicing Rule (16-19)
                if len(clean_name) >= 19 and clean_name[15:19].isdigit():
                    angle_key = int(clean_name[15:19])
                    # Preserve the internal file track name so we can pull it later
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
    🎯 TARGETED PARSER: Pulls only the selected single HDF file from the archive path,
    sifts dimensions using metadata hooks, and commits records to SQLite.
    """
    target_hdf_dest = os.path.join(DROP_FOLDER, os.path.basename(internal_hdf_name))
    os.makedirs(DROP_FOLDER, exist_ok=True)
    
    # Extract only the selected HDF target file to the local drop folder
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Check for multi-tiered nested components
            try:
                with open(target_hdf_dest, 'wb') as f:
                    f.write(z.read(internal_hdf_name))
            except KeyError:
                # If nested deeply, unpack using temporary structural discovery steps
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
        print("❌ Error: Target HDF file extraction vanished from output space.")
        return None, None

    base_name = os.path.basename(target_hdf_dest)
    name_part = base_name.split('.')[0]
    model_name = name_part[:15] if len(name_part) >= 15 else name_part
    wind_angle = int(int(target_angle) / 10)

    try:
        hdf_obj = SD(target_hdf_dest, SDC.READ)
        datasets_dict = hdf_obj.datasets()
    except Exception as e:
        print(f"❌ HDF4 Open Error: {e}")
        if os.path.exists(target_hdf_dest): os.remove(target_hdf_dest)
        return None, None

    # Identify array blocks safely using metadata shapes
    main_ds_name = None
    max_size = 0
    ds_info_map = {}
    
    for ds_name in datasets_dict.keys():
        try:
            ds_select = hdf_obj.select(ds_name)
            _, rank, dims, _, _ = ds_select.info()
            ds_select.endaccess()
            
            ds_info_map[ds_name] = dims
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

    data_matrix = hdf_obj.select(main_ds_name).get()
    total_taps, time_steps = data_matrix.shape

    stats_matrix = None
    location_matrix = None

    for ds_name, dims in ds_info_map.items():
        if ds_name.lower() == "tap_max_min_mean_rms":
            stats_matrix = hdf_obj.select(ds_name).get()
        elif ds_name != main_ds_name and ds_name.lower() != "tap_max_min_mean_rms":
            if len(dims) == 2:
                if (dims[0] == total_taps and dims[1] == 4) or (dims[1] == total_taps and dims[0] == 4):
                    location_matrix = hdf_obj.select(ds_name).get() if dims[0] == total_taps else hdf_obj.select(ds_name).get().T

    hdf_obj.end()
    if os.path.exists(target_hdf_dest):
        try: os.remove(target_hdf_dest)
        except Exception: pass

    mean_vector, rms_vector = None, None
    if stats_matrix is not None:
        if stats_matrix.shape[0] == 4:
            mean_vector = stats_matrix[2, :]
            rms_vector = stats_matrix[3, :]
        elif stats_matrix.shape[1] == 4:
            mean_vector = stats_matrix[:, 2]
            rms_vector = stats_matrix[:, 3]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        
        cursor.execute("INSERT OR IGNORE INTO origin_models (model_name, total_taps) VALUES (?, ?)", (model_name, total_taps))
        cursor.execute("SELECT id FROM origin_models WHERE model_name = ?", (model_name,))
        model_id = cursor.fetchone()[0]
        
        for tap_idx in range(total_taps):
            raw_series = data_matrix[tap_idx, :]

            if mean_vector is not None and rms_vector is not None:
                mean_val = float(mean_vector[tap_idx])
                std_val = float(rms_vector[tap_idx])
            else:
                mean_val = float(np.mean(raw_series))
                std_val = float(np.std(raw_series))

            tap_number, face_code, x_coord, y_coord = tap_idx + 1, 1, None, None
            if location_matrix is not None:
                try:
                    tap_number = int(round(location_matrix[tap_idx, 0]))
                    face_code = int(round(location_matrix[tap_idx, 1]))
                    x_coord = float(location_matrix[tap_idx, 2])
                    y_coord = float(location_matrix[tap_idx, 3])
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