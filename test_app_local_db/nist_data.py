import os
import io
import zipfile
import tempfile
import sqlite3
from pyhdf.SD import SD, SDC
import numpy as np

DB_FILE = "local_wind_data.db"
SAMPLE_DATA_DIR = "./sample_data"

def process_hdf_file(hdf_path, hdf_name, model_id, cursor):
    """
    Parses an open HDF4 file. Maps the 2D 'flat_tap' layout with the 
    pre-calculated 'Tap_Max_Min_Mean_RMS' table directly.
    """
    print(f"🚀 Processing 2D summary matrix for: {hdf_name}")
    
    hdf_interface = SD(hdf_path, SDC.READ)
    
    try:
        available_datasets = list(hdf_interface.datasets().keys())
        
        tap_2d_key = 'Flat_Tap_Coordinates'
        stats_key = 'Tap_Max_Min_Mean_RMS'

        if not (tap_2d_key and stats_key):
            print(f"❌ Missing critical keys inside HDF structure. Available: {available_datasets}")
            return

        tap_2d_coordinates = hdf_interface.select(tap_2d_key).get()
        summary_stats_matrix = hdf_interface.select(stats_key).get()

        # Handle Column-Major Transpositions
        if tap_2d_coordinates.shape[0] == 4:
            tap_2d_coordinates = tap_2d_coordinates.T
            
        if summary_stats_matrix.shape[0] == 5:
            summary_stats_matrix = summary_stats_matrix.T

        # Map Tap Numbers directly to stats
        # Row Layout: [0: Tap Number, 1: Max, 2: Min, 3: Mean, 4: Fluctuating RMS (Std Dev)]
        stats_lookup = {}
        for row in summary_stats_matrix:
            tap_num = int(row[0])
            mean_cp = float(row[3])
            std_dev_cp = float(row[4])  # Taken directly from the file!
            
            stats_lookup[tap_num] = (mean_cp, std_dev_cp)

        bulk_taps_buffer = []

        # Loop over the 2D coordinates matrix
        for row in tap_2d_coordinates:
            tap_num = int(row[0])
            face_no = int(row[1])
            x_coord = float(row[2])
            y_coord = float(row[3])
            
            # Skip internal taps and respect the 6-face rule
            if face_no == 0 or face_no > 6:
                continue 
                
            if tap_num in stats_lookup:
                mean_cp, std_cp = stats_lookup[tap_num]
            else:
                mean_cp, std_cp = 0.0, 0.0

            bulk_taps_buffer.append((
                model_id,
                face_no,
                x_coord,
                y_coord,
                mean_cp,
                std_cp
            ))

        # Write clean data directly to SQLite
        cursor.executemany("""
            INSERT INTO taps (model_id, face_no, x_coordinate, y_coordinate, mean_pressure, std_dev_pressure)
            VALUES (?, ?, ?, ?, ?, ?);
        """, bulk_taps_buffer)
        
        print(f"   ✅ Successfully loaded {len(bulk_taps_buffer)} individual taps from 2D mapping.")

    finally:
        hdf_interface.end()


def process_nist_archive(zip_filename):
    zip_path = os.path.join(SAMPLE_DATA_DIR, zip_filename)
    if not os.path.exists(zip_path):
        zip_path = zip_filename if os.path.exists(zip_filename) else zip_path
        if not os.path.exists(zip_path):
            print(f"❌ Cannot find zip file at: {zip_path}")
            return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    try:
        with zipfile.ZipFile(zip_path, 'r') as outer_archive:
            for item in outer_archive.namelist():
                if item.lower().endswith('.zip'):
                    print(f"\n📂 Opening nested zip archive: {item}")
                    with outer_archive.open(item) as inner_zip_bytes:
                        nested_zip_stream = io.BytesIO(inner_zip_bytes.read())
                        with zipfile.ZipFile(nested_zip_stream) as inner_archive:
                            for hdf_item in [f for f in inner_archive.namelist() if f.lower().endswith('.hdf')]:
                                cursor.execute("INSERT OR IGNORE INTO origin_models (filename) VALUES (?);", (hdf_item,))
                                cursor.execute("SELECT model_id FROM origin_models WHERE filename = ?;", (hdf_item,))
                                model_id = cursor.fetchone()[0]
                                
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    temp_hdf_path = inner_archive.extract(hdf_item, temp_dir)
                                    process_hdf_file(temp_hdf_path, hdf_item, model_id, cursor)

                elif item.lower().endswith('.hdf'):
                    cursor.execute("INSERT OR IGNORE INTO origin_models (filename) VALUES (?);", (item,))
                    cursor.execute("SELECT model_id FROM origin_models WHERE filename = ?;", (item,))
                    model_id = cursor.fetchone()[0]
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_hdf_path = outer_archive.extract(item, temp_dir)
                        process_hdf_file(temp_hdf_path, item, model_id, cursor)

        conn.commit()
        print("\n🏆 DATABASE PIPELINE SYNCHRONIZATION COMPLETE.")

    except Exception as pipeline_error:
        conn.rollback()
        print(f"💥 Pipeline Execution Halted: {pipeline_error}")
    finally:
        conn.close()

if __name__ == "__main__":
    process_nist_archive("ou1.zip")