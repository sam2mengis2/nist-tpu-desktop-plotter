import os
import sqlite3
import numpy as np
from scipy.io import loadmat

DB_FILE = "local_wind_data.db"
TPU_DATA_DIR = "./tpu_data"  # Drop your .mat files in this folder

def process_tpu_mat_file(file_path):
    filename = os.path.basename(file_path)
    print(f"\n🚀 Opening TPU MATLAB Workspace: {filename}")
    
    # 1. Load the .mat file into a Python dictionary
    try:
        mat_data = loadmat(file_path)
    except Exception as e:
        print(f"💥 Failed to read MATLAB file: {e}")
        return

    # 2. Extract the core data matrices using the exact variable names
    # Note: loadmat preserves the exact casing from the MATLAB workspace
    try:
        wind_matrix = mat_data['Wind_pressure_coefficients']
        location_matrix = mat_data['Location_of_measured_points']
    except KeyError as e:
        print(f"❌ Missing critical variable key in workspace: {e}")
        print(f"Available keys inside this file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        return

    print(f"📊 Array Diagnostics -> Time-Series Shape: {wind_matrix.shape} | Location Matrix Shape: {location_matrix.shape}")

    # 3. CRUNCH STATISTICS (Rows = Time, Columns = Taps)
    # Compress the time axis (axis=0) down to 1D vectors of length (Number of Taps)
    print("🔢 Calculating Mean and Standard Deviation over the timeline matrix...")
    mean_cp_array = np.mean(wind_matrix, axis=0)
    std_cp_array = np.std(wind_matrix, axis=0)

    # 4. TRANPOSE COORDINATES MATRIX
    # Change layout from (4, N) to (N, 4) to loop through rows as individual taps
    clean_coords = location_matrix.T

    # 5. OPEN DATABASE TRANSACTION
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    try:
        # Log the file arrival into origin_models
        cursor.execute("INSERT OR IGNORE INTO origin_models (filename) VALUES (?);", (filename,))
        cursor.execute("SELECT model_id FROM origin_models WHERE filename = ?;", (filename,))
        model_id = cursor.fetchone()[0]

        bulk_taps_buffer = []

        # 6. MESH COORDINATES AND STATS TOGETHER
        # Row layout spec: [0: x_coordinate, 1: y_coordinate, 2: point_number, 3: surface_number]
        for row in clean_coords:
            x_coord = float(row[0])
            y_coord = float(row[1])
            tap_num = int(row[2])
            face_no = int(row[3])

            # ⚠️ CRITICAL BRIDGE: MATLAB uses 1-based indexing for arrays.
            # Point 5 corresponds to column index 4 in Python's 0-indexed arrays.
            python_array_idx = tap_num - 1

            # Safeguard against unexpected index boundaries
            if python_array_idx >= len(mean_cp_array):
                continue

            mean_cp = float(mean_cp_array[python_array_idx])
            std_cp = float(std_cp_array[python_array_idx])

            bulk_taps_buffer.append((
                model_id,
                face_no,
                x_coord,
                y_coord,
                mean_cp,
                std_cp
            ))

        # 7. BULK DUMP TO SQLITE
        cursor.executemany("""
            INSERT INTO taps (model_id, face_no, x_coordinate, y_coordinate, mean_pressure, std_dev_pressure)
            VALUES (?, ?, ?, ?, ?, ?);
        """, bulk_taps_buffer)

        conn.commit()
        print(f"   ✅ Successfully loaded {len(bulk_taps_buffer)} taps into your local database.")

    except Exception as pipeline_error:
        conn.rollback()
        print(f"💥 Database Injection Halted: {pipeline_error}")
    finally:
        conn.close()


def run_tpu_pipeline():
    """Loops through a targeted data directory to process all available .mat files."""
    if not os.path.exists(TPU_DATA_DIR):
        os.makedirs(TPU_DATA_DIR)
        print(f"📁 Created an empty folder at '{TPU_DATA_DIR}'. Drop your TPU .mat files inside it and run again!")
        return

    mat_files = [f for f in os.listdir(TPU_DATA_DIR) if f.lower().endswith('.mat')]
    
    if not mat_files:
        print(f"ℹ️ No .mat files found inside '{TPU_DATA_DIR}' directory.")
        return

    print(f"📦 Discovered {len(mat_files)} MATLAB dataset(s) for ingestion.")
    for mat_file in mat_files:
        full_path = os.path.join(TPU_DATA_DIR, mat_file)
        process_tpu_mat_file(full_path)
    
    print("\n🏆 PIPELINE RUN COMPLETE.")


if __name__ == "__main__":
    run_tpu_pipeline()