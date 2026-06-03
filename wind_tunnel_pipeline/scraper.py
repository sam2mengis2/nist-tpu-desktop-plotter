import os
import requests
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from itertools import islice
import io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
from scipy.io import loadmat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tempfile
import zipfile
import os
import pandas as pd
# --- CONFIGURATION ---
DB_PASSWORD = "$Web4now$03"
DOWNLOAD_DIR = "/home/azureuser/wind_tunnel_pipeline"

def get_db_connection():
    return psycopg2.connect(
        host="aws-1-us-east-1.pooler.supabase.com",
        port="5432",
        database="postgres",
        user="postgres.nanddzdspaucmwlyoyoc",
        password=DB_PASSWORD
    )

# ==========================================
# PHASE 1: SCRAPE & DOWNLOAD
# ==========================================
def fetch_nist_dataset(page_url):
    """Finds the dataset link and streams it to disk with clear chunk telemetry."""
    print(f"Scraping {page_url}...")
    response = requests.get(page_url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the zip anchor tag
    link = soup.find('a', href=lambda href: href and '.zip' in href.lower())
    if not link:
        raise Exception(f"Could not find any .zip download links on {page_url}")
        
    zip_url = link['href']
    if zip_url.startswith('/'):
        zip_url = "https://www.nist.gov" + zip_url

    file_name = zip_url.split('/')[-1].split('?')[0]
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    save_path = os.path.join(DOWNLOAD_DIR, file_name)

    print(f"\n[TELEMETRY] Target Download URL: {zip_url}")
    print(f"Streaming to disk...")
    
    # Use a explicit tuple: (connect_timeout, read_timeout)
    with requests.get(zip_url, stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        
        chunk_count = 0
        with open(save_path, 'wb') as f:
            # Drop chunk size to 4KB to catch slow connections faster
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    chunk_count += 1
                    
                    # Print an update every 50 chunks (~200 KB) so you see signs of life
                    if chunk_count % 50 == 0:
                        print(f"   -> Progress: {chunk_count * 4096 / 1024:.0f} KB written to disk...", end="\r")
                        
    print(f"\n✅ Download complete: {file_name}")
    return save_path, file_name

# ==========================================
# FILENAME PARSER
# ==========================================
def parse_nist_filename(file_path):
    """
    Extracts strictly positional metadata from the NIST HDF file name.
    Example: 'ADW100o100S048a1800.HDF'
    """
    filename = os.path.basename(file_path)
    base_name = filename.upper().replace('.HDF', '')

    try:
        exposure_map = {'O': 'Open', 'S': 'Suburban', 'W': 'WERFL'}
        leak_map = {'D': 'Distributed', 'S': 'Small', 'L': 'Large', 'N': 'None', 'B': 'Basic'}

        return {
            "roof_slope": int(base_name[3:6]),
            "exposure_val": exposure_map.get(base_name[6], 'Unknown'),
            "model_scale": int(base_name[7:10]),
            "leakage": leak_map.get(base_name[10], 'Unknown'),
            "eave_height": int(base_name[11:14]),
            "angle": float(base_name[15:19]) / 10.0
        }
    except (IndexError, ValueError) as e:
        print(f"Filename parsing error on {filename}: {e}")
        return None
    
def parse_single_hdf4_file(file_path):
    """
    Extracts spatial layout and time-series arrays, automatically handling 
    HDF4 matrix transpositions, multiplier conversions, and ghost tap filtering.
    """
    hdf = None
    try:
        hdf = SD(file_path, SDC.READ)
        
        # --- 1. EXTRACT SPATIAL COORDINATES ---
        coords_matrix = hdf.select('Flat_Tap_Coordinates')[:] # Shape: (4, 515)
        
        # Because it is 4x515, we slice specific rows
        # Row 0: Tap numbers, Row 2: X, Row 3: Y
        tap_labels_spatial = coords_matrix[0, :].astype(int).astype(str)
        x_vector = coords_matrix[2, :].astype(float)
        y_vector = coords_matrix[3, :].astype(float)
        
        df_spatial = pd.DataFrame({
            "tap_no": tap_labels_spatial,
            "x": x_vector,
            "y": y_vector
        })
        
        # --- 2. EXTRACT TIME-SERIES AND CONVERT ---
        # Get the actual pressure data and the specific labels for all 688 channels
        pressure_matrix = hdf.select('Time_Series')[:] # Shape: (688, 49792)
        ts_labels_matrix = hdf.select('Tap_Position_List')[:] # Shape: (688, 1)
        
        # Extract the conversion multiplier (e.g., 0.001) and apply it to the matrix!
        multiplier = float(hdf.select('Ts_Multiplier')[:][0][0])
        pressure_matrix = pressure_matrix * multiplier
        
        # Flatten the 688x1 label matrix into a simple list of strings
        ts_labels = ts_labels_matrix.flatten().astype(int).astype(str)
        
        # Compress the data alongside fast statistics
        series_records = []
        for index, label in enumerate(ts_labels):
            
            # TRAP 3 FIX: Only process this time-series if we have its X/Y coordinates!
            if label not in tap_labels_spatial:
                continue 
                
            timeline = pressure_matrix[index, :].tolist()
            series_records.append({
                "tap_no": label,
                "mean_cp": float(np.mean(timeline)),
                "stddev_cp": float(np.std(timeline)),
                "cp_time_series": timeline
            })
            
        df_series = pd.DataFrame(series_records)
        return df_spatial, df_series
        
    except Exception as error:
        print(f"Failed parsing HDF4 structure in {os.path.basename(file_path)}: {error}")
        return None, None
    finally:
        if hdf:
            hdf.end()

# ==========================================
# PHASE 2: DATABASE INGESTION
# ==========================================
def push_to_supabase(hdf_filename, df_taps, df_pressure):
    """Handles the 3-tier relational upload to Supabase matching the exact ERD layout."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # --- STEP A: Parse Metadata & Create Parent Record ---
        print(f"\n1. Parsing filename and creating Parent Record in NIST_Table...")
        
        meta = parse_nist_filename(hdf_filename)
        if not meta:
            raise Exception("Failed to parse metadata. Pipeline halted.")
        
        # 💡 THE FIX: Safely catch whichever key name you used in your filename parser!
        actual_angle = meta.get('wind_angle', meta.get('angle', 0.0))
        data_origin_val = 'NIST'

        cur.execute("""
            INSERT INTO "Origin_Table" (roof_slope, exposure_val, model_scale, leakage, eave_height, angle, data_origin)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            meta['roof_slope'], 
            meta['exposure_val'], 
            meta['model_scale'], 
            meta['leakage'], 
            meta['eave_height'],
            actual_angle,  # Drops cleanly into your 'angle' column
            data_origin_val
        ))
        
        model_id = cur.fetchone()[0]
        print(f"   -> Model created with ID: {model_id} at {actual_angle}°")

        # --- STEP B: Upload Spatial Taps ---
        print("2. Uploading Spatial Taps...")
        
        df_taps['model_id'] = model_id
        
        tap_insert_query = """
            INSERT INTO taps (model_id, tap_no, x, y) 
            VALUES %s
            RETURNING id, tap_no;
        """
        tap_values = [tuple(x) for x in df_taps[['model_id', 'tap_no', 'x', 'y']].to_numpy()]
        
        inserted_taps = execute_values(cur, tap_insert_query, tap_values, fetch=True)
        tap_id_map = {row[1]: row[0] for row in inserted_taps}
        print(f"   -> Successfully linked {len(tap_id_map)} taps.")

        # --- STEP C: Upload Time-Series Arrays (BATCHED VERSION) ---
        print("3. Uploading Pressure Time-Series Arrays...")
        
        # 1. Map the newly generated tap database IDs to our pressure rows
        df_pressure['tap_id'] = df_pressure['tap_no'].map(tap_id_map)
        
        pressure_insert_query = """
            INSERT INTO pressure_series (tap_id, mean_cp, stddev_cp, cp_time_series)
            VALUES %s;
        """
        
        # Match columns exactly to your pressure_series table layout
        pressure_columns = ['tap_id', 'mean_cp', 'stddev_cp', 'cp_time_series']
        pressure_values = [tuple(x) for x in df_pressure[pressure_columns].to_numpy()]
        
        # 💡 THE BATCHING ENGINE: Process 25 taps at a time
        BATCH_SIZE = 25
        total_rows = len(pressure_values)
        
        print(f"   -> Splitting {total_rows} rows into batches of {BATCH_SIZE}...")
        
        for start_idx in range(0, total_rows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch = pressure_values[start_idx:end_idx]
            
            # Print a live updating counter in the terminal
            print(f"   -> Uploading rows {start_idx} to {end_idx} of {total_rows}...   ", end="\r")
            
            # Fire just this batch over the network
            execute_values(cur, pressure_insert_query, batch)
            
            # Optional: Commit each batch dynamically so you can watch them populate in Supabase live!
            conn.commit() 
            
        print(f"\n   -> Successfully uploaded all {total_rows} time-series records.")
        print(f"✅ Clean run for {os.path.basename(hdf_filename)} committed completely!")

    except Exception as e:
        print(f"Database Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()
# ==========================================
# PHASE 3: MASTER EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    import tempfile
    import zipfile
    import os
    import pandas as pd
    
    # Your target database link
    NIST_PAGE_URL = "https://www.nist.gov/el/mssd/nist-aerodynamic-database/university-western-ontario-data-sets/data-sets-test-number#data-sets-from-phase-1"
    
    try:
        print("--- STARTING AUTOMATED PIPELINE ---")
        
        # 1. Download the master archive
        zip_path, _ = fetch_nist_dataset(NIST_PAGE_URL)
        
        # 2. Create a self-cleaning temporary staging area
        print("\n--- UNPACKING DATASETS ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Unzip the master file to the temporary hard drive space
            with zipfile.ZipFile(zip_path, 'r') as master_zip:
                master_zip.extractall(temp_dir)
            
            # Unpack all nested zip archives discovered inside
            nested_zips = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(".zip"):
                        nested_zips.append(os.path.join(root, file))
            
            if nested_zips:
                print(f"Found {len(nested_zips)} nested ZIP files. Unpacking...")
                for n_zip in nested_zips:
                    with zipfile.ZipFile(n_zip, 'r') as nested_archive:
                        nested_archive.extractall(os.path.dirname(n_zip))
            
            # Gather all unzipped .HDF target files
            hdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.upper().endswith(".HDF"):
                        hdf_files.append(os.path.join(root, file))
                        
            if not hdf_files:
                raise Exception("No .HDF files found in the extracted archive.")
            
            print(f"Found {len(hdf_files)} .HDF files for different wind angles.")
            
            # 3. Process data and upload file-by-file
            print("\n--- PARSING & RELATIONAL DATABASE INGESTION ---")
            
            for hdf_path in hdf_files:
                filename = os.path.basename(hdf_path)
                print(f"\nProcessing target: {filename}")
                
                # Extract arrays using the pyhdf matrix script we built
                df_taps, df_pressure = parse_single_hdf4_file(hdf_path) 
                
                # --- THE SAFETY NET ---
                # Skip if a file is unreadable, preventing an unexpected crash midway through
                if df_taps is None or df_pressure is None:
                    print(f"⚠️ Warning: {filename} data parsing failed. Skipping entry.")
                    continue
                
                # ✅ SCHEMA ALIGNMENT FIX: 
                # Upload directly to Supabase file-by-file. This generates a unique Parent 
                # Model row for each angle, preserving your exact relational architecture.
                push_to_supabase(hdf_path, df_taps, df_pressure)
                
        print("\n✅ PIPELINE COMPLETE! Temporary storage flushed. Database populated.")
        
    except Exception as e:
        print(f"\n❌ Pipeline crashed: {e}")