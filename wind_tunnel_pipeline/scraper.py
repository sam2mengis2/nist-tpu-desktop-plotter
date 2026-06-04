import os
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
from pyhdf.SD import SD, SDC

# --- CONFIGURATION ---
DB_PASSWORD = "$Web4now$03"
DOWNLOAD_DIR = "/home/azureuser/wind_tunnel_pipeline"  # Optimized Linux Path

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
    
    with requests.get(zip_url, stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        chunk_count = 0
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    chunk_count += 1
                    if chunk_count % 250 == 0:
                        print(f"   -> Progress: {chunk_count * 4096 / 1024 / 1024:.1f} MB written to disk...", end="\r")
                        
    print(f"\n✅ Download complete: {file_name}")
    return save_path, file_name

# ==========================================
# PHASE 2: FILENAME PARSER
# ==========================================
def parse_nist_filename(file_path):
    """Extracts strictly positional metadata from the NIST HDF file name."""
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

# ==========================================
# PHASE 3: FEATURE MATRIX MATRIX TRANSFORMATION
# ==========================================
def parse_single_hdf4_file(file_path):
    """
    Extracts spatial components and compresses temporal arrays into 6 statistical 
    moments and 32 low-frequency Real FFT coefficients natively.
    """
    hdf = None
    try:
        hdf = SD(file_path, SDC.READ)
        
        # --- 1. EXTRACT SPATIAL COORDINATES ---
        coords_matrix = hdf.select('Flat_Tap_Coordinates')[:] 
        tap_labels_spatial = coords_matrix[0, :].astype(int).astype(str)
        x_vector = coords_matrix[2, :].astype(float)
        y_vector = coords_matrix[3, :].astype(float)
        
        df_spatial = pd.DataFrame({
            "tap_no": tap_labels_spatial,
            "x": x_vector,
            "y": y_vector
        })
        
        # --- 2. EXTRACT TIME-SERIES AND REDUCE FEATURES ---
        pressure_matrix = hdf.select('Time_Series')[:] 
        ts_labels_matrix = hdf.select('Tap_Position_List')[:] 
        
        multiplier = float(hdf.select('Ts_Multiplier')[:][0][0])
        pressure_matrix = pressure_matrix * multiplier
        ts_labels = ts_labels_matrix.flatten().astype(int).astype(str)
        
        series_records = []
        for index, label in enumerate(ts_labels):
            if label not in tap_labels_spatial:
                continue 
                
            timeline_arr = np.array(pressure_matrix[index, :])
            ts_series = pd.Series(timeline_arr)
            
            # Extract high-value statistical parameters
            mean_val = float(np.mean(timeline_arr))
            std_val = float(np.std(timeline_arr))
            min_val = float(np.min(timeline_arr))
            max_val = float(np.max(timeline_arr))
            
            # Handle edge cases for perfectly static values gracefully
            skew_val = float(ts_series.skew()) if not pd.isna(ts_series.skew()) else 0.0
            kurt_val = float(ts_series.kurtosis()) if not pd.isna(ts_series.kurtosis()) else 0.0
            
            # Extract dominant low-frequency macro macro dynamics via Real FFT
            fft_magnitudes = np.abs(np.fft.rfft(timeline_arr))
            top_32_frequencies = fft_magnitudes[:32].tolist()
            
            series_records.append({
                "tap_no": label,
                "mean_cp": mean_val,
                "stddev_cp": std_val,
                "min_cp": min_val,
                "max_cp": max_val,
                "skew_cp": skew_val,
                "kurtosis_cp": kurt_val,
                "fft_magnitude": top_32_frequencies
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
# PHASE 4: RELATIONAL INGESTION ENGINE
# ==========================================
def push_to_supabase(hdf_filename, df_taps, df_pressure):
    """Handles the 3-tier relational upload matching the clean schema."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # --- STEP A: Create Parent Metadata Record ---
        print(f"\n1. Creating Parent Record in Origin_Table...")
        meta = parse_nist_filename(hdf_filename)
        if not meta:
            raise Exception("Failed to parse metadata. Pipeline halted.")
        
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
            meta['angle'],
            data_origin_val
        ))
        
        model_id = cur.fetchone()[0]
        print(f"   -> Parent created with ID: {model_id} at {meta['angle']}°")

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
        print(f"   -> Successfully linked {len(tap_id_map)} spatial taps.")

        # --- STEP C: Upload Optimized Feature Records ---
        print("3. Uploading Pressure Feature Metrics...")
        df_pressure['tap_id'] = df_pressure['tap_no'].map(tap_id_map)
        
        pressure_insert_query = """
            INSERT INTO pressure_series (tap_id, mean_cp, stddev_cp, min_cp, max_cp, skew_cp, kurtosis_cp, fft_magnitude)
            VALUES %s;
        """
        
        pressure_columns = ['tap_id', 'mean_cp', 'stddev_cp', 'min_cp', 'max_cp', 'skew_cp', 'kurtosis_cp', 'fft_magnitude']
        pressure_values = [tuple(x) for x in df_pressure[pressure_columns].to_numpy()]
        
        # Aggressive batch sizes are safe since rows are tiny (no massive raw arrays)
        BATCH_SIZE = 100
        total_rows = len(pressure_values)
        
        for start_idx in range(0, total_rows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch = pressure_values[start_idx:end_idx]
            
            print(f"   -> Transmitting rows {start_idx} to {end_idx} of {total_rows}...   ", end="\r")
            execute_values(cur, pressure_insert_query, batch)
            conn.commit() 
            
        print(f"\n   -> Successfully uploaded all {total_rows} feature records.")
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
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    NIST_PAGE_URL = "https://www.nist.gov/el/mssd/nist-aerodynamic-database/university-western-ontario-data-sets/data-sets-test-number#data-sets-from-phase-1"
    
    try:
        print("--- STARTING AUTOMATED PIPELINE ---")
        
        # 1. Stream master archive
        zip_path, _ = fetch_nist_dataset(NIST_PAGE_URL)
        
        # 2. Extract into safe, self-cleaning staging disk space
        print("\n--- UNPACKING DATASETS ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as master_zip:
                master_zip.extractall(temp_dir)
            
            nested_zips = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(".zip"):
                        nested_zips.append(os.path.join(root, file))
            
            if nested_zips:
                print(f"Found {len(nested_zips)} nested ZIP archives. Unpacking maps...")
                for n_zip in nested_zips:
                    with zipfile.ZipFile(n_zip, 'r') as nested_archive:
                        nested_archive.extractall(os.path.dirname(n_zip))
            
            hdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.upper().endswith(".HDF"):
                        hdf_files.append(os.path.join(root, file))
                        
            if not hdf_files:
                raise Exception("No .HDF files found in the extracted archive.")
            
            print(f"Found {len(hdf_files)} .HDF files for different wind angles.")
            print("\n--- PARSING & RELATIONAL DATABASE INGESTION ---")
            
            # Sort files so they run sequentially from 0 to 360 degrees
            for hdf_path in sorted(hdf_files):
                filename = os.path.basename(hdf_path)
                print(f"\nProcessing target: {filename}")
                
                df_taps, df_pressure = parse_single_hdf4_file(hdf_path) 
                
                if df_taps is None or df_pressure is None:
                    print(f"⚠️ Warning: {filename} data parsing failed. Skipping.")
                    continue
                
                push_to_supabase(hdf_path, df_taps, df_pressure)
                
        print("\n✅ PIPELINE COMPLETE! Temporary storage flushed. Database fully populated.")
        
    except Exception as e:
        print(f"\n❌ Pipeline crashed: {e}")