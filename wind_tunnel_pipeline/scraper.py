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
# PHASE 1: LINK HARVESTING & STREAMING
# ==========================================
def get_all_nist_zip_urls(page_url):
    """Scrapes the page and gathers every unique .zip download link available in the data tables."""
    print(f"Scraping page index for all datasets: {page_url}...")
    response = requests.get(page_url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find every anchor tag that references a .zip archive
    anchors = soup.find_all('a', href=lambda href: href and '.zip' in href.lower())
    
    zip_urls = []
    for anchor in anchors:
        url = anchor['href']
        if url.startswith('/'):
            url = "https://www.nist.gov" + url
        if url not in zip_urls:
            zip_urls.append(url)
            
    print(f"🎯 Successfully harvested {len(zip_urls)} unique master dataset links from the page tables.")
    return zip_urls

def download_single_zip(zip_url):
    """Streams a single target zip file to the local SSD cache layer."""
    file_name = zip_url.split('/')[-1].split('?')[0]
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    save_path = os.path.join(DOWNLOAD_DIR, file_name)

    print(f"\n[STREAMING] Downloading: {file_name}")
    with requests.get(zip_url, stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        chunk_count = 0
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    chunk_count += 1
                    if chunk_count % 500 == 0:
                        print(f"   -> Progress: {chunk_count * 4096 / 1024 / 1024:.1f} MB compiled...", end="\r")
                        
    print(f"\n downloaded: {file_name}")
    return save_path

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
# PHASE 3: FEATURE MATRIX TRANSFORMATION
# ==========================================
def parse_single_hdf4_file(file_path):
    """Extracts spatial layout and compresses temporal arrays into statistical & FFT features."""
    hdf = None
    try:
        hdf = SD(file_path, SDC.READ)
        
        # --- 1. SPATIAL DATA FRAMING ---
        coords_matrix = hdf.select('Flat_Tap_Coordinates')[:] 
        tap_labels_spatial = coords_matrix[0, :].astype(int).astype(str)
        x_vector = coords_matrix[2, :].astype(float)
        y_vector = coords_matrix[3, :].astype(float)
        
        df_spatial = pd.DataFrame({"tap_no": tap_labels_spatial, "x": x_vector, "y": y_vector})
        
        # --- 2. TIME-SERIES FEATURE REDUCTION ---
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
            
            mean_val = float(np.mean(timeline_arr))
            std_val = float(np.std(timeline_arr))
            min_val = float(np.min(timeline_arr))
            max_val = float(np.max(timeline_arr))
            skew_val = float(ts_series.skew()) if not pd.isna(ts_series.skew()) else 0.0
            kurt_val = float(ts_series.kurtosis()) if not pd.isna(ts_series.kurtosis()) else 0.0
            
            fft_magnitudes = np.abs(np.fft.rfft(timeline_arr))
            top_32_frequencies = fft_magnitudes[:32].tolist()
            
            series_records.append({
                "tap_no": label, "mean_cp": mean_val, "stddev_cp": std_val,
                "min_cp": min_val, "max_cp": max_val, "skew_cp": skew_val,
                "kurtosis_cp": kurt_val, "fft_magnitude": top_32_frequencies
            })
            
        return df_spatial, pd.DataFrame(series_records)
    except Exception as error:
        print(f"Failed parsing HDF4 structure in {os.path.basename(file_path)}: {error}")
        return None, None
    finally:
        if hdf: hdf.end()

# ==========================================
# PHASE 4: DB INGESTION
# ==========================================
def push_to_supabase(hdf_filename, df_taps, df_pressure):
    """Inserts processed layout and feature tables into Supabase relational structure."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        meta = parse_nist_filename(hdf_filename)
        if not meta: raise Exception("Failed to parse metadata.")
        
        # Parent Insertion
        cur.execute("""
            INSERT INTO "Origin_Table" (roof_slope, exposure_val, model_scale, leakage, eave_height, angle, data_origin)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """, (meta['roof_slope'], meta['exposure_val'], meta['model_scale'], meta['leakage'], meta['eave_height'], meta['angle'], 'NIST'))
        
        model_id = cur.fetchone()[0]

        # Spatial Taps Insertion
        df_taps['model_id'] = model_id
        tap_insert_query = "INSERT INTO taps (model_id, tap_no, x, y) VALUES %s RETURNING id, tap_no;"
        tap_values = [tuple(x) for x in df_taps[['model_id', 'tap_no', 'x', 'y']].to_numpy()]
        inserted_taps = execute_values(cur, tap_insert_query, tap_values, fetch=True)
        tap_id_map = {row[1]: row[0] for row in inserted_taps}

        # Features Vector Insertion
        df_pressure['tap_id'] = df_pressure['tap_no'].map(tap_id_map)
        pressure_insert_query = """
            INSERT INTO pressure_series (tap_id, mean_cp, stddev_cp, min_cp, max_cp, skew_cp, kurtosis_cp, fft_magnitude)
            VALUES %s;
        """
        pressure_columns = ['tap_id', 'mean_cp', 'stddev_cp', 'min_cp', 'max_cp', 'skew_cp', 'kurtosis_cp', 'fft_magnitude']
        pressure_values = [tuple(x) for x in df_pressure[pressure_columns].to_numpy()]
        
        BATCH_SIZE = 100
        for start_idx in range(0, len(pressure_values), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(pressure_values))
            execute_values(cur, pressure_insert_query, pressure_values[start_idx:end_idx])
            conn.commit() 
            
    except Exception as e:
        print(f"Database Error: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

# ==========================================
# MASTER AUTOMATION EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    NIST_PAGE_URL = "https://www.nist.gov/el/mssd/nist-aerodynamic-database/university-western-ontario-data-sets/data-sets-test-number#data-sets-from-phase-1"
    
    try:
        print("--- STARTING MASTER DISTRIBUTED AUTOMATION PIPELINE ---")
        
        # 1. Harvest every master dataset link visible on the webpage
        master_zip_urls = get_all_nist_zip_urls(NIST_PAGE_URL)
        
        # 2. Sequential Processing Loop
        for index, target_url in enumerate(master_zip_urls, 1):
            filename_zip = target_url.split('/')[-1].split('?')[0]
            
            # 💡 THE GUARD CLAUSE: Instantly skip ee1.zip processing
            if "ee1" in filename_zip.lower():
                print(f"\n⏭️ Skipping batch entry {index} ({filename_zip}) because it is already ingested.")
                continue
                
            print(f"\n==================================================================")
            print(f"PROCESSING BATCH ENTRY {index} OF {len(master_zip_urls)}")
            print(f"Target: {filename_zip}")
            print(f"==================================================================")
            
            local_zip_path = None
            try:
                # Step A: Download the single master zip archive
                local_zip_path = download_single_zip(target_url)
                
                # Step B: Unpack, calculate features, and stream to Supabase
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(local_zip_path, 'r') as master_zip:
                        master_zip.extractall(temp_dir)
                    
                    nested_zips = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(".zip"):
                                nested_zips.append(os.path.join(root, file))
                    
                    if nested_zips:
                        for n_zip in nested_zips:
                            with zipfile.ZipFile(n_zip, 'r') as nested_archive:
                                nested_archive.extractall(os.path.dirname(n_zip))
                    
                    hdf_files = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.upper().endswith(".HDF"):
                                hdf_files.append(os.path.join(root, file))
                    
                    print(f" -> Found {len(hdf_files)} aerodynamic files inside this dataset archive.")
                    
                    for hdf_path in sorted(hdf_files):
                        filename = os.path.basename(hdf_path)
                        print(f"    -> Parsing matrices: {filename}...", end="\r")
                        
                        df_taps, df_pressure = parse_single_hdf4_file(hdf_path)
                        if df_taps is None or df_pressure is None:
                            continue
                            
                        push_to_supabase(hdf_path, df_taps, df_pressure)
                
                print(f"✅ Master dataset {index} successfully uploaded completely.")
                
            except Exception as item_error:
                print(f"⚠️ Error running dataset bundle {target_url}: {item_error}")
                
            finally:
                if local_zip_path and os.path.exists(local_zip_path):
                    os.remove(local_zip_path)
                    print(f"🧹 Local cache cleared: Removed {os.path.basename(local_zip_path)} from server disk.")
                    
        print("\n🏆 GLOBAL DATA EXTRACTION COMPLETE! Entire NIST Phase 1 table ingested.")
        
    except Exception as e:
        print(f"\n❌ Global Pipeline Failure: {e}")