# -*- coding: utf-8 -*-
"""
tpu_scraper.py - Main Control App with Full Matrix CSV Generation & Contour Layouts
"""

import os
import re
import ssl
import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import sqlite3
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from TPU_parser import (
    populate_database_from_mat, 
    initialize_local_database, 
    clear_session_data, 
    DB_PATH,
    DROP_FOLDER as TEMP_DROP_PATH
)

PORTAL_CATALOG = {
    "High-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm",
    "Low-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/Homepage/homepageLDF.htm",
    "Low-Rise Buildings with Eaves": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/eaves/Homepage/homepageEDF.htm",
    "Low-Rise Buildings (Non-Isolated)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/grouplowrise/mainpage.html"
}


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT@SECLEVEL=1:HIGH:!DH:!EDH")
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, ssl_context=ctx, **pool_kwargs)


def create_legacy_session():
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def find_dropdown_recursively(session, url, headers, visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return None, None
    visited.add(url)

    try:
        response = session.get(url, headers=headers, timeout=12, verify=False)
        response.raise_for_status()
    except Exception:
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    select_tag = soup.find("select", attrs={"name": "mysel"}) or soup.find("select", attrs={"name": "urlsel"})
    if select_tag:
        return select_tag, url

    for frame in soup.find_all(["frame", "iframe"]):
        src = frame.get("src")
        if src:
            tag, found_url = find_dropdown_recursively(session, urljoin(url, src), headers, visited)
            if tag:
                return tag, found_url
    return None, None


def parse_dropdown_from_url(session, url, headers):
    select_tag, actual_content_url = find_dropdown_recursively(session, url, headers)
    if not select_tag:
        return {}, None

    options_map = {}
    for option in select_tag.find_all("option"):
        value = option.get("value")
        text = option.get_text().strip()
        if value == "def" or not value or "please select" in text.lower():
            continue
        options_map[text] = urljoin(actual_content_url, value)
    return options_map, actual_content_url


def run_data_session_dashboard(model_id, wind_angle):
    if plt is None:
        print("\n⚠️ Note: 'matplotlib' not detected in your current environment.")

    while True:
        print(f"\n=========================================")
        print(f"📊 ACTIVE TPU LIVE ANALYSIS DASHBOARD")
        print(f"=========================================")
        print(" [1] Export FULL Tabular Time-Series to CSV (All Taps)")
        print(" [2] Export Statistical Tap Grid to CSV (X/Y: Summaries per Tap Location)")
        print(" [3] Render Spatial Surface Contour Plot")
        print(" [b] Return to Scraper Wizard / Load Another Model")
        print(" [q] Terminate Session & Securely Wipe Database")
        
        choice = input("\n👉 Select a dashboard capability index: ").strip().lower()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if choice == 'q':
            conn.close()
            clear_session_data()
            print("Goodbye!")
            sys.exit(0)

        elif choice == 'b':
            conn.close()
            return "back"

        elif choice == '1':
            print("⏳ Extracting comprehensive time-history matrix from database cache...")
            try:
                cursor.execute("""
                    SELECT tap_number, time_history 
                    FROM tap_measurements 
                    WHERE model_id=? AND wind_angle=? 
                    ORDER BY tap_number
                """, (model_id, wind_angle))
                
                rows = cursor.fetchall()
                if not rows:
                    print("❌ No sensor rows located in active session database cache.")
                    continue
                
                csv_data = {}
                max_len = 0
                
                for row in rows:
                    tap_num = row[0]
                    time_series = np.frombuffer(row[1], dtype=np.float32)
                    csv_data[f"Tap_{tap_num}_Cp"] = time_series
                    if len(time_series) > max_len:
                        max_len = len(time_series)

                output_csv = os.path.join(TEMP_DROP_PATH, f"tpu_tabular_time_history_angle_{wind_angle}.csv")
                print(f"✍️ Compiling {max_len} time steps into CSV. Writing to disk...")
                
                with open(output_csv, "w") as f:
                    headers = ["TimeStep"] + list(csv_data.keys())
                    f.write(",".join(headers) + "\n")
                    arrays = [csv_data[k] for k in csv_data.keys()]
                    
                    for step in range(max_len):
                        vals = [str(step)] + [str(arr[step]) if step < len(arr) else "" for arr in arrays]
                        f.write(",".join(vals) + "\n")
                
                print(f"🎉 Complete Time-Series CSV successfully exported to:\n📁 {output_csv}")
            except Exception as e:
                print(f"❌ Tabular extraction failure: {e}")
            finally:
                conn.close()

        elif choice == '2':
            try:
                # Includes raw face integer directly in statistical spreadsheet export
                cursor.execute("SELECT tap_number, mean_cp, std_cp, face FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number", (model_id, wind_angle))
                rows = cursor.fetchall()
                
                output_csv = os.path.join(TEMP_DROP_PATH, f"tpu_spatial_summary_grid_angle_{wind_angle}.csv")
                with open(output_csv, "w") as f:
                    f.write("Tap_Number,Mean_Cp,Std_Dev_Cp,Raw_Face_Code\n")
                    for row in rows:
                        f.write(f"{row[0]},{row[1]},{row[2]},{row[3] if row[3] is not None else ''}\n")
                print(f"🎉 Spatial Grid CSV successfully exported to:\n📁 {output_csv}")
            except Exception as e:
                print(f"❌ Statistical mapping failure: {e}")
            finally:
                conn.close()

        elif choice == '3':
            if plt is None:
                print("❌ Plotting unavailable. Run 'pip install matplotlib'.")
                conn.close()
                continue
            
            try:
                cursor.execute("SELECT mean_cp FROM tap_measurements WHERE model_id=? AND wind_angle=? ORDER BY tap_number")
                means = [r[0] for r in cursor.fetchall()]
                
                if not means:
                    print("❌ Empty data sequence. Cannot construct distribution map plots.")
                    conn.close()
                    continue

                total = len(means)
                cols = int(np.ceil(np.sqrt(total)))
                rows = int(np.ceil(total / cols))
                
                padded_means = np.zeros(rows * cols)
                padded_means[:total] = means
                grid_z = padded_means.reshape((rows, cols))

                plt.figure(figsize=(8, 6))
                plt.title(f"TPU Wind Pressure Distribution Contour Map (Angle: {wind_angle}°)")
                contour = plt.contourf(grid_z, cmap='RdBu_r', levels=15)
                plt.colorbar(contour, label="Mean Pressure Coefficient ($C_p$)")
                plt.xlabel("Horizontal Grid Space Axis (X)")
                plt.ylabel("Vertical Grid Space Axis (Y)")
                
                print("🎨 Displaying plot canvas window...")
                plt.show()
            except Exception as e:
                print(f"❌ Visualization renderer fault: {e}")
            finally:
                conn.close()


def process_final_data_page(session, page_url, headers):
    try:
        res = session.get(page_url, headers=headers, timeout=12, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to parse data page: {e}")
        return

    data_file_row, direction_row = None, None
    for row in soup.find_all("tr"):
        a_tags = row.find_all("a", href=True)
        if len([a for a in a_tags if ".mat" in a["href"].lower()]) > 1:
            data_file_row = row
            break

    if data_file_row:
        current_row = data_file_row.find_previous_sibling("tr")
        while current_row:
            if "Wind direction" in current_row.get_text():
                direction_row = current_row
                break
            current_row = current_row.find_previous_sibling("tr")

    if not direction_row or not data_file_row:
        print("❌ Error parsing the wind angle matrix grid layout.")
        return

    angles = []
    for td in direction_row.find_all(["td", "th"]):
        match = re.search(r"(\d+)", td.get_text().strip())
        if match:
            angles.append(int(match.group(1)))

    links = [urljoin(page_url, td.find("a")["href"]) for td in data_file_row.find_all("td") if td.find("a") and ".mat" in td.find("a")["href"].lower()]
    angle_map = dict(zip(angles, links[:len(angles)]))
    available_angles = sorted(list(angle_map.keys()))

    while True:
        print(f"\n📊 Reached Final Model Configuration Grid.")
        print(f"Available Wind Angles: {available_angles}")
        angle_choice = input("👉 Enter target Wind Angle to process into DB (or 'b' to go back): ").strip().lower()

        if angle_choice == 'b':
            return "back"

        if not angle_choice.isdigit() or int(angle_choice) not in angle_map:
            print("❌ Invalid wind angle choice selection.")
            continue

        target_angle = int(angle_choice)
        download_url = angle_map[target_angle]
        
        original_file_name = download_url.split('/')[-1]
        if not original_file_name.endswith('.mat'):
            original_file_name = f"tpu_angle_{target_angle}.mat"

        local_path = os.path.join(TEMP_DROP_PATH, original_file_name)
        os.makedirs(TEMP_DROP_PATH, exist_ok=True)
        
        print(f"📥 Extracting streaming file artifact from source server...")
        try:
            with session.get(download_url, stream=True, timeout=30, verify=False) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            model_id, wind_angle = populate_database_from_mat(local_path)
            if model_id is not None:
                run_data_session_dashboard(model_id, wind_angle)
                return "complete"

        except Exception as e:
            print(f"❌ Processing failure inside session ingestion routine: {e}")
        finally:
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass


def run_pipeline_wizard():
    print("=== TPU Complete Automated Self-Cleaning Ingestion Engine ===")
    initialize_local_database()
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    session = create_legacy_session()

    print("\n🌐 SELECT A WIND TUNNEL DATABASE CATEGORY PORTAL:")
    portals_list = list(PORTAL_CATALOG.keys())
    for idx, category in enumerate(portals_list, 1):
        print(f" [{idx}] {category}")
    print(f" [{len(portals_list) + 1}] Custom Entry URL (Paste manually)")

    while True:
        portal_choice = input(f"\n🎯 Enter category number selection (1-{len(portals_list)+1}): ").strip()
        if portal_choice.isdigit():
            choice_idx = int(portal_choice)
            if 1 <= choice_idx <= len(portals_list):
                selected_category = portals_list[choice_idx - 1]
                base_start_url = PORTAL_CATALOG[selected_category]
                break
            elif choice_idx == len(portals_list) + 1:
                base_start_url = input("🔗 Paste your custom target TPU portal URL: ").strip()
                if base_start_url:
                    selected_category = "Custom Configuration Link"
                    break
        print("❌ Invalid entry configuration.")

    url_history = [base_start_url]
    menu_names_history = [selected_category]

    try:
        while True:
            current_url = url_history[-1]
            print(f"\n🔍 Analyzing layout context layer: {menu_names_history[-1]}")
            
            menu_options, resolved_url = parse_dropdown_from_url(session, current_url, headers)

            if not menu_options:
                action = process_final_data_page(session, resolved_url or current_url, headers)
                if action == "back":
                    url_history.pop()
                    menu_names_history.pop()
                    continue
                break

            options_list = list(menu_options.keys())
            print(f"📋 CHOOSE FROM THE EXTRACTED OPTIONS:")
            for idx, text in enumerate(options_list, 1):
                print(f" [{idx}] {text}")

            back_indicator = " (or 'b' to go back, 'q' to quit)" if len(url_history) > 1 else " (or 'q' to quit)"
            choice = input(f"\n🎯 Select option index number{back_indicator}: ").strip().lower()

            if choice == 'q':
                break
            elif choice == 'b' and len(url_history) > 1:
                url_history.pop()
                menu_names_history.pop()
                continue
            elif choice.isdigit() and (1 <= int(choice) <= len(options_list)):
                selected_text = options_list[int(choice) - 1]
                url_history.append(menu_options[selected_text])
                menu_names_history.append(selected_text)
            else:
                print("❌ Choice index position out of range bounds.")
                
    finally:
        clear_session_data()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    run_pipeline_wizard()