# -*- coding: utf-8 -*-
"""
tpu_scraper.py - Automated Lifecycle Ingestion & Self-Cleaning Engine
Crawls the TPU database, streams target datasets, processes them into SQLite, 
and automatically deletes the downloaded files to keep your drive clear.
"""

import os
import re
import ssl
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# 🎯 FIX: Import BOTH the initialization function and the data parser function
from TPU_parser import populate_database_from_mat, initialize_local_database

# Temporary landing folder for files before processing
TEMP_DROP_PATH = r"C:\FINAL_SUMMER_PROJ\final_local_db_pipeline\file_drop"

# Catalog of primary TPU database landing portals
PORTAL_CATALOG = {
    "High-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm",
    "Low-Rise Buildings (Isolated Building)": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/Homepage/homepageLDF.htm",
    "Low-Rise Buildings with Eaves": "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/eaves/Homepage/homepageEDF.htm",
    "Low Rise Buildings (Non-Isolated)": 'https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/grouplowrise/mainpage.html'
}


class LegacySSLAdapter(HTTPAdapter):
    """Bypasses weak Diffie-Hellman handshakes on legacy academic servers."""
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT@SECLEVEL=1:HIGH:!DH:!EDH")
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, ssl_context=ctx, **pool_kwargs
        )


def create_legacy_session():
    """Returns a requests session configured for older server architectures."""
    session = requests.Session()
    adapter = LegacySSLAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def find_dropdown_recursively(session, url, headers, visited=None):
    """Scans the current URL and nested framesets to locate either 'mysel' or 'urlsel' dropdowns."""
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
    
    select_tag = soup.find("select", attrs={"name": "mysel"})
    if not select_tag:
        select_tag = soup.find("select", attrs={"name": "urlsel"})
        
    if select_tag:
        return select_tag, url

    for frame in soup.find_all(["frame", "iframe"]):
        src = frame.get("src")
        if src:
            full_frame_url = urljoin(url, src)
            tag, found_url = find_dropdown_recursively(session, full_frame_url, headers, visited)
            if tag:
                return tag, found_url

    return None, None


def parse_dropdown_from_url(session, url, headers):
    """Fetches a URL and returns a clean text-to-link mapping of its dropdown options."""
    select_tag, actual_content_url = find_dropdown_recursively(session, url, headers)
    if not select_tag:
        return {}, None

    options_map = {}
    for option in select_tag.find_all("option"):
        value = option.get("value")
        text = option.get_text().strip()

        if value == "def" or not value or "please select" in text.lower():
            continue

        full_url = urljoin(actual_content_url, value)
        options_map[text] = full_url

    return options_map, actual_content_url


def process_final_data_page(session, page_url, headers):
    """Processes the final results page, parses files, populates DB, and cleans up the storage."""
    try:
        res = session.get(page_url, headers=headers, timeout=12, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to parse data page: {e}")
        return

    data_file_row = None
    direction_row = None

    # Scan rows to isolate the true time-series data link block
    for row in soup.find_all("tr"):
        a_tags = row.find_all("a", href=True)
        mat_links = [a for a in a_tags if ".mat" in a["href"].lower()]
        if len(mat_links) > 1:
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

    # Extract available angles
    angles = []
    for td in direction_row.find_all(["td", "th"]):
        text = td.get_text().strip()
        match = re.search(r"(\d+)", text)
        if match:
            angles.append(int(match.group(1)))

    # Collect matching download URLs
    links = []
    for td in data_file_row.find_all("td"):
        a_tag = td.find("a")
        if a_tag and "href" in a_tag.attrs and ".mat" in a_tag["href"].lower():
            links.append(urljoin(page_url, a_tag["href"]))

    angle_map = dict(zip(angles, links[:len(angles)]))
    available_angles = sorted(list(angle_map.keys()))

    if not available_angles:
        print("❌ No available wind angles could be extracted from this model grid layout.")
        return

    while True:
        print(f"\n📊 SUCCESS! Reached Final Model Configuration Page.")
        print(f"Available Wind Angles: {available_angles}")
        angle_choice = input("👉 Enter target Wind Angle to process into DB (or 'b' to go back): ").strip().lower()

        if angle_choice == 'b':
            return "back"

        if not angle_choice.isdigit() or int(angle_choice) not in angle_map:
            print(f"❌ Invalid choice. Please pick an angle from: {available_angles}")
            continue

        target_angle = int(angle_choice)
        download_url = angle_map[target_angle]
        
        original_file_name = download_url.split('/')[-1]
        if not original_file_name.endswith('.mat'):
            original_file_name = f"tpu_angle_{target_angle}.mat"

        local_path = os.path.join(TEMP_DROP_PATH, original_file_name)
        os.makedirs(TEMP_DROP_PATH, exist_ok=True)
        
        print(f"📥 Downloading transient dataset file...")
        try:
            with session.get(download_url, stream=True, timeout=30, verify=False) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Step 2: Handoff file directly to database ingestion function
            print("⚡ Pipeline Connected! Calculating statistical metrics and updating local database...")
            populate_database_from_mat(local_path)
            print("🎉 Database successfully updated with calculations!")

        except Exception as e:
            print(f"❌ Processing failure during pipeline stream: {e}")
            
        finally:
            # Step 3: Self-Cleaning Action Execution Block
            if os.path.exists(local_path):
                try:
                    print(f"🧹 Cleaning up: Deleting temporary file '{original_file_name}' from drive...")
                    os.remove(local_path)
                    print("✨ Temporary file successfully wiped clean.")
                except Exception as cleanup_err:
                    print(f"⚠️ Warning: Could not delete temporary file from disk: {cleanup_err}")


def run_pipeline_wizard():
    print("=== TPU Complete Automated Self-Cleaning Ingestion Engine ===")
    
    # 🎯 FIX: Automatically check/initialize the database tables on startup
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
        print("❌ Invalid entry configuration. Please select a valid category option.")

    url_history = [base_start_url]
    menu_names_history = [selected_category]

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
            print("Pipeline execution closed.")
            break
        elif choice == 'b' and len(url_history) > 1:
            url_history.pop()
            menu_names_history.pop()
            continue
        elif choice.isdigit() and (1 <= int(choice) <= len(options_list)):
            selected_text = options_list[int(choice) - 1]
            next_url = menu_options[selected_text]
            
            url_history.append(next_url)
            menu_names_history.append(selected_text)
        else:
            print(f"❌ Out of scope index selection. Enter numbers 1 to {len(options_list)}.")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    run_pipeline_wizard()