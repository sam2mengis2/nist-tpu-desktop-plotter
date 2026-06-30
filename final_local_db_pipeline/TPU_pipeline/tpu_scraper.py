import os
import re
import ssl
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# Import your database function from your existing script
from TPU_parser import populate_database_from_mat

# Root URL container where the chain begins
BASE_URL = "https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm"


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
    
    # Look for 'mysel' first. If it's missing, look for the 'urlsel' tag configuration
    select_tag = soup.find("select", attrs={"name": "mysel"})
    if not select_tag:
        select_tag = soup.find("select", attrs={"name": "urlsel"})
        
    if select_tag:
        return select_tag, url

    # Fallback: Look inside frame tags if we are dealing with an old frameset container
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
    """Processes the final results grid page using Direct Row Lookup."""
    try:
        res = session.get(page_url, headers=headers, timeout=12, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to parse data page: {e}")
        return

    # 🎯 FIX: Direct Row Lookup Strategy
    data_file_row = None
    direction_row = None

    # 1. Target the file links row directly by finding the element containing '.mat'
    mat_text_node = soup.find(string=re.compile(r"\.mat|Data files", re.IGNORECASE))
    if mat_text_node:
        data_file_row = mat_text_node.find_parent("tr")

    # 2. Look backwards through sibling rows directly above it to find the header values
    if data_file_row:
        current_row = data_file_row.find_previous_sibling("tr")
        while current_row:
            if "Wind direction" in current_row.get_text():
                direction_row = current_row
                break
            current_row = current_row.find_previous_sibling("tr")

    if not direction_row or not data_file_row:
        print("❌ Error parsing the wind angle matrix grid layout.")
        print(f"Debug Info - Current URL location: {page_url}")
        return

    # Extract clean integers for angles
    angles = []
    for td in direction_row.find_all(["td", "th"]):
        text = td.get_text().strip()
        match = re.search(r"(\d+)", text)
        if match:
            angles.append(int(match.group(1)))

    # Extract down matching absolute links
    links = []
    for td in data_file_row.find_all("td"):
        a_tag = td.find("a")
        if a_tag and "href" in a_tag.attrs:
            links.append(urljoin(page_url, a_tag["href"]))

    angle_map = dict(zip(angles, links[:len(angles)]))
    available_angles = sorted(list(angle_map.keys()))

    if not available_angles:
        print("❌ No available wind angles parsed from this grid layout.")
        return

    while True:
        print(f"\n📊 SUCCESS! Reached Final Model Configuration Page.")
        print(f"Available Wind Angles: {available_angles}")
        angle_choice = input("👉 Enter target Wind Angle to parse & save to DB (or 'b' to go back): ").strip().lower()

        if angle_choice == 'b':
            return "back"

        if not angle_choice.isdigit() or int(angle_choice) not in angle_map:
            print(f"❌ Invalid choice. Please pick a vector from: {available_angles}")
            continue

        target_angle = int(angle_choice)
        download_url = angle_map[target_angle]
        filename = f"tpu_scraped_angle_{target_angle}.mat"
        local_path = os.path.join("downloads", filename)

        os.makedirs("downloads", exist_ok=True)
        print("📥 Downloading binary wind tunnel dataset...")
        try:
            with session.get(download_url, stream=True, timeout=30, verify=False) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print("⚡ Pipeline Connected! Ingesting values into local SQLite database...")
            populate_database_from_mat(local_path)
            print("🎉 Database successfully updated with calculations!")
        except Exception as e:
            print(f"❌ Processing failure: {e}")


def run_pipeline_wizard():
    print("=== TPU Complete Automated Lifecycle Ingestion Engine ===")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    session = create_legacy_session()

    url_history = [BASE_URL]
    menu_names_history = ["Main Directory Container"]

    while True:
        current_url = url_history[-1]
        print(f"\n🔍 Analyzing layout context layer: {menu_names_history[-1]}")
        
        # Step 1: Look for any available dropdown options at the current URL stage
        menu_options, resolved_url = parse_dropdown_from_url(session, current_url, headers)

        # Step 2: If no dropdown menu is found, the selection chain is complete!
        if not menu_options:
            action = process_final_data_page(session, resolved_url or current_url, headers)
            if action == "back":
                url_history.pop()
                menu_names_history.pop()
                continue
            break

        # Step 3: Present dropdown choices to user
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