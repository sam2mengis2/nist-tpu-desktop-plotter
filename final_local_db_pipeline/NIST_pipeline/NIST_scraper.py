# -*- coding: utf-8 -*-
"""
NIST_pipeline/NIST_scraper.py - Isolated Ingestion Web Scraper Backend
Resolves multi-row rowspan nested page layouts using 2D grid cell expansion loops.
"""

import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Global cache to preserve states during drop-down drill down executions
NIST_TABLE_CACHE = {}


def analyze_nist_architecture(session, url, headers):
    """
    🎯 2D Grid Interpolation Engine: Unwraps complex, variable rowspan table cell layouts
    globally into flat row records to prevent out-of-index column shifting.
    """
    url = url.strip()
    
    # Check if we are running in a virtual path router state
    if url.startswith("nist_state://"):
        path_clean = url.replace("nist_state://", "")
        parts = path_clean.split("/")
        portal_key = parts[0]
        flat_rows = NIST_TABLE_CACHE.get(portal_key, [])
        if not flat_rows: 
            return {}, url, False
            
        # Step A: Extract Unique Test Configurations
        if len(parts) == 1:
            options_map = {}
            unique_tests = set()
            for r in flat_rows:
                unique_tests.add((r['test_num'], r['scale'], r['slope'], r['dimensions']))
            for t_num, scale, slope, dims in sorted(unique_tests):
                label = f"Test {t_num} (Scale {scale}, Slope {slope}, {dims})"
                options_map[label] = f"nist_state://{portal_key}/{t_num}"
            return options_map, url, False
            
        # Step B: Extract Eave Heights for the Selected Profile
        if len(parts) == 2:
            target_test = parts[1]
            options_map = {}
            unique_heights = set()
            for r in flat_rows:
                if r['test_num'] == target_test: 
                    unique_heights.add(r['height'])
            for h in sorted(unique_heights):
                options_map[f"Eave Height: {h}"] = f"nist_state://{portal_key}/{target_test}/{h}"
            return options_map, url, False

        # Step C: Extract Terrains/Roughness Exposures
        if len(parts) == 3:
            target_test, target_height = parts[1], parts[2]
            options_map = {}
            unique_terrains = set()
            for r in flat_rows:
                if r['test_num'] == target_test and r['height'] == target_height: 
                    unique_terrains.add(r['terrain'])
            for t in sorted(unique_terrains):
                options_map[f"Terrain: {t}"] = f"nist_state://{portal_key}/{target_test}/{target_height}/{t}"
            return options_map, url, False

        return {}, url, len(parts) == 4

    # Perform active HTTP fetch on original target server endpoints
    response = session.get(url, headers=headers, timeout=12, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table: 
        return {}, url, False

    rows = table.find_all("tr")
    matrix = {}
    
    # Map the asymmetrical layout to a true uniform 2D grid matrix
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (r_idx, c_idx) in matrix: 
                c_idx += 1
                
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = cell.get_text().strip()
            links = [(a.get_text().strip(), urljoin(url, a["href"])) for a in cell.find_all("a", href=True)]
            
            # Interpolate the value across all virtual slots spanned by the element
            for dr in range(rowspan):
                for dc in range(colspan):
                    matrix[(r_idx + dr, c_idx + dc)] = {
                        "text": text, 
                        "links": links, 
                        "is_header": cell.name == "th"
                    }
            c_idx += colspan

    max_r = max([k[0] for k in matrix.keys()]) if matrix else 0
    flat_rows = []
    portal_key = str(hash(url))
    
    for r in range(max_r + 1):
        c0 = matrix.get((r, 0))
        if not c0 or c0["is_header"] or not c0["text"]: 
            continue
            
        # Compile structured clean datasets mapped correctly to all parents
        flat_rows.append({
            'test_num': matrix.get((r, 0), {}).get("text", ""),
            'scale': matrix.get((r, 1), {}).get("text", ""),
            'slope': matrix.get((r, 2), {}).get("text", ""),
            'dimensions': f"{matrix.get((r,3),{}).get('text','')} x {matrix.get((r,4),{}).get('text','')}".strip(),
            'height': matrix.get((r, 5), {}).get("text", ""),
            'terrain': matrix.get((r, 6), {}).get("text", ""),
            'links': matrix.get((r, 7), {}).get("links", [])
        })
    
    NIST_TABLE_CACHE[portal_key] = flat_rows
    return analyze_nist_architecture(session, f"nist_state://{portal_key}", headers)


def get_cached_links_for_leaf(resolved_url):
    """Retrieves download file vectors belonging to the targeted active configuration."""
    path_clean = resolved_url.replace("nist_state://", "")
    parts = path_clean.split("/")
    portal_key, target_test, target_height, target_terrain = parts[0], parts[1], parts[2], parts[3]
    
    flat_rows = NIST_TABLE_CACHE.get(portal_key, [])
    for r in flat_rows:
        if r['test_num'] == target_test and r['height'] == target_height and r['terrain'] == target_terrain:
            return r['links']
    return []