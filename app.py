import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import re
import copy
from datetime import datetime, timedelta
from collections import Counter  # Added for duplicate resource counting
import textwrap  # For dedenting multi-line strings
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Border, Side
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment

# -----------------------
# CONFIGURATION & CONSTANTS
# -----------------------
st.set_page_config(layout="wide")

TRADE_CIRCLE_SIZE = 6  # 6 players per circle, each gets 2 resources

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def get_resource_1_2(row):
    """
    Return a string with Resource 1 and Resource 2 in the format "Resource 1, Resource 2".
    In the event of unrecognized characters, these are safely encoded to ASCII, dropping any problematic bytes.
    """
    def safe_str(val):
        if pd.isnull(val):
            return ""
        # Convert value to string and force encoding to ASCII (ignoring unrecognized characters)
        return str(val).strip().encode("ascii", errors="ignore").decode("ascii")

    s1 = safe_str(row.get("Resource 1"))
    s2 = safe_str(row.get("Resource 2"))
    if s1 and s2:
        return f"{s1}, {s2}"
    else:
        # Fallback: parse from the "Current Resources" column
        current = safe_str(row.get("Current Resources", ""))
        resources = [r.strip() for r in current.split(",") if r.strip()]
        if len(resources) >= 2:
            return f"{resources[0]}, {resources[1]}"
        elif resources:
            return resources[0]
        return ""

def add_nation_drill_url(df):
    """
    Append a 'Nation Drill URL' column to the DataFrame based on the 'Nation ID' column.
    """
    df = df.copy()
    if 'Nation ID' in df.columns:
        df['Nation Drill URL'] = "https://www.cybernations.net/nation_drill_display.asp?Nation_ID=" + df['Nation ID'].astype(str)
    return df
    
def get_current_resources(row, resource_cols):
    """Return a comma-separated string of non-blank resources sorted alphabetically."""
    resources = sorted([str(x).strip() for x in row[resource_cols] if pd.notnull(x) and str(x).strip() != ''])
    return ", ".join(resources)

def count_empty_slots(row, resource_cols):
    """Count blank resource cells and determine trade slots (each slot covers 2 resources)."""
    count = sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '')
    return count // 2
    
def is_valid_resource_pair(current_pair_str, valid_combos):
    """
    Check if the given current resource pair string (e.g., 'A, B')
    is one of the valid combinations (each valid combo is expected to be a sorted list of two resources).
    """
    # If current_pair_str is None or empty, return False immediately.
    if not current_pair_str or not isinstance(current_pair_str, str):
        return False
    
    current_pair = sorted([r.strip() for r in current_pair_str.split(",") if r.strip()])
    for combo in valid_combos:
        if current_pair == combo:
            return True
    return False

def display_trade_circle_df(circle, condition):
    """Display a trade circle in a Streamlit dataframe."""
    circle_data = []
    for player in circle:
        current_resources_str = player.get('Current Resources', '')
        circle_data.append({
            'Trade Circle ID': player.get('Trade Circle ID', ''),  # New column added here
            'Nation ID': player.get('Nation ID', ''),
            'Ruler Name': player.get('Ruler Name', ''),
            'Nation Name': player.get('Nation Name', ''),
            'Team': player.get('Team', ''),
            'Current Resources': current_resources_str,
            'Current Resource 1+2': get_resource_1_2(player),
            'Activity': player.get('Activity', ''),
            'Days Old': player.get('Days Old', ''),
            f'Assigned {condition} Resources': ", ".join(player.get('Assigned Resources', [])) if player.get('Assigned Resources') else "None"
        })
    circle_df = pd.DataFrame(circle_data)
    st.dataframe(circle_df, use_container_width=True)

def highlight_none(val):
    """Return a gray font color if the value is 'None', otherwise no styling."""
    if val == "None":
        return 'color: gray'
    return ''

# -----------------------
# DOWNLOAD & DATA LOADING FUNCTIONS
# -----------------------
def download_and_extract_zip(url):
    """Download a zip file from the given URL and extract its first file as a DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        # Comment out or remove the st.error message to suppress it.
        # st.error(f"Error downloading file from {url}: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_list = z.namelist()
        if not file_list:
            # st.error("The zip file is empty.")
            return None
        file_name = file_list[0]
        with z.open(file_name) as file:
            try:
                # Suppress warnings during CSV reading.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(file, delimiter="|", encoding="ISO-8859-1")
                return df
            except Exception as e:
                # st.error(f"Error reading the CSV file: {e}")
                return None

# -----------------------
# MAIN APP
# -----------------------
def main():
    st.title("Cyber Nations | Nation Statistics & Trade Circle Formation Tool")
    
    # About section
    st.markdown(
        """
        **About This Tool:**
        This application downloads and processes nation statistics from CyberNations,
        providing functionality to filter data and form trade circles for both peacetime
        and wartime scenarios. It helps organize players into groups based on their available
        resources and activity status, streamlining trade circle formation.
        """
    )
    
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = False
    
    if not st.session_state.password_verified:
        password = st.text_input("Enter Password", type="password", key="password_input")
        if password:
            if password == "secret":
                st.session_state.password_verified = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
    
    # Only proceed if the password is verified.
    if st.session_state.password_verified:
        # Automatically download the Nation Statistics ZIP on UI load.
        with st.spinner("Retrieving Nation Statistics..."):
            today = datetime.now()
            base_url = "https://www.cybernations.net/assets/CyberNations_SE_Nation_Stats_"
            # Create a list of dates to try: current day, previous day, and next day.
            dates_to_try = [today, today - timedelta(days=1), today + timedelta(days=1)]
            df = None  # to hold the DataFrame if a download succeeds

            # Iterate through the dates and try both URL variants
            for dt in dates_to_try:
                date_str = f"{dt.month}{dt.day}{dt.year}"
                url1 = base_url + date_str + "510001.zip"
                url2 = base_url + date_str + "510002.zip"
                
                df = download_and_extract_zip(url1)
                if df is None:
                    df = download_and_extract_zip(url2)
                
                if df is not None:
                    break

            if df is not None:
                st.session_state.df = df
            else:
                st.error("Failed to load data from any of the constructed URLs.")

        # Proceed if data is loaded
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df.copy()

            # -----------------------
            # ORIGINAL DATA SECTION
            # -----------------------
            with st.expander("Original Data"):
                st.dataframe(df, use_container_width=True)

            # -----------------------
            # FILTERING UI SECTION
            # -----------------------
            with st.expander("Filter Data"):
                if not df.empty:
                    column_options = list(df.columns)
                    default_index = column_options.index("Alliance") if "Alliance" in column_options else 0
                    selected_column = st.selectbox("Select column to filter", column_options, index=default_index, key="filter_select")
                    
                    if selected_column == "Alliance":
                        alliances = sorted(df["Alliance"].dropna().unique())
                        default_alliances = ["Freehold of The Wolves"] if "Freehold of The Wolves" in alliances else alliances
                        selected_alliances = st.multiselect("Select Alliance(s) to filter", alliances, default=default_alliances, key="alliance_filter")
                        st.session_state.selected_alliances = selected_alliances  # Save the selection
                        if selected_alliances:
                            filtered_df = df[df["Alliance"].isin(selected_alliances)]
                        else:
                            filtered_df = df.copy()
                    else:
                        search_text = st.text_input("Filter by text (separate words by comma)", value="Freehold of the Wolves", key="filter_text")
                        if search_text:
                            # Support multiple filters separated by comma
                            filters = [f.strip() for f in search_text.split(",") if f.strip()]
                            pattern = "|".join(filters)
                            filtered_df = df[df[selected_column].astype(str).str.contains(pattern, case=False, na=False)]
                        else:
                            st.info("Enter text to filter the data.")
                            filtered_df = df.copy()
                            
                    st.write(f"Showing results where **{selected_column}** is filtered:")
                    # Use the filtered table's own "Resource 1" and "Resource 2" columns for current resource display
                    if "Resource 1" in filtered_df.columns and "Resource 2" in filtered_df.columns:
                        filtered_df = filtered_df.copy()
                        filtered_df['Current Resource 1+2'] = filtered_df.apply(lambda row: get_resource_1_2(row), axis=1)

                    if "Created" in filtered_df.columns:
                        # Adjust date_format as needed based on your data
                        date_format = "%m/%d/%Y %I:%M:%S %p"  
                        filtered_df['Created'] = pd.to_datetime(filtered_df['Created'], format=date_format, errors='coerce')
                        current_date = datetime.now()
                        filtered_df['Days Old'] = (current_date - filtered_df['Created']).dt.days

                    # Sort by "Ruler Name" and reset index
                    filtered_df = filtered_df.sort_values(by="Ruler Name", key=lambda col: col.str.lower()).reset_index(drop=True)

                    st.dataframe(filtered_df, use_container_width=True)
                    # Save the filtered DataFrame and CSV content to session state for later use.
                    st.session_state.filtered_df = filtered_df
                    csv_content = filtered_df.to_csv(index=False)
                    st.session_state.filtered_csv = csv_content
                    st.download_button("Download Filtered CSV", csv_content, file_name="filtered_nation_stats.csv", mime="text/csv", key="download_csv")
                else:
                    st.warning("DataFrame is empty, nothing to filter.")

            # -----------------------
            # TRADE CIRCLE & RESOURCE PROCESSING (automatically triggered)
            # -----------------------
            if "df" in st.session_state:
                # Use the filtered DataFrame directly if available, otherwise fall back to the original DataFrame.
                if "filtered_df" in st.session_state:
                    df_to_use = st.session_state.filtered_df.copy()
                else:
                    df_to_use = df.copy()

                # Re-calculate the 'Created' and 'Days Old' columns on the filtered DataFrame.
                if "Created" in df_to_use.columns:
                    date_format = "%m/%d/%Y %I:%M:%S %p"  # Adjust as needed.
                    df_to_use['Created'] = pd.to_datetime(df_to_use['Created'], format=date_format, errors='coerce')
                    current_date = datetime.now()
                    df_to_use['Days Old'] = (current_date - df_to_use['Created']).dt.days

                # Continue processing using df_to_use.
                resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
                mask_empty = df_to_use[resource_cols].isnull().any(axis=1) | (
                    df_to_use[resource_cols].apply(lambda col: col.astype(str).str.strip() == '').any(axis=1)
                )
                players_empty = df_to_use[mask_empty].copy()

                players_empty['Current Resources'] = players_empty.apply(lambda row: get_current_resources(row, resource_cols), axis=1)
                players_empty['Current Resource 1+2'] = players_empty.apply(lambda row: get_resource_1_2(row), axis=1)
                players_empty['Empty Slots Count'] = players_empty.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)

                # Filter out inactive players.
                players_empty = players_empty[~players_empty['Activity'].isin(["Active Three Weeks Ago", "Active More Than Three Weeks Ago"])]

                if "Alliance Status" in players_empty.columns:
                    players_empty = players_empty[players_empty["Alliance Status"] != "Pending"]

                # -----------------------
                # PLAYERS WITH EMPTY TRADE SLOTS
                # -----------------------
                with st.expander("Players with empty trade slots (active recently)"):
                    display_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Team',
                                    'Current Resources', 'Current Resource 1+2',
                                    'Empty Slots Count', 'Activity', 'Days Old']
                    players_empty = players_empty.sort_values(by="Ruler Name", key=lambda col: col.str.lower()).reset_index(drop=True)
                    st.dataframe(players_empty[display_cols].reset_index(drop=True), use_container_width=True)

                # ---- New Filter: Exclude players with Alliance Status "Pending" ----
                if "Alliance Status" in players_empty.columns:
                    players_empty = players_empty[players_empty["Alliance Status"] != "Pending"]
                
                # -----------------------
                # PLAYERS WITH COMPLETE TRADE CIRCLES
                # -----------------------
                with st.expander("Players with a complete trade circle (no empty slots)"):
                    players_full = df_to_use[~mask_empty].copy()
                    # Compute "Current Resources" for players with complete resource sets
                    players_full['Current Resources'] = players_full.apply(lambda row: get_current_resources(row, resource_cols), axis=1)
                    # Use CSV-based "Resource 1" and "Resource 2" for Current Resource 1+2 (if available)
                    players_full['Current Resource 1+2'] = players_full.apply(lambda row: get_resource_1_2(row), axis=1)
                    # Also compute "Empty Slots Count" to verify these players have complete resource sets (should be 0)
                    players_full['Empty Slots Count'] = players_full.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
                    players_full = players_full.sort_values(by="Ruler Name", key=lambda col: col.str.lower()).reset_index(drop=True)
                    if "Alliance Status" in players_full.columns:
                        players_full = players_full[players_full["Alliance Status"] != "Pending"]
                    st.dataframe(players_full[display_cols].reset_index(drop=True), use_container_width=True)

                # -----------------------
                # RESOURCE MISMATCHES
                # -----------------------
                with st.expander("Resource Mismatches"):
                    st.markdown(
                        """
                        In this section, we check for existing complete trade circles to see if they have the correct available combinations of resources.
 
                        ### Understanding Peace Mode Levels
                    
                        - **Peace Mode Level A:**  
                          Nations that are less than **1000 days old**. This level is intended for newer or rapidly developing nations, which may still be adjusting their resource management.
                    
                        - **Peace Mode Level B:**  
                          Nations that are between **1000 and 2000 days old**. These nations are moderately established; their resource combinations may be evolving as they fine-tune their trade strategies.
                    
                        - **Peace Mode Level C:**  
                          Nations that are **2000 days or older**. These are mature nations with longstanding resource setups, typically expecting more stable and optimized resource combinations.
                        """
                    )

                    st.markdown("### Valid Resource Combinations Input")
                    # Text box for Peace Mode - Level A with default combinations.
                    peace_a_text = st.text_area(
                        "Peace Mode - Level A (one combination per line)",
                        value="""Cattle, Coal, Fish, Gems, Gold, Lead, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Gold, Lead, Oil, Pigs, Rubber, Spices, Sugar, Uranium, Wheat
Coal, Fish, Furs, Gems, Gold, Lead, Oil, Rubber, Silver, Uranium, Wheat, Wine
Coal, Fish, Gems, Gold, Lead, Oil, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Coal, Fish, Gems, Gold, Lead, Lumber, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Coal, Fish, Gems, Gold, Lead, Oil, Pigs, Rubber, Silver, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, Marble, Oil, Rubber, Uranium, Wheat
Coal, Fish, Gems, Gold, Lead, Marble, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Gold, Lead, Lumber, Oil, Rubber, Spices, Sugar, Uranium, Wheat""",
                        height=100
                    )
                    peace_b_text = st.text_area(
                        "Peace Mode - Level B (one combination per line)",
                        value="""Aluminum, Cattle, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Water, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Sugar, Uranium, Wheat
Aluminum, Coal, Fish, Gems, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Pigs, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Silver, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Marble, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Aluminum, Cattle, Coal, Fish, Iron, Lumber, Marble, Rubber, Spices, Uranium, Water, Wheat""",
                        height=100
                    )
                    peace_c_text = st.text_area(
                        "Peace Mode - Level C (one combination per line)",
                        value="""Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Pigs, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Gems, Gold, Pigs, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Water, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Oil, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Pigs, Rubber, Silver, Spices, Uranium, Wine
Cattle, Coal, Fish, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat, Wine""",
                        height=100
                    )
                    war_text = st.text_area(
                        "War Mode (one combination per line)",
                        value="""Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Spices, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Wheat
Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Cattle, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Sugar, Uranium
Aluminum, Coal, Furs, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Silver, Uranium
Aluminum, Coal, Gems, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Wine
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Water""",
                        height=100
                    )
                    
                    # Helper: parse input text into a list of sorted resource lists.
                    def parse_combinations(input_text):
                        combos = []
                        for line in input_text.strip().splitlines():
                            line = line.strip()
                            if line:
                                resources = sorted([res.strip() for res in line.split(",") if res.strip()])
                                if resources:
                                    combos.append(resources)
                        return combos
                
                    # Helper: compare two lists of resources and compute missing and extra resources.
                    def compare_resources(nation_resources, valid_combo):
                        nation_set = set(nation_resources)
                        valid_set = set(valid_combo)
                        missing = valid_set - nation_set
                        extra = nation_set - valid_set
                        return missing, extra
                
                    # Helper: find the best matching valid combination that minimizes total mismatches.
                    def find_best_match(nation_resources, valid_combinations):
                        best_combo = None
                        best_missing = set()
                        best_extra = set()
                        best_score = float('inf')
                        for combo in valid_combinations:
                            missing, extra = compare_resources(nation_resources, combo)
                            score = len(missing) + len(extra)
                            if score < best_score:
                                best_score = score
                                best_combo = combo
                                best_missing = missing
                                best_extra = extra
                        return best_combo, best_missing, best_extra, best_score
                
                    # Parse the text inputs into lists of valid combinations.
                    peace_a_combos = parse_combinations(peace_a_text)
                    peace_b_combos = parse_combinations(peace_b_text)
                    peace_c_combos = parse_combinations(peace_c_text)
                    war_combos     = parse_combinations(war_text)
                    
                    st.markdown("**Total valid combinations provided:**")
                    st.write(f"Peace Mode Level A: {len(peace_a_combos)}")
                    st.write(f"Peace Mode Level B: {len(peace_b_combos)}")
                    st.write(f"Peace Mode Level C: {len(peace_c_combos)}")
                    st.write(f"War Mode: {len(war_combos)}")
                    
                    # Create four lists to hold mismatches for each category.
                    mismatch_peace_a = []
                    mismatch_peace_b = []
                    mismatch_peace_c = []
                    mismatch_war     = []
                    
                    for idx, row in players_full.iterrows():
                        # Get current resources as per the CSV-based list.
                        current_resources = [res.strip() for res in row['Current Resources'].split(',') if res.strip()]
                        # Ensure Resource 1 and Resource 2 are included if not already present.
                        if "Resource 1" in row and pd.notnull(row["Resource 1"]):
                            res1 = str(row["Resource 1"]).strip()
                            if res1 and res1 not in current_resources:
                                current_resources.append(res1)
                        if "Resource 2" in row and pd.notnull(row["Resource 2"]):
                            res2 = str(row["Resource 2"]).strip()
                            if res2 and res2 not in current_resources:
                                current_resources.append(res2)
                        
                        # Create a sorted list for proper comparison.
                        current_resources_sorted = sorted(current_resources)
                        current_set_str = ", ".join(current_resources_sorted)
                        # Calculate duplicate resources.
                        duplicates = [res for res, count in Counter(current_resources).items() if count > 1]
                        dup_str = ", ".join(sorted(duplicates)) if duplicates else "None"
                        
                        base_info = {
                            'Nation ID': row['Nation ID'],
                            'Ruler Name': row['Ruler Name'],
                            'Nation Name': row['Nation Name'],
                            'Current Resources': row['Current Resources'],
                            'Current Resource 1+2': get_resource_1_2(row),
                            'Duplicate Resources': dup_str,
                            'Current Sorted Resources': current_set_str,
                            'Activity': row['Activity'],
                            'Days Old': row['Days Old']
                        }
                        
                        # Check against Peace Mode Level A.
                        best_combo, missing, extra, score = find_best_match(current_resources_sorted, peace_a_combos)
                        if score != 0:  # mismatch exists if any resource is missing or extra
                            rec = base_info.copy()
                            rec.update({
                                'Valid Combination': ", ".join(best_combo),
                                'Missing Resources': ", ".join(sorted(missing)) if missing else "None",
                                'Extra Resources': ", ".join(sorted(extra)) if extra else "None"
                            })
                            mismatch_peace_a.append(rec)
                        
                        # Check against Peace Mode Level B.
                        best_combo, missing, extra, score = find_best_match(current_resources_sorted, peace_b_combos)
                        if score != 0:
                            rec = base_info.copy()
                            rec.update({
                                'Valid Combination': ", ".join(best_combo),
                                'Missing Resources': ", ".join(sorted(missing)) if missing else "None",
                                'Extra Resources': ", ".join(sorted(extra)) if extra else "None"
                            })
                            mismatch_peace_b.append(rec)
                        
                        # Check against Peace Mode Level C.
                        best_combo, missing, extra, score = find_best_match(current_resources_sorted, peace_c_combos)
                        if score != 0:
                            rec = base_info.copy()
                            rec.update({
                                'Valid Combination': ", ".join(best_combo),
                                'Missing Resources': ", ".join(sorted(missing)) if missing else "None",
                                'Extra Resources': ", ".join(sorted(extra)) if extra else "None"
                            })
                            mismatch_peace_c.append(rec)
                        
                        # Check against War Mode.
                        best_combo, missing, extra, score = find_best_match(current_resources_sorted, war_combos)
                        if score != 0:
                            rec = base_info.copy()
                            rec.update({
                                'Valid Combination': ", ".join(best_combo),
                                'Missing Resources': ", ".join(sorted(missing)) if missing else "None",
                                'Extra Resources': ", ".join(sorted(extra)) if extra else "None"
                            })
                            mismatch_war.append(rec)
                    
                    # Convert each list into a DataFrame and apply age filters.
                    # Peace Mode Level A
                    df_peace_a = pd.DataFrame(mismatch_peace_a)
                    # Filter to include only nations under 1000 days old.
                    df_peace_a = df_peace_a[df_peace_a['Days Old'] < 1000]
                    if not df_peace_a.empty:
                        df_peace_a = df_peace_a.sort_values(by='Ruler Name', key=lambda col: col.str.lower()).reset_index(drop=True)
                        styled_peace_a = df_peace_a.style.applymap(highlight_none, subset=['Duplicate Resources'])
                        st.markdown(f"**Peace Mode Level A Mismatches: {len(df_peace_a)}**")
                        st.dataframe(styled_peace_a, use_container_width=True)
                    else:
                        st.info("No mismatches found for Peace Mode Level A (nations under 1000 days old).")

                    # Peace Mode Level B
                    df_peace_b = pd.DataFrame(mismatch_peace_b)
                    # Filter to include only nations 1000-2000 days old.
                    df_peace_b = df_peace_b[(df_peace_b['Days Old'] >= 1000) & (df_peace_b['Days Old'] < 2000)]
                    if not df_peace_b.empty:
                        df_peace_b = df_peace_b.sort_values(by='Ruler Name', key=lambda col: col.str.lower()).reset_index(drop=True)
                        styled_peace_b = df_peace_b.style.applymap(highlight_none, subset=['Duplicate Resources'])
                        st.markdown(f"**Peace Mode Level B Mismatches: {len(df_peace_b)}**")
                        st.dataframe(styled_peace_b, use_container_width=True)
                    else:
                        st.info("No mismatches found for Peace Mode Level B (nations 1000-2000 days old).")

                    # Peace Mode Level C
                    df_peace_c = pd.DataFrame(mismatch_peace_c)
                    # Filter to include only nations over 2000 days old.
                    df_peace_c = df_peace_c[df_peace_c['Days Old'] >= 2000]
                    if not df_peace_c.empty:
                        df_peace_c = df_peace_c.sort_values(by='Ruler Name', key=lambda col: col.str.lower()).reset_index(drop=True)
                        styled_peace_c = df_peace_c.style.applymap(highlight_none, subset=['Duplicate Resources'])
                        st.markdown(f"**Peace Mode Level C Mismatches: {len(df_peace_c)}**")
                        st.dataframe(styled_peace_c, use_container_width=True)
                    else:
                        st.info("No mismatches found for Peace Mode Level C (nations over 2000 days old).")

                    # War Mode
                    df_war = pd.DataFrame(mismatch_war)
                    # No age filtering for War Mode.
                    if not df_war.empty:
                        df_war = df_war.sort_values(by='Ruler Name', key=lambda col: col.str.lower()).reset_index(drop=True)
                        styled_war = df_war.style.applymap(highlight_none, subset=['Duplicate Resources'])
                        st.markdown(f"**War Mode Mismatches: {len(df_war)}**")
                        st.dataframe(styled_war, use_container_width=True)
                    else:
                        st.info("No mismatches found for War Mode.")

                    # --- Consolidate mismatch DataFrames for later use (summary, Excel export) ---
                    # Consolidate only the filtered Peace Mode DataFrames.
                    if not (df_peace_a.empty and df_peace_b.empty and df_peace_c.empty):
                        peacetime_df = pd.concat([df_peace_a, df_peace_b, df_peace_c], ignore_index=True)
                    else:
                        peacetime_df = pd.DataFrame()
                    wartime_df = df_war.copy()

                # -----------------------
                # RECOMMENDED TRADE CIRCLES (UPDATED SECTION)
                # -----------------------
                with st.expander("Recommended Trade Circles"):
                    st.markdown("### Paste Trade Circle Data")
                    trade_circle_text = st.text_area(
                        "Enter Trade Circle data below. Each line should contain the following tab-separated fields:\n"
                        "Ruler Name | Resource 1+2 | Alliance | Team | Days Old | Nation Drill Link | Activity\n\n"
                        "Empty slots are denoted by a line that starts with an 'x' or has an empty Ruler Name. Separate Trade Circle blocks with an empty line.",
                        height=200
                    )
                    
                    st.markdown("### Filter Out Players")
                    filter_text = st.text_area(
                        "Enter one value per line to filter out players by matching Ruler Name, Nation Name, or Nation ID:",
                        height=100
                    )
                    
                    # Build a filter set for quick exclusion:
                    filter_set = set(line.strip().lower() for line in filter_text.splitlines() if line.strip())
                    
                    # ---------------------------------------------------------------------
                    # Determine Majority Team from Filter Data (assumes filtered_df exists)
                    # ---------------------------------------------------------------------
                    majority_team = None
                    if "filtered_df" in st.session_state:
                        df_filtered = st.session_state.filtered_df
                        if not df_filtered["Team"].empty:
                            majority_team = df_filtered["Team"].mode().iloc[0]
                    
                    # ---------------------------------------------------------------------
                    # Extend eligibility: only include players who have Activity < 14, are in a selected alliance,
                    # AND are in the majority team of that alliance.
                    # ---------------------------------------------------------------------
                    def eligible(player):
                        try:
                            act = float(player.get("Activity", 0))
                        except:
                            act = 99  # treat non-numeric as ineligible
                        # Basic check on activity and alliance:
                        if act >= 14 or (player.get("Alliance") not in st.session_state.selected_alliances):
                            return False
                        # Also require that the player's team matches the majority team (if defined)
                        if majority_team and player.get("Team") != majority_team:
                            return False
                        return True

                    # ---------------------------------------------------------------------
                    # Parse pasted Trade Circle text into circle blocks.
                    # ---------------------------------------------------------------------
                    blocks = []
                    current_block = []
                    for line in trade_circle_text.splitlines():
                        line = line.strip()
                        if not line:
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
                        else:
                            current_block.append(line)
                    if current_block:
                        blocks.append(current_block)
                    
                    # Build initial list of circles from pasted data.
                    pasted_circles = []
                    for block in blocks:
                        circle = []
                        for line in block:
                            fields = [f.strip() for f in line.split("\t")]
                            if len(fields) < 7:
                                continue  # skip if not enough fields
                            ruler_name = fields[0].strip().lower() if fields[0] else ""
                            nation_name = fields[2].strip().lower() if len(fields) > 2 else ""
                            # Exclude any players in the filter set:
                            if ruler_name in filter_set or nation_name in filter_set:
                                continue
                            # Mark empty slots
                            if not fields[0] or fields[0].lower() == "x":
                                circle.append({
                                    "Ruler Name": None,
                                    "Resource 1+2": fields[1] if len(fields) > 1 else None,
                                    "Alliance": fields[2] if len(fields) > 2 else None,
                                    "Team": fields[3] if len(fields) > 3 else None,
                                    "Days Old": None,
                                    "Nation Drill Link": fields[5] if len(fields) > 5 else None,
                                    "Activity": fields[6] if len(fields) > 6 else None,
                                    "Empty": True
                                })
                            else:
                                try:
                                    days_old = float(fields[4])
                                except:
                                    days_old = None
                                player = {
                                    "Ruler Name": fields[0],
                                    "Resource 1+2": fields[1],
                                    "Alliance": fields[2],
                                    "Team": fields[3],
                                    "Days Old": days_old,
                                    "Nation Drill Link": fields[5],
                                    "Activity": fields[6],
                                    "Empty": False
                                }
                                if eligible(player):
                                    circle.append(player)
                        if circle:
                            pasted_circles.append(circle)
                    
                    # ---------------------------------------------------------------------
                    # Build a free pool from the filtered DataFrame (from Filter Data section)
                    # ---------------------------------------------------------------------
                    free_pool = []
                    if "filtered_df" in st.session_state:
                        df_free = st.session_state.filtered_df.copy()
                        df_free = df_free[df_free.apply(lambda row: eligible(row.to_dict()), axis=1)]
                        for idx, row in df_free.iterrows():
                            free_pool.append(row.to_dict())
                    
                    # ---------------------------------------------------------------------
                    # Categorize players by Peace Mode Level (A, B, C)
                    # ---------------------------------------------------------------------
                    def get_peace_mode_level(player):
                        try:
                            d = float(player.get("Days Old", 0))
                        except:
                            return None
                        if d < 1000:
                            return "A"
                        elif d < 2000:
                            return "B"
                        else:
                            return "C"
                    
                    # Add Peace Mode Level to each pasted player and free pool player:
                    for circle in pasted_circles:
                        for p in circle:
                            p["Peace Mode Level"] = get_peace_mode_level(p)
                    for p in free_pool:
                        p["Peace Mode Level"] = get_peace_mode_level(p)
                    
                    # Build category-specific tables to display for Levels A, B, and C:
                    levels = ["A", "B", "C"]
                    category_tables = {}
                    for lvl in levels:
                        players_lvl = []
                        # Add players from pasted circles (ignoring empty slots)
                        for circle in pasted_circles:
                            for p in circle:
                                if p.get("Peace Mode Level") == lvl and not p.get("Empty", False):
                                    players_lvl.append(p)
                        # Also add players from the free pool for completeness.
                        for p in free_pool:
                            if p.get("Peace Mode Level") == lvl:
                                players_lvl.append(p)
                        df_lvl = pd.DataFrame(players_lvl).drop_duplicates(subset=["Ruler Name", "Nation Drill Link"])
                        if not df_lvl.empty:
                            category_tables[lvl] = df_lvl.sort_values(by="Ruler Name", key=lambda col: col.str.lower()).reset_index(drop=True)
                            st.markdown(f"#### Peace Mode Level {lvl} Players")
                            st.dataframe(category_tables[lvl], use_container_width=True)
                    
                    # ---------------------------------------------------------------------
                    # Helper: Fill incomplete circles from a given pool using candidate scoring.
                    # (Assumes 'find_best_match' is defined as in the original script.)
                    # ---------------------------------------------------------------------
                    def fill_incomplete_circle(circle, pool, valid_combos):
                        # Get indices of empty slots in the circle.
                        empty_slots = [idx for idx, p in enumerate(circle) if p.get("Empty", False)]
                        for idx in empty_slots:
                            best_candidate = None
                            best_score = float('inf')
                            for candidate in pool:
                                # Skip if candidate already appears in this circle.
                                if any(candidate.get("Ruler Name") == existing.get("Ruler Name") for existing in circle if existing.get("Ruler Name")):
                                    continue
                                # Evaluate matching score based on candidate's Resource 1+2 vs. valid combos.
                                # (find_best_match returns a tuple: (best_combo, missing, extra, score))
                                best_combo, missing, extra, score = find_best_match(
                                    sorted([r.strip() for r in candidate.get("Resource 1+2", "").split(",") if r.strip()]),
                                    valid_combos
                                )
                                if score < best_score:
                                    best_score = score
                                    best_candidate = candidate
                            if best_candidate:
                                # Fill the slot and remove the candidate from the pool.
                                circle[idx] = best_candidate
                                pool.remove(best_candidate)
                        return circle
                    
                    # Dummy valid combinations for the three Peace Mode Levels.
                    # (In your actual code, these would be parsed from your text input areas.)
                    if "peace_a_combos" not in globals():
                        peace_a_combos = []
                    if "peace_b_combos" not in globals():
                        peace_b_combos = []
                    if "peace_c_combos" not in globals():
                        peace_c_combos = []
                    
                    # ---------------------------------------------------------------------
                    # Form New Trade Circles for Peace Mode
                    # ---------------------------------------------------------------------
                    new_circles = {lvl: [] for lvl in levels}
                    for lvl in levels:
                        # Process pasted circles where the (non-empty) players have the same peace mode level.
                        for circle in pasted_circles:
                            # Determine the circle’s level by checking the first filled player.
                            circle_levels = [p.get("Peace Mode Level") for p in circle if not p.get("Empty", False)]
                            if circle_levels and circle_levels[0] == lvl:
                                # If the circle has no empty slot, preserve it.
                                if not any(p.get("Empty", False) for p in circle):
                                    new_circles[lvl].append(circle)
                                else:
                                    # For incomplete circles, try to fill empty slots.
                                    pool_candidates = [p for p in free_pool if p.get("Peace Mode Level") == lvl]
                                    valid_combos = peace_a_combos if lvl == "A" else (peace_b_combos if lvl == "B" else peace_c_combos)
                                    filled_circle = fill_incomplete_circle(circle, pool_candidates, valid_combos)
                                    new_circles[lvl].append(filled_circle)
                        # For any remaining free players in this level not already in a circle, form new circles.
                        free_lvl = [p for p in free_pool if p.get("Peace Mode Level") == lvl]
                        while free_lvl:
                            circle = []
                            for i in range(TRADE_CIRCLE_SIZE):
                                if free_lvl:
                                    circle.append(free_lvl.pop(0))
                                else:
                                    circle.append({"Empty": True})
                            new_circles[lvl].append(circle)
                    
                    # Combine and sort all Peace Mode circles (complete ones first).
                    combined_peace_circles = []
                    for lvl in levels:
                        combined_peace_circles.extend(new_circles[lvl])
                    
                    def count_empty(circle):
                        return sum(1 for p in circle if p.get("Empty", False))
                    combined_peace_circles.sort(key=count_empty)
                    
                    # Within each circle, sort players alphabetically by Ruler Name.
                    for circle in combined_peace_circles:
                        circle.sort(key=lambda p: p.get("Ruler Name") if p.get("Ruler Name") else "")
                    
                    st.markdown("#### New Peace Mode Trade Circles")
                    for i, circle in enumerate(combined_peace_circles, start=1):
                        st.markdown(f"**Trade Circle {i} (Empty Slots: {count_empty(circle)})**")
                        display_trade_circle_df(circle, condition="Peace Mode")
                    
                    # ---------------------------------------------------------------------
                    # War Mode – Form Trade Circles without category constraints.
                    # ---------------------------------------------------------------------
                    st.markdown("### War Mode Trade Circle Matching")
                    # Create a war mode pool from remaining free players (or you might use a different source).
                    war_pool = free_pool.copy()
                    war_circles = []
                    while war_pool:
                        circle = []
                        for i in range(TRADE_CIRCLE_SIZE):
                            if war_pool:
                                circle.append(war_pool.pop(0))
                            else:
                                circle.append({"Empty": True})
                        war_circles.append(circle)
                    # (Optionally, use war_combos for candidate matching as done above.)
                    if "war_combos" not in globals():
                        war_combos = []
                    filled_war_circles = []
                    for circle in war_circles:
                        filled_circle = fill_incomplete_circle(circle, war_pool, war_combos)
                        filled_war_circles.append(filled_circle)
                    
                    st.markdown("#### New War Mode Trade Circles")
                    for i, circle in enumerate(filled_war_circles, start=1):
                        st.markdown(f"**War Mode Trade Circle {i} (Empty Slots: {count_empty(circle)})**")
                        display_trade_circle_df(circle, condition="War Mode")
                    
                    # ---------------------------------------------------------------------
                    # Identify Nations Not Placed in a Full Trade Circle.
                    # ---------------------------------------------------------------------
                    included_nations = set()
                    for circle in combined_peace_circles + filled_war_circles:
                        for p in circle:
                            if p.get("Ruler Name"):
                                included_nations.add(p.get("Ruler Name").lower())
                    remaining_players = []
                    for p in free_pool:
                        if p.get("Ruler Name", "").lower() not in included_nations:
                            remaining_players.append(p)
                    if remaining_players:
                        st.markdown("#### Nations Remaining Without a Full Trade Circle")
                        df_remaining = pd.DataFrame(remaining_players).sort_values(by="Ruler Name", key=lambda col: col.str.lower()).reset_index(drop=True)
                        st.dataframe(df_remaining, use_container_width=True)
                    
                    # ---------------------------------------------------------------------
                    # Trade Circle Coverage Summary
                    # ---------------------------------------------------------------------
                    total_peace_circles = len(combined_peace_circles)
                    total_war_circles = len(filled_war_circles)
                    total_circles = total_peace_circles + total_war_circles
                    total_nations_in_circles = sum(len([p for p in circle if p.get("Ruler Name")]) for circle in combined_peace_circles + filled_war_circles)
                    summary_data = {
                        "Total Peace Mode Circles": [total_peace_circles],
                        "Total War Mode Circles": [total_war_circles],
                        "Total Circles": [total_circles],
                        "Total Nations in Circles": [total_nations_in_circles],
                        "Total Nations Unmatched": [len(remaining_players)]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.markdown("#### Trade Circle Coverage Summary")
                    st.dataframe(summary_df, use_container_width=True)

                # -----------------------
                # EXPORT/WRITE EXCEL FILE FOR DOWNLOAD WITH ADDITIONAL WORKSHEETS
                # -----------------------
                sheets = {}  # Ensure sheets is initialized
                final_circles = combined_peace_circles + filled_war_circles

                # Build trade circle export entries based on final_circles
                trade_circle_entries = []
                for circle in final_circles:
                    circle_type = circle[0].get("Trade Circle Category", "Uncategorized")
                    trade_circle_id = ".".join([str(p.get('Nation ID', '')) for p in circle if p.get('Nation ID')])
                    for player in circle:
                        trade_circle_entries.append({
                            "Category": f"{circle_type} Recommended Trade Circle",
                            "Circle Type": circle_type,
                            "Trade Circle ID": trade_circle_id,
                            "Nation ID": player.get('Nation ID', ''),
                            "Ruler Name": player.get('Ruler Name', ''),
                            "Nation Name": player.get('Nation Name', ''),
                            "Team": player.get('Team', ''),
                            "Current Resources": player.get('Current Resources', ''),
                            "Current Resource 1+2": get_resource_1_2(player),
                            "Activity": player.get('Activity', ''),
                            "Days Old": player.get('Days Old', ''),
                            "Assigned Resources": ", ".join(player.get('Assigned Resource 1+2', [])) if player.get('Assigned Resource 1+2') else "None"
                        })
                
                if trade_circle_entries:
                    trade_circle_df = pd.DataFrame(trade_circle_entries)
                    sheets["Trade Circles"] = add_nation_drill_url(trade_circle_df)
                    
                # -----------------------
                # GENERATE MESSAGE TEMPLATES FOR TRADE CIRCLES FROM final_circles
                # -----------------------
                def generate_trade_circle_messages(circles):
                    # First, filter out any empty circles.
                    non_empty_circles = [circle for circle in circles if circle]
                    messages = []
                    for circle in non_empty_circles:
                        circle_type = circle[0].get("Trade Circle Category", "Uncategorized")
                        nation_names = [player.get('Ruler Name', '') for player in circle if player.get('Ruler Name')]
                        for player in circle:
                            partners = [name for name in nation_names if name != player.get('Ruler Name', '')]
                            msg = (
                                f"To The Ruler: {player.get('Ruler Name', '')}, please join a Trade Circle with partners: "
                                f"{', '.join(partners)}. Your assigned resource pair is "
                                f"{', '.join(player.get('Assigned Resource 1+2', [])) if player.get('Assigned Resource 1+2') else 'None'}. -Lord of Growth."
                            )
                            messages.append({"Message Type": f"{circle_type} Trade Circle", "Message": msg})
                    return messages
                
                message_entries = generate_trade_circle_messages(final_circles)
                
                if message_entries:
                    messages_df = pd.DataFrame(message_entries)
                    sheets["Message Templates"] = messages_df.copy()

                # -----------------------
                # COMPARATIVE ALLIANCE STATS (EXAMPLE)
                # -----------------------
                if "Alliance" in df.columns:
                    original_df = st.session_state.df.copy()
                    if "Alliance Status" in original_df.columns:
                        original_df = original_df[original_df["Alliance Status"] != "Pending"]
                    # Filter out rows where "Alliance" is exactly "None"
                    original_df = original_df[original_df["Alliance"] != "None"]
                    resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
                    mask_empty_all = original_df[resource_cols].isnull().any(axis=1) | (
                        original_df[resource_cols].apply(lambda col: col.astype(str).str.strip() == '').any(axis=1)
                    )
                    players_empty_all = original_df[mask_empty_all].copy()
                    players_full_all = original_df[~mask_empty_all].copy()
                
                    alliances = original_df['Alliance'].unique()
                    comp_stats = []
                    for alliance in alliances:
                        total_players = len(original_df[original_df['Alliance'] == alliance])
                        empty_players = len(players_empty_all[players_empty_all['Alliance'] == alliance])
                        full_players = len(players_full_all[players_full_all['Alliance'] == alliance])
                        empty_percentage = (empty_players / total_players * 100) if total_players else 0
                
                        comp_stats.append({
                            "Alliance": alliance,
                            "Total Alliance Members": total_players,
                            "Players with Empty Trade Slots": empty_players,
                            "Empty Trade Slot (%)": empty_percentage,
                            "Players in Complete Trade Circle": full_players,
                        })
                    
                    comp_stats_df = pd.DataFrame(comp_stats)
                    comp_stats_df = comp_stats_df.sort_values("Empty Trade Slot (%)", ascending=True).reset_index(drop=True)
                    comp_stats_df.index = comp_stats_df.index + 1
                
                    with st.expander("Comparative Alliance Stats"):
                        st.dataframe(comp_stats_df.style.format({"Empty Trade Slot (%)": "{:.2f}%"}), use_container_width=True)
                
                    sheets["Comparative Alliance Stats"] = comp_stats_df.copy()

                # -----------------------
                # SUMMARY OVERVIEW SECTION (UI)
                # -----------------------
                with st.expander("Summary Overview"):
                    # Determine the heading based on the Alliance filter selection stored in session state.
                    if "selected_alliances" in st.session_state:
                        alliances = st.session_state.selected_alliances
                    else:
                        alliances = []
                
                    if not alliances or len(alliances) == 0:
                        heading_alliance = "Alliances"
                    elif len(alliances) == 1:
                        heading_alliance = alliances[0]
                    else:
                        heading_alliance = ", ".join(alliances)
                    
                    st.subheader(f"General Statistics for {heading_alliance}")

                    # Total players in either group (empty slots + complete)
                    total_players = len(players_empty) + len(players_full)
                    empty_percentage = (len(players_empty) / total_players * 100) if total_players else 0

                    # For players in complete trade circles, count unique mismatches using consolidated DataFrames.
                    total_full = len(players_full)
                    unique_peacetime_mismatch = peacetime_df['Nation ID'].nunique() if not peacetime_df.empty else 0
                    unique_wartime_mismatch = wartime_df['Nation ID'].nunique() if not wartime_df.empty else 0
                    peacetime_mismatch_percentage = (unique_peacetime_mismatch / total_full * 100) if total_full else 0
                    wartime_mismatch_percentage = (unique_wartime_mismatch / total_full * 100) if total_full else 0

                    st.write(f"**Total Alliance Members:** {total_players} (Not Including Pending)")
                    st.write(f"**Members with Empty Trade Slots:** {len(players_empty)} ({empty_percentage:.2f}%)")
                    st.write(f"**Members in Complete Trade Circle:** {total_full}")
                    
                    # Break down the peacetime mismatches by Peace Mode Level
                    unique_peaceA = df_peace_a['Nation ID'].nunique() if not df_peace_a.empty else 0
                    unique_peaceB = df_peace_b['Nation ID'].nunique() if not df_peace_b.empty else 0
                    unique_peaceC = df_peace_c['Nation ID'].nunique() if not df_peace_c.empty else 0
                    
                    total_peace_mismatch = unique_peaceA + unique_peaceB + unique_peaceC
                    
                    st.markdown("#### Peacetime Mismatches Breakdown by Level")
                    st.write(f"- **Level A** (< 1000 days old): **{unique_peaceA}**")
                    st.write(f"- **Level B** (1000 to 2000 days old): **{unique_peaceB}**")
                    st.write(f"- **Level C** (>= 2000 days old): **{unique_peaceC}**")
                    st.write(f"Total Peacetime Mismatch among Complete Trade Circles: {total_peace_mismatch} ({peacetime_mismatch_percentage:.2f}%)")
                    st.write(f"Wartime Mismatch among Complete Trade Circles: {unique_wartime_mismatch} ({wartime_mismatch_percentage:.2f}%)")
                    st.markdown('---')

                    st.subheader("Action Plan for Alliance Management")
                    action_plan = textwrap.dedent("""\
                    **1. Identify Affected Trade Circles:**
                    - Review the **Peacetime Resource Mismatches** and **Wartime Resource Mismatches** reports.
                    - For each entry, note the following:
                      - **Player Identification:** Nation Name, Nation ID, Ruler Name.
                      - **Resources:** The extra resources (listed under *Extra Resources*) and the missing resources (listed under *Missing Resources*).
    
                    **2. Notify Affected Players:**
                    - For each player with a peacetime mismatch, send a message:
                      - *"To The Ruler: [Ruler Name], your Trade Circle currently has mismatched/duplicate resource(s) [Extra Resources] which must be exchanged for the missing resource(s) [Missing Resources] to meet peacetime trade requirements. Please can you either change your resources or get in contact with your trade partners. -Lord of Growth."*
                    - For each player with a wartime mismatch, send a similar message.
    
                    **3. Reconfigure Incomplete Trade Circles:**
                    - Review the **Players with Empty Trade Slots** report and arrange a meeting for any player not in a full trade circle.
    
                    **4. Document and Follow-Up:**
                    - Log each notification with details for follow-up.
                    """)
                    st.markdown(action_plan)

                # -----------------------
                # WRITE EXCEL FILE FOR DOWNLOAD
                # -----------------------
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df_sheet in sheets.items():
                        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    workbook = writer.book
                    from openpyxl.utils.dataframe import dataframe_to_rows
                    from openpyxl.styles import Font, Border, Side, Alignment
                    header_font = Font(bold=True)
                    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                                         top=Side(style='thin'), bottom=Side(style='thin'))
                    
                    def format_header(ws):
                        for cell in ws[1]:
                            cell.font = header_font
                            cell.border = thin_border
                    
                    # Add any additional formatting or worksheets if needed.
                    # No need to call writer.save() explicitly within the context.
                    
                output.seek(0)
                excel_data = output.read()
                # -----------------------
                # DOWNLOAD ALL DATA EXCEL (positioned at the bottom of the page)
                # -----------------------
                st.markdown("### Download All Processed Data")
                if excel_data:
                    st.download_button("Download Summary Report", excel_data, 
                                       file_name="full_summary_report.xlsx", 
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                       key="download_report")
                else:
                    st.info("No data available for download.")
    else:
        st.info("Please enter the correct password to access the functionality.")

if __name__ == "__main__":
    main()
