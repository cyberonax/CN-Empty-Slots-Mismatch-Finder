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

# -----------------------
# CONFIGURATION & CONSTANTS
# -----------------------
st.set_page_config(layout="wide")
# Resource sets for Peacetime and Wartime (12 resources each)
peacetime_resources = [
    'Aluminum', 'Cattle', 'Fish', 'Iron', 'Lumber',
    'Marble', 'Pigs', 'Spices', 'Sugar', 'Uranium',
    'Water', 'Wheat'
]
wartime_resources = [
    'Aluminum', 'Coal', 'Fish', 'Gold', 'Iron',
    'Lead', 'Lumber', 'Marble', 'Oil', 'Pigs',
    'Rubber', 'Uranium'
]

sorted_peacetime = sorted(peacetime_resources)
sorted_wartime = sorted(wartime_resources)

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

def get_current_resources(row, resource_cols):
    """Return a comma-separated string of non-blank resources sorted alphabetically."""
    resources = sorted([str(x).strip() for x in row[resource_cols] if pd.notnull(x) and str(x).strip() != ''])
    return ", ".join(resources)

def count_empty_slots(row, resource_cols):
    """Count blank resource cells and determine trade slots (each slot covers 2 resources)."""
    count = sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '')
    return count // 2

# -----------------------
# IMPROVED TRADE CIRCLE FORMATION LOGIC
# -----------------------
def form_trade_circles(players, recommended_resources, circle_size=TRADE_CIRCLE_SIZE):
    """
    Incremental Trade Circle Formation (improved):

    - Preserves previously established trade circles if a 'Trade Circle ID' is present.
    - Merges partial circles with unassigned players to fill empty slots.
    - First attempts to form circles based solely on players' existing "Current Resource 1+2" values.
      For a valid circle, the 6 players’ resource pairs (each exactly 2 resources) must be disjoint and,
      when combined, exactly equal the recommended resource list.
      In such a case, Assigned Resources remains None.
    - If no feasible combination exists from the players with valid resource pairs,
      then the remaining players are grouped in descending order of Days Old and assigned resources from the recommended list.
    
    Returns a tuple: (complete_trade_circles, leftover_players)
    """
    # Dictionaries to hold circles keyed by Trade Circle ID and a list for unassigned players.
    circles = {}
    unassigned = []
    
    # --- Step 1: Preserve existing circles (if any) ---
    for player in players:
        if player.get('Trade Circle ID'):
            tid = player.get('Trade Circle ID')
            circles.setdefault(tid, []).append(player)
        else:
            unassigned.append(player)
    
    # --- Step 2: Fill missing slots for partially complete circles ---
    for tid, members in list(circles.items()):
        if len(members) < circle_size:
            needed = circle_size - len(members)
            # Take as many as available from unassigned players.
            to_add = unassigned[:needed]
            if to_add:
                members.extend(to_add)
                unassigned = unassigned[needed:]
            # Recalculate Trade Circle ID based on sorted Nation IDs so that the ID is consistent.
            sorted_ids = sorted([str(player.get('Nation ID','')) for player in members])
            new_tid = ".".join(sorted_ids)
            for player in members:
                # For preserved circles, do not reassign resources.
                # They keep their existing Assigned Resources.
                player['Trade Circle ID'] = new_tid
            # Update dictionary with new circle id (if changed)
            if new_tid != tid:
                circles[new_tid] = circles.pop(tid)
    
    # --- Step 3: Form new circles from remaining unassigned players ---
    if unassigned:
        # First, try to form circles using players' existing "Current Resource 1+2" values.
        # We require that each player's "Current Resource 1+2" parses to exactly 2 resources,
        # and that these resources (when combined with others in the circle) exactly match the recommended list.
        recommended_set = set(recommended_resources)
        
        # Build a pool of players with valid current resource pairs (both non-empty and exactly 2 resources)
        valid_pool = []
        invalid_pool = []
        for player in unassigned:
            rpair_str = player.get('Current Resource 1+2', '')
            parts = [res.strip() for res in rpair_str.split(",") if res.strip()]
            if len(parts) == 2:
                # Only include if the pair is a subset of the recommended set.
                if set(parts).issubset(recommended_set):
                    # Also add the parsed resource pair to the player dict for easier checking later.
                    player['_parsed_resource_pair'] = set(parts)
                    valid_pool.append(player)
                else:
                    invalid_pool.append(player)
            else:
                invalid_pool.append(player)
        
        # Use backtracking to try to form circles from valid_pool.
        def find_exact_circle(pool, current_list, start, current_union):
            if len(current_list) == circle_size:
                if current_union == recommended_set:
                    return list(current_list)
                return None
            for i in range(start, len(pool)):
                player = pool[i]
                pair = player['_parsed_resource_pair']
                # They must be disjoint with the current union.
                if current_union & pair:
                    continue
                new_union = current_union | pair
                # If new_union already goes outside recommended_set, skip.
                if not new_union.issubset(recommended_set):
                    continue
                current_list.append(player)
                result = find_exact_circle(pool, current_list, i + 1, new_union)
                if result:
                    return result
                current_list.pop()
            return None

        matching_circles = []
        remaining_valid_pool = valid_pool.copy()
        while True:
            circle = find_exact_circle(remaining_valid_pool, [], 0, set())
            if circle:
                # For circles formed using the existing pairs, do not assign new resources.
                sorted_ids = sorted([str(player.get('Nation ID', '')) for player in circle])
                new_tid = ".".join(sorted_ids)
                for player in circle:
                    player['Trade Circle ID'] = new_tid
                    player['Assigned Resources'] = None
                matching_circles.append(circle)
                # Remove the players in this circle from the valid pool.
                for player in circle:
                    remaining_valid_pool.remove(player)
            else:
                break
        
        # Remove the players used in matching circles from unassigned.
        used_in_matching = set()
        for circle in matching_circles:
            for player in circle:
                used_in_matching.add(player.get('Nation ID'))
        remaining_unassigned = []
        for player in unassigned:
            # Remove temporary helper if present.
            if '_parsed_resource_pair' in player:
                del player['_parsed_resource_pair']
            if player.get('Nation ID') not in used_in_matching:
                remaining_unassigned.append(player)
        
        # Now form default circles from remaining players (including those that were not eligible for matching)
        # Sorted by Days Old (oldest first).
        remaining_unassigned = sorted(remaining_unassigned, key=lambda p: p.get('Days Old', 0), reverse=True)
        default_circles = []
        current_circle = []
        for player in remaining_unassigned:
            current_circle.append(player)
            if len(current_circle) == circle_size:
                default_circles.append(current_circle)
                current_circle = []
        # FIX: If there is a partially complete group remaining, add them to circles under a new key so they are not lost.
        if current_circle:
            merged = False
            for tid, members in circles.items():
                if len(members) < circle_size:
                    members.extend(current_circle)
                    merged = True
                    break
            if not merged:
                circles["leftover"] = circles.get("leftover", []) + current_circle
        
        # For each default circle, assign new resources based on the recommended list.
        for circle in default_circles:
            sorted_ids = sorted([str(player.get('Nation ID','')) for player in circle])
            new_tid = ".".join(sorted_ids)
            for j, player in enumerate(circle):
                player['Assigned Resources'] = recommended_resources[2*j:2*j+2] if len(recommended_resources) >= 2*(j+1) else []
                player['Trade Circle ID'] = new_tid
            circles[new_tid] = circle
        
        # Also add the matching circles to the circles dictionary.
        for circle in matching_circles:
            tid = circle[0].get('Trade Circle ID')
            circles[tid] = circle
        
        # Retrieve any leftovers stored in circles.
        leftover = circles.get("leftover", [])
    else:
        leftover = []
    
    # --- Step 4: Compile complete circles and leftovers ---
    complete_circles = []
    incomplete_players = []
    for circle in circles.values():
        if circle and len(circle) == circle_size:
            complete_circles.append(circle)
        elif circle and len(circle) != circle_size:
            incomplete_players.extend(circle)
            
    # If there are still enough leftover players among themselves to form a full circle, group them.
    if incomplete_players and len(incomplete_players) >= circle_size:
        extra_groups = [incomplete_players[i:i+circle_size] for i in range(0, len(incomplete_players), circle_size)
                        if len(incomplete_players[i:i+circle_size]) == circle_size]
        complete_circles.extend(extra_groups)
        remainder = incomplete_players[len(extra_groups)*circle_size:]
    else:
        remainder = incomplete_players

    return complete_circles, remainder

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
        st.error(f"Error downloading file from {url}: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_list = z.namelist()
        if not file_list:
            st.error("The zip file is empty.")
            return None
        file_name = file_list[0]
        with z.open(file_name) as file:
            try:
                # Adjust delimiter and encoding as needed
                df = pd.read_csv(file, delimiter="|", encoding="ISO-8859-1")
                return df
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
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
    
    # Password protection block
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = False

    if not st.session_state.password_verified:
        # Added key to preserve the value in session state
        password = st.text_input("Enter Password", type="password", key="password_input")
        if password:
            if password == "secret":
                st.session_state.password_verified = True
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
                    # Add keys to preserve filter selections.
                    selected_column = st.selectbox("Select column to filter", column_options, index=default_index, key="filter_select")
                    search_text = st.text_input("Filter by text (separate words by comma)", value="Freehold of the Wolves", key="filter_text")
                            
                    if search_text:
                        # Support multiple filters separated by comma
                        filters = [f.strip() for f in search_text.split(",") if f.strip()]
                        pattern = "|".join(filters)
                        filtered_df = df[df[selected_column].astype(str).str.contains(pattern, case=False, na=False)]
                        st.write(f"Showing results where **{selected_column}** contains any of {filters}:")
                        # Use the filtered table's own "Resource 1" and "Resource 2" columns for current resource display
                        if "Resource 1" in filtered_df.columns and "Resource 2" in filtered_df.columns:
                            filtered_df = filtered_df.copy()
                            filtered_df['Current Resource 1+2'] = filtered_df.apply(lambda row: get_resource_1_2(row), axis=1)
                        st.dataframe(filtered_df, use_container_width=True)
                        # Save the filtered DataFrame and CSV content to session state for later use.
                        st.session_state.filtered_df = filtered_df
                        csv_content = filtered_df.to_csv(index=False)
                        st.session_state.filtered_csv = csv_content
                        st.download_button("Download Filtered CSV", csv_content, file_name="filtered_nation_stats.csv", mime="text/csv", key="download_csv")
                    else:
                        st.info("Enter text to filter the data.")
                else:
                    st.warning("DataFrame is empty, nothing to filter.")

            # -----------------------
            # TRADE CIRCLE & RESOURCE PROCESSING (automatically triggered)
            # -----------------------
            if "df" in st.session_state:
                # Instead of using st.session_state.filtered_df, reload the filtered CSV if available
                if "filtered_csv" in st.session_state:
                    filtered_csv = st.session_state.filtered_csv
                    # Read the CSV content into a DataFrame
                    df_to_use = pd.read_csv(io.StringIO(filtered_csv))
                else:
                    df_to_use = df

                # Assume that the resource columns are named "Connected Resource 1" to "Connected Resource 10"
                resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
                # Identify players with at least one blank in any resource column
                mask_empty = df_to_use[resource_cols].isnull().any(axis=1) | (
                    df_to_use[resource_cols].apply(lambda col: col.astype(str).str.strip() == '').any(axis=1)
                )
                players_empty = df_to_use[mask_empty].copy()

                # Compute "Current Resources" column (for full resource list)
                players_empty['Current Resources'] = players_empty.apply(lambda row: get_current_resources(row, resource_cols), axis=1)
                # Now use the CSV's Resource 1 and Resource 2 for the Current Resource 1+2 column
                players_empty['Current Resource 1+2'] = players_empty.apply(lambda row: get_resource_1_2(row), axis=1)
                # Compute empty trade slots (each slot covers 2 resources)
                players_empty['Empty Slots Count'] = players_empty.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
                # Convert "Created" to datetime and compute age in days (optional, still displayed)
                date_format = "%m/%d/%Y %I:%M:%S %p"  # Adjust if necessary
                players_empty['Created'] = pd.to_datetime(players_empty['Created'], format=date_format, errors='coerce')
                current_date = pd.to_datetime("now")
                players_empty['Days Old'] = (current_date - players_empty['Created']).dt.days

                # Filter out players who are inactive based on the "Activity" column.
                # Only include players whose Activity is NOT "Active Three Weeks Ago" or "Active More Than Three Weeks Ago"
                players_empty = players_empty[~players_empty['Activity'].isin(["Active Three Weeks Ago", "Active More Than Three Weeks Ago"])]

                # ---- New Filter: Exclude players with Alliance Status "Pending" ----
                if "Alliance Status" in players_empty.columns:
                    players_empty = players_empty[players_empty["Alliance Status"] != "Pending"]

                # -----------------------
                # PLAYERS WITH EMPTY TRADE SLOTS
                # -----------------------
                with st.expander("Players with empty trade slots (active recently)"):
                    display_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Team', 'Current Resources', 'Current Resource 1+2', 'Empty Slots Count', 'Activity', 'Days Old']
                    st.dataframe(players_empty[display_cols].reset_index(drop=True), use_container_width=True)
                
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
                    # Process "Created" and "Days Old"
                    players_full['Created'] = pd.to_datetime(players_full['Created'], format=date_format, errors='coerce')
                    players_full['Days Old'] = (current_date - players_full['Created']).dt.days

                    st.dataframe(players_full[display_cols].reset_index(drop=True), use_container_width=True)
                
                # -----------------------
                # RESOURCE MISMATCHES
                # -----------------------
                with st.expander("Resource Mismatches"):
                    peacetime_mismatch = []
                    wartime_mismatch = []
                
                    for idx, row in players_full.iterrows():
                        # Parse current resources from the "Current Resources" column
                        current_resources = [res.strip() for res in row['Current Resources'].split(',') if res.strip()]
                        # If the row contains "Resource 1" and "Resource 2", add them as well
                        if "Resource 1" in row and pd.notnull(row["Resource 1"]):
                            res1 = str(row["Resource 1"]).strip()
                            if res1 and res1 not in current_resources:
                                current_resources.append(res1)
                        if "Resource 2" in row and pd.notnull(row["Resource 2"]):
                            res2 = str(row["Resource 2"]).strip()
                            if res2 and res2 not in current_resources:
                                current_resources.append(res2)
                                        
                        # Calculate duplicate resources
                        duplicates = [res for res, count in Counter(current_resources).items() if count > 1]
                        dup_str = ", ".join(sorted(duplicates)) if duplicates else "None"
                                        
                        current_set = set(current_resources)
                        peacetime_set = set(peacetime_resources)
                        wartime_set = set(wartime_resources)
                                        
                        missing_peace = peacetime_set - current_set
                        extra_peace = current_set - peacetime_set
                        missing_war = wartime_set - current_set
                        extra_war = current_set - wartime_set
                                        
                        # Only add to the list if there is a mismatch for peacetime resources
                        if missing_peace or extra_peace:
                            peacetime_mismatch.append({
                                'Nation ID': row['Nation ID'],
                                'Ruler Name': row['Ruler Name'],
                                'Nation Name': row['Nation Name'],
                                'Current Resources': row['Current Resources'],
                                'Current Resource 1+2': get_resource_1_2(row),
                                'Duplicate Resources': dup_str,
                                'Missing Peacetime Resources': ", ".join(sorted(missing_peace)) if missing_peace else "None",
                                'Extra Resources': ", ".join(sorted(extra_peace)) if extra_peace else "None",
                                'Activity': row['Activity'],
                                'Days Old': row['Days Old']
                            })
                                        
                        # Only add to the list if there is a mismatch for wartime resources
                        if missing_war or extra_war:
                            wartime_mismatch.append({
                                'Nation ID': row['Nation ID'],
                                'Ruler Name': row['Ruler Name'],
                                'Nation Name': row['Nation Name'],
                                'Current Resources': row['Current Resources'],
                                'Current Resource 1+2': get_resource_1_2(row),
                                'Duplicate Resources': dup_str,
                                'Missing Wartime Resources': ", ".join(sorted(missing_war)) if missing_war else "None",
                                'Extra Resources': ", ".join(sorted(extra_war)) if extra_war else "None",
                                'Activity': row['Activity'],
                                'Days Old': row['Days Old']
                            })
                                        
                    st.markdown("**Peacetime Resource Mismatches:**")
                    peacetime_df = pd.DataFrame(peacetime_mismatch).reset_index(drop=True)
                    
                    # Only style and display if there's data
                    if not peacetime_df.empty:
                        subset_cols = ['Duplicate Resources', 'Extra Resources']
                        # Get only the subset columns that are present
                        existing_subset = [col for col in subset_cols if col in peacetime_df.columns]
                        if existing_subset:
                            styled_peace = peacetime_df.style.applymap(highlight_none, subset=existing_subset)
                            st.dataframe(styled_peace, use_container_width=True)
                        else:
                            st.dataframe(peacetime_df, use_container_width=True)
                    else:
                        st.info("No peacetime resource mismatches found.")
                    
                    st.markdown("**Wartime Resource Mismatches:**")
                    wartime_df = pd.DataFrame(wartime_mismatch).reset_index(drop=True)
                    
                    if not wartime_df.empty:
                        existing_subset_war = [col for col in subset_cols if col in wartime_df.columns]
                        if existing_subset_war:
                            styled_war = wartime_df.style.applymap(highlight_none, subset=existing_subset_war)
                            st.dataframe(styled_war, use_container_width=True)
                        else:
                            st.dataframe(wartime_df, use_container_width=True)
                    else:
                        st.info("No wartime resource mismatches found.")

                # -----------------------
                # RECOMMENDED TRADE CIRCLES
                # -----------------------
                with st.expander("Recommended Trade Circles"):
                    # Sort players by Nation ID (or another criterion) from the empty slots list
                    players_empty_sorted = players_empty.sort_values('Nation ID')
                    players_list = players_empty_sorted.to_dict('records')
                    
                    # -----------------------
                    # FIX: Use deep copies for separate trade circle formations
                    # -----------------------
                    players_list_peace = copy.deepcopy(players_list)
                    players_list_war = copy.deepcopy(players_list)
                    
                    # Use the improved formation logic:
                    trade_circles_peace, leftover_peace = form_trade_circles(players_list_peace, sorted_peacetime)
                    trade_circles_war, leftover_war = form_trade_circles(players_list_war, sorted_wartime)
                
                    # Display Peacetime Trade Circles
                    if trade_circles_peace:
                        st.markdown("**Recommended Peacetime Trade Circles:**")
                        for idx, circle in enumerate(trade_circles_peace, start=1):
                            st.markdown(f"--- **Peacetime Trade Circle #{idx}** ---")
                            display_trade_circle_df(circle, "Peacetime")
                    else:
                        st.info("No full Peacetime trade circles could be formed.")
                
                    # Display Wartime Trade Circles
                    if trade_circles_war:
                        st.markdown("**Recommended Wartime Trade Circles:**")
                        for idx, circle in enumerate(trade_circles_war, start=1):
                            st.markdown(f"--- **Wartime Trade Circle #{idx}** ---")
                            display_trade_circle_df(circle, "Wartime")
                    else:
                        st.info("No full Wartime trade circles could be formed.")
                
                    # Display leftover players (if any) from both approaches
                    # Combine leftovers from both formations and remove duplicates based on Nation ID.
                    combined_leftovers = leftover_peace + leftover_war
                    unique_leftovers = {player.get('Nation ID'): player for player in combined_leftovers}.values()
                    
                    if unique_leftovers:
                        leftover_data = []
                        for player in unique_leftovers:
                            leftover_data.append({
                                'Nation ID': player.get('Nation ID', ''),
                                'Ruler Name': player.get('Ruler Name', ''),
                                'Nation Name': player.get('Nation Name', ''),
                                'Team': player.get('Team', ''),
                                'Current Resources': player.get('Current Resources', ''),
                                'Current Resource 1+2': get_resource_1_2(player),
                                'Activity': player.get('Activity', ''),
                                'Days Old': player.get('Days Old', '')
                            })
                        st.markdown("**Players remaining without a full trade circle:**")
                        st.dataframe(pd.DataFrame(leftover_data), use_container_width=True)
                    else:
                        st.success("All players have been grouped into trade circles.")

                # -----------------------
                # PREPARE DATA FOR EXCEL DOWNLOAD WITH SEPARATE WORKSHEETS
                # -----------------------
                # Function to add Nation Drill URL column
                def add_nation_drill_url(df):
                    df = df.copy()
                    if 'Nation ID' in df.columns:
                        df['Nation Drill URL'] = "https://www.cybernations.net/nation_drill_display.asp?Nation_ID=" + df['Nation ID'].astype(str)
                    return df

                sheets = {}
                # Empty slots data
                empty_slots_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Team', 'Current Resources', 'Current Resource 1+2', 'Empty Slots Count', 'Activity', 'Days Old']
                empty_slots_df = players_empty[empty_slots_cols].copy()
                empty_slots_df['Category'] = 'Empty Slots'
                sheets["Empty Slots"] = add_nation_drill_url(empty_slots_df)

                # Complete trade circles data
                complete_slots_df = players_full[empty_slots_cols].copy()
                complete_slots_df['Category'] = 'Complete Trade Circle'
                sheets["Complete Trade Circle"] = add_nation_drill_url(complete_slots_df)

                # Mismatched resources data - peacetime
                if not peacetime_df.empty:
                    peacetime_df_copy = peacetime_df.copy()
                    peacetime_df_copy['Category'] = 'Peacetime Resource Mismatch'
                    sheets["Peacetime Mismatch"] = add_nation_drill_url(peacetime_df_copy)
                # Mismatched resources data - wartime
                if not wartime_df.empty:
                    wartime_df_copy = wartime_df.copy()
                    wartime_df_copy['Category'] = 'Wartime Resource Mismatch'
                    sheets["Wartime Mismatch"] = add_nation_drill_url(wartime_df_copy)

                # Recommended trade circles data
                trade_circle_entries = []
                for circle_type, circles in [("Peacetime", trade_circles_peace), ("Wartime", trade_circles_war)]:
                    if circles:
                        for circle in circles:
                            # Retrieve the Trade Circle ID from the first player in the circle
                            trade_circle_id = circle[0].get('Trade Circle ID', '')
                            for player in circle:
                                trade_circle_entries.append({
                                    "Category": f"{circle_type} Recommended Trade Circle",
                                    "Circle Type": circle_type,
                                    "Trade Circle ID": trade_circle_id,  # Updated column instead of Circle Number
                                    "Nation ID": player.get('Nation ID', ''),
                                    "Ruler Name": player.get('Ruler Name', ''),
                                    "Nation Name": player.get('Nation Name', ''),
                                    "Team": player.get('Team', ''),
                                    "Current Resources": player.get('Current Resources', ''),
                                    "Current Resource 1+2": get_resource_1_2(player),
                                    "Activity": player.get('Activity', ''),
                                    "Days Old": player.get('Days Old', ''),
                                    "Assigned Resources": ", ".join(player.get('Assigned Resources', [])) if player.get('Assigned Resources') else "None"
                                })
                if trade_circle_entries:
                    trade_circle_df = pd.DataFrame(trade_circle_entries)
                    sheets["Trade Circles"] = add_nation_drill_url(trade_circle_df)

                # -----------------------
                # COMPARATIVE STATISTICS (NEW SECTION)
                # -----------------------
                # We compute per-alliance stats using the original data (st.session_state.df)
                if "Alliance" in df.columns:
                    original_df = st.session_state.df.copy()
                    resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
                    # Determine players with empty trade slots in the original data
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
                        # Empty Trade Slot (%) stored as a numerical value.
                        empty_percentage = (empty_players / total_players * 100) if total_players else 0

                        comp_stats.append({
                            "Alliance": alliance,
                            "Total Alliance Members": total_players,
                            "Players with Empty Trade Slots": empty_players,
                            "Empty Trade Slot (%)": empty_percentage,
                            "Players in Complete Trade Circle": full_players,
                        })
                    
                comp_stats_df = pd.DataFrame(comp_stats)
                # Sort by the percentage column in ascending order.
                comp_stats_df = comp_stats_df.sort_values("Empty Trade Slot (%)", ascending=True).reset_index(drop=True)
                comp_stats_df.index = comp_stats_df.index + 1
                
                with st.expander("Comparative Alliance Stats"):
                    st.dataframe(comp_stats_df.style.format({"Empty Trade Slot (%)": "{:.2f}%"}), use_container_width=True)

                    # Add Comparative Alliance Stats to the Excel sheets
                    sheets["Comparative Alliance Stats"] = comp_stats_df.copy()

                # -----------------------
                # SUMMARY OVERVIEW SECTION (UI)
                # -----------------------
                with st.expander("Summary Overview"):
                    selected_alliance = ""
                    if "filtered_df" in st.session_state and "Alliance" in st.session_state.filtered_df.columns:
                        # Remove any null values and get unique alliances
                        alliances = st.session_state.filtered_df["Alliance"].dropna().unique()
                        if len(alliances) == 1:
                            selected_alliance = alliances[0]
                        elif len(alliances) > 1:
                            selected_alliance = "Multiple Alliances: " + ", ".join(alliances)
                        else:
                            selected_alliance = "All Alliances"
                    else:
                        selected_alliance = "All Alliances"
                        
                    st.subheader(f"General Statistics for {selected_alliance}")

                    # Total players in either group (empty slots + complete)
                    total_players = len(players_empty) + len(players_full)
                    empty_percentage = (len(players_empty) / total_players * 100) if total_players else 0

                    # For players in complete trade circles, count unique mismatches
                    total_full = len(players_full)
                    unique_peacetime_mismatch = peacetime_df['Nation ID'].nunique() if not peacetime_df.empty else 0
                    unique_wartime_mismatch = wartime_df['Nation ID'].nunique() if not wartime_df.empty else 0
                    peacetime_mismatch_percentage = (unique_peacetime_mismatch / total_full * 100) if total_full else 0
                    wartime_mismatch_percentage = (unique_wartime_mismatch / total_full * 100) if total_full else 0

                    st.write(f"**Total Players (Empty + Complete):** {total_players}")
                    st.write(f"**Players with Empty Trade Slots:** {len(players_empty)} ({empty_percentage:.2f}%)")
                    st.write(f"**Players in Complete Trade Circle:** {total_full}")
                    st.write(f"**Peacetime Mismatch among Complete Trade Circles:** {unique_peacetime_mismatch} ({peacetime_mismatch_percentage:.2f}%)")
                    st.write(f"**Wartime Mismatch among Complete Trade Circles:** {unique_wartime_mismatch} ({wartime_mismatch_percentage:.2f}%)")
                    st.markdown('---')

                    st.subheader("Action Plan for Alliance Management")
                    # Use dedent to ensure proper bullet point formatting
                    action_plan = textwrap.dedent("""
                    **1. Identify Affected Trade Circles:**
                    - Review the **Peacetime Resource Mismatches** and **Wartime Resource Mismatches** reports.
                    - For each entry, note the following:
                      - **Player Identification:** Nation Name, Nation ID, Ruler Name.
                      - **Resources:** The extra resources (listed under *Extra Resources*) and the missing resources (listed under *Missing Peacetime Resources* or *Missing Wartime Resources*).

                    **2. Notify Affected Players:**
                    - For each player with a peacetime mismatch, send a message:
                      - *"To The Ruler: [Ruler Name], your Trade Circle currently has mismatched/duplicate resource(s) [list Extra Resources] which must be exchanged for the missing resource(s) [list Missing Peacetime Resources] to meet peacetime trade requirements. Please can you either change your resources or get in contact with your trade partners to coordinate adjustments to your agreements. Peacetime resources must be: Aluminum, Cattle, Fish, Iron, Lumber, Marble, Pigs, Spices, Sugar, Uranium, Water, Wheat. -Lord of Growth."*
                    - For each player with a wartime mismatch, send a similar message:
                      - *"To The Ruler: [Ruler Name], your Trade Circle currently has mismatched/duplicate resource(s) [list Extra Resources] which must be exchanged for the missing resource(s) [list Missing Wartime Resources] to meet wartime trade requirements. Please can you either change your resources or get in contact with your trade partners to coordinate adjustments to your agreements. Wartime resources must be: Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, and Uranium. -Lord of Growth."*

                    **3. Reconfigure Incomplete Trade Circles:**
                    - Review the **Players with Empty Trade Slots** report.
                    - For players not fully assigned, identify the recommended trade circle from the **Recommended Trade Circles** report.
                    - For each affected player, send a message:
                      - *"To The Ruler: [Ruler Name], please join a Trade Circle with partners: [list partner Nation Names]. Your assigned resource pair is [Assigned Resources]. Confirm your participation immediately. -Lord of Growth."*
                    - For any leftover players, send an individual notification to arrange an immediate meeting for reconfiguration.

                    **4. Document and Follow-Up:**
                    - Log each notification with the following details: Nation ID, Nation Name, Ruler Name, category (Peacetime/Wartime/Trade Circle), and specific action required.
                    - Set a follow-up review after 48-72 hours to ensure all players have confirmed the required changes.
                    - Re-run the analysis after adjustments and update the report accordingly.
                    """)
                    st.markdown(action_plan)

                # -----------------------
                # WRITE EXCEL FILE FOR DOWNLOAD WITH ADDITIONAL WORKSHEETS
                # -----------------------
                if sheets:
                    output = io.BytesIO()
                    # Using openpyxl engine instead of xlsxwriter
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Write existing sheets
                        for sheet_name, df_sheet in sheets.items():
                            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Get access to the workbook
                        workbook = writer.book
                        
                        # Add Summary Overview worksheet
                        summary_ws = workbook.create_sheet("Summary Overview")
                        summary_text = (
                            f"Total Players (Empty + Complete): {total_players}\n"
                            f"Players with Empty Trade Slots: {len(players_empty)} ({empty_percentage:.2f}%)\n"
                            f"Players in Complete Trade Circle: {total_full}\n"
                            f"Peacetime Mismatch among Complete Trade Circles: {unique_peacetime_mismatch} ({peacetime_mismatch_percentage:.2f}%)\n"
                            f"Wartime Mismatch among Complete Trade Circles: {unique_wartime_mismatch} ({wartime_mismatch_percentage:.2f}%)"
                        )
                        summary_ws["A1"] = summary_text
                        from openpyxl.styles import Alignment
                        summary_ws.column_dimensions["A"].width = 100
                        summary_ws["A1"].alignment = Alignment(wrapText=True)
                        
                        # Add Message Templates worksheet
                        messages_ws = workbook.create_sheet("Message Templates")
                        messages = []
                        # Peacetime mismatch messages:
                        if not peacetime_df.empty:
                            for idx, row in peacetime_df.iterrows():
                                extra = row["Extra Resources"]
                                dup = row["Duplicate Resources"]
                                # Determine which resource(s) to change:
                                if extra == "None" and dup != "None":
                                    resources_to_change = dup
                                elif extra != "None" and dup != "None":
                                    resources_to_change = f"{extra} and {dup}"
                                else:
                                    resources_to_change = extra
                                msg = (
                                    f'To The Ruler: {row["Ruler Name"]}, your Trade Circle currently has mismatched/duplicate resource(s) '
                                    f'{resources_to_change} which must be exchanged for any of the missing resource(s) '
                                    f'{row["Missing Peacetime Resources"]} to meet peacetime trade requirements. '
                                    f'Please can you either change your resources or get in contact with '
                                    f'your trade partners to coordinate adjustments to your agreements. '
                                    f'Peacetime resources must be: Aluminum, Cattle, Fish, Iron, Lumber, Marble, Pigs, '
                                    f'Spices, Sugar, Uranium, Water, Wheat. -Lord of Growth.'
                                )
                                messages.append({"Message Type": "Peacetime Resource Mismatch", "Message": msg})
                        # Wartime mismatch messages:
                        if not wartime_df.empty:
                            for idx, row in wartime_df.iterrows():
                                extra = row["Extra Resources"]
                                dup = row["Duplicate Resources"]
                                # Determine which resource(s) to change:
                                if extra == "None" and dup != "None":
                                    resources_to_change = dup
                                elif extra != "None" and dup != "None":
                                    resources_to_change = f"{extra} and {dup}"
                                else:
                                    resources_to_change = extra
                                msg = (
                                    f'To The Ruler: {row["Ruler Name"]}, your Trade Circle currently has mismatched/duplicate resource(s) '
                                    f'{resources_to_change} which must be exchanged for any of the missing resource(s) '
                                    f'{row["Missing Wartime Resources"]} to meet wartime trade requirements. '
                                    f'Please can you either change your resources or get in contact with '
                                    f'your trade partners to coordinate adjustments to your agreements. '
                                    f'Wartime resources must be: Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, '
                                    f'Marble, Oil, Pigs, Rubber, and Uranium. -Lord of Growth.'
                                )
                                messages.append({"Message Type": "Wartime Resource Mismatch", "Message": msg})

                        # Trade circle messages: include partner information
                        def generate_trade_circle_messages(circles, circle_type):
                            for circle in circles:
                                nation_names = [player.get('Ruler Name','') for player in circle]
                                for player in circle:
                                    partners = [name for name in nation_names if name != player.get('Ruler Name','')]
                                    msg = f'To The Ruler: {player.get("Ruler Name","")}, please can you join a Trade Circle with partners: {", ".join(partners)}. Your assigned resource pair is {", ".join(player.get("Assigned Resources", [])) if player.get("Assigned Resources") else "None"}. Confirm your participation immediately. -Lord of Growth.'
                                    messages.append({"Message Type": f"{circle_type} Trade Circle", "Message": msg})

                        if trade_circles_peace:
                            generate_trade_circle_messages(trade_circles_peace, "Peacetime")
                        if trade_circles_war:
                            generate_trade_circle_messages(trade_circles_war, "Wartime")
                        
                        messages_df = pd.DataFrame(messages)
                        # Write the messages DataFrame starting at A1 in Message Templates worksheet
                        from openpyxl.utils.dataframe import dataframe_to_rows
                        for r in dataframe_to_rows(messages_df, index=False, header=True):
                            messages_ws.append(r)
                        # Adjust column widths for Message Templates
                        messages_ws.column_dimensions["A"].width = 30
                        messages_ws.column_dimensions["B"].width = 150
                    output.seek(0)
                    excel_data = output.read()
                else:
                    excel_data = None

            # -----------------------
            # DOWNLOAD ALL DATA EXCEL (positioned at the bottom of the page)
            # -----------------------
            st.markdown("### Download All Processed Data")
            if excel_data:
                st.download_button("Download Summary Report", excel_data, file_name="full_summary_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_report")
            else:
                st.info("No data available for download.")
    else:
        st.info("Please enter the correct password to access the functionality.")

if __name__ == "__main__":
    main()
