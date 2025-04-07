import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import re
import copy
from datetime import datetime

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
# TRADE CIRCLE PROCESSING FUNCTIONS
# -----------------------
def get_current_resources(row, resource_cols):
    """Return a comma-separated string of non-blank resources."""
    resources = [str(x).strip() for x in row[resource_cols] if pd.notnull(x) and str(x).strip() != '']
    return ", ".join(resources)

def count_empty_slots(row, resource_cols):
    """Count blank resource cells and determine trade slots (each slot covers 2 resources)."""
    count = sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '')
    return count // 2

def form_trade_circles(players, sorted_resources, circle_size=TRADE_CIRCLE_SIZE):
    """Group players into full trade circles and assign resource pairs."""
    trade_circles = []
    full_groups = [players[i:i+circle_size] for i in range(0, len(players), circle_size) if len(players[i:i+circle_size]) == circle_size]
    for group in full_groups:
        for j, player in enumerate(group):
            # Each player gets two resources from the sorted list.
            assigned_resources = sorted_resources[2*j:2*j+2]
            player['Assigned Resources'] = assigned_resources
        trade_circles.append(group)
    return trade_circles

def display_trade_circle_df(circle, condition):
    """Display a trade circle in a Streamlit dataframe."""
    circle_data = []
    for player in circle:
        # Compute the "Current Resource 1+Resource2" column.
        current_resources_str = player.get('Current Resources', '')
        current_resources_list = [r.strip() for r in current_resources_str.split(',') if r.strip()]
        if len(current_resources_list) >= 2:
            current_resource_sum = f"{current_resources_list[0]}+{current_resources_list[1]}"
        else:
            current_resource_sum = current_resources_str

        circle_data.append({
            'Nation ID': player.get('Nation ID', ''),
            'Ruler Name': player.get('Ruler Name', ''),
            'Nation Name': player.get('Nation Name', ''),
            'Team': player.get('Team', ''),
            'Current Resources': player.get('Current Resources', ''),
            'Current Resource 1+Resource2': current_resource_sum,
            'Activity': player.get('Activity', ''),
            'Days Old': player.get('Days Old', ''),
            f'Assigned {condition} Resources': ", ".join(player.get('Assigned Resources', []))
        })
    circle_df = pd.DataFrame(circle_data)
    st.dataframe(circle_df, use_container_width=True)

# -----------------------
# MAIN APP
# -----------------------
def main():
    st.title("CyberNations | Nation Statistics & Trade Circle Formation Tool")
    
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
        password = st.text_input("Enter Password", type="password")
        if password:
            if password == "secret":
                st.session_state.password_verified = True
                st.success("Password accepted!")
            else:
                st.error("Incorrect password. Please try again.")
    
    # Only display the download functionality if the password is verified.
    if st.session_state.password_verified:
        if st.button("Download and Display Nation Statistics"):
            with st.spinner("Constructing download link and retrieving data..."):
                # Construct the URL based on the current date.
                today = datetime.now()
                # Format: MonthDayYear (e.g., April 7, 2025 -> "472025")
                date_str = f"{today.month}{today.day}{today.year}"
                base_url = "https://www.cybernations.net/assets/CyberNations_SE_Nation_Stats_"
                # Construct two potential links
                url1 = base_url + date_str + "510001.zip"
                url2 = base_url + date_str + "510002.zip"
                st.write(f"Attempting to download from: {url1}")
                df = download_and_extract_zip(url1)
                if df is None:
                    st.write(f"Trying alternative link: {url2}")
                    df = download_and_extract_zip(url2)
                if df is not None:
                    st.success("Data loaded successfully!")
                    st.session_state.df = df
                else:
                    st.error("Failed to load data from both constructed URLs.")

        # Proceed if data is loaded
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df.copy()
            st.subheader("Original Data")
            st.dataframe(df, use_container_width=True)

            # -----------------------
            # FILTERING UI
            # -----------------------
            st.subheader("Filter Data")
            if not df.empty:
                column_options = list(df.columns)
                default_index = column_options.index("Alliance") if "Alliance" in column_options else 0
                selected_column = st.selectbox("Select column to filter", column_options, index=default_index)
                search_text = st.text_input("Filter by text (separate words by comma)", value="Freehold of the Wolves")
                        
                if search_text:
                    # Support multiple filters separated by comma
                    filters = [f.strip() for f in search_text.split(",") if f.strip()]
                    pattern = "|".join(filters)
                    filtered_df = df[df[selected_column].astype(str).str.contains(pattern, case=False, na=False)]
                    st.write(f"Showing results where **{selected_column}** contains any of {filters}:")
                    st.dataframe(filtered_df, use_container_width=True)
                    csv = filtered_df.to_csv(index=False)
                    st.download_button("Download Filtered CSV", csv, file_name="filtered_nation_stats.csv", mime="text/csv")
                else:
                    st.info("Enter text to filter the data.")
            else:
                st.warning("DataFrame is empty, nothing to filter.")

            # -----------------------
            # TRADE CIRCLE FORMATION SECTION
            # -----------------------
            st.subheader("Trade Circle Formation")
            st.markdown(
                """
                **Note:**
                Only players with at least one empty resource slot are considered. Additionally, players who have been inactive 
                for three weeks or more and those with an Alliance Status of "Pending" are excluded from trade circle inclusion.
                """
            )
            if st.button("Form Trade Circles"):
                # For this example, we use the filtered data if available; otherwise, use the full DataFrame.
                df_to_use = filtered_df if "filtered_df" in locals() and not filtered_df.empty else df

                # Assume that the resource columns are named "Connected Resource 1" to "Connected Resource 10"
                resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
                # Identify players with at least one blank in any resource column
                mask_empty = df_to_use[resource_cols].isnull().any(axis=1) | (
                    df_to_use[resource_cols].apply(lambda col: col.astype(str).str.strip() == '').any(axis=1)
                )
                players_empty = df_to_use[mask_empty].copy()

                # Compute "Current Resources" column
                players_empty['Current Resources'] = players_empty.apply(lambda row: get_current_resources(row, resource_cols), axis=1)
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

                # Display summary of players with empty trade slots and active recently
                display_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Team', 'Current Resources', 'Empty Slots Count', 'Activity', 'Days Old']
                st.markdown("**Players with empty trade slots (active recently):**")
                st.dataframe(players_empty[display_cols].reset_index(drop=True), use_container_width=True)
                
                # ---- New Table: Players with a complete trade circle (no empty slots) ----
                players_full = df_to_use[~mask_empty].copy()
                # Compute "Current Resources" for players with complete resource sets
                players_full['Current Resources'] = players_full.apply(lambda row: get_current_resources(row, resource_cols), axis=1)
                # Also compute "Empty Slots Count" to verify these players have complete resource sets (should be 0)
                players_full['Empty Slots Count'] = players_full.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
                # Process "Created" and "Days Old"
                players_full['Created'] = pd.to_datetime(players_full['Created'], format=date_format, errors='coerce')
                players_full['Days Old'] = (current_date - players_full['Created']).dt.days
                # Filter out inactive players and Alliance Pending just as above
                players_full = players_full[~players_full['Activity'].isin(["Active Three Weeks Ago", "Active More Than Three Weeks Ago"])]
                if "Alliance Status" in players_full.columns:
                    players_full = players_full[players_full["Alliance Status"] != "Pending"]

                st.markdown("**Players with a complete trade circle (no empty slots):**")
                st.dataframe(players_full[display_cols].reset_index(drop=True), use_container_width=True)

                # Sort players by Nation ID (or another criterion) from the empty slots list
                players_empty_sorted = players_empty.sort_values('Nation ID')
                players_list = players_empty_sorted.to_dict('records')
                
                # -----------------------
                # FIX: Use deep copies for separate trade circle formations
                # -----------------------
                players_list_peace = copy.deepcopy(players_list)
                players_list_war = copy.deepcopy(players_list)
                trade_circles_peace = form_trade_circles(players_list_peace, sorted_peacetime)
                trade_circles_war = form_trade_circles(players_list_war, sorted_wartime)

                # Determine leftover players (those not in a full group)
                num_full_groups = len(players_list) // TRADE_CIRCLE_SIZE
                leftover_players = players_list[num_full_groups * TRADE_CIRCLE_SIZE:]

                # Display Peacetime Trade Circles
                if trade_circles_peace:
                    st.markdown(f"**Recommended Peacetime Trade Circles:**")
                    for idx, circle in enumerate(trade_circles_peace, start=1):
                        st.markdown(f"--- **Peacetime Trade Circle #{idx}** ---")
                        display_trade_circle_df(circle, "Peacetime")
                else:
                    st.info("No full Peacetime trade circles could be formed.")

                # Display Wartime Trade Circles
                if trade_circles_war:
                    st.markdown(f"**Recommended Wartime Trade Circles:**")
                    for idx, circle in enumerate(trade_circles_war, start=1):
                        st.markdown(f"--- **Wartime Trade Circle #{idx}** ---")
                        display_trade_circle_df(circle, "Wartime")
                else:
                    st.info("No full Wartime trade circles could be formed.")

                # Display leftover players (if any)
                if leftover_players:
                    leftover_data = []
                    for player in leftover_players:
                        leftover_data.append({
                            'Nation ID': player.get('Nation ID', ''),
                            'Ruler Name': player.get('Ruler Name', ''),
                            'Nation Name': player.get('Nation Name', ''),
                            'Team': player.get('Team', ''),
                            'Current Resources': player.get('Current Resources', ''),
                            'Activity': player.get('Activity', ''),
                            'Days Old': player.get('Days Old', '')
                        })
                    st.markdown("**Players remaining without a full trade circle:**")
                    st.dataframe(pd.DataFrame(leftover_data), use_container_width=True)
                else:
                    st.success("All players have been grouped into trade circles.")

                # -----------------------
                # DOWNLOAD TRADE CIRCLES CSV
                # -----------------------
                trade_circle_entries = []
                # Combine both Peacetime and Wartime trade circles for CSV download
                for circle_type, circles in [("Peacetime", trade_circles_peace), ("Wartime", trade_circles_war)]:
                    if circles:
                        for idx, circle in enumerate(circles, start=1):
                            for player in circle:
                                trade_circle_entries.append({
                                    "Circle Type": circle_type,
                                    "Circle Number": idx,
                                    "Nation ID": player.get('Nation ID', ''),
                                    "Ruler Name": player.get('Ruler Name', ''),
                                    "Nation Name": player.get('Nation Name', ''),
                                    "Team": player.get('Team', ''),
                                    "Current Resources": player.get('Current Resources', ''),
                                    "Activity": player.get('Activity', ''),
                                    "Days Old": player.get('Days Old', ''),
                                    "Assigned Resources": ", ".join(player.get('Assigned Resources', [])),
                                    "Nation Drill URL": "https://www.cybernations.net/nation_drill_display.asp?Nation_ID=" + str(player.get('Nation ID', ''))
                                })
                if trade_circle_entries:
                    trade_circles_df = pd.DataFrame(trade_circle_entries)
                    csv_trade_circles = trade_circles_df.to_csv(index=False)
                    st.download_button("Download Trade Circles CSV", csv_trade_circles, file_name="trade_circles.csv", mime="text/csv")
                else:
                    st.info("No trade circles data available for download.")
    else:
        st.info("Please enter the correct password to access the functionality.")

if __name__ == "__main__":
    main()
