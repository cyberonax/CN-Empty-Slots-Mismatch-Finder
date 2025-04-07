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
# The original base number is "452025510002" where "452025" represents Month/Day/Year (4/5/2025)
# We'll replace that prefix with a dynamic one based on the current date.
ORIGINAL_BASE_NUMBER = "452025510002"
# Extract the suffix (last 6 digits) from the original number.
SUFFIX = ORIGINAL_BASE_NUMBER[6:]
BASE_URL = "https://www.cybernations.net/assets/"

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
#wartime_resources = [
#    'Aluminum', 'Cattle', 'Coal', 'Fish', 'Iron',
#    'Lumber', 'Marble', 'Oil', 'Pigs', 'Rubber',
#    'Uranium', 'Wheat'
#]

sorted_peacetime = sorted(peacetime_resources)
sorted_wartime = sorted(wartime_resources)

TRADE_CIRCLE_SIZE = 6  # 6 players per circle, each gets 2 resources

# -----------------------
# AUTO-DETECT WORKING ZIP URL FUNCTION
# -----------------------
def get_working_zip_url(debug=False):
    """
    Build the zip file URL dynamically by adjusting the date prefix according
    to the current month, day, and year. Then, probe the links with numerical differences
    of +/- 2 from the computed base number to find a working Nation Statistics zip file.
    
    The file number is composed as:
      <date_prefix><suffix>
    where:
	  date_prefix = f"{current_month}{current_day}{current_year}"
      suffix = constant extracted from the original base number (e.g., "510002")
    
    Returns:
        str: A working zip file URL if found, or None.
    """
    # Get current date values
    today = datetime.now()
    # Build the dynamic prefix: e.g., for April 6, 2025, it becomes "462025"
    current_prefix = f"{today.month}{today.day}{today.year}"
    # Create the new base number by combining the dynamic prefix with the constant suffix.
    new_base_str = f"{current_prefix}{SUFFIX}"
    try:
        new_base = int(new_base_str)
    except ValueError:
        st.error("Error constructing the dynamic base number.")
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.93 Safari/537.36"
        )
    }
    if debug:
        st.write(f"Dynamic base number: {new_base_str}")

    # Probe candidate numbers +/-20 from new_base
    for offset in range(-20, 21):  # Check new_base-20, to new_base+20
        candidate_number = new_base + offset
        candidate_filename = f"CyberNations_SE_Nation_Stats_{candidate_number}.zip"
        candidate_url = BASE_URL + candidate_filename
        if debug:
            st.write(f"Trying URL: {candidate_url}")
        try:
            response = requests.get(candidate_url, headers=headers)
            if response.status_code == 200:
                if debug:
                    st.write(f"Working URL found: {candidate_url}")
                return candidate_url
            else:
                if debug:
                    st.write(f"Status {response.status_code} for URL: {candidate_url}")
        except Exception as e:
            if debug:
                st.write(f"Error fetching {candidate_url}: {e}")
    st.error("No working zip file URL found within the given range.")
    return None

# -----------------------
# DOWNLOAD & DATA LOADING FUNCTIONS
# -----------------------
def download_and_extract_zip(url):
    """Download a zip file from the given URL and extract its first file as a DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Error downloading file: {e}")
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
        nation_id = player.get('Nation ID', '')
        circle_data.append({
            'Nation ID': nation_id,
            'Nation URL': f"https://www.cybernations.net/nation_drill_display.asp?Nation_ID={nation_id}",
            'Ruler Name': player.get('Ruler Name', ''),
            'Nation Name': player.get('Nation Name', ''),
            'Current Resources': player.get('Current Resources', ''),
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
    
    # Only display the download button if the password is verified.
    if st.session_state.password_verified:
        if "df" not in st.session_state:
            st.session_state.df = None

        if st.button("Download and Display Nation Statistics"):
            with st.spinner("Checking for a working zip file link..."):
                # Set debug=True to see detailed output; set to False once confirmed.
                working_zip_url = get_working_zip_url(debug=False)
                if working_zip_url is None:
                    st.error("Could not detect a working zip file URL.")
                else:
                    #st.write(f"Using zip file: {working_zip_url}")
                    st.session_state.df = download_and_extract_zip(working_zip_url)
            if st.session_state.df is not None:
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data.")

        # Proceed if data is loaded
        if st.session_state.df is not None:
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
                display_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Current Resources', 'Empty Slots Count', 'Activity', 'Days Old']
                st.markdown("**Players with empty trade slots (active recently):**")
                st.dataframe(players_empty[display_cols].reset_index(drop=True), use_container_width=True)

                # Sort players by Nation ID (or another criterion)
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
                            'Ruler Name': player.get('Ruler Name', ''),
                            'Nation Name': player.get('Nation Name', ''),
                            'Current Resources': player.get('Current Resources', ''),
                            'Activity': player.get('Activity', ''),
                            'Days Old': player.get('Days Old', ''),
                            'Nation URL': f"https://www.cybernations.net/nation_drill_display.asp?Nation_ID={player.get('Nation ID', '')}"
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
                                    "Ruler Name": player.get('Ruler Name', ''),
                                    "Nation Name": player.get('Nation Name', ''),
                                    "Current Resources": player.get('Current Resources', ''),
                                    "Activity": player.get('Activity', ''),
                                    "Days Old": player.get('Days Old', ''),
                                    "Assigned Resources": ", ".join(player.get('Assigned Resources', [])),
                                    "Nation URL": f"https://www.cybernations.net/nation_drill_display.asp?Nation_ID={player.get('Nation ID', '')}"
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
