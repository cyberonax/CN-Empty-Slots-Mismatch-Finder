            # -----------------------
            # TRADE CIRCLE & RESOURCE PROCESSING (automatically triggered)
            # -----------------------
            # Instead of using st.session_state.filtered_df, reload the filtered CSV if available.
            if "filtered_csv" in st.session_state:
                filtered_csv = st.session_state.filtered_csv
                # Read the CSV content into a DataFrame
                df_to_use = pd.read_csv(io.StringIO(filtered_csv))
            else:
                df_to_use = df

            # Assume that the resource columns are named "Connected Resource 1" to "Connected Resource 10"
            resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]

            # FIRST: Compute uniform "Days Old" data for the entire DataFrame.
            date_format = "%m/%d/%Y %I:%M:%S %p"  # Adjust if needed
            df_to_use['Created'] = pd.to_datetime(df_to_use['Created'], format=date_format, errors='coerce')
            current_date = pd.to_datetime("now")
            df_to_use['Days Old'] = (current_date - df_to_use['Created']).dt.days

            # Identify players with at least one blank in any resource column
            mask_empty = df_to_use[resource_cols].isnull().any(axis=1) | (
                df_to_use[resource_cols].apply(lambda col: col.astype(str).str.strip() == '').any(axis=1)
            )

            # -----------------------
            # PLAYERS WITH EMPTY TRADE SLOTS (active recently)
            # -----------------------
            players_empty = df_to_use[mask_empty].copy()
            # Compute "Current Resources" column for full resource list
            players_empty['Current Resources'] = players_empty.apply(lambda row: ", ".join(sorted([str(x).strip() for x in row[resource_cols] 
                                                                          if pd.notnull(x) and str(x).strip() != ''])), axis=1)
            # Compute "Current Resource 1+2" using Resource 1 and Resource 2 (with fallback logic)
            players_empty['Current Resource 1+2'] = players_empty.apply(lambda row: get_resource_1_2(row), axis=1)
            # Compute empty trade slots (each slot covers 2 resources)
            players_empty['Empty Slots Count'] = players_empty.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
            # (No need to re-compute Created and Days Old here since it was applied to df_to_use uniformly.)

            # Filter out players who are inactive based on the "Activity" column.
            players_empty = players_empty[~players_empty['Activity'].isin(["Active Three Weeks Ago", "Active More Than Three Weeks Ago"])]

            # ---- New Filter: Exclude players with Alliance Status "Pending" ----
            if "Alliance Status" in players_empty.columns:
                players_empty = players_empty[players_empty["Alliance Status"] != "Pending"]

            with st.expander("Players with empty trade slots (active recently)"):
                display_cols = ['Nation ID', 'Ruler Name', 'Nation Name', 'Team', 'Current Resources',
                                'Current Resource 1+2', 'Empty Slots Count', 'Activity', 'Days Old']
                st.dataframe(players_empty[display_cols].reset_index(drop=True), use_container_width=True)


            # -----------------------
            # PLAYERS WITH A COMPLETE TRADE CIRCLE (no empty slots)
            # -----------------------
            players_full = df_to_use[~mask_empty].copy()
            # Compute "Current Resources" for players with complete resource sets.
            players_full['Current Resources'] = players_full.apply(lambda row: ", ".join(sorted([str(x).strip() for x in row[resource_cols]
                                                                          if pd.notnull(x) and str(x).strip() != ''])), axis=1)
            # Use CSV-based "Resource 1" and "Resource 2" for Current Resource 1+2 (if available)
            players_full['Current Resource 1+2'] = players_full.apply(lambda row: get_resource_1_2(row), axis=1)
            # Also compute "Empty Slots Count" to verify these players have complete resource sets (should be 0)
            players_full['Empty Slots Count'] = players_full.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
            # The uniform "Days Old" has already been computed above.

            with st.expander("Players with a complete trade circle (no empty slots)"):
                st.dataframe(players_full[display_cols].reset_index(drop=True), use_container_width=True)


            # -----------------------
            # RESOURCE MISMATCHES
            # -----------------------
            with st.expander("Resource Mismatches"):
                st.markdown(
                    """
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
                # (The text area sections for peace_a_text, peace_b_text, peace_c_text, war_text remain unchanged.)
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
                
                # Process each nation from players_full.
                for idx, row in players_full.iterrows():
                    # Get current resources from the full set (calculated in "Current Resources")
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
                st.markdown("**Peace Mode Level A Mismatches:**")
                df_peace_a = pd.DataFrame(mismatch_peace_a).reset_index(drop=True)
                # Filter to include only nations under 1000 days old.
                df_peace_a = df_peace_a[df_peace_a['Days Old'] < 1000]
                if not df_peace_a.empty:
                    styled_peace_a = df_peace_a.style.applymap(highlight_none, subset=['Duplicate Resources'])
                    st.dataframe(styled_peace_a, use_container_width=True)
                else:
                    st.info("No mismatches found for Peace Mode Level A (nations under 1000 days old).")
                
                st.markdown("**Peace Mode Level B Mismatches:**")
                df_peace_b = pd.DataFrame(mismatch_peace_b).reset_index(drop=True)
                # Filter to include only nations 1000-2000 days old.
                df_peace_b = df_peace_b[(df_peace_b['Days Old'] >= 1000) & (df_peace_b['Days Old'] < 2000)]
                if not df_peace_b.empty:
                    styled_peace_b = df_peace_b.style.applymap(highlight_none, subset=['Duplicate Resources'])
                    st.dataframe(styled_peace_b, use_container_width=True)
                else:
                    st.info("No mismatches found for Peace Mode Level B (nations 1000-2000 days old).")
                
                st.markdown("**Peace Mode Level C Mismatches:**")
                df_peace_c = pd.DataFrame(mismatch_peace_c).reset_index(drop=True)
                # Filter to include only nations over 2000 days old.
                df_peace_c = df_peace_c[df_peace_c['Days Old'] >= 2000]
                if not df_peace_c.empty:
                    styled_peace_c = df_peace_c.style.applymap(highlight_none, subset=['Duplicate Resources'])
                    st.dataframe(styled_peace_c, use_container_width=True)
                else:
                    st.info("No mismatches found for Peace Mode Level C (nations over 2000 days old).")
                
                st.markdown("**War Mode Mismatches:**")
                df_war = pd.DataFrame(mismatch_war).reset_index(drop=True)
                # (No age filtering for War Mode; adjust here if needed.)
                if not df_war.empty:
                    styled_war = df_war.style.applymap(highlight_none, subset=['Duplicate Resources'])
                    st.dataframe(styled_war, use_container_width=True)
                else:
                    st.info("No mismatches found for War Mode.")
                
                # (Additional consolidation of mismatch DataFrames for later use can follow here as in your original code.)
