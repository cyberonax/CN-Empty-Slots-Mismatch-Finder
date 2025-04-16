                from itertools import combinations

                # --- Utility: parse pasted blocks into raw level‑tagged lists ---
                raw = st.text_area(
                    "Paste your Trade Circle blocks. Each line:\n"
                    "`Ruler Name<TAB>Resource 1+2<TAB>Alliance<TAB>Team<TAB>Days Old"
                    "<TAB>Nation Drill Link<TAB>Activity`\n"
                    "Skip blank or ‘x’. Separate circles with an empty line.",
                    height=200
                )
                exclude = {x.strip().lower() for x in st.text_area(
                    "Exclude these Ruler or Nation names (one per line):", height=80
                ).splitlines() if x.strip()}

                # majority‑team filter
                majority = {}
                if "selected_alliances" in st.session_state and "filtered_df" in st.session_state:
                    for A in st.session_state.selected_alliances:
                        dfA = st.session_state.filtered_df.query("Alliance==@A")
                        if not dfA.empty:
                            majority[A] = dfA["Team"].mode()[0]

                def eligible(p):
                    if p["Alliance"] not in st.session_state.selected_alliances:
                        return False
                    try:
                        if float(p["Activity"]) >= 14:
                            return False
                    except:
                        return False
                    tm = majority.get(p["Alliance"])
                    if tm and p["Team"] != tm:
                        return False
                    if p["Ruler Name"].lower() in exclude or p.get("Nation Name","").lower() in exclude:
                        return False
                    return True

                # parse into blocks
                blocks, cur = [], []
                for L in raw.splitlines():
                    if not L.strip():
                        if cur:
                            blocks.append(cur); cur=[]
                    else:
                        cur.append(L)
                if cur:
                    blocks.append(cur)

                # build initial per‑level blocks
                init_by_level = {"A":[],"B":[],"C":[]}
                for blk in blocks:
                    seen = set()
                    circ = []
                    for L in blk:
                        f = [x.strip() for x in L.split("\t")]
                        if len(f)<7:
                            continue
                        name, res12, alli, team, days_s, link, act = f[:7]
                        if not name or name.lower()=="x" or name in seen:
                            continue
                        seen.add(name)
                        try:
                            days = float(days_s)
                        except:
                            days = None
                        p = {
                            "Ruler Name": name,
                            "Resource 1+2": res12,
                            "Alliance": alli,
                            "Team": team,
                            "Days Old": days,
                            "Nation Drill Link": link,
                            "Activity": act,
                            "Peace Level": get_peace_level(days)
                        }
                        if eligible(p):
                            lvl = p["Peace Level"]
                            circ.append(p)
                    if circ:
                        # keep only one peace level per block
                        lvl = Counter(p["Peace Level"] for p in circ).most_common(1)[0][0]
                        init_by_level[lvl].append(circ)

                # build full pools from filtered_df
                pools = {"A":[], "B":[], "C":[]}
                if "filtered_df" in st.session_state:
                    for _, row in st.session_state.filtered_df.iterrows():
                        p = {
                            "Ruler Name": row["Ruler Name"],
                            "Resource 1+2": row["Current Resource 1+2"],
                            "Alliance": row["Alliance"],
                            "Team": row["Team"],
                            "Days Old": row["Days Old"],
                            "Nation Drill Link": f"https://www.cybernations.net/nation_drill_display.asp?Nation_ID={row['Nation ID']}",
                            "Activity": row["Activity"],
                            "Peace Level": get_peace_level(row["Days Old"])
                        }
                        if eligible(p):
                            pools[p["Peace Level"]].append(p)

                assigned = set()
                final_circles = []

                def choose_and_assign(circ, valid_combos):
                    # pick best combo + pairing with Hungarian over 66 pairs
                    best_cost, best = float("inf"), None
                    for combo in valid_combos:
                        pairs = list(combinations(combo, 2))
                        n, m = len(circ), len(pairs)
                        cost = np.zeros((n, m))
                        for i, p in enumerate(circ):
                            curr = set(r.strip() for r in p["Resource 1+2"].split(",") if r.strip())
                            for j,pair in enumerate(pairs):
                                want = set(pair)
                                cost[i,j] = len((want-curr)) + len((curr-want))
                        r,c = linear_sum_assignment(cost)
                        tot = cost[r,c].sum()
                        assigned_pairs = [pairs[j] for j in c]
                        flat = [r for pair in assigned_pairs for r in pair]
                        if tot<best_cost and len(flat)==len(set(flat))==len(combo):
                            best_cost, best = (tot, (combo, r, c, pairs))
                    if best is None:
                        combo, r, c, pairs = valid_combos[0], range(len(circ)), range(len(circ))
                    else:
                        combo, r, c, pairs = best
                    # commit
                    for i,j in zip(r,c):
                        circ[i]["Assigned Resource 1+2"] = list(pairs[j])
                    # store full 12‑list
                    for p in circ:
                        p["Assigned Resource Combination"] = ", ".join(combo)
                    return circ

                # now per‑level: fill initial, then make new from leftovers
                for lvl in ["A","B","C"]:
                    valid = {"A":peace_a_combos,"B":peace_b_combos,"C":peace_c_combos}[lvl]
                    # fill initial blocks
                    for blk in init_by_level[lvl]:
                        members = [p for p in blk if p["Ruler Name"] not in assigned]
                        slots = TRADE_CIRCLE_SIZE - len(members)
                        for cand in pools[lvl]:
                            if slots<=0:
                                break
                            if cand["Ruler Name"] not in assigned:
                                members.append(cand)
                                assigned.add(cand["Ruler Name"])
                                slots -= 1
                        if len(members)==TRADE_CIRCLE_SIZE:
                            final_circles.append((lvl, choose_and_assign(members, valid)))

                    # leftover grouping
                    left = [p for p in pools[lvl] if p["Ruler Name"] not in assigned]
                    for i in range(len(left)//TRADE_CIRCLE_SIZE):
                        grp = left[i*6:(i+1)*6]
                        for p in grp:
                            assigned.add(p["Ruler Name"])
                        final_circles.append((lvl, choose_and_assign(grp, valid)))

                # DISPLAY
                st.markdown("### New Recommended Trade Circles (Peace Mode)")
                for lvl in ["A","B","C"]:
                    label = {
                      "A":"Level A (<1000 days)",
                      "B":"Level B (1000–2000 days)",
                      "C":"Level C (>=2000 days)"
                    }[lvl]
                    st.markdown(f"#### Peace {label}")
                    for idx,(L,circ) in enumerate(final_circles,1):
                        if L!=lvl: continue
                        st.markdown(f"**Trade Circle {idx}:**")
                        rows = []
                        for p in circ:
                            rows.append({
                                "Ruler Name": p["Ruler Name"],
                                "Resource 1+2": p["Resource 1+2"],
                                "Alliance": p["Alliance"],
                                "Team": p["Team"],
                                "Days Old": p["Days Old"],
                                "Nation Drill Link": p["Nation Drill Link"],
                                "Activity": p["Activity"],
                                "Assigned Resource 1+2": ", ".join(p["Assigned Resource 1+2"]),
                                "Assigned Resource Combination": p["Assigned Resource Combination"]
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # LEFTOVERS
                leftovers = [p for lvl in ["A","B","C"] for p in pools[lvl] if p["Ruler Name"] not in assigned]
                st.markdown("### Remaining Unmatched Players")
                if leftovers:
                    st.dataframe(pd.DataFrame(leftovers)[[
                        "Ruler Name","Resource 1+2","Alliance","Team",
                        "Days Old","Nation Drill Link","Activity"
                    ]], use_container_width=True)
                else:
                    st.info("All eligible players placed in Trade Circles.")

                # SUMMARY
                st.markdown("### Match Summary by Alliance")
                dfF = st.session_state.filtered_df.copy()
                if "Alliance Status" in dfF.columns:
                    dfF = dfF[dfF["Alliance Status"]!="Pending"]
                summary = []
                for A in st.session_state.selected_alliances:
                    tot = len(dfF.query("Alliance==@A"))
                    mat = sum(1 for lvl,c in final_circles for p in c
                              if p["Alliance"]==A)
                    pct = f"{(mat/tot*100) if tot else 0:.2f}%"
                    summary.append({
                        "Alliance": A,
                        "Total Members": tot,
                        "Matched Players": mat,
                        "Percent Matched": pct
                    })
                st.dataframe(pd.DataFrame(summary), use_container_width=True)
