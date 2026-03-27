# ui/sidebar.py
import streamlit as st

def setup_sidebar(rules_all, selected_rules):
    # -----------------------------
    # Rule selection
    # -----------------------------
    rule_names = list(rules_all.keys())
    selected_rules = st.sidebar.multiselect(
        "Select Rules", rule_names, default=selected_rules[:3]
    )

    # -----------------------------
    # Input source
    # -----------------------------
    source_option = st.sidebar.radio("Input Source", ["Webcam", "Upload MP4 Video"])

    # -----------------------------
    # Floor controls
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Floor Line**")

    def nudge_floor(delta):
        st.session_state["floor_y"] = max(0, min(1080, st.session_state["floor_y"] + delta))

    def sync_from_slider():
        st.session_state["floor_y"] = st.session_state["_floor_slider"]

    def sync_from_number():
        st.session_state["floor_y"] = st.session_state["_floor_number"]

    col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
    col_down.button("▼", on_click=nudge_floor, args=(-5,))
    col_up.button("▲", on_click=nudge_floor, args=(+5,))
    col_val.number_input(
        "Y", min_value=0, max_value=1080,
        value=st.session_state["floor_y"],
        step=1, key="_floor_number",
        on_change=sync_from_number, label_visibility="collapsed"
    )
    st.sidebar.slider(
        "Fine-tune floor", 0, 1080,
        value=st.session_state["floor_y"],
        key="_floor_slider", on_change=sync_from_slider,
        label_visibility="collapsed"
    )

    # -----------------------------
    # Webcam start checkbox
    # -----------------------------
    run_webcam = False
    if source_option == "Webcam":
        run_webcam = st.sidebar.checkbox("Start Webcam", value=False)

    return selected_rules, source_option, run_webcam

# # ui/sidebar.py
# import streamlit as st
# from posture_analysis import load_rule_groups


# def setup_sidebar(rules_all, default_rules):
#     """
#     Returns (selected_rule_names, source_option, run_webcam, exercise_name)

#     selected_rule_names : list[str]  – rule keys active for the chosen exercise
#     source_option       : str
#     run_webcam          : bool
#     exercise_name       : str        – display name for the score card
#     """

#     # ── Exercise selector ─────────────────────────────────────────────────────
#     st.sidebar.markdown("### 🏃 Exercise")

#     # Parse groups once and cache in session_state
#     if "_rule_groups" not in st.session_state:
#         st.session_state["_rule_groups"] = load_rule_groups("assets/rules.txt")

#     rule_groups    = st.session_state["_rule_groups"]
#     exercise_names = list(rule_groups.keys())

#     if not exercise_names:
#         st.sidebar.warning("No rule groups found in assets/rules.txt")
#         selected_rules = default_rules
#         exercise_name  = "Exercise"
#     else:
#         exercise_name = st.sidebar.radio(
#             "Select exercise",
#             exercise_names,
#             key="selected_exercise",
#             label_visibility="collapsed",
#         )
#         selected_rules = rule_groups.get(exercise_name, [])

#         # Compact expander showing active rules + feature ranges
#         with st.sidebar.expander("📋 Active rules", expanded=False):
#             for r in selected_rules:
#                 rule_def = rules_all.get(r, {})
#                 parts    = ", ".join(
#                     f"`{feat}` [{lo}–{hi}]"
#                     for feat, (lo, hi) in rule_def.items()
#                 )
#                 st.markdown(f"- **{r}** — {parts if parts else '—'}")

#     # ── Input source ──────────────────────────────────────────────────────────
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 📷 Input Source")
#     source_option = st.sidebar.radio(
#         "Source",
#         ["Webcam", "Upload MP4 Video"],
#         key="source_option",
#         label_visibility="collapsed",
#     )

#     # ── Floor line ────────────────────────────────────────────────────────────
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 📐 Floor Line")

#     def nudge_floor(delta):
#         st.session_state["floor_y"] = max(0, min(1080, st.session_state["floor_y"] + delta))

#     def sync_from_slider():
#         st.session_state["floor_y"] = st.session_state["_floor_slider"]

#     def sync_from_number():
#         st.session_state["floor_y"] = st.session_state["_floor_number"]

#     col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
#     col_down.button("▼", on_click=nudge_floor, args=(-5,))
#     col_up.button("▲",   on_click=nudge_floor, args=(+5,))
#     col_val.number_input(
#         "Y", min_value=0, max_value=1080,
#         value=st.session_state["floor_y"],
#         step=1, key="_floor_number",
#         on_change=sync_from_number, label_visibility="collapsed",
#     )
#     st.sidebar.slider(
#         "Fine-tune floor", 0, 1080,
#         value=st.session_state["floor_y"],
#         key="_floor_slider", on_change=sync_from_slider,
#         label_visibility="collapsed",
#     )

#     # ── Webcam checkbox ───────────────────────────────────────────────────────
#     run_webcam = False
#     if source_option == "Webcam":
#         run_webcam = st.sidebar.checkbox("▶ Start Webcam", value=False)

#     return selected_rules, source_option, run_webcam, exercise_name