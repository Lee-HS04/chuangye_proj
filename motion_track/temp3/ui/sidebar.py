# # ui/sidebar.py
# import streamlit as st

# def setup_sidebar(rules_all, selected_rules):
#     # -----------------------------
#     # Rule selection
#     # -----------------------------
#     rule_names = list(rules_all.keys())
#     selected_rules = st.sidebar.multiselect(
#         "Select Rules", rule_names, default=selected_rules[:3]
#     )

#     # -----------------------------
#     # Input source
#     # -----------------------------
#     source_option = st.sidebar.radio("Input Source", ["Webcam", "Upload MP4 Video"])

#     # -----------------------------
#     # Floor controls
#     # -----------------------------
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("**Floor Line**")

#     def nudge_floor(delta):
#         st.session_state["floor_y"] = max(0, min(1080, st.session_state["floor_y"] + delta))

#     def sync_from_slider():
#         st.session_state["floor_y"] = st.session_state["_floor_slider"]

#     def sync_from_number():
#         st.session_state["floor_y"] = st.session_state["_floor_number"]

#     col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
#     col_down.button("▼", on_click=nudge_floor, args=(-5,))
#     col_up.button("▲", on_click=nudge_floor, args=(+5,))
#     col_val.number_input(
#         "Y", min_value=0, max_value=1080,
#         value=st.session_state["floor_y"],
#         step=1, key="_floor_number",
#         on_change=sync_from_number, label_visibility="collapsed"
#     )
#     st.sidebar.slider(
#         "Fine-tune floor", 0, 1080,
#         value=st.session_state["floor_y"],
#         key="_floor_slider", on_change=sync_from_slider,
#         label_visibility="collapsed"
#     )

#     # -----------------------------
#     # Webcam start checkbox
#     # -----------------------------
#     run_webcam = False
#     if source_option == "Webcam":
#         run_webcam = st.sidebar.checkbox("Start Webcam", value=False)

#     return selected_rules, source_option, run_webcam
#---------------------------------------------------------------------------------------
# # ui/sidebar.py
# import streamlit as st

# def setup_sidebar(rules_all, selected_rules):
#     # -----------------------------
#     # Rule selection (UI improved)
#     # -----------------------------
#     st.sidebar.markdown("### 📋 Select Rules")

#     rule_names = list(rules_all.keys())

#     # persist selection
#     if "selected_rules" not in st.session_state:
#         st.session_state["selected_rules"] = selected_rules[:3]

#     selected_rules = st.sidebar.multiselect(
#         "Rules",
#         rule_names,
#         default=st.session_state["selected_rules"],
#         label_visibility="collapsed",
#     )

#     st.session_state["selected_rules"] = selected_rules

#     # Optional: show rule details (from your commented version)
#     with st.sidebar.expander("📖 Rule Details", expanded=False):
#         for r in selected_rules:
#             rule_def = rules_all.get(r, {})
#             parts = ", ".join(
#                 f"`{feat}` [{lo}–{hi}]"
#                 for feat, (lo, hi) in rule_def.items()
#             )
#             st.markdown(f"- **{r}** — {parts if parts else '—'}")

#     # -----------------------------
#     # Input source (clean UI)
#     # -----------------------------
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 📷 Input Source")

#     source_option = st.sidebar.radio(
#         "Source",
#         ["Webcam", "Upload MP4 Video"],
#         label_visibility="collapsed",
#     )

#     # -----------------------------
#     # Floor controls (same logic, nicer UI)
#     # -----------------------------
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 📐 Floor Line")

#     # ensure exists
#     if "floor_y" not in st.session_state:
#         st.session_state["floor_y"] = 500

#     def nudge_floor(delta):
#         st.session_state["floor_y"] = max(
#             0, min(1080, st.session_state["floor_y"] + delta)
#         )

#     def sync_from_slider():
#         st.session_state["floor_y"] = st.session_state["_floor_slider"]

#     def sync_from_number():
#         st.session_state["floor_y"] = st.session_state["_floor_number"]

#     col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])

#     col_down.button("▼", on_click=nudge_floor, args=(-5,))
#     col_up.button("▲", on_click=nudge_floor, args=(+5,))

#     col_val.number_input(
#         "Y",
#         min_value=0,
#         max_value=1080,
#         value=st.session_state["floor_y"],
#         step=1,
#         key="_floor_number",
#         on_change=sync_from_number,
#         label_visibility="collapsed",
#     )

#     st.sidebar.slider(
#         "Fine-tune floor",
#         0,
#         1080,
#         value=st.session_state["floor_y"],
#         key="_floor_slider",
#         on_change=sync_from_slider,
#         label_visibility="collapsed",
#     )

#     # -----------------------------
#     # Webcam control (UI improved)
#     # -----------------------------
#     run_webcam = False
#     if source_option == "Webcam":
#         st.sidebar.markdown("---")
#         run_webcam = st.sidebar.checkbox("▶ Start Webcam", value=False)

#     # -----------------------------
#     # Return (UNCHANGED)
#     # -----------------------------
#     return selected_rules, source_option, run_webcam


# ui/sidebar.py
import streamlit as st

# ── Utility: load rules from current rules.txt ──────────────────────────────
import os

def load_rule_groups(filepath):
    """
    Parses rules.txt in format:
        Exercise|feature|min|max
    Returns:
        dict: { exercise_name: [feature1, feature2, ...] }
    """
    groups = {}
    
    # Make filepath absolute relative to the temp3 root directory
    if not os.path.isabs(filepath):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        filepath = os.path.join(base_dir, filepath)
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            exercise_name, feature, *_ = parts
            if exercise_name not in groups:
                groups[exercise_name] = []

            groups[exercise_name].append(feature)
    return groups

# ── Main sidebar setup ─────────────────────────────────────────────────────
def setup_sidebar(rules_all, selected_rules):
    # -----------------------------
    # Exercise selection
    # -----------------------------
    st.sidebar.markdown("### 🏃 Exercise")

    if "_rule_groups" not in st.session_state:
        st.session_state["_rule_groups"] = load_rule_groups("assets/rules.txt")

    rule_groups = st.session_state["_rule_groups"]

    # only show exercise names
    exercise_name = st.sidebar.radio(
        "Select Exercise",
        list(rule_groups.keys()),
        label_visibility="collapsed",
    )

    # internal rules for the chosen exercise
    selected_rules = rule_groups[exercise_name]

    # Optional: show features in an expander
    with st.sidebar.expander("📋 Exercise Details", expanded=False):
        for feature in selected_rules:
            st.markdown(f"- {feature}")

    # -----------------------------
    # Input source
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📷 Input Source")

    source_option = st.sidebar.radio(
        "Source",
        ["Webcam", "Upload MP4 Video"],
        label_visibility="collapsed",
    )
    
    # Camera Lens setting to help GVHMR
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Camera Lens (Focal Length)")
    lens_type = st.sidebar.selectbox(
        "For better 3D tracking, what kind of lens was this video shot on?",
        [
            "Standard / 1x (24mm)", 
            "Ultra Wide / 0.5x (13mm)", 
            "Telephoto / 2x (48mm)", 
            "Telephoto / 3x (77mm)",
            "Default / Unknown"
        ]
    )
    
    # Map selection to mm integer
    f_mm_map = {
        "Ultra Wide / 0.5x (13mm)": 13,
        "Standard / 1x (24mm)": 24,
        "Telephoto / 2x (48mm)": 48,
        "Telephoto / 3x (77mm)": 77,
        "Default / Unknown": None
    }
    st.session_state["camera_f_mm"] = f_mm_map[lens_type]

    # -----------------------------
    # Floor controls
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Floor Line")

    if "floor_y" not in st.session_state:
        st.session_state["floor_y"] = 500

    def nudge_floor(delta):
        st.session_state["floor_y"] = max(
            0, min(3000, st.session_state["floor_y"] + delta)
        )

    def sync_from_slider():
        st.session_state["floor_y"] = st.session_state["_floor_slider"]

    def sync_from_number():
        st.session_state["floor_y"] = st.session_state["_floor_number"]

    # Synchronize widget state blocks to match the source of truth BEFORE drawing widgets
    if "_floor_number" in st.session_state and st.session_state["_floor_number"] != st.session_state["floor_y"]:
        st.session_state["_floor_number"] = st.session_state["floor_y"]
    if "_floor_slider" in st.session_state and st.session_state["_floor_slider"] != st.session_state["floor_y"]:
        st.session_state["_floor_slider"] = st.session_state["floor_y"]

    col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
    col_down.button("▼", on_click=nudge_floor, args=(-5,))
    col_up.button("▲", on_click=nudge_floor, args=(+5,))

    col_val.number_input(
        "Y",
        min_value=0,
        max_value=3000,
        step=1,
        key="_floor_number",
        on_change=sync_from_number,
        label_visibility="collapsed",
    )

    st.sidebar.slider(
        "Fine-tune floor",
        0,
        3000,
        key="_floor_slider",
        on_change=sync_from_slider,
        label_visibility="collapsed",
    )

    # -----------------------------
    # Webcam checkbox
    # -----------------------------
    run_webcam = False
    if source_option == "Webcam":
        st.sidebar.markdown("---")
        run_webcam = st.sidebar.checkbox("▶ Start Webcam", value=False)

    # -----------------------------
    # Return
    # -----------------------------
    return selected_rules, source_option, run_webcam, exercise_name