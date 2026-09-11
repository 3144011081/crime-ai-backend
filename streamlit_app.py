import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Import 3-Model AI Video Processing & Adaptive Zoom Engine
import video_inference1

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aerial Surveillance AI Monitor",
    page_icon="🚁",
    layout="wide"
)

st.title("🚁 Aerial Surveillance & Crime Detection Dashboard")
st.caption("3-Model AI Architecture (R(2+1)D Crime Classifier + Drone Person YOLO + Weapon YOLO + High-Altitude Pixel Zooming)")

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Sidebar Configuration & 3-Model System Status
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model & Detection Settings")

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "models" / "crime_aerial_augmented_best1.pth")
weights_path = st.sidebar.text_input(
    "Crime Model (.pth)",
    value=DEFAULT_CHECKPOINT
)

confidence_threshold = st.sidebar.slider(
    "Alert Confidence Threshold (%)",
    min_value=10,
    max_value=90,
    value=35,
    step=5
) / 100.0

clip_stride = st.sidebar.select_slider(
    "Temporal Sliding Window Stride (Frames)",
    options=[4, 8, 16],
    value=8,
    help="Smaller stride = higher temporal precision; Larger stride = faster inference."
)

st.sidebar.divider()
st.sidebar.subheader("🎨 Image & Video Preprocessing")
preprocess_selection = st.sidebar.selectbox(
    "Preprocessing Enhancement Mode",
    options=[
        "Auto (Adaptive Scene Analysis)",
        "Aerial Drone (CLAHE + Dehaze)",
        "Low-Light / Night (Shadow Lift + AGCWD)",
        "YOLO Small Target Booster (Edge + Contrast)",
        "None (Passthrough Raw)"
    ],
    index=0,
    help="Preprocesses frames before Stage 1 (YOLO) and Stage 2 (R(2+1)D) to boost detection recall and temporal stability."
)

# Map human-readable option to internal mode name
MODE_MAP = {
    "Auto (Adaptive Scene Analysis)": "auto",
    "Aerial Drone (CLAHE + Dehaze)": "aerial_drone",
    "Low-Light / Night (Shadow Lift + AGCWD)": "low_light_night",
    "YOLO Small Target Booster (Edge + Contrast)": "yolo_enhanced",
    "None (Passthrough Raw)": "none"
}
selected_mode_key = MODE_MAP.get(preprocess_selection, "auto")

st.sidebar.divider()
st.sidebar.subheader("🤖 3 Trained AI Models Pipeline")
st.sidebar.info("1. 🚁 **Crime Classifier**: `crime_aerial_augmented_best1.pth`\n2. 👤 **People Detector**: `drone_person_detector_best.pt`\n3. ⚠️ **Weapon Detector**: `weapon_detector_best.pt`")

st.sidebar.divider()
st.sidebar.subheader("🔎 Adaptive Video Acquisition")
st.sidebar.caption("Applies multi-scale pixel ROI zooming when drone is 50m+ high or scene is unclear.")

# ---------------------------------------------------------
# Video Upload & Selection Section
# ---------------------------------------------------------
col_u1, col_u2 = st.columns([3, 2])

with col_u1:
    uploaded_file = st.file_uploader("Upload Drone/Aerial Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])

with col_u2:
    available_samples = sorted([f.name for f in VIDEOS_DIR.glob("*.mp4")]) if VIDEOS_DIR.exists() else []
    sample_choice = st.selectbox(
        "Or select a sample video from repository:",
        options=["-- Select Sample --"] + available_samples
    )

target_video_path = None

if uploaded_file is not None:
    try:
        clean_name = Path(uploaded_file.name).name
        save_path = UPLOADS_DIR / clean_name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_video_path = str(save_path)
    except Exception as err:
        st.error(f"Error saving uploaded file: {err}")
elif sample_choice != "-- Select Sample --":
    candidate = VIDEOS_DIR / sample_choice
    if candidate.exists():
        target_video_path = str(candidate)

# ---------------------------------------------------------
# Video Analysis Execution with Real-Time Progress Bar
# ---------------------------------------------------------
if target_video_path and os.path.exists(target_video_path):
    st.divider()
    v_col1, v_col2 = st.columns([3, 2])

    with v_col1:
        st.subheader("📹 Input Video Source")
        st.video(target_video_path)

    with v_col2:
        st.subheader("🚀 Trigger AI Analysis")
        st.write("Click below to run 3-Model AI spatio-temporal crime classification, entity tracking, high-altitude pixel zooming, and evidence extraction.")
        start_analysis = st.button("🚀 Start Aerial Crime Detection", type="primary", use_container_width=True)

    if start_analysis:
        progress_bar = st.progress(0, text="Initializing 3-Model AI Engine...")
        status_box = st.empty()

        def update_gui_progress(pct, msg):
            try:
                val = min(1.0, max(0.0, float(pct)))
                progress_bar.progress(val, text=f"Processing Video... {int(val * 100)}%")
                status_box.caption(f"⚙️ **Pipeline Status**: {msg}")
            except Exception:
                pass

        try:
            if hasattr(video_inference1, "drone_preprocessor") and video_inference1.drone_preprocessor is not None:
                video_inference1.drone_preprocessor.mode = selected_mode_key
            results = video_inference1.process_video(target_video_path, progress_callback=update_gui_progress)
            st.session_state["last_results"] = results
            progress_bar.progress(1.0, text="Analysis Complete! 100%")
            status_box.empty()
            st.success("🎉 AI 3-Model Analysis & Adaptive Zooming Complete!")
        except Exception as err:
            st.error(f"Error during video processing: {err}")

    # Render results if available
    if "last_results" in st.session_state:
        res = st.session_state["last_results"]
        st.divider()

        # High Altitude Zoom Status Pill
        acq_mode = res.get("acquisition_mode", "original")
        if acq_mode != "original":
            st.info(f"🔍 **High-Altitude Pixel Zooming Activated (`{acq_mode}`)**: Drone camera was high or scene was unclear. Multi-scale pixel ROI zooming was applied to enhance subject resolution and AI detection confidence.")
        else:
            st.success("🟢 **Original Aerial View**: Scene was clear and visible. No pixel zooming required.")

        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        is_crime = res.get("crime", False)
        crime_type = res.get("type", "Normal")
        conf_pct = res.get("confidence", 0.0) * 100

        m1.metric("Overall Status", "CRIME ALERT" if is_crime else "NORMAL", delta="CRITICAL" if is_crime else "SAFE", delta_color="inverse" if is_crime else "normal")
        m2.metric("Classification", crime_type)
        m3.metric("Confidence", f"{conf_pct:.1f}%")
        m4.metric("People Count", res.get("people_count", 0))
        m5.metric("Weapons Count", res.get("weapon_count", 0))

        # Output Video & Details
        col_res_vid, col_res_info = st.columns([3, 2])

        with col_res_vid:
            st.subheader("🎯 Real-Time Detection Feed (Annotated)")
            output_vid_name = res.get("output_video", "")
            output_vid_path = OUTPUT_DIR / output_vid_name if output_vid_name else None
            if output_vid_path and output_vid_path.exists():
                st.video(str(output_vid_path))
            elif output_vid_name and os.path.exists(output_vid_name):
                st.video(output_vid_name)
            else:
                st.warning("Output video rendering in progress or not found.")

        with col_res_info:
            st.subheader("📊 Detected Activities & Telemetry")
            activities = res.get("detected_activities", [])
            if activities:
                for act in activities:
                    st.info(f"🔹 {act}")
            else:
                st.write("No specific suspicious activities logged.")

            st.write("#### Description")
            st.write(res.get("description", "No detailed description."))

        # Evidence Clips Gallery
        crime_clips = res.get("crime_clips", [])
        if crime_clips:
            st.divider()
            st.subheader("🚨 Extracted Evidence Clips")
            clip_cols = st.columns(min(len(crime_clips), 3))
            for idx, clip_file in enumerate(crime_clips):
                c_path = EVIDENCE_DIR / clip_file
                with clip_cols[idx % 3]:
                    st.write(f"**Clip {idx+1}: {clip_file}**")
                    if c_path.exists():
                        st.video(str(c_path))
                    else:
                        st.caption("Clip file unavailable.")

        # Download Report Section
        st.divider()
        st.subheader("📥 Export Surveillance Audit Report")
        r_col1, r_col2 = st.columns(2)

        json_data = json.dumps(res, indent=2, default=str)
        with r_col1:
            st.download_button(
                label="Download JSON Audit Report",
                data=json_data,
                file_name="crime_surveillance_report.json",
                mime="application/json"
            )

        # Build CSV data from results list
        window_results = res.get("results", [])
        if window_results:
            df = pd.DataFrame(window_results)
            csv_data = df.to_csv(index=False).encode("utf-8")
            with r_col2:
                st.download_button(
                    label="Download CSV Telemetry Log",
                    data=csv_data,
                    file_name="crime_surveillance_log.csv",
                    mime="text/csv"
                )

# ---------------------------------------------------------
# Standalone execution launcher
# ---------------------------------------------------------
if __name__ == "__main__":
    pass
