from datetime import datetime
import time
from user_input_converter import convert_additional_parameters
import streamlit as st
import os
import cv2
import numpy as np
import subprocess
import base64
import zipfile
import io
import pandas as pd
import ast
from process_skeleton_video import process_video
from hand_heatmap_from_video import hand_heatmap_two_color
from streamlit_javascript import st_javascript
import streamlit.components.v1 as components
import People_Tracker_with_Optical_Flow as PT
from ultralytics import YOLO

# Directories
IMPORTS_DIR = "imports"
EXPORTS_DIR = "exports"

# Create directories if not exist
for directory in [IMPORTS_DIR, EXPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set wide layout
st.set_page_config(layout="wide")


# --- Cached YOLO model for Optical Flow Tracker ---
@st.cache_resource
def get_tracker_model():
    return YOLO("yolov8n.pt")


# --- Méret beállítások ---
DEFAULT_RESULT_WIDTH = 800  # nagyobb alapértelmezett méret

if "result_width" not in st.session_state:
    st.session_state.result_width = DEFAULT_RESULT_WIDTH

st.markdown(f"""
<style>
div.stButton > button {{
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 16px;
    font-weight: 500;
    width: 100%;
    text-align: left;
    border: none;
    background-color: transparent;
    color: #1A73E8;
    margin-bottom: 6px;
    transition: all 0.2s ease-in-out;
}}
div.stButton > button:hover:enabled {{
    background-color: #e8f0fe !important;
    color: #174ea6 !important;
    cursor: pointer;
}}
div.stButton > button:disabled {{
    background-color: #1A73E8 !important;
    color: white !important;
    font-weight: 600;
    cursor: default;
}}
.centered-content {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""<style>
/* Make radio buttons smaller */
div[data-testid="stRadio"] {
    max-width: 90%;
    width: 90%;
    min-width: 100px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make Streamlit tabs bigger */
div[data-testid="stTabs"] button {
    font-size: 1.2rem !important;
    padding: 1rem 2rem !important;
    min-height: 48px !important;
}
div[data-testid="stTabs"] {
    min-height: 60px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Set a fixed width for text inputs (basic information) */
div[data-testid="stTextInput"] input {
    width: 80% !important;
    max-width: 80% !important;
    min-width: 80% !important;
}

/* Set the same fixed width for the file uploader */
div[data-testid="stFileUploader"] {
    width: 90% !important;
    max-width: 90% !important;
    min-width: 90% !important;
}

/* Set the same fixed width for sliders */
div[data-testid="stSlider"] {
    width: 90% !important;
    max-width: 90% !important;
    min-width: 90% !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make all text bigger */
html, body, [data-testid="stAppViewContainer"] * {
    font-size: 1.15rem !important;
}
/* Make headers even bigger */
h1, .stMarkdown h1 {
    font-size: 2.2rem !important;
}
h2, .stMarkdown h2, .stSubheader {
    font-size: 1.8rem !important;
}
h3, .stMarkdown h3 {
    font-size: 1.4rem !important;
}
</style>
""", unsafe_allow_html=True)


# Helper to show video with width control and center
def show_video_centered(video_path, width=500):
    """
    Displays a video centered and responsive in the Streamlit layout.
    Uses st.video for best performance and compatibility.
    """
    st.markdown(f'<div class="centered-content" style="max-width:{width}px;">', unsafe_allow_html=True)
    st.video(video_path)
    st.markdown('</div>', unsafe_allow_html=True)


# Initialize session state
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Ergonomy Assessment"
if "processing_video" not in st.session_state:
    st.session_state.processing_video = False
if "show_slider" not in st.session_state:
    st.session_state.show_slider = False
if "assessment_method" not in st.session_state:
    st.session_state.assessment_method = ""
if "side_choose" not in st.session_state:
    st.session_state.side_choose = False
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "last_results" not in st.session_state:
    st.session_state.last_results = {}
if "selected_frame_idx" not in st.session_state:
    st.session_state.selected_frame_idx = 0


# Button-based navigation
def nav_button(label):
    is_active = st.session_state.selected_menu == label
    if is_active:
        st.button(label, key=label, use_container_width=True, disabled=True)
    else:
        if st.button(label, key=label, use_container_width=True):
            st.session_state.selected_menu = label
            st.rerun()


def convert_to_h264(input_path, output_path):
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e}")


def process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, assess_method, processing_rate):
    print(f"process_streamlit_video: Input video is in {file_path}")
    try:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        temp_video_path = os.path.join(EXPORTS_DIR, f"temp_{os.path.basename(file_path)}")
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        temp_out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        if not cap.isOpened():
            raise Exception("Could not open input video file.")
        if start_frame >= end_frame or start_frame < 0:
            raise Exception("Invalid frame range specified.")

        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = end_frame - start_frame
        frames_to_process = len(list(range(start_frame, end_frame, processing_rate)))
        print(f"start_frame={start_frame}, end_frame={end_frame}, processing_rate={processing_rate}, frames_to_process={frames_to_process}")

        if frames_to_process < 1 or total_frames < 1:
            st.error("The selected range and processing rate result in zero frames to process. Please select a wider range or lower processing rate.")
            return

        processed_count = 0
        status_placeholder = st.empty()
        frame_num = start_frame
        while frame_num < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {frame_num} could not be read, stopping.")
                break
            temp_out.write(frame)

            processed_count += 1
            progress = processed_count / frames_to_process
            progress_bar.progress(progress)
            status_placeholder.text(f"Processing frame {processed_count}/{frames_to_process} (video frame {frame_num + 1}/{end_frame})...")

            frame_num += processing_rate

        temp_out.release()
        cap.release()
        converted_temp_path = temp_video_path.replace(".mp4", "_h264.mp4")
        convert_to_h264(temp_video_path, converted_temp_path)
        print("Converted temp video exists:", os.path.exists(converted_temp_path))
        print("Converted temp video size:", os.path.getsize(converted_temp_path) if os.path.exists(converted_temp_path) else "N/A")


        status_placeholder.text("Performing assessment...")

        if assess_method == "hand_heatmap":
            output_video_file_path, _ = hand_heatmap_two_color(
                filename=os.path.basename(converted_temp_path), status_placeholder = status_placeholder)
        else:
            process_video(
                filename=converted_temp_path,
                assess_method=assess_method,
                start_frame=start_frame,
                end_frame=end_frame,
                processing_rate=processing_rate,
                additional_data= additional_data,
                assessor=assessor,
                task=task,
                workstation=workstation
            )
            output_video_file_path = None

        progress_bar.progress(0.75)
        status_placeholder.text("Converting video to H.264 format...")

        filename_short = os.path.splitext(os.path.basename(converted_temp_path))[0]
        output_name = f'output_{filename_short}_{assess_method}.mp4'
        processed_video_path = os.path.join(EXPORTS_DIR, output_name)

        if not os.path.exists(processed_video_path):
            raise Exception(f"Processed video not found: {processed_video_path}")

        final_output_path = os.path.join(EXPORTS_DIR, f"converted_{output_name}")
        convert_to_h264(processed_video_path, final_output_path)

        progress_bar.progress(1.0)
        status_placeholder.text("Processing complete!")
        st.success("Processing complete!")

        st.session_state.last_results = {
            "final_output_path": final_output_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "side_view": "Left side",
            "fps": fps,
            "assess_method": assess_method,
            "processing_rate": processing_rate,
            "additional_data": additional_data
        }
        st.session_state.show_results = True
        st.rerun()

    except Exception as e:
        st.error(f"Processing error: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.session_state.is_processing = False


# === Optical Flow Tracker helpers ===
def _opt_apply_params_to_module(param_dict: dict):
    """Frissíti a PT.params-ot és a megfelelő modul globálisokat (WIN, LEVELS, stb.)."""
    try:
        if hasattr(PT, "params") and isinstance(PT.params, dict):
            PT.params.update(param_dict)
        else:
            PT.params = dict(param_dict)
    except Exception:
        PT.params = dict(param_dict)

    for k, v in param_dict.items():
        if k == "text_color" and hasattr(PT, "TEXT_COLOR"):
            try:
                setattr(PT, "TEXT_COLOR", tuple(v) if isinstance(v, (list, tuple)) else v)
                continue
            except Exception:
                pass
        UPPER = k.upper()
        if hasattr(PT, UPPER):
            try:
                setattr(PT, UPPER, v)
            except Exception:
                pass

    if hasattr(PT, "WIN") and hasattr(PT, "LEVELS"):
        try:
            PT.lk_params = dict(
                winSize=(int(PT.WIN), int(PT.WIN)),
                maxLevel=int(PT.LEVELS),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1e-10),
            )
        except Exception:
            pass


def _opt_run_yolo_and_build_detections(model, frame, w, h):
    """YOLO detektálás + People_Tracker formátumú detekciók építése."""
    results = model(frame, classes=[0], iou=0.4, verbose=True)
    # Eredeti stílusú log
    try:
        res0 = results[0] if isinstance(results, (list, tuple)) else results
        n_persons = len(getattr(res0, "boxes", []) or [])
        spd = getattr(res0, "speed", {}) or {}
        pp = float(spd.get("preprocess", 0.0))
        inf = float(spd.get("inference", 0.0))
        post = float(spd.get("postprocess", 0.0))
        log_h, log_w = frame.shape[:2]
        print(f"0: {log_h}x{log_w} {n_persons} persons, {inf:.1f}ms")
        print(f"Speed: {pp:.1f}ms preprocess, {inf:.1f}ms inference, {post:.1f}ms postprocess per image at shape (1, 3, {log_h}, {log_w})")
    except Exception:
        pass

    detections = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            p0 = PT.make_support_grid(cx, cy, PT.NEIGH_RADIUS, PT.GRID_SPACING, w, h)
            if p0 is None:
                p0 = np.array([[[cx, cy]]], dtype=np.float32)
            detections.append((cx, cy, p0, (x1, y1, x2, y2)))
    return detections


def _opt_assign_only_list(ret):
    """Ha a hozzárendelő tuple-t ad vissza (list, extra), csak a listát veszi ki."""
    try:
        if isinstance(ret, tuple) and len(ret) >= 1:
            return ret[0]
    except Exception:
        pass
    return ret


# UI Layout
col1, col2 = st.columns([1, 5])

with col1:
    with st.container():
        st.image("assets/PE_logo_blue.png", width=120)
        st.markdown("###")
        nav_button("Ergonomy Assessment")
        nav_button("Optical Flow Tracker")
        nav_button("Activity Assessment")
        nav_button("Work Instructions")
        nav_button("Chatbot")
        nav_button("Result Files")
        st.markdown("###")
        st.image("assets/MK_logo_blue.png", width=120)
        st.markdown("###")
        st.slider("Result width", min_value=300, max_value=1200, value=st.session_state.result_width, key="result_width")

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Ergonomy Assessment -------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
with col2:
    if st.session_state.selected_menu == "Ergonomy Assessment":
        st.header("Upload and edit a video file")
        if st.session_state.show_results and st.session_state.last_results:
            results = st.session_state.last_results
            st.success("Processing complete!")
            # --- Video középre igazítva, szabályozható szélességgel ---
            try:
                show_video_centered(results["final_output_path"], width=200)
            except Exception:
                st.warning("The result video could not be loaded.")

            side_view = st.selectbox(
                "Choose the side of the view",
                ['Left side', 'Right side'],
                index=0 if results["side_view"] == "Left side" else 1
            )

            label_map = {
            "neck_twisted": "Neck: Twisted?",
            "neck_bent": "Neck: Side bending?",
            "trunk_twisted": "Trunk: Twisted?",
            "trunk_bent": "Trunk: Side bending?",
            "upper_arms_raised_l": "Upper Arms: Shoulder raised (Left)",
            "upper_arms_raised_r": "Upper Arms: Shoulder raised (Right)",
            "upper_arms_abducted_l": "Upper Arms: Abducted? (Left)",
            "upper_arms_abducted_r": "Upper Arms: Abducted? (Right)",
            "upper_arms_supported_l": "Upper Arms: Supported or leaning? (Left)",
            "upper_arms_supported_r": "Upper Arms: Supported or leaning? (Right)",
            "lower_arms_mid_l": "Lower Arms: Arm mid or side position (Left)",
            "lower_arms_mid_r": "Lower Arms: Arm mid or side position (Right)",
            "wrists_bent_l": "Wrists: Wrist bent/twisted from midline? (Left)",
            "wrists_bent_r": "Wrists: Wrist bent/twisted from midline? (Right)",
            "wrist_twist_endline_l": "Wrists: Left wrist action",
            "wrist_twist_endline_r": "Wrists: Right wrist action",
            "load_force_arms_l": "Load (Arms): Weight on left hand",
            "load_force_arms_r": "Load (Arms): Weight on right hand",
            "load_force_upper_body": "Load (Upper Body): Weight",
            "load_force_legs": "Load (Legs): Weight",
            "load_shock": "Load: Shock or rapid build-up?",
            "muscle_arms_l": "Muscle Use: Left arm",
            "muscle_arms_r": "Muscle Use: Right arm",
            "muscle_legs": "Muscle Use: Legs used for support",
            "coupling_score_l": "Coupling: Left hand",
            "coupling_score_r": "Coupling: Right hand",
            "activity_score": "Activity Score (total)",
        }

            st.markdown("Choosen additional parameters:")
            additional_params = results.get("additional_data", {})
            num_columns = 3
            keys = list(additional_params.keys())
            columns = st.columns(num_columns)

            for idx, key in enumerate(keys):
                label = label_map.get(key, key)  # fallback to key if label missing
                value = additional_params[key]
                col = columns[idx % num_columns]
                if key == "load_force_arms_l":
                    load_weight_str = st.session_state.get("load_weight_l", "N/A")
                    value_display = f"{load_weight_str}"
                elif key == "load_force_arms_r":
                    load_weight_str = st.session_state.get("load_weight_r", "N/A")
                    value_display = f"{load_weight_str}"
                elif key == "load_force_upper_body":
                    load_weight_str = st.session_state.get("load_weight_upper_body", "N/A")
                    value_display = f"{load_weight_str}"
                elif key == "load_force_legs":
                    load_weight_str = st.session_state.get("load_weight_legs", "N/A")
                    value_display = f"{load_weight_str}"
                elif value is None:
                    value_display = "N/A"
                elif isinstance(value, str):
                    value_display = value
                elif isinstance(value, (int, bool)):
                    value_display = "Yes" if value else "No"
                else:
                    value_display = str(value)
                with col:
                    st.markdown(f"**{label}**: {value_display}")

            # --- Kép középre igazítva és slider szélesség igazítása ---
            st.markdown('<div class="centered-content">', unsafe_allow_html=True)
            chart_img_path = os.path.join(
                EXPORTS_DIR, f"line_chart_{'left' if side_view == 'Left side' else 'right'}.png"
            )
            print(f'The chart will be write to {chart_img_path}')
            if os.path.exists(chart_img_path):
                st.image(chart_img_path, caption=f"{side_view} risk chart", width=st.session_state.result_width)
            else:
                st.warning("The selected line chart image is not available.")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Frame slider and video jump under the chart ---
            st.markdown('<div class="centered-content">', unsafe_allow_html=True)
            cap = cv2.VideoCapture(results["final_output_path"])
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0 and fps > 0:
                st.markdown(
                    f"""
                    <style>
                    div[data-testid="stSlider"] {{
                        width: {st.session_state.result_width}px !important;
                        max-width: 100% !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                selected_idx = st.slider(
                    "Jump to frame:",
                    min_value=0,
                    max_value=total_frames - 1,
                    value=0,
                    step=1,
                    key="selected_frame_idx_slider"
                )
                # Mindig a slider visszaadott értékét használd!
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, channels="RGB", caption=f"Frame {selected_idx}", width=st.session_state.result_width)
                else:
                    st.warning("Could not load the selected frame.")
            cap.release()
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("New analysis"):
                st.session_state.show_results = False
                st.session_state.last_results = {}
                st.session_state.selected_frame_idx = 0
                st.rerun()
        else:
            uploaded_file = st.file_uploader("Select a video...", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                file_path = os.path.join(IMPORTS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded video: {uploaded_file.name}")

            video_files = os.listdir(IMPORTS_DIR)
            video_options = ["Select a video..."] + video_files
            if video_files:
                selected_video = st.selectbox("Select a video to process:", video_options, disabled=st.session_state.is_processing)
                if selected_video != "Select a video...":
                    left_col, right_col = st.columns([2, 3])

                    with left_col:
                        file_path = os.path.join(IMPORTS_DIR, selected_video)
                        # Limit video width to 500px for better fit
                        st.markdown('<div class="centered-content" style="max-width:500px;">', unsafe_allow_html=True)
                        st.video(file_path)
                        st.markdown('</div>', unsafe_allow_html=True)
                        cap = cv2.VideoCapture(file_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = total_frames / fps if fps > 0 else 0
                        cap.release()
                        st.write(f"Video duration: {duration:.2f} seconds")
                        start_time, end_time = st.slider(
                            "Select processing time range (in seconds):",
                            min_value=0.0,
                            max_value=float(duration),
                            value=(0.0, float(duration)),
                            step=0.1
                        )

                    with right_col:
                        st.text(f"Add basic information of the video:")
                        assessor = st.text_input("Assessor", key="assessor")
                        task = st.text_input("Task", key="task")
                        workstation = st.text_input("Workstation", key="workstation")
                        st.text("Select the additional parameters for the analysis:")
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Posture", "Load", "Muscle Use", "Coupling", "Activity Score"])

                        with tab1:
                            st.subheader("Neck:")
                            neck_twisted = st.checkbox("Twisted?", key="neck_twisted")
                            neck_bent = st.checkbox("Side bending?", key="neck_bent")
                            st.subheader("Trunk:")
                            trunk_twisted = st.checkbox("Twisted?", key="trunk_twisted")
                            trunk_bent = st.checkbox("Side bending?", key="trunk_bent")
                            st.subheader("Upper Arms:")
                            upper_arms_raised_l = st.checkbox("Shoulder raised (Left)", key="ua_raised_l")
                            upper_arms_raised_r = st.checkbox("Shoulder raised (Right)", key="ua_raised_r")
                            upper_arms_abducted_l = st.checkbox("Abducted?(Left)", key="ua_abducted_l")
                            upper_arms_abducted_r = st.checkbox("Abducted?(Right)", key="ua_abducted_r")
                            upper_arms_supported_l = st.checkbox("Supported or leaning?(Left)", key="ua_supported-l")
                            upper_arms_supported_r = st.checkbox("Supported or leaning?(Right)", key="ua_supported_r")
                            st.subheader("Lower Arms:")
                            lower_arms_mid_l = st.checkbox("Arm mid or side position(Left)", key="lower_arms_mid_l")
                            lower_arms_mid_r = st.checkbox("Arm mid or side position (Right)", key="lower_arms_mid_r")
                            st.subheader("Wrists:")
                            wrists_bent_l = st.checkbox("Wrist bent/twisted from midline?(Left)", key="wrists_bent_l")
                            wrists_bent_r = st.checkbox("Wrist bent/twisted from midline?(Right)", key="wrists_bent_r")
                            wrist_twist_endline_l = st.radio("Left wrist action", ("None", "Wrist twisted in mid-range", "Wrist is at or near end of range"), key="wrist_twist_endline_l")
                            wrist_twist_endline_r = st.radio("Right wrist action", ("None", "Wrist twisted in mid-range", "Wrist is at or near end of range"), key="wrist_twist_endline_r")
                        with tab2:
                            st.subheader("Load:")
                            st.subheader("For Arms:")
                            load_weight_arms_l = st.text_input("Load weight for left hand(kg)", key="load_weight_l")
                            load_weight_arms_r = st.text_input("Load weight for right hand(kg)", key="load_weight_r")
                            load_variation_arms_l = st.radio("Load movement type for left arm", ("Intermittent", "Static or repeated"), key="load_variation_l")
                            load_variation_arms_r = st.radio("Load movement type for right arm", ("Intermittent", "Static or repeated"), key="load_variation_r")
                            st.subheader("For Legs:")
                            load_shock = st.checkbox("Shock or rapid build-up?", key="load_shock")
                            load_weight_legs = st.text_input("Load weight for legs(kg)", key="load_weight_legs")
                            st.subheader("For Upper Body:")
                            load_weight_upper_body = st.text_input("Load weight for upper body(kg)", key="load_weight_upper_body")
                            load_variation_upper_body = st.radio("Load movement type for upper body", ("Intermittent", "Static or repeated"), key="load_variation_upper_body")
                        with tab3:
                            st.subheader("Muscle Use:")
                            st.text("Posture mainly static(held > 1 minute) or action repeated 4X per minute")
                            st.subheader("For Arms:")
                            muscle_arms_l = st.checkbox("Left hand", key="muscle_l")
                            muscle_arms_r = st.checkbox("Right hand", key="muscle_r")
                            st.subheader("For Legs:")
                            muscle_legs = st.checkbox("Legs used for support?", key="muscle_legs")
                        with tab4:
                            st.subheader("Coupling Scores for Arms:")
                            coupling = ['Well fitting Handle and mid range power grip', 'Accepltable but not ideal hand hold or coupling','Acceptable with another body part'
                                        'Hand hold not acceptable but possible', 'No handles, awkward, unsafe with any body part'
                                        ]
                            coupling_score_l = st.selectbox("pick one", coupling, key="coupling_score_l")
                            coupling_score_r = st.selectbox("pick one", coupling, key="coupling_score_r")
                        with tab5:
                            st.subheader("Activity Score:")
                            activity_score_1 = st.checkbox("1 or more body parts are held for longer than 1 minute(static)", key="activity_score_1")
                            activity_score_2 = st.checkbox("Repeated small range actions(more than 4X per minute)", key="activity_score_2")
                            activity_score_3 = st.checkbox("Action causes rapid large range changes in postures or unstable base", key="activity_score_3")

                    wrist_twist_endline_l, wrist_twist_endline_r, coupling_score_l, coupling_score_r , load_force_arms_l, load_force_arms_r , load_force_legs, load_force_upper_body = convert_additional_parameters(wrist_twist_endline_l, wrist_twist_endline_r, coupling_score_l, coupling_score_r, load_weight_arms_l, load_weight_arms_r , load_weight_legs,load_weight_upper_body, load_variation_arms_l, load_variation_arms_r, load_variation_upper_body)

                    additional_data = {
                        "neck_twisted": int(neck_twisted),
                        "neck_bent": int(neck_bent),
                        "trunk_twisted": int(trunk_twisted),
                        "trunk_bent": int(trunk_bent),
                        "upper_arms_raised_l": int(upper_arms_raised_l),
                        "upper_arms_raised_r": int(upper_arms_raised_r),
                        "upper_arms_abducted_l": int(upper_arms_abducted_l),
                        "upper_arms_abducted_r": int(upper_arms_abducted_r),
                        "upper_arms_supported_l": -1 * int(upper_arms_supported_l),
                        "upper_arms_supported_r": -1 * int(upper_arms_supported_r),
                        "lower_arms_mid_l": int(lower_arms_mid_l),
                        "lower_arms_mid_r": int(lower_arms_mid_r),
                        "wrists_bent_l": int(wrists_bent_l),
                        "wrists_bent_r": int(wrists_bent_r),
                        "wrist_twist_endline_l": 2 if wrist_twist_endline_l else 0,
                        "wrist_twist_endline_r": 2 if wrist_twist_endline_r else 0,
                        "load_force_arms_l": int(load_force_arms_l),
                        "load_force_arms_r": int(load_force_arms_r),
                        "load_force_upper_body": int(load_force_upper_body),
                        "load_force_legs": int(load_force_legs),
                        "load_shock": int(load_shock),
                        "muscle_arms_l": int(muscle_arms_l),
                        "muscle_arms_r": int(muscle_arms_r),
                        "muscle_legs": int(muscle_legs),
                        "coupling_score_l": coupling_score_l,
                        "coupling_score_r": coupling_score_r,
                        "activity_score": int(activity_score_1) + int(activity_score_2) + int(activity_score_3),
                    }

                    view_type = st.selectbox(
                        "Choose a view type",
                        ['Top view', 'Side view', 'Front view'],
                        index=1,
                        disabled=st.session_state.is_processing)

                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)
                    assessment_method = ""

                    if view_type == 'Top view':
                        if st.button('Hand heatmap', disabled=st.session_state.is_processing):
                            st.session_state.is_processing = True
                            output_path = os.path.join(EXPORTS_DIR, f"output_temp_{selected_video.split('.')[0]}_hand_heatmap.mp4")
                            progress_bar = st.progress(0)
                            process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, "hand_heatmap", 1)

                    if view_type in ['Side view', 'Front view']:
                        if st.button("REBA calculation", disabled=st.session_state.is_processing):
                            st.session_state.assessment_method = "REBA"
                            st.session_state.show_slider = True

                        if st.button("RULA calculation", disabled=st.session_state.is_processing):
                            st.session_state.assessment_method = "RULA"
                            st.session_state.show_slider = True

                        if st.session_state.show_slider:

                            processing_rate = st.slider(
                                "Processing rate (process every Nth frame):",
                                min_value=1,
                                max_value=60,
                                value=1,
                                step=1,
                                help="Increase to process fewer frames for faster but less detailed analysis."
                            )
                            frames_to_process = len(list(range(start_frame, end_frame, processing_rate)))
                            st.write(f"To be processed: {frames_to_process} frames.")
                            if frames_to_process < 1:
                                st.warning("The selected range and processing rate result in zero frames to process. " \
                                "Please select a wider range or lower processing rate.")

                            if st.button("Process video", disabled=st.session_state.is_processing or frames_to_process < 1):
                                st.session_state.is_processing = True
                                st.session_state.show_slider = False
                                output_path = os.path.join(EXPORTS_DIR, f"output_temp_{selected_video.split('.')[0]}_{st.session_state.assessment_method}.mp4")
                                # output_path = os.path.join(EXPORTS_DIR, f"converted_output_temp_{selected_video.split('.')[0]}_{st.session_state.assessment_method}.mp4")
                                print(f"The file for analysis is in {output_path}")
                                progress_bar = st.progress(0)
                                process_streamlit_video(
                                    file_path, output_path, start_frame, end_frame, fps, progress_bar,
                                    st.session_state.assessment_method, processing_rate
                                )

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Result Files --------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
    elif st.session_state.selected_menu == "Result Files":
        if os.path.exists(EXPORTS_DIR):
            st.header("Result Files")
            video_files = os.listdir(IMPORTS_DIR)
            video_options = ["Select a video..."] + video_files
            chosen_import_file = st.selectbox("Select the input video:", video_options, disabled=st.session_state.is_processing)
            result_files = os.listdir(EXPORTS_DIR)
            result_options = ["Select a file..."] + result_files

            if chosen_import_file != "Select a video...":
                st.write(f"Result files for {chosen_import_file}:")
                search_query = st.text_input("Search result files by date/time (e.g. '2025-07-17' or '14-04-33')", "")
                filtered_files = [f for f in result_files if search_query in f]

                # Pagination setup
                PAGE_SIZE = 10
                if "result_files_page" not in st.session_state:
                    st.session_state.result_files_page = 0
                total_pages = max(1, (len(filtered_files) + PAGE_SIZE - 1) // PAGE_SIZE)
                page = st.session_state.result_files_page

                # Show only files for the current page
                start_idx = page * PAGE_SIZE
                end_idx = start_idx + PAGE_SIZE
                paged_files = filtered_files[start_idx:end_idx]

                st.caption(f"Page {page+1} of {total_pages}")

                if "selected_result_files" not in st.session_state:
                    st.session_state.selected_result_files = set()
                if st.button("Select all"):
                    st.session_state.selected_result_files = set(filtered_files)
                if st.button("Clear selection"):
                    st.session_state.selected_result_files = set()

                # List files with checkboxes
                for file in paged_files:
                    checked = file in st.session_state.selected_result_files
                    if st.checkbox(file, value=checked, key=f"chk_{file}"):
                        st.session_state.selected_result_files.add(file)
                    else:
                        st.session_state.selected_result_files.discard(file)

                # Navigation buttons
                col_prev, col_next = st.columns([1, 1])
                with col_prev:
                    if st.button("Back", disabled=page == 0):
                        st.session_state.result_files_page = max(0, page - 1)
                        st.rerun()
                with col_next:
                    if st.button("Next", disabled=page >= total_pages - 1):
                        st.session_state.result_files_page = min(total_pages - 1, page + 1)
                        st.rerun()

                # Download button
                if st.button("Download selected"):
                    if not st.session_state.selected_result_files:
                        st.warning("No files selected for download.")
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                            for file in st.session_state.selected_result_files:
                                file_path = os.path.join(EXPORTS_DIR, file)
                                if os.path.exists(file_path):
                                    zip_file.write(file_path, arcname=file)
                                else:
                                    st.warning(f"File not found: {file_path}")
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download ZIP",
                            data=zip_buffer,
                            file_name="selected_results.zip",
                            mime="application/zip"
                        )

            else:
                st.info("Please select a video to see the results.")

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Optical Flow Tracker ------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
    elif st.session_state.selected_menu == "Optical Flow Tracker":
        ctrl = st.container()
        with ctrl:
            # --- Felső vezérlők (mindig látszanak) ---
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            btn_manual_init = c1.button("I – manuális reinit")
            btn_reset       = c2.button("D – reset")
            btn_toggle_save = c3.button("S – mentés ki/be")
            btn_toggle_dash = c4.button("B – dashboard ki/be")
            run             = c5.toggle("Futás", value=st.session_state.get("opt_run", False), key="opt_run")

            st.divider()
            col_src1, col_src2 = st.columns([1, 2])
            src_mode = col_src1.radio(
                "Bemenet", ["Webkamera", "Videófájl"],
                index=st.session_state.get("opt_src_mode_idx", 1), key="opt_src_mode"
            )
            vf = None
            if st.session_state["opt_src_mode"] == "Videófájl":
                vf = col_src2.file_uploader(
                    "Videó (mp4/avi/mkv/mov)", type=["mp4", "avi", "mkv", "mov"], key="opt_video_file"
                )

            st.divider()
            # --- param.xlsx feltöltő a csúszkák ELŐTT ---
            st.markdown("**Paraméterek** (alapértékek; fájl betöltése után a csúszkák frissülnek)")
            uploaded_param = st.file_uploader(
                "param.xlsx (két oszlop: param, value)", type=["xlsx"], key="param_xlsx"
            )

            # --- Param csúszkák (defaults; fájlbetöltés után frissülnek) ---
            param_spec = {
                "MEAN_DRIFT_THRESH": ("Mean drift threshold (px)", 0.0, 200.0, 0.5, float),
                "POINT_DROP_RATIO":  ("Point drop ratio",        0.0,   1.0, 0.01, float),
                "ASSIGN_DIST":       ("Assign distance (px)",    0.0, 300.0, 1.0,  float),
                "REINIT_WINDOW":     ("Reinit window (frames)",  1,     240, 1,    int),
                "VISFRAME":          ("Dashboard history (frames)", 50, 1000, 10,   int),
                "N_SAVE_FRAME":      ("Autosave period (frames)",   1,  1000, 1,    int),
                "WIN":               ("LK window (px)",            5,     41, 2,    int),
                "LEVELS":            ("LK pyramid levels",         0,      3, 1,    int),
                "GRID_SPACING":      ("Support grid spacing (px)", 2,     32, 1,    int),
                "NEIGH_RADIUS":      ("Support grid radius (px)",  2,     64, 1,    int),
            }

            def _opt_seed_slider_defaults_from_PT():
                for name in param_spec.keys():
                    if hasattr(PT, name):
                        val = getattr(PT, name)
                    elif hasattr(PT, "params") and isinstance(PT.params, dict):
                        val = PT.params.get(name.lower())
                    else:
                        lo, hi = param_spec[name][1], param_spec[name][2]
                        val = 0.5 * (lo + hi)
                    st.session_state[f"param_{name}"] = val

            # param.xlsx beolvasás és alkalmazás
            if uploaded_param is not None:
                try:
                    dfp = pd.read_excel(uploaded_param)
                    raw = {row["param"]: row["value"] for _, row in dfp.iterrows()}
                    for k, v in list(raw.items()):
                        if isinstance(v, str):
                            try:
                                raw[k] = ast.literal_eval(v)
                            except Exception:
                                pass
                    # Modul frissítése (PT.params + megfelelő globálisok)
                    _opt_apply_params_to_module(raw)
                    # Csúszkák default értékeinek újrainicializálása
                    _opt_seed_slider_defaults_from_PT()
                    st.success("param.xlsx beolvasva és alkalmazva a trackerhez.")
                except Exception as e:
                    st.error(f"param.xlsx beolvasási hiba: {e}")
            else:
                # Első betöltéskor seedeljük a csúszkákat a PT aktuális értékeiből
                for name in param_spec.keys():
                    st.session_state.setdefault(f"param_{name}", None)
                if any(st.session_state[f"param_{n}"] is None for n in param_spec.keys()):
                    _opt_seed_slider_defaults_from_PT()

            sc1, sc2 = st.columns(2)
            _opt_params_updated = {}
            for i, (name, (label, vmin, vmax, step, typ)) in enumerate(param_spec.items()):
                key = f"param_{name}"
                default_val = st.session_state.get(key)
                if default_val is None:
                    if hasattr(PT, name):
                        default_val = getattr(PT, name)
                    elif hasattr(PT, "params") and isinstance(PT.params, dict):
                        default_val = PT.params.get(name.lower())
                    if default_val is None:
                        lo, hi = param_spec[name][1], param_spec[name][2]
                        default_val = 0.5 * (lo + hi)
                col = sc1 if (i % 2 == 0) else sc2
                if typ is float:
                    val = col.slider(label, vmin, vmax, float(default_val), step, key=key)
                else:
                    val = col.slider(label, vmin, vmax, int(default_val), step, key=key)
                _opt_params_updated[name] = val

            # Változások visszaírása a PT modulba és PT.params-ba
            _changed = False
            for k, v in _opt_params_updated.items():
                try:
                    if getattr(PT, k, None) != v:
                        setattr(PT, k, v); _changed = True
                except Exception:
                    pass
                try:
                    if not hasattr(PT, "params") or not isinstance(PT.params, dict):
                        PT.params = {}
                    if PT.params.get(k.lower()) != v:
                        PT.params[k.lower()] = v; _changed = True
                except Exception:
                    pass

            # LK paraméterek frissítése, ha kell
            if _changed and hasattr(PT, "WIN") and hasattr(PT, "LEVELS"):
                try:
                    PT.lk_params = dict(
                        winSize=(int(PT.WIN), int(PT.WIN)),
                        maxLevel=int(PT.LEVELS),
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1e-10),
                    )
                except Exception:
                    pass

        # Ergonomy-szerű kétoszlopos elrendezés (bal: nagy vászon, jobb: infó)
        col_main, col_side = st.columns([3, 1])
        img_slot = col_main.empty()
        stats_box = col_side.container()
        stats_placeholder = stats_box.empty()

        model = get_tracker_model()

        # Session state init a trackerhez
        if "opt_state_init" not in st.session_state:
            st.session_state["opt_state_init"] = True
            st.session_state["opt_trackers"] = []
            st.session_state["opt_prev_gray"] = None
            st.session_state["opt_frame_count"] = 0
            st.session_state["opt_video_name"] = None
            if hasattr(PT, "REINIT_ACTIVE"): PT.REINIT_ACTIVE.clear()
            if hasattr(PT, "ID_COLORS"):      PT.ID_COLORS.clear()
            if hasattr(PT, "NEXT_ID"):        PT.NEXT_ID = 0
            if hasattr(PT, "id_frame_count"): PT.id_frame_count = 0
            if hasattr(PT, "track_buffer"):   PT.track_buffer.clear()
            if hasattr(PT, "current_filename"): PT.current_filename = None
            if hasattr(PT, "dashboard_data"):
                PT.dashboard_data["positions"].clear()
                PT.dashboard_data["points"].clear()
                PT.dashboard_data["reinits"].clear()
                PT.dashboard_data["frames"].clear()

        # Forrás megnyitása
        cap = None
        if run:
            if st.session_state["opt_src_mode"] == "Webkamera":
                cap = cv2.VideoCapture(0)
                st.session_state["opt_video_name"] = "camera0"
            else:
                if vf is not None:
                    path = os.path.join("/mnt/data", vf.name)
                    with open(path, "wb") as f: f.write(vf.read())
                    cap = cv2.VideoCapture(path)
                    st.session_state["opt_video_name"] = os.path.splitext(os.path.basename(path))[0]

        # Reset / toggles
        if btn_reset:
            if getattr(PT, "SAVE_DATA", False) and hasattr(PT, "save_tracks_to_excel_async"):
                PT.save_tracks_to_excel_async()
                st.session_state['opt_last_autosave'] = int(getattr(PT, 'id_frame_count', st.session_state.get('opt_frame_count', 0)))
            if hasattr(PT, "track_buffer"): PT.track_buffer.clear()
            st.session_state["opt_trackers"].clear()
            if hasattr(PT, "REINIT_ACTIVE"): PT.REINIT_ACTIVE.clear()
            if hasattr(PT, "ID_COLORS"):     PT.ID_COLORS.clear()
            if hasattr(PT, "NEXT_ID"):       PT.NEXT_ID = 0
            if hasattr(PT, "id_frame_count"): PT.id_frame_count = 0
            if hasattr(PT, "current_filename"):
                PT.current_filename = PT.get_filename() if hasattr(PT, "get_filename") else None
            if hasattr(PT, "dashboard_data"):
                PT.dashboard_data["positions"].clear()
                PT.dashboard_data["points"].clear()
                PT.dashboard_data["reinits"].clear()
                PT.dashboard_data["frames"].clear()
            st.session_state["opt_prev_gray"] = None
            st.session_state["opt_frame_count"] = 0
            st.success("RESET megtörtént.")

        if btn_toggle_save and hasattr(PT, "SAVE_DATA"):
            PT.SAVE_DATA = not PT.SAVE_DATA
            if hasattr(PT, "save_state"):
                PT.save_state = "Deactivate saving data" if PT.SAVE_DATA else "Activate saving data"
            st.info(f"SAVE_DATA = {PT.SAVE_DATA}")

        if btn_toggle_dash and hasattr(PT, "DASHBOARD_ACTIVE"):
            PT.DASHBOARD_ACTIVE = not PT.DASHBOARD_ACTIVE
            if hasattr(PT, "dashboard_state"):
                PT.dashboard_state = "Deactivate dashboard" if PT.DASHBOARD_ACTIVE else "Activate dashboard"
            if (not PT.DASHBOARD_ACTIVE) and hasattr(PT, "dashboard_data"):
                PT.dashboard_data["positions"].clear()
                PT.dashboard_data["points"].clear()
                PT.dashboard_data["reinits"].clear()
                PT.dashboard_data["frames"].clear()
            st.info(f"DASHBOARD_ACTIVE = {PT.DASHBOARD_ACTIVE}")

        # Fő ciklus
        if run and cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                st.error("Forrás nem olvasható.")
                cap.release()
            else:
                h, w = frame.shape[:2]
                st.session_state["opt_prev_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                st.session_state["opt_frame_count"] = 0
                if hasattr(PT, "id_frame_count"): PT.id_frame_count = 0
                video_name = st.session_state["opt_video_name"] or "video"
                t0 = time.perf_counter()

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.info("A videó véget ért.")
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Log buffer
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for tr in st.session_state["opt_trackers"]:
                        if hasattr(PT, "track_buffer"):
                            PT.track_buffer.append([getattr(PT, "id_frame_count", 0), timestamp, tr["id"], tr["cx"], tr["cy"]])

                    if getattr(PT, "SAVE_DATA", False) and hasattr(PT, "N_SAVE_FRAME") and hasattr(PT, "save_tracks_to_excel_async"):
                        if getattr(PT, "id_frame_count", 0) % max(1, int(PT.N_SAVE_FRAME)) == 0 and getattr(PT, "id_frame_count", 0) > 0:
                            PT.save_tracks_to_excel_async()
                            # -> utolsó autosave frame frissítése VALÓS mentésnél
                            st.session_state['opt_last_autosave'] = int(getattr(PT, 'id_frame_count', st.session_state['opt_frame_count']))

                    if getattr(PT, "DASHBOARD_ACTIVE", True) and hasattr(PT, "dashboard_data"):
                        PT.dashboard_data["frames"].append(getattr(PT, "id_frame_count", 0))
                        for tr in st.session_state["opt_trackers"]:
                            tid = tr["id"]
                            PT.dashboard_data["positions"][tid]["x"].append(tr["cx"])
                            PT.dashboard_data["positions"][tid]["y"].append(tr["cy"])
                            PT.dashboard_data["points"][tid].append(len(tr["p0"]))
                            if hasattr(PT, "REINIT_ACTIVE") and tid in PT.REINIT_ACTIVE:
                                PT.dashboard_data["reinits"][tid].append(getattr(PT, "id_frame_count", 0))

                    # Initial detection
                    if len(st.session_state["opt_trackers"]) == 0:
                        detections = _opt_run_yolo_and_build_detections(model, frame, w, h)
                        st.session_state["opt_trackers"] = _opt_assign_only_list(PT.assign_detections_to_trackers(
                            detections, st.session_state["opt_trackers"], frame, video_name, st.session_state["opt_frame_count"], PT.ASSIGN_DIST
                        ))

                    # LK update + reinit
                    updated = []
                    need_reinit_ids = set()

                    for tr in st.session_state["opt_trackers"]:
                        cx, cy, p0 = tr["cx"], tr["cy"], tr["p0"]
                        p1g, p0g = PT.forward_backward_filter(st.session_state["opt_prev_gray"], gray, p0)

                        if (p1g is not None) and (len(p1g) > 0):
                            pts = p1g.reshape(-1, 2)
                            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                            k = max(1, int(len(dists) * 0.1))
                            worst_mean = float(np.mean(np.sort(dists)[-k:]))

                            init_pts = int(tr.get("init_pts", len(p1g)))
                            init_pts = max(1, init_pts)
                            too_few = (len(p1g) < PT.POINT_DROP_RATIO * init_pts)

                            if too_few or (worst_mean > PT.MEAN_DRIFT_THRESH):
                                need_reinit_ids.add(tr["id"])
                                current_frame = st.session_state["opt_frame_count"]
                                if worst_mean > PT.MEAN_DRIFT_THRESH:
                                    reason = f"mean drift too large ={worst_mean:.1f}px)"
                                else:
                                    reason = f"too few points: ({len(p1g)}/{init_pts})"
                                if hasattr(PT, "REINIT_ACTIVE"):
                                    if tr["id"] not in PT.REINIT_ACTIVE:
                                        PT.REINIT_ACTIVE[tr["id"]] = current_frame + PT.REINIT_WINDOW
                                        print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit START: {reason}")
                                    else:
                                        if current_frame <= PT.REINIT_ACTIVE[tr["id"]]:
                                            print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} retrying reinit ({reason})")
                                        else:
                                            print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit FAILED (timeout)")
                                            del PT.REINIT_ACTIVE[tr["id"]]
                            else:
                                disp = (p1g - p0g).reshape(-1, 2)
                                dx, dy = float(np.mean(disp[:, 0])), float(np.mean(disp[:, 1]))
                                cx, cy = PT.clamp_point(cx + dx, cy + dy, w, h)
                                for idx, pt in enumerate(p1g.reshape(-1, 2)):
                                    if idx < len(tr["trails"]):
                                        tr["trails"][idx].append((int(pt[0]), int(pt[1])))
                                tr["cx"], tr["cy"], tr["p0"] = cx, cy, p1g.copy()
                                tr["hist"].append((cx, cy))
                                if hasattr(PT, "REINIT_ACTIVE") and tr["id"] in PT.REINIT_ACTIVE:
                                    del PT.REINIT_ACTIVE[tr["id"]]
                                    print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit SUCCESS")
                                updated.append(tr)
                        else:
                            need_reinit_ids.add(tr["id"])
                            current_frame = st.session_state["opt_frame_count"]
                            if hasattr(PT, "REINIT_ACTIVE"):
                                if tr["id"] not in PT.REINIT_ACTIVE:
                                    PT.REINIT_ACTIVE[tr["id"]] = current_frame + PT.REINIT_WINDOW
                                    print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit START: optical flow lost")
                                else:
                                    if current_frame <= PT.REINIT_ACTIVE[tr["id"]]:
                                        print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit RETRY (optical flow lost)")
                                    else:
                                        print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit FAILED (timeout)")
                                        del PT.REINIT_ACTIVE[tr["id"]]

                    if need_reinit_ids and any(st.session_state["opt_frame_count"] <= PT.REINIT_ACTIVE.get(tid, -1) for tid in need_reinit_ids):
                        dets = _opt_run_yolo_and_build_detections(model, frame, w, h)
                        new_list = _opt_assign_only_list(PT.assign_detections_to_trackers(
                            dets, st.session_state["opt_trackers"], frame, video_name, st.session_state["opt_frame_count"], PT.ASSIGN_DIST
                        ))
                        for tr in new_list:
                            if tr["id"] in need_reinit_ids and tr.get("p0") is not None:
                                init_pts2 = int(tr.get("init_pts", len(tr["p0"])))
                                init_pts2 = max(1, init_pts2)
                                if len(tr["p0"]) >= PT.POINT_DROP_RATIO * init_pts2:
                                    if tr["id"] in PT.REINIT_ACTIVE:
                                        del PT.REINIT_ACTIVE[tr["id"]]
                                        print(f"[Frame {st.session_state['opt_frame_count']}] Tracker {tr['id']} reinit SUCCESS")
                        st.session_state["opt_trackers"] = new_list
                    else:
                        st.session_state["opt_trackers"] = updated if len(updated) > 0 else st.session_state["opt_trackers"]

                    if btn_manual_init:
                        print(f"[Frame {st.session_state['opt_frame_count']}] Manual reinitialization")
                        detections = _opt_run_yolo_and_build_detections(model, frame, w, h)
                        try:
                            st.session_state["opt_trackers"] = _opt_assign_only_list(PT.assign_detections_to_trackers(
                                detections, st.session_state["opt_trackers"], frame, video_name, st.session_state["opt_frame_count"], PT.ASSIGN_DIST, manual=True
                            ))
                        except TypeError:
                            st.session_state["opt_trackers"] = _opt_assign_only_list(PT.assign_detections_to_trackers(
                                detections, st.session_state["opt_trackers"], frame, video_name, st.session_state["opt_frame_count"], PT.ASSIGN_DIST
                            ))

                    # Rajzolás + dashboard merge
                    draw = frame.copy()
                    for tr in st.session_state["opt_trackers"]:
                        color = PT.get_id_color(tr["id"])
                        cv2.circle(draw, (int(tr["cx"]), int(tr["cy"])), 6, color, -1)
                        cv2.putText(draw, f"ID {tr['id']}", (int(tr["cx"]) + 10, int(tr["cy"]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        for trail in tr["trails"]:
                            if len(trail) > 1:
                                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(draw, [pts], False, color, 1)
                        if tr["p0"] is not None:
                            for (x, y) in tr["p0"].reshape(-1, 2):
                                cv2.circle(draw, (int(x), int(y)), 3, color, 1)

                    st.session_state["opt_frame_count"] += 1
                    if hasattr(PT, "id_frame_count"): PT.id_frame_count += 1
                    fps = st.session_state["opt_frame_count"] / (time.perf_counter() - t0 + 1e-6)

                    if getattr(PT, "DASHBOARD_ACTIVE", True):
                        dashboard = PT.draw_dashboard_canvas(600, 600, history=getattr(PT, "VISFRAME", 200), trackers=st.session_state["opt_trackers"])
                        merged = PT.merge_frame_and_dashboard(draw, dashboard)
                        cv2.putText(merged, f"Frame: {getattr(PT, 'id_frame_count', st.session_state['opt_frame_count'])} | FPS {fps:.2f} | Trackers={len(st.session_state['opt_trackers'])}", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, getattr(PT, "TEXT_COLOR", (255, 255, 255)), 2)
                        labels = ["Frame", "Tracked IDs", "Tracked points"]

                        # A feliratokat a MERGED kép koordinátarendszeréhez igazítjuk
                        mH, mW = merged.shape[:2]
                        col_width = mW // 3

                        for i, text in enumerate(labels):
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            x = i * col_width + (col_width - text_w) // 2
                            y = 95
                            cv2.putText(
                                merged, text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                getattr(PT, "TEXT_COLOR", (255, 255, 255)), 2, cv2.LINE_AA
                            )
                        img_slot.image(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        cv2.putText(draw, f"Frame: {getattr(PT, 'id_frame_count', st.session_state['opt_frame_count'])} | FPS {fps:.2f} | Trackers={len(st.session_state['opt_trackers'])}", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, getattr(PT, "TEXT_COLOR", (255, 255, 255)), 2)
                        img_slot.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), use_container_width=True)

                    st.session_state["opt_prev_gray"] = gray
                    time.sleep(0.001)

                cap.release()
    else:
        st.markdown(f"### {st.session_state.selected_menu}")
        st.info("This section is under construction.")
