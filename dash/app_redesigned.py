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
from process_skeleton_video import process_video
from hand_heatmap_from_video import hand_heatmap_two_color
from streamlit_javascript import st_javascript
import streamlit.components.v1 as components

# Directories
IMPORTS_DIR = "imports"
EXPORTS_DIR = "exports"

# Create directories if not exist
for directory in [IMPORTS_DIR, EXPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set wide layout
st.set_page_config(layout="wide")

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


# UI Layout
col1, col2 = st.columns([1, 5])

with col1:
    with st.container():
        st.image("assets/PE_logo_blue.png", width=120)
        st.markdown("###")
        nav_button("Ergonomy Assessment")
        nav_button("Optical Flow Analysis")
        nav_button("Activity Assessment")
        nav_button("Work Instructions")
        nav_button("Chatbot")
        nav_button("Result Files")
        st.markdown("###")
        st.image("assets/MK_logo_blue.png", width=120)
        st.markdown("###")
        st.slider("Result width", min_value=300, max_value=1200, value=st.session_state.result_width, key="result_width")

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
                        index= 1,
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

    else:
        st.markdown(f"### {st.session_state.selected_menu}")
        st.info("This section is under construction.")