import time
import streamlit as st
import os
import cv2
import subprocess
from process_skeleton_video import process_video
from hand_heatmap_from_video import hand_heatmap_two_color

# Az imports és exports mappák elérési útjai
IMPORTS_DIR = "imports"
EXPORTS_DIR = "exports"

# Ellenőrizzük, hogy az imports és exports mappák léteznek-e, ha nem, létrehozzuk
for directory in [IMPORTS_DIR, EXPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Streamlit alkalmazás címe
st.title("Videó Feltöltés és Feldolgozás")

# Oldalsáv (side menu)
st.sidebar.title("Navigáció")
selected_action = st.sidebar.radio("Válassz egy műveletet:", ["Videó feltöltése", "Feldolgozás indítása"])

# Inicializáljuk a session state változókat
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Függvények
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
        print(f"Video successfully converted to H.264: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during H.264 conversion: {e}")
        raise Exception(f"FFmpeg conversion failed: {e}")

def process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, assess_method):
    try:
        # Biztosítsuk, hogy a start_frame és end_frame egész szám legyen
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        temp_video_path = os.path.join(EXPORTS_DIR, f"temp_{os.path.basename(file_path)}")
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        if not cap.isOpened():
            raise Exception("Error: Could not open input video file.")
        if start_frame >= end_frame or start_frame < 0:
            raise Exception("Invalid frame range specified.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = end_frame - start_frame

        # Feldolgozás előrehaladásának frissítése
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_num}.")
                break
            temp_out.write(frame)
            progress_bar.progress((frame_num - start_frame + 1) / (2 * total_frames))  # 50%-ig frissítjük

        cap.release()
        temp_out.release()

        print(f"Temporary video created at: {temp_video_path}")

        # Másik process_video függvény meghívása
        if assess_method == "hand_heatmap":
            output_video_file_path, output_heatmap_path = hand_heatmap_two_color(
                filename=os.path.basename(temp_video_path)
            )
        else:
            process_video(
                filename=os.path.basename(temp_video_path),
                assess_method=assess_method,
                start_frame=start_frame,
                end_frame=end_frame,
                processing_rate=1  # Példa: feldolgozási sebesség
            )

        # Progress bar frissítése a másik feldolgozás során
        progress_bar.progress(0.75)  # 75%-ra állítjuk

        # A process_video által generált fájl keresése
        processed_video_name = f"output_{os.path.basename(temp_video_path).split('.')[0]}_{assess_method}.mp4"
        processed_video_path = os.path.join(EXPORTS_DIR, processed_video_name)

        if output_video_file_path is not None:
            processed_video_path = os.path.join(EXPORTS_DIR, output_video_file_path)

        if not os.path.exists(processed_video_path):
            raise Exception(f"Processed video not found: {processed_video_path}")

        # Konvertálás H.264 formátumba
        final_output_path = os.path.join(EXPORTS_DIR, f"converted_{processed_video_name}")

        convert_to_h264(processed_video_path, final_output_path)

        # Az eredeti feldolgozott fájl törlése
        #os.remove(processed_video_path)

        # Progress bar teljesre állítása
        progress_bar.progress(1.0)

        st.success(f"Feldolgozás befejezve! A feldolgozott videó elérhető itt: {final_output_path}")
        st.video(final_output_path)

    except Exception as e:
        st.error(f"Hiba történt a feldolgozás során: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.session_state.is_processing = False  # Feldolgozás vége
        #st.rerun()

def start_processing(file_path, output_path, start_frame, end_frame, fps):
    """Külön szálon futtatja a videó feldolgozást."""
    progress_bar = st.progress(0)
    try:
        process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar)
    finally:
        print('Processing completed.')
        st.session_state.is_processing = False  # Feldolgozás vége

# Videó feltöltése
if selected_action == "Videó feltöltése":
    uploaded_file = st.file_uploader("Tölts fel egy videót", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        file_path = os.path.join(IMPORTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"A videó sikeresen feltöltve: {uploaded_file.name}")
        st.video(file_path)

# Feldolgozás indítása
if selected_action == "Feldolgozás indítása":
    video_files = os.listdir(IMPORTS_DIR)
    if not video_files:
        st.warning("Nincs feltöltött videó az 'imports' mappában.")
    else:
        selected_video = st.selectbox(
            "Válassz egy videót feldolgozáshoz:",
            video_files,
            disabled=st.session_state.is_processing
        )

        if selected_video:
            file_path = os.path.join(IMPORTS_DIR, selected_video)

            if not st.session_state.is_processing:
                # Videó megjelenítése
                st.video(file_path)

                # Videó információk kiolvasása
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()

                st.write(f"Videó hossza: {duration:.2f} másodperc")
                start_time, end_time = st.slider(
                    "Válaszd ki a feldolgozandó időtartományt (másodpercben):",
                    min_value=0.0,
                    max_value=float(duration),
                    value=(0.0, float(duration)),
                    step=0.1
                )
            
            view_type = st.selectbox(
            "Choose a view type",
            ['Top view', 'Side view', 'Front view'],
            disabled=st.session_state.is_processing
            )

            if view_type == 'Top view':

                if st.button('Hand heatmap', disabled=st.session_state.is_processing):
                    st.session_state.is_processing = True

                    # Másodpercek konvertálása frame-ekre
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)

                    # Output path beállítása
                    output_path = os.path.join(EXPORTS_DIR, f"output_temp_{selected_video.split('.')[0]}_hand_heatmap.mp4")

                    # Progress bar létrehozása
                    progress_bar = st.progress(0)
                    st.session_state.is_processing = True

                    # Feldolgozás indítása
                    process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, assess_method="hand_heatmap")

                    # Feldolgozás vége

            if view_type == 'Side view' or view_type == 'Front view':


                if st.button("REBA calculation", disabled=st.session_state.is_processing):
                    st.session_state.is_processing = True

                    # Másodpercek konvertálása frame-ekre
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)

                    # Output path beállítása
                    output_path = os.path.join(EXPORTS_DIR, f"output_temp_{selected_video.split('.')[0]}_REBA.mp4")

                    # Progress bar létrehozása
                    progress_bar = st.progress(0)
                    st.session_state.is_processing = True

                    # Feldolgozás indítása
                    process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, assess_method="REBA")

                    # Feldolgozás vége

                if st.button("RULA calculation", disabled=st.session_state.is_processing):
                    st.session_state.is_processing = True

                    # Másodpercek konvertálása frame-ekre
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)

                    # Output path beállítása
                    output_path = os.path.join(EXPORTS_DIR, f"output_temp_{selected_video.split('.')[0]}_REBA.mp4")

                    # Progress bar létrehozása
                    progress_bar = st.progress(0)
                    st.session_state.is_processing = True
                    

                    # Feldolgozás indítása
                    process_streamlit_video(file_path, output_path, start_frame, end_frame, fps, progress_bar, assess_method="RULA")

                    # Feldolgozás vége
