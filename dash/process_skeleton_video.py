# Import libraries
import cv2 as cv
import mediapipe as mp
import os
import numpy as np
import streamlit as st
import convert_to_excel as cte
import pandas as pd
from datetime import datetime
import create_pdf_report as topdf
from calculate_video import VideoElements as vid
from create_visuals import VisualElements as visual
from convert_to_piechart import PieChart as pie
from convert_to_linechart import LineChartElements as line
from get_score_RULA import GetRULAScores as rula
from create_risk_map import RiskMapElements as riskmap
from sqlalchemy.orm import Session

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
PIE_CHART_NAME = "pie_chart.png"
RISK_MAP_NAME = "risk_map.png"
NECK_NAME = "max_neck.png"
TRUNK_NAME = "max_trunk.png"
LEG_NAME = "max_legs.png"
U_ARM_NAME = "max_up_arm.png"
L_ARM_NAME = "max_lo_arm.png"
WRIST_NAME = "max_wrist.png"
CRITICAL_LEFT_NAME = "max_left.png"
CRITICAL_RIGHT_NAME = "max_right.png"


def process_video(filename : str, assess_method : str, start_frame: int, end_frame: int, processing_rate: int, additional_data: dict,
                  assessor : str , task : str, workstation : str):
    # processing_rate = 1
    print(f"Processing video: {filename}")
    if os.path.isabs(filename) or os.path.exists(filename):
        input_video_file_path = filename
    else:
        input_video_file_path = os.path.join(EXPORTS_DIR, filename)
    print("Take input video from:", input_video_file_path)
    print('Processing rate:', processing_rate)
    vis = visual()
    calcvid = vid()
    piechart = pie(assess_method)
    analysis_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # processing_rate = 10  # the rate to process the video, original = 30 fps
    input_param = [
         {'shoulder_raised_l': additional_data['upper_arms_raised_l'], 'arm_abducted_l': additional_data['upper_arms_abducted_l'], 'supported_leaning_l': additional_data['upper_arms_supported_l'],
          'shoulder_raised_r': additional_data['upper_arms_raised_r'], 'arm_abducted_r': additional_data['upper_arms_abducted_r'], 'supported_leaning_r': additional_data['upper_arms_supported_r'],
          "arm_mid_l": additional_data['lower_arms_mid_l'], "arm_mid_r": additional_data['lower_arms_mid_r'],
          "wrist_bent_l": additional_data['wrists_bent_l'], "wrist_twisted_mid_l" : additional_data['wrist_twist_endline_l'],
          "wrist_bent_r": additional_data['wrists_bent_r'], "wrist_twisted_mid_r" : additional_data['wrist_twist_endline_r'],
          "muscle_arms_l": additional_data['muscle_arms_l'], "muscle_arms_r": additional_data['muscle_arms_r'],
          "force_load_l": additional_data['load_force_arms_l'], "force_load_r": additional_data['load_force_arms_r'], 
          "load_force_upper_body" : additional_data['load_force_upper_body'],
          # neck, trunk and leg
          "neck_twisted": additional_data['neck_twisted'], "neck_side": additional_data['neck_bent'],
          "trunk_twisted": additional_data['trunk_twisted'], "trunk_side": additional_data['trunk_bent'],
          "muscle_leg": additional_data['muscle_legs'], "force_load_leg": additional_data['load_force_legs'],
          "load_shock" : additional_data['load_shock'],
          # coupling score
          "coupling_l": additional_data['coupling_score_l'], "coupling_r": additional_data['coupling_score_r'],
          # activity score
          "act_score": additional_data['activity_score']
          }
         ]

    assess_type = ['RULA', 'REBA']
    if assess_method not in assess_type:
        raise ValueError("Invalid type. Expected one of: %s" % assess_type)

    print("The assessment method is:", assess_method)

    filename_short = os.path.splitext(os.path.basename(filename))[0]
    # set up the output folder
    output_name = f'output_{filename_short}_{assess_method}.mp4'
    output_path = os.path.join(EXPORTS_DIR, output_name)
    print("The output video will be in:", output_path)
    score_filename = f'scores_{filename_short}_{assess_method}_{analysis_time}.xlsx'
    score_filepath = os.path.join(EXPORTS_DIR, score_filename)
    # reba_parent_dir = os.path.dirname(score_filepath)
    # if not os.path.exists(reba_parent_dir):
    #     os.makedirs(reba_parent_dir)

    # initialize Pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_drawing_styles = mp.solutions.drawing_styles

    # % Start video streaming
    try:
        cam = cv.VideoCapture(input_video_file_path)
        print(f"Try to open video in {input_video_file_path}")
        if not cam.isOpened():
            raise Exception("Cant open input video. Nem sikerült megnyitni a videót!")
    except Exception as ex:
        print("Input video invalid!", ex)
    # Video setting
    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = height / width
    # Reduce image size (downscale for speed)
    scaled_h = 960   # downscaled_width, downscaled_height = 540, 960
    scaled_w = int(scaled_h / aspect_ratio)
    score_data = []  # df to store the ergonomic scores
    frame_number = 0  # Frame counter for time tracking
    orig_frame_number = start_frame
    fps = cam.get(cv.CAP_PROP_FPS)
    # create visualization
    blank_canvas = visual.create_dashboard(vis, assess_method)
    vid_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'MP4V'),
                                30, (int(scaled_w + 600), scaled_h))

    while True:
        _ret, frame = cam.read()
        if not _ret:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (scaled_w, scaled_h))  # Resize
        frame_number += 1    

        # Calculate timestamp for the current frame
        # timestamp_seconds = frame_number / fps  # Get timestamp in seconds
        elapsed_seconds = orig_frame_number / fps

        # Convert seconds to hh:mm:ss format
        timestamp = f"{int(elapsed_seconds // 3600):02}:{int((elapsed_seconds % 3600) // 60):02}:{int(elapsed_seconds % 60):02}"

        # start processing each frame
        # if frame_number % processing_rate == 0:  # processing rate

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB
        image.flags.writeable = False
        results = pose.process(image)  # get the skeleton
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Convert back to BGR

        if results.pose_landmarks:  # Check if pose landmarks are detected.
            landmarks = results.pose_landmarks.landmark  # Extract landmarks
            input_dict = vid.create_input_dict(calcvid, landmarks, mp_pose)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5),
                )

            # face mask
            vid.face_mask(calcvid, image, input_dict['ear_c'], input_dict['nose'])

            # input parameters from user
            input_var = input_param[0]   # take the set of input in correspondence with frame i-th
            # Start the assessment
            if assess_method == "REBA":
                score_dict, action_required, msd_risk_level = vid.calculate_REBA(calcvid, input_dict, input_var)
                # converting to excel
                ergonomic_scores = cte.ExcelElements.getNewElementREBA(frame_number, orig_frame_number, timestamp,
                    input_dict, input_var, score_dict)
                score_data.append(ergonomic_scores)

            if assess_method == "RULA":
                score_dict, action_required, msd_risk_level = vid.calculate_RULA(calcvid, input_dict, input_var)
                # converting to excel
                ergonomic_scores = cte.ExcelElements.getNewElementRULA(frame_number, orig_frame_number, timestamp,
                    input_dict, input_var, score_dict)
                score_data.append(ergonomic_scores)

            if score_dict:
                # create visualization
                updated_canvas = visual.update_canvas(
                    vis, assess_method, blank_canvas,
                    input_dict, input_var, score_dict)

                # create a black image with only the body of the operator
                xs = [lm.x for lm in landmarks]
                x_min, x_max = min(xs), max(xs)
                center_x = (x_min + x_max) / 2
                box_width = x_max - x_min

                scale = 1.8  # scale factor to determine width around body
                scaled_width = box_width * scale

                new_x_min = max(center_x - scaled_width / 2, 0)
                new_x_max = min(center_x + scaled_width / 2, 1)

                x1 = int(new_x_min * scaled_w)
                x2 = int(new_x_max * scaled_w)

                # Make sure x1 and x2 are even numbers (for codec compatibility)
                # x1 = (x1 + 1) // 2 * 2
                # x2 = (x2 + 1) // 2 * 2

                # Crop the body region
                # body_region = image[:, x1:x2]

                # Create black canvas same size as original image
                black_canvas = np.zeros_like(image)
                black_canvas[:, x1:x2] = image[:, x1:x2]

                # Calculate horizontal offset to center the cropped body region
                # body_width = x2 - x1
                # offset_x = (scaled_w - body_width) // 2

                # # Paste the body region onto the black canvas centered horizontally
                # black_canvas[:, offset_x:offset_x + body_width] = body_region

                merged_canvas = cv.hconcat([black_canvas, updated_canvas])
                # merged_canvas = cv.hconcat([image, updated_canvas])
                cv.imshow('Assessment result', merged_canvas)

                # write the output frame
                vid_writer.write(merged_canvas)

        orig_frame_number += processing_rate

        #ch = cv.waitKey(1)
        #if ch == 27:
        #    print("Quitting the program!")
        #    break
    # cleanup the camera and close any open windows
    cam.release()
    vid_writer.release()
    cv.destroyAllWindows()

    # Save score data to excel file
    df = pd.DataFrame(score_data)
    df.to_excel(score_filepath, index=False)
    print("Excel file of scores saved in ", score_filepath)

    # get the index of critical frames
    crit_idx = vid.get_critical_index(calcvid, df, assess_method)
    # extract the images
    vid.get_critical_img(calcvid, input_video_file_path, crit_idx , analysis_time)

    # create risk map
    risk_map = riskmap(score_dict, assess_method)

    # visualization with risk map and pie chart
    piechart_path = piechart.create_piechart(df, analysis_time)
    riskmap_path = risk_map.createMap(analysis_time)

    total_frames, fps, duration_seconds, duration_str = vid.get_video_duration(calcvid, input_video_file_path)
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration (seconds): {duration_seconds:.2f}")
    print(f"Duration (formatted): {duration_str}")

    print("Critical_idx:", crit_idx)

    # Example Usage
    data = {
        "date": f'{datetime.now().date()}' + ' ' + f'{datetime.now().time().hour}' + ':' + f'{datetime.now().time().minute}' + ' : ' + f'{datetime.now().time().second}',
        "assessor": assessor,
        "assessment": assess_method,
        "workstation": workstation,
        "task": task,
        "video_length": str(total_frames),
        "working_duration": duration_str,
        "highest_score": 'Left: ' + str(df['C_score_l'].max()) + ' , ' + 'Right: ' + str(df['C_score_r'].max()),
        "msd_risk_level": msd_risk_level,
        "action_required": action_required
    }

    linechart = line(assess_method, analysis_time)
    linechart.categorize_color()
    linechart.visualization_chart(df, side="left")
    linechart.visualization_chart(df, side="right")


    elements = topdf.PdfElements(
        filename=f'{filename}_ergonomic_report-{analysis_time}.pdf',
        documentTitle="Ergonomic Report",
        opdata=data,
        piechart= piechart_path,
        riskmap= riskmap_path,
        maximages={
            "neck": os.path.join(EXPORTS_DIR, f"max_neck-{analysis_time}.png"),
            "trunk": os.path.join(EXPORTS_DIR, f"max_trunk-{analysis_time}.png") ,
            "leg": os.path.join(EXPORTS_DIR, f"max_legs-{analysis_time}.png"),
            "u_arm": os.path.join(EXPORTS_DIR, f"max_up_arm-{analysis_time}.png"),
            "l_arm": os.path.join(EXPORTS_DIR, f"max_lo_arm-{analysis_time}.png"),
            "wrist": os.path.join(EXPORTS_DIR, f"max_wrist-{analysis_time}.png")
        },
        criticalidx=crit_idx,
        criticalimages={
            "criticalleft" : os.path.join(EXPORTS_DIR, f"max_left-{analysis_time}.png"),
            "criticalright" : os.path.join(EXPORTS_DIR , f"max_right-{analysis_time}.png")
        }
    )

    print(piechart)
    elements.createPdfFile()
    print("DEBUG - SCORE_DICT:")
    print(score_dict)
    
    st.session_state.analysis_time = analysis_time


