# Import libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")


# %%
def hand_heatmap_two_color(filename,status_placeholder, movement_min=3, movement_max=10):
    print('Hand heatmap two color')
    # %%###############################################################################
    # Adjustable parameters
    buffer_time = 20  # Duration in seconds to keep data
    # movement_min, movement_max = 3, 10  # Threshold for detecting movements in pixels
    point_radius = 7  # Radius of points to draw in the heatmap
    # intensity_decay_factor = 0.9  # decay factor for the intensity of the heatmap over time

    # %%###############################################################################
    # initialize mediapipe pose estimator
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Fixed parameters
    fps = 30  # Frames per second of the video feed
    frame_buffer_size = fps * buffer_time
    # Buffers for left and right hands
    left_hand_buffer = deque(maxlen=frame_buffer_size)
    right_hand_buffer = deque(maxlen=frame_buffer_size)
    previous_landmarks = {"Left": None, "Right": None}  # Store previous landmarks separately for left and right hands
    finger_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]  # Indices for the first three landmarks of each finger
    input_video_file_path = os.path.join(EXPORTS_DIR, filename)
    print(f'Input video file path: {input_video_file_path}')
    filename_without_extension = filename.split('.')[0]
    output_video_file_name = f'output_{filename_without_extension}.mp4'
    output_video_file_path = os.path.join(EXPORTS_DIR, output_video_file_name)
    output_heatmap_name = f'overall_heatmap_{filename_without_extension}.jpg'
    output_heatmap_path = os.path.join(EXPORTS_DIR, output_heatmap_name)

    # %% Start video streaming
    # input_video_file_path = "C:/Users/anh.tt/Documents/operation50_dash/operator-50-dashboard/dash/imports/GX010068.MP4"
    try:
        cam = cv2.VideoCapture(input_video_file_path)
    except Exception as e:
        print(f"Error opening video file: {e}")

    if not cam.isOpened():
        raise Exception(f"Nem sikerült megnyitni a videót: {input_video_file_path}")
    
    # Get total frame count
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ################################################################################
    # Video setting
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = height / width
    # Reduce image size (downscale for speed)
    downscaled_height = int(height / 2)  # 540
    downscaled_width = int(downscaled_height / aspect_ratio)
    # Initialize map to track how many times each point has been visited
    overall_heatmap_left = np.zeros((downscaled_height, downscaled_width), dtype=np.float32)
    overall_heatmap_right = np.zeros((downscaled_height, downscaled_width), dtype=np.float32)

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_video_file_path, fourcc, fps, (width, height))

    frame_count = 0
    print(">>> cam open ok:", cam.isOpened())
    while True:
        _ret, orig_frame = cam.read()
        if not _ret:
            print('No frames grabbed or end of videos!')
            break
        frame = orig_frame.copy()
        print(f'Frame: {frame_count}')
        status_placeholder.text(f"Processing frame {frame_count + 1} / {total_frames} ...")
        frame_count += 1
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (downscaled_width, downscaled_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ################################################################################
        # Hand recognition and heatmap
        ################################################################################
        # Process the frame for hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Determine if the hand is left or right
                label = handedness.classification[0].label  # 'Left' or 'Right'

                # Collect hand landmark positions
                frame_landmarks = []
                for i in finger_indices:
                    landmark = hand_landmarks.landmark[i]
                    x, y = int(landmark.x * downscaled_width), int(landmark.y * downscaled_height)
                    x = min(max(x, 0), downscaled_width - 1)
                    y = min(max(y, 0), downscaled_height - 1)
                    frame_landmarks.append((x, y))

                # Check movement and update buffers
                if previous_landmarks[label]:
                    for (x, y), (prev_x, prev_y) in zip(frame_landmarks, previous_landmarks[label]):
                        # Calculate movement distance
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        if movement_max > distance > movement_min:
                            intensity = min(int(distance * 2), 255)  # Scale intensity, capped at 255
                            if label == "Left":
                                left_hand_buffer.append((x, y, intensity))
                                for dx in range(-3, 4):
                                    for dy in range(-3, 4):
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < downscaled_width and 0 <= ny < downscaled_height:
                                            distance_from_center = np.sqrt(dx**2 + dy**2)
                                            if distance_from_center <= 3:
                                                overall_heatmap_left[ny, nx] += intensity / (distance_from_center + 1)
                            elif label == "Right":
                                right_hand_buffer.append((x, y, intensity))
                                for dx in range(-3, 4):
                                    for dy in range(-3, 4):
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < downscaled_width and 0 <= ny < downscaled_height:
                                            distance_from_center = np.sqrt(dx**2 + dy**2)
                                            if distance_from_center <= 3:
                                                overall_heatmap_right[ny, nx] += intensity / (distance_from_center + 1)

                previous_landmarks[label] = frame_landmarks  # Update previous landmarks

        # Create separate heatmaps for left and right hands
        left_heatmap = np.zeros((downscaled_height, downscaled_width), dtype=np.float32)
        right_heatmap = np.zeros((downscaled_height, downscaled_width), dtype=np.float32)

        # Update heatmaps from buffers
        for x, y, intensity in left_hand_buffer:
            # Draw a circle on the heatmap with intensity based on speed
            cv2.circle(right_heatmap, (x, y), point_radius, intensity, -1)

        for x, y, intensity in right_hand_buffer:
            cv2.circle(left_heatmap, (x, y), point_radius, intensity, -1)
            # # Increase the intensity in a radius around the point to simulate a larger point size
            # for dx in range(-point_radius, point_radius + 1):
            #     for dy in range(-point_radius, point_radius + 1):
            #         nx, ny = x + dx, y + dy
            #         # Ensure (nx, ny) is within bounds
            #         if 0 <= nx < downscaled_width and 0 <= ny < downscaled_height:
            #             distance_from_center = np.sqrt(dx**2 + dy**2)
            #             if distance_from_center <= point_radius:  # Only affect points within the radius
            #                 heatmap[ny, nx] += intensity / (distance_from_center + 1)

        # Normalize and colorize the heatmaps
        left_heatmap = np.clip(left_heatmap * 10, 0, 255).astype(np.uint8)
        right_heatmap = np.clip(right_heatmap * 10, 0, 255).astype(np.uint8)
        overall_heatmap_left_normalized = np.clip(overall_heatmap_left * 10, 0, 255).astype(np.uint8)
        overall_heatmap_right_normalized = np.clip(overall_heatmap_right * 10, 0, 255).astype(np.uint8)

        # left_heatmap_colored = cv2.applyColorMap(left_heatmap, cv2.COLORMAP_SPRING)  # Green
        # right_heatmap_colored = cv2.applyColorMap(right_heatmap, cv2.COLORMAP_AUTUMN)  # Yellow
        overall_heatmap_left_colored = cv2.applyColorMap(overall_heatmap_left_normalized, cv2.COLORMAP_OCEAN)  # Green
        overall_heatmap_right_colored = cv2.applyColorMap(overall_heatmap_right_normalized, cv2.COLORMAP_SUMMER)  # Yellow

        # Resize heatmaps to original frame size and overlay
        # left_resized = cv2.resize(left_heatmap_colored, (width, height))
        # right_resized = cv2.resize(right_heatmap_colored, (width, height))

        # Combine the two heatmaps into one
        combined_heatmap = cv2.addWeighted(overall_heatmap_left_colored, 0.5, overall_heatmap_right_colored, 0.5, 0)
        # Create a mask for non-zero intensities in both heatmaps
        mask = (overall_heatmap_left_normalized > 0) | (overall_heatmap_right_normalized > 0)
        combined_heatmap[~mask] = 0  # Set background to black where both heatmaps have zero intensity

        # Resize for blending with the original frame
        combined_heatmap_resized = cv2.resize(combined_heatmap, (width, height))
        final_frame = cv2.addWeighted(frame, 0.7, combined_heatmap_resized, 0.5, 0)

        # Show the overlayed frame with the heatmap and hand landmarks
        # cv2.imshow('Hand Tracking with Heatmap', final_frame)
        video_writer.write(final_frame)        # Write the original frames into original video

        # Show the 1-minute frequency map in a separate window
        # cv2.imshow('Overall Heatmap', overall_heatmap_resized)

        ####################################################################
        # End modifying from here
        ####################################################################

        #ch = cv2.waitKey(1)
        #if ch == ord('q'):
            #break

    # cleanup the camera and close any open windows
    cam.release()
    video_writer.release()
    cv2.imwrite(output_heatmap_path, combined_heatmap_resized)
    print('Successfully saved heatmap!')
    cv2.destroyAllWindows()
    return output_video_file_path, output_heatmap_path
