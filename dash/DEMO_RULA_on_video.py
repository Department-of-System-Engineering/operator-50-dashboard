# Import libraries
import numpy as np
import cv2 as cv
import mediapipe as mp
import math
computer_name = "C:/Users/raszt" 
import os
#os.chdir(computer_name + "/OneDrive/Asztali gép/input_videos")
from calculate_RULA import GetRULAScores as rc
from calculate_REBA import GetREBAScores as rec
video_folder = computer_name + "/OneDrive/Asztali gép/input_videos"
filename = "IMG_1014.MOV"

def calculate_MiddlePoint(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    midpoint = (p1 + p2) / 2
    return tuple(midpoint)

# Define a function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    # Ensure angle is between 0 and 180
    # if angle > 180.0:
    #     angle = 360 - angle
    return angle

def calculate_angle_between_vectors(a, b, c, d):
    """
    Calculate the angle between vectors AB and CD.

    Parameters:
    a (tuple): Coordinates of point A (x1, y1).
    b (tuple): Coordinates of point B (x2, y2).
    c (tuple): Coordinates of point C (x3, y3).
    d (tuple): Coordinates of point D (x4, y4).

    Returns:
    float: The angle in degrees.
    """
    # Vector AB
    ab = (b[0] - a[0], b[1] - a[1])
    # Vector CD
    cd = (d[0] - c[0], d[1] - c[1])

    # Dot product of AB and CD
    dot_product = ab[0] * cd[0] + ab[1] * cd[1]
    # Magnitudes of AB and CD
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_cd = math.sqrt(cd[0]**2 + cd[1]**2)

    # Cosine of the angle
    cos_angle = dot_product / (magnitude_ab * magnitude_cd)

    # Avoid floating-point errors outside the range [-1, 1]
    cos_angle = max(-1, min(1, cos_angle))

    # Calculate the angle in radians and convert to degrees
    angle = math.acos(cos_angle)
    return math.degrees(angle)

def calculate_distance(point1, point2):
    """
    Calculate the distance between two 2D points.

    Parameters:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: The distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

#% create the dashboard with fixed variables
def create_dashboard():
    # downscaled_width, downscaled_height = 540, 960
    # line_space  = 40
    # Create a black image
    canvas_h, canvas_w = 960, 600
    right_rule, left_rule = 360, 490    
    blank_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    cv.putText(blank_canvas, 'RIGHT', (right_rule, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 192, 203), 2)
    cv.putText(blank_canvas, 'LEFT', (left_rule, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv.putText(blank_canvas, 'ARM & WRIST: ', (10, 80),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.putText(blank_canvas, '1. Upper arm: ', (10, 120),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted (+):  ', (10, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, '2. Lower arm: ', (10, 180),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted (+): ', (10, 200),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, '3. Wrist position: ', (10, 240),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted (+):  ', (10, 260),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, '4. Wrist twist: ', (10, 300),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.putText(blank_canvas, '5. Score A: ', (10, 340),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, '6. Muscle use (+):  ', (10, 380),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, '7. Force/load (+):  ', (10, 420),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted score: ', (10, 460),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.putText(blank_canvas, 'NECK, TRUNK & LEG: ', (10, 550),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.putText(blank_canvas, '9. Neck position: ', (10, 590),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted (+):  ', (10, 610),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(blank_canvas, '10. Trunk position: ', (10, 650),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
    cv.putText(blank_canvas, 'Adjusted (+): ', (10, 670),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(blank_canvas, '11. Legs: ', (10, 710),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, '12. Score B: ', (10, 750),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, '13. Muscle use (+):  ', (10, 790),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, '14. Force/load (+):  ', (10, 830),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, 'Adjusted score: ', (10, 870),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv.putText(blank_canvas, 'FINAL SCORE: ', (10, 935),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return blank_canvas

# Function to update the dashboard
def update_canvas(args : int , final_score : bool):
    # Clear the image for dynamic updates
    updated_canvas = blank_canvas.copy()
    right_rule, left_rule = 360, 490    
    center_rule = int((right_rule + left_rule ) / 2)
    if(final_score):
        if (args == 1 or args == 2) :
            return (0, 255, 0)
        elif (args == 3 or args == 4) :
            return (100, 255, 255)
        elif (args == 5 or args == 6) :
            return (0, 255, 255)
        elif (args > 6 ) :
            return (0, 0, 255)
        
    right_color = (255, 192, 203)
    left_color = (0, 255, 255)
           
    # show the variables here
    cv.putText(updated_canvas, f'{int(upper_arm_r)}', (right_rule, 120),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{upper_arm_score_r} => {upper_arm_adjusted_r}', (right_rule, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
    cv.putText(updated_canvas, f'{int(lower_arm_r)}', (right_rule, 180),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas,  f'{lower_arm_score_r} => {lower_arm_adjusted_r}', (right_rule, 200),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
    cv.putText(updated_canvas, f'{int(wrist_r)}', (right_rule, 240),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{wrist_score_r} => {wrist_adjusted_r}', (right_rule, 260),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)    
    cv.putText(updated_canvas, f'{wrist_twist_r}', (right_rule, 300),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{A_score_r}', (right_rule, 340),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{muscle_use_r}', (right_rule, 380),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{force_load_r}', (right_rule, 420),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    cv.putText(updated_canvas, f'{A_score_adjusted_r}', (right_rule, 460),
               cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
    
    cv.putText(updated_canvas, f'{int(upper_arm_l)}', (left_rule, 120),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{upper_arm_score_l} => {upper_arm_adjusted_l}', (left_rule, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
    cv.putText(updated_canvas, f'{int(lower_arm_r)}', (left_rule, 180),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{lower_arm_score_l} => {lower_arm_adjusted_l}', (left_rule, 200),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
    cv.putText(updated_canvas, f'{int(wrist_l)}', (left_rule, 240),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{wrist_score_l} => {wrist_adjusted_l}', (left_rule, 260),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)    
    cv.putText(updated_canvas, f'{wrist_twist_l}', (left_rule, 300),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{A_score_l}', (left_rule, 340),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{muscle_use_l}', (left_rule, 380),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{force_load_l}', (left_rule, 420),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    cv.putText(updated_canvas, f'{A_score_adjusted_l}', (left_rule, 460),
               cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
    
    cv.putText(updated_canvas, f"{int(neck_angle)}", (center_rule, 590),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(updated_canvas, f"{neck_score} => {neck_adjusted_score}", (center_rule, 610),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(updated_canvas, f'{int(trunk_angle)}', (center_rule, 650),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
    cv.putText(updated_canvas, f"{trunk_score} => {trunk_adjusted_score}", (center_rule, 670),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(updated_canvas, f'{leg_score}', (center_rule, 710),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(updated_canvas, f'{B_score}', (center_rule, 750),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(updated_canvas, f'{muscle_use_leg}', (center_rule, 790),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(updated_canvas, f'{force_load_leg}', (center_rule, 830),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(updated_canvas, f'{int(B_score_adjusted)}', (center_rule, 870),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    color_l = update_canvas(C_score_l , True)
    color_r = update_canvas(C_score_r , True)

    cv.putText(updated_canvas, f'{C_score_r}', (right_rule, 935),
               cv.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)
    cv.putText(updated_canvas, f'{C_score_l}', (left_rule, 935),
               cv.FONT_HERSHEY_SIMPLEX, 1, color_l, 2)
    
    print(update_canvas(C_score_l , True))
    print(update_canvas(C_score_r , True))


    return updated_canvas
    
#%%
def process_skeleton_video_file_with_reba_score(filename):
    input_video_file_path = os.path.join("exports", filename)
    filename_without_extension = filename.split('.')[0]
    output_video_file_name = f'output_reba_{filename_without_extension}.mp4'
    output_video_file_path = os.path.join("exports", output_video_file_name)

    # initialize Pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
    mp_drawing_styles = mp.solutions.drawing_styles
    total_frames = 0

    #% Start video streaming
    cam = cv.VideoCapture(input_video_file_path)
    # Video setting
    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Reduce image size (downscale for speed)
    scale_factor = 0.25  # Downscale factor (0.5 means reducing the size by half)
    downscaled_width = int(width * scale_factor)
    downscaled_height = int(height * scale_factor)

    frame_count = 0  # Frame counter for time tracking
    fps = 30
    # create visualization
    blank_canvas = np.zeros((downscaled_height, int(downscaled_width * 2), 3), dtype=np.uint8)
    # vid_writer = cv.VideoWriter(video_name[:-4] + "_output.avi", cv.VideoWriter_fourcc(*'XVID'), fps, (blank_canvas.shape[1], downscaled_height))
    vid_writer = cv.VideoWriter(output_video_file_path, cv.VideoWriter_fourcc(*'MP4V'), fps, (blank_canvas.shape[1], downscaled_height))


    while True:
        _ret, frame = cam.read()
        if not _ret:
            print('No frames grabbed!')
            break

        # Resize frame for faster processing
        frame = cv.resize(frame, (downscaled_width, downscaled_height))
        # Convert the BGR image to RGB.
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        # process the RGB frame to get the skeleton
        results = pose.process(image)
        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Check if pose landmarks are detected.
        if results.pose_landmarks:
            # Extract relevant landmarks
            landmarks = results.pose_landmarks.landmark

            # Shoulder, Hip, and Ear coordinates (assuming back view)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            left_ankle =  [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Calculate mid-points
            ear_center = calculate_MiddlePoint(left_ear, right_ear)
            ankle_center = calculate_MiddlePoint(left_ankle , right_ankle)
            shoulder_center = calculate_MiddlePoint(left_shoulder, right_shoulder)
            hip_center = calculate_MiddlePoint(left_hip, right_hip)
            knee_center = calculate_MiddlePoint(left_knee, right_knee)
            hand_center_l = calculate_MiddlePoint(left_pinky, left_index)
            hand_center_r = calculate_MiddlePoint(right_pinky, right_index)
            # angles
            upper_arm_r = calculate_angle_between_vectors(right_shoulder, right_elbow , shoulder_center, hip_center)
            upper_arm_l = calculate_angle_between_vectors(left_shoulder, left_elbow , shoulder_center, hip_center)
            lower_arm_r = calculate_angle_between_vectors(right_elbow, right_wrist, right_shoulder, right_elbow)
            lower_arm_l = calculate_angle_between_vectors(left_elbow, left_wrist, left_shoulder, left_elbow)
            wrist_l = abs(180 - calculate_angle(left_elbow, left_wrist, hand_center_l))
            wrist_r = abs(180 - calculate_angle(right_elbow, right_wrist, hand_center_r))
            neck_angle = abs(180 - calculate_angle(hip_center, shoulder_center, ear_center))
            trunk_angle = abs(180 - calculate_angle(knee_center, hip_center, shoulder_center))
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image = cv.line(image, (int(ear_center[0]), int(ear_center[1])), (int(shoulder_center[0]), int(shoulder_center[1])), (255, 0, 0), 2)

        # Display the image
        # cv.putText(image, f'Upper right arm angle: {int(upper_arm)}, RULA score {int(upper_arm_score)}', (50, 50),
        # cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # landmark = landmarks[:-1]

        # create visualization
        blank_canvas = np.zeros((downscaled_height, int(downscaled_width * 2), 3), dtype=np.uint8)

        blank_canvas[0: downscaled_height, 0: downscaled_width] = image
        # cv.putText(image, f'Upper right arm angle: {int(upper_arm)} ,{reba.getRebaScore(landmark)}', (50, 50),
        # cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv.putText(blank_canvas, 'A. UPPER ARM ANGLE: ', (downscaled_width + 50, 100),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, f'Right arm: {int(upper_arm_r)}', (downscaled_width + 50, 150),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, f'Left arm: {int(upper_arm_l)}', (downscaled_width + 50, 200),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, 'A. LOWER ARM ANGLE: ', (downscaled_width + 50, 250),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, f'Right arm: {int(lower_arm_r)}', (downscaled_width + 50, 300),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, f'Left arm: {int(lower_arm_l)}', (downscaled_width + 50, 350),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, 'A. WRIST ANGLE: ', (downscaled_width + 50, 400),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, f'Right arm: {int(wrist_r)}', (downscaled_width + 50, 450),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, f'Left arm: {int(wrist_l)}', (downscaled_width + 50, 500),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, 'B. NECK: ', (downscaled_width + 50, 550),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, f'Neck angle: {int(neck_angle)}', (downscaled_width + 50, 600),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv.putText(blank_canvas, 'B. TRUNK: ', (downscaled_width + 50, 650),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, f'Trunk angle: {int(trunk_angle)}', (downscaled_width + 50, 700),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        upper_arm_score_l = rc.getUpperArmRULA(upper_arm_l , 0 , 0 , 0)
        lower_arm_score_l = rc.getLowerArmRULA(lower_arm_l , 0)
        wrist_score_l  = rc.getWristRULA(wrist_l , 0 , 1)
        neck_score = rc.getNeckRULA(neck_angle , 0 , 0)
        trunk_score = rc.getTrunkRULA(trunk_angle , 0, 0)
        leg_score_l = rc.getLegRULA(1)
        supp_l = 1
        print("Supp_l:",supp_l)

        #getting the RULA-table scores
        left_A_score = rc.get_Table_A_Score(upper_arm_score_l , lower_arm_score_l , wrist_score_l , supp_l)
        left_B_score = rc.get_Table_B_Score(neck_score , trunk_score , leg_score_l)

        #final score
        left_C_score = rc.get_Table_C_Score(left_A_score , left_B_score)
        print("RULA score for left side: " , left_C_score)

        #example usage of the REBA-calculator for left side
        reba_neck_score = rec.getNeckREBA(neck_angle , 1 , 0)
        reba_trunk_score = rec.getTrunkREBA(trunk_angle , 1 , 0)
        reba_legs_score = rec.getLegsREBA(leg_l , 1)
        reba_upper_arm_score = rec.getUpperArmREBA(upper_arm_l , 1 ,1 ,0)
        reba_lower_arm_score = rec.getLowerArmREBA(lower_arm_l)
        reba_wrist_score = rec.getWristREBA(wrist_l , 1)

        #getting the REBA-table scores
        reba_left_A_score = rec.get_Table_A_Score(reba_trunk_score , reba_neck_score , reba_legs_score)
        reba_left_B_score = rec.get_Table_B_Score(reba_upper_arm_score , reba_lower_arm_score , reba_wrist_score)

        #final score
        reba_left_C_score = rc.get_Table_C_Score(reba_left_A_score , reba_left_B_score)
        print("REBA score for left side: " , reba_left_C_score)

        #####################################################################################################
        cv.imshow('Ergonomics Assessment', blank_canvas)
        # Increment frame counter
        frame_count += 1
        total_frames += 1
        # print(landmarks)

        # #write the output frame
        vid_writer.write(blank_canvas)

        ch = cv.waitKey(1)
        if ch == ord('q'):
            break
    # cleanup the camera and close any open windows
    cam.release()
    vid_writer.release()
    cv.destroyAllWindows()
#%%

input_video_file_path = os.path.join(video_folder, filename)
filename_without_extension = filename.split('.')[0]
output_video_file_name = f'{filename_without_extension}_output_rula.mp4'
output_video_file_path = os.path.join("./output/", output_video_file_name)

#% initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing_styles = mp.solutions.drawing_styles

# % Start video streaming
cam = cv.VideoCapture(input_video_file_path)

#list and variable for creating the excel file
data = []
frame_number = 0

# Video setting
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
# Reduce image size (downscale for speed)
scale_factor = 0.25  # Downscale factor (0.5 means reducing the size by half)
downscaled_width = int(width * scale_factor)
downscaled_height = int(height * scale_factor)

fps = 30
# create visualization
blank_canvas = create_dashboard()
vid_writer = cv.VideoWriter(output_video_file_path, cv.VideoWriter_fourcc(*'MP4V'), fps, (blank_canvas.shape[1], downscaled_height))

while True:
    _ret, frame = cam.read()
    if not _ret:
        print('No frames grabbed!')
        break

    # Resize frame for faster processing
    frame = cv.resize(frame, (downscaled_width, downscaled_height))
    # Convert the BGR image to RGB.
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    # process the RGB frame to get the skeleton
    results = pose.process(image)
    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Check if pose landmarks are detected.
    if results.pose_landmarks:
        # Extract relevant landmarks
        landmarks = results.pose_landmarks.landmark

        # Shoulder, Hip, and Ear coordinates (assuming back view)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y]
        left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
        right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y]
        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]

        # Calculate mid-points
        ear_center = calculate_MiddlePoint(left_ear, right_ear)
        # ankle_center = calculate_MiddlePoint(left_ankle, right_ankle)
        shoulder_center = calculate_MiddlePoint(left_shoulder, right_shoulder)
        hip_center = calculate_MiddlePoint(left_hip, right_hip)
        knee_center = calculate_MiddlePoint(left_knee, right_knee)
        hand_center_l = calculate_MiddlePoint(left_pinky, left_index)
        hand_center_r = calculate_MiddlePoint(right_pinky, right_index)
        # angles
        upper_arm_r = calculate_angle_between_vectors(right_shoulder, right_elbow, shoulder_center, hip_center)
        upper_arm_l = calculate_angle_between_vectors(left_shoulder, left_elbow, shoulder_center, hip_center)
        lower_arm_r = calculate_angle_between_vectors(right_elbow, right_wrist, right_shoulder, right_elbow)
        lower_arm_l = calculate_angle_between_vectors(left_elbow, left_wrist, left_shoulder, left_elbow)
        wrist_l = abs(180 - calculate_angle(left_elbow, left_wrist, hand_center_l))
        wrist_r = abs(180 - calculate_angle(right_elbow, right_wrist, hand_center_r))
        neck_angle = abs(180 - calculate_angle(hip_center, shoulder_center, ear_center))
        trunk_angle = abs(180 - calculate_angle(knee_center, hip_center, shoulder_center))
        legs_angle = calculate_angle_between_vectors(right_knee, right_ankle, left_knee, left_ankle)
        # distances between knee and ankle
        leg_l = calculate_distance(right_knee, right_ankle)
        leg_r = calculate_distance(left_knee, left_ankle)
        print("legs angle: ", legs_angle)
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv.line(image, (int(ear_center[0]), int(ear_center[1])), (int(shoulder_center[0]), int(shoulder_center[1])), (255, 255, 0), 2)

    # RULA-calculator
    # input parameters
    shoulder_raised_l, arm_abducted_l, supported_leaning_l = 0, 0 , 0
    shoulder_raised_r, arm_abducted_r, supported_leaning_r = 0, 0 , 0
    arm_mid_l = 0
    arm_mid_r = 0
    wrist_bent_l, wrist_twisted_mid_l, wrist_end_range_l = 0, 1, 0
    wrist_bent_r, wrist_twisted_mid_r, wrist_end_range_r = 0, 1, 0
    muscle_use_l = 0
    muscle_use_r = 0
    force_load_l = 0
    force_load_r = 0      
    # arm and wrist
    upper_arm_score_l = rc.getUpperArmRULA(upper_arm_l)
    upper_arm_adjusted_l = upper_arm_score_l + shoulder_raised_l + arm_abducted_l + supported_leaning_l
    lower_arm_score_l = rc.getLowerArmRULA(lower_arm_l)
    lower_arm_adjusted_l = lower_arm_score_l + arm_mid_l
    wrist_score_l = rc.getWristRULA(wrist_l)
    wrist_adjusted_l = wrist_score_l + wrist_bent_l
    wrist_twist_l = rc.getWristTwistRULA(wrist_twisted_mid_l, wrist_end_range_l)

    upper_arm_score_r = rc.getUpperArmRULA(upper_arm_r)
    upper_arm_adjusted_r = upper_arm_score_l + shoulder_raised_r + arm_abducted_r + supported_leaning_r
    lower_arm_score_r = rc.getLowerArmRULA(lower_arm_r)
    lower_arm_adjusted_r = lower_arm_score_r + arm_mid_r
    wrist_score_r = rc.getWristRULA(wrist_r)
    wrist_adjusted_r = wrist_score_r + wrist_bent_r
    wrist_twist_r = rc.getWristTwistRULA(wrist_twisted_mid_r, wrist_end_range_r)
    
    # getting the RULA-table scores
    A_score_l = rc.get_Table_A_Score(upper_arm_score_l, lower_arm_score_l, wrist_score_l, wrist_twist_l)
    A_score_adjusted_l = A_score_l + muscle_use_l + force_load_l
    A_score_r = rc.get_Table_A_Score(upper_arm_score_r, lower_arm_score_r, wrist_score_r, wrist_twist_r)
    A_score_adjusted_r = A_score_r + muscle_use_r + force_load_r

    # getting the REBA-table scores
    #REBA_A_score_l = rc.get_Table_A_Score(upper_arm_score_l, lower_arm_score_l, wrist_score_l, wrist_twist_l)
    #REBA_A_score_adjusted_l = A_score_l + muscle_use_l + force_load_l
    #REBA_A_score_r = rc.get_Table_A_Score(upper_arm_score_r, lower_arm_score_r, wrist_score_r, wrist_twist_r)
    #REBA_A_score_adjusted_r = A_score_r + muscle_use_r + force_load_r
    
    # neck, trunk and leg 
    neck_twisted, neck_side = 0, 0
    trunk_twisted, trunk_side = 0, 0
    muscle_use_leg = 0
    force_load_leg = 0
    neck_score = rc.getNeckRULA(neck_angle)
    neck_adjusted_score = neck_score + neck_twisted + neck_side
    trunk_score = rc.getTrunkREBA(trunk_angle)    
    trunk_adjusted_score = trunk_score + trunk_twisted + trunk_side  
    leg_score = rc.getLegREBA(legs_angle, leg_l, leg_r)
    
    # getting the RULA-table scores
    B_score = rc.get_Table_B_Score(neck_adjusted_score, trunk_score, leg_score)
    B_score_adjusted = B_score + muscle_use_leg + force_load_leg
    # final score
    C_score_l = rc.get_Table_C_Score(A_score_adjusted_l, B_score_adjusted)
    C_score_r = rc.get_Table_C_Score(A_score_adjusted_r, B_score_adjusted)

    #####################################################################################################
    # create visualization
    # Define the position where the embedded window will be placed
    updated_canvas = update_canvas(blank_canvas , False)
    merged_canvas = cv.hconcat([image, updated_canvas]) 

    #####################################################################################################
    cv.imshow('RULA Assessment', merged_canvas)

    # #write the output frame
    vid_writer.write(merged_canvas)

    ch = cv.waitKey(1)
    if ch == ord('q'):
        break
# cleanup the camera and close any open windows
cam.release()
vid_writer.release()
cv.destroyAllWindows()
