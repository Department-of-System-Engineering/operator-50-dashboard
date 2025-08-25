# Import libraries
import numpy as np
import cv2 as cv
import mediapipe as mp
import math
import os
import convert_to_piechart as pie
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from get_score_REBA import GetREBAScores as reba
from get_score_RULA import GetRULAScores as rula

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")


class VideoElements:

    def __init__(self):
        pass

    def calculate_MiddlePoint(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        midpoint = (p1 + p2) / 2
        return tuple(midpoint)

    # Define a function to calculate angle between three points
    def calculate_angle(self, a, b, c):

        a = np.array(a)  # First point
        b = np.array(b)  # Middle point
        c = np.array(c)  # Last point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        # Ensure angle is between 0 and 180
        # if angle > 180.0:
        #     angle = 360 - angle
        return angle

    def determine_color(self, args: int):
        if (args == 1 or args == 2):
            return (0, 255, 0)
        elif (args == 3 or args == 4):
            return (100, 255, 255)
        elif (args == 5 or args == 6):
            return (0, 255, 255)
        elif (args > 6):
            return (0, 0, 255)

    def angle_btw_vectors(self, a, b, c, d):
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

    def get_video_duration(self, video_path):
        # Open the video file
        cap = cv.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        # Get the total number of frames
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Get the frames per second (FPS)
        fps = cap.get(cv.CAP_PROP_FPS)

        # Calculate the duration in seconds
        duration_seconds = total_frames / fps

        # Convert the duration to minutes and seconds format
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)

        cap.release()  # Release the video file
        return total_frames, fps, duration_seconds, f"{minutes} min {seconds} sec"

    # Define a function to calculate the distance between two points
    def calculate_distance(self, point1, point2):
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

    # get critical index
    def get_critical_index(self, df, assess_method):
        if assess_method == 'REBA': 
            crit_idx = {}
            max_neck = df['neck_adjusted'].idxmax()
            crit_idx['max_neck'] = (df.loc[max_neck, 'frame_number'], df.loc[max_neck, 'neck_adjusted'])

            max_trunk = df['trunk_adjusted'].idxmax()
            crit_idx['max_trunk'] = (df.loc[max_trunk, 'frame_number'], df.loc[max_trunk, 'trunk_adjusted'])

            max_legs = df['leg_score'].idxmax()
            crit_idx['max_legs'] = (df.loc[max_legs, 'frame_number'], df.loc[max_legs, 'leg_score'])

            max_up_arm = df[['u_arm_adjusted_l', 'u_arm_adjusted_r']].values.max(axis=1).argmax()
            crit_idx['max_up_arm'] = (df.loc[max_up_arm, 'frame_number'], 
                                df[['u_arm_adjusted_l', 'u_arm_adjusted_r']].iloc[max_up_arm].max())

            max_lo_arm = df[['lower_arm_score_l', 'lower_arm_score_r']].values.max(axis=1).argmax()
            crit_idx['max_lo_arm'] = (df.loc[max_lo_arm, 'frame_number'], 
                                df[['lower_arm_score_l', 'lower_arm_score_r']].iloc[max_lo_arm].max())

            max_wrist = df[['wrist_adjusted_l', 'wrist_adjusted_r']].values.max(axis=1).argmax()
            crit_idx['max_wrist'] = (df.loc[max_wrist, 'frame_number'], 
                                df[['wrist_adjusted_l', 'wrist_adjusted_r']].iloc[max_wrist].max())

            max_left = df['C_score_l'].idxmax()
            crit_idx['max_left'] = (df.loc[max_left, 'frame_number'], df.loc[max_left, 'C_score_l'])
            max_right = df['C_score_r'].idxmax()
            crit_idx['max_right'] = (df.loc[max_right, 'frame_number'], df.loc[max_right, 'C_score_r'])

            return crit_idx

        if assess_method == 'RULA': 
            crit_idx = {}
            max_neck = df['neck_adjusted'].idxmax()
            crit_idx['max_neck'] = (df.loc[max_neck, 'frame_number'], df.loc[max_neck, 'neck_adjusted'])

            max_trunk = df['trunk_adjusted'].idxmax()
            crit_idx['max_trunk'] = (df.loc[max_trunk, 'frame_number'], df.loc[max_trunk, 'trunk_adjusted'])

            max_legs = df['leg_score'].idxmax()
            crit_idx['max_legs'] = (df.loc[max_legs, 'frame_number'], df.loc[max_legs, 'leg_score'])

            max_up_arm = df[['u_arm_adjusted_l', 'u_arm_adjusted_r']].values.max(axis=1).argmax()
            crit_idx['max_up_arm'] = (df.loc[max_up_arm, 'frame_number'], 
                                df[['u_arm_adjusted_l', 'u_arm_adjusted_r']].iloc[max_up_arm].max())

            max_lo_arm = df[['lo_arm_adjusted_l', 'lo_arm_adjusted_r']].values.max(axis=1).argmax()
            crit_idx['max_lo_arm'] =  (df.loc[max_lo_arm, 'frame_number'], 
                                df[['lo_arm_adjusted_l', 'lo_arm_adjusted_r']].iloc[max_lo_arm].max())

            max_wrist = df[['wrist_adjusted_l', 'wrist_adjusted_r']].values.max(axis=1).argmax()
            crit_idx['max_wrist'] = (df.loc[max_wrist, 'frame_number'], 
                                df[['wrist_adjusted_l', 'wrist_adjusted_r']].iloc[max_wrist].max())

            max_left = df['C_score_l'].idxmax()
            crit_idx['max_left'] = (df.loc[max_left, 'frame_number'], df.loc[max_left, 'C_score_l'])
            max_right = df['C_score_r'].idxmax()
            crit_idx['max_right'] = (df.loc[max_right, 'frame_number'], df.loc[max_right, 'C_score_r'])

            return crit_idx

    # face mask
    def face_mask(self, frame, ear_c, nose):
        ear_nose_distance = self.calculate_distance(ear_c, nose)
        # Calculate radius (4-5 times the distance between ears)
        radius = int(2 * ear_nose_distance * frame.shape[1])  # Scale radius to frame size
        # Calculate nose position
        ear_x = int(ear_c[0] * frame.shape[1])
        ear_y = int(ear_c[1] * frame.shape[0])
        # Draw a big circle over the nose
        cv.circle(frame, (ear_x, ear_y), radius, (255, 0, 0), -1)  # Black-filled circle

    # get critical images
    def get_critical_img(self, input_video_file_path, crit_idx, analysis_time):

        # initialize Pose estimator
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        mp_drawing_styles = mp.solutions.drawing_styles

        # Open video
        cap = cv.VideoCapture(input_video_file_path)
        image_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        for key, frame_data in crit_idx.items():
            # Set the video to the specific frame
            # print('FRAME_DATA : ', frame_data)
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_data[0])

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_data[0]}. Skipping...")
                continue

            # Convert frame to RGB for Mediapipe
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Perform pose detection
            results = pose.process(frame_rgb)

            # Draw the pose skeleton on the frame
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark  # Extract landmarks
                nose = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                        landmarks[mp_pose.PoseLandmark.NOSE].y]
                ear_l = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]
                ear_r = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y]
                ear_c = self.calculate_MiddlePoint(ear_l, ear_r)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=10, circle_radius=20),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=10, circle_radius=20),
                )
                # face mask
                self.face_mask(frame, ear_c, nose)

                # trim the image
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Calculate center, width, height of bounding box
                box_width = x_max - x_min
                box_height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                # Scale width and height by 1.3
                scale = 1.8
                scaled_width = box_width * scale
                scaled_height = box_height * scale

                # Calculate new bounding box coordinates (normalized)
                new_x_min = max(center_x - scaled_width / 2, 0)
                new_x_max = min(center_x + scaled_width / 2, 1)
                new_y_min = max(center_y - scaled_height / 2, 0)
                new_y_max = min(center_y + scaled_height / 2, 1)

                # Convert normalized coordinates to pixel coordinates
                x1 = int(new_x_min * image_width)
                x2 = int(new_x_max * image_width)
                y1 = int(new_y_min * image_height)
                y2 = int(new_y_max * image_height)

                # Crop the image
                trimmed_frame = frame[y1:y2, x1:x2]

            # Save the frame with skeleton pose
            img_filename = f"{key}-{analysis_time}.png"
            img_filepath = os.path.join(EXPORTS_DIR, img_filename)
            # cv.imwrite(img_filepath, frame)
            cv.imwrite(img_filepath, trimmed_frame)
            # print(f"Saved frame {key} to {img_filepath}")

        cap.release()

    def calculate_REBA(self, input_dict, input_var):
        score_dict = {}
        # arm and wrist
        score_dict["u_arm_score_l"], score_dict["u_arm_adjusted_l"] = reba.getUpperArmREBA(input_dict['upper_arm_l'], input_var['shoulder_raised_l'], input_var['arm_abducted_l'], input_var['supported_leaning_l'])          
        score_dict["lo_arm_score_l"] = reba.getLowerArmREBA(input_dict['lower_arm_l'])
        score_dict["wrist_score_l"], score_dict['wrist_adjusted_l'] = reba.getWristREBA(input_dict['wrist_l'], input_var['wrist_twisted_mid_l'])

        score_dict['u_arm_score_r'], score_dict['u_arm_adjusted_r'] = reba.getUpperArmREBA(input_dict['upper_arm_r'], input_var['shoulder_raised_r'], input_var['arm_abducted_r'], input_var['supported_leaning_r'])
        score_dict['lo_arm_score_r'] = reba.getLowerArmREBA(input_dict['lower_arm_r'])
        score_dict['wrist_score_r'], score_dict['wrist_adjusted_r'] = reba.getWristREBA(input_dict['wrist_l'], input_var['wrist_twisted_mid_r'])

        # neck - trunk - leg
        score_dict['neck_score'], score_dict['neck_adjusted'] = reba.getNeckREBA(input_dict['neck_angle'], input_var['neck_twisted'] + input_var['neck_side'])
        score_dict['trunk_score'], score_dict['trunk_adjusted'] = reba.getTrunkREBA(input_dict['trunk_angle'], input_var['trunk_twisted'], input_var['trunk_side'])
        score_dict['leg_score'], score_dict['leg_adjusted'] = reba.getLegsREBA(input_dict['final_leg_angle'], input_dict["legs_down"])
        print('Leg angle: ' , input_dict['legs_angle'])

        # getting the REBA-table scores
        score_dict['A_score'] = reba.get_Table_A_Score(score_dict['trunk_score'], score_dict['neck_score'], score_dict['leg_score'])
        score_dict['A_adjusted'] = score_dict['A_score'] + input_var["force_load_leg"]

        score_dict['B_score_l'] = reba.get_Table_B_Score(score_dict['u_arm_score_l'], score_dict['lo_arm_score_l'], score_dict['wrist_score_l'])
        score_dict['B_adjusted_l'] = score_dict['B_score_l'] + input_var["coupling_l"]
        score_dict['B_score_r'] = reba.get_Table_B_Score(score_dict['u_arm_score_r'], score_dict['lo_arm_score_r'], score_dict['wrist_score_r'])
        score_dict['B_adjusted_r'] = score_dict['B_score_r'] + input_var["coupling_r"]

        # final score: REBA has additional activity score 
        score_dict['C_score_l'] = reba.get_Table_C_Score(score_dict['A_adjusted'], score_dict['B_adjusted_l']) + input_var["act_score"]
        score_dict['C_score_r'] = reba.get_Table_C_Score(score_dict['A_adjusted'], score_dict['B_adjusted_r']) + input_var["act_score"]
        print("REBA score for left side: ", score_dict['C_score_l'])
        print("REBA score for right side: ", score_dict['C_score_r'])

        if score_dict['C_score_l'] == 1 or score_dict['C_score_r'] == 1:
            action_required = "No action required"
            msd_risk_level = "Neglitible"
        elif score_dict['C_score_l'] >= 2 and score_dict['C_score_l'] <= 3 or score_dict['C_score_r'] >= 2 and score_dict['C_score_r'] <= 3:
            action_required = "May need action"
            msd_risk_level = "Low"
        elif score_dict['C_score_l'] >= 4 and score_dict['C_score_l'] <= 7 or score_dict['C_score_r'] >= 4 and score_dict['C_score_r'] <= 7:
            action_required = "Change soon"
            msd_risk_level = "Medium"
        elif score_dict['C_score_l'] >= 8 and score_dict['C_score_l'] <= 10 or score_dict['C_score_r'] >= 8 and score_dict['C_score_r'] <= 10:
            action_required = "Investigation needed"
            msd_risk_level = "High"
        elif score_dict['C_score_l'] >= 11 or score_dict['C_score_r'] >= 11:
            action_required = "Change immediately"
            msd_risk_level = "Very High"

        return score_dict, action_required , msd_risk_level

    def calculate_RULA(self, input_dict, input_var):
        score_dict = {}
        # arm and wrist
        score_dict['u_arm_score_l'], score_dict['u_arm_adjusted_l'] = rula.getUpperArmRULA(input_dict['upper_arm_l'], input_var['shoulder_raised_l'], input_var['arm_abducted_l'], input_var['supported_leaning_l'])
        score_dict['lo_arm_score_l'], score_dict['lo_arm_adjusted_l'] = rula.getLowerArmRULA(input_dict['lower_arm_l'], input_var['arm_mid_l'])
        score_dict['wrist_score_l'], score_dict['wrist_adjusted_l'] = rula.getWristRULA(input_dict['wrist_l'], input_var['wrist_bent_l'])
        score_dict['wrist_twist_l'] = rula.getWristTwistRULA(input_var["wrist_twisted_mid_l"], input_var["wrist_end_l"])

        score_dict['u_arm_score_r'], score_dict['u_arm_adjusted_r'] = rula.getUpperArmRULA(input_dict['upper_arm_r'], input_var['shoulder_raised_r'], input_var['arm_abducted_r'], input_var['supported_leaning_r'])
        score_dict['lo_arm_score_r'], score_dict['lo_arm_adjusted_r'] = rula.getLowerArmRULA(input_dict['lower_arm_r'], input_var['arm_mid_r'])
        score_dict['wrist_score_r'], score_dict['wrist_adjusted_r'] = rula.getWristRULA(input_dict['wrist_r'], input_var['wrist_bent_r'])
        score_dict['wrist_twist_r'] = rula.getWristTwistRULA(input_var["wrist_twisted_mid_r"], input_var["wrist_end_r"])

        # neck - trunk - leg
        score_dict['neck_score'], score_dict['neck_adjusted'] = rula.getNeckRULA(input_dict['neck_angle'], input_var['neck_twisted'], input_var['neck_side'])
        score_dict['trunk_score'], score_dict['trunk_adjusted'] = rula.getTrunkRULA(input_dict['trunk_angle'], input_var['trunk_twisted'], input_var['trunk_side'])
        score_dict['leg_score'] = rula.getLegRULA(input_var["legs_supported"])  # yes = 1 , no = 2

        # getting the RULA-table scores
        score_dict['A_score_l'] = rula.get_Table_A_Score(score_dict['u_arm_score_l'], score_dict['lo_arm_score_l'],
                                        score_dict['wrist_score_l'], score_dict['wrist_twist_l'])
        score_dict['A_adjusted_l'] = score_dict['A_score_l'] + input_var["muscle_arms_l"] + input_var["force_load_l"]
        score_dict['A_score_r'] = rula.get_Table_A_Score(score_dict['u_arm_score_r'], score_dict['lo_arm_score_r'],
                                        score_dict['wrist_score_r'], score_dict['wrist_twist_r'])
        score_dict['A_adjusted_r'] = score_dict['A_score_r'] + input_var["muscle_arms_r"] + input_var["force_load_r"]

        # getting the RULA-table scores
        score_dict['B_score'] = rula.get_Table_B_Score(score_dict['neck_adjusted'], score_dict['trunk_score'], score_dict['leg_score'])
        score_dict['B_adjusted'] = score_dict['B_score'] + input_var["muscle_leg"] + input_var["load_force_upper_body"]
        # final score
        score_dict['C_score_l'] = rula.get_Table_C_Score(score_dict['A_adjusted_l'], score_dict['B_adjusted'])
        score_dict['C_score_r'] = rula.get_Table_C_Score(score_dict['A_adjusted_r'], score_dict['B_adjusted'])
        print("RULA score for left side: ", score_dict['C_score_l'])
        print("RULA score for right side: ", score_dict['C_score_r'])

        if score_dict['C_score_l'] >= 1 and score_dict['C_score_l'] <=2 or score_dict['C_score_r'] >= 1 and score_dict['C_score_r'] <= 2:
            action_required = "No action required"
            msd_risk_level = "Low"
        elif score_dict['C_score_l'] >= 3 and score_dict['C_score_l'] <= 4 or score_dict['C_score_r'] >= 3 and score_dict['C_score_r'] <= 4:
            action_required = "Investigation needed"
            msd_risk_level = "Medium"
        elif score_dict['C_score_l'] >= 5 and score_dict['C_score_l'] <= 6 or score_dict['C_score_r'] >= 5 and score_dict['C_score_r'] <= 6:
            action_required = "Change soon"
            msd_risk_level = "High"
        elif score_dict['C_score_l'] >= 7 or score_dict['C_score_r'] >= 7:
            action_required = "Change immediately"
            msd_risk_level = "Very High"

        return score_dict, action_required, msd_risk_level

    def create_input_dict(self, landmarks, mp_pose):
        input_dict = {}
        input_dict['nose'] = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                landmarks[mp_pose.PoseLandmark.NOSE].y]
        # Shoulder, Hip, and Ear coordinates (assuming back view)
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ear_l = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]
        ear_r = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y]
        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        pinky_l = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x,
                landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y]
        index_l = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x,
                landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
        pinky_r = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y]
        index_r = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]

        # Calculate mid-points
        input_dict['ear_c'] = self.calculate_MiddlePoint(ear_l, ear_r)
        shoulder_c = self.calculate_MiddlePoint(shoulder_l, shoulder_r)
        hip_c = self.calculate_MiddlePoint(hip_l, hip_r)
        knee_c = self.calculate_MiddlePoint(knee_l, knee_r)
        hand_c_l = self.calculate_MiddlePoint(pinky_l, index_l)
        hand_c_r = self.calculate_MiddlePoint(pinky_r, index_r)
        # angles
        input_dict['upper_arm_r'] = int(self.angle_btw_vectors(shoulder_r, elbow_r,
                                        shoulder_c, hip_c))
        input_dict['upper_arm_l'] = int(self.angle_btw_vectors(shoulder_l, elbow_l,
                                        shoulder_c, hip_c))
        input_dict['lower_arm_r'] = int(self.angle_btw_vectors(elbow_r, wrist_r,
                                        shoulder_r, elbow_r))
        input_dict['lower_arm_l'] = int(self.angle_btw_vectors(elbow_l, wrist_l,
                                        shoulder_l, elbow_l))
        input_dict['wrist_l'] = int(abs(180 - self.calculate_angle(elbow_l, wrist_l, hand_c_l)))
        input_dict['wrist_r'] = int(abs(180 - self.calculate_angle(elbow_r, wrist_r, hand_c_r)))
        input_dict['neck_angle'] = int(abs(180 - self.calculate_angle(hip_c, shoulder_c, input_dict['ear_c'])))
        input_dict['trunk_angle'] = int(abs(180 - self.calculate_angle(knee_c, hip_c, shoulder_c)))
        input_dict['legs_angle'] = int(self.angle_btw_vectors(knee_r, ankle_r, knee_l, ankle_l))

        # Calculate the angle of the legs based on hip, knee, and ankle positions
        input_dict['leg_angle_l'] = int(abs(180 - self.calculate_angle(hip_l, knee_l, ankle_l)))
        input_dict['leg_angle_r'] = int(abs(180 - self.calculate_angle(hip_r, knee_r, ankle_r)))

        # Compute the final leg angle as the average of both leg angles
        input_dict['final_leg_angle'] = int((input_dict['leg_angle_l'] + input_dict['leg_angle_r']) / 2)

        # Compute shank lengths as distances between knee and ankle
        shank_l = self.calculate_distance(knee_l, ankle_l)
        shank_r = self.calculate_distance(knee_r, ankle_r)

        # Thresholds
        # height_threshold = ((shank_l + shank_r)/2)*(1/10)  # 10% of the shanks average

        # Check if knees and ankles are at the same height (both legs down)
        # knees_same_height = abs(knee_l[1] - knee_r[1]) < height_threshold # no need for this constraint
        if shank_r != 0:  # avoid the case of 0 value
            shanks_equal = 0.6 < shank_l / shank_r < 1.4
        else:
            shanks_equal = False

        shanks_parallel = abs(input_dict["legs_angle"]) < 10  # Small angle means legs are parallel 

        # Condition 1: Both legs are down
        if shanks_equal and shanks_parallel:
            input_dict['legs_down'] = 1  # Both legs are down
        else:
            input_dict["legs_down"] = 2  # At least one leg is up

        print('Shank equal: ', shanks_equal, ', shank parallel: ', shanks_parallel)
        return input_dict
    

