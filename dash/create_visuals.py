# Import libraries
import numpy as np
import cv2 as cv
import convert_to_piechart as pie


class VisualElements:
    layout = {
        "line_space": 20,
        "arm_title": 80,
        "Upper arm": 120, "Lower arm": 180, "Wrist position": 240,
        "Arm/Wrist Score": 320, "adjusted arm score": 400,
        "neck_title": 530,
        "Neck position": 570, "Trunk position": 630, "Legs": 690,
        "Neck/Trunk Score": 730, "adjusted_neck": 810,
        "final_line": 935
        }
    
    def categorize_color_REBA(self, score) :
        if score == 1:
             return (0,255,255)
        elif score >= 2 and score <= 3 :
              return (50, 205, 50)
        elif score >= 4 and score <= 7:
              return (255, 255, 0)
        elif score >= 8 and score <= 10:
            return (255, 165, 0)
        elif score >= 11 :
            return (255, 0, 0)
    

    def categorize_color_RULA(self, score) :
        if score >= 1 and score <= 2:
            return (0,255,255)
        elif score >= 3 and score <= 4:
            return (50, 205, 50)
        elif score >= 5 and score <= 6:
            return (255, 255, 0)
        elif score == 7:
            return (255, 0, 0)
        
  
    # create the dashboard with fixed variables, width, height = 540, 960
    def categorize_risk_REBA(self, score):
        if score == 1:
            return "Negligible Risk"
        elif score >= 2 and score <= 3:
            return "Low Risk"
        elif score >= 4 and score <= 7:
            return "Medium Risk"
        elif score >= 8 and score <= 10:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def categorize_risk_RULA(self, score):
        if score >= 1 and score <= 2:
            return "Acceptable posture"
        elif score >= 3 and score <= 4:
            return "Further investigation, change may be needed"
        elif score >= 5 and score <= 6:
            return "Further investigation, change soon"
        elif score == 7:
            return "Investigate and implement change"
        
        
    def create_dashboard(self, assess_method):
        layout = self.layout
        # Create a black image
        canvas_h, canvas_w = 960, 600
        right_rule, left_rule = 360, 490
        blank_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cv.putText(blank_canvas, f'{str(assess_method)}', (100, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'RIGHT', (right_rule, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 192, 203), 2)
        cv.putText(blank_canvas, 'LEFT', (left_rule, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv.putText(blank_canvas, 'ARM & WRIST: ', (10, layout["arm_title"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, '1. Upper arm: ', (10, layout["Upper arm"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'Adjusted (+):  ', (10, layout["Upper arm"] + layout["line_space"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv.putText(blank_canvas, '2. Lower arm: ', (10, layout["Lower arm"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, '3. Wrist position: ', (10, layout["Wrist position"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'Adjusted (+):  ', (10, layout["Wrist position"] + layout["line_space"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv.putText(blank_canvas, '4. Arm/Wrist Score: ', (10, layout["Arm/Wrist Score"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, 'Adjusted score: ', (10, layout["adjusted arm score"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, 'NECK, TRUNK & LEG: ', (10, layout["neck_title"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, '1. Neck position: ', (10, layout["Neck position"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'Adjusted (+):  ', (10, layout["Neck position"] + layout["line_space"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(blank_canvas, '2. Trunk position: ', (10, layout["Trunk position"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'Adjusted (+): ', (10, layout["Trunk position"] + layout["line_space"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(blank_canvas, '3. Legs (+): ', (10, layout["Legs"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, '4. Neck/Trunk Score: ', (10, layout["Neck/Trunk Score"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(blank_canvas, 'Force/load (+):  ', (10, layout["Neck/Trunk Score"] + layout["line_space"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(blank_canvas, 'Adjusted score: ', (10, layout["adjusted_neck"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.putText(blank_canvas, 'FINAL SCORE: ', (10, layout["final_line"]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Second: the variables that are the different between RULA/REBA
        if assess_method == "RULA":
                # only RULA has adjusted score for lower arm
                cv.putText(blank_canvas, 'Adjusted (+): ', (10, layout["Lower arm"] + layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # only RULA has wrist twist
                cv.putText(blank_canvas, 'Wrist twist: ', (10, layout["Wrist position"] + 2 * layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # only RULA has muscle use and force load for arms
                cv.putText(blank_canvas, 'Muscle use (+):  ', (10, layout["Arm/Wrist Score"] + layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(blank_canvas, 'Force/load (+):  ', (10, layout["Arm/Wrist Score"] + 2 * layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # only RULA has muscle use and force load for legs
                cv.putText(blank_canvas, 'Muscle use (+):  ', (10, layout["Neck/Trunk Score"] + 2 * layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if assess_method == "REBA":
                # only RULA has adjusted score for lower arm
                cv.putText(blank_canvas, 'Lower arm score: ', (10, layout["Lower arm"] + layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # REBA has coupling score
                cv.putText(blank_canvas, 'Coupling score (+):  ', (10, layout["Arm/Wrist Score"] + layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # REBA has activity score
                cv.putText(blank_canvas, 'Activity score (+):  ', (10, layout["adjusted_neck"] + layout["line_space"]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return blank_canvas

    # Function to update the dashboard
    def update_canvas(self, assess_method, blank_canvas, input_dict, input_var, score_dict):
        layout = self.layout
        # Clear the image for dynamic updates
        updated_canvas = blank_canvas.copy()
        right_rule, left_rule = 360, 490
        center_rule = int((right_rule + left_rule) / 2)
        right_color = (255, 192, 203)
        left_color = (0, 255, 255)
        white_color = (255, 255, 255)

        # First: the variables that are the same
        # upper arm angle
        cv.putText(updated_canvas, f'{input_dict["upper_arm_r"]}',
                   (right_rule, layout["Upper arm"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
        cv.putText(updated_canvas, f'{input_dict["upper_arm_l"]}',
                   (left_rule, layout["Upper arm"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

        # upper arm original score and adjusted
        cv.putText(updated_canvas, f'{score_dict["u_arm_score_r"]} => {score_dict["u_arm_adjusted_r"]}',
                   (right_rule, layout["Upper arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        cv.putText(updated_canvas, f'{score_dict["u_arm_score_l"]} => {score_dict["u_arm_adjusted_l"]}',
                   (left_rule, layout["Upper arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

        # lower arm angle
        cv.putText(updated_canvas, f'{input_dict["lower_arm_r"]}',
                   (right_rule, layout["Lower arm"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
        cv.putText(updated_canvas, f'{input_dict["lower_arm_l"]}',
                   (left_rule, layout["Lower arm"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

        # wrist score
        cv.putText(updated_canvas, f'{input_dict["wrist_r"]}',
                   (right_rule, layout["Wrist position"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
        cv.putText(updated_canvas, f'{input_dict["wrist_l"]}',
                   (left_rule, layout["Wrist position"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
        cv.putText(updated_canvas, f'{score_dict["wrist_score_r"]} => {score_dict["wrist_adjusted_r"]}',
                   (right_rule, layout["Wrist position"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        cv.putText(updated_canvas, f'{score_dict["wrist_score_l"]} => {score_dict["wrist_adjusted_l"]}',
                   (left_rule, layout["Wrist position"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

        # neck score
        cv.putText(updated_canvas, f"{input_dict['neck_angle']}",
                   (center_rule, layout["Neck position"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)
        cv.putText(updated_canvas, f"{score_dict['neck_score']} => {score_dict['neck_adjusted']}",
                   (center_rule, layout["Neck position"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, .5, white_color, 1)

        # trunk score
        cv.putText(updated_canvas, f"{input_dict['trunk_angle']}",
                   (center_rule, layout["Trunk position"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)
        cv.putText(updated_canvas, f"{score_dict['trunk_score']} => {score_dict['trunk_adjusted']}",
                   (center_rule, layout["Trunk position"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, .5, white_color, 1)

        # # force load for leg
        # cv.putText(updated_canvas, f'{input_var['force_load_leg']}',
        #         (center_rule, 830), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

        # Second: the variables that are the different between RULA/REBA
        if assess_method == "RULA":
                # upper arm original score and adjusted (only RULA has the adjusted scores)
                cv.putText(updated_canvas, f"{score_dict['lo_arm_score_r']} => {score_dict['lo_arm_adjusted_r']}",
                           (right_rule, layout["Lower arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{score_dict['lo_arm_score_l']} => {score_dict['lo_arm_adjusted_l']}",
                           (left_rule, layout["Lower arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

                # only RULA has wrist twist score
                cv.putText(updated_canvas, f"{input_var['wrist_twisted_mid_r']}",
                           (right_rule, layout["Wrist position"] + 2 * layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{input_var['wrist_twisted_mid_l']}",
                           (left_rule, layout["Wrist position"] + 2 * layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

                # arm/wrist score:
                cv.putText(updated_canvas, f"{score_dict['A_score_r']}",
                           (right_rule, layout["Arm/Wrist Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
                cv.putText(updated_canvas, f"{score_dict['A_score_l']}",
                           (left_rule, layout["Arm/Wrist Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

                # only RULA has muscle and force load scores:
                cv.putText(updated_canvas, f"{input_var['muscle_r']}",
                           (right_rule, layout["Arm/Wrist Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{input_var['muscle_l']}",
                           (left_rule, layout["Arm/Wrist Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
                cv.putText(updated_canvas, f"{input_var['force_load_r']}",
                           (right_rule, layout["Arm/Wrist Score"] + 2 * layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{input_var['force_load_l']}",
                           (left_rule, layout["Arm/Wrist Score"] + 2 * layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

                # arm/wrist score:
                cv.putText(updated_canvas, f"{score_dict['A_adjusted_r']}",
                           (right_rule, layout["adjusted arm score"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
                cv.putText(updated_canvas, f"{score_dict['A_adjusted_l']}",
                           (left_rule, layout["adjusted arm score"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

                # leg score
                cv.putText(updated_canvas, f"{score_dict['leg_score']}",
                           (center_rule, layout["Legs"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

                # neck/trunk/leg score:
                cv.putText(updated_canvas, f"{score_dict['B_score']}",
                           (center_rule, layout["Neck/Trunk Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

                # RULA has muscle scores for legs:
                cv.putText(updated_canvas, f"{input_var['force_load_leg']}",
                           (center_rule, layout["Neck/Trunk Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 1)
                cv.putText(updated_canvas, f"{input_var['muscle_leg']}",
                           (center_rule, layout["Neck/Trunk Score"] + 2 * layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 1)

                # neck/trunk/leg adjusted score:
                cv.putText(updated_canvas, f"{int(score_dict['B_adjusted'])}",
                           (center_rule, layout["adjusted_neck"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

        if assess_method == "REBA":
                # upper arm original score (only RULA has the adjusted scores)
                cv.putText(updated_canvas, f"{score_dict['lo_arm_score_r']}",
                           (right_rule, layout["Lower arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{score_dict['lo_arm_score_l']}",
                           (left_rule, layout["Lower arm"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

                # arm/wrist score:
                cv.putText(updated_canvas, f"{score_dict['B_score_r']}",
                           (right_rule, layout["Arm/Wrist Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
                cv.putText(updated_canvas, f"{score_dict['B_score_l']}",
                           (left_rule, layout["Arm/Wrist Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

                # REBA has coupling scores:
                cv.putText(updated_canvas, f"{input_var['coupling_r']}",
                           (right_rule, layout["Arm/Wrist Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                cv.putText(updated_canvas, f"{input_var['coupling_l']}",
                           (left_rule, layout["Arm/Wrist Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

                # arm/wrist adjusted score:
                cv.putText(updated_canvas, f"{score_dict['B_adjusted_r']}",
                           (right_rule, layout["adjusted arm score"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
                cv.putText(updated_canvas, f"{score_dict['B_adjusted_l']}",
                           (left_rule, layout["adjusted arm score"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

                # REBA has leg adjusted score
                cv.putText(updated_canvas, f"{score_dict['leg_score']} => {score_dict['leg_adjusted']}",
                           (center_rule, layout["Legs"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

                # neck/trunk/leg score:
                cv.putText(updated_canvas, f"{score_dict['A_score']}",
                           (center_rule, layout["Neck/Trunk Score"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

                # only REBA has force/load scores for legs:
                cv.putText(updated_canvas, f"{input_var['force_load_leg']}",
                           (center_rule, layout["Neck/Trunk Score"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 1)

                # neck/trunk/leg adjusted score:
                cv.putText(updated_canvas, f"{int(score_dict['A_adjusted'])}",
                           (center_rule, layout["adjusted_neck"]), cv.FONT_HERSHEY_SIMPLEX, 1, white_color, 2)

                # activity score:
                cv.putText(updated_canvas, f"{int(input_var['act_score'])}",
                           (center_rule, layout["adjusted_neck"] + layout["line_space"]), cv.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 1)

        cv.putText(updated_canvas, f"{score_dict['C_score_r']}",
                   (right_rule, layout["final_line"]), cv.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
        cv.putText(updated_canvas, f"{score_dict['C_score_l']}",
                   (left_rule, layout["final_line"]), cv.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)

        return updated_canvas
