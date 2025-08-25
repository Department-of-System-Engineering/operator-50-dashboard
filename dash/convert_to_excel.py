class ExcelElements:

    def getNewElementREBA(frame_number: float, orig_frame_number: float, timestamp : str,
                          input_dict, input_var, score_dict):
        return {
            'frame_number': frame_number, 'orig_frame_number': orig_frame_number, 'timestamp': timestamp,
            'neck_angle': input_dict['neck_angle'], 'trunk_angle': input_dict['trunk_angle'],
            'leg_score': score_dict['leg_score'],
            'upper_arm_l': input_dict['upper_arm_l'], 'upper_arm_r': input_dict['upper_arm_r'],
            'lower_arm_l': input_dict['lower_arm_l'], 'lower_arm_r': input_dict['lower_arm_r'],
            'wrist_l': input_dict['wrist_l'], 'wrist_r': input_dict['wrist_r'],
            'wrist_adjusted_l': score_dict['wrist_adjusted_l'],
            'wrist_adjusted_r': score_dict['wrist_adjusted_r'],

            'neck_adjusted': score_dict['neck_adjusted'],
            'trunk_adjusted': score_dict['trunk_adjusted'],

            'u_arm_adjusted_l': score_dict['u_arm_adjusted_l'],
            'u_arm_adjusted_r': score_dict['u_arm_adjusted_r'],

            'lower_arm_score_l': score_dict['lo_arm_score_l'],
            'lower_arm_score_r': score_dict['lo_arm_score_r'],

            'A_adjusted': score_dict['A_adjusted'],
            'B_adjusted_l': score_dict['B_adjusted_l'], 'B_adjusted_r': score_dict['B_adjusted_r'],
            'C_score_l': score_dict['C_score_l'], 'C_score_r': score_dict['C_score_r']
            }

    def getNewElementRULA(frame_number: float, orig_frame_number: float, timestamp: str,
                          input_dict, input_var, score_dict):

        return {
            'frame_number': frame_number, 'orig_frame_number': orig_frame_number, 'timestamp': timestamp,
            'neck_angle': input_dict['neck_angle'], 'trunk_angle': input_dict['trunk_angle'],
            'leg_score': score_dict['leg_score'],
            'upper_arm_l': input_dict['upper_arm_l'], 'upper_arm_r': input_dict['upper_arm_r'],
            'lower_arm_l': input_dict['lower_arm_l'], 'lower_arm_r': input_dict['lower_arm_r'],
            'wrist_l': input_dict['wrist_l'], 'wrist_r': input_dict['wrist_r'],
            'wrist_adjusted_l': score_dict['wrist_adjusted_l'],
            'wrist_adjusted_r': score_dict['wrist_adjusted_r'],

            'neck_adjusted': score_dict['neck_adjusted'],
            'trunk_adjusted': score_dict['trunk_adjusted'],
            'neck_adjusted': score_dict['neck_adjusted'],
            'trunk_adjusted': score_dict['trunk_adjusted'],

            'u_arm_adjusted_l': score_dict['u_arm_adjusted_l'],
            'u_arm_adjusted_r': score_dict['u_arm_adjusted_r'],
            'lo_arm_adjusted_l': score_dict['lo_arm_adjusted_l'],
            'lo_arm_adjusted_r': score_dict['lo_arm_adjusted_r'],

            'A_adjusted_l': score_dict['A_adjusted_l'], 'A_adjusted_r': score_dict['A_adjusted_r'],
            'B_adjusted': score_dict['B_adjusted'],
            'C_score_l': score_dict['C_score_l'], 'C_score_r': score_dict['C_score_r']
        }
