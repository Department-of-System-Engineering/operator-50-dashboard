import numpy as np
import os


class GetREBAScores:

    def getNeckREBA(neck_angle: float, neck_twisted_or_side_bending: int):
        angle_limits = [[-100, 0], [0, 20], [20, 100]]
        neck_score = 0
        adjusted_score = neck_twisted_or_side_bending

        # scoring for neck score
        if neck_angle >= angle_limits[0][0] and neck_angle <= angle_limits[0][1]:
            neck_score += 2
            adjusted_score += 2
        elif neck_angle >= angle_limits[1][0] and neck_angle <= angle_limits[1][1]:
            neck_score += 1
            adjusted_score += 1
        elif neck_angle >= angle_limits[2][0] and neck_angle <= angle_limits[2][1]:
            neck_score += 2
            adjusted_score += 2

        return neck_score, adjusted_score

    def getTrunkREBA(trunk_angle: float, trunk_twisted: int, trunk_side_bending: int):
        angle_limits = [[0], [-20, 0], [0, 20], [20, 60], [-90, -20], [60, 100]]
        trunk_score = 0
        adjusted_score = trunk_twisted + trunk_side_bending
        # scoring for trunk score
        if trunk_angle == angle_limits[0][0]:
            trunk_score += 1
            adjusted_score += 1
        elif trunk_angle > angle_limits[1][0] and trunk_angle <= angle_limits[1][1]:
            trunk_score += 2
            adjusted_score += 2
        elif trunk_angle > angle_limits[2][0] and trunk_angle <= angle_limits[2][1]:
            trunk_score += 2
            adjusted_score += 2
        elif trunk_angle > angle_limits[3][0] and trunk_angle <= angle_limits[3][1]:
            trunk_score += 3
            adjusted_score += 3
        elif trunk_angle > angle_limits[4][0] and trunk_angle <= angle_limits[4][1]:
            trunk_score += 3
            adjusted_score += 3
        elif trunk_angle > angle_limits[5][0] and trunk_angle <= angle_limits[5][1]:
            trunk_score += 4
            adjusted_score += 4

        return trunk_score, adjusted_score

    def getLegsREBA(leg_angle: float, one_or_both_down: int):
        angle_limits = [30, 60]  
        leg_score = one_or_both_down
        adjusted_score = one_or_both_down

        # scoring for leg score
        if leg_angle > angle_limits[0] and leg_angle <= angle_limits[1]:
            adjusted_score += 1
        elif leg_angle > angle_limits[1] :
            adjusted_score += 2

        return leg_score, adjusted_score

    def getUpperArmREBA(upper_arm_angle: float, raised: int, abducted: int, supported_leaning: int):
        angle_limits = [[-20, 20], [-90, -20], [20, 45], [45, 90], [90, 180]]
        upper_arm_score = 0
        adjusted_score = raised + abducted + supported_leaning

        # scoring for upper arm score
        if upper_arm_angle > angle_limits[0][0] and upper_arm_angle <= angle_limits[0][1]:
            upper_arm_score += 1
            adjusted_score += 1
        elif upper_arm_angle > angle_limits[1][0] and upper_arm_angle <= angle_limits[1][1]:
            upper_arm_score += 2
            adjusted_score += 2
        elif upper_arm_angle > angle_limits[2][0] and upper_arm_angle <= angle_limits[2][1]:
            upper_arm_score += 2
            adjusted_score += 2
        elif upper_arm_angle > angle_limits[3][0] and upper_arm_angle <= angle_limits[3][1]:
            upper_arm_score += 3
            adjusted_score += 3
        elif upper_arm_angle > angle_limits[4][0] and upper_arm_angle <= angle_limits[4][1]:
            upper_arm_score += 4
            adjusted_score += 4

        return upper_arm_score, adjusted_score

    def getLowerArmREBA(lower_arm_angle: float):
        angle_limits = [[60, 100], [0, 60], [100, 180]]
        lower_arm_score = 0
        # scoring for any side
        if lower_arm_angle > angle_limits[0][0] and lower_arm_angle < angle_limits[0][1]:
            lower_arm_score += 1
        elif lower_arm_angle > angle_limits[1][0] and lower_arm_angle <= angle_limits[1][1]:
            lower_arm_score += 2
        elif lower_arm_angle > angle_limits[2][0] and lower_arm_angle <= angle_limits[2][1]:
            lower_arm_score += 2

        return lower_arm_score

    def getWristREBA(wrist_angle: float, wrist_midline_or_twisted: int):
        angle_limits = [[-15, 15], [15, 45], [-45, -15]]
        wrist_score = 0
        adjusted_score = wrist_midline_or_twisted

        # scoring for wrist score
        if wrist_angle > angle_limits[0][0] and wrist_angle <= angle_limits[0][1]:
            wrist_score += 1
            adjusted_score += 1
        elif wrist_angle > angle_limits[1][0] and wrist_angle <= angle_limits[1][1]:
            wrist_score += 2
            adjusted_score += 2
        elif wrist_angle > angle_limits[2][0] and wrist_angle <= angle_limits[2][1]:
            wrist_score += 2
            adjusted_score += 2

        return wrist_score, adjusted_score

    def get_Table_A_Score(trunk_score: float, neck_score: float, leg_score: float):
        scores = np.round([trunk_score, neck_score, leg_score])
        # print(scores)
        scores = scores.astype(int)

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "REBA_tables", "tableA.csv")
        # print("The table is in: ", file_path)
        table = np.genfromtxt(file_path, delimiter=',', dtype=int)

        # generating x-coordinate
        if scores[0] == 1:
            x = 0
        elif scores[0] == 2:
            x = 1
        elif scores[0] == 3:
            x = 2
        elif scores[0] == 4:
            x = 3
        elif scores[0] == 5:
            x = 4
        else:
            x = 0
            print("Error!x-value is invalid!(Table-A)")

        # generating y-coordinate
        if scores[1] == 1 and scores[2] == 1:
            y = 0
        elif scores[1] == 1 and scores[2] == 2:
            y = 1
        elif scores[1] == 1 and scores[2] == 3:
            y = 2
        elif scores[1] == 1 and scores[2] == 4:
            y = 3
        elif scores[1] == 2 and scores[2] == 1:
            y = 4
        elif scores[1] == 2 and scores[2] == 2:
            y = 5
        elif scores[1] == 2 and scores[2] == 3:
            y = 6
        elif scores[1] == 2 and scores[2] == 4:
            y = 7
        elif scores[1] == 3 and scores[2] == 1:
            y = 8
        elif scores[1] == 3 and scores[2] == 2:
            y = 9
        elif scores[1] == 3 and scores[2] == 3:
            y = 10
        elif scores[1] == 3 and scores[2] == 4:
            y = 11
        else:
            y = 0
            print("Error!y-value is invalid!(Table-A)")

        # print("Score Table A:")
        # print("x: ", str(x), "; y: ", str(y))
        # print(str(table[x, y]))

        return table[x, y]

    def get_Table_B_Score(upper_arm_score: float, lower_arm_score: float, wrist_score: float):
        scores = np.round([upper_arm_score, lower_arm_score, wrist_score])
        scores = scores.astype(int)

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "REBA_tables", "tableB.csv")
        table = np.genfromtxt(file_path, delimiter=',', dtype=int)

        # generating x-coordinates
        if scores[0] == 1:
            x = 0
        elif scores[0] == 2:
            x = 1
        elif scores[0] == 3:
            x = 2
        elif scores[0] == 4:
            x = 3
        elif scores[0] == 5:
            x = 4
        elif scores[0] == 6:
            x = 5
        else:
            x = 0
            print("Error!x-value is invalid!(Table-B)")

        # generating y-coordinates
        if scores[1] == 1 and scores[2] == 1:
            y = 0
        elif scores[1] == 1 and scores[2] == 2:
            y = 1
        elif scores[1] == 1 and scores[2] == 3:
            y = 2
        elif scores[1] == 2 and scores[2] == 1:
            y = 3
        elif scores[1] == 2 and scores[2] == 2:
            y = 4
        elif scores[1] == 2 and scores[2] == 3:
            y = 5
        else:
            y = 0
            print("Error!y-value is invalid!(Table-B)")

        # print("Score Table B:")
        # print("x: " + str(x) + "; y: " + str(y))
        # print(str(table[x, y]))

        return table[x, y]

    def get_Table_C_Score(table_A_score: float, table_B_score: float):
        """
        generates the final Rula Score based, on values of upper_table and lower_table
        """
        scores = np.round([table_A_score, table_B_score])
        scores = scores.astype(int)

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "REBA_tables", "tableC.csv")
        # print("The C table is in: ", file_path)
        table = np.genfromtxt(file_path, delimiter=',', dtype=int)

        # generating x-coordinate
        if scores[0] == 1:
            x = 0
        elif scores[0] == 2:
            x = 1
        elif scores[0] == 3:
            x = 2
        elif scores[0] == 4:
            x = 3
        elif scores[0] == 5:
            x = 4
        elif scores[0] == 6:
            x = 5
        elif scores[0] == 7:
            x = 6
        elif scores[0] == 8:
            x = 7
        elif scores[0] == 9:
            x = 8
        elif scores[0] == 10:
            x = 9
        elif scores[0] == 11:
            x = 10
        elif scores[0] == 12:
            x = 11
        else:
            x = 0
            print("Error!x-value is invalid!(Table-C)")

        # generating y-coordinate
        if scores[1] == 1:
            y = 0
        elif scores[1] == 2:
            y = 1
        elif scores[1] == 3:
            y = 2
        elif scores[1] == 4:
            y = 3
        elif scores[1] == 5:
            y = 4
        elif scores[1] == 6:
            y = 5
        elif scores[1] == 7:
            y = 6
        elif scores[1] == 8:
            y = 7
        elif scores[1] == 9:
            y = 8
        elif scores[1] == 10:
            y = 9
        elif scores[1] == 11:
            y = 10
        elif scores[1] == 12:
            y = 11
        else:
            y = 0
            print("Error!y-value is invalid!(Table-C)")

        # print("Score Table C:")
        # print("x: " + str(x) + "; y: " + str(y))
        # print("==============")
        # print("FINAL SCORE:")
        # print("\t" + str(table[x, y]))
        # print("==============")

        return table[x, y]
