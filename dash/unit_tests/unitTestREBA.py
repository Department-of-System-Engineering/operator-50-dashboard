import unittest
import calculate_REBA as reba

class TestRULACalculator(unittest.TestCase) :
    def setUp(self):
        self.reba_score = reba

    def test_single_frame(self) :
        #TODO : makeing the unit test for the given frame
        # Define input parameters for a known scenario
        right_upper_arm = 4
        shoulder_raised = 1
        arm_abducted = 0
        arm_supported = 0

        right_lower_arm = 2

        right_wrist = 1
        wrist_twist_mid_or_end_of_range = 1

        static_A = 1
        load_A = 0

        neck = 2
        neck_twisted = 0
        neck_side_bending = 0

        trunk = 3
        trunk_twisted = 1
        trunk_side_bending = 0

        leg = 2
        leg_down_raised = 1

        static_B = 1
        load_B = 0

        supp = wrist_twist_mid_or_end_of_range
        #calculating the scores
        right_upper_arm_score = right_upper_arm + shoulder_raised + arm_abducted + arm_supported
        right_lower_arm_score = right_lower_arm 
        right_wrist_score = right_wrist + wrist_twist_mid_or_end_of_range 
        neck_score = neck + neck_twisted + neck_side_bending
        trunk_score = trunk + trunk_twisted + trunk_side_bending
        legs_score = leg + leg_down_raised
        # Expected RULA score for these parameters
        posture_A = 7
        posture_B = 6
        posture_C = 9

        # Calculate the RULA score using the calculator
        actual_score_A = self.reba_score.GetREBAScores.get_Table_A_Score(trunk_score , legs_score , neck_score)
        actual_score_B = self.reba_score.GetREBAScores.get_Table_B_Score(right_upper_arm_score , right_lower_arm_score , right_wrist_score) 
        actual_score_C = self.reba_score.GetREBAScores.get_Table_C_Score(actual_score_A ,actual_score_B)
        
        # Assert that the actual score matches the expected score
        self.assertEqual(actual_score_A , posture_A)
        self.assertEqual(actual_score_B , posture_B)
        self.assertEqual(actual_score_C , posture_C)  #Ran 1 test in 0.004s OK


if __name__ == '__main__':
    unittest.main()