def arm_load_force_calculator(load_weight, load_variation):
        if load_weight < 2 :
                return 0
        elif load_weight >= 2 and load_weight <= 10 and load_variation == "Intermittent":
                return 1
        elif load_weight >= 2 and load_weight <= 10 and load_variation == "Static or repeated":
                return 2
        elif load_weight > 10 :
                return 3

def leg_load_force_calculator(load_weight):
        if load_weight < 5 :
                return 0
        elif load_weight >= 5 and load_weight <= 10:
                return 1
        elif load_weight > 10 :
                return 2   

def upper_body_load_force_calculator(load_weight, load_variation):
        if load_weight < 2 :
                return 0
        elif load_weight >= 2 and load_weight <= 10 and load_variation == "Intermittent":
                return 1
        elif load_weight >= 2 and load_weight <= 10 and load_variation == "Static or repeated":
                return 2
        elif load_weight > 10 :
                return 3

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def convert_additional_parameters(wrist_twist_endline_l, wrist_twist_endline_r, coupling_score_l, coupling_score_r, load_weight_arms_l, load_weight_arms_r , load_weight_legs, load_weight_upper_body,
                                  load_variation_arms_l,load_variation_arms_r, load_variation_upper_body):
                # Map selections to single integer values
                action_map = {
                "None": 0,
                "Twist": 1,
                "End": 2
                }

                # Mapping descriptions to scores
                coupling_map = {
                'Well fitting Handle and mid range power grip': 0,
                'Acceptable but not ideal hand hold or coupling': 1,
                'Acceptable with another body part': 1,
                'Hand hold not acceptable but possible': 2,
                'No handles, awkward, unsafe with any body part': 3
                }


                action_val_l = action_map.get(wrist_twist_endline_l, 0)
                action_val_r = action_map.get(wrist_twist_endline_r, 0)
                coupling_val_l = coupling_map.get(coupling_score_l, 0)
                coupling_val_r = coupling_map.get(coupling_score_r, 0)

                load_force_arms_l = arm_load_force_calculator(safe_int(load_weight_arms_l), safe_int(load_variation_arms_l))
                load_force_arms_r = arm_load_force_calculator(safe_int(load_weight_arms_r), safe_int(load_variation_arms_r))
                load_force_legs = leg_load_force_calculator(safe_int(load_weight_legs))
                load_force_upper_body = upper_body_load_force_calculator(safe_int(load_weight_upper_body), safe_int(load_variation_upper_body))

                
                return action_val_l, action_val_r, coupling_val_l, coupling_val_r, load_force_arms_l, load_force_arms_r , load_force_legs, load_force_upper_body

                
                