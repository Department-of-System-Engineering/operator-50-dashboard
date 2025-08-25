from create_visuals import VisualElements as visual
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches


# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


HUMAN_SILHOUETTE_NAME = "human_silhouette.jpg"


class RiskMapElements:

    def __init__(self, score_dict: dict, assess_method: str):
        # self.image = "assets/human_silhouette.jpg"
        # self.save_path = "exports/risk_map.png"
        self.image = os.path.join(ASSETS_DIR, HUMAN_SILHOUETTE_NAME)
        self.score_dict = score_dict
        self.assess_method = assess_method
            
    def createMap(self, analysis_time):
        #Load the image as an array
        save_path = os.path.join(EXPORTS_DIR, f"risk_map-{analysis_time}.png")
        image_array = cv2.imread(self.image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) 
        height, width, _ = image_array.shape
        # Plot the image
        plt.figure(figsize=(6, 12))
        plt.imshow(image_array)
        plt.grid(True)  # Add grid for debugging
        plt.xticks(np.linspace(0, width, 10))
        plt.yticks(np.linspace(0, height, 10))


        body_scores =  {
            "neck" : self.score_dict["neck_adjusted"],
            "trunk" : self.score_dict["trunk_adjusted"],
            "leg" : self.score_dict["leg_score"],
            "upper_arm_l" : self.score_dict["u_arm_adjusted_l"],
            "upper_arm_r" : self.score_dict["u_arm_adjusted_r"],
            "lower_arm_l" : self.score_dict["lo_arm_score_l"],
            "lower_arm_r" : self.score_dict["lo_arm_score_r"],
            "wrist_l" : self.score_dict["wrist_adjusted_l"],
            "wrist_r" : self.score_dict["wrist_adjusted_r"]
        }

        body_points = {
            "neck": (width * 0.5, height * 0.1),
            "trunk": (width * 0.5, height * 0.4),
            "leg": (width * 0.5, height * 0.8),
            "upper_arm_l": (140, 100),
            "upper_arm_r": (220, 100),
            "lower_arm_l": (137, 140),
            "lower_arm_r": (224, 140),
            "wrist_l": (125, 170),
            "wrist_r": (235, 170),
        }

        # 🔹 Flip Y-coordinates to match Matplotlib
        for part in body_points:
            x, y = body_points[part]
            body_points[part] = (x,y)  # Matplotlib flips Y-axis
        
        # Overlay dots based on REBA/RULA scores
        visual_instance = visual()
        risk_colors = {}
        for part, (x,y) in body_points.items():
            score = body_scores[part]
            if score == 0:
                continue  # Skip plotting if score is 0

            if(self.assess_method == "REBA") :
                color = visual_instance.categorize_color_REBA(body_scores[part])
            elif(self.assess_method == "RULA") :
                color = visual_instance.categorize_color_RULA(body_scores[part])

            plt.scatter(x, y, color=np.array(color)/255.0, s=200, edgecolors="black")      

            # Store unique colors and their associated risk levels
            risk_colors[tuple(color)] = body_scores[part]

        # 🔹 Create a Legend for Risk Levels
        unique_colors = set(risk_colors.keys())  # Get unique color values
        legend_patches = []

        for color in unique_colors:
            risk_level = risk_colors[color]
            color_norm = np.array(color) / 255.0  # Normalize to (0,1) range for Matplotlib
            legend_patches.append(mpatches.Patch(color=color_norm, label=f"Score Level: {risk_level}"))

        # 🔹 Add legend to the figure
        plt.legend(handles=legend_patches, loc="upper right", fontsize=12, title="Score Levels")


         # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches="tight")   
        print(f"Risk map saved at: {save_path}")

        return save_path  # Return the path of the saved image

            
