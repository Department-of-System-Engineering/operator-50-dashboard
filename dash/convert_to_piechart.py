import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from create_visuals import VisualElements as visual


# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")


class PieChart:

    def __init__(self, assess_method: str):
        self.assess_method = assess_method

    def create_piechart(self, df, analysis_time):
        # Apply categorization to both Left and Right scores
        visual_instance = visual()
        if self.assess_method == "REBA":
            df['Left_Risk'] = df['C_score_l'].apply(visual_instance.categorize_risk_REBA)
            df['Right_Risk'] = df['C_score_r'].apply(visual_instance.categorize_risk_REBA)
        if self.assess_method == "RULA":
            df['Left_Risk'] = df['C_score_l'].apply(visual_instance.categorize_risk_RULA)
            df['Right_Risk'] = df['C_score_r'].apply(visual_instance.categorize_risk_RULA)

        # Count the frequencies for each category
        left_counts = df['Left_Risk'].value_counts(normalize=True) * 100
        right_counts = df['Right_Risk'].value_counts(normalize=True) * 100
        # left_counts = left_counts.sort_index()  # Ensure consistent order
        # right_counts = right_counts.sort_index()  # Ensure consistent order

        # Ensure both series have the same categories (fill missing with 0)
        if self.assess_method == "REBA":
            categories = ['Negligible Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
            # Define categories and corresponding colors
            color_map = {
                "Negligible Risk": "cyan",
                "Low Risk": "limegreen",  
                "Medium Risk": "yellow",  
                "High Risk": "orange",  
                "Very High Risk": "red"      
            }

        if self.assess_method == "RULA":
            categories = ['Acceptable posture', 'Further investigation, change maybe needed', 'Further investigation, change soon', 'Investigate and implement change']
            color_map = {
                "Acceptable posture": "cyan",   
                "Further investigation, change maybe needed": "limegreen",  
                "Further investigation, change soon": "yellow",  
                "Investigate and implement change": "red"      
            }

        left_counts = left_counts.reindex(categories, fill_value=0)
        right_counts = right_counts.reindex(categories, fill_value=0)
        # Filter out categories with zero count (if any)
        left_counts = left_counts[left_counts > 0]
        right_counts = right_counts[right_counts > 0]
        # Dynamically determine colors for the present categories
        left_colors = [color_map[category] for category in left_counts.index if category in color_map]
        right_colors = [color_map[category] for category in right_counts.index if category in color_map]

        # Create side-by-side pie charts (will be saved as images, not displayed)
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Ensure the pie charts remain circular
        for ax in axs:
            ax.set_box_aspect(1)

        # Left Side Pie Chart
        axs[0].pie(left_counts.values.tolist(), labels=left_counts.index.tolist(), autopct='%1.1f%%', startangle=90, colors=left_colors, textprops = {'fontsize' : 16})
        axs[0].set_title('Left Side Score Distribution', fontsize=20, pad=20)  # Adjusted spacing with pad

        # Right Side Pie Chart
        axs[1].pie(right_counts.values.tolist(), labels=right_counts.index.tolist(), autopct='%1.1f%%', startangle=90, colors=right_colors , textprops = {'fontsize' : 16})
        axs[1].set_title('Right Side Score Distribution', fontsize=20, pad=20)  # Adjusted spacing with pad

        # Create legends for each pie chart
        legend_left = [mpatches.Patch(color=left_colors[i], label=label) for i, label in enumerate(left_counts.index)]
        legend_right = [mpatches.Patch(color=right_colors[i], label=label) for i, label in enumerate(right_counts.index)]

        axs[0].legend(handles=legend_left, title="Risk categories", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
        axs[1].legend(handles=legend_right, title="Risk categories", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)

        # Adjust layout to prevent titles from being cut off
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)  # Increase top margin
        plt.tight_layout()
        # plt.show(block=False)

        # Save the pie charts as image files
        pie_chart_file = f"pie_chart-{analysis_time}.png"
        # Save the pie charts to files
        img_filepath = os.path.join(EXPORTS_DIR, pie_chart_file)
        fig.savefig(img_filepath)
        # Close the figure to free memory
        plt.close(fig)
        # Inform user that data has been saved
        print(f"Pie chart saved as: {img_filepath}")

        return img_filepath
