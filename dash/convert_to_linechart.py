
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")


class LineChartElements:

    def __init__(self, assess_method, analysis_time):
        self.assess_method = assess_method
        self.analysis_time = analysis_time

    def categorize_color(self):
        if self.assess_method == 'REBA':
            self.color_categories = {
                    "Negligible": {"range": (0, 1), "color": "cyan"},
                    "Low": {"range": (2, 3), "color": "green"},
                    "Medium": {"range": (4, 7), "color": "yellow"},
                    "High": {"range": (8, 10), "color": "orange"},
                    "Very high": {"range": (11, 12), "color": "red"},
                }
        elif self.assess_method == 'RULA':
            self.color_categories = {
                "Negligible": {"range": (0, 2), "color": "green"},
                "Low": {"range": (3, 4), "color": "yellow"},
                "Medium": {"range": (5, 6), "color": "orange"},
                "High": {"range": (7, 8), "color": "red"},
            }

    def visualization_chart(self, df, side="left"):
        self.categorize_color()

        fig, ax = plt.subplots(figsize=(14, 7), dpi=200)

        if side == "left":
            ax.plot(df.index + 1, df["Left_Risk"], marker='o', color='blue', label="Left_Risk")
            ax.set_title("Risk Score on Left Side", fontsize=24, pad=10)
        elif side == "right":
            ax.plot(df.index + 1, df["Right_Risk"], marker='o', color='orange', label="Right_Risk")
            ax.set_title("Risk Score on Right Side", fontsize=24, pad=10)
        else:
            raise ValueError("side must be 'left' or 'right'")

        ax.legend(loc="upper left")
        ax.grid(axis="both", linestyle="--", alpha=0.5)

        if self.assess_method == "REBA":
            ax.set_ylim(0, 12)
        if self.assess_method == "RULA":
            ax.set_ylim(0, 7)

        upper_limits = [r["range"][1] for r in self.color_categories.values()]
        for category, details in self.color_categories.items():
            y_min, y_max = details["range"]
            ax.axhspan(y_min - 0.5, y_max + 0.5, color=details["color"], alpha=0.3, label=f'{category} Range')
        ax.set_yticks(upper_limits)
        ax.set_yticklabels(self.color_categories.keys(), fontsize=20)
        # Set x-axis ticks from 1 to length of df
        x_ticks = range(1, len(df) + 1)
        ax.set_xticks(x_ticks)

        plt.tight_layout()

        # Save the chart to file
        # chart_file = f"line_chart_{side}-{self.analysis_time}.png"  # cant be found to display
        chart_file = f"line_chart_{side}.png"

        chart_img_filepath = os.path.join(EXPORTS_DIR, chart_file)
        fig.savefig(chart_img_filepath)
        plt.close(fig)

        print(f"Line chart saved as: {chart_img_filepath}")
