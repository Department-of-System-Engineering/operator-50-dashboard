from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib import colors 
import os
from PIL import Image

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TRUETYPE_FONT_NAME = "Helvetica.ttf"


class PdfElements:

    def __init__(self, filename: str, documentTitle: str, opdata: dict, piechart: str, riskmap: str, maximages: dict, criticalidx: dict, criticalimages: dict):
        self.filename = filename
        self.documentTitle = documentTitle
        self.opdata = opdata
        self.piechart = piechart
        self.riskmap = riskmap
        self.maximages = maximages
        self.criticalidx = criticalidx
        self.criticalimages = criticalimages

    def createPdfFile(self):
        pdf = canvas.Canvas(self.filename)
        pdf.setTitle(self.documentTitle)

        # Register the font
        font_path = os.path.join(ASSETS_DIR, TRUETYPE_FONT_NAME)
        pdfmetrics.registerFont(TTFont('Helvetica', font_path))

        # Add Title
        pdf.setFont('Helvetica', 18)
        pdf.drawCentredString(300, 800, "Video-based Ergonomic Report")

        # Add headers
        pdf.setFont('Helvetica', 12)
        pdf.drawString(50, 760, f"Date: {self.opdata.get('date', '_________')}")
        pdf.drawString(50, 740, f"Assessor: {self.opdata.get('assessor', '_________')}")
        pdf.drawString(300, 760, f"Type of assessment: {self.opdata.get('assessment', '_________')}")
        pdf.drawString(300, 740, f"Workstation: {self.opdata.get('workstation', '_________')}")
        pdf.drawString(50, 720, f"Task: {self.opdata.get('task', '_________')}")
        pdf.drawString(300, 720, f"Video length: {self.opdata.get('video_length', '_________')} frames")
        pdf.drawString(50, 700, f"Working duration: {self.opdata.get('working_duration', '_________')}%")

        # Add Pie Chart
        # Open the pie chart image to get original dimensions
        pdf.drawString(50, 650, "I. Overall")
        img = Image.open(self.piechart)
        orig_width, orig_height = img.size  # Get original width and height

        # Define desired width (adjust as needed)
        target_width = 400  

        # Maintain aspect ratio
        aspect_ratio = orig_height / orig_width  
        target_height = target_width * aspect_ratio  # Compute height based on aspect ratio

        # Draw image while maintaining aspect ratio
        pdf.drawInlineImage(self.piechart, 100, 450, width=target_width, height=target_height)

        # Add Risk Map
        pdf.drawString(50, 470, "II. Ergonomic Risk Map")
        pdf.drawInlineImage(self.riskmap, 50, 250, width=200, height=200)

        # Add critical movements
        for key, image_path in self.maximages.items():  # Loop through key-value pairs
            if os.path.exists(image_path):
                self.placeMaxImages(image_path, pdf, key)
                
            else:
                print(f"Warning: Image not found -> {image_path}")

        # Add critical movements section
        pdf.drawString(50, 220, "III. Top Critical Movements")
        
        for key, image_path in self.criticalimages.items():
              if os.path.exists(image_path):
                    self.placeCriticalImages(image_path, pdf, key)
                    
              else:
                    print(f"Warning: Image not found -> {image_path}")

        # Add score table (manual text placement for now)
        pdf.drawString(50, 30, f"Highest score: {self.opdata.get('highest_score', '_________')}")
        pdf.drawString(250, 30, f"MSD risk level: {self.opdata.get('msd_risk_level', '_________')}")
        pdf.drawString(400, 30, f"Action required: {self.opdata.get('action_required', '_________')}")

        # Save PDF
        pdf.save()
        print(f"Pdf report saved in: {self.filename}")

    def placeMaxImages(self, image_path: str, pdf: canvas , key : str):
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            return

        img = Image.open(image_path)

        # Get original dimensions
        orig_width, orig_height = img.size 

        # Set fixed height and calculate width proportionally
        fixed_height = 75  # Adjust this as needed
        aspect_ratio = orig_width / orig_height
        new_width = int(fixed_height * aspect_ratio)  # Maintain aspect ratio

        # Resizing the image 
        img = img.resize((new_width ,fixed_height), Image.Resampling.LANCZOS) 

        # x coordinates
        left_x = 280
        right_x = 450

        # Placement logic for images
        if key == 'neck':
                # Left-side image
                pdf.drawInlineImage(img, left_x, 400, width=new_width, height=fixed_height)
                pdf.drawString(left_x + 70, 400 + 60, "Neck")
                pdf.drawString(left_x + 70, 400 + 40, "Frame: " + f"{self.criticalidx['max_neck'][0]}")
                pdf.drawString(left_x + 70, 400 + 20, "Score: " + f"{self.criticalidx['max_neck'][1]}")
        elif key == 'trunk':
        # Right-side: Top row, first column
                pdf.drawInlineImage(img, left_x, 310, width=new_width, height=fixed_height)
                pdf.drawString(left_x + 70, 310 + 60, "Trunk")
                pdf.drawString(left_x + 70, 310 + 40, "Frame: " + f"{self.criticalidx['max_trunk'][0]}")
                pdf.drawString(left_x + 70, 310 + 20, "Score: " + f"{self.criticalidx['max_trunk'][1]}")
        elif key == 'leg':
        # Right-side: Top row, second column
                pdf.drawInlineImage(img, left_x , 220, width=new_width, height=fixed_height)
                pdf.drawString(left_x  + 70, 220 + 60, "Leg")
                pdf.drawString(left_x  + 70, 220 + 40, "Frame: " + f"{self.criticalidx['max_legs'][0]}")
                pdf.drawString(left_x  + 70, 220 + 20, "Score: " + f"{self.criticalidx['max_legs'][1]}")
        elif key == 'u_arm':
        # Right-side: Second row, first column
                pdf.drawInlineImage(img, right_x, 400, width=new_width, height=fixed_height)
                pdf.drawString(right_x + 70, 400 + 60, "Upper arm")
                pdf.drawString(right_x + 70, 400 + 40, "Frame: " + f"{self.criticalidx['max_up_arm'][0]}")
                pdf.drawString(right_x + 70, 400 + 20, "Score: " + f"{self.criticalidx['max_up_arm'][1]}")
        elif key == 'l_arm':
                # Right-side: Second row, second column
                pdf.drawInlineImage(img, right_x, 310, width=new_width, height=fixed_height)
                pdf.drawString(right_x + 70, 310 + 60, "Lower arm")
                pdf.drawString(right_x + 70, 310 + 40, "Frame: " + f"{self.criticalidx['max_lo_arm'][0]}")
                pdf.drawString(right_x + 70, 310 + 20, "Score: " + f"{self.criticalidx['max_lo_arm'][1]}")
        elif key == 'wrist':
                # Right-side: Third row, first column
                pdf.drawInlineImage(img, right_x, 220, width=new_width, height=fixed_height)
                pdf.drawString(right_x + 70, 220 + 60, "Wrist")
                pdf.drawString(right_x + 70, 220 + 40, "Frame: " + f"{self.criticalidx['max_wrist'][0]}")
                pdf.drawString(right_x + 70, 220 + 20, "Score: " + f"{self.criticalidx['max_wrist'][1]}")

    def placeCriticalImages(self, image_path : str, pdf : canvas, key: str):               
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            return

        img = Image.open(image_path)
        
        # Get original dimensions
        orig_width, orig_height = img.size 

        # Set fixed height and calculate width proportionally
        fixed_height = 150  # Adjust this as needed
        aspect_ratio = orig_width / orig_height
        new_width = int(fixed_height * aspect_ratio)  # Maintain aspect ratio

        # Resizing the image 
        img = img.resize((new_width ,fixed_height), Image.Resampling.LANCZOS) 

        if key == 'criticalleft' :
             pdf.drawInlineImage(img, 50, 50, width=new_width, height=fixed_height) 
        elif key == 'criticalright' :
             pdf.drawInlineImage(img, 350, 50, width=new_width, height=fixed_height) 