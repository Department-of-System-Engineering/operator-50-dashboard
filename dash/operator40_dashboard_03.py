import dash
import cv2
import os
import base64
import threading
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from flask import Flask
from flask import send_from_directory
from hand_heatmap_from_video import hand_heatmap_two_color
from convert_video_file import reencode_video
from process_skeleton_video import process_video
import subprocess
# from process_skeleton_video import process_video_reba

# Define the Flask server
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server, suppress_callback_exceptions=True)

# Shared state to track processing status
processing_status = {"is_processing": False, "message": "", "result": None}

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "imports" and "exports" folder
IMPORTS_DIR = os.path.join(BASE_DIR, "imports")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

# Ensure the folders exists
if not os.path.exists(IMPORTS_DIR):
    os.makedirs(IMPORTS_DIR)
if not os.path.exists(EXPORTS_DIR):
    os.makedirs(EXPORTS_DIR)

# Create a directory for uploads and exports if they don't exist
# os.makedirs('imports', exist_ok=True)
# os.makedirs('exports', exist_ok=True)

# Define styles for the sidebar and content
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# Function to encode images for the sidebar (if needed)
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


# Sidebar layout
sidebar = html.Div(
    [
        html.H5("Video Analysis", className="display-5"),
        html.Hr(),

        # PE logo at the current position
        html.Div(
            dcc.Link(
                html.Img(
                    src='/assets/PE_logo_blue.png',
                    height="auto",
                    width=200,
                    style={'marginTop': '20px', 'marginBottom': '20px'}
                ),
                href='https://uni-pannon.hu/',  # Link for the PE logo
                target="_blank"
            ),
            style={'textAlign': 'center', 'marginTop': '10px'}  # Center the MK logo
        ),

        # Other sidebar items
        dbc.Nav(
            [
                dbc.NavLink("Ergonomy Assessment", href="/video_analysis", active="exact"),
                dbc.NavLink("Activity Assessment", href="/activity_assessment", active="exact"),
                dbc.NavLink("Work Instructions", href="/work-instructions", active="exact"),
                dbc.NavLink("Chatbot", href="/chatbot", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),

        # MK logo at the bottom of the sidebar
        html.Div(
            dcc.Link(
                html.Img(
                    src='/assets/MK_logo_blue.png',
                    height="auto",
                    width=120,
                    style={'marginTop': '20px', 'marginBottom': '20px'}
                ),
                href='https://mk.uni-pannon.hu/',  # Link for the MK logo
                target="_blank"
            ),
            style={'textAlign': 'center', 'marginTop': '50px'}  # Center the MK logo
        ),

    ],
    style=SIDEBAR_STYLE,
)

progress_modal = dbc.Modal(
    [
        dbc.ModalBody(id="progress-message"),
    ],
    id="progress_modal",
    is_open=False,  # Initially hidden
    backdrop="static",  # Prevent closing by clicking outside
    centered=True,
)

progress_interval = dcc.Interval(
    id="progress_interval",
    interval=1000,  # Check every 1 second
    n_intervals=0,  # Start at 0
    disabled=True  # Initially disabled
)

# Main content area
content = html.Div(id="page-content", style=CONTENT_STYLE)

# App layout with sidebar and content
app.layout = html.Div([dcc.Location(id="url"), sidebar, content, progress_modal,progress_interval], style={'fontFamily': 'Roboto, sans-serif'})


# Callback to manage page navigation
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):

    # Redirect from root "/" to "/video_analysis"
    if pathname == "/" or pathname is None:
        return dcc.Location(href="/video_analysis", id="redirect-video-analysis")

    if pathname == "/video_analysis":
        return html.Div([
            html.H2("Upload and edit a video file", style={'textAlign': 'center'}),

            # Video upload area and existing elements
            dcc.Upload(
                id='upload-video',
                children=html.Div([
                    'Drag and drop or ',
                    html.A('select a video file...')
                ]),
                style={
                    'width': '30%',
                    'height': '50px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px auto',
                },
                multiple=False  # Set to False since we're uploading one video at a time
            ),

            dcc.Loading(id="loading", type="circle", children=[html.Div(id='loading-output')]),
            html.Div(id='error_message', style={'color': 'red', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Video(id='video-display', controls=True, style={'width': '100px','height' : '100px', 'display': 'none', 'margin': '0 auto'}),
            html.Div(
                dcc.RangeSlider(
                    id='time_slider',
                    min=0,
                    max=100,
                    value=[0, 15],
                    marks={}  # Initialize with empty marks, which will be updated dynamically
                ),
                id='slider-container',
                style={'width': '75%', 'margin': '0 auto', 'display': 'none'}
            ),

            html.Div(id='controls_container', style={'display': 'none'}, children=[

                html.Div([
                    dbc.Label('Time from: ', style={'marginTop': '15px', 'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                    dcc.Input(
                        id='time-from-input', type='number', value=0,
                        min=0, max=100, step=0.01, style={'width': '50px', 'textAlign': 'center', 'marginRight': '30px'}
                    ),
                    dbc.Label('Time to: ', style={'marginTop': '15px', 'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                    dcc.Input(
                        id='time-to-input', type='number', value=15,
                        min=0, max=100, step=0.01, style={'width': '50px', 'textAlign': 'center'}
                    )
                ], style={'textAlign': 'center'}),

                html.Div([
                    dbc.Label('Total Duration:', style={'marginTop': '15px', 'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.Div(id='video-duration', style={'margin': '10px 0', 'textAlign': 'center'}),
                ],
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'justifyContent': 'center'}),

                html.Div([
                    dbc.Label('Selected Time Range:', style={'marginTop': '15px', 'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.Div(id='selected_times', style={'margin': '10px 0', 'textAlign': 'center'}),
                ],
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'justifyContent': 'center'}),

                # Metadata info to save in SQL ... LATER
                html.H5(
                    "Related metadata for the video snippet:",
                    style={'marginTop': '10px', 'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '20px', 'color': 'blue'}
                ),
                html.Div(
                    [
                        dbc.Label('Camera position: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        dcc.Dropdown(
                            id='camera_position_input',
                            options=[
                                {'label': 'Top View', 'value': 'top-view'},
                                {'label': 'Side View', 'value': 'side-view'},
                                {'label': 'Back View', 'value': 'back-view'}
                            ],
                            placeholder='Select a camera position',
                            style={'width': '220px', 'marginRight': '30px'}
                        ),
                        html.Div(id='output-message'),
                        # dbc.Label('Activity type: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        # dcc.Input(
                        #     id='activity-type-input', type='text', value='',
                        #     style={'width': '150px', 'marginRight': '30px'}
                        # ),
                        # dbc.Label('Job location: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        # dcc.Input(
                        #     id='job-location-input', type='text', value='',
                        #     style={'width': '150px', 'marginRight': '30px'}
                        # ),
                        # dbc.Label('Workstation: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        # dcc.Input(
                        #     id='workstation-input', type='text', value='',
                        #     style={'width': '150px', 'marginRight': '30px'}
                        # ),
                        # dbc.Label('Operator skill level: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        # dcc.Input(
                        #     id='operator-skill-level-input', type='text', value='',
                        #     style={'width': '150px'}
                        # ),
                        # dbc.Label('Operator weight: ', style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        # dcc.Input(
                        #     id='operator-weight-input', type='text', value='',
                        #     style={'width': '150px'}
                        # ),
                    ],
                    style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '15px'}
                ),
                # Metadata info to save in SQL ... LATER

                html.H5("Analyse the video snippet and the related metadata:", style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '22px', 'color': 'darkblue'}),
                html.Div([
                    # html.Button('Analyse Video', id='export-button', n_clicks=0)
                    html.Button(
                        'Hand heatmap',
                        id='export-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#4CAF50',  # Green background
                            'color': 'white',  # White text
                            'border': 'none',  # No border
                            'padding': '15px 32px',  # Padding for better spacing
                            'textAlign': 'center',  # Center align the text
                            'textDecoration': 'none',  # No underline
                            'display': 'inline-block',  # Inline block
                            'fontSize': '18px',  # Font size
                            'borderRadius': '12px',  # Rounded corners
                            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
                            'cursor': 'pointer',  # Pointer cursor on hover
                            'transition': 'all 0.3s ease',  # Smooth transition for hover effect
                            'marginRight': '30px'
                        }
                    ),
                    html.Button(
                        'REBA calculation',
                        id='reba-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#4CAF50',  # Green background
                            'color': 'white',  # White text
                            'border': 'none',  # No border
                            'padding': '15px 32px',  # Padding for better spacing
                            'textAlign': 'center',  # Center align the text
                            'textDecoration': 'none',  # No underline
                            'display': 'inline-block',  # Inline block
                            'fontSize': '18px',  # Font size
                            'borderRadius': '12px',  # Rounded corners
                            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
                            'cursor': 'pointer',  # Pointer cursor on hover
                            'transition': 'all 0.3s ease',  # Smooth transition for hover effect
                            'marginLeft': '30px'
                        }
                    ),
                    html.Button(
                        'RULA calculation',
                        id='rula-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#4CAF50',  # Green background
                            'color': 'white',  # White text
                            'border': 'none',  # No border
                            'padding': '15px 32px',  # Padding for better spacing
                            'textAlign': 'center',  # Center align the text
                            'textDecoration': 'none',  # No underline
                            'display': 'inline-block',  # Inline block
                            'fontSize': '18px',  # Font size
                            'borderRadius': '12px',  # Rounded corners
                            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
                            'cursor': 'pointer',  # Pointer cursor on hover
                            'transition': 'all 0.3s ease',  # Smooth transition for hover effect
                            'marginLeft': '60px'
                        }
                    )

                ], style={'textAlign': 'center', 'margin': '100px 0'}),

                html.Div(id='output_message', style={'color': 'darkorange', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px'}),

                html.Div(id='video_analysis_result', children=[], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'justifyContent': 'center'}),

                # Processing Modal
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Processing..."), close_button=False),
                        dbc.ModalBody("Please wait while the video is being processed."),
                    ],
                    id="progress-popup",
                    centered=True,
                    is_open=False,
                    backdrop="static",
                )

            ]),

            # html.H2("2. - Select the analyses mode", style={'marginTop': '100px', 'textAlign': 'center', 'marginBottom': '20px'}),
            # html.H5("Ergonomy assessment", style={'textAlign': 'center', 'marginBottom': '20px'}),
            # html.H5("Activity assessments", style={'textAlign': 'center'})
        ])

    elif pathname == "/activity_assessment":
        return html.Div([
            html.H2("Activity Assessment", style={'textAlign': 'center'}),
            html.P("Activity Assessment functionality will be implemented here.", style={'textAlign': 'center'}),
        ])

    elif pathname == "/work-instructions":
        return html.Div([
            html.H2("Work Instructions", style={'textAlign': 'center'}),
            html.P("Instructions will be provided here.", style={'textAlign': 'center'}),
        ])

    elif pathname == "/chatbot":
        return html.Div([
            html.H2("Chatbot", style={'textAlign': 'center'}),
            html.P("Chatbot functionality will be implemented here.", style={'textAlign': 'center'}),
        ])

    return html.Div([
        html.H1("404: Not Found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {pathname} does not exist."),
    ], className="p-3 bg-light rounded-3")


@app.callback(
    [Output('export-button', 'style'),
     Output('reba-button', 'style'),
     Output('rula-button', 'style')],
    [Input('camera_position_input', 'value')],
)
def update_buttons(selected_value):
    hidden = {'display': 'none'}
    visible = {'backgroundColor': '#4CAF50',  # Green background
               'color': 'white',  # White text
               'border': 'none',  # No border
               'padding': '15px 32px',  # Padding for better spacing
               'textAlign': 'center',  # Center align the text
               'textDecoration': 'none',  # No underline
               'display': 'inline-block',  # Inline block
               'fontSize': '18px',  # Font size
               'borderRadius': '12px',  # Rounded corners
               'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
               'cursor': 'pointer',  # Pointer cursor on hover
               'transition': 'all 0.3s ease',  # Smooth transition for hover effect
               'marginRight': '30px'
               }  # Show buttons with spacing

    if selected_value == 'top-view':
        return visible, hidden, hidden
    elif selected_value == 'side-view':
        return hidden, visible, visible
    elif selected_value == 'back-view':
        return hidden, hidden, hidden
    else:
        return visible, visible, visible  # Hide all if nothing is selected

def convert_to_h264(input_path, output_path):
    try:
        # FFmpeg parancs a H.264 konvertáláshoz
        command = [
            "ffmpeg", "-y",  # Felülírja a meglévő fájlt
            "-i", input_path,  # Bemeneti fájl
            "-c:v", "libx264",  # H.264 kódoló
            "-preset", "fast",  # Gyors kódolás
            "-crf", "23",  # Minőség (23 alapértelmezett, kisebb érték = jobb minőség)
            output_path  # Kimeneti fájl
        ]
        subprocess.run(command, check=True)
        print(f"Video successfully converted to H.264: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during H.264 conversion: {e}")

def process_video_in_background(filename, input_path, output_path, start_frame, end_frame, fps, processing_type, processing_rate=1):
    print(f"Processing video: {filename}")
    try:
        # Create a temporary video file with the desired frame range
        temp_video_path = os.path.join(EXPORTS_DIR, f"temp_{filename}")
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        if not cap.isOpened():
            raise Exception("Error: Could not open input video file.")
        if start_frame >= end_frame or start_frame < 0:
            raise Exception("Invalid frame range specified.")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_num}.")
                break
            temp_out.write(frame)

        cap.release()
        temp_out.release()

        print(f"Temporary video created at: {temp_video_path}")

        # Call the process_video function on the temporary video
        process_video(
            filename=os.path.basename(temp_video_path),
            assess_method=processing_type,
            start_frame=start_frame,
            end_frame=end_frame,
            processing_rate=processing_rate
        )

        print(f"Processing {processing_type} complete.")

        # Move the processed video to the desired output path
        processed_video_path = os.path.join(EXPORTS_DIR, f"output_temp_{filename.split('.')[0]}_{processing_type}.mp4")
        print(f"Processed video path: {processed_video_path}")
        if not os.path.exists(processed_video_path):
            raise Exception("Processed video file not found.")

        # Convert the processed video to H.264 format
        convert_to_h264(processed_video_path, output_path)

        # Update processing status
        snippet_path = f"/get-video/exports/{os.path.basename(output_path)}"
        processing_status["message"] = f"{processing_type} processing complete."
        processing_status["result"] = snippet_path
    except Exception as e:
        # Handle errors
        processing_status["message"] = f"Error: {e}"
        processing_status["result"] = None
        print(f"Error during {processing_type} processing: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # Mark processing as complete
        processing_status["is_processing"] = False

def run_heatmap_and_reencode(output_filename, snippet_filename_before_conversion, snippet_filename_after_conversion):
    try:
        print("Starting hand_heatmap_two_color...")
        hand_heatmap_two_color(output_filename)
        print("Finished hand_heatmap_two_color.")

        print("Starting reencode_video...")
        reencode_video(snippet_filename_before_conversion, snippet_filename_after_conversion)
        print("Finished reencode_video.")

        snippet_path = f"/get-video/exports/{snippet_filename_after_conversion}"
        processing_status["message"] = "Hand heatmap processing complete."
        processing_status["result"] = snippet_path
    except Exception as e:
        print(f"Error during hand heatmap processing: {e}")
        processing_status["message"] = f"Hiba a feldolgozás során: {e}"
        processing_status["result"] = None
    finally:
        print(">>> FINALLY: is_processing False-ra állítva")
        print(">>> result jelenlegi állapot:", processing_status.get("result"))
        if processing_status.get("result") is None:
            print(">>> FIGYELEM: result nincs beállítva!!!")
        processing_status["is_processing"] = False

@app.callback(
    Output('video-display', 'src'),
    Output('video-display', 'style'),
    Output('video-duration', 'children'),
    Output('selected_times', 'children'),
    Output('output_message', 'children'),
    Output('error_message', 'children'),
    Output('time_slider', 'max'),
    Output('slider-container', 'style'),
    Output('time_slider', 'marks'),
    Output('time-from-input', 'value'),
    Output('time-to-input', 'value'),
    Output('time-from-input', 'max'),
    Output('time-to-input', 'max'),
    Output('controls_container', 'style'),
    Output('video_analysis_result', 'children'),
    Output("progress-popup", "is_open"),
    Output("progress_interval", "disabled"),
    Input('upload-video', 'contents'),
    State('upload-video', 'filename'),
    Input('time_slider', 'value'),
    Input('export-button', 'n_clicks'),
    Input('reba-button', 'n_clicks'),
    Input('rula-button', 'n_clicks'),
    Input('time-from-input', 'value'),
    Input('time-to-input', 'value'),
    State('camera_position_input', 'value'),
    Input("progress_interval", "n_intervals"),
    prevent_initial_call=True
)
def manage_all_callbacks(upload_contents, filename, time_range, export_clicks, reba_clicks, rula_clicks, time_from, time_to, camera_position, n_intervals):
    global processing_status

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    print(f"Triggered ID: {triggered_id}")

    # Initialize variables
    message = ""
    error_message = ""
    video_src = None
    video_style = {'display': 'none'}
    slider_style = {'display': 'none', 'width': '80%', 'margin': '0 auto'}
    video_duration = ""
    selected_times = ""
    max_duration = 100
    marks = {}
    analysis_result_children = []
    is_open = False
    interval_disabled = True

    # Handle video upload
    if triggered_id == "upload-video":
        if not upload_contents or not filename:
            error_message = "No video uploaded."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)

        # Save the uploaded video
        data = upload_contents.encode("utf8").split(b";base64,")[1]
        video_path = os.path.join(IMPORTS_DIR, filename)
        with open(video_path, "wb") as fp:
            fp.write(base64.decodebytes(data))
        print(f"Video saved at: {video_path}")

        # Calculate video duration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_message = "Error: Could not open video file."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Update UI elements
        video_src = f"/get-video/imports/{filename}"
        video_style = {'display': 'block', 'width': '80%', 'margin': '0 auto'}
        slider_style = {'display': 'block', 'width': '80%', 'margin': '0 auto'}
        video_duration = f"{duration:.2f} seconds"
        max_duration = duration
        marks = {0: "0 sec", **{round(i * (duration / 9), 1): f"{round(i * (duration / 9), 1)} sec" for i in range(1, 9)}, int(duration): f"{int(duration)} sec"}

        return (
        video_src, video_style, video_duration, selected_times, message, error_message, max_duration,
        slider_style, marks, time_from, time_to, max_duration, max_duration, {'display': 'block'},
        analysis_result_children, is_open, interval_disabled
    )

    elif triggered_id == "export-button":
        print("Processing hand heatmap...")
        if not upload_contents or not filename:
            error_message = "No video uploaded."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)

        # Prepare file paths and parameters
        input_path = os.path.join(IMPORTS_DIR, filename)
        output_filename = f"video_export_{filename.split('.')[0]}_{str(time_from).replace('.', '-')}_{str(time_to).replace('.', '-')}.mp4"
        output_path = os.path.join(EXPORTS_DIR, output_filename)
        file_extension = filename.lower().split('.')[-1]
        output_filename_with_extension = f'{output_filename}.{file_extension}'

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            error_message = "Error: Could not open video file or invalid FPS."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)
        start_frame, end_frame = int(time_from * fps), int(time_to * fps)
        cap.release()

        # Update processing status
        processing_status["is_processing"] = True
        processing_status["message"] = "Processing hand heatmap... Please wait."
        processing_status["result"] = None

        # Start the hand heatmap processing in a separate thread
        snippet_filename_before_conversion = f'output_{output_filename}'
        snippet_filename_after_conversion = f'converted_output_{output_filename}'
        t1 = threading.Thread(target=run_heatmap_and_reencode,
                         args=(output_filename, snippet_filename_before_conversion, snippet_filename_after_conversion))
        t1.start()

        message = "Result of the hand assessment video analytics:"
        snippet_path = f'/exports/{snippet_filename_after_conversion}'
        heatmap_filename = f'overall_heatmap_{output_filename}.jpg'
        # heatmap_path = os.path.join(EXPORTS_DIR, heatmap_filename)
        heatmap_path = f'/exports/{heatmap_filename}'
        analysis_result_children = [
                html.Video(src=snippet_path, controls=True, style={'display': 'block', 'width': '50%', 'margin': '0 auto'}),
                html.Img(src=heatmap_path, style={'width': '40%', 'height': '40%', 'margin': '10px'})
        ]
        # Enable the progress modal and interval
        is_open = True
        interval_disabled = False

    # Handle time range updates
    elif triggered_id in ["time-from-input", "time-to-input"]:
        # Update the selected time range
        selected_times = f"From {time_from:.2f} sec to {time_to:.2f} sec"
        return (dash.no_update, dash.no_update, dash.no_update, selected_times, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, time_from, time_to, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)    

    #Handle REBA processing
    elif triggered_id == "reba-button":
        print("Processing REBA video analytics...")
        if not upload_contents or not filename:
            error_message = "No video uploaded."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)
        
        #prepare file paths and parameters
        input_path = os.path.join(IMPORTS_DIR, filename)
        output_filename = f"video_export_{filename.split('.')[0]}_{str(time_from).replace('.', '-')}_{str(time_to).replace('.', '-')}.mp4"
        output_path = os.path.join(EXPORTS_DIR, output_filename)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            error_message = "Error: Could not open video file or invalid FPS."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)
        start_frame, end_frame = int(time_from * fps), int(time_to * fps)
        cap.release()

        threading.Thread(target=process_video_in_background, args=(filename,input_path, output_path, start_frame, end_frame, fps, "REBA" , 10)).start()

        #Update processing status
        processing_status["is_processing"] = True
        processing_status["message"] = "Processing REBA video analytics... Please wait."
        processing_status["result"] = None


        #enable the progress modal and interval
        is_open = True
        interval_disabled = False

    # Handle RULA processing
    elif triggered_id == "rula-button":
        print("Processing RULA video analytics...")
        if not upload_contents or not filename:
            error_message = "No video uploaded."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)

        # Prepare file paths and parameters
        input_path = os.path.join(IMPORTS_DIR, filename)
        output_filename = f"video_export_{filename.split('.')[0]}_{str(time_from).replace('.', '-')}_{str(time_to).replace('.', '-')}.mp4"
        output_path = os.path.join(EXPORTS_DIR, output_filename)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            error_message = "Error: Could not open video file or invalid FPS."
            return (None, video_style, "", "", "", error_message, max_duration, slider_style, marks, 0, 15, max_duration, max_duration, {'display': 'none'}, [], False, True)
        start_frame, end_frame = int(time_from * fps), int(time_to * fps)
        cap.release()


        # Start the RULA processing in a separate thread
        threading.Thread(target=process_video_in_background, args=(filename,input_path, output_path, start_frame, end_frame, fps, "RULA" , 10)).start()


        # Update processing status
        processing_status["is_processing"] = True
        processing_status["message"] = "Processing RULA video analytics... Please wait."
        processing_status["result"] = None

        # Enable the progress modal and interval
        is_open = True
        interval_disabled = False

        return (
        video_src, video_style, video_duration, selected_times, message, error_message, max_duration,
        slider_style, marks, time_from, time_to, max_duration, max_duration, {'display': 'block'},
        analysis_result_children, is_open, interval_disabled
        )

    # Handle progress updates
    elif triggered_id == "progress_interval":
        print(f"Processing status: {processing_status['is_processing']}")
        if processing_status["is_processing"]:
            # Processing is still ongoing
            return (video_src, video_style, video_duration, selected_times, message, processing_status["message"], max_duration,
                    slider_style, marks, time_from, time_to, max_duration, max_duration, {'display': 'block'}, [], True, False)

        if processing_status["result"]:
            # Processing succeeded
            snippet_path = processing_status["result"]
            print(f"Video snippet saved at: {snippet_path}")
            analysis_result_children = [
                html.Video(src=snippet_path, controls=True, style={'display': 'block', 'width': '50%', 'margin': '0 auto'})
            ]
            return (video_src, video_style, video_duration, selected_times, "Processing complete.", "", max_duration,
                    slider_style, marks, time_from, time_to, max_duration, max_duration, {'display': 'none'}, analysis_result_children, False, True)
        else:
            # Processing failed
            print(f"Processing failed: {processing_status['message']}")
            return (video_src, video_style, video_duration, selected_times, "", processing_status["message"], max_duration,
                    slider_style, marks, time_from, time_to, max_duration, max_duration, {'display': 'none'}, [], False, True)

# IMPORT_VIDEO_FOLDER = '/imports'
# EXPORT_VIDEO_FOLDER = '/exports'


@app.server.route('/get-video/exports/<filename>')
def serve_export_video(filename):
    return send_from_directory(EXPORTS_DIR, filename)


@app.server.route('/get-video/imports/<filename>')
def serve_import_video(filename):
    return send_from_directory(IMPORTS_DIR, filename)


# IMAGE_FOLDER = '/exports'


@app.server.route('/exports/<filename>')
def serve_image(filename):
    return send_from_directory(EXPORTS_DIR, filename)


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
