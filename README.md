# Video-Based Ergonomy Analysis Web Application

With this web application, users can upload videos and perform video-based ergonomics analysis.

## Important Notice
To use this application, the host machine **must have the FFmpeg application installed**.
This is because videos generated using the `opencv-python` library are not properly displayed in Dash applications.

### Cause
The `opencv-python` library uses the H.264 video codec, which is not natively supported for playback in Dash applications.

### Effect
Videos generated with `opencv-python` cannot be played in the Dash app.

### Solution
Install FFmpeg on the host machine to re-encode the videos for proper playback.

- **[Download FFmpeg here](https://ffmpeg.org/download.html)**
For example: click Windows builds (e.g., from Gyan.dev).
Download the "Full" or "Essentials" build (usually a .zip file).
- Extract the .zip file to a folder (e.g., C:\ffmpeg).
- Add FFmpeg to System PATH (Recommended).
1. Open Start and search for "Environment Variables".
2. Click Edit the system environment variables.
3. In the System Properties window, click Environment Variables.
4. Under System Variables, find Path, then click Edit.
5. Click New and add: C:\ffmpeg\bin
6. Click OK on all windows and restart your computer.
- Verify the installation

---

## Usage
1. Upload a video file through the web interface.
2. Start video-based ergonomics analysis.
3. View the processed video and analysis results.

Ensure that FFmpeg is properly installed and configured on your system for smooth functionality.
