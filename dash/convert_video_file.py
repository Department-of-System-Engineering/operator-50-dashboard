from moviepy import VideoFileClip
import ffmpeg
import os

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path for the "exports" folder
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")


def reencode_video(input_file, output_file):
    print(f"Re-encoding video: {input_file} to {output_file}")
    try:
        # Use ffmpeg to re-encode the video
        input_path = os.path.join(EXPORTS_DIR, input_file)
        output_path = os.path.join(EXPORTS_DIR, output_file)
        ffmpeg.input(input_path).output(
            output_path,
            vcodec='libx264',  # H.264 codec
            pix_fmt='yuv420p',  # Proper pixel format
            acodec='aac',  # AAC audio codec for compatibility
            strict='experimental'  # Allow experimental features (if needed)
        ).run(overwrite_output=True)
        # print(f"Re-encoded video saved as: {output_path}")
    except ffmpeg.Error as e:
        print("An error occurred:", e)
        print(e.stderr.decode())
    finally:
        print("Re-encoding process completed.")


def convert_video(filename, input_format, output_format):
    """
    Convert a video file to the specified format (AVI or MP4) using MoviePy.

    Parameters:
        filename (str): Input filename.
        input_format (str): Input file format.
        output_format (str): Desired output file format ('avi' or 'mp4').
    """
    # Validate the output format
    if output_format not in ['avi', 'mp4']:
        raise ValueError("Output format must be 'avi' or 'mp4'")

    try:
        input_path = f'{filename}.{input_format}'
        output_path = f'converted_{filename}.{output_format}'
        # Load the video
        clip = VideoFileClip(input_path)

        # Specify codec and audio settings based on output format
        if output_format == 'mp4':
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        elif output_format == 'avi':
            clip.write_videofile(output_path, codec="libxvid", fps=30)
        print(f"Conversion successful! Video saved as: {output_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
