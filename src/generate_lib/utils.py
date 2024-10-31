import cv2  
import base64
import numpy as np
from typing import Dict
import random
random.seed(42)


def read_video(video_path: str, total_frames: int):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file.")
    try:
        # Initialize a list to store base64 encoded frames
        base64_frames = []
        
        # Read frames in a loop
        while True:
            success, frame = video.read()
            if not success:
                break  # No more frames or error occurred

            # Encode the frame as a JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Convert the image to base64 string
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_base64)

        
        print(f"Number of frames input to the model is: {total_frames}")
        if total_frames == 1:
            selected_indices = [np.random.choice(range(total_frames))]
        else:
            selected_indices = np.linspace(0, len(base64_frames) - 1, total_frames, dtype=int)

        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0

        return selected_base64_frames, duration
    finally:
        # Release the video capture object
        video.release()


def parse_result(data_string):
    import re
    import json
    # Extract potential JSON using regex
    json_match = re.search(r'```json\n+\s*({.*?})\s*\n+```', data_string, re.DOTALL)
    if not json_match:
        return "No JSON found"

    json_string = json_match.group(1)
    corrections_needed = True

    while corrections_needed:
        try:
            # Try parsing the JSON
            parsed_json = json.loads(json_string)
            return parsed_json  # Return successfully parsed JSON
        except json.JSONDecodeError as e:
            # Get the position of the error
            error_pos = e.pos
            # Attempt to fix common JSON errors (like missing quotes)
            if 'Expecting property name enclosed in double quotes' in str(e):
                # Insert a quote before and after the suspected unquoted key
                json_string = (
                    json_string[:error_pos] + '"' + json_string[error_pos:]
                )
                json_string = (
                    json_string[:json_string.find(':', error_pos) + 1] + '"' + json_string[json_string.find(':', error_pos) + 1:]
                )
            elif 'Expecting value' in str(e):
                # Insert quotes around an unquoted value
                value_start = json_string.rfind(' ', 0, error_pos) + 1
                value_end = json_string.find(',', error_pos)
                if value_end == -1:
                    value_end = json_string.find('}', error_pos)
                json_string = (
                    json_string[:value_start] + '"' + json_string[value_start:value_end] + '"' + json_string[value_end:]
                )
            else:
                return f"Error parsing JSON after corrections: {str(e)}"
