import os
import cv2
import math
from pathlib import Path

def format_timestamp(milliseconds):
    """Converts milliseconds into a safe, readable filename string."""
    # Ensure we don't process negative times
    milliseconds = max(0, milliseconds)
    seconds = milliseconds / 1000.0
    
    # Calculate hours, minutes, seconds, and remaining milliseconds
    s = int(seconds % 60)
    ms = int(math.modf(seconds)[0] * 1000)
    
    return f"{s:02d}s-{ms:03d}ms"

def extract_frames_with_time(root_folder):
    # Common video formats
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}
    root_path = Path(root_folder)
    
    if not root_path.exists() or not root_path.is_dir():
        print("Error: The provided folder path is invalid.")
        return

    for file_path in root_path.rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            
            output_dir = file_path.parent / file_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            video_name = file_path.stem
            folder_name = file_path.parent.name

            print(f"Processing: {file_path.name}")
            print(f"Saving to:  {output_dir}")
            
            # Open the video using OpenCV
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                print(f"❌ Error: Could not open {file_path.name}\n")
                continue
            
            # 3 frames per second means we want a frame roughly every 333.33 milliseconds
            target_interval_ms = 1000.0 / 3.0 
            next_target_ms = 0.0
            saved_count = 0
            
            while True:
                # Read the next frame
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                
                # Get the current timestamp of the video in milliseconds
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                # If we've reached or passed our next target time, save the frame
                if current_ms >= next_target_ms:
                    timestamp_str = format_timestamp(current_ms)
                    output_filename = output_dir / f"{folder_name}_{video_name}_time_{timestamp_str}.jpg"
                    
                    # Save the image to the folder
                    cv2.imwrite(str(output_filename), frame)
                    saved_count += 1
                    
                    # Advance the target to the next 1/3rd of a second
                    next_target_ms += target_interval_ms
            
            # Release the video file from memory
            cap.release()
            print(f"✅ Success! Extracted {saved_count} timestamped frames.\n")

if __name__ == "__main__":
    print("--- Video Frame Extractor (Timestamp Edition) ---")
    target_folder = input("Enter the absolute path to your root folder: ")
    
    # Strip quotes in case you copy-pasted a path with quotes around it
    target_folder = target_folder.strip('"').strip("'")
    
    extract_frames_with_time(target_folder)
    print("Done!")