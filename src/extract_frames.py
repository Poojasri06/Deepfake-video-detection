# src/extract_frames.py

import cv2
import os
from tqdm import tqdm

def extract_frames_from_videos(video_dir, output_dir, frame_rate=5):
    os.makedirs(output_dir, exist_ok=True)
    for video_file in tqdm(os.listdir(video_dir)):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_rate == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            success, frame = cap.read()
            frame_count += 1
        cap.release()

if __name__ == "__main__":
    extract_frames_from_videos(r"D:\projects\deepfake_detection\data\celebDFv2_real", 
                               r"D:\projects\deepfake_detection\data\frames_real")
    extract_frames_from_videos(r"D:\projects\deepfake_detection\data\celebDFv2_fake", 
                               r"D:\projects\deepfake_detection\data\frames_fake")
