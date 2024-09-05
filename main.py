from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

def extract_frames(video_path, frame_dir):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    
    num_frames = 0
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        frame_image = Image.fromarray(frame)
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        frame_image.save(frame_path)
        num_frames += 1
    
    print(f"Extracted {num_frames} frames.")
    return fps, duration, [os.path.join(frame_dir, f"frame_{i:04d}.png") for i in range(num_frames)]

def extract_features(frame_paths):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    
    features = []
    for frame_path in frame_paths:
        # Load and preprocess the image
        image = Image.open(frame_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            features.append(logits)
    
    print(f"Extracted features from {len(features)} frames.")
    return features

def generate_timestamps(features, fps):
    timestamps = []
    
    # Example logic to generate timestamps from features
    for i, feature in enumerate(features):
        timestamp = i / fps  # Simplistic example
        timestamps.append(timestamp)
    
    print(f"Generated Timestamps: {timestamps}")
    return timestamps

def main():
    video_path = 'video.mp4'
    frame_dir = 'images'
    
    fps, duration, frame_paths = extract_frames(video_path, frame_dir)

    print(fps)
    
    features = extract_features(frame_paths)
    
    timestamps = generate_timestamps(features, fps)
    
    print(f"Final Timestamps: {timestamps}")

if __name__ == "__main__":
    main()
