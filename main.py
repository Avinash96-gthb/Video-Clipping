import os
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import ollama

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
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    
    features = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            features.append(logits)
    
    print(f"Extracted features from {len(features)} frames.")
    return features, model

def generate_timestamps(features, fps, model):
    timestamps = []
    for i, feature in enumerate(features):
        predicted_class_idx = feature.argmax().item()
        predicted_class = model.config.id2label[predicted_class_idx]
        description = f"Frame contains: {predicted_class}"
        print(description)
#         prompt = f"Analyze this frame description and determine if it's a good timestamp: '{description}'. If it's a good timestamp, respond with 'GOOD'. Otherwise, respond with 'BAD'."
        
#         try:
#             response = ollama.chat(model='phi3', messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt,
#                 }
#             ])
            
#             if 'GOOD' in response['message']['content'].upper():
#                 timestamps.append(i / fps)
        
#         except Exception as e:
#             print(f"Error communicating with Ollama: {e}")
    
#     print(f"Generated Timestamps: {timestamps}")
#     return timestamps

def main():
    video_path = 'video.mp4'
    frame_dir = 'images'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    fps, duration, frame_paths = extract_frames(video_path, frame_dir)
    print(f"Video FPS: {fps}")
    
    features, model = extract_features(frame_paths)
    
    timestamps = generate_timestamps(features, fps, model)
    
    print(f"Final Timestamps: {timestamps}")

if __name__ == "__main__":
    main()