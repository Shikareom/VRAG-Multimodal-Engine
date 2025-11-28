# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process



import os
import cv2
import base64
import whisper
import ollama
import json
from moviepy import VideoFileClip

# --- CONFIGURATION ---
VIDEO_PATH = "data/videoplayback.mp4"  # Put your video in a 'data' folder

OUTPUT_FILE = "data/video_index.json"
FRAME_INTERVAL = 10  # Analyze a frame every 10 seconds (Lower = more detailed but slower)

def extract_and_transcribe_audio(video_path):
    print("🎧 Starting Audio Transcription...")
    
    # Safety Check
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise ValueError("❌ Error: Video file is empty or missing. Please upload again.")

    # 1. Extract Audio
    audio_path = "temp_audio.mp3"
    
    # Use 'with' to ensure the file is closed automatically
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
    except Exception as e:
        print(f"Error reading video file: {e}")
        if 'video' in locals(): video.close()
        raise e
    finally:
        # FORCE CLOSE THE FILE HANDLE
        if 'video' in locals(): video.close()
    
    # 2. Transcribe with Whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    # 3. Format results
    audio_docs = []
    for segment in result['segments']:
        audio_docs.append({
            "content": segment['text'],
            "start": segment['start'],
            "end": segment['end'],
            "type": "audio"
        })
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    print(f"✅ Audio processed: {len(audio_docs)} segments found.")
    return audio_docs

def analyze_visual_frames(video_path):
    print("👁️ Starting Visual Analysis (This requires a GPU and time)...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * FRAME_INTERVAL)
    
    visual_docs = []
    count = 0
    frame_count_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Only process every Nth frame
        if count % target_frame == 0:
            # Convert frame to base64 for the AI to "see"
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            try:
                # Ask Ollama to describe the image
                response = ollama.chat(model='moondream', messages=[{
                    'role': 'user',
                    'content': 'Describe this scene in detail.',
                    'images': [jpg_as_text]
                }])
                
                description = response['message']['content']
                timestamp = count / fps
                
                visual_docs.append({
                    "content": f"Visual Scene: {description}",
                    "start": timestamp,
                    "end": timestamp + FRAME_INTERVAL,
                    "type": "visual"
                })
                print(f"   👉 Frame at {int(timestamp)}s analyzed.")
                frame_count_processed += 1
                
            except Exception as e:
                print(f"   ❌ Error analyzing frame: {e}")
                
        count += 1
    
    cap.release()
    print(f"✅ Visual analysis complete: {frame_count_processed} frames described.")
    return visual_docs

if __name__ == "__main__":
    # Ensure data folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Please put a video file at {VIDEO_PATH}")
    else:
        # 1. Run Audio Analysis
        audio_data = extract_and_transcribe_audio(VIDEO_PATH)
        
        # 2. Run Visual Analysis
        visual_data = analyze_visual_frames(VIDEO_PATH)
        
        # 3. Save Everything
        full_index = audio_data + visual_data
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(full_index, f, indent=4)
            
        print(f"Success! Index saved to {OUTPUT_FILE}")