import os
import glob
import subprocess

# Settings
INPUT_DIR = "audio_samples2"
TARGET_SAMPLE_RATE = "16000"
TARGET_CHANNELS = "1"
TARGET_CODEC = "pcm_s16le"

def convert_mp3s():
    # Find all mp3s
    mp3_files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in '{INPUT_DIR}'")
        return

    print(f"found {len(mp3_files)} MP3 files. Converting...")
    
    for mp3_path in mp3_files:
        # Create new filename: "audio.mp3" -> "audio.wav"
        wav_path = mp3_path.rsplit('.', 1)[0] + ".wav"
        
        # Skip if it already exists
        if os.path.exists(wav_path):
            print(f" Skipping (WAV exists): {os.path.basename(wav_path)}")
            continue
            
        print(f" Converting: {os.path.basename(mp3_path)}...")
        
        try:
            # Run ffmpeg command
            command = [
                "ffmpeg", "-i", mp3_path,
                "-ar", TARGET_SAMPLE_RATE,
                "-ac", TARGET_CHANNELS,
                "-c:a", TARGET_CODEC,
                "-y",  # Overwrite enabled
                wav_path
            ]
            # Run silently (remove stdout/stderr capture to see errors if it fails)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   Created: {os.path.basename(wav_path)}")
        except Exception as e:
            print(f"   Failed to convert {mp3_path}: {e}")

if __name__ == "__main__":
    convert_mp3s()