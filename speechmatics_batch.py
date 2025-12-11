# import asyncio
# import websockets
# import json
# import os
# import glob
# import time
# from pathlib import Path
# from dotenv import load_dotenv

# # Load API Key
# load_dotenv()
# API_KEY = os.getenv("SPEECHMATICS_API_KEY")

# # Configuration
# INPUT_DIR = "audio_samples"
# OUTPUT_DIR = "speechmatics_transcript"
# SAMPLE_RATE = 16000
# LANGUAGE = "hi"  # Change to 'en' or other codes if needed
# # Standard EU Region URL. Use 'eu2.rt.speechmatics.com' if you are an Enterprise user.
# WEBSOCKET_URL = "wss://neu.rt.speechmatics.com/v2" 

# class SpeechmaticsClient:
#     def __init__(self):
#         self.ws = None
#         self.transcript_parts = []
#         self.is_connected = False

#     async def connect(self):
#         print(f"   🔌 Connecting to Speechmatics...", end="\r")
#         try:
#             # Use 'additional_headers' for modern websockets library
#             self.ws = await websockets.connect(
#                 WEBSOCKET_URL,
#                 additional_headers={"Authorization": f"Bearer {API_KEY}"},
#                 ping_interval=None  # Disable auto-ping to prevent timeouts during processing
#             )
            
#             # Configuration Message
#             start_msg = {
#                 "message": "StartRecognition",
#                 "audio_format": {
#                     "type": "raw",
#                     "encoding": "pcm_s16le",
#                     "sample_rate": SAMPLE_RATE
#                 },
#                 "transcription_config": {
#                     "language": LANGUAGE,
#                     "enable_partials": True, # Required to keep connection alive/active
#                     "operating_point": "enhanced"
#                 }
#             }
#             await self.ws.send(json.dumps(start_msg))
#             self.is_connected = True
#             # print("   ✅ Connected.")
#         except Exception as e:
#             print(f"   ❌ Connection failed: {e}")
#             self.is_connected = False
#             raise

#     async def stream_audio(self, file_path):
#         """Reads audio file and streams it to the WebSocket"""
#         chunk_size = 3200 # 100ms chunks
        
#         with open(file_path, "rb") as audio:
#             while True:
#                 data = audio.read(chunk_size)
#                 if not data:
#                     break
#                 await self.ws.send(data)
#                 # Small sleep to simulate real-time and prevent buffer overflow on server
#                 await asyncio.sleep(0.05) 
        
#         # Signal end of audio stream
#         await self.ws.send(json.dumps({"message": "EndOfStream"}))

#     async def listen_for_results(self):
#         """Listens for transcripts until EndOfTranscription is received"""
#         try:
#             async for message in self.ws:
#                 data = json.loads(message)
#                 msg_type = data.get('message')

#                 if msg_type == 'AddTranscript':
#                     # This is a final segment
#                     text = data.get('metadata', {}).get('transcript', '')
#                     if text:
#                         self.transcript_parts.append(text)
#                         print(f"   📝 Received: {text[:50]}...", end="\r")

#                 elif msg_type == 'EndOfTranscription':
#                     # Server is done processing
#                     break
                
#                 elif msg_type == 'Error':
#                     print(f"\n   ❌ Server Error: {data.get('reason')}")
#                     break

#         except Exception as e:
#             print(f"\n   ⚠️ Listen loop error: {e}")

#     async def process_file(self, file_path):
#         filename = os.path.basename(file_path)
#         print(f"\n🎧 Processing: {filename}")
        
#         self.transcript_parts = []
        
#         try:
#             await self.connect()
            
#             # Run streaming and listening concurrently
#             # We wait for the listener to finish (which happens after EndOfStream is sent and processed)
#             stream_task = asyncio.create_task(self.stream_audio(file_path))
#             listen_task = asyncio.create_task(self.listen_for_results())
            
#             await stream_task
#             await listen_task
            
#             return " ".join(self.transcript_parts)

#         except Exception as e:
#             print(f"   ❌ Failed: {e}")
#             return None
#         finally:
#             if self.ws:
#                 await self.ws.close()

# async def main():
#     # 1. Setup Directories
#     if not os.path.exists(INPUT_DIR):
#         print(f"❌ Input directory '{INPUT_DIR}' not found!")
#         return
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 2. Get WAV Files
#     wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
#     if not wav_files:
#         print("❌ No .wav files found. Please run convert_files.py first.")
#         return

#     print(f"🚀 Found {len(wav_files)} files. Starting Batch Speechmatics Processing...")
#     print(f"📂 Output Directory: {OUTPUT_DIR}/")
#     print("-" * 50)

#     # 3. Process Each File
#     processor = SpeechmaticsClient()
    
#     for i, wav_file in enumerate(wav_files):
#         start_time = time.time()
#         transcript = await processor.process_file(wav_file)
#         duration = time.time() - start_time
        
#         if transcript:
#             # Save to file
#             base_name = Path(wav_file).stem
#             output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            
#             with open(output_path, "w", encoding="utf-8") as f:
#                 f.write(transcript)
            
#             print(f"\n   ✅ Saved to: {output_path}")
#             print(f"   ⏱️  Time taken: {duration:.2f}s")
#         else:
#             print("\n   ⚠️ No transcript generated.")
        
#         # Brief pause between files
#         await asyncio.sleep(1)

#     print("\n🎉 Batch Processing Complete!")

# if __name__ == "__main__":
#     if not API_KEY:
#         print("❌ Error: SPEECHMATICS_API_KEY not found in .env file.")
#     else:
#         try:
#             asyncio.run(main())
#         except KeyboardInterrupt:
#             print("\n🛑 Stopped by user.")

import asyncio
import websockets
import json
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SPEECHMATICS_API_KEY")

INPUT_DIR = "audio_samples"
OUTPUT_DIR = "speechmatics_transcript_metrics"
SAMPLE_RATE = 16000
WEBSOCKET_URL = "wss://neu.rt.speechmatics.com/v2" 

class SpeechmaticsClient:
    def __init__(self):
        self.ws = None
        self.transcript_parts = []
        self.stream_start_time = 0
        self.first_token_time = None
        self.confidences = []

    async def connect(self):
        print(f"   🔌 Connecting to Speechmatics...", end="\r")
        self.ws = await websockets.connect(
            WEBSOCKET_URL,
            additional_headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        start_msg = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            "transcription_config": {
                "language": "hi",
                "enable_partials": True,
                "operating_point": "enhanced"
            }
        }
        await self.ws.send(json.dumps(start_msg))

    async def stream_audio(self, file_path):
        self.stream_start_time = time.time()
        chunk_size = 3200
        
        with open(file_path, "rb") as audio:
            while True:
                data = audio.read(chunk_size)
                if not data: break
                await self.ws.send(data)
                await asyncio.sleep(0.05) 
        
        await self.ws.send(json.dumps({"message": "EndOfStream"}))

    async def listen_for_results(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get('message')

                if msg_type == 'AddTranscript':
                    if self.first_token_time is None:
                        self.first_token_time = time.time()

                    results = data.get('results', [])
                    for res in results:
                        alts = res.get('alternatives', [])
                        if alts:
                            self.confidences.append(alts[0]['confidence'])

                    text = data.get('metadata', {}).get('transcript', '')
                    if text:
                        self.transcript_parts.append(text)
                        print(f"   Received: {text[:50]}...", end="\r")

                elif msg_type == 'EndOfTranscription':
                    break
        except Exception as e:
            print(f"\n   Listen error: {e}")

    async def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"\n Processing: {filename}")
        
        self.transcript_parts = []
        self.confidences = []
        self.first_token_time = None
        
        try:
            await self.connect()
            stream_task = asyncio.create_task(self.stream_audio(file_path))
            listen_task = asyncio.create_task(self.listen_for_results())
            
            await stream_task
            await listen_task
            
            ttft = 0.0
            if self.first_token_time:
                ttft = self.first_token_time - self.stream_start_time
            
            avg_conf = 0.0
            if self.confidences:
                avg_conf = sum(self.confidences) / len(self.confidences)

            return " ".join(self.transcript_parts), ttft, avg_conf

        except Exception as e:
            print(f"    Failed: {e}")
            return None, 0, 0
        finally:
            if self.ws: await self.ws.close()

async def main():
    if not os.path.exists(INPUT_DIR): return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    print(f"Starting Speechmatics Batch Processing...")
    print(f" Output: {OUTPUT_DIR}/")
    processor = SpeechmaticsClient()
    
    for wav_file in wav_files:
        transcript, ttft, conf = await processor.process_file(wav_file)
        
        if transcript:
            base_name = Path(wav_file).stem
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"TTFT: {ttft:.4f}\n")
                f.write(f"Confidence: {conf:.4f}\n")
                f.write(f"Transcript: {transcript}\n")
            print(f"\n   Saved: {output_path}")
        else:
            print("\n   No transcript.")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())