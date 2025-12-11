# import asyncio
# import websockets
# import json
# import os
# import glob
# import time
# import base64
# import httpx
# from pathlib import Path
# from dotenv import load_dotenv

# # Load API Key
# load_dotenv()
# API_KEY = os.getenv("GLADIA_API_KEY")

# # Configuration
# INPUT_DIR = "audio_samples"
# OUTPUT_DIR = "gladia_transcript"
# SAMPLE_RATE = 16000

# class GladiaClient:
#     def __init__(self):
#         self.ws = None
#         self.transcript_parts = []

#     async def connect(self):
#         print(f"   🔌 Connecting to Gladia...", end="\r")
#         try:
#             # 1. Initialize Session via REST
#             async with httpx.AsyncClient() as client:
#                 resp = await client.post(
#                     "https://api.gladia.io/v2/live",
#                     headers={"x-gladia-key": API_KEY},
#                     json={
#                         "sample_rate": SAMPLE_RATE,
#                         "encoding": "wav/pcm",
#                         "language_config": {"languages": ["hi", "en"]} 
#                     }
#                 )
#                 if resp.status_code not in [200, 201]:
#                     raise Exception(f"Init failed: {resp.status_code} - {resp.text}")
                
#                 data = resp.json()
#                 url = data['url']

#             # 2. Connect WebSocket
#             self.ws = await websockets.connect(url)
#             # print("   ✅ Connected.")

#         except Exception as e:
#             print(f"   ❌ Connection failed: {e}")
#             raise

#     async def stream_audio(self, file_path):
#         chunk_size = 3200 
#         with open(file_path, "rb") as audio:
#             while True:
#                 data = audio.read(chunk_size)
#                 if not data:
#                     break
                
#                 # Gladia V2 expects base64 encoded chunks
#                 payload = {
#                     "type": "audio_chunk",
#                     "data": {"chunk": base64.b64encode(data).decode("utf-8")}
#                 }
#                 await self.ws.send(json.dumps(payload))
#                 await asyncio.sleep(0.05)
        
#         # Stop recording
#         await self.ws.send(json.dumps({"type": "stop_recording"}))

#     async def listen_for_results(self):
#         try:
#             async for message in self.ws:
#                 data = json.loads(message)
                
#                 if data.get('type') == 'transcript':
#                     # Check for final flag
#                     is_final = data.get('data', {}).get('is_final', False)
#                     text = data.get('data', {}).get('utterance', {}).get('text', '')

#                     if is_final and text:
#                         self.transcript_parts.append(text)
#                         print(f"   📝 Received: {text[:50]}...", end="\r")

#         except websockets.exceptions.ConnectionClosed:
#             pass # Normal closure
#         except Exception as e:
#             print(f"\n   ⚠️ Listen loop error: {e}")

#     async def process_file(self, file_path):
#         filename = os.path.basename(file_path)
#         print(f"\n🎧 Processing: {filename}")
#         self.transcript_parts = []
        
#         try:
#             await self.connect()
#             stream_task = asyncio.create_task(self.stream_audio(file_path))
#             listen_task = asyncio.create_task(self.listen_for_results())
            
#             await stream_task
#             # Wait a bit for final results before closing
#             await asyncio.sleep(2) 
#             if self.ws:
#                 await self.ws.close()
#             await listen_task
            
#             return " ".join(self.transcript_parts)
#         except Exception as e:
#             print(f"   ❌ Failed: {e}")
#             return None

# async def main():
#     if not os.path.exists(INPUT_DIR):
#         print(f"❌ Input directory '{INPUT_DIR}' not found!")
#         return
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
#     print(f"🚀 Starting Gladia Batch Processing ({len(wav_files)} files)...")
#     print(f"📂 Output: {OUTPUT_DIR}/")
#     print("-" * 50)

#     processor = GladiaClient()
#     for wav_file in wav_files:
#         start_time = time.time()
#         transcript = await processor.process_file(wav_file)
#         duration = time.time() - start_time
        
#         if transcript:
#             base_name = Path(wav_file).stem
#             output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
#             with open(output_path, "w", encoding="utf-8") as f:
#                 f.write(transcript)
#             print(f"\n   ✅ Saved to: {output_path}")
#             print(f"   ⏱️  Time: {duration:.2f}s")
#         else:
#             print("\n   ⚠️ No transcript.")
#         await asyncio.sleep(1)

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import websockets
import json
import os
import glob
import time
import base64
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GLADIA_API_KEY")

INPUT_DIR = "audio_samples"
OUTPUT_DIR = "gladia_transcript_metrics"
SAMPLE_RATE = 16000

class GladiaClient:
    def __init__(self):
        self.ws = None
        self.transcript_parts = []
        self.stream_start_time = 0
        self.first_token_time = None
        self.confidences = []

    async def connect(self):
        print(f"   🔌 Connecting to Gladia...", end="\r")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.gladia.io/v2/live",
                headers={"x-gladia-key": API_KEY},
                json={
                    "sample_rate": SAMPLE_RATE,
                    "encoding": "wav/pcm",
                    "language_config": {"languages": ["hi", "en"]} 
                }
            )
            data = resp.json()
            url = data['url']
            self.ws = await websockets.connect(url)

    async def stream_audio(self, file_path):
        self.stream_start_time = time.time()
        chunk_size = 3200 
        with open(file_path, "rb") as audio:
            while True:
                data = audio.read(chunk_size)
                if not data: break
                
                payload = {
                    "type": "audio_chunk",
                    "data": {"chunk": base64.b64encode(data).decode("utf-8")}
                }
                await self.ws.send(json.dumps(payload))
                await asyncio.sleep(0.05)
        
        await self.ws.send(json.dumps({"type": "stop_recording"}))

    async def listen_for_results(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                if data.get('type') == 'transcript':
                    # 1. Capture TTFT
                    if self.first_token_time is None:
                        self.first_token_time = time.time()

                    # 2. Capture Confidence
                    utt = data.get('data', {}).get('utterance', {})
                    if 'confidence' in utt:
                        self.confidences.append(utt['confidence'])

                    is_final = data.get('data', {}).get('is_final', False)
                    text = utt.get('text', '')

                    if is_final and text:
                        self.transcript_parts.append(text)
                        print(f"   Received: {text[:50]}...", end="\r")

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"\n    Listen loop error: {e}")

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
            await asyncio.sleep(2) 
            if self.ws: await self.ws.close()
            await listen_task
            
            ttft = 0.0
            if self.first_token_time:
                ttft = self.first_token_time - self.stream_start_time
            
            avg_conf = 0.0
            if self.confidences:
                avg_conf = sum(self.confidences) / len(self.confidences)

            return " ".join(self.transcript_parts), ttft, avg_conf
            
        except Exception as e:
            print(f"   Failed: {e}")
            return None, 0, 0

async def main():
    if not os.path.exists(INPUT_DIR): return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    print(f" Starting Gladia Batch Processing...")
    print(f" Output: {OUTPUT_DIR}/")
    processor = GladiaClient()
    
    for wav_file in wav_files:
        transcript, ttft, conf = await processor.process_file(wav_file)
        
        if transcript:
            base_name = Path(wav_file).stem
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"TTFT: {ttft:.4f}\n")
                f.write(f"Confidence: {conf:.4f}\n")
                f.write(f"Transcript: {transcript}\n")
            print(f"\n    Saved: {output_path} (Conf: {conf:.2f})")
        else:
            print("\n    No transcript.")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())