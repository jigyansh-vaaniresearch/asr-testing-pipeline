# import asyncio
# import os
# import glob
# import time
# from pathlib import Path
# from dotenv import load_dotenv

# # Import AssemblyAI SDK v3
# import assemblyai as aai
# from assemblyai.streaming.v3 import (
#     StreamingClient,
#     StreamingClientOptions,
#     StreamingParameters,
#     StreamingEvents,
#     TurnEvent,
#     StreamingError
# )

# # Load API Key
# load_dotenv()
# API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# # Configuration
# INPUT_DIR = "audio_samples"
# OUTPUT_DIR = "assemblyai_transcript"
# SAMPLE_RATE = 16000

# class AssemblyBatchClient:
#     def __init__(self):
#         self.client = None
#         self.transcript_parts = []

#     def on_turn(self, sender, event: TurnEvent):
#         # We only collect final transcripts (end_of_turn=True) to avoid duplication
#         if event.transcript and event.end_of_turn:
#             self.transcript_parts.append(event.transcript)
#             print(f"   📝 Received: {event.transcript[:50]}...", end="\r")

#     def on_error(self, sender, error: StreamingError):
#         print(f"\n   ❌ Error: {error}")

#     async def process_file(self, file_path):
#         filename = os.path.basename(file_path)
#         print(f"\n🎧 Processing: {filename}")
#         self.transcript_parts = []

#         try:
#             # 1. Setup Client
#             self.client = StreamingClient(
#                 StreamingClientOptions(
#                     api_key=API_KEY,
#                     api_host="streaming.assemblyai.com"
#                 )
#             )
            
#             self.client.on(StreamingEvents.Turn, self.on_turn)
#             self.client.on(StreamingEvents.Error, self.on_error)

#             # 2. Connect
#             # ✅ FIX: Pass sample_rate directly (SDK handles encoding default to pcm_s16le)
#             self.client.connect(
#                 StreamingParameters(
#                     sample_rate=SAMPLE_RATE,
#                     encoding="pcm_s16le", 
#                     format_turns=True
#                 )
#             )

#             # 3. Stream Audio
#             chunk_size = 3200
#             with open(file_path, "rb") as audio:
#                 while True:
#                     data = audio.read(chunk_size)
#                     if not data:
#                         break
#                     self.client.stream(data)
#                     # Sleep allows the async loop to process incoming WebSocket messages
#                     await asyncio.sleep(0.05)

#             # 4. Finish - Use asyncio.to_thread to prevent blocking the loop
#             if self.client:
#                 # ✅ FIX: disconnect() is the correct method, not close()
#                 # Using terminate=True forces immediate closure
#                 await asyncio.to_thread(self.client.disconnect, True)
                
#             return " ".join(self.transcript_parts)

#         except Exception as e:
#             print(f"   ❌ Failed: {e}")
#             return None

# async def main():
#     # Ensure directories exist
#     if not os.path.exists(INPUT_DIR):
#         print(f"❌ Input directory '{INPUT_DIR}' not found!")
#         return
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
#     if not wav_files:
#         print(f"❌ No .wav files found in {INPUT_DIR}")
#         return
    
#     print(f"🚀 Starting AssemblyAI Batch Processing ({len(wav_files)} files)...")
#     print(f"📂 Saving transcripts to: {OUTPUT_DIR}/")
#     print("-" * 50)

#     processor = AssemblyBatchClient()
    
#     try:
#         for wav_file in wav_files:
#             start_time = time.time()
#             transcript = await processor.process_file(wav_file)
#             duration = time.time() - start_time
            
#             if transcript:
#                 base_name = Path(wav_file).stem
#                 output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
                
#                 # Write to file
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     f.write(transcript)
                    
#                 print(f"\n   ✅ Saved to: {output_path}")
#                 print(f"   ⏱️  Time: {duration:.2f}s")
#             else:
#                 print("\n   ⚠️ No transcript received.")
            
#             # Small buffer between files
#             await asyncio.sleep(1)
            
#     except KeyboardInterrupt:
#         print("\n🛑 Process interrupted by user. Exiting...")
#         if processor.client:
#             try:
#                 # Use correct disconnect args
#                 await asyncio.to_thread(processor.client.disconnect, True)
#             except:
#                 pass

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         pass

import asyncio
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv

import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions, StreamingParameters,
    StreamingEvents, TurnEvent, StreamingError
)

load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

INPUT_DIR = "audio_samples"
OUTPUT_DIR = "assemblyai_transcript_metrics"
SAMPLE_RATE = 16000

class AssemblyBatchClient:
    def __init__(self):
        self.client = None
        self.transcript_parts = []
        self.stream_start_time = 0
        self.first_token_time = None
        self.confidences = []

    def on_turn(self, sender, event: TurnEvent):
        if self.first_token_time is None:
            self.first_token_time = time.time()

        if event.transcript and event.end_of_turn:
            if hasattr(event, 'confidence'):
                self.confidences.append(event.confidence)
            
            self.transcript_parts.append(event.transcript)
            print(f"    Received: {event.transcript[:50]}...", end="\r")

    def on_error(self, sender, error: StreamingError):
        print(f"\n    Error: {error}")

    async def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"\n🎧 Processing: {filename}")
        
        self.transcript_parts = []
        self.confidences = []
        self.first_token_time = None

        try:
            self.client = StreamingClient(
                StreamingClientOptions(api_key=API_KEY, api_host="streaming.assemblyai.com")
            )
            
            self.client.on(StreamingEvents.Turn, self.on_turn)
            self.client.on(StreamingEvents.Error, self.on_error)

            self.client.connect(
                StreamingParameters(
                    sample_rate=SAMPLE_RATE,
                    encoding="pcm_s16le", 
                    format_turns=True
                )
            )

            self.stream_start_time = time.time()
            chunk_size = 3200
            
            with open(file_path, "rb") as audio:
                while True:
                    data = audio.read(chunk_size)
                    if not data: break
                    self.client.stream(data)
                    await asyncio.sleep(0.05)

            if self.client:
                await asyncio.to_thread(self.client.disconnect, True)
            
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
    
    print(f" Starting AssemblyAI Batch Processing...")
    print(f" Output: {OUTPUT_DIR}/")
    processor = AssemblyBatchClient()
    
    try:
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
                print("\n   No transcript.")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        if processor.client:
            await asyncio.to_thread(processor.client.disconnect, True)

if __name__ == "__main__":
    asyncio.run(main())