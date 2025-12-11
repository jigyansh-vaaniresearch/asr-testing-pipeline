# import asyncio
# import os
# import glob
# import time
# from pathlib import Path
# from dotenv import load_dotenv
# from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

# # Load API Key
# load_dotenv()
# API_KEY = os.getenv("DEEPGRAM_API_KEY")

# # Configuration
# INPUT_DIR = "audio_samples"
# OUTPUT_DIR = "deepgram_transcript"

# class DeepgramBatchClient:
#     def __init__(self):
#         self.client = DeepgramClient(API_KEY)
#         self.transcript_parts = []
#         self.dg_connection = None

#     async def process_file(self, file_path):
#         filename = os.path.basename(file_path)
#         print(f"\n🎧 Processing: {filename}")
#         self.transcript_parts = []

#         try:
#             # 1. Connect
#             self.dg_connection = self.client.listen.live.v("1")
            
#             # Define handlers
#             def on_message(self, result, **kwargs):
#                 sentence = result.channel.alternatives[0].transcript
#                 if len(sentence) > 0 and result.is_final:
#                     # We need to access the outer class list
#                     processor.transcript_parts.append(sentence)
#                     print(f"   📝 Received: {sentence[:50]}...", end="\r")

#             self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

#             # Start Connection
#             options = LiveOptions(
#                 model="nova-2",
#                 language="hi",
#                 smart_format=True,
#                 encoding="linear16",
#                 sample_rate=16000,
#                 interim_results=True
#             )
            
#             if self.dg_connection.start(options) is False:
#                 raise Exception("Failed to start Deepgram")

#             # 2. Stream Audio
#             chunk_size = 3200
#             with open(file_path, "rb") as audio:
#                 while True:
#                     data = audio.read(chunk_size)
#                     if not data:
#                         break
#                     self.dg_connection.send(data)
#                     await asyncio.sleep(0.05)

#             # 3. Finish
#             self.dg_connection.finish()
#             return " ".join(self.transcript_parts)

#         except Exception as e:
#             print(f"   ❌ Failed: {e}")
#             return None

# # Global instance for callback access
# processor = DeepgramBatchClient()

# async def main():
#     if not os.path.exists(INPUT_DIR):
#         print(f"❌ Input directory '{INPUT_DIR}' not found!")
#         return
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
#     print(f"🚀 Starting Deepgram Batch Processing ({len(wav_files)} files)...")
#     print(f"📂 Output: {OUTPUT_DIR}/")
#     print("-" * 50)

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
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")

INPUT_DIR = "audio_samples"
OUTPUT_DIR = "deepgram_transcript_metrics"

class DeepgramBatchClient:
    def __init__(self):
        self.client = DeepgramClient(API_KEY)
        self.transcript_parts = []
        self.dg_connection = None
        
        # Metrics storage
        self.stream_start_time = 0
        self.first_token_time = None
        self.confidences = []

    async def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"\n🎧 Processing: {filename}")
        
        self.transcript_parts = []
        self.confidences = []
        self.first_token_time = None
        self.stream_start_time = 0

        try:
            self.dg_connection = self.client.listen.live.v("1")
            
            def on_message(self, result, **kwargs):
                # 1. Capture TTFT
                if processor.first_token_time is None:
                    processor.first_token_time = time.time()

                sentence = result.channel.alternatives[0].transcript
                
                # 2. Capture Confidence
                words = result.channel.alternatives[0].words
                if words:
                    avg_sentence_conf = sum(w.confidence for w in words) / len(words)
                    processor.confidences.append(avg_sentence_conf)

                if len(sentence) > 0 and result.is_final:
                    processor.transcript_parts.append(sentence)
                    print(f"    Received: {sentence[:50]}...", end="\r")

            self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                language="hi",
                smart_format=True,
                encoding="linear16",
                sample_rate=16000,
                interim_results=True
            )
            
            if self.dg_connection.start(options) is False:
                raise Exception("Failed to start Deepgram")

            self.stream_start_time = time.time()
            
            chunk_size = 3200
            with open(file_path, "rb") as audio:
                while True:
                    data = audio.read(chunk_size)
                    if not data: break
                    self.dg_connection.send(data)
                    await asyncio.sleep(0.05)

            self.dg_connection.finish()
            
            full_transcript = " ".join(self.transcript_parts)
            
            ttft = 0.0
            if self.first_token_time:
                ttft = self.first_token_time - self.stream_start_time
            
            avg_conf = 0.0
            if self.confidences:
                avg_conf = sum(self.confidences) / len(self.confidences)

            return full_transcript, ttft, avg_conf

        except Exception as e:
            print(f"    Failed: {e}")
            return None, 0, 0

processor = DeepgramBatchClient()

async def main():
    if not os.path.exists(INPUT_DIR):
        print(f" Input directory '{INPUT_DIR}' not found!")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    print(f" Starting Deepgram Batch Processing...")
    print(f" Output: {OUTPUT_DIR}/")

    for wav_file in wav_files:
        transcript, ttft, conf = await processor.process_file(wav_file)
        
        if transcript:
            base_name = Path(wav_file).stem
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"TTFT: {ttft:.4f}\n")
                f.write(f"Confidence: {conf:.4f}\n")
                f.write(f"Transcript: {transcript}\n")
                
            print(f"\n    Saved: {output_path} (TTFT: {ttft:.2f}s | Conf: {conf:.2f})")
        else:
            print("\n   No transcript.")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())