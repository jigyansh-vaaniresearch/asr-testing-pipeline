# import asyncio
# import yaml
# import time
# import os
# import argparse
# from dotenv import load_dotenv
# import jiwer
# from models import get_model_handler

# load_dotenv()

# # Load Config
# with open("config.yaml", "r") as f:
#     CONFIG = yaml.safe_load(f)

# # Simulation Settings
# CHUNK_SIZE = CONFIG['experiment'].get('chunk_size', 3200)
# CHUNK_DURATION = CONFIG['experiment'].get('chunk_duration', 0.1)

# async def stream_audio_generator(file_path):
#     """
#     Async generator that reads audio file and yields chunks 
#     simulating real-time delay.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Audio file not found: {file_path}")

#     with open(file_path, "rb") as audio_file:
#         while True:
#             data = audio_file.read(CHUNK_SIZE)
#             if not data:
#                 break
#             yield data
#             # CRITICAL: await asyncio.sleep allows other tasks (WebSocket listeners)
#             # to run while we simulate the delay between audio chunks.
#             # Plain time.sleep() would BLOCK everything.
#             await asyncio.sleep(CHUNK_DURATION)

# def calculate_metrics(model_name, hypothesis, ground_truth_path, latency):
#     print("\n" + "="*50)
#     print(f"📊 REPORT FOR: {model_name.upper()}")
#     print("="*50)
    
#     # 1. Latency
#     print(f"⏱️  First Token Latency: {latency:.2f} ms")

#     # 2. Accuracy (WER/MER)
#     if os.path.exists(ground_truth_path):
#         with open(ground_truth_path, "r", encoding="utf-8") as f:
#             reference = f.read()
        
#         wer = jiwer.wer(reference, hypothesis)
        
#         print(f"✅ Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
#         print("-" * 50)
#         print(f"📝 Ref: {reference[:100]}...")
#         print(f"🗣️  Hyp: {hypothesis[:100]}...")
#     else:
#         print("⚠️  Ground truth file not found. WER skipped.")
#         wer = 0.0

#     print("="*50)

#     # Save Results
#     os.makedirs("results", exist_ok=True)
#     with open(f"results/result_9016{model_name}.txt", "w", encoding="utf-8") as f:
#         f.write(f"Model: {model_name}\nLatency: {latency}\nWER: {wer}\nTranscript: {hypothesis}")

# async def run_pipeline():
#     model_name = CONFIG['active_model']
#     print(f"🚀 Starting Pipeline for: {model_name}")
    
#     # 1. Initialize & Connect
#     try:
#         handler = get_model_handler(model_name, CONFIG)
#         await handler.connect()
#     except Exception as e:
#         print(f"❌ Initialization Failed: {e}")
#         return

#     # 2. Stream Audio
#     audio_path = CONFIG['experiment']['audio_input_path']
#     print(f"🎙️  Streaming: {audio_path}")
    
#     # We set the start time explicitly before the first chunk to ensure
#     # we measure latency from the moment audio *starts* flowing.
    
#     async for chunk in stream_audio_generator(audio_path):
#         await handler.process_audio(chunk)

#     # 3. Finish & Get Results
#     print("\n🛑 Audio finished. Waiting for final transcript...")
#     final_transcript = await handler.finish()

#     # 4. Metrics
#     calculate_metrics(
#         model_name, 
#         final_transcript, 
#         CONFIG['experiment']['ground_truth_path'], 
#         handler.get_latency()
#     )

# if __name__ == "__main__":
#     try:
#         asyncio.run(run_pipeline())
#     except KeyboardInterrupt:
#         print("\nPipeline stopped by user.")/

import asyncio
import yaml
import time
import os
import argparse
from dotenv import load_dotenv
import jiwer
from models import get_model_handler

load_dotenv()

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Simulation Settings
CHUNK_SIZE = CONFIG['experiment'].get('chunk_size', 3200)
CHUNK_DURATION = CONFIG['experiment'].get('chunk_duration', 0.1)

# async def stream_audio_generator(file_path):
#     """
#     Async generator that reads audio file and yields chunks 
#     simulating real-time delay.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Audio file not found: {file_path}")

#     with open(file_path, "rb") as audio_file:
#         while True:
#             data = audio_file.read(CHUNK_SIZE)
#             if not data:
#                 break
#             yield data
#             await asyncio.sleep(CHUNK_DURATION)

async def stream_audio_generator(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    with open(file_path, "rb") as audio_file:
        if file_path.endswith(".wav"):
            audio_file.seek(44)

        while True:
            data = audio_file.read(CHUNK_SIZE)
            if not data:
                break
            yield data
            await asyncio.sleep(CHUNK_DURATION)

def calculate_metrics(model_name, hypothesis, ground_truth_path, latency):
    print("\n" + "="*50)
    print(f" REPORT FOR: {model_name.upper()}")
    print("="*50)
    
    # 1. Latency (Handle None case)
    if latency is None:
        print("  First Token Latency: N/A (No tokens received)")
    else:
        print(f"  First Token Latency: {latency:.2f} ms")

    # 2. Accuracy (WER/MER)
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            reference = f.read()
        
        # Handle empty hypothesis to avoid jiwer errors
        if not hypothesis:
            hypothesis = ""
            
        wer = jiwer.wer(reference, hypothesis)
        
        print(f" Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
        print("-" * 50)
        print(f" Ref: {reference[:100]}...")
        print(f"🗣️  Hyp: {hypothesis[:100]}...")
    else:
        print("  Ground truth file not found. WER skipped.")
        wer = 0.0

    print("="*50)

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open(f"results/result_9016{model_name}.txt", "w", encoding="utf-8") as f:
        # Safe conversion for file writing
        lat_str = f"{latency:.2f}" if latency is not None else "N/A"
        f.write(f"Model: {model_name}\nLatency: {lat_str}\nWER: {wer}\nTranscript: {hypothesis}")

async def run_pipeline():
    model_name = CONFIG['active_model']
    print(f" Starting Pipeline for: {model_name}")
    
    # 1. Initialize & Connect
    try:
        handler = get_model_handler(model_name, CONFIG)
        await handler.connect()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Stream Audio
    audio_path = CONFIG['experiment']['audio_input_path']
    print(f"🎙️  Streaming: {audio_path}")
    
    if hasattr(handler, 'set_start_time'):
        handler.set_start_time()
    
    async for chunk in stream_audio_generator(audio_path):
        await handler.process_audio(chunk)

    # 3. Finish & Get Results
    print("\n Audio finished. Waiting for final transcript...")
    final_transcript = await handler.finish()

    # 4. Metrics
    calculate_metrics(
        model_name, 
        final_transcript, 
        CONFIG['experiment']['ground_truth_path'], 
        handler.get_latency()
    )

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")