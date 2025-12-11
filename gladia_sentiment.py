import asyncio
import websockets
import json
import time
import os
import glob
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration
AUDIO_DIR = "audio_samples"
CHUNK_SIZE = 3200  # 100ms chunks (16kHz mono 16-bit)
CHUNK_SLEEP = 0.1  # Real-time simulation delay

class GladiaEmotionClient:
    def __init__(self):
        self.api_key = os.getenv("GLADIA_API_KEY")
        if not self.api_key:
            raise ValueError("GLADIA_API_KEY not found in .env")

    async def get_ws_url(self):
        """Initialize session and get WebSocket URL"""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.gladia.io/v2/live",
                headers={"x-gladia-key": self.api_key},
                json={
                    "sample_rate": 16000,
                    "encoding": "wav/pcm",
                    "language_config": {"languages": ["en", "hi"]},
                    "messages_config": {
                        "receive_partial_transcripts": True,
                        "receive_final_transcripts": True,
                        "receive_speech_events": True
                    }
                }
            )
            if resp.status_code not in [200, 201]:
                raise Exception(f"Init failed: {resp.status_code} - {resp.text}")
            return resp.json()['url']

    async def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"\n🎧 Processing: {filename}")
        print("-" * 40)
        
        start_time = 0
        ttfe = None
        ws_url = await self.get_ws_url()

        try:
            async with websockets.connect(ws_url) as ws:
                start_time = time.time()
                
                async def sender():
                    with open(file_path, "rb") as f:
                        while True:
                            data = f.read(CHUNK_SIZE)
                            if not data: break
                            payload = {
                                "type": "audio_chunk",
                                "data": {"chunk": base64.b64encode(data).decode("utf-8")}
                            }
                            await ws.send(json.dumps(payload))
                            await asyncio.sleep(CHUNK_SLEEP)
                    await ws.send(json.dumps({"type": "stop_recording"}))

                async def receiver():
                    nonlocal ttfe
                    async for msg in ws:
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        # 1. Print Real-Time Transcript
                        if msg_type == "transcript":
                            text = data.get("data", {}).get("utterance", {}).get("text", "")
                            is_final = data.get("data", {}).get("is_final")
                            
                            if text:
                                if is_final:
                                    # Print final line clearly
                                    print(f"   📝 [FINAL]: {text}")
                                    
                                    # Check if sentiment is attached to final transcript metadata
                                    sentiment = data.get("data", {}).get("sentiment")
                                    if sentiment:
                                        if ttfe is None:
                                            ttfe = (time.time() - start_time) * 1000
                                        print(f"      🎭 Sentiment (Metadata): {json.dumps(sentiment)}")
                                else:
                                    # Print partials on same line (optional, can be noisy)
                                    print(f"   ... {text}", end="\r")

                        # 2. Check for explicit Audio Intelligence events
                        if msg_type == "audio_intelligence":
                             sentiment_data = data.get("data", {}).get("sentiment")
                             if sentiment_data:
                                 if ttfe is None:
                                     ttfe = (time.time() - start_time) * 1000
                                 print(f"   ✨ [INTELLIGENCE] Emotion Detected: {json.dumps(sentiment_data)}")
                        
                        if msg_type == "stop_recording":
                            break

                sender_task = asyncio.create_task(sender())
                receiver_task = asyncio.create_task(receiver())
                
                await sender_task
                try:
                    await asyncio.wait_for(receiver_task, timeout=5.0)
                except asyncio.TimeoutError:
                    pass

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

        return ttfe

async def main():
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Error: '{AUDIO_DIR}' not found.")
        return

    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    if not wav_files:
        print(f"❌ No .wav files found in '{AUDIO_DIR}'")
        return

    print(f"🚀 Starting Gladia Sentiment Latency Test ({len(wav_files)} files)...")
    print("-" * 60)

    client = GladiaEmotionClient()
    results = []

    for wav_file in wav_files:
        latency = await client.process_file(wav_file)
        if latency:
            results.append(latency)
            print(f"   ⏱️  Time to First Emotion: {latency:.2f} ms")
        else:
            print("   ⚠️ No emotion detected.")
        
        await asyncio.sleep(1)

    print("\n" + "="*60)
    print("🏆 GLADIA SENTIMENT LATENCY REPORT")
    print("="*60)
    if results:
        avg_latency = sum(results) / len(results)
        print(f"Files Processed: {len(results)}/{len(wav_files)}")
        print(f"Average Latency: {avg_latency:.2f} ms")
    else:
        print("❌ No sentiment data captured (Feature likely unavailable on this plan/endpoint).")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")