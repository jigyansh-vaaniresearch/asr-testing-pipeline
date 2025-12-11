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
CHUNK_SIZE = 3200
CHUNK_SLEEP = 0.1

class GladiaEmotionClient:
    def __init__(self):
        self.api_key = os.getenv("GLADIA_API_KEY")
        if not self.api_key:
            raise ValueError("GLADIA_API_KEY not found in .env")

    async def get_ws_url(self):
        """Initialize session and get WebSocket URL"""
        async with httpx.AsyncClient() as client:
            # Fix: Removed 'audio_intelligence_config' which caused 400 error
            # We will try to enable sentiment via a simpler config or just test basic connection first
            # If sentiment is not supported in V2 Live Init for your plan, this prevents the crash.
            # We added 'messages_config' to ensure we get all event types.
            resp = await client.post(
                "https://api.gladia.io/v2/live",
                headers={"x-gladia-key": self.api_key},
                json={
                    "sample_rate": 16000,
                    "encoding": "wav/pcm",
                    "language_config": {"languages": ["en", "hi"]},
                    # Request all message types to see if intelligence comes through
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
                        
                        # Debug: Print types to see what we get
                        # print(f"Msg Type: {data.get('type')}") 

                        if data.get("type") == "audio_intelligence":
                            if ttfe is None:
                                ttfe = (time.time() - start_time) * 1000
                                print(f"   ✨ Emotion Detected in {ttfe:.2f} ms")
                        
                        if data.get("type") == "stop_recording":
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
        else:
            print("   ⚠️ No emotion detected (Feature might be disabled/unavailable).")
        
        await asyncio.sleep(1)

    print("\n" + "="*60)
    print("🏆 GLADIA SENTIMENT LATENCY REPORT")
    print("="*60)
    if results:
        avg_latency = sum(results) / len(results)
        print(f"Files Processed: {len(results)}/{len(wav_files)}")
        print(f"Average Latency: {avg_latency:.2f} ms")
    else:
        print("❌ No sentiment data captured.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")