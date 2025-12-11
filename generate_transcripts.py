import asyncio
import websockets
import json
import os
import glob
import time
import base64
import httpx
import requests
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIGURATION ---
INPUT_DIR = "audio_samples2"
OUTPUT_DIR = "raw_transcripts2"
SAMPLE_RATE = 16000
LANGUAGE = "hi"

load_dotenv()
KEYS = {
    "DEEPGRAM": os.getenv("DEEPGRAM_API_KEY"),
    "ASSEMBLYAI": os.getenv("ASSEMBLYAI_API_KEY"),
    "GLADIA": os.getenv("GLADIA_API_KEY"),
    "SPEECHMATICS": os.getenv("SPEECHMATICS_API_KEY"),
    "ELEVENLABS": os.getenv("ELEVENLABS_API_KEY"),
    "CARTESIA": os.getenv("CARTESIA_API_KEY"),
}

def save_result(path, transcript, confidence):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Confidence: {confidence}\n")
            f.write(f"Transcript: {transcript}")
    except Exception as e:
        print(f"Error saving {path}: {e}")

# 1. CLIENTS (returning Tuple: text, confidence)

class SarvamGenerator:
    def run(self, audio_path):
        try:
            url = "https://api.sarvam.ai/speech-to-text"
            # Sarvam uses 'api-subscription-key' in the header
            headers = {"api-subscription-key": KEYS["SARVAM"]}
            
            with open(audio_path, "rb") as f:
                # Multipart form upload
                files = {
                    "file": (os.path.basename(audio_path), f, "audio/wav")
                }
                data = {
                    "model": "saarika:v2.5", # Standard model for STT
                    "language_code": "hi-IN", # Explicitly set Hindi
                    "with_timestamps": "false"
                }
                
                resp = requests.post(url, headers=headers, files=files, data=data)
            
            if resp.status_code != 200:
                return "", 0.0
            
            result = resp.json()
            transcript = result.get("transcript", "")
            

            return transcript, 0.0 

        except Exception as e:
            print(f"Sarvam Error: {e}")
            return "", 0.0
class GladiaGenerator:
    async def run(self, audio_path):
        transcript_parts = []
        confidences = []
        try:
            # Init
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.gladia.io/v2/live",
                    headers={"x-gladia-key": KEYS["GLADIA"]},
                    json={
                        "sample_rate": SAMPLE_RATE,
                        "encoding": "wav/pcm",
                        "language_config": {"languages": ["hi", "en"]}
                    }
                )
                if resp.status_code not in [200, 201]: return "", 0.0
                url = resp.json()['url']

            # Socket
            async with websockets.connect(url) as ws:
                async def send():
                    with open(audio_path, "rb") as f:
                        while chunk := f.read(3200):
                            payload = {"type": "audio_chunk", "data": {"chunk": base64.b64encode(chunk).decode("utf-8")}}
                            await ws.send(json.dumps(payload))
                            await asyncio.sleep(0.01)
                    await ws.send(json.dumps({"type": "stop_recording"}))

                async def receive():
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get('type') == 'transcript' and data.get('data', {}).get('is_final'):
                            utt = data['data']['utterance']
                            transcript_parts.append(utt['text'])
                            if 'confidence' in utt: confidences.append(utt['confidence'])
                
                send_task = asyncio.create_task(send())
                recv_task = asyncio.create_task(receive())
                await send_task
                await asyncio.sleep(2)
                if not recv_task.done(): recv_task.cancel()
            
            avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
            return " ".join(transcript_parts), round(avg_conf, 4)
        except: return "", 0.0

class SpeechmaticsGenerator:
    async def run(self, audio_path):
        try:
            url = "https://asr.api.speechmatics.com/v2/jobs"
            headers = {"Authorization": f"Bearer {KEYS['SPEECHMATICS']}"}
            
            with open(audio_path, "rb") as f:
                files = {
                    "data_file": (os.path.basename(audio_path), f, "application/octet-stream"),
                    "config": (None, json.dumps({"type": "transcription", "transcription_config": { "language": LANGUAGE }}), "application/json")
                }
                resp = requests.post(url, headers=headers, files=files)
            
            if resp.status_code != 201: return "", 0.0
            job_id = resp.json()['id']

            # Poll
            job_url = f"{url}/{job_id}"
            while True:
                status = requests.get(job_url, headers=headers).json()['job']['status']
                if status == 'done': break
                elif status in ['rejected', 'failed']: return "", 0.0
                await asyncio.sleep(2)

            # Get JSON for Confidence
            res = requests.get(f"{job_url}/transcript?format=json-v2", headers=headers).json()
            
            words = []
            confs = []
            for item in res.get('results', []):
                alt = item['alternatives'][0]
                words.append(alt['content'])
                confs.append(alt['confidence'])
            
            avg_conf = sum(confs)/len(confs) if confs else 0.0
            return " ".join(words), round(avg_conf, 4)

        except: return "", 0.0

class CartesiaGenerator:
    async def run(self, audio_path):
        transcript_parts = []
        try:
            url = f"wss://api.cartesia.ai/stt/websocket?api_key={KEYS['CARTESIA']}&cartesia_version=2024-06-10"
            async with websockets.connect(url) as ws:
                # FIXED: int(SAMPLE_RATE)
                await ws.send(json.dumps({
                    "type": "start",
                    "model_id": "ink-whisper",
                    "transcription_config": {"language": LANGUAGE, "diarize": False, "encoding": "pcm_s16le", "sample_rate": int(SAMPLE_RATE)}
                }))

                with open(audio_path, "rb") as f:
                    f.seek(44) # Skip WAV header
                    while chunk := f.read(8192):
                        await ws.send(chunk)
                        await asyncio.sleep(0.01)
                await ws.send(json.dumps({"type": "close"}))

                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "transcription":
                        transcript_parts.append(data.get("text", ""))
                    elif data.get("type") == "done":
                        break
            
            # Cartesia currently doesn't expose confidence in WS text messages
            return " ".join(transcript_parts), 0.99 # Mock confidence as placeholder
        except: return "", 0.0

# SDK WRAPPERS
def run_deepgram(audio_path):
    from deepgram import DeepgramClient, PrerecordedOptions
    try:
        dg = DeepgramClient(KEYS["DEEPGRAM"])
        with open(audio_path, "rb") as f: payload = {"buffer": f.read()}
        opts = PrerecordedOptions(model="nova-2", language=LANGUAGE, smart_format=True)
        res = dg.listen.rest.v("1").transcribe_file(payload, opts)
        alt = res.results.channels[0].alternatives[0]
        return alt.transcript, alt.confidence
    except: return "", 0.0

def run_assembly(audio_path):
    import assemblyai as aai
    try:
        aai.settings.api_key = KEYS["ASSEMBLYAI"]
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(language_code=LANGUAGE)
        res = transcriber.transcribe(audio_path, config=config)
        return res.text, res.confidence
    except: return "", 0.0

def run_elevenlabs(audio_path):
    from elevenlabs.client import ElevenLabs
    try:
        client = ElevenLabs(api_key=KEYS["ELEVENLABS"])
        with open(audio_path, "rb") as f:
            res = client.speech_to_text.convert(file=f, model_id="scribe_v1")
        return res.text, 0.0 # No confidence available
    except: return "", 0.0

# ORCHESTRATOR
async def main():
    if not os.path.exists(INPUT_DIR): return
    files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    gladia = GladiaGenerator()
    speechmatics = SpeechmaticsGenerator()
    cartesia = CartesiaGenerator()

    for f_path in files:
        stem = Path(f_path).stem
        print(f"\nProcessing: {stem}")

        tasks = {
            "Deepgram": lambda: run_deepgram(f_path),
            "AssemblyAI": lambda: run_assembly(f_path),
            "ElevenLabs": lambda: run_elevenlabs(f_path),
            "Gladia": lambda: gladia.run(f_path),
            "Speechmatics": lambda: speechmatics.run(f_path),
            "Cartesia": lambda: cartesia.run(f_path),
            "Sarvam": lambda: sarvam.run(f_path)
        }

        for model, func in tasks.items():
            dest_file = os.path.join(OUTPUT_DIR, model, f"{stem}.txt")
            if os.path.exists(dest_file): continue
            
            print(f"   🎙️  {model}...", end="\r")
            try:
                if model in ["Gladia", "Speechmatics", "Cartesia"]:
                    txt, conf = await func()
                else:
                    txt, conf = func()
                
                save_result(dest_file, txt, conf)
                print(f"   ✅ {model} (Conf: {conf})     ")
            except Exception as e:
                print(f"   ❌ {model} Failed")

if __name__ == "__main__":
    asyncio.run(main())