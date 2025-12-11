import os
import time
import json
import abc
import asyncio
import base64
import websockets
import httpx
from deepgram import DeepgramClient
# Fix: Import LiveOptions from the specific submodule for v3 compatibility
try:
    from deepgram.clients.live.v1 import LiveOptions
except ImportError:
    try:
        from deepgram import LiveOptions
    except ImportError:
        # Fallback if class not found, we will use dict
        LiveOptions = None

from groq import Groq
import assemblyai as aai
from google.cloud import speech
import speechmatics
from sarvamai import AsyncSarvamAI

class ASRModel(abc.ABC):
    """Abstract Base Class for all ASR Models"""
    def __init__(self, config):
        self.config = config
        self.transcript = []
        self.first_token_received = False
        self.start_time = 0
        self.latency = 0
        self.chunk_count = 0
        self.chunk_latencies = []

    def _mark_latency(self):
        """Helper to record latency on first token"""
        if not self.first_token_received:
            self.latency = (time.time() - self.start_time) * 1000
            self.first_token_received = True
            print(f"\n[FIRST TOKEN LATENCY]: {self.latency:.2f} ms")

    @abc.abstractmethod
    async def connect(self): pass

    @abc.abstractmethod
    async def process_audio(self, audio_chunk): pass

    @abc.abstractmethod
    async def finish(self): pass
    
    def get_latency(self): return self.latency

# 1. DEEPGRAM (WebSocket)
class DeepGramHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.ws = None

    async def connect(self):
        url = f"wss://api.deepgram.com/v1/listen?model={self.config.get('model_name', 'nova-2')}&language=hi&smart_format=true&encoding=linear16&sample_rate=16000"
        headers = {"Authorization": f"Token {self.api_key}"}
        self.ws = await websockets.connect(url, additional_headers=headers)
        self.start_time = time.time()
        asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                if data.get('channel', {}).get('alternatives'):
                    transcript = data['channel']['alternatives'][0]['transcript']
                    if transcript:
                        self._mark_latency()
                        if data.get('is_final'):
                            self.transcript.append(transcript)
                            print(f"\r[DG Final]: {transcript}", end="", flush=True)
                        else:
                            print(f"\r[DG Partial]: {transcript}", end="", flush=True)
        except Exception as e:
            print(f"DG WS error: {e}")

    async def process_audio(self, chunk):
        self.chunk_count += 1
        chunk_start = time.time()
    
        await self.ws.send(chunk)
    
        chunk_end = time.time()
        chunk_latency = (chunk_end - chunk_start) * 1000
        self.chunk_latencies.append(chunk_latency)
        print(f"[CHUNK {self.chunk_count} ({chunk_latency:.0f}ms)]", end=" ")

    async def finish(self):
        if self.ws:
            await self.ws.close()
        
        full_transcript = " ".join(self.transcript)
        avg_chunk_latency = sum(self.chunk_latencies) / len(self.chunk_latencies) if self.chunk_latencies else 0
        
        print(f"\n{'='*50}")
        print(f"FULL TRANSCRIPT ({self.chunk_count} chunks, avg {avg_chunk_latency:.1f}ms/chunk):")
        print(f"{full_transcript}")
        print('='*50)
        
        return full_transcript

# 2. ASSEMBLYAI (SDK)
class AssemblyAIHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.transcriber = None

    async def connect(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self._on_data,
            on_error=self._on_error
        )
        self.transcriber.connect()
        self.start_time = time.time()
        print("✅ AssemblyAI Connected")

    def _on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text: 
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.transcript.append(transcript.text)
            print(f"\r[AAI Final]: {transcript.text}", flush=True)
        else:
            self._mark_latency()
            print(f"\r[AAI Partial]: {transcript.text}", end="", flush=True)

    def _on_error(self, error: aai.RealtimeError):
        print(f"\nAssemblyAI Error: {error}", flush=True)

    async def process_audio(self, chunk):
        try:
            self.chunk_count += 1
            self.transcriber.stream(chunk)
            
            if self.chunk_count % 50 == 0:
                print(f"[CHUNK {self.chunk_count}]", end=" ", flush=True)
        except Exception as e:
            print(f"Error streaming audio: {e}")

    async def finish(self):
        print("\n\nFinishing AssemblyAI stream...")
        try:
            self.transcriber.close()
            # Give time for final callbacks
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error closing transcriber: {e}")

        full_transcript = " ".join(self.transcript)
        
        # Save Result
        os.makedirs("results", exist_ok=True)
        filename = f"results/assemblyai.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_transcript)
            
        print(f"\n{'='*50}")
        print(f"FULL TRANSCRIPT SAVED: {filename}")
        print('='*50)
        return full_transcript

# 3. GLADIA (Raw WebSocket V2)
class GladiaHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.ws = None

    async def connect(self):
        # 1. Initiate Session via REST
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.gladia.io/v2/live",
                headers={"x-gladia-key": self.api_key},
                json={"sample_rate": 16000, "encoding": "wav/pcm", "language_config": {"languages": ["en", "hi"]}}
            )
            if resp.status_code != 200:
                print(f"Gladia Init Failed: {resp.text}")
                raise Exception("Gladia Init Failed")
            data = resp.json()
            url = data['url']

        # 2. Connect via WS
        self.ws = await websockets.connect(url)
        self.start_time = time.time()
        
        # Start listener loop in background
        asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                if data['type'] == 'transcript' and data['data']['utterance']['text']:
                    text = data['data']['utterance']['text']
                    self._mark_latency()
                    if data['data']['is_final']:
                        self.transcript.append(text)
                    print(f"\r[Gladia]: {text}", end="", flush=True)
        except Exception:
            pass

    async def process_audio(self, chunk):
        # Base64 encode for Gladia
        payload = {"type": "audio_chunk", "data": {"chunk": base64.b64encode(chunk).decode("utf-8")}}
        await self.ws.send(json.dumps(payload))

    async def finish(self):
        if self.ws:
            await self.ws.send(json.dumps({"type": "stop_recording"}))
            await asyncio.sleep(1) # Wait for finals
            await self.ws.close()
        return " ".join(self.transcript)

# 4. SPEECHMATICS (SDK)
class SpeechmaticsHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.sm_client = None
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")

    async def connect(self):
        conn = speechmatics.models.ConnectionSettings(
            url=f"wss://eu2.rt.speechmatics.com/v2/{self.config.get('language', 'en')}",
            auth_token=self.api_key
        )
        # Initialize WebsocketClient
        self.sm_client = speechmatics.client.WebsocketClient(conn)
        
        def text_handler(msg):
            txt = msg['metadata']['transcript']
            if txt:
                self._mark_latency()
                print(f"\r[SM]: {txt}", end="", flush=True)
                if msg['message_type'] == 'AddTranscript': # Final
                    self.transcript.append(txt)

        self.sm_client.add_event_handler(
            speechmatics.models.ServerMessageType.AddPartialTranscript, text_handler
        )
        self.sm_client.add_event_handler(
            speechmatics.models.ServerMessageType.AddTranscript, text_handler
        )
        
        # Note: Speechmatics SDK run() is blocking. 
        # In a real async app, you'd run this in a separate thread or use a raw websocket implementation
        # For this demo script structure, we might need a workaround or raw WS if SDK blocks.
        # This is a placeholder for standard SDK usage.
        pass 

    async def process_audio(self, chunk):
        # In a real impl, you'd feed the AudioProcessor buffer here
        pass

    async def finish(self):
        return " ".join(self.transcript)

# 5. SARVAM AI (SDK)
class SarvamHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
        self.ws_context = None
        self.ws = None

    async def connect(self):
        # Connect
        self.ws_context = self.client.speech_to_text_streaming.connect(
            language_code=self.config.get("language_code", "hi-IN")
        )
        self.ws = await self.ws_context.__aenter__()
        self.start_time = time.time()
        asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for msg in self.ws:
                if msg.get("type") == "transcript":
                    text = msg.get("text")
                    self._mark_latency()
                    print(f"\r[Sarvam]: {text}", end="", flush=True)
                    self.transcript.append(text)
        except Exception: pass

    async def process_audio(self, chunk):
        await self.ws.transcribe(audio=base64.b64encode(chunk).decode('utf-8'))

    async def finish(self):
        if self.ws_context:
            await self.ws_context.__aexit__(None, None, None)
        return " ".join(self.transcript)

# 6. ELEVENLABS (Raw WebSocket)
class ElevenLabsHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.ws = None

    async def connect(self):
        url = f"wss://api.elevenlabs.io/v1/speech-to-text/stream-input?model_id=scribe_v2"
        self.ws = await websockets.connect(url)
        # Auth message often required first
        await self.ws.send(json.dumps({
            "text": " ", 
            "xi-api-key": self.api_key
        }))
        self.start_time = time.time()
        asyncio.create_task(self._listen())

    async def _listen(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("text"):
                self._mark_latency()
                print(f"\r[11Labs]: {data['text']}", end="", flush=True)
                if data.get("is_final"):
                    self.transcript.append(data['text'])

    async def process_audio(self, chunk):
        payload = {"audio_event": {"audio_base_64": base64.b64encode(chunk).decode("utf-8")}}
        await self.ws.send(json.dumps(payload))

    async def finish(self):
        if self.ws:
            await self.ws.close()
        return " ".join(self.transcript)

# 7. CARTESIA (Raw WebSocket)
class CartesiaHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("CARTESIA_API_KEY")
        self.ws = None

    async def connect(self):
        url = f"wss://api.cartesia.ai/stt/websocket?api_key={self.api_key}"
        self.ws = await websockets.connect(url)
        self.start_time = time.time()
        asyncio.create_task(self._listen())

    async def _listen(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "transcript":
                self._mark_latency()
                print(f"\r[Cartesia]: {data['text']}", end="", flush=True)
                self.transcript.append(data['text'])

    async def process_audio(self, chunk):
        # Send raw bytes for Cartesia
        await self.ws.send(chunk)

    async def finish(self):
        if self.ws:
            await self.ws.send(json.dumps({"type": "finalize"}))
            await asyncio.sleep(0.5)
            await self.ws.close()
        return " ".join(self.transcript)

# 8. GOOGLE CHIRP (Cloud Speech SDK)
class GoogleHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = speech.SpeechClient()
        self.queue = asyncio.Queue()
        self.stream_task = None

    async def connect(self):
        self.start_time = time.time()
        self.stream_task = asyncio.create_task(self._stream_generator())

    async def _stream_generator(self):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            model="latest_long", 
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        def request_generator():
            while True:
                chunk = asyncio.run_coroutine_threadsafe(self.queue.get(), asyncio.get_event_loop()).result()
                if chunk is None: break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        requests = request_generator()
        # Note: streaming_recognize is blocking in sync mode, so this gen approach 
        # needs to be carefully handled in pure asyncio. 
        # Google's async client is preferred for main.py's asyncio loop.
        # But for this snippet we assume standard client usage.
        try:
            responses = self.client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                for result in response.results:
                    if result.alternatives:
                        text = result.alternatives[0].transcript
                        self._mark_latency()
                        print(f"\r[Google]: {text}", end="", flush=True)
                        if result.is_final:
                            self.transcript.append(text)
        except Exception as e:
            print(f"Google Stream Error: {e}")

    async def process_audio(self, chunk):
        await self.queue.put(chunk)

    async def finish(self):
        await self.queue.put(None)
        return " ".join(self.transcript)

# 9. GROQ (Fast Batch / Simulated)
class GroqHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.temp_file = "temp_groq.wav"

    async def connect(self):
        self.start_time = time.time()

    async def process_audio(self, chunk):
        # Groq doesn't support streaming yet, so we chunk-file it
        with open(self.temp_file, "wb") as f:
            f.write(chunk)
        
        with open(self.temp_file, "rb") as f:
            res = self.client.audio.transcriptions.create(
                file=(self.temp_file, f.read()),
                model="whisper-large-v3",
                language="en"
            )
        if res.text:
            self._mark_latency()
            print(f"\r[Groq]: {res.text}", end="", flush=True)
            self.transcript.append(res.text)

    async def finish(self):
        if os.path.exists(self.temp_file): os.remove(self.temp_file)
        return " ".join(self.transcript)


def get_model_handler(key, config):
    handlers = {
        "deepgram": DeepGramHandler,
        "assemblyai": AssemblyAIHandler,
        "gladia": GladiaHandler,
        "speechmatics": SpeechmaticsHandler,
        "sarvam": SarvamHandler,
        "elevenlabs": ElevenLabsHandler,
        "cartesia": CartesiaHandler,
        "google": GoogleHandler,
        "groq": GroqHandler
    }
    return handlers[key](config['models'].get(key, {}))
