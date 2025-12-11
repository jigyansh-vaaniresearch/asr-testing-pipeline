import os
import time
import json
import abc
import asyncio
import base64
import websockets
import httpx
import urllib.parse

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents


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

        # NEW latency variables
        self.audio_start_time = None
        self.first_token_time = None
        self.latency = None

    def mark_audio_start(self):
        """Mark when the FIRST audio chunk is sent"""
        if self.audio_start_time is None:
            self.audio_start_time = time.time()

    def _mark_latency(self):
        """Mark when the FIRST token arrives"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            self.latency = (self.first_token_time - self.audio_start_time) * 1000
            print(f"\n[LATENCY]: {self.latency:.2f} ms")

    @abc.abstractmethod
    async def connect(self): pass

    @abc.abstractmethod
    async def process_audio(self, audio_chunk): pass

    @abc.abstractmethod
    async def finish(self): pass
    
    def get_latency(self):
        return self.latency


# 1. DEEPGRAM (WebSocket)


class DeepGramHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
        self.dg_socket = None

    async def connect(self):
        try:
            options = LiveOptions(
                model=self.config.get("model_name", "nova-2"),
                language=self.config.get("language", "hi"),
                smart_format=True,
                encoding="linear16", 
                sample_rate=16000,
                interim_results=True
            )

            def on_message(sender, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return
                
                self._mark_latency()
                
                if result.is_final:
                    self.transcript.append(sentence)
                    print(f"\r[DG Final]: {sentence}", flush=True)
                else:
                    print(f"\r[DG Partial]: {sentence}", end="", flush=True)

            def on_error(sender, error, **kwargs):
                print(f"\n[DG Error]: {error}")

            self.dg_socket = self.client.listen.live.v("1")

            self.dg_socket.on(LiveTranscriptionEvents.Transcript, on_message)
            self.dg_socket.on(LiveTranscriptionEvents.Error, on_error)

            if self.dg_socket.start(options) is False:
                raise Exception("Failed to start Deepgram connection")

            print(" Deepgram Connected")

        except Exception as e:
            print(f" Deepgram Connection Failed: {e}")
            raise

    async def process_audio(self, chunk):
        self.mark_audio_start()
        if self.dg_socket:
            self.dg_socket.send(chunk)

    async def finish(self):
        if self.dg_socket:
            self.dg_socket.finish() 
        return " ".join(self.transcript)

# 2. ASSEMBLYAI (SDK)
# class AssemblyAIHandler(ASRModel):
#     def __init__(self, config):
#         super().__init__(config)
#         aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
#         self.transcriber = None

#     async def connect(self):
#         self.transcriber = aai.RealtimeTranscriber(
#             sample_rate=16000,
#             on_data=self._on_data,
#             on_error=self._on_error
#         )
#         self.transcriber.connect()
#         self.start_time = time.time()

#     def _on_data(self, transcript: aai.RealtimeTranscript):
#         if not transcript.text: return
#         if isinstance(transcript, aai.RealtimeFinalTranscript):
#             self.transcript.append(transcript.text)
#             print(f"\r[AAI Final]: {transcript.text}", flush=True)
#         else:
#             self._mark_latency()
#             print(f"\r[AAI Partial]: {transcript.text}", end="", flush=True)

#     def _on_error(self, error: aai.RealtimeError):
#         print("An error occured:", error)

#     async def process_audio(self, chunk):
#         self.transcriber.stream(chunk)

#     async def finish(self):
#         self.transcriber.close()
#         return " ".join(self.transcript)
# 2. ASSEMBLYAI (SDK)
import os
import assemblyai as aai
# Import the new V3 streaming classes
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    TurnEvent,
    StreamingError
)

# 2. ASSEMBLYAI (SDK v3 - Universal Streaming)
# 2. ASSEMBLYAI (SDK v3 - Universal Streaming)
class AssemblyAIHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.client = None

    async def connect(self):
        try:
            # 1. Configure the Client
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com"
                )
            )

            # 2. Define Callbacks
            def on_turn(sender, event: TurnEvent):
                if not event.transcript: return
                self._mark_latency()
                
                if event.end_of_turn:
                    self.transcript.append(event.transcript)
                    print(f"\r[AAI Final]: {event.transcript}", flush=True)
                else:
                    print(f"\r[AAI Partial]: {event.transcript}", end="", flush=True)

            def on_error(sender, error: StreamingError):
                print(f"\n[AAI Error]: {error}")

            # 3. Register Callbacks
            self.client.on(StreamingEvents.Turn, on_turn)
            self.client.on(StreamingEvents.Error, on_error)

            # 4. Connect
            self.client.connect(
                StreamingParameters(
                    sample_rate=16000,
                    encoding="pcm_s16le",
                    format_turns=True
                )
            )
            
            self.start_time = time.time()
            print(" AssemblyAI Connected (Universal Streaming)")

        except Exception as e:
            print(f" AssemblyAI Connection Failed: {e}")
            raise

    async def process_audio(self, chunk):
        self.mark_audio_start()
        if self.client:
            self.client.stream(chunk)

    async def finish(self):
        if self.client:
            # Use 'disconnect()' instead of 'close()'
            self.client.disconnect()
        return " ".join(self.transcript)
# 3. GLADIA (Raw WebSocket V2)
class GladiaHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.ws = None

    async def connect(self):
        try:
            # 1. Initiate Session via REST
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.gladia.io/v2/live",
                    headers={"x-gladia-key": self.api_key},
                    json={
                        "sample_rate": 16000, 
                        "encoding": "wav/pcm", 
                        "language_config": {"languages": ["en", "hi"]}
                    }
                )
                
                if resp.status_code != 200 and resp.status_code != 201:
                    print(f"Gladia Init Failed: {resp.text}")
                    raise Exception(f"Gladia Init Failed with status {resp.status_code}")
                
                data = resp.json()
                url = data['url']

            # 2. Connect via WS
            self.ws = await websockets.connect(url)
            self.start_time = time.time()
            print(" Gladia Connected")
            
            # Start listener loop in background
            asyncio.create_task(self._listen())

        except Exception as e:
            print(f" Gladia Connection Failed: {e}")
            raise

    async def _listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                # Gladia V2 structure: type='transcript' and data.is_final
                if data.get('type') == 'transcript':
                    utterance = data.get('data', {}).get('utterance', {})
                    text = utterance.get('text', '')
                    
                    if text:
                        self._mark_latency()
                        if data['data'].get('is_final'):
                            self.transcript.append(text)
                        
                        print(f"\r[Gladia]: {text}", end="", flush=True)
        except Exception as e:
            # Silent fail on disconnect is common in streaming
            pass

    async def process_audio(self, chunk):
        self.mark_audio_start()
        # Base64 encode for Gladia V2
        # V2 Format: { "type": "audio_chunk", "data": { "chunk": "base64..." } }
        if self.ws:
            payload = {
                "type": "audio_chunk", 
                "data": {"chunk": base64.b64encode(chunk).decode("utf-8")}
            }
            await self.ws.send(json.dumps(payload))

    async def finish(self):
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "stop_recording"}))
                await asyncio.sleep(1) # Wait for final transcripts
                await self.ws.close()
            except Exception:
                pass
        return " ".join(self.transcript)

# 4. SPEECHMATICS (Raw WebSocket )
class SpeechmaticsHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")
        self.ws = None

    async def connect(self):
        try:
            language = self.config.get('language', 'en')
            
          
            # 'neu.rt.speechmatics.com' is for Standard/Trial users (EU1).
            # 'eu2.rt.speechmatics.com' is for Enterprise (EU2).
            # If 'neu' fails, check your Speechmatics Dashboard for your specific region.
            url = "wss://neu.rt.speechmatics.com/v2"
            
            print(f"🔌 Connecting to: {url}")
            
            # Old versions used 'extra_headers', which caused your error.
            self.ws = await websockets.connect(
                url, 
                additional_headers={"Authorization": f"Bearer {self.api_key}"}
            )

            # 3. Send Configuration Message
            start_msg = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000
                },
                "transcription_config": {
                    "language": language,
                    "enable_partials": True, 
                    "operating_point": "enhanced" 
                }
            }
            await self.ws.send(json.dumps(start_msg))
            
            self.start_time = time.time()
            print(" Speechmatics Connected")

            # 4. Start Listening Loop
            asyncio.create_task(self._listen())

        except Exception as e:
            print(f" Speechmatics Connection Failed: {e}")
            raise

    async def _listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                msg_type = data.get('message')

                if msg_type == 'Error':
                    print(f"\n[SM Server Error]: {data.get('reason')}")
                    return

                if msg_type in ['AddTranscript', 'AddPartialTranscript']:
                    metadata = data.get('metadata', {})
                    text = metadata.get('transcript', '')
                    
                    if text:
                        self._mark_latency()
                        if msg_type == 'AddTranscript':
                            self.transcript.append(text)
                            print(f"\r[SM Final]: {text}", flush=True)
                        else:
                            print(f"\r[SM Partial]: {text}", end="", flush=True)
        
        except Exception:
            pass

    async def process_audio(self, chunk):
        self.mark_audio_start()
        if self.ws:
            try:
                await self.ws.send(chunk)
            except Exception:
                pass

    async def finish(self):
        if self.ws:
            try:
                await self.ws.send(json.dumps({"message": "EndOfStream"}))
                await asyncio.sleep(1)
                await self.ws.close()
            except Exception:
                pass
        return " ".join(self.transcript)

# 5. SARVAM AI (SDK)
# 5. SARVAM AI (SDK)
import base64
import os
import asyncio
from sarvamai import AsyncSarvamAI

class SarvamHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
        self.ws_context = None
        self.ws = None

    async def connect(self):
        try:
            # 1. TELL THE TRUTH HERE: "I am sending raw PCM 16-bit Little Endian"
            # This 'input_audio_codec' is what the server actually uses to decode.
            self.ws_context = self.client.speech_to_text_streaming.connect(
                model=self.config.get("model_name", "saarika:v2.5"),
                language_code=self.config.get("language_code", "hi-IN"),
                input_audio_codec="pcm_s16le",  # <--- CRITICAL SERVER CONFIG
                sample_rate=16000
            )
            self.ws = await self.ws_context.__aenter__()
            self.start_time = time.time()
            print(" Sarvam AI Connected")
            
            asyncio.create_task(self._listen())
        except Exception as e:
            print(f" Sarvam Connection Failed: {e}")
            raise

    async def _listen(self):
        try:
            async for msg in self.ws:
                # Sarvam sends multiple message types; look for transcripts
                if msg.get("type") == "transcript":
                    text = msg.get("text")
                    if text and text.strip():
                        self._mark_latency()
                        print(f"\r[Sarvam]: {text}", end="", flush=True)
                        self.transcript.append(text)
        except Exception:
            pass

    async def process_audio(self, chunk):
        self.mark_audio_start()
        if self.ws:
            b64_audio = base64.b64encode(chunk).decode('utf-8')
            
            # 2. LIE TO THE SDK HERE: "This is audio/wav"
            # We do this solely to bypass the Pydantic 'literal_error'.
            # The server ignores this because we set 'pcm_s16le' in connect().
            await self.ws.transcribe(
                audio=b64_audio,
                encoding="audio/wav",  # <--- CRITICAL SDK BYPASS
                sample_rate=16000
            )

    async def finish(self):
        if self.ws_context:
            try:
                await self.ws_context.__aexit__(None, None, None)
            except Exception:
                pass
        return " ".join(self.transcript)

# 6. ELEVENLABS (Scribe v2 - Realtime STT)
class ElevenLabsHandler(ASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.ws = None

    async def connect(self):
        try:
            url = (
                "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
                f"?model_id=scribe_v2"
                f"&audio_format=pcm_16000"
                f"&token={self.api_key}"
            )
            
            print(f"🔌 Connecting to ElevenLabs Scribe...")

            # Connect without extra headers (Auth is in URL now)
            self.ws = await websockets.connect(url)

            self.start_time = time.time()
            print(" ElevenLabs Connected")
            
            asyncio.create_task(self._listen())

        except Exception as e:
            print(f" ElevenLabs Connection Failed: {e}")
            raise

    async def _listen(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                msg_type = data.get("type")
                
                # Scribe v2 Events
                if msg_type in ["partial_transcript", "final_transcript"]:
                    # Extract text from the event structure
                    text = data.get("text")
                    if not text and "channel" in data:
                        text = data["channel"]["alternatives"][0]["transcript"]

                    if text:
                        self._mark_latency()
                        is_final = data.get("is_final") or msg_type == "final_transcript"
                        
                        if is_final:
                            self.transcript.append(text)
                            print(f"\r[11Labs Final]: {text}", flush=True)
                        else:
                            print(f"\r[11Labs Partial]: {text}", end="", flush=True)

        except Exception:
            pass

    async def process_audio(self, chunk):
        if self.ws:
            try:
                # Must be a JSON object with "audio_chunk" (Base64)
                payload = {
                    "type": "audio_chunk",
                    "audio_chunk": base64.b64encode(chunk).decode("utf-8")
                }
                await self.ws.send(json.dumps(payload))
            except Exception:
                pass

    async def finish(self):
        if self.ws:
            try:
                # Send empty chunk or close to signal end
                await self.ws.send(json.dumps({"type": "audio_chunk", "audio_chunk": ""}))
                await asyncio.sleep(0.5)
                await self.ws.close()
            except Exception:
                pass
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
        self.mark_audio_start()
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