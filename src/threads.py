from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from loguru import logger
import asyncio
import soundcard as sc
import websockets
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
    LiveTranscriptionEvents,
    LiveOptions
)
import numpy as np
import json

from .constants import DEEPGRAM_API_KEY
from .constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)

SPEAKER_ID = str(sc.default_speaker().name)

is_running = True
is_finals = []
is_done = False
transcribed_data = ""


# 1. Start Transcription
def start_transcription():
    global is_finals, transcribed_data, is_done
    global dg_connection

    logger.debug("Starting transcription...")

    # Initialize Deepgram connection with LiveOptions
    options = LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        encoding="linear16",
        interim_results=False,
        sample_rate=SAMPLE_RATE,
        channels=1,
    )

    # Establish Deepgram WebSocket connection
    dg_connection = deepgram_client.listen.websocket.v("1")

    # Define event listeners
    dg_connection.on(LiveTranscriptionEvents.Transcript, handle_transcription)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)

    # Start the connection
    if not dg_connection.start(options):
        logger.error("Failed to connect to Deepgram")
        return False

    is_running = True
    is_finals = []
    is_done = False
    transcribed_data = ""

    return True

# 2. Process Audio (Recording)
async def process_audio():
    global is_running

    mic = sc.get_microphone(id=SPEAKER_ID, include_loopback=True)

    try:
        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            logger.info("Started recording system audio...")
            cnt = 0
            while is_running:
                # Record a small chunk of audio data
                audio_chunk = recorder.record(numframes=SAMPLE_RATE // 10)  # 0.1 second chunks
                audio_chunk_int16 = np.int16(audio_chunk * 32767)
                audio_bytes = audio_chunk_int16.tobytes()

                # Send the bytes to the WebSocket connection
                dg_connection.send(audio_bytes)

                if cnt % 30 == 0:
                    keep_alive_msg = json.dumps({"type": "KeepAlive"})
                    dg_connection.send(keep_alive_msg)

                cnt += 1

    except KeyboardInterrupt as e:
        logger.error(f"Error while recording or streaming audio: {e}")

def stop_transcription():
    global is_running, is_done

    is_running = False

    # Send a finalize message to Deepgram
    finalize_msg = json.dumps({"type": "Finalize"})
    dg_connection.send(finalize_msg)

    # Wait for final transcription to be done
    while not is_done:
        pass

    # Close the connection
    dg_connection.finish()
    logger.debug("Transcription done")

    return transcribed_data

# 3. Handle Transcription (Callback)
def handle_transcription(self, result, **kwargs):
    global is_finals, transcribed_data, is_done

    sentence = result.channel.alternatives[0].transcript
    if len(sentence) == 0:
        return

    if result.is_final:
        is_finals.append(sentence)
        if result.speech_final:
            utterance = " ".join(is_finals)
            transcribed_data += utterance
            print(f"Final Transcription: {utterance}")
            is_finals = []

    if result.from_finalize:
        is_done = True

def on_open(self, open, **kwargs):
    logger.debug("Connection Open")

def on_metadata(self, metadata, **kwargs):
    logger.debug(f"Metadata: {metadata}")

def on_close(self, close, **kwargs):
    logger.debug("Connection Closed")

def on_error(self, error, **kwargs):
    logger.debug(f"Handled Error: {error}")

class ChatGPTThread(QThread):
    answer_ready = pyqtSignal(str)

    def __init__(self, llm, audio_transcript, short_answer, temperature):
        super().__init__()
        self.llm = llm
        self.audio_transcript = audio_transcript
        self.short_answer = short_answer
        self.temperature = temperature

    def run(self):
        print("do i even run")
        answer = self.llm.generate_answer(
            self.audio_transcript,
            short_answer=self.short_answer,
            temperature=self.temperature
        )
        self.answer_ready.emit(answer)