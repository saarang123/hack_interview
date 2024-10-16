from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from loguru import logger
import soundcard as sc
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
    LiveTranscriptionEvents,
    LiveOptions
)

from .constants import DEEPGRAM_API_KEY
from .constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

URL = f"wss://api.deepgram.com/v1/listen?access_token={DEEPGRAM_API_KEY}"

SPEAKER_ID = str(sc.default_speaker().name)

class RecordingThread(QThread):
    transcription_done = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.is_final = False
        self.transcribed_data = None
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        self.dg_connection = self.deepgram.listen.live.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
        
        self.options = LiveOptions(
            model="nova-2", 
            language="en-US", 
            smart_format=True,
        )
        
        # STEP 6: Start the connection

    def run(self):
        # change to post to deepgram socket
        logger.debug("Recording thread started")
        self.dg_connection.start(self.options)
        mic = sc.get_microphone(id=SPEAKER_ID, include_loopback=True)
        try:
            # Open the microphone recorder
            with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
                logger.info("Started recording system audio...")

                while self.is_running:
                    # Record a small chunk of audio data
                    audio_chunk = recorder.record(numframes=SAMPLE_RATE // 10)  # 0.1 second chunks
                    
                    # Convert the audio data to bytes
                    audio_bytes = audio_chunk.tobytes()

                    # Send the bytes to the WebSocket connection
                    self.dg_connection.send(audio_bytes)
                    self.is_final = False
                # recorder.__exit__(None, None, None)
        except Exception as e:
            logger.error(f"Error while recording or streaming audio: {e}")

        logger.debug("Recording thread finished")

    def on_message(self, client, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if result.speech_final:
            self.is_final = True
        logger.debug("received transcription from deepgram socket")
        if len(sentence) == 0:
            return
        self.transcription_data += sentence
        print(f"speaker: {sentence}")

    def on_metadata(self, client, metadata, **kwargs):
        logger.debug("received metadata from deepgram socket")
        print(f"\n\n{metadata}\n\n")
        print("huh")

    def on_error(self, client, error, **kwargs):
        logger.debug("error transcribing from deepgram socket")
        print(f"\n\n{error}\n\n")

    def stop(self):
        self.is_running = False
        self.dg_connection.finish()
        print("stop RecordingThread")
        while not self.is_final:
            pass
        logger.debug("Transcription done")
        return transcription_data

class ChatGPTThread(QThread):
    answer_ready = pyqtSignal(str)

    def __init__(self, llm, audio_transcript, short_answer, temperature):
        super().__init__()
        self.llm = llm
        self.audio_transcript = audio_transcript
        self.short_answer = short_answer
        self.temperature = temperature

    def run(self):
        answer = self.llm.generate_answer(
            self.audio_transcript,
            short_answer=self.short_answer,
            temperature=self.temperature
        )
        self.answer_ready.emit(answer)
