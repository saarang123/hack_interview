from dotenv import load_dotenv
from time import sleep
import soundcard as sc
import numpy as np
import json

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

# from constants import DEEPGRAM_API_KEY
DEEPGRAM_API_KEY = "e7c8746b1d34c6d239d393a9e7e18cabbb2f2958"
from constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

SPEAKER_ID = str(sc.default_speaker().name)

# We will collect the is_final=true messages here so we can use them when the person finishes speaking
is_finals = []

def main():
    try:
        # Initialize the Deepgram client with default options
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY)

        dg_connection = deepgram.listen.websocket.v("1")

        def on_open(self, open, **kwargs):
            print("Connection Open")

        def on_message(self, result, **kwargs):
            global is_finals
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            if result.is_final:
                print(f"Message: {result.to_json()}")
                is_finals.append(sentence)

                if result.speech_final:
                    utterance = " ".join(is_finals)
                    print(f"Speech Final: {utterance}")
                    is_finals = []
                else:
                    print(f"Is Final: {sentence}")
            else:
                print(f"Interim Results: {sentence}")

        def on_metadata(self, metadata, **kwargs):
            print(f"Metadata: {metadata}")

        def on_speech_started(self, speech_started, **kwargs):
            print("Speech Started")

        def on_utterance_end(self, utterance_end, **kwargs):
            global is_finals
            if len(is_finals) > 0:
                utterance = " ".join(is_finals)
                print(f"Utterance End: {utterance}")
                is_finals = []

        def on_close(self, close, **kwargs):
            print("Connection Closed")

        def on_error(self, error, **kwargs):
            print(f"Handled Error: {error}")

        def on_unhandled(self, unhandled, **kwargs):
            print(f"Unhandled Websocket Message: {unhandled}")

        options: LiveOptions = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            interim_results=False,
            sample_rate=SAMPLE_RATE,
            channels=1,
        )

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to connect to Deepgram")
            return

        print("Recording system audio...")

        mic = sc.get_microphone(id=SPEAKER_ID, include_loopback=True)
        try:
            # Open the microphone recorder
            with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
                cnt = 0
                while True:
                    # Record a small chunk of audio data
                    audio_chunk = recorder.record(numframes=SAMPLE_RATE // 10)  # 0.1 second chunks
                    audio_chunk_int16 = np.int16(audio_chunk * 32767)

                    # Convert to bytes for transmission
                    audio_bytes = audio_chunk_int16.tobytes()

                    dg_connection.send(audio_bytes)

                    if cnt % 30 == 0:
                        keep_alive_msg = json.dumps({"type": "KeepAlive"})
                        dg_connection.send(keep_alive_msg)

        except KeyboardInterrupt:
            print("Stopping recording...")

        # Indicate that we've finished
        dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


if __name__ == "__main__":
    main()


# from deepgram import DeepgramClient, PrerecordedOptions

# DEEPGRAM_API_KEY = "e7c8746b1d34c6d239d393a9e7e18cabbb2f2958"

# AUDIO_URL = {
#     "url": "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
# }

# def main():
#     try:
#         deepgram = DeepgramClient(DEEPGRAM_API_KEY)

#         options = PrerecordedOptions(
#             model="nova-2",
#             language="en",
#             smart_format=True,
#         )

#         response = deepgram.listen.prerecorded.v("1").transcribe_url(AUDIO_URL, options)
#         print(response.to_json(indent=4))

#     except Exception as e:
#         print(f"Exception: {e}")

# if __name__ == "__main__":
#     main()