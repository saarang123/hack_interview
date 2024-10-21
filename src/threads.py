from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from loguru import logger
import asyncio
import soundcard as sc
import time
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramClientOptions
)
import numpy as np
import json
import sys
import openai

from .constants import DEEPGRAM_API_KEY, OPENAI_API_KEY
from .constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE


SYSTEM_PROMPT = f"""You are a sales agent for Avoca Air Condioning company.
You will receive an audio transcription of the question. It may not be complete. 
Treat this as a phone call you are the sales agent. The audio you receive is the users response on the call. 
You need to understand the question and write an answer to it based on the following script: \n
Start with an introduction: Thank you for calling Dooley Service Pro, this is Sarah your virtual assistant how may I help you today!

If you have already greeted/introduced yourself to the user, there is no need to do it again. 

If the user is querying about service, frame a response to collect information on:
Problem / issue they are facing
Age of their system
Name
Address
Callback Number
Email

Further clarifications after this could be based on when they are interested in scheduling it and appropriately responding saying its been scheduled.

FAQ:
What hours are you open?
8-5 Monday Though Friday, 5 days a week
When can we speak to a live agent?
The earliest that someone will return your call is between 730 and 8:30 AM the next day.
What time can you come out?
We do offer open time frames. Our dispatcher will keep you updated throughout the day. 
Is there a service fee to come out?
It’s just $79 for the diagnostic fee unless you are looking to replace your system in which case we can offer a free quote.

Below consists the history of conversation between you and the user. 
The format is User query: <transcript of users query> GPT Response: <response you gave the user in the past>
Don't ask for the same information multiple times and/or request redundant information if the info is in the message history.

"""

SHORTER_INSTRACT = "Concisely respond, limiting your answer to 70 words."
LONGER_INSTRACT = (
    "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."
)

config = DeepgramClientOptions(
    options={"keepalive": "true"} # Comment this out to see the effect of not using keepalive
)

deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, config)
openai.api_key = OPENAI_API_KEY


SPEAKER_ID = str(sc.default_speaker().name)

is_running = True
is_finals = []
is_done = False
transcribed_data = ""

msg_history = ""


# 1. Start Transcription
def start_transcription():
    global is_finals, transcribed_data, is_done, is_running
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
        endpointing=10000
    )

    # Establish Deepgram WebSocket connection
    dg_connection = deepgram_client.listen.websocket.v("1")

    # Define event listeners
    dg_connection.on(LiveTranscriptionEvents.Transcript, handle_transcription)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)

    is_running = True
    is_finals = []
    is_done = False
    transcribed_data = ""

    # Start the connection
    if not dg_connection.start(options):
        logger.error("Failed to connect to Deepgram")
        return False

    return True

# 2. Process Audio (Recording)
def process_audio():
    global is_running

    mic = sc.get_microphone(id=SPEAKER_ID, include_loopback=True)

    try:
        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            logger.info("Started recording system audio...")
            cnt = 0
            assert(is_running)
            while is_running:
                # Record a small chunk of audio data
                audio_chunk = recorder.record(numframes=SAMPLE_RATE // 10)  # 0.1 second chunks
                audio_chunk_int16 = np.int16(audio_chunk * 32767)
                audio_bytes = audio_chunk_int16.tobytes()

                # Send the bytes to the WebSocket connection
                dg_connection.send(audio_bytes)

                if cnt % 5 == 0:
                    keep_alive_msg = json.dumps({"type": "KeepAlive"})
                    dg_connection.send(keep_alive_msg)

                cnt += 1
            print("recording done", cnt)

    except KeyboardInterrupt as e:
        logger.error(f"Error while recording or streaming audio: {e}")

def stop_transcription():
    global is_running, is_done

    is_running = False

    # Send a finalize message to Deepgram
    finalize_msg = json.dumps({"type": "Finalize"})
    dg_connection.send(finalize_msg)

    # Wait for final transcription to be done
    time.sleep(0.2)

    # Close the connection
    dg_connection.finish()
    logger.debug("Transcription done")

    return transcribed_data

# 3. Handle Transcription (Callback)
def handle_transcription(self, result, **kwargs):
    global is_finals, transcribed_data, is_done

    sentence = result.channel.alternatives[0].transcript
    
    # print(result)

    if result.is_final:
        transcribed_data += sentence + " "
        print(f"Final Transcription: {sentence}")

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
    sys.exit(1)

def generate_answer(transcript: str, short_answer: bool = True, temperature: float = 0.4) -> str:
    """
    Generates an answer based on the given transcript using the OpenAI GPT-3.5-turbo model.

    Args:
        transcript (str): The transcript to generate an answer from.
        short_answer (bool): Whether to generate a short answer or not. Defaults to True.
        temperature (float): The temperature parameter for controlling the randomness of the generated answer.

    Returns:
        str: The generated answer.

    Example:
        ```python
        transcript = "Can you tell me about the weather?"
        answer = generate_answer(transcript, short_answer=False, temperature=0.8)
        print(answer)
        ```

    Raises:
        Exception: If the LLM fails to generate an answer.
    """
    global msg_history

    if short_answer:
        system_prompt = SYSTEM_PROMPT + SHORTER_INSTRACT
    else:
        system_prompt = SYSTEM_PROMPT + LONGER_INSTRACT
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt + msg_history},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error
    resp = response["choices"][0]["message"]["content"]
    msg_history += "User query:\n" + transcript + "\n"
    msg_history += "GPT Response:\n" + resp + "\n"
    return resp