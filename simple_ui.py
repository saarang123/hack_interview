import numpy as np
import PySimpleGUI as sg
from loguru import logger
import asyncio

from src.constants import APPLICATION_WIDTH, OFF_IMAGE, ON_IMAGE

from src.threads import (
    start_transcription, 
    process_audio, 
    handle_transcription, 
    stop_transcription,
    generate_answer
)


logger.add("debug.log", level="DEBUG", rotation="3 MB", compression="zip")

def get_text_area(text: str, size: tuple) -> sg.Text:
    """
    Create a text area widget with the given text and size.

    Parameters:
        text (str): The initial text to display in the text area.
        size (tuple): The size of the text area widget.

    Returns:
        sg.Text: The created text area widget.
    """
    return sg.Text(
        text,
        size=size,
        background_color=sg.theme_background_color(),
        text_color="white",
    )


class BtnInfo:
    def __init__(self, state=False):
        self.state = state

with open(ON_IMAGE, 'rb') as f: on_data = f.read()
with open(OFF_IMAGE, 'rb') as f: off_data = f.read()

# All the stuff inside your window:
sg.theme("DarkAmber")  # Add a touch of color
record_status_button = sg.Button(
    image_data=off_data,
    k="-TOGGLE1-",
    border_width=0,
    button_color=(sg.theme_background_color(), sg.theme_background_color()),
    disabled_button_color=(sg.theme_background_color(), sg.theme_background_color()),
    metadata=BtnInfo(),
)

analyzed_text_label = get_text_area("", size=(APPLICATION_WIDTH, 2))
quick_chat_gpt_answer = get_text_area("", size=(APPLICATION_WIDTH, 5))
full_chat_gpt_answer = get_text_area("", size=(APPLICATION_WIDTH, 12))


layout = [
    [sg.Text("Press R to start recording", size=(int(APPLICATION_WIDTH * 0.8), 2)), record_status_button],
    [sg.Text("Press A to analyze the recording")],
    [analyzed_text_label],
    [sg.Text("Short answer:")],
    [quick_chat_gpt_answer],
    [sg.Text("Full answer:")],
    [full_chat_gpt_answer],
    [sg.Button("Cancel")],
]
WINDOW = sg.Window("Avoca AI", layout, return_keyboard_events=True, use_default_focus=False)

audio_transcript = None

while True:
    event, values = WINDOW.read()
    print("in while true", event, values)
    if event in ["Cancel", sg.WIN_CLOSED]:
        logger.debug("Closing...")
        break

    if event == "r:27":  # start recording
        print("event r")
        record_status_button.metadata.state = not record_status_button.metadata.state
        if record_status_button.metadata.state:
            logger.debug("Starting recording...")
            is_transcribed = False
            audio_transcript = None
            assert(start_transcription())
            # asyncio.run(process_audio())
            WINDOW.perform_long_operation(process_audio, "-RECORDING-")
        else:
            logger.debug("Stopping recording...")
            audio_transcript = stop_transcription()
            is_transcribed = True
        record_status_button.update(image_data=on_data if record_status_button.metadata.state else off_data)

    elif event == "a:38":  # send audio to OpenAI Whisper model
        logger.debug("Analyzing audio...")
        analyzed_text_label.update("Start analyzing...")
        
        if not is_transcribed or audio_transcript is None:
            analyzed_text_label.update("Stop recording before analyzing")
            continue

        analyzed_text_label.update(audio_transcript)

        # Generate quick answer:
        quick_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: generate_answer(audio_transcript, short_answer=True, temperature=0.3),
            "-CHAT_GPT SHORT ANSWER-",
        )

        # Generate full answer:
        full_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: generate_answer(audio_transcript, short_answer=False, temperature=0.7),
            "-CHAT_GPT LONG ANSWER-",
        )

    elif event == "-WHISPER COMPLETED-":
        is_transcribed = True
        audio_transcript = values["-WHISPER COMPLETED-"]
        analyzed_text_label.update(audio_transcript)

        # Generate quick answer:
        quick_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: generate_answer(audio_transcript, short_answer=True, temperature=0.3),
            "-CHAT_GPT SHORT ANSWER-",
        )

        # Generate full answer:
        full_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: generate_answer(audio_transcript, short_answer=False, temperature=0.7),
            "-CHAT_GPT LONG ANSWER-",
        )
    elif event == "-CHAT_GPT SHORT ANSWER-":
        quick_chat_gpt_answer.update(values["-CHAT_GPT SHORT ANSWER-"])
    elif event == "-CHAT_GPT LONG ANSWER-":
        full_chat_gpt_answer.update(values["-CHAT_GPT LONG ANSWER-"])