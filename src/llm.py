import openai
from loguru import logger
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

from src.constants import DEEPGRAM_API_KEY, OPENAI_API_KEY, OUTPUT_FILE_NAME


SYSTEM_PROMPT = f"""You are a sales agent for Avoca Air Condioning company.
You will receive an audio transcription of the question. It may not be complete. 
Treat this as a phone call you are the sales agent. The audio you receive is the users response on the call. 
You need to understand the question and write an answer to it based on the following script: \n
Start with an introduction: Thank you for calling Dooley Service Pro, this is Sarah your virtual assistant how may I help you today!

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

"""
SHORTER_INSTRACT = "Concisely respond, limiting your answer to 70 words."
LONGER_INSTRACT = (
    "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."
)


# Install the Deepgram Python SDK
# pip install deepgram-sdk==3.*

class LLMInference:
    def __init__(self):
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        openai.api_key = OPENAI_API_KEY

    def transcribe_audio(self, path_to_file: str = OUTPUT_FILE_NAME) -> str:
        """
        Transcribes an audio file into text.

        Args:
            path_to_file (str, optional): The path to the audio file to be transcribed.

        Returns:
            str: The transcribed text.

        Raises:
            Exception: If the audio file fails to transcribe.
        """
        print("transcribe: ", path_to_file)
        with open(path_to_file, "rb") as audio_file:
            try:
                buffer_data = audio_file.read()

                payload: FileSource = {
                    "buffer": buffer_data,
                }

                options = PrerecordedOptions(
                    model="nova-2",
                    smart_format=True,
                )

                response = self.deepgram.listen.rest.v("1").transcribe_file(payload, options)

                print(response.to_json(indent=4))

            except Exception as error:
                logger.error(f"Can't transcribe audio: {error}")
                raise error
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]


    def generate_answer(self, transcript: str, short_answer: bool = True, temperature: float = 0.4) -> str:
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
        if short_answer:
            system_prompt = SYSTEM_PROMPT + SHORTER_INSTRACT
        else:
            system_prompt = SYSTEM_PROMPT + LONGER_INSTRACT
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
            )
        except Exception as error:
            logger.error(f"Can't generate answer: {error}")
            raise error
        return response["choices"][0]["message"]["content"]



# llm = LLMInference()

# test = 