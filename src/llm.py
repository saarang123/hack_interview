import openai
import numpy as np
from loguru import logger


from src.constants import DEEPGRAM_API_KEY, OPENAI_API_KEY, OUTPUT_FILE_NAME
from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE


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


class LLMInference:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

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