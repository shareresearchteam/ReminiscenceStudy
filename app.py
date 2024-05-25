import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from datetime import datetime
from rich.logging import RichHandler, Console
import logging
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService
import torch
import pyttsx3
import pyaudio
import wave
import sys

fillers = ["hmm","i_wonder","in_that_case","interesting","let_me_think","lets_see"]

TEXT_ONLY = -1
NO_RAILS = 1
FIVE_PROMPTS = 2
PRESET_QUESTIONS = 3
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024

print("Using GPU" if torch.cuda.is_available() else "Not using GPU")

print("Setting up...")
mode = int(sys.argv[1]) if len(sys.argv) > 1 else NO_RAILS
stt = whisper.load_model("base.en")
tts = pyttsx3.init()  #TextToSpeechService()
voices = tts.getProperty('voices')       #getting details of current voice
tts.setProperty('voice', voices[1].id)
tts.setProperty('rate', 150)

audio = pyaudio.PyAudio()
audio.open(format=INPUT_FORMAT, 
    channels=INPUT_CHANNELS,
    rate=INPUT_RATE, 
    input=True,
    frames_per_buffer=INPUT_CHUNK).close()

if mode == PRESET_QUESTIONS:
    print("Using preset questions")

    template = """
        You are Misty, a empathetic listener. You are speaking to a person. They are telling you a story about an influential person in her life.

        You ask open-ended questions to encourage them to share more. 
        Your responses should be less than 40 words. 
        Your response must be one of the following:
        - How did that make you feel?
        - What did you learn from that experience?
        - How did that experience change you?
        - What was the most rewarding part of that experience?
        - What was the most surprising part of that experience?
        - Can you tell me more about that?
        - What was the most challenging part of that experience?
        - What part of that do you think about the most?
        - What was the most difficult part of that experience?
        - Was there anything you didn't like about that?

        The conversation transcript is as follows:
        {history}

        And here is the user's follow-up: {input}

        Your response:
        """
else:
    print("Using default template")
    template = """
        You are Misty, a empathetic listener. You are speaking to a person. They is telling you a story about an influential person in his life.

        You ask open-ended questions to encourage them to share more. 
        Your responses should be less than 40 words. 
        Your response must contain a question asking more about the story.

        The conversation transcript is as follows:
        {history}

        And here is the user's follow-up: {input}

        Your response:
        """
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model='gemma:2b'),
)



def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response

# NOT USED
def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()

def play_file(filename, wait=False):
    if wait:
        time.sleep(1)

    chunk = 1024 #measured in bytes

    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    data = wf.readframes(chunk)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()

    p.terminate()

def say_filler():
    filler = fillers[np.random.randint(0, len(fillers))]
    play_file(f"audio/{filler}.wav")    

def make_text_to_speech(text):
    tempPath = "temp.wav"
    tts.save_to_file(text , tempPath)
    tts.runAndWait()
    wf = wave.open(tempPath, 'rb')
    # open stream based on the wave object which has been input.
    stream = audio.open(format =
                    audio.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    chunkSize = 1024
    chunk = wf.readframes(chunkSize)
    while chunk:
        stream.write(chunk)
        chunk = wf.readframes(chunkSize)

    wf.close()   

class PrintLogger:
    def __init__(self, filename):
        self.console = Console()
        self.log_output = Console(file=filename)

        logging.basicConfig(format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO,
                            handlers=[RichHandler(console=self.log_output, markup=True)])

        self.logger = logging.getLogger()

    def write(self, string, manual=False):
        if not manual:
            self.console.print(string)
            self.logger.info(string)
        else:
            self.logger.info(string)

    def flush(self):
        pass


if __name__ == "__main__":
    print("mode: ", mode)

    with open(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".log", "w") as f:
        logger = PrintLogger(f)

        # checks if running a valid mode
        if mode > 3 or mode < -1:
            logger.write("[red]Invalid mode. Exiting...")
            exit()
        
        # Start up
        logger.write("[cyan]Assistant started! Press Ctrl+C to exit.")
        play_file("audio/intro.wav")
        input("Press Enter to start the session...")

        try:
            play_file("audio/starting_question.wav")
            while True:
                if mode != TEXT_ONLY:
                    logger.write(
                        "[green]Recording started, press Enter to stop..."
                    )
                    data_queue = Queue()  # type: ignore[var-annotated]
                    stop_event = threading.Event()
                    recording_thread = threading.Thread(
                        target=record_audio,
                        args=(stop_event, data_queue),
                    )
                    recording_thread.start()
                    input()
                    logger.write("Recording ended.", manual=True)

                    stop_event.set()
                    recording_thread.join()

                    audio_data = b"".join(list(data_queue.queue))
                    audio_np = (
                        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                else:
                    logger.write("[green]Type to talk to the assistant. Press Enter to send the message.")
                    audio_np = np.array([1])

                if audio_np.size > 0:
                    if mode != TEXT_ONLY:
                        logger.write("Transcribing started...", manual=True)
                        with logger.console.status("Transcribing...", spinner="earth"):
                            text = transcribe(audio_np)
                        logger.write(f"[yellow]Transcribed: {text}")

                        if text.lower() == "end session." or text.lower() == "end session":
                            logger.write("\n[red]Participant ended session.")
                            play_file("audio/outro.wav")
                            audio.terminate()
                            exit()
                    else:
                        logger.write("[yellow]You: ")


                    if mode != TEXT_ONLY:
                        time.sleep(1.5)
                        audio_thread = threading.Thread(target=say_filler)
                        audio_thread.start()


                    if mode == TEXT_ONLY:
                        with logger.console.status("Generating response...", spinner="earth"):
                            response = get_llm_response(text)
                        logger.write(f"[cyan]Assistant: {response}")

                    if mode == NO_RAILS or mode == PRESET_QUESTIONS:
                        logger.write("Generating response...", manual=True)
                        with logger.console.status("Generating response...", spinner="earth"):
                            response = get_llm_response(text)
                        logger.write(f"[cyan]Generated Response: {response}")

                        while True:
                            user_input = input("Enter 'y' to continue or any other key to retry: ")
                            if user_input.lower() == 'y':
                                logger.write("Response accepted.")
                                break
                            with logger.console.status("Re-generating response...", spinner="earth"):
                                logger.write("Re-generating response...", manual=True)
                                response = get_llm_response(text)
                                logger.write(f"[cyan]Generated Response: {response}")
                    elif mode == FIVE_PROMPTS:
                        responses = []

                        logger.write("Generating three responses...", manual=True)
                        for i in range(3):
                            new_response = get_llm_response(text)
                            responses.append(new_response)
                            logger.write(f"[cyan]{i}: {new_response}")

                        logger.write("Three responses generated.", manual=True)
                        text = input("Enter the number to respond with: ")
                        logger.write(f"Selected response: {responses[int(text)]}", manual=True)
                        response = responses[int(text)]                    

                    if mode != TEXT_ONLY:
                        audio_thread.join()
                        make_text_to_speech(response)
                        logger.write("Playing response...", manual=True)

                else:
                    logger.write(
                        "[red]No audio recorded. Please ensure your microphone is working."
                    )

        except KeyboardInterrupt:
            logger.write("\n[red]Researcher ended session.")

            if mode != TEXT_ONLY:
                play_file("audio/outro.wav")
            audio.terminate()


        logger.write("[blue]Session ended.")
