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
import torch
import pyttsx3
import pyaudio
import wave
import sys

fillers = ["hmm","i_wonder","in_that_case","interesting","let_me_think","lets_see"]

# modes 
TEXT_ONLY = -1
NO_RAILS = 1
THREE_PROMPTS = 2
PRESET_QUESTIONS = 3

# audio settings
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024

print("Using GPU" if torch.cuda.is_available() else "Not using GPU")

print("Setting up...")
# sets up tts and whisper
mode = int(sys.argv[1]) if len(sys.argv) > 1 else NO_RAILS
stt = whisper.load_model("base.en")
tts = pyttsx3.init()  #TextToSpeechService()
voices = tts.getProperty('voices')       #getting details of current voice
tts.setProperty('voice', voices[1].id)
tts.setProperty('rate', 100)

# sets up audio playing
audio = pyaudio.PyAudio()
audio.open(format=INPUT_FORMAT, 
    channels=INPUT_CHANNELS,
    rate=INPUT_RATE, 
    input=True,
    frames_per_buffer=INPUT_CHUNK).close()

# sets up the language model
if mode == PRESET_QUESTIONS:
    print("Using preset questions")

    template = """
        You are Misty, a empathetic listener. You are speaking to a person. They are telling you about an influential person in their life.

        You ask open-ended questions to encourage them to share more. 
        Your responses should be less than 30 words. 
        Your response must be one of the following:
        - Can you tell me more about that?
        - How did that make you feel?
        - What did you learn from that experience?
        - How did that experience change you?
        - What was the most rewarding part of that experience?
        - What was the most surprising part of that experience?
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
        You are Misty, a empathetic therapist. You are speaking to a person. They are telling you about an influential person in their life.

        You ask open-ended questions to encourage them to share more. 
        Your responses should be less than 30 words. 
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
    llm=Ollama(model='gemma:2b-instruct-v1.1-q4_K_M'),
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


def get_llm_response(text: str, retry=False) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.
        retry (bool): Whether this is a retry generating a response, tries to generate a different response.
    Returns:
        str: The generated response.
    """
    if retry: 
        text += "Ask a different question."
    response = chain.predict(input=text)
    # all of the following is to remove the garbage text before the actual response
    # list is not exhaustive, more cases can be added
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    if response.startswith("Sure, here is the response you requested:"):
        response = response[len("Sure, here is the response you requested:") :].strip()
    if response.startswith("Generated Response:"):
        response = response[len("Generated Response:") :].strip()
    if response.startswith("Sure, here's a different question that encourages the person to share more:"):
        response = response[len("Sure, here's a different question that encourages the person to share more:") :].strip()
    if response.startswith("Sure, here's the response you requested:"):
        response = response[len("Sure, here's the response you requested:") :].strip()
    if response.startswith("Sure, here is the answer to the user's follow-up question:"):
        response = response[len("Sure, here is the answer to the user's follow-up question:") :].strip()
    if response.startswith("Sure, here is the revised response:"):
        response = response[len("Sure, here is the revised response:") :].strip()
    if response.startswith("Sure, here is Misty's response:"):
        response = response[len("Sure, here is Misty's response:") :].strip()
    if response.startswith("Sure, here's a response to the user's question:"):
        response = response[len("Sure, here's a response to the user's question:") :].strip()
    if response.startswith("Sure, here is the answer to the follow-up question:"):
        response = response[len("Sure, here is the answer to the follow-up question:") :].strip()
    if response.startswith("Sure, here is the response to the follow-up question:"):
        response = response[len("Sure, here is the response to the follow-up question:") :].strip()
    if response.startswith("Sure, here is the response to the user's follow-up question:"):
        response = response[len("Sure, here is the response to the user's follow-up question:") :].strip()
    
    return response


def play_file(filename, wait=False):
    """
    Plays an audio file that's passed in.

    Args:
        filename (str): the name of the audio file to be played.
        wait (bool): waits before playing file.

    Returns:
        nothing.
    """
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
    '''
    Plays a filler sound to make the conversation more natural.
    Picks a random filler from the list of fillers at the top of this file.
    '''
    filler = fillers[np.random.randint(0, len(fillers))]
    play_file(f"audio/{filler}.wav")    

def make_text_to_speech(text):
    '''
    Converts text to speech and plays it using computers default tts service.
    Saves it as temp.wav and plays it using pyaudio.

    Args:
        text (str): the text to be converted to speech.
    '''
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

# This is a silly class I wrote so that using logger.write will print to console and log to a file at the same time.
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
        '''
        Writes the string to the console and the log file.
        '''
        if not manual:
            self.console.print(string)
            self.logger.info(string)
        else:
            self.logger.info(string)

    def flush(self):
        # i actually don't know what this does and I'm too scared to remove it
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

        try: # for catching keyboard interrupt
            # play starting question
            play_file("audio/starting_question.wav")
            while True:
                if mode != TEXT_ONLY: # if in an audio mode, record the audio
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
                    if mode != TEXT_ONLY: # if in audio mode transcribe the audio
                        logger.write("Transcribing started...", manual=True)
                        with logger.console.status("Transcribing...", spinner="earth"):
                            text = transcribe(audio_np)
                        logger.write(f"[yellow]Transcribed: {text}")

                        # catch if participant ended session, doesn't always work
                        if text.lower() == "end session." or text.lower() == "end session":
                            logger.write("\n[red]Participant ended session.")
                            play_file("audio/outro.wav")
                            audio.terminate()
                            exit()
                    else:
                        logger.write("[yellow]You: ")


                    if mode != TEXT_ONLY: # if in audio mode, wait 3 seconds and play filler sound
                        time.sleep(3)
                        audio_thread = threading.Thread(target=say_filler)
                        audio_thread.start()


                    if mode == TEXT_ONLY:
                        with logger.console.status("Generating response...", spinner="earth"):
                            response = get_llm_response(text)
                        logger.write(f"[cyan]Assistant: {response}")

                    if mode == NO_RAILS or mode == PRESET_QUESTIONS:
                        # generates a single response
                        logger.write("Generating response...", manual=True)
                        with logger.console.status("Generating response...", spinner="earth"):
                            response = get_llm_response(text)
                        logger.write(f"[cyan]Generated Response: {response}")

                        while True:
                            # loops until researcher is satisfied with response
                            user_input = input("Enter 'y' to continue or any other key to retry: ")
                            if user_input.lower() == 'y':
                                logger.write("Response accepted.")
                                break
                            with logger.console.status("Re-generating response...", spinner="earth"):
                                logger.write("Re-generating response...", manual=True)
                                response = get_llm_response(text, retry=True)
                                logger.write(f"[cyan]Generated Response: {response}")
                    elif mode == THREE_PROMPTS:
                        # generates three responses, one generic and asks researcher to pick one
                        responses = []

                        logger.write("Generating three responses...", manual=True)
                        for i in range(3):
                            new_response = get_llm_response(text, retry=True if i>0 else False)
                            responses.append(new_response)
                            logger.write(f"[cyan]{i}: {new_response}")
                        responses.append("Can you tell me more about that?")
                        logger.write(f"[cyan]{3}: {responses[3]}")

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
            # if session is ended by researcher, play outro and end session
            logger.write("\n[red]Researcher ended session.")

            if mode != TEXT_ONLY:
                play_file("audio/outro.wav")
            audio.terminate()


        logger.write("[blue]Session ended.")
