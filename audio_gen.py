import torch
import pyttsx3
import pyaudio
import wave
import sys
import threading

print(torch.cuda.is_available())
print(torch.__version__)


INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
tts = pyttsx3.init()  #TextToSpeechService()
voices = tts.getProperty('voices')       #getting details of current voice
tts.setProperty('voice', voices[1].id)
tts.setProperty('rate', 100)
audio = pyaudio.PyAudio()
audio.open(format=INPUT_FORMAT, 
    channels=INPUT_CHANNELS,
    rate=INPUT_RATE, 
    input=True,
    frames_per_buffer=INPUT_CHUNK).close()


# phrases: let me think, let's see, interesting, I wonder, in that case, 

def generate_audio(text, filename):
    tempPath = filename
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

# WRITE TEXT HERE AND FILE NAME TO CREATE AN AUDIO FILE
generate_audio("Thank you for sharing with me today. I hope you have a good day. Good bye.", "audio/outro.wav")