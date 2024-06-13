## Reminiscence Study Code

Based on https://blog.duy-huynh.com/build-your-own-voice-assistant-and-run-it-locally/

To setup:
- install Ollama (https://ollama.com/)
    - It's just an executable
- run `ollama pull gemma:2b-instruct-v1.1-q4_K_M`
    - Replace `gemma:2b-instruct-v1.1-q4_K_M` with any other model from Ollama if desired, just requires a one line change in app.py where the model is setup
- run `ollama serve`
- run `pip install requirements.txt`

To run:
- in text only mode for debugging `python app.py -1`
- in no_rails mode which only lets researcher confirm or regnerate `python app.py 1`
- in three prompts mode which generates 3 prompts for the researcher to choose from `python app.py 2`
- in safe mode which has better? prompt engineering `python app.py 3`
