# Advanced PoC Development with NVIDIA Riva 

## Resume evaluator based on NVIDIA Riva’s ASR and TTS services and ChatGPT 
A resume evaluator system is able to evaluate resumes from candidates (PDF format) based on the job requirements given by the user in an audio file (WAV format).

## Explanation
1. The system receives an audio file (WAV format) with a job description (Spanish or English). For example: “Machine Learning engineer with experience in TensorFlow, Keras, and Tensorboard.” 

2. NVIDIA ASR converts this audio file to text. 

3. Previous job descriptions in text format and a Resume from a candidate (PDF format) are injected into ChatGPT. 

4. The output of ChatGPT in text format is injected into the NVIDIA TTS service and an audio file (WAV format) is given as output. 

## Test
If you want to run it in English mode, run:
- ```python main.py --language="english"```

If you want to run it in Spanish mode, run:
- ```python main.py --language="spanish"```