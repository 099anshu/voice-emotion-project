import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf
from utils.audio_preprocessing import extract_features
import speech_recognition as sr

MODEL = tf.keras.models.load_model("models/emotion_model.h5")
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise"]

def record_audio(duration=5, sr=22050):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()

    # Save PCM WAV (Google Speech Recognition requires this)
    write("record.wav", sr, audio)
    print("Saved as record.wav (16-bit PCM)")

def transcribe_audio(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return "Could not understand audio"

while True:
    input("Press Enter to record...")   # waits for user input
    record_audio()                      # records audio → saves record.wav

    # 1. Speech-to-text
    text = transcribe_audio("record.wav")
    print("You said:", text)

    # 2. Extract ML features
    features = extract_features("record.wav")

    # 3. Predict emotion
    prediction = MODEL.predict(np.array([features]))[0]
    emotion = EMOTIONS[np.argmax(prediction)]

    print("\nPredicted Emotion:", emotion)
