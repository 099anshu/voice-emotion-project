import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf
from utils.audio_preprocessing import extract_features

# -------------------------
# LOAD MODEL
# -------------------------
model = tf.keras.models.load_model("models/lstm_emotion_model.keras")
idx_to_emotion = {0:"angry", 1:"happy", 2:"sad", 3:"fear", 4:"neutral", 5:"surprise"}

# -------------------------
# RECORD AUDIO
# -------------------------
def record_audio(filename="record.wav", duration=3, fs=22050):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    # Convert float32 to int16 for WAV format
    recording_int16 = np.int16(recording * 32767)
    write(filename, fs, recording_int16)
    print(f"Saved as {filename}")

# -------------------------
# PREDICT EMOTION
# -------------------------
def predict_emotion(filename="record.wav"):
    features = extract_features(filename)  # shape: (time_steps, n_mfcc)
    features = features.reshape(1, features.shape[0], features.shape[1])
    prediction = model.predict(features)
    emotion = idx_to_emotion[np.argmax(prediction)]
    print(f"Predicted Emotion: {emotion}")
    return emotion

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    input("Press Enter to record...")
    record_audio()
    predicted_emotion = predict_emotion()
    print("-" * 40)
